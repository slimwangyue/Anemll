#!/usr/bin/env python3
"""
Long-context compaction validation for ctx4096 Qwen3.5-4B.

Fills the KV cache near the 4096 limit, triggers real compaction,
then evaluates whether post-compaction generation is coherent,
relevant, non-repetitive, and logically continuous.

Uses combined_LUT4_dedup multifunction models with prefill batching.

Usage:
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/long_context_compaction_ctx4096.py
"""
import sys, os, time, textwrap, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import coremltools as ct
from collections import deque

# ── Configuration ─────────────────────────────────────────────────
MODEL_DIR = "artifacts/v4_all_chunks_ctx4096"
TOKENIZER_DIR = "qwen3_5_stable_lut4ffn_lut6em_fp32"
NUM_CHUNKS = 9
CTX = 4096
BATCH_SIZE = 1024   # prefill batch width
COMPUTE_UNIT = ct.ComputeUnit.ALL  # ANE for speed; cache symlinked to SSD

# Per-chunk linear-state shapes: chunk0=(3,LLL), chunks1-7=(4,FLLL), chunk8=(1,F)
CONV_SHAPES = {0: (3, 1024, 32), 8: (1, 1024, 32)}
REC_SHAPES  = {0: (3, 32, 128, 128), 8: (1, 32, 128, 128)}
DEFAULT_CONV = (4, 1024, 32)
DEFAULT_REC  = (4, 32, 128, 128)

# ── Helpers ───────────────────────────────────────────────────────
def banner(title):
    print(f"\n{'='*76}\n  {title}\n{'='*76}")


def repetition_score(tokens, window=20):
    """Fraction of overlapping n-grams in sliding windows."""
    if len(tokens) < window * 2:
        return 0.0
    ngrams = set()
    repeats = 0
    for i in range(len(tokens) - window + 1):
        ng = tuple(tokens[i:i+window])
        if ng in ngrams:
            repeats += 1
        ngrams.add(ng)
    return repeats / max(1, len(tokens) - window + 1)


# ── Load models ───────────────────────────────────────────────────
banner("Loading models")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

combined_dir = os.path.join(MODEL_DIR, "combined_LUT4_dedup")

# Embed / lmhead — single-token (decode)
embed_path = os.path.join(MODEL_DIR, "embed_lmhead_combined.mlpackage")
print("Loading embed (single) ...", end="", flush=True)
embed = ct.models.MLModel(embed_path, compute_units=ct.ComputeUnit.CPU_ONLY, function_name="embed")
print(" ok")
print("Loading lmhead ...", end="", flush=True)
lmhead = ct.models.MLModel(embed_path, compute_units=ct.ComputeUnit.CPU_ONLY, function_name="lmhead")
print(" ok")

# Embed prefill (batch)
embed_pf_path = os.path.join(MODEL_DIR, "embed_prefill.mlpackage")
print("Loading embed_prefill ...", end="", flush=True)
embed_prefill = ct.models.MLModel(embed_pf_path, compute_units=ct.ComputeUnit.CPU_ONLY)
print(" ok")

# FFN chunks — combined (infer + prefill function)
ffns = []
prefills = []
for ci in range(NUM_CHUNKS):
    path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
    print(f"  chunk {ci} ...", end="", flush=True)
    t0 = time.time()
    ffns.append(ct.models.MLModel(path, compute_units=COMPUTE_UNIT, function_name="infer"))
    prefills.append(ct.models.MLModel(path, compute_units=COMPUTE_UNIT, function_name="prefill"))
    print(f" {time.time()-t0:.0f}s")

KV_NAMES = ['k_cache', 'v_cache']
SEQ_AXIS = 2  # (layers, heads, CTX, head_dim)


# ── Engine ────────────────────────────────────────────────────────
class Engine:
    """Minimal inference engine with batch prefill and compaction."""

    def __init__(self):
        self.states = [ffns[ci].make_state() for ci in range(NUM_CHUNKS)]
        self.lin_convs = [np.zeros(CONV_SHAPES.get(ci, DEFAULT_CONV), dtype=np.float16) for ci in range(NUM_CHUNKS)]
        self.lin_recs  = [np.zeros(REC_SHAPES.get(ci, DEFAULT_REC),  dtype=np.float16) for ci in range(NUM_CHUNKS)]
        self.pos = 0
        self.rope_offset = 0
        self.token_history = deque(maxlen=CTX * 4)
        self.compaction_count = 0

        # Reusable buffers — single token
        self._tb = np.zeros((1, 1), dtype=np.int32)
        self._mb = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        self._pb = np.zeros(1, dtype=np.int32)
        self._rb = np.zeros(1, dtype=np.int32)

        # Reusable buffers — prefill batch
        self._btb = np.zeros((1, BATCH_SIZE), dtype=np.int32)
        self._bmb = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
        self._bpos = np.zeros(BATCH_SIZE, dtype=np.int32)
        self._bcur = np.zeros(1, dtype=np.int32)
        self._bvl  = np.zeros(1, dtype=np.int32)

    def reset(self):
        self.__init__()

    # ── Single-token step (no lmhead) ──
    def _step_kv(self, tok_id, pos):
        self._tb[0, 0] = tok_id
        hid = list(embed.predict({"input_ids": self._tb}).values())[0]
        self._mb[:] = -65504.0
        self._mb[:, :, :, :pos+1] = 0
        self._pb[0] = pos
        self._rb[0] = pos + self.rope_offset
        for ci in range(NUM_CHUNKS):
            out = ffns[ci].predict({
                "hidden_states": hid.astype(np.float16),
                "position_ids": self._rb,
                "causal_mask":  self._mb,
                "current_pos":  self._pb,
                "linear_conv_state":     self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }, state=self.states[ci])
            hid = out["output_hidden_states"]
            self.lin_convs[ci] = out['linear_conv_state_out']
            self.lin_recs[ci]  = out['linear_recurrent_state_out']
        return hid

    # ── Single-token step WITH lmhead ──
    def _step(self, tok_id, pos):
        hid = self._step_kv(tok_id, pos)
        lo = lmhead.predict({"hidden_states": hid.astype(np.float16)})
        logits = list(lo.values())[0].flatten().astype(np.float32)
        return int(np.argmax(logits))

    # ── Batch prefill ──
    def batch_prefill(self, token_ids, block_start):
        valid_len = len(token_ids)
        assert 1 <= valid_len <= BATCH_SIZE

        self._btb[0, :] = 0
        self._btb[0, :valid_len] = token_ids
        hid = list(embed_prefill.predict({"input_ids": self._btb}).values())[0]

        self._bmb[:] = -65504.0
        for i in range(valid_len):
            self._bmb[0, 0, i, :block_start + i + 1] = 0

        self._bpos[:valid_len] = np.arange(
            block_start + self.rope_offset,
            block_start + self.rope_offset + valid_len, dtype=np.int32)
        self._bpos[valid_len:] = 0
        self._bcur[0] = block_start
        self._bvl[0] = valid_len

        for ci in range(NUM_CHUNKS):
            out = prefills[ci].predict({
                "hidden_states": hid.astype(np.float16),
                "position_ids": self._bpos,
                "causal_mask":  self._bmb,
                "current_pos":  self._bcur,
                "linear_conv_state":     self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
                "valid_len":    self._bvl,
            }, state=self.states[ci])
            hid = out["output_hidden_states"]
            self.lin_convs[ci] = out['linear_conv_state_out']
            self.lin_recs[ci]  = out['linear_recurrent_state_out']

        if hid.ndim >= 3 and hid.shape[1] > 1:
            hid = hid[:, valid_len - 1:valid_len, :]

        lo = lmhead.predict({"hidden_states": hid.astype(np.float16)})
        logits = list(lo.values())[0].flatten().astype(np.float32)
        next_id = int(np.argmax(logits))
        self.pos = block_start + valid_len
        return next_id

    # ── Full prefill of token list (auto-batched) ──
    def prefill_tokens(self, token_ids):
        """Prefill all tokens using batch prefill. Returns next_id."""
        total = len(token_ids)
        next_id = None
        i = 0
        while i < total:
            chunk_len = min(BATCH_SIZE, total - i)
            chunk = token_ids[i:i+chunk_len]
            next_id = self.batch_prefill(chunk, self.pos)
            self.token_history.extend(chunk)
            i += chunk_len
        return next_id

    # ── Greedy decode ──
    def generate(self, first_id, n_tokens, *, compact_at=None, label="gen"):
        """Generate n_tokens greedily. Returns (ids, compaction_info_or_None)."""
        ids = []
        nxt = first_id
        compact_info = None
        t_start = time.time()
        for step_i in range(n_tokens):
            if self.pos >= CTX:
                print(f"  [generate] Hit CTX limit at step {step_i}, pos={self.pos}")
                break
            # check compaction trigger
            if compact_at is not None and self.pos >= compact_at and compact_info is None:
                compact_info = self.compact_cache()
            self.token_history.append(nxt)
            nxt = self._step(nxt, self.pos)
            self.pos += 1
            ids.append(nxt)
            # Progress every 100 tokens
            if (step_i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                tps = (step_i + 1) / elapsed
                print(f"    [{label}] {step_i+1}/{n_tokens} tokens, "
                      f"pos={self.pos}, {tps:.1f} tok/s")
        return ids, compact_info

    # ── Compaction (Strategy A: shift KV + keep linear) ──
    def compact_cache(self):
        keep_count = min(CTX // 2, self.pos, len(self.token_history))
        if keep_count <= 0:
            return None

        old_logical = self.pos + self.rope_offset
        old_physical = self.pos
        discard = old_physical - keep_count
        new_rope_offset = old_logical - keep_count

        t0 = time.time()
        for ci in range(NUM_CHUNKS):
            for sname in KV_NAMES:
                kv = self.states[ci].read_state(name=sname)
                shifted = np.zeros_like(kv)
                src = [slice(None)] * kv.ndim
                dst = [slice(None)] * kv.ndim
                src[SEQ_AXIS] = slice(discard, old_physical)
                dst[SEQ_AXIS] = slice(0, keep_count)
                shifted[tuple(dst)] = kv[tuple(src)]
                self.states[ci].write_state(name=sname, value=shifted)
        elapsed = time.time() - t0

        info = {
            "physical_before": old_physical,
            "logical_before": old_logical,
            "discard": discard,
            "keep": keep_count,
            "physical_after": keep_count,
            "new_rope_offset": new_rope_offset,
            "shift_ms": elapsed * 1000,
        }

        self.pos = keep_count
        self.rope_offset = new_rope_offset

        kept = list(self.token_history)[-keep_count:]
        self.token_history = deque(kept, maxlen=CTX * 4)
        self.compaction_count += 1

        print(f"  [compact] physical {old_physical}→{self.pos}, "
              f"discard {discard}, keep {keep_count}, "
              f"rope_offset {new_rope_offset}, "
              f"shift {elapsed*1000:.0f}ms")
        return info


# ── Scenarios ─────────────────────────────────────────────────────
SCENARIOS = []

# Scenario 1: Long technical/coding discussion
SCENARIOS.append({
    "name": "Technical / Coding",
    "prompt": textwrap.dedent("""\
    You are a senior software engineer working on a large-scale distributed system. \
    A junior developer has asked you to review the following Python code for a distributed \
    task queue system. Please provide a detailed code review.

    ```python
    import asyncio
    import json
    import hashlib
    import redis
    from dataclasses import dataclass, field
    from typing import Optional, List, Dict, Any, Callable
    from datetime import datetime, timedelta
    import logging

    logger = logging.getLogger(__name__)

    @dataclass
    class Task:
        id: str
        queue: str
        payload: Dict[str, Any]
        priority: int = 0
        max_retries: int = 3
        retry_count: int = 0
        created_at: datetime = field(default_factory=datetime.utcnow)
        scheduled_at: Optional[datetime] = None
        timeout_seconds: int = 300
        result: Optional[Any] = None
        error: Optional[str] = None
        status: str = "pending"

    class TaskQueue:
        def __init__(self, redis_url: str, worker_count: int = 4):
            self.redis = redis.from_url(redis_url)
            self.worker_count = worker_count
            self.handlers: Dict[str, Callable] = {}
            self._running = False
            self._workers: List[asyncio.Task] = []

        def register(self, queue_name: str):
            def decorator(func):
                self.handlers[queue_name] = func
                return func
            return decorator

        async def enqueue(self, task: Task) -> str:
            task_data = json.dumps({
                'id': task.id,
                'queue': task.queue,
                'payload': task.payload,
                'priority': task.priority,
                'max_retries': task.max_retries,
                'retry_count': task.retry_count,
                'created_at': task.created_at.isoformat(),
                'timeout_seconds': task.timeout_seconds,
                'status': 'queued',
            })
            score = task.priority * 1000000 + task.created_at.timestamp()
            self.redis.zadd(f'queue:{task.queue}', {task_data: score})
            self.redis.set(f'task:{task.id}', task_data, ex=86400)
            logger.info(f"Enqueued task {task.id} to {task.queue}")
            return task.id

        async def _process_task(self, queue_name: str, task_data: str):
            task_dict = json.loads(task_data)
            task = Task(**{k: v for k, v in task_dict.items() 
                         if k in Task.__dataclass_fields__})
            handler = self.handlers.get(queue_name)
            if not handler:
                logger.error(f"No handler for queue {queue_name}")
                return
            try:
                self.redis.hset(f'task:{task.id}', 'status', 'processing')
                result = await asyncio.wait_for(
                    handler(task.payload),
                    timeout=task.timeout_seconds
                )
                task.result = result
                task.status = 'completed'
                self.redis.hset(f'task:{task.id}', mapping={
                    'status': 'completed',
                    'result': json.dumps(result),
                })
            except asyncio.TimeoutError:
                task.status = 'timeout'
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.scheduled_at = datetime.utcnow() + timedelta(
                        seconds=2 ** task.retry_count * 10
                    )
                    await self.enqueue(task)
                else:
                    self.redis.hset(f'task:{task.id}', 'status', 'failed')
            except Exception as e:
                logger.exception(f"Task {task.id} failed: {e}")
                task.error = str(e)
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    await self.enqueue(task)

        async def _worker(self, queue_name: str):
            while self._running:
                items = self.redis.zpopmin(f'queue:{queue_name}', count=1)
                if items:
                    task_data, _ = items[0]
                    await self._process_task(queue_name, task_data)
                else:
                    await asyncio.sleep(0.1)

        async def start(self):
            self._running = True
            for queue_name in self.handlers:
                for _ in range(self.worker_count):
                    self._workers.append(
                        asyncio.create_task(self._worker(queue_name))
                    )

        async def stop(self):
            self._running = False
            await asyncio.gather(*self._workers, return_exceptions=True)
    ```

    Please review this code carefully. Focus on:
    1. Thread safety and concurrency issues
    2. Error handling completeness
    3. Redis usage patterns and potential race conditions
    4. Data serialization and deserialization correctness
    5. Memory management and resource cleanup
    6. Retry logic correctness
    7. Overall architecture and design patterns

    After reviewing the code, please suggest specific improvements with code examples.
    """),
    "gen_tokens": 600,
    "compact_trigger_offset": 200,  # compact 200 tokens before CTX limit
})

# Scenario 2: Long Chinese conversation / instruction
SCENARIOS.append({
    "name": "Chinese Conversation / Reasoning",
    "prompt": textwrap.dedent("""\
    你是一位资深的人工智能研究员，专门研究大型语言模型的推理能力。现在有一位博士生想要了解关于LLM推理方面的前沿研究。

    请详细回答以下几个问题：

    **问题一：思维链（Chain of Thought）推理的局限性**

    思维链提示已经成为提升LLM推理能力的标准技术。但它存在哪些根本性的局限？
    - 它在哪些类型的推理任务上表现不佳？
    - 思维链是真正的"推理"还是只是模式匹配的延伸？
    - 最近有哪些研究试图超越思维链？

    请用具体的例子和实验数据来支持你的观点。

    **问题二：搜索与规划能力**

    LLM在需要搜索和规划的任务中表现如何？例如：
    - 数学定理证明
    - 程序合成
    - 博弈论中的策略推理
    - 组合优化问题

    与传统的搜索算法（如MCTS、A*等）相比，LLM有什么独特的优势和劣势？

    **问题三：多步推理中的错误累积**

    在多步推理过程中，每一步都有微小的错误概率，这些错误会如何累积？
    - 有哪些量化研究分析了这种错误累积现象？
    - 什么策略可以缓解这个问题？
    - 自我验证（self-verification）方法的效果如何？

    **问题四：世界模型与因果推理**

    LLM是否能够构建内部的世界模型？
    - 对于因果推理任务（区分相关性和因果性），LLM的表现如何？
    - LLM能否学习到真正的因果结构，还是仅仅是统计关联？
    - 最近的研究中，有没有evidence表明LLM可以进行某种形式的因果推断？

    请分别对以上四个问题进行深入分析，每个问题至少用一千字来回答。
    """),
    "gen_tokens": 700,
    "compact_trigger_offset": 150,
})

# Scenario 3: Long reasoning / Q&A chain
SCENARIOS.append({
    "name": "Multi-turn Reasoning QA",
    "prompt": textwrap.dedent("""\
    Context: You are helping me solve a complex business optimization problem. I run a chain of 12 coffee shops \
    across a metropolitan area. Each shop has different characteristics:

    Shop 1 (Downtown Core): 60 seats, rent $12,000/mo, avg daily customers 340, peak hours 7-9am and 12-2pm
    Shop 2 (University District): 45 seats, rent $8,500/mo, avg daily customers 280, peak hours 8-11am and 2-5pm
    Shop 3 (Business Park): 35 seats, rent $7,000/mo, avg daily customers 190, peak hours 7-9am and 12-1pm
    Shop 4 (Suburban Mall): 50 seats, rent $9,200/mo, avg daily customers 220, peak hours 10am-2pm and 5-8pm
    Shop 5 (Train Station): 20 seats, rent $15,000/mo, avg daily customers 410, peak hours 6-9am and 4-7pm
    Shop 6 (Hospital Area): 30 seats, rent $6,800/mo, avg daily customers 165, peak hours 7-10am and 1-3pm
    Shop 7 (Tech Hub): 40 seats, rent $10,500/mo, avg daily customers 255, peak hours 8-10am and 12-2pm
    Shop 8 (Residential Area): 55 seats, rent $5,500/mo, avg daily customers 145, peak hours 8-11am weekdays, 9am-3pm weekends
    Shop 9 (Airport Road): 25 seats, rent $11,000/mo, avg daily customers 180, peak hours vary widely
    Shop 10 (Cultural District): 65 seats, rent $13,500/mo, avg daily customers 195, peak hours 10am-1pm and 6-9pm
    Shop 11 (Sports Arena): 35 seats, rent $8,000/mo, avg daily customers 120 (spikes to 500+ on event days)
    Shop 12 (Waterfront): 70 seats, rent $14,000/mo, avg daily customers 210, peak hours 11am-3pm and 5-8pm

    Additional data:
    - Average ticket: $6.50 (drinks only) to $12.80 (drinks + food)
    - Staff cost: $17/hr base, $22/hr peak premium
    - Each shop needs minimum 2 staff during off-peak, 4 during peak hours
    - Food waste averages 8% of food inventory
    - Supply chain: main warehouse delivers 3x/week, emergency deliveries cost 3x normal
    - Customer satisfaction scores range from 3.2 to 4.7 out of 5
    - We're considering closing 2 underperforming shops and opening 1 new location

    Question: Given all this data, please help me analyze:

    1. Which 2 shops should we consider closing? Show the financial analysis including:
       - Revenue per seat per day
       - Rent-to-revenue ratio
       - Customer density efficiency
       - Location strategic value

    2. What characteristics should the new location have? Consider:
       - Optimal seat count
       - Target rent range
       - Ideal customer traffic patterns
       - What demographic/location type

    3. For the remaining 11 shops, propose a staffing optimization plan that:
       - Reduces total labor cost by at least 12%
       - Maintains service quality during peak hours
       - Accounts for cross-training and flexibility
       - Includes specific shift schedules for each shop

    4. Design a supply chain optimization that:
       - Reduces food waste from 8% to under 4%
       - Minimizes emergency deliveries
       - Accounts for demand variability across shops

    Please provide detailed numerical analysis with specific recommendations.
    """),
    "gen_tokens": 500,
    "compact_trigger_offset": 250,
})


# ── Run scenarios ─────────────────────────────────────────────────
results = []

for sc_idx, scenario in enumerate(SCENARIOS):
    banner(f"Scenario {sc_idx+1}/{len(SCENARIOS)}: {scenario['name']}")

    eng = Engine()

    # Tokenize prompt
    prompt_ids = tokenizer.encode(scenario["prompt"], add_special_tokens=False)
    prompt_len = len(prompt_ids)
    print(f"  Prompt length: {prompt_len} tokens")

    # If prompt is shorter than ~3500 tokens, we'll generate more to fill
    target_fill = CTX - scenario["compact_trigger_offset"]
    gen_to_fill = max(0, target_fill - prompt_len)

    # ── Phase 1: Prefill prompt ──
    print(f"  Phase 1: Prefilling {prompt_len} tokens via batch prefill ...")
    t0 = time.time()
    first_id = eng.prefill_tokens(prompt_ids)
    prefill_time = time.time() - t0
    print(f"    Prefill done in {prefill_time:.1f}s,  pos={eng.pos}")

    # ── Phase 2: Generate to fill context close to limit ──
    pre_compact_ids = []
    if gen_to_fill > 0:
        print(f"  Phase 2: Generating {gen_to_fill} tokens to fill context near {target_fill} ...")
        t0 = time.time()
        fill_ids, _ = eng.generate(first_id, gen_to_fill, label="fill")
        fill_time = time.time() - t0
        pre_compact_ids = fill_ids
        first_id = fill_ids[-1] if fill_ids else first_id
        print(f"    Generated {len(fill_ids)} tokens in {fill_time:.1f}s,  pos={eng.pos}")
    else:
        print(f"  Phase 2: Prompt already fills context sufficiently (pos={eng.pos})")

    pos_before_compact = eng.pos
    logical_before_compact = eng.pos + eng.rope_offset

    # ── Phase 3: Generate with compaction trigger ──
    # Try to generate enough that we cross the compaction trigger
    gen_target = scenario["gen_tokens"]
    # Compaction will trigger when pos >= CTX - compact_trigger_offset
    # Actually let's set it to trigger at a fixed position
    compact_trigger_pos = CTX - scenario["compact_trigger_offset"]

    print(f"  Phase 3: Generating {gen_target} tokens with compaction at pos>={compact_trigger_pos} ...")
    print(f"    Current pos={eng.pos}, rope_offset={eng.rope_offset}")
    t0 = time.time()
    post_ids, compact_info = eng.generate(first_id, gen_target, compact_at=compact_trigger_pos, label="post")
    gen_time = time.time() - t0
    print(f"    Generated {len(post_ids)} tokens in {gen_time:.1f}s")

    # ── Collect results ──
    all_gen_ids = pre_compact_ids + [first_id] + post_ids if gen_to_fill > 0 else post_ids
    full_text = tokenizer.decode(all_gen_ids, skip_special_tokens=True)

    # Find where compaction happened in post_ids
    compact_boundary = None
    if compact_info:
        # Compaction happened — figure out which token in post_ids
        # post generation started at pos_before_compact
        compact_boundary = compact_info["physical_before"] - pos_before_compact

    # Pre-compaction text (last 200 tokens before compaction)
    pre_text = tokenizer.decode(pre_compact_ids[-200:] if pre_compact_ids else [], skip_special_tokens=True)
    # Post-compaction text
    post_text = tokenizer.decode(post_ids, skip_special_tokens=True)

    rep_score = repetition_score(post_ids)

    result = {
        "scenario": scenario["name"],
        "prompt_tokens": prompt_len,
        "gen_to_fill": gen_to_fill,
        "pre_compact_gen": len(pre_compact_ids),
        "post_compact_gen": len(post_ids),
        "compact_info": compact_info,
        "pos_before_compact": pos_before_compact,
        "logical_before_compact": logical_before_compact,
        "final_pos": eng.pos,
        "final_rope_offset": eng.rope_offset,
        "repetition_score": rep_score,
        "total_gen_time": (fill_time if gen_to_fill > 0 else 0) + gen_time,
    }
    results.append(result)

    # ── Print report ──
    banner(f"Results — {scenario['name']}")

    print(f"  Prompt tokens:       {prompt_len}")
    print(f"  Generated to fill:   {len(pre_compact_ids)}")
    print(f"  Post-compact gen:    {len(post_ids)}")
    print(f"  Final pos:           {eng.pos}")
    print(f"  Final rope_offset:   {eng.rope_offset}")
    if compact_info:
        print(f"\n  ── Compaction Details ──")
        print(f"  Physical pos before: {compact_info['physical_before']}")
        print(f"  Logical pos before:  {compact_info['logical_before']}")
        print(f"  Tokens discarded:    {compact_info['discard']}")
        print(f"  Tokens kept:         {compact_info['keep']}")
        print(f"  Physical pos after:  {compact_info['physical_after']}")
        print(f"  New rope_offset:     {compact_info['new_rope_offset']}")
        print(f"  Shift time:          {compact_info['shift_ms']:.0f}ms")
    else:
        print(f"\n  ⚠ Compaction did NOT trigger (pos={pos_before_compact} < {compact_trigger_pos})")

    print(f"\n  Repetition score (20-gram): {rep_score:.4f}")
    print(f"  (0 = no repetition, >0.1 = concerning)")

    print(f"\n  ── Pre-compaction text (last ~200 tokens) ──")
    print(textwrap.fill(pre_text[:500], width=100, initial_indent="    ", subsequent_indent="    "))
    print(f"\n  ── Post-compaction generated text ──")
    # Show first 1000 chars
    print(textwrap.fill(post_text[:1500], width=100, initial_indent="    ", subsequent_indent="    "))

    # Check for obvious pathologies
    pathologies = []
    if rep_score > 0.1:
        pathologies.append("HIGH REPETITION")
    if len(post_ids) < 20:
        pathologies.append("TOO FEW TOKENS GENERATED")
    # Check for degenerate tokens (all same token)
    if len(set(post_ids[-50:])) < 5:
        pathologies.append("DEGENERATE (very few unique tokens)")
    # Check for EOS flood
    eos_count = sum(1 for t in post_ids if t in (tokenizer.eos_token_id, 151643, 151645))
    if eos_count > 10:
        pathologies.append(f"EXCESSIVE EOS ({eos_count} stop tokens)")

    if pathologies:
        print(f"\n  ⚠ PATHOLOGIES DETECTED: {', '.join(pathologies)}")
    else:
        print(f"\n  ✓ No obvious pathologies detected")

    del eng

# ── Final summary ─────────────────────────────────────────────────
banner("FINAL SUMMARY")

all_pass = True
for r in results:
    name = r["scenario"]
    ci = r["compact_info"]
    status = "✓" if ci else "✗ (no compaction)"
    rep = r["repetition_score"]
    rep_tag = "✓" if rep < 0.05 else ("~" if rep < 0.15 else "✗")

    print(f"\n  {name}:")
    print(f"    Compaction:    {status}")
    if ci:
        print(f"      pos {ci['physical_before']}→{ci['physical_after']}, "
              f"discard {ci['discard']}, keep {ci['keep']}, "
              f"rope_off {ci['new_rope_offset']}")
    print(f"    Post-gen:      {r['post_compact_gen']} tokens")
    print(f"    Repetition:    {rep:.4f} {rep_tag}")
    print(f"    Final pos:     {r['final_pos']}  (rope_off={r['final_rope_offset']})")

    if not ci:
        all_pass = False
    if rep > 0.15:
        all_pass = False

print(f"\n{'='*76}")
if all_pass:
    print("  VERDICT: ✓ All scenarios passed — compaction is viable for ctx4096.")
else:
    print("  VERDICT: ✗ Issues detected — see details above.")
print(f"{'='*76}")
print("\nDone.")
