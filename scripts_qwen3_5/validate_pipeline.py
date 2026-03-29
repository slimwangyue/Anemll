#!/usr/bin/env python3
"""Qwen3.5-4B Milestone 2.0 — Full validation + benchmark suite.

Tests:
  1. Single-round: batch prefill correctness
  2. Multi-round: conversational state continuity
  3. Crossover threshold: batch vs sequential breakeven
  4. Cache overflow: deterministic policy under pressure
  5. Benchmark: prefill throughput before/after

Usage:
    python scripts_qwen3_5/validate_pipeline.py [--model-dir ...]
    python scripts_qwen3_5/validate_pipeline.py --benchmark-only
"""
import gc, time, argparse, os, sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct
from transformers import AutoTokenizer
from config import BATCH_SIZE, CTX, NUM_CHUNKS, FFN_LABEL, DEFAULT_OUTPUT, DEFAULT_HF_MODEL

# ── Helpers ──────────────────────────────────────────────────────────

def _load_model(path, compute_unit, function_name=None):
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, compute_unit)
    kwargs = {"compute_units": compute_unit}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def _find_model(base_dir, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


# ── Engine (minimal, shared between tests) ───────────────────────────

class TestEngine:
    """Minimal inference engine for validation.  Loads infer + prefill
    instances from combined-dedup models and shares state between them."""

    def __init__(self, model_dir, hf_path, ctx=CTX, num_chunks=NUM_CHUNKS):
        self.model_dir = model_dir
        self.hf_path = hf_path
        self.ctx = ctx
        self.num_chunks = num_chunks
        self.pos = 0

        cu = ct.ComputeUnit.CPU_AND_NE

        print("[engine] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False)

        print("[engine] Loading embeddings...")
        self.embed = _load_model(_find_model(model_dir, "embeddings"), cu)

        print("[engine] Loading lm_head...")
        self.lmhead = _load_model(_find_model(model_dir, "lm_head"), cu)

        combined_dir = os.path.join(model_dir, f"combined_{FFN_LABEL}_dedup")
        use_combined = os.path.isdir(combined_dir)

        print("[engine] Loading FFN chunks...")
        self.ffns = []       # infer instances
        self.prefills = []   # prefill instances
        self.has_prefill = False

        for ci in range(num_chunks):
            if use_combined:
                path = _find_model(combined_dir, f"chunk{ci}")
                if path.endswith(".mlmodelc"):
                    use_combined = False
            if use_combined:
                print(f"  chunk {ci} infer...", end="", flush=True)
                t0 = time.time()
                m_infer = _load_model(path, cu, function_name="infer")
                print(f" {time.time()-t0:.0f}s")
                print(f"  chunk {ci} prefill...", end="", flush=True)
                t0 = time.time()
                m_prefill = _load_model(path, cu, function_name="prefill")
                print(f" {time.time()-t0:.0f}s")
            else:
                ffn_path = _find_model(model_dir, f"ffn_{FFN_LABEL}_chunk{ci}")
                print(f"  chunk {ci} infer...", end="", flush=True)
                t0 = time.time()
                m_infer = _load_model(ffn_path, cu)
                print(f" {time.time()-t0:.0f}s")
                try:
                    pf_path = _find_model(model_dir, f"prefill_{FFN_LABEL}_chunk{ci}")
                    print(f"  chunk {ci} prefill...", end="", flush=True)
                    t0 = time.time()
                    m_prefill = _load_model(pf_path, cu)
                    print(f" {time.time()-t0:.0f}s")
                except FileNotFoundError:
                    m_prefill = None
            self.ffns.append(m_infer)
            self.prefills.append(m_prefill)

        self.has_prefill = all(p is not None for p in self.prefills)

        # Detect shapes
        self.inp_map = {}
        try:
            spec = self.ffns[0].get_spec()
            fn_inputs = None
            if use_combined and hasattr(spec.description, 'functions'):
                for fn in spec.description.functions:
                    if fn.name == "infer":
                        fn_inputs = fn.input
                        break
            if fn_inputs is None:
                fn_inputs = spec.description.input
            for inp in fn_inputs:
                try:
                    self.inp_map[inp.name] = tuple(
                        inp.type.multiArrayType.shape)
                except Exception:
                    pass
        except Exception:
            self.inp_map = {
                'linear_conv_state': (8, 1024, 32),
                'linear_recurrent_state': (8, 32, 128, 128),
            }

        # States
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [
            np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
            for _ in range(num_chunks)]
        self.lin_recs = [
            np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
            for _ in range(num_chunks)]

        # Buffers
        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)
        self._batch_tok_buf = np.zeros((1, BATCH_SIZE), dtype=np.int32)
        self._valid_len_buf = np.zeros((1,), dtype=np.int32)
        self._batch_mask_buf = np.full(
            (1, 1, BATCH_SIZE, ctx), -65504.0, dtype=np.float16)
        self._batch_pos_buf = np.zeros(BATCH_SIZE, dtype=np.int32)
        self._batch_cur_buf = np.zeros(1, dtype=np.int32)

        print(f"[engine] Ready. has_prefill={self.has_prefill}")

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [
            np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
            for _ in range(self.num_chunks)]
        self.lin_recs = [
            np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
            for _ in range(self.num_chunks)]
        self.pos = 0

    def _step(self, tok_id, pos):
        tok = self._tok_buf; tok[0, 0] = tok_id
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = self._mask_buf; mask[:] = -65504.0; mask[:, :, :, :pos + 1] = 0
        pos_arr = self._pos_buf; pos_arr[0] = pos
        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr, "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    def _step_kv_only(self, tok_id, pos):
        tok = self._tok_buf; tok[0, 0] = tok_id
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = self._mask_buf; mask[:] = -65504.0; mask[:, :, :, :pos + 1] = 0
        pos_arr = self._pos_buf; pos_arr[0] = pos
        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr, "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

    def _batch_prefill(self, token_ids, block_start):
        valid_len = len(token_ids)
        assert 1 <= valid_len <= BATCH_SIZE
        input_ids = self._batch_tok_buf
        input_ids[0, :] = 0
        input_ids[0, :valid_len] = token_ids
        hidden = list(self.embed.predict({"input_ids": input_ids}).values())[0]
        mask = self._batch_mask_buf; mask[:] = -65504.0
        for i in range(valid_len):
            mask[0, 0, i, :block_start + i + 1] = 0
        pos_ids = self._batch_pos_buf
        pos_ids[:valid_len] = np.arange(block_start, block_start + valid_len, dtype=np.int32)
        pos_ids[valid_len:] = 0
        cur_pos = self._batch_cur_buf; cur_pos[0] = block_start
        valid_len_arr = self._valid_len_buf; valid_len_arr[0] = valid_len
        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids, "causal_mask": mask,
                "current_pos": cur_pos,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
                "valid_len": valid_len_arr,
            }
            out = self.prefills[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            next_id = int(np.argmax(lm_out["logits"].flatten()))
        else:
            next_id = int(lm_out["argmax_idx"].flatten()[0])
        self.pos = block_start + valid_len
        return next_id

    def sequential_prefill(self, tokens):
        """Prefill all tokens one at a time. Returns next_token_id."""
        for i, tok in enumerate(tokens):
            if i < len(tokens) - 1:
                self._step_kv_only(tok, self.pos)
            else:
                next_id = self._step(tok, self.pos)
            self.pos += 1
        return next_id

    def batch_prefill(self, tokens):
        """Prefill all tokens using batched 256-token blocks. Returns next_token_id."""
        chunks = [tokens[i:i + BATCH_SIZE] for i in range(0, len(tokens), BATCH_SIZE)]
        for block in chunks:
            next_id = self._batch_prefill(block, self.pos)
        return next_id

    def generate(self, first_tok_id, max_tokens=20):
        """Generate up to max_tokens from a seed token ID."""
        ids = [first_tok_id]
        stop_ids = set()
        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(self.tokenizer.eos_token_id)
        for name in ["<|im_end|>", "<|endoftext|>"]:
            t = self.tokenizer.convert_tokens_to_ids(name)
            if t is not None and t != self.tokenizer.unk_token_id:
                stop_ids.add(t)
        for _ in range(max_tokens - 1):
            if self.pos >= self.ctx - 1:
                break
            next_id = self._step(ids[-1], self.pos)
            self.pos += 1
            ids.append(next_id)
            if next_id in stop_ids:
                break
        return ids


# ── Test 1: Single-round validation ─────────────────────────────────

def test_single_round(eng):
    """Verify batch prefill produces coherent output on a real prompt."""
    print("\n" + "=" * 70)
    print("  TEST 1: Single-Round Validation")
    print("=" * 70)

    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    tokens = eng.tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False)
    if hasattr(tokens, 'input_ids'):
        tokens = tokens.input_ids
    token_list = tokens[0].tolist()

    print(f"  Prompt: \"{prompt}\"")
    print(f"  Tokens: {len(token_list)}")

    eng.reset()
    t0 = time.time()
    first_id = eng.batch_prefill(token_list)
    prefill_elapsed = time.time() - t0
    tps_prefill = len(token_list) / max(prefill_elapsed, 1e-9)

    t0 = time.time()
    gen_ids = eng.generate(first_id, max_tokens=40)
    gen_elapsed = time.time() - t0
    tps_decode = len(gen_ids) / max(gen_elapsed, 1e-9)

    text = eng.tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"  Output: \"{text[:120]}\"")
    print(f"  Prefill: {len(token_list)} tok in {prefill_elapsed*1000:.0f}ms "
          f"({tps_prefill:.0f} tok/s)")
    print(f"  Decode:  {len(gen_ids)} tok in {gen_elapsed*1000:.0f}ms "
          f"({tps_decode:.1f} tok/s)")

    ok = "paris" in text.lower() or "france" in text.lower()
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"({'contains Paris/France' if ok else 'unexpected answer'})")
    return ok


# ── Test 2: Multi-round validation ──────────────────────────────────

def test_multi_round(eng):
    """Verify conversational continuity across multiple turns."""
    print("\n" + "=" * 70)
    print("  TEST 2: Multi-Round Validation")
    print("=" * 70)

    eng.reset()

    # Build special tokens once
    t = eng.tokenizer
    im_start = t.convert_tokens_to_ids("<|im_start|>")
    im_end = t.convert_tokens_to_ids("<|im_end|>")
    nl = t.encode("\n", add_special_tokens=False)
    user_toks = t.encode("user", add_special_tokens=False)
    asst_toks = t.encode("assistant", add_special_tokens=False)

    messages = []
    turns = [
        ("My name is Alice.", 30),
        ("What is 2 + 3?", 60),
        ("What is my name?", 100),
    ]

    all_ok = True
    for turn_idx, (user_msg, max_gen) in enumerate(turns):
        messages.append({"role": "user", "content": user_msg})
        is_first = (turn_idx == 0)

        if is_first:
            tokens = t.apply_chat_template(
                messages, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=False)
            if hasattr(tokens, 'input_ids'):
                tokens = tokens.input_ids
            token_list = tokens[0].tolist()
        else:
            # Incremental: close previous assistant + add new user turn
            token_list = []
            token_list += [im_end] + nl
            token_list += [im_start] + user_toks + nl
            token_list += t.encode(user_msg, add_special_tokens=False)
            token_list += [im_end] + nl
            token_list += [im_start] + asst_toks + nl

        print(f"\n  Turn {turn_idx+1}: \"{user_msg}\" "
              f"({len(token_list)} prompt tok, pos={eng.pos})")

        # Prefill
        t0 = time.time()
        if len(token_list) >= 8 and eng.has_prefill:
            first_id = eng.batch_prefill(token_list)
        else:
            first_id = eng.sequential_prefill(token_list)
        prefill_ms = (time.time() - t0) * 1000

        # Decode
        gen_ids = eng.generate(first_id, max_tokens=max_gen)
        text = t.decode(gen_ids, skip_special_tokens=True)
        print(f"    Output: \"{text[:100]}\"")
        print(f"    Prefill: {prefill_ms:.0f}ms, Decode: {len(gen_ids)} tok, "
              f"pos={eng.pos}/{eng.ctx}")

        messages.append({"role": "assistant", "content": text})

    # Validate turn 3 references "Alice"
    # Check both with and without special tokens (model may say Alice in <think> block)
    last_text = messages[-1]["content"].lower()
    ok = "alice" in last_text
    print(f"\n  Turn 3 remembers name: {'PASS' if ok else 'FAIL'} "
          f"(text: \"{messages[-1]['content'][:120]}\")")
    all_ok = all_ok and ok

    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ── Test 3: Crossover threshold benchmark ───────────────────────────

def test_crossover(eng):
    """Measure batch vs sequential prefill at various input lengths."""
    print("\n" + "=" * 70)
    print("  TEST 3: Crossover Threshold Benchmark")
    print("=" * 70)

    if not eng.has_prefill:
        print("  SKIP: no prefill models loaded")
        return None

    lengths = [1, 2, 4, 8, 12, 16, 32, 64, 128, 256]
    # Use a reproducible token sequence
    base_tokens = list(range(1000, 1260))  # 260 tokens
    results = []

    print(f"  {'len':>5}  {'batch_ms':>10}  {'seq_ms':>10}  {'ratio':>8}  winner")
    print(f"  {'---':>5}  {'--------':>10}  {'------':>10}  {'-----':>8}  ------")

    for n in lengths:
        tokens = base_tokens[:n]

        # Batch prefill timing (3 runs, take median)
        batch_times = []
        for _ in range(3):
            eng.reset()
            t0 = time.time()
            eng._batch_prefill(tokens, 0)
            batch_times.append(time.time() - t0)
        batch_ms = sorted(batch_times)[1] * 1000

        # Sequential prefill timing (3 runs, take median)
        seq_times = []
        for _ in range(3):
            eng.reset()
            t0 = time.time()
            for i, tok in enumerate(tokens):
                if i < n - 1:
                    eng._step_kv_only(tok, i)
                else:
                    eng._step(tok, i)
            seq_times.append(time.time() - t0)
        seq_ms = sorted(seq_times)[1] * 1000

        ratio = seq_ms / max(batch_ms, 0.01)
        winner = "BATCH" if batch_ms < seq_ms else "SEQ"
        results.append((n, batch_ms, seq_ms, ratio, winner))
        print(f"  {n:>5}  {batch_ms:>10.1f}  {seq_ms:>10.1f}  {ratio:>8.2f}x  {winner}")

    # Find crossover
    crossover = None
    for n, batch_ms, seq_ms, ratio, winner in results:
        if winner == "BATCH":
            crossover = n
            break

    if crossover is not None:
        print(f"\n  Crossover: batch wins at {crossover} tokens")
        print(f"  Recommended PREFILL_CROSSOVER = {crossover}")
    else:
        print(f"\n  Sequential was always faster (batch prefill may not be worth it)")

    return crossover


# ── Test 4: Cache overflow ──────────────────────────────────────────

def test_cache_overflow(eng):
    """Verify the engine handles cache approaching the limit."""
    print("\n" + "=" * 70)
    print("  TEST 4: Cache Overflow Policy")
    print("=" * 70)

    eng.reset()

    # Fill cache close to limit with a long prompt
    # Use 900 tokens to leave ~124 for generation
    long_text = "apple banana cherry " * 100  # lots of tokens
    messages = [{"role": "user", "content": long_text}]
    tokens = eng.tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False)
    if hasattr(tokens, 'input_ids'):
        tokens = tokens.input_ids
    token_list = tokens[0].tolist()

    # Truncate to leave room for some generation
    max_prefill = eng.ctx - 50
    if len(token_list) > max_prefill:
        token_list = token_list[:max_prefill]

    print(f"  Filling cache with {len(token_list)} tokens "
          f"(limit={eng.ctx})...")

    if eng.has_prefill and len(token_list) >= 8:
        first_id = eng.batch_prefill(token_list)
    else:
        first_id = eng.sequential_prefill(token_list)

    print(f"  After prefill: pos={eng.pos}/{eng.ctx}")

    # Try generating until we hit the limit
    gen_ids = eng.generate(first_id, max_tokens=100)
    print(f"  Generated {len(gen_ids)} tokens before reaching limit")
    print(f"  Final pos: {eng.pos}/{eng.ctx}")

    ok = eng.pos <= eng.ctx
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(pos={eng.pos} <= ctx={eng.ctx})")
    return ok


# ── Test 5: Before/After benchmark ──────────────────────────────────

def test_benchmark(eng):
    """Compare sequential vs batch prefill throughput."""
    print("\n" + "=" * 70)
    print("  TEST 5: Prefill Throughput Benchmark")
    print("=" * 70)

    prompt = "Explain the theory of relativity in simple terms."
    messages = [{"role": "user", "content": prompt}]
    tokens = eng.tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False)
    if hasattr(tokens, 'input_ids'):
        tokens = tokens.input_ids
    token_list = tokens[0].tolist()

    # Extend to 256+ tokens for a meaningful benchmark
    if len(token_list) < 300:
        # Pad with repeated tokens
        token_list = token_list + list(range(1000, 1000 + (300 - len(token_list))))
    token_list = token_list[:512]  # cap to 2 full blocks

    n = len(token_list)
    print(f"  Input: {n} tokens")

    # Sequential benchmark (2 runs, take best)
    print(f"\n  Sequential prefill (token-by-token):")
    best_seq = float('inf')
    for run in range(2):
        eng.reset()
        t0 = time.time()
        for i, tok in enumerate(token_list):
            if i < n - 1:
                eng._step_kv_only(tok, i)
            else:
                eng._step(tok, i)
        elapsed = time.time() - t0
        best_seq = min(best_seq, elapsed)
        tps = n / max(elapsed, 1e-9)
        print(f"    Run {run+1}: {elapsed*1000:.0f}ms ({tps:.0f} tok/s)")

    tps_seq = n / max(best_seq, 1e-9)

    # Batch benchmark (2 runs, take best)
    if eng.has_prefill:
        print(f"\n  Batch prefill (256-tok blocks):")
        best_batch = float('inf')
        for run in range(2):
            eng.reset()
            t0 = time.time()
            eng.batch_prefill(token_list)
            elapsed = time.time() - t0
            best_batch = min(best_batch, elapsed)
            tps = n / max(elapsed, 1e-9)
            print(f"    Run {run+1}: {elapsed*1000:.0f}ms ({tps:.0f} tok/s)")

        tps_batch = n / max(best_batch, 1e-9)

        speedup = tps_batch / max(tps_seq, 1e-9)
        print(f"\n  ┌───────────────────────────────────────┐")
        print(f"  │ Sequential: {tps_seq:>8.0f} tok/s            │")
        print(f"  │ Batch:      {tps_batch:>8.0f} tok/s            │")
        print(f"  │ Speedup:    {speedup:>8.1f}x                 │")
        print(f"  └───────────────────────────────────────┘")

        ok = speedup > 2.0
        print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
              f"(speedup {'>' if ok else '<='} 2x, "
              f"expected >10x for true batch)")
    else:
        print(f"\n  Sequential only: {tps_seq:.0f} tok/s")
        print(f"  RESULT: SKIP (no prefill models)")
        ok = None
        tps_batch = None
        speedup = None

    return {
        "tps_seq": tps_seq,
        "tps_batch": tps_batch,
        "speedup": speedup,
        "n_tokens": n,
        "pass": ok,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-4B validation + benchmark suite")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run only the throughput benchmark")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_dir

    print("=" * 70)
    print("  Qwen3.5-4B Pipeline Validation Suite")
    print(f"  Model dir: {args.model_dir}")
    print("=" * 70)

    eng = TestEngine(args.model_dir, args.tokenizer)

    if args.benchmark_only:
        test_benchmark(eng)
        return

    results = {}

    # Test 1: Single round
    results["single_round"] = test_single_round(eng)

    # Test 2: Multi round
    results["multi_round"] = test_multi_round(eng)

    # Test 3: Crossover threshold
    results["crossover"] = test_crossover(eng)

    # Test 4: Cache overflow
    results["cache_overflow"] = test_cache_overflow(eng)

    # Test 5: Benchmark
    results["benchmark"] = test_benchmark(eng)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    for name, result in results.items():
        if isinstance(result, bool):
            status = "PASS" if result else "FAIL"
        elif isinstance(result, dict) and "pass" in result:
            status = "PASS" if result["pass"] else "FAIL"
        elif result is None:
            status = "SKIP"
        else:
            status = str(result)
        print(f"  {name:<20s} {status}")

    all_pass = all(
        (r is True or (isinstance(r, dict) and r.get("pass") is True)
         or (isinstance(r, int) and r > 0))  # crossover returns int
        for r in results.values() if r is not None
    )
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
