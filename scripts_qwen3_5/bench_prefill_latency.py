#!/usr/bin/env python3
"""Benchmark: batched prefill vs sequential infer latency on ANE.

Compares two prefill strategies for a realistic prompt:
  A) Batched prefill  — 1 predict() per chunk  (full_attn batch + linear_attn batch)
  B) Sequential infer — N predict() per chunk  (full_attn seq   + linear_attn seq)

Both use the same compiled models and the same ANE compute unit.
Reports per-chunk and total wall-clock latency.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CTX, NUM_CHUNKS, BATCH_SIZE
import coremltools as ct

MODEL_DIR = '/Users/yw68/Anemll/qwen3_5_stable_models_6chunk'
FFN_DIR   = os.path.join(MODEL_DIR, 'combined_LUT6_dedup')
EMBED     = os.path.join(MODEL_DIR, 'embeddings.mlpackage')
TOKENIZER = '/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B'

cu = ct.ComputeUnit.CPU_AND_NE
WARMUP = 2
TRIALS = 5

# ── Load models ──────────────────────────────────────────────────
print('Loading models...')
embed = ct.models.MLModel(EMBED, compute_units=cu)
ffns_infer = []
ffns_prefill = []
for ci in range(NUM_CHUNKS):
    p = os.path.join(FFN_DIR, f'chunk{ci}.mlpackage')
    ffns_infer.append(ct.models.MLModel(p, compute_units=cu, function_name='infer'))
    ffns_prefill.append(ct.models.MLModel(p, compute_units=cu, function_name='prefill'))
    print(f'  chunk{ci} loaded')

# ── Detect per-chunk state shapes ────────────────────────────────
conv_shapes, rec_shapes = [], []
for ci in range(NUM_CHUNKS):
    spec = ffns_infer[ci].get_spec()
    for fn in spec.description.functions:
        if fn.name == 'infer':
            cs, rs = (6, 1024, 32), (6, 32, 128, 128)
            for inp in fn.input:
                if inp.name == 'linear_conv_state':
                    cs = tuple(inp.type.multiArrayType.shape)
                if inp.name == 'linear_recurrent_state':
                    rs = tuple(inp.type.multiArrayType.shape)
            conv_shapes.append(cs)
            rec_shapes.append(rs)
            break
    print(f'  chunk{ci}: conv={conv_shapes[-1]}, rec={rec_shapes[-1]}')

# ── Build a realistic prompt ─────────────────────────────────────
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
msgs = [{'role': 'user', 'content': '1+1等于几？'}]
prompt_ids = list(tok.apply_chat_template(
    msgs, add_generation_prompt=True, tokenize=True,
    enable_thinking=False, return_dict=False))
valid_len = len(prompt_ids)
print(f'\nPrompt: {valid_len} tokens')

# ── Helper: reset states ─────────────────────────────────────────
def make_states():
    """Create fresh zero states for all chunks."""
    lin_convs = [np.zeros(conv_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
    lin_recs  = [np.zeros(rec_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
    # CoreML stateful predict needs state objects
    states_infer = [ffns_infer[ci].make_state() for ci in range(NUM_CHUNKS)]
    states_prefill = [ffns_prefill[ci].make_state() for ci in range(NUM_CHUNKS)]
    return lin_convs, lin_recs, states_infer, states_prefill


# ══════════════════════════════════════════════════════════════════
# BENCHMARK A: Batched prefill (1 predict per chunk)
# ══════════════════════════════════════════════════════════════════
def run_batched_prefill():
    """Time one full batched prefill pass."""
    lin_convs, lin_recs, _, states_pf = make_states()

    # Embed
    input_ids = np.zeros((1, BATCH_SIZE), dtype=np.int32)
    input_ids[0, :valid_len] = prompt_ids
    hidden = list(embed.predict({"input_ids": input_ids}).values())[0]

    # Causal mask
    mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
    for i in range(valid_len):
        mask[0, 0, i, :i + 1] = 0

    pos_ids = np.zeros((BATCH_SIZE,), dtype=np.int32)
    pos_ids[:valid_len] = np.arange(valid_len, dtype=np.int32)
    cur_pos = np.array([0], dtype=np.int32)
    valid_len_arr = np.array([valid_len], dtype=np.int32)

    chunk_times = []
    t_start = time.perf_counter()
    for ci in range(NUM_CHUNKS):
        tc = time.perf_counter()
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": cur_pos,
            "linear_conv_state": lin_convs[ci],
            "linear_recurrent_state": lin_recs[ci],
            "valid_len": valid_len_arr,
        }
        out = ffns_prefill[ci].predict(inp, state=states_pf[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_convs[ci] = out['linear_conv_state_out']
            lin_recs[ci] = out['linear_recurrent_state_out']
        chunk_times.append(time.perf_counter() - tc)
    total = time.perf_counter() - t_start
    return total, chunk_times


# ══════════════════════════════════════════════════════════════════
# BENCHMARK B: Sequential infer (N predicts per chunk)
# ══════════════════════════════════════════════════════════════════
def run_sequential_infer():
    """Time one full sequential infer pass (all prompt tokens)."""
    lin_convs, lin_recs, states_inf, _ = make_states()

    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    pos_arr = np.array([0], dtype=np.int32)
    tok_buf = np.zeros((1, 1), dtype=np.int32)

    chunk_times = [0.0] * NUM_CHUNKS
    t_start = time.perf_counter()

    for t in range(valid_len):
        tok_buf[0, 0] = prompt_ids[t]
        hidden = list(embed.predict({"input_ids": tok_buf}).values())[0]

        mask[:, :, :, :] = -65504.0
        mask[:, :, :, :t + 1] = 0
        pos_arr[0] = t

        for ci in range(NUM_CHUNKS):
            tc = time.perf_counter()
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
            }
            out = ffns_infer[ci].predict(inp, state=states_inf[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
            chunk_times[ci] += time.perf_counter() - tc

    total = time.perf_counter() - t_start
    return total, chunk_times


# ══════════════════════════════════════════════════════════════════
# Run benchmarks
# ══════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'BENCHMARK CONFIG: {valid_len} tokens, BATCH_SIZE={BATCH_SIZE}, '
      f'CTX={CTX}, {NUM_CHUNKS} chunks')
print(f'Warmup={WARMUP}, Trials={TRIALS}')
print(f'{"="*70}')

# ── Warmup + measure batched prefill ─────────────────────────────
print(f'\n--- Batched Prefill (1 predict/chunk) ---')
for w in range(WARMUP):
    t, _ = run_batched_prefill()
    print(f'  warmup {w+1}: {t*1000:.1f} ms')

batch_totals = []
batch_chunks = []
for trial in range(TRIALS):
    t, ct_list = run_batched_prefill()
    batch_totals.append(t)
    batch_chunks.append(ct_list)
    print(f'  trial {trial+1}: {t*1000:.1f} ms  '
          f'[{", ".join(f"{c*1000:.1f}" for c in ct_list)}]')

# ── Warmup + measure sequential infer ────────────────────────────
print(f'\n--- Sequential Infer ({valid_len} predicts/chunk) ---')
for w in range(WARMUP):
    t, _ = run_sequential_infer()
    print(f'  warmup {w+1}: {t*1000:.1f} ms')

seq_totals = []
seq_chunks = []
for trial in range(TRIALS):
    t, ct_list = run_sequential_infer()
    seq_totals.append(t)
    seq_chunks.append(ct_list)
    print(f'  trial {trial+1}: {t*1000:.1f} ms  '
          f'[{", ".join(f"{c*1000:.1f}" for c in ct_list)}]')

# ── Summary ──────────────────────────────────────────────────────
print(f'\n{"="*70}')
print(f'SUMMARY ({valid_len} tokens)')
print(f'{"="*70}')

b_avg = np.mean(batch_totals) * 1000
b_std = np.std(batch_totals) * 1000
s_avg = np.mean(seq_totals) * 1000
s_std = np.std(seq_totals) * 1000
ratio = s_avg / b_avg if b_avg > 0 else float('inf')

print(f'\n  Batched prefill:    {b_avg:8.1f} ± {b_std:.1f} ms')
print(f'  Sequential infer:   {s_avg:8.1f} ± {s_std:.1f} ms')
print(f'  Slowdown ratio:     {ratio:8.1f}×')

print(f'\n  Per-chunk breakdown (avg ms):')
b_chunk_avg = np.mean(batch_chunks, axis=0) * 1000
s_chunk_avg = np.mean(seq_chunks, axis=0) * 1000
print(f'  {"chunk":>8s}  {"batched":>10s}  {"sequential":>10s}  {"ratio":>8s}')
print(f'  {"-"*42}')
for ci in range(NUM_CHUNKS):
    r = s_chunk_avg[ci] / b_chunk_avg[ci] if b_chunk_avg[ci] > 0 else float('inf')
    print(f'  {ci:>8d}  {b_chunk_avg[ci]:>10.1f}  {s_chunk_avg[ci]:>10.1f}  {r:>8.1f}×')

print(f'\n  Estimated force_recurrent prefill:')
print(f'  (Single predict/chunk with unrolled recurrent loop)')
print(f'  Expected to be between batched and sequential.')
print(f'  Lower bound: {b_avg:.0f} ms (if ANE parallelizes perfectly)')
print(f'  Upper bound: {s_avg:.0f} ms (if no benefit vs individual calls)')
print(f'  Rough estimate: ~{b_avg * 3:.0f}-{b_avg * 5:.0f} ms '
      f'(3-5× batched, {s_avg/b_avg/3:.0f}-{s_avg/b_avg/5:.0f}× faster than sequential)')
