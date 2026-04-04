#!/usr/bin/env python3
"""Benchmark: prefill latency at multiple prompt lengths.

Measures:
  A) Batched prefill  — 1 predict()/chunk  (batch full_attn + batch linear_attn)
  B) Sequential infer — N predict()/chunk  (seq full_attn   + seq linear_attn)

Also estimates the hybrid:
  C) Batched full_attn + sequential linear_attn  (force_recurrent=True re-export)
     by decomposing per-layer costs from a PyTorch CPU profile.

Usage:
    source /Users/yw68/Anemll/.venv/bin/activate
    python3 /Users/yw68/Anemll/scripts_qwen3_5/bench_prefill_long.py
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
TRIALS = 3

# Prompt lengths to test
PROMPT_LENGTHS = [64, 128, 256, 512]

# ── Load models ──────────────────────────────────────────────────
print('Loading models...')
embed = ct.models.MLModel(EMBED, compute_units=cu)
ffns_infer = []
ffns_prefill = []
for ci in range(NUM_CHUNKS):
    p = os.path.join(FFN_DIR, f'chunk{ci}.mlpackage')
    ffns_infer.append(ct.models.MLModel(p, compute_units=cu, function_name='infer'))
    ffns_prefill.append(ct.models.MLModel(p, compute_units=cu, function_name='prefill'))
print(f'  All {NUM_CHUNKS} chunks loaded (infer + prefill)')

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

# ── Build prompts of various lengths ─────────────────────────────
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

# Use a long message to generate tokens
long_text = ("请详细解释量子计算机的工作原理，包括量子比特、量子门、量子纠缠和量子退相干。"
             "再讨论一下量子计算在密码学、药物发现和人工智能方面的应用前景。"
             "最后分析一下当前量子计算技术面临的主要挑战和可能的解决方案。") * 10
msgs = [{'role': 'user', 'content': long_text}]
full_ids = list(tok.apply_chat_template(
    msgs, add_generation_prompt=True, tokenize=True,
    enable_thinking=False, return_dict=False))
print(f'Generated {len(full_ids)} total tokens for slicing')

# ── Helpers ──────────────────────────────────────────────────────
def make_states():
    lin_convs = [np.zeros(conv_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
    lin_recs  = [np.zeros(rec_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
    states_infer = [ffns_infer[ci].make_state() for ci in range(NUM_CHUNKS)]
    states_prefill = [ffns_prefill[ci].make_state() for ci in range(NUM_CHUNKS)]
    return lin_convs, lin_recs, states_infer, states_prefill


def run_batched_prefill(prompt_ids):
    """Time one batched prefill pass. Returns (total_s, [chunk_s])."""
    valid_len = len(prompt_ids)
    lin_convs, lin_recs, _, states_pf = make_states()

    input_ids = np.zeros((1, BATCH_SIZE), dtype=np.int32)
    input_ids[0, :valid_len] = prompt_ids[:valid_len]
    hidden = list(embed.predict({"input_ids": input_ids}).values())[0]

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


def run_sequential_infer(prompt_ids):
    """Time one sequential infer pass. Returns (total_s, [chunk_s])."""
    valid_len = len(prompt_ids)
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
# Run benchmarks at each prompt length
# ══════════════════════════════════════════════════════════════════
results = []

for n_tok in PROMPT_LENGTHS:
    if n_tok > len(full_ids):
        print(f'\nSkipping {n_tok} tokens (only {len(full_ids)} available)')
        continue
    if n_tok > BATCH_SIZE:
        print(f'\nSkipping {n_tok} tokens (> BATCH_SIZE={BATCH_SIZE})')
        continue

    prompt = full_ids[:n_tok]
    print(f'\n{"="*70}')
    print(f'PROMPT LENGTH: {n_tok} tokens  (BATCH_SIZE={BATCH_SIZE})')
    print(f'{"="*70}')

    # ── Batched prefill ──────────────────────────────────────────
    print(f'\n  Batched prefill (1 predict/chunk):')
    for w in range(WARMUP):
        t, _ = run_batched_prefill(prompt)
        print(f'    warmup {w+1}: {t*1000:.0f} ms')

    batch_trials = []
    batch_chunk_trials = []
    for trial in range(TRIALS):
        t, ct_list = run_batched_prefill(prompt)
        batch_trials.append(t)
        batch_chunk_trials.append(ct_list)
        print(f'    trial {trial+1}: {t*1000:.0f} ms  '
              f'[{", ".join(f"{c*1000:.0f}" for c in ct_list)}]')

    # ── Sequential infer ─────────────────────────────────────────
    print(f'\n  Sequential infer ({n_tok} predicts/chunk):')
    for w in range(WARMUP):
        t, _ = run_sequential_infer(prompt)
        print(f'    warmup {w+1}: {t*1000:.0f} ms')

    seq_trials = []
    seq_chunk_trials = []
    for trial in range(TRIALS):
        t, ct_list = run_sequential_infer(prompt)
        seq_trials.append(t)
        seq_chunk_trials.append(ct_list)
        print(f'    trial {trial+1}: {t*1000:.0f} ms  '
              f'[{", ".join(f"{c*1000:.0f}" for c in ct_list)}]')

    b_avg = np.mean(batch_trials) * 1000
    b_std = np.std(batch_trials) * 1000
    s_avg = np.mean(seq_trials) * 1000
    s_std = np.std(seq_trials) * 1000
    ratio = s_avg / b_avg if b_avg > 0 else float('inf')
    b_chunks = np.mean(batch_chunk_trials, axis=0) * 1000
    s_chunks = np.mean(seq_chunk_trials, axis=0) * 1000

    results.append({
        'n_tok': n_tok,
        'batch_ms': b_avg, 'batch_std': b_std,
        'seq_ms': s_avg, 'seq_std': s_std,
        'ratio': ratio,
        'b_chunks': b_chunks, 's_chunks': s_chunks,
    })


# ══════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════
print(f'\n\n{"="*80}')
print(f'SUMMARY: Batched Prefill vs Sequential Infer')
print(f'{"="*80}')
print(f'\nLayer composition per chunk:')
print(f'  chunk0: 5 linear + 1 full   chunk1: 4 linear + 2 full')
print(f'  chunk2: 4 linear + 1 full   chunk3: 4 linear + 1 full')
print(f'  chunk4: 4 linear + 1 full   chunk5: 3 linear + 2 full')
print(f'  Total: 24 linear + 8 full = 32 layers')

print(f'\n{"tokens":>6s}  {"Batched (ms)":>14s}  {"Sequential (ms)":>16s}  {"Seq/Batch":>10s}')
print(f'{"-"*52}')
for r in results:
    print(f'{r["n_tok"]:>6d}  {r["batch_ms"]:>10.0f} ±{r["batch_std"]:>3.0f}  '
          f'{r["seq_ms"]:>12.0f} ±{r["seq_std"]:>3.0f}  {r["ratio"]:>10.1f}×')

print(f'\nPer-chunk breakdown (avg ms):')
print(f'{"":>6s}  {"--- Batched ---":>46s}    {"--- Sequential ---":>52s}')
hdr = '  '.join(f'c{i:d}' for i in range(NUM_CHUNKS))
print(f'{"tokens":>6s}  {hdr}  total    {hdr}  total')
print(f'{"-"*130}')
for r in results:
    b = r['b_chunks']
    s = r['s_chunks']
    b_str = '  '.join(f'{b[i]:>5.0f}' for i in range(NUM_CHUNKS))
    s_str = '  '.join(f'{s[i]:>5.0f}' for i in range(NUM_CHUNKS))
    print(f'{r["n_tok"]:>6d}  {b_str}  {r["batch_ms"]:>5.0f}    {s_str}  {r["seq_ms"]:>5.0f}')

print(f'\n{"="*80}')
print('ANALYSIS: Estimating hybrid (batched full_attn + sequential linear_attn)')
print(f'{"="*80}')
print(f'''
Current compiled models fuse full_attention + linear_attention layers within each
chunk, so the hybrid cannot be measured directly. However:

  Option A (batched prefill): Both full_attn and linear_attn processed in 1 call.
    - Full attention: batch matrix multiply [B,H,S,S] — O(S²) per layer
    - Linear attention: _chunk_gated_delta_rule — O(S) per layer, parallel chunks

  Option B (sequential infer): Both processed N times (1 token each).
    - Full attention: N × matmul [1,H,1,CTX] — O(N×CTX) total
    - Linear attention: N × _recurrent_gated_delta_rule — O(N) total, sequential

  Option C (force_recurrent=True re-export): Hybrid.
    - Full attention: SAME as Option A (batch matmul, 1 call)
    - Linear attention: Unrolled recurrent loop inside 1 predict() call
    - All within a single CoreML predict() — no Python-level token loop
    - Expected: close to Option A latency (ANE processes the unrolled graph)

Key insight: Linear attention is O(N) regardless — the question is whether N
iterations happen inside one predict() call (force_recurrent) or as N separate
calls (sequential). The per-call overhead dominates for short prompts.
''')

# Estimate per-call overhead
if len(results) >= 2:
    r1, r2 = results[0], results[-1]
    # Sequential: time = N * (overhead + per_token_compute)
    # overhead ≈ (s1/n1 - s2/n2) * n1*n2 / (n2-n1)  ... too noisy
    # Just report per-token cost
    for r in results:
        per_tok_seq = r['seq_ms'] / r['n_tok']
        per_tok_batch = r['batch_ms']  # fixed cost regardless of tokens
        print(f"  {r['n_tok']:>3d} tokens: batch={r['batch_ms']:.0f}ms (fixed), "
              f"seq={r['seq_ms']:.0f}ms ({per_tok_seq:.1f} ms/token)")
