#!/usr/bin/env python3
"""
Comprehensive root-cause analysis: batch-tail vs sequential-tail divergence.

Tests (all on real CoreML models, CPU_ONLY for reproducibility):
  1. A vs B strict cache/state comparison
  2. Padding leakage test (same valid_len, different padding content)
  3. Four-condition matrix (zero/nonzero state × full/partial batch)
  4. Summary table

Usage:
  python tests/dev/diag_rca_comprehensive.py
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine

# ── Constants ────────────────────────────────────────────────
MODEL_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3'
HF_PATH   = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'
CTX       = 4096
N_CHUNKS  = 9
CU        = ct.ComputeUnit.CPU_ONLY   # deterministic

# ── Helpers ──────────────────────────────────────────────────
def cos_sim(a, b):
    a_f, b_f = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    if a_f.size == 0 or b_f.size == 0:
        return float('nan')
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    if na < 1e-30 or nb < 1e-30:
        return float('nan')
    return float(np.dot(a_f, b_f) / (na * nb))

def max_abs_diff(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.max(d)) if d.size else float('nan')

def mean_abs_diff(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.mean(d)) if d.size else float('nan')

def snapshot_states(engine):
    """Deep-copy all mutable state: lin_convs, lin_recs, MLState KV caches."""
    convs = [c.copy() for c in engine.lin_convs]
    recs  = [r.copy() for r in engine.lin_recs]
    kv = []
    for ci in range(engine.num_chunks):
        chunk_kv = {}
        for sn in engine.kv_state_names:
            chunk_kv[sn] = engine.states[ci].read_state(name=sn).copy()
        kv.append(chunk_kv)
    return {'convs': convs, 'recs': recs, 'kv': kv, 'pos': engine.pos}

def compare_snapshots(snap_a, snap_b, label_a='A', label_b='B'):
    """Compare two state snapshots and print summary."""
    rows = []
    for ci in range(len(snap_a['convs'])):
        ca, cb = snap_a['convs'][ci], snap_b['convs'][ci]
        ra, rb = snap_a['recs'][ci], snap_b['recs'][ci]
        rows.append({
            'chunk': ci,
            'conv_cos': cos_sim(ca, cb),
            'conv_mad': max_abs_diff(ca, cb),
            'rec_cos': cos_sim(ra, rb),
            'rec_mad': max_abs_diff(ra, rb),
        })
    # KV cache comparison
    kv_rows = []
    for ci in range(len(snap_a['kv'])):
        for sn in snap_a['kv'][ci]:
            ka, kb = snap_a['kv'][ci][sn], snap_b['kv'][ci][sn]
            kv_rows.append({
                'chunk': ci, 'name': sn,
                'cos': cos_sim(ka, kb),
                'mad': max_abs_diff(ka, kb),
                'mean_ad': mean_abs_diff(ka, kb),
            })
    return rows, kv_rows

def print_state_table(rows, kv_rows, title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"  {'chunk':>5}  {'conv_cos':>10}  {'conv_mad':>10}  {'rec_cos':>10}  {'rec_mad':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in rows:
        print(f"  {r['chunk']:>5}  {r['conv_cos']:>10.6f}  {r['conv_mad']:>10.4f}  "
              f"{r['rec_cos']:>10.6f}  {r['rec_mad']:>10.4f}")
    print(f"\n  KV cache:")
    print(f"  {'chunk':>5}  {'name':>12}  {'cos':>10}  {'mad':>10}  {'mean_ad':>10}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in kv_rows:
        print(f"  {r['chunk']:>5}  {r['name']:>12}  {r['cos']:>10.6f}  "
              f"{r['mad']:>10.4f}  {r['mean_ad']:>10.6f}")

def run_batch_prefill_instrumented(engine, token_ids, block_start, collect_hidden=True):
    """Run batch prefill, collecting per-chunk hidden states at last valid token."""
    valid_len = len(token_ids)
    bs = engine._prefill_bs

    input_ids = engine._batch_tok_buf
    input_ids[0, :] = 0
    input_ids[0, :valid_len] = token_ids
    hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
    if valid_len < bs:
        hidden[:, valid_len:, :] = 0.0

    mask = engine._batch_mask_buf
    mask[:, :, :, :] = -65504.0
    for i in range(valid_len):
        mask[0, 0, i, :block_start + i + 1] = 0
    for i in range(valid_len, bs):
        mask[0, 0, i, 0] = 0.0

    pos_ids = engine._batch_pos_buf
    pos_ids[:valid_len] = np.arange(
        block_start + engine.rope_offset,
        block_start + engine.rope_offset + valid_len, dtype=np.int32)
    pos_ids[valid_len:] = 0
    cur_pos = engine._batch_cur_buf
    cur_pos[0] = block_start
    valid_len_arr = engine._valid_len_buf
    valid_len_arr[0] = valid_len

    hiddens = []
    for ci in range(engine.num_chunks):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": cur_pos,
            "linear_conv_state": engine.lin_convs[ci],
            "linear_recurrent_state": engine.lin_recs[ci],
            "valid_len": valid_len_arr,
        }
        out = engine.prefills[ci].predict(inp, state=engine.states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            engine.lin_convs[ci] = out['linear_conv_state_out']
            engine.lin_recs[ci] = out['linear_recurrent_state_out']
        if valid_len < bs:
            hidden[:, valid_len:, :] = 0.0
        if collect_hidden:
            hiddens.append(hidden[:, valid_len-1:valid_len, :].copy())

    # lm_head
    h_final = hidden[:, valid_len-1:valid_len, :] if hidden.shape[1] > 1 else hidden
    lm_out = engine.lmhead.predict({"hidden_states": h_final.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits = engine._extract_logits(lm_out)
        next_id = int(np.argmax(logits))
    else:
        logits = None
        next_id = int(lm_out["argmax_idx"].flatten()[0])

    engine.pos = block_start + valid_len
    return next_id, logits, hiddens

def run_sequential_prefill_instrumented(engine, token_ids, start_pos, collect_hidden=True):
    """Run sequential prefill, collecting per-chunk hidden for last token."""
    n = len(token_ids)
    hiddens = []

    # All but last: _step_kv_only
    for ti, tok_id in enumerate(token_ids[:-1]):
        engine._step_kv_only(tok_id, start_pos + ti)

    # Last token: manual per-chunk to collect hidden states
    last_tok = token_ids[-1]
    last_pos = start_pos + n - 1
    tok = engine._tok_buf
    tok[0, 0] = last_tok
    hidden = list(engine.embed.predict({"input_ids": tok}).values())[0]

    mask_s = engine._mask_buf
    mask_s[:, :, :, :] = -65504.0
    mask_s[0, 0, 0, :last_pos+1] = 0.0
    rope_arr = engine._rope_buf
    rope_arr[0] = last_pos + engine.rope_offset

    for ci in range(engine.num_chunks):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": rope_arr,
            "causal_mask": mask_s,
            "current_pos": np.array([last_pos], dtype=np.int32),
            "linear_conv_state": engine.lin_convs[ci],
            "linear_recurrent_state": engine.lin_recs[ci],
        }
        out = engine.ffns[ci].predict(inp, state=engine.states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            engine.lin_convs[ci] = out['linear_conv_state_out']
            engine.lin_recs[ci] = out['linear_recurrent_state_out']
        if collect_hidden:
            hiddens.append(hidden.copy())

    lm_out = engine.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits = engine._extract_logits(lm_out)
        next_id = int(np.argmax(logits))
    else:
        logits = None
        next_id = int(lm_out["argmax_idx"].flatten()[0])

    engine.pos = start_pos + n
    return next_id, logits, hiddens

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
print("Loading engine...")
engine = ChatEngine(
    model_dir=MODEL_DIR, hf_path=HF_PATH, ctx=CTX, num_chunks=N_CHUNKS,
    compute_unit=CU,
)
engine.load()
bs = engine._prefill_bs
print(f"Batch size: {bs}")

# Build a prompt that gives 256+35 = 291 tokens
prompt = (
    "Please translate the following passage into Chinese.\n\n"
    "In recent work on efficient transformer inference, a method referred to as "
    "Hierarchical Context Distillation has been proposed to address the growing "
    "cost of long-context processing."
)
messages = [{'role': 'user', 'content': prompt}]
tokens = engine._tokenize_messages(messages, enable_thinking=False)
# Adjust to get exactly 256 + 35 = 291 tokens
target = bs + 35
if len(tokens) < target:
    # extend with repeated content
    extra = " The central idea is to iteratively compress representations."
    while len(tokens) < target:
        messages = [{'role': 'user', 'content': prompt + extra}]
        tokens = engine._tokenize_messages(messages, enable_thinking=False)
        extra += " Additional context for longer input."
# Truncate to exactly target
tokens = tokens[:target]
block1 = tokens[:bs]
tail   = tokens[bs:]
print(f"Total tokens: {len(tokens)}, block1: {len(block1)}, tail: {len(tail)}")

# ══════════════════════════════════════════════════════════════
#  TEST 1: A vs B strict state comparison
# ══════════════════════════════════════════════════════════════
print(f"\n{'#'*72}")
print(f"#  TEST 1: A (256 batch + 35 sequential) vs B (256 batch + 35 batch)")
print(f"{'#'*72}")

# ── Case A: 256 batch + 35 sequential ──
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
snap_after_block1 = snapshot_states(engine)
next_A, logits_A, hiddens_A = run_sequential_prefill_instrumented(engine, tail, engine.pos)
snap_A = snapshot_states(engine)

# ── Case B: 256 batch + 35 batch ──
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_B, logits_B, hiddens_B = run_batch_prefill_instrumented(engine, tail, engine.pos)
snap_B = snapshot_states(engine)

# Compare
rows_AB, kv_AB = compare_snapshots(snap_A, snap_B, 'A(seq)', 'B(batch)')
print_state_table(rows_AB, kv_AB, "TEST 1: Linear state after tail — A(seq) vs B(batch)")

print(f"\n  Hidden states at last valid token (per chunk):")
print(f"  {'chunk':>5}  {'cos':>10}  {'mad':>10}  {'mean_ad':>10}")
print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")
for ci in range(min(len(hiddens_A), len(hiddens_B))):
    ha, hb = hiddens_A[ci], hiddens_B[ci]
    print(f"  {ci:>5}  {cos_sim(ha,hb):>10.6f}  {max_abs_diff(ha,hb):>10.4f}  {mean_abs_diff(ha,hb):>10.6f}")

print(f"\n  Next token — A: {next_A} ({engine.tokenizer.decode([next_A])!r})")
print(f"  Next token — B: {next_B} ({engine.tokenizer.decode([next_B])!r})")
print(f"  MATCH: {next_A == next_B}")

if logits_A is not None and logits_B is not None:
    topk = 10
    top_A = np.argsort(logits_A)[-topk:][::-1]
    top_B = np.argsort(logits_B)[-topk:][::-1]
    print(f"\n  Top-{topk} A: {[f'{engine.tokenizer.decode([t])!r}({logits_A[t]:.2f})' for t in top_A]}")
    print(f"  Top-{topk} B: {[f'{engine.tokenizer.decode([t])!r}({logits_B[t]:.2f})' for t in top_B]}")
    logit_cos = cos_sim(logits_A, logits_B)
    print(f"  Logits cos: {logit_cos:.6f}")

# ══════════════════════════════════════════════════════════════
#  TEST 2: Padding leakage (same valid_len, different padding)
# ══════════════════════════════════════════════════════════════
print(f"\n{'#'*72}")
print(f"#  TEST 2: Padding leakage — same valid_len={len(tail)}, different padding")
print(f"{'#'*72}")

def run_tail_with_padding(engine, block1_tokens, tail_tokens, padding_hidden_fn, label):
    engine._reset_states()
    engine.pos = 0
    _ = engine._batch_prefill(block1_tokens, 0)
    vl = len(tail_tokens)
    bsz = engine._prefill_bs
    input_ids = engine._batch_tok_buf
    input_ids[0, :] = 0
    input_ids[0, :vl] = tail_tokens
    hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
    hdim = hidden.shape[2]
    if hidden.shape[1] > vl:
        pad_h = padding_hidden_fn(1, hidden.shape[1] - vl, hdim)
        hidden[:, vl:, :] = pad_h

    mask = engine._batch_mask_buf
    mask[:, :, :, :] = -65504.0
    for i in range(vl):
        mask[0, 0, i, :bs + i + 1] = 0
    for i in range(vl, bsz):
        mask[0, 0, i, 0] = 0.0
    pos_ids = engine._batch_pos_buf
    pos_ids[:vl] = np.arange(bs + engine.rope_offset, bs + engine.rope_offset + vl, dtype=np.int32)
    pos_ids[vl:] = 0
    cur_pos = engine._batch_cur_buf
    cur_pos[0] = bs
    valid_len_arr = engine._valid_len_buf
    valid_len_arr[0] = vl

    hiddens = []
    for ci in range(engine.num_chunks):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_ids, "causal_mask": mask,
            "current_pos": cur_pos,
            "linear_conv_state": engine.lin_convs[ci],
            "linear_recurrent_state": engine.lin_recs[ci],
            "valid_len": valid_len_arr,
        }
        out = engine.prefills[ci].predict(inp, state=engine.states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            engine.lin_convs[ci] = out['linear_conv_state_out']
            engine.lin_recs[ci] = out['linear_recurrent_state_out']
        if hidden.shape[1] > vl:
            pad_h2 = padding_hidden_fn(1, hidden.shape[1] - vl, hdim)
            hidden[:, vl:, :] = pad_h2
        if hidden.shape[1] >= vl:
            hiddens.append(hidden[:, vl-1:vl, :].copy())
        else:
            hiddens.append(hidden[:, -1:, :].copy())
    return hiddens, snapshot_states(engine)

pad_zero = lambda b, n, d: np.zeros((b, n, d), dtype=np.float16)
pad_rand = lambda b, n, d: np.random.RandomState(42).randn(b, n, d).astype(np.float16)
pad_ones = lambda b, n, d: np.ones((b, n, d), dtype=np.float16)

hZ, sZ = run_tail_with_padding(engine, block1, tail, pad_zero, "zeros")
hR, sR = run_tail_with_padding(engine, block1, tail, pad_rand, "random")
hO, sO = run_tail_with_padding(engine, block1, tail, pad_ones, "ones")

print(f"\n  Hidden at last valid token: ZEROS vs RANDOM vs ONES")
print(f"  {'chunk':>5}  {'Z-R cos':>10}  {'Z-R mad':>10}  {'Z-O cos':>10}  {'Z-O mad':>10}")
print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for ci in range(len(hZ)):
    print(f"  {ci:>5}  {cos_sim(hZ[ci],hR[ci]):>10.6f}  {max_abs_diff(hZ[ci],hR[ci]):>10.4f}  "
          f"{cos_sim(hZ[ci],hO[ci]):>10.6f}  {max_abs_diff(hZ[ci],hO[ci]):>10.4f}")

# Conv/rec state
print(f"\n  Conv/rec state: ZEROS vs RANDOM")
print(f"  {'chunk':>5}  {'conv_cos':>10}  {'rec_cos':>10}")
print(f"  {'-'*5}  {'-'*10}  {'-'*10}")
for ci in range(len(sZ['convs'])):
    print(f"  {ci:>5}  {cos_sim(sZ['convs'][ci], sR['convs'][ci]):>10.6f}  "
          f"{cos_sim(sZ['recs'][ci], sR['recs'][ci]):>10.6f}")

# ══════════════════════════════════════════════════════════════
#  TEST 3: Four conditions — zero/nonzero state × full/partial batch
# ══════════════════════════════════════════════════════════════
print(f"\n{'#'*72}")
print(f"#  TEST 3: Four conditions — batch-prefill vs sequential reference")
print(f"{'#'*72}")

conditions = {}

# Condition 1: zero state + full batch (block1 from pos=0)
print("\n  Condition 1: zero state + full batch...")
engine._reset_states(); engine.pos = 0
_, _, h_batch_1 = run_batch_prefill_instrumented(engine, block1, 0)
s_batch_1 = snapshot_states(engine)

engine._reset_states(); engine.pos = 0
_, _, h_seq_1 = run_sequential_prefill_instrumented(engine, block1, 0)
s_seq_1 = snapshot_states(engine)

conditions['1_zero_full'] = (h_batch_1, h_seq_1, s_batch_1, s_seq_1)

# Condition 2: zero state + partial tail (from pos=0, only tail tokens)
print("  Condition 2: zero state + partial tail...")
engine._reset_states(); engine.pos = 0
_, _, h_batch_2 = run_batch_prefill_instrumented(engine, tail, 0)
s_batch_2 = snapshot_states(engine)

engine._reset_states(); engine.pos = 0
_, _, h_seq_2 = run_sequential_prefill_instrumented(engine, tail, 0)
s_seq_2 = snapshot_states(engine)

conditions['2_zero_partial'] = (h_batch_2, h_seq_2, s_batch_2, s_seq_2)

# Condition 3: non-zero state + full batch (block1 first, then another full block)
# We need 2*bs tokens for this. Duplicate block1 for block2.
block2 = block1  # same tokens, different position
print("  Condition 3: non-zero state + full batch...")
engine._reset_states(); engine.pos = 0
_ = engine._batch_prefill(block1, 0)  # prime the state
_, _, h_batch_3 = run_batch_prefill_instrumented(engine, block2, engine.pos)
s_batch_3 = snapshot_states(engine)

engine._reset_states(); engine.pos = 0
_ = engine._batch_prefill(block1, 0)
_, _, h_seq_3 = run_sequential_prefill_instrumented(engine, block2, engine.pos)
s_seq_3 = snapshot_states(engine)

conditions['3_nonzero_full'] = (h_batch_3, h_seq_3, s_batch_3, s_seq_3)

# Condition 4: non-zero state + partial tail (the actual problem case)
print("  Condition 4: non-zero state + partial tail...")
engine._reset_states(); engine.pos = 0
_ = engine._batch_prefill(block1, 0)
_, _, h_batch_4 = run_batch_prefill_instrumented(engine, tail, engine.pos)
s_batch_4 = snapshot_states(engine)

engine._reset_states(); engine.pos = 0
_ = engine._batch_prefill(block1, 0)
_, _, h_seq_4 = run_sequential_prefill_instrumented(engine, tail, engine.pos)
s_seq_4 = snapshot_states(engine)

conditions['4_nonzero_partial'] = (h_batch_4, h_seq_4, s_batch_4, s_seq_4)

# Print comparison table
print(f"\n  {'='*72}")
print(f"  Four-condition hidden-state comparison (batch vs sequential)")
print(f"  {'='*72}")
for cname, (hb, hs, sb, ss) in conditions.items():
    print(f"\n  {cname}:")
    print(f"    {'chunk':>5}  {'cos':>10}  {'mad':>10}")
    print(f"    {'-'*5}  {'-'*10}  {'-'*10}")
    for ci in range(min(len(hb), len(hs))):
        print(f"    {ci:>5}  {cos_sim(hb[ci],hs[ci]):>10.6f}  {max_abs_diff(hb[ci],hs[ci]):>10.4f}")

print(f"\n  {'='*72}")
print(f"  Four-condition linear state comparison (batch vs sequential)")
print(f"  {'='*72}")
for cname, (hb, hs, sb, ss) in conditions.items():
    rows, _ = compare_snapshots(sb, ss)
    print(f"\n  {cname}:")
    print(f"    {'chunk':>5}  {'conv_cos':>10}  {'rec_cos':>10}")
    print(f"    {'-'*5}  {'-'*10}  {'-'*10}")
    for r in rows:
        print(f"    {r['chunk']:>5}  {r['conv_cos']:>10.6f}  {r['rec_cos']:>10.6f}")

# ══════════════════════════════════════════════════════════════
#  TEST 4: First decode step comparison
# ══════════════════════════════════════════════════════════════
print(f"\n{'#'*72}")
print(f"#  TEST 4: First decode step comparison after A vs B")
print(f"{'#'*72}")

# Re-run A and B, then do one decode step
engine._reset_states(); engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_A, logits_A_tail, _ = run_sequential_prefill_instrumented(engine, tail, engine.pos)
next_A2, logits_A2 = engine._step(next_A, engine.pos)
engine.pos += 1

engine._reset_states(); engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_B, logits_B_tail, _ = run_batch_prefill_instrumented(engine, tail, engine.pos)
next_B2, logits_B2 = engine._step(next_B, engine.pos)
engine.pos += 1

print(f"\n  After tail prefill:")
print(f"    A first token: {next_A} ({engine.tokenizer.decode([next_A])!r})")
print(f"    B first token: {next_B} ({engine.tokenizer.decode([next_B])!r})")
print(f"\n  After first decode step:")
print(f"    A second token: {next_A2} ({engine.tokenizer.decode([next_A2])!r})")
print(f"    B second token: {next_B2} ({engine.tokenizer.decode([next_B2])!r})")

if logits_A_tail is not None and logits_B_tail is not None:
    logit_cos_tail = cos_sim(logits_A_tail, logits_B_tail)
    print(f"    Tail logits cos: {logit_cos_tail:.6f}")

if logits_A2 is not None and logits_B2 is not None:
    logit_cos_dec = cos_sim(logits_A2, logits_B2)
    print(f"    Decode logits cos: {logit_cos_dec:.6f}")

# ══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'#'*72}")
print(f"#  SUMMARY")
print(f"{'#'*72}")
print(f"""
Key findings:
  1. Padding leakage: check ZEROS vs RANDOM cos above
     - If cos=1.0 at all chunks: valid_len masking works, padding does NOT leak
     - If cos<1.0: valid_len masking is broken

  2. Four-condition matrix:
     - zero_full     : expected cos ~0.999 (minimal divergence)
     - zero_partial  : expected cos ~0.999 if masking works
     - nonzero_full  : if cos drops, chunked-vs-recurrent precision diverges
     - nonzero_partial: if cos drops more, padding amplifies the issue

  3. Root cause hierarchy (check which line broke first):
     a) Cross-model (chunked vs recurrent delta rule) → hidden cos at chunk2+
     b) Non-zero state amplification → non-zero conditions much worse
     c) Padding interaction → partial worse than full with non-zero state
     d) KV cache contamination → check KV cos at chunks with full-attention layers

  Token match: A={next_A}, B={next_B}, MATCH={next_A==next_B}
""")
