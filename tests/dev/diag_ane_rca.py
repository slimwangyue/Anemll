#!/usr/bin/env python3
"""
ANE-path RCA diagnostics for batch tail prefill divergence.

Same structure as diag_deployed_rca.py but runs on CPU_AND_NE (ANE path)
instead of CPU_ONLY.

Tests:
  0) ANE determinism baseline — same run twice, check exact match
  1) Case A (seq tail) vs Case B (batch tail) on ANE
  2) Padding isolation on ANE (zero vs random padding)
  3) 4-condition matrix on ANE (zero/nonzero × full/partial)
  4) Divergence timeline on ANE
  5) Per-chunk divergence growth on ANE
  6) Direct padding audit on conv/rec states (ANE)
"""

import sys, os, copy, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine

# ─── Configuration ───────────────────────────────────────────────────
MODEL_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3'
HF_PATH   = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'
COMPUTE   = ct.ComputeUnit.CPU_AND_NE  # *** ANE path ***

# ─── Helpers ─────────────────────────────────────────────────────────
def cosine(a, b):
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    if na < 1e-30 and nb < 1e-30:
        return 1.0
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))

def maxabs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))

def meanabs(a, b):
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def snapshot_all(engine):
    snap = {}
    for ci in range(engine.num_chunks):
        snap[ci] = {}
        for sn in engine.kv_state_names:
            snap[ci][sn] = engine.states[ci].read_state(name=sn).copy()
        snap[ci]['conv'] = engine.lin_convs[ci].copy()
        snap[ci]['rec']  = engine.lin_recs[ci].copy()
    return snap


def compare_snaps(snap_a, snap_b, label_a="A", label_b="B", num_chunks=9):
    rows = []
    for ci in range(num_chunks):
        if ci not in snap_a or ci not in snap_b:
            continue
        for key in snap_a[ci]:
            a = snap_a[ci][key]
            b = snap_b[ci][key]
            c = cosine(a, b)
            mx = maxabs(a, b)
            mn = meanabs(a, b)
            rows.append((ci, key, c, mx, mn))
    return rows


def print_table(rows, title=""):
    if title:
        print(f"\n  {title}")
    print(f"  {'chunk':>5}  {'state':<12}  {'cosine':>10}  {'max_abs':>12}  {'mean_abs':>12}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*12}")
    for ci, key, c, mx, mn in rows:
        print(f"  {ci:>5}  {key:<12}  {c:>10.6f}  {mx:>12.6f}  {mn:>12.8f}")


def run_batch_prefill_instrumented(engine, token_ids, block_start, zero_padding=True, random_padding=False):
    """Run batch prefill, returning per-chunk hidden states AND final state snapshot."""
    valid_len = len(token_ids)
    bs = engine._prefill_bs
    assert 1 <= valid_len <= bs

    # Embed
    input_ids = engine._batch_tok_buf.copy()
    input_ids[0, :] = 0
    input_ids[0, :valid_len] = token_ids
    hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
    if valid_len < bs:
        if random_padding:
            hidden[:, valid_len:, :] = np.random.randn(1, bs - valid_len, hidden.shape[2]).astype(hidden.dtype) * 0.1
        else:
            hidden[:, valid_len:, :] = 0.0

    # Mask
    mask = engine._batch_mask_buf.copy()
    mask[:, :, :, :] = -65504.0
    for i in range(valid_len):
        mask[0, 0, i, :block_start + i + 1] = 0
    for i in range(valid_len, bs):
        mask[0, 0, i, 0] = 0.0

    # Position IDs
    pos_ids = engine._batch_pos_buf.copy()
    pos_ids[:valid_len] = np.arange(block_start + engine.rope_offset,
                                     block_start + engine.rope_offset + valid_len, dtype=np.int32)
    pos_ids[valid_len:] = 0

    cur_pos = engine._batch_cur_buf.copy()
    cur_pos[0] = block_start

    valid_len_arr = engine._valid_len_buf.copy()
    valid_len_arr[0] = valid_len

    hiddens_per_chunk = {}
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

        hiddens_per_chunk[ci] = hidden.copy()

        if valid_len < bs:
            hidden[:, valid_len:, :] = 0.0

    # LM head on last valid token
    if hidden.ndim >= 3 and hidden.shape[1] > 1:
        last_h = hidden[:, valid_len - 1:valid_len, :]
    else:
        last_h = hidden
    lm_out = engine.lmhead.predict({"hidden_states": last_h.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits = engine._extract_logits(lm_out)
        next_id = int(np.argmax(logits))
    else:
        logits = None
        next_id = int(lm_out["argmax_idx"].flatten()[0])

    engine.pos = block_start + valid_len
    return next_id, logits, hiddens_per_chunk


def run_sequential_tail(engine, token_ids, start_pos):
    """Run tokens sequentially using infer model."""
    n = len(token_ids)
    hiddens_per_chunk = {}

    for ti, tok_id in enumerate(token_ids):
        is_last = (ti == n - 1)
        pos = start_pos + ti
        tok = engine._tok_buf
        tok[0, 0] = tok_id
        hidden = list(engine.embed.predict({"input_ids": tok}).values())[0]

        mask = engine._mask_buf
        mask[:, :, :, :] = -65504.0
        mask[:, :, :, :pos + 1] = 0

        pos_arr = engine._pos_buf
        pos_arr[0] = pos

        rope_arr = engine._rope_buf
        rope_arr[0] = pos + engine.rope_offset

        for ci in range(engine.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": rope_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": engine.lin_convs[ci],
                "linear_recurrent_state": engine.lin_recs[ci],
            }
            out = engine.ffns[ci].predict(inp, state=engine.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                engine.lin_convs[ci] = out['linear_conv_state_out']
                engine.lin_recs[ci] = out['linear_recurrent_state_out']

            if is_last:
                hiddens_per_chunk[ci] = hidden.copy()

        engine.pos = pos + 1

    # Get logits for last token
    lm_out = engine.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits = engine._extract_logits(lm_out)
        next_id = int(np.argmax(logits))
    else:
        logits = None
        next_id = int(lm_out["argmax_idx"].flatten()[0])

    return next_id, logits, hiddens_per_chunk


# ─── Main ────────────────────────────────────────────────────────────
print("=" * 72)
print("  ANE-PATH RCA DIAGNOSTICS")
print(f"  Compute unit: {COMPUTE}")
print("=" * 72)

print("\nLoading engine on ANE...")
print("  Using separate .mlmodelc files to avoid runtime ANE compilation")
t_load = time.time()
engine = ChatEngine(model_dir=MODEL_DIR, hf_path=HF_PATH, ctx=4096, num_chunks=9,
                    compute_unit=COMPUTE)
# Force separate model loading: use pre-compiled .mlmodelc files only.
# This avoids runtime ANE compilation which fails when boot volume is low on space.
# Monkey-patch _find_model to prefer .mlmodelc over .mlpackage for this session.
import scripts_qwen3_5.chat_server as _cs
_orig_find_model = _cs._find_model
def _find_model_mlmodelc_first(base_dir, name):
    """Prefer .mlmodelc (pre-compiled, no runtime compilation needed)."""
    import os as _os
    for ext in (".mlmodelc", ".mlpackage"):
        p = _os.path.join(base_dir, name + ext)
        if _os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")
_cs._find_model = _find_model_mlmodelc_first
# Also force separate (non-combined) loading
engine.use_combined = False
engine.combined_dir = None
engine.load()
# Restore original
_cs._find_model = _orig_find_model
bs = engine._prefill_bs
NC = engine.num_chunks
print(f"  Loaded in {time.time()-t_load:.1f}s")
print(f"  Batch size: {bs}, Chunks: {NC}, KV states: {engine.kv_state_names}")

# Build prompt tokens
prompt = (
    "Please translate the following passage into Chinese.\n\n"
    "In recent work on efficient transformer inference, a method referred to as "
    "Hierarchical Context Distillation has been proposed to address the growing "
    "cost of long-context processing. The approach combines ideas from knowledge "
    "distillation with hierarchical attention mechanisms to produce more compact "
    "representations during inference."
)
messages = [{'role': 'user', 'content': prompt}]
tokens = engine._tokenize_messages(messages, enable_thinking=False)
target = bs + 35
while len(tokens) < target:
    prompt += " This approach reduces memory footprint significantly."
    messages = [{'role': 'user', 'content': prompt}]
    tokens = engine._tokenize_messages(messages, enable_thinking=False)
tokens = tokens[:target]
block1, tail = tokens[:bs], tokens[bs:]
print(f"  Block1: {len(block1)}, Tail: {len(tail)}")


########################################################################
# TEST 0: ANE Determinism Baseline
########################################################################
print("\n" + "=" * 72)
print("  TEST 0: ANE Determinism Baseline")
print("  Running same operation twice to check if ANE is deterministic")
print("=" * 72)

# Run 1
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_d1, logits_d1, hidden_d1 = run_batch_prefill_instrumented(engine, tail, engine.pos)
snap_d1 = snapshot_all(engine)

# Run 2 (identical)
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_d2, logits_d2, hidden_d2 = run_batch_prefill_instrumented(engine, tail, engine.pos)
snap_d2 = snapshot_all(engine)

print(f"\n  Tokens: run1={next_d1}, run2={next_d2}, match={next_d1==next_d2}")
rows_det = compare_snaps(snap_d1, snap_d2, "run1", "run2", NC)
non_identical = [(ci,k,c,mx,mn) for ci,k,c,mx,mn in rows_det if c < 1.0 or mx > 0]
if non_identical:
    print(f"  ANE NON-DETERMINISM DETECTED ({len(non_identical)} non-identical states):")
    print_table(non_identical, "Non-identical states between identical runs")
else:
    print("  ANE is DETERMINISTIC — all states identical across runs ✓")

# Also check hidden states
print("  Hidden determinism:")
for ci in range(NC):
    h1 = hidden_d1.get(ci)
    h2 = hidden_d2.get(ci)
    if h1 is not None and h2 is not None:
        c = cosine(h1, h2)
        mx = maxabs(h1, h2)
        if c < 1.0 or mx > 0:
            print(f"    chunk{ci}: cos={c:.8f}  max_abs={mx:.8f}  ← NON-DETERMINISTIC")
        else:
            print(f"    chunk{ci}: IDENTICAL")

if logits_d1 is not None and logits_d2 is not None:
    c_l = cosine(logits_d1, logits_d2)
    mx_l = maxabs(logits_d1, logits_d2)
    print(f"  Logits: cos={c_l:.8f}  max_abs={mx_l:.8f}")


########################################################################
# TEST 1: Case A vs Case B on ANE
########################################################################
print("\n" + "=" * 72)
print("  TEST 1: Case A (seq tail) vs Case B (batch tail) — ANE")
print("=" * 72)

# ─── Case A: 256 batch + 35 sequential ───
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
snap_after_block1_A = snapshot_all(engine)

next_A, logits_A, hidden_A = run_sequential_tail(engine, tail, engine.pos)
snap_A = snapshot_all(engine)

# ─── Case B: 256 batch + 35 batch ───
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
snap_after_block1_B = snapshot_all(engine)

next_B, logits_B, hidden_B = run_batch_prefill_instrumented(engine, tail, engine.pos)
snap_B = snapshot_all(engine)

# Verify block1 states identical
print("\n  Block1 state comparison (should be identical):")
rows_b1 = compare_snaps(snap_after_block1_A, snap_after_block1_B, num_chunks=NC)
divergent_b1 = [r for r in rows_b1 if r[2] < 0.9999]
if divergent_b1:
    print_table(divergent_b1, "Block1 DIVERGENT states (ANE non-determinism?)")
else:
    print("  All block1 states identical (cos > 0.9999) ✓")

# Compare after tail
print("\n  After-tail state comparison (A seq vs B batch) — ANE:")
rows = compare_snaps(snap_A, snap_B, num_chunks=NC)
print_table(rows, "A(seq) vs B(batch) — all states after tail [ANE]")

# Hidden state comparison
print("\n  Hidden states at last valid token position [ANE]:")
for ci in range(NC):
    h_a = hidden_A.get(ci)
    h_b = hidden_B.get(ci)
    if h_a is None or h_b is None:
        continue
    if h_b.ndim >= 3 and h_b.shape[1] > 1:
        h_b_last = h_b[:, len(tail) - 1:len(tail), :]
    else:
        h_b_last = h_b
    if h_a.ndim >= 3 and h_a.shape[1] > 1:
        h_a = h_a[:, -1:, :]
    c = cosine(h_a, h_b_last)
    mx = maxabs(h_a, h_b_last)
    print(f"    chunk{ci}: cos={c:.6f}  max_abs={mx:.6f}")

# Token comparison
print(f"\n  First decode token: A={next_A}, B={next_B}, match={next_A==next_B}")
if logits_A is not None and logits_B is not None:
    c_logits = cosine(logits_A, logits_B)
    print(f"  Logits cosine: {c_logits:.6f}")
    topk_A = np.argsort(logits_A.ravel())[::-1][:10]
    topk_B = np.argsort(logits_B.ravel())[::-1][:10]
    print(f"  Top-10 A: {topk_A.tolist()}")
    print(f"  Top-10 B: {topk_B.tolist()}")

first_div = None
for ci, key, c, mx, mn in rows:
    if c < 0.99 and first_div is None:
        first_div = (ci, key, c)
print(f"\n  First large divergence (cos < 0.99): chunk{first_div[0]}/{first_div[1]} cos={first_div[2]:.6f}" if first_div else "\n  No large divergence found")


########################################################################
# TEST 2: Padding leakage on ANE
########################################################################
print("\n" + "=" * 72)
print("  TEST 2: Padding leakage (zero vs random padding) — ANE")
print("=" * 72)

# Zero padding
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
pos_tail = engine.pos
snap_before_tail_Z = snapshot_all(engine)

next_Z, logits_Z, hidden_Z = run_batch_prefill_instrumented(engine, tail, pos_tail, zero_padding=True)
snap_Z = snapshot_all(engine)

# Random padding (fresh from identical starting state)
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
snap_before_tail_R = snapshot_all(engine)
pre_match = all(cosine(snap_before_tail_Z[ci][k], snap_before_tail_R[ci][k]) > 0.9999
                for ci in range(NC) for k in snap_before_tail_Z[ci])
print(f"  Pre-tail states match: {pre_match}")

# Check if pre-tail states are EXACTLY identical on ANE (determinism)
pre_exact = all(maxabs(snap_before_tail_Z[ci][k], snap_before_tail_R[ci][k]) == 0.0
                for ci in range(NC) for k in snap_before_tail_Z[ci])
print(f"  Pre-tail states EXACTLY identical: {pre_exact}")
if not pre_exact:
    print("  *** ANE non-determinism in block1 — padding test results will include this noise ***")
    for ci in range(NC):
        for k in snap_before_tail_Z[ci]:
            mx = maxabs(snap_before_tail_Z[ci][k], snap_before_tail_R[ci][k])
            if mx > 0:
                c = cosine(snap_before_tail_Z[ci][k], snap_before_tail_R[ci][k])
                print(f"    chunk{ci}/{k}: cos={c:.8f} max_abs={mx:.8f}")

next_R, logits_R, hidden_R = run_batch_prefill_instrumented(engine, tail, engine.pos, random_padding=True)
snap_R = snapshot_all(engine)

# Compare states
print("\n  Zero-padding vs Random-padding states [ANE]:")
rows_pad = compare_snaps(snap_Z, snap_R, "ZERO", "RAND", NC)
print_table(rows_pad, "ZERO vs RANDOM padding — state comparison [ANE]")

# Compare hidden at valid positions
print("\n  Hidden state comparison at VALID token positions [ANE]:")
for ci in range(NC):
    hz = hidden_Z.get(ci)
    hr = hidden_R.get(ci)
    if hz is None or hr is None:
        continue
    vl = len(tail)
    hz_v = hz[:, :vl, :]
    hr_v = hr[:, :vl, :]
    c_val = cosine(hz_v, hr_v)
    mx_val = maxabs(hz_v, hr_v)
    hz_p = hz[:, vl:, :]
    hr_p = hr[:, vl:, :]
    if hz_p.size > 0 and hr_p.size > 0:
        c_pad = cosine(hz_p, hr_p)
        mx_pad = maxabs(hz_p, hr_p)
        print(f"    chunk{ci}: valid cos={c_val:.6f} max_abs={mx_val:.6f}  |  padding cos={c_pad:.6f} max_abs={mx_pad:.6f}")
    else:
        print(f"    chunk{ci}: valid cos={c_val:.6f} max_abs={mx_val:.6f}  |  (no padding — last chunk outputs 1 token)")

print(f"\n  Token: ZERO={next_Z}, RANDOM={next_R}, match={next_Z==next_R}")
if logits_Z is not None and logits_R is not None:
    c_l = cosine(logits_Z, logits_R)
    mx_l = maxabs(logits_Z, logits_R)
    print(f"  Logits cosine: {c_l:.6f}  max_abs: {mx_l:.6f}")

# Padding hidden norms
print("\n  Padding hidden state norms (intra-chunk leakage) [ANE]:")
vl = len(tail)
for ci in range(NC):
    hz = hidden_Z.get(ci)
    if hz is None:
        continue
    pad_part = hz[:, vl:, :]
    if pad_part.size > 0:
        pad_norm = float(np.linalg.norm(pad_part.astype(np.float64)))
    else:
        pad_norm = 0.0
    val_norm = float(np.linalg.norm(hz[:, :vl, :].astype(np.float64)))
    ratio = pad_norm / max(val_norm, 1e-30)
    print(f"    chunk{ci}: valid_norm={val_norm:.2f}  pad_norm={pad_norm:.2f}  ratio={ratio:.6f}")


########################################################################
# TEST 3: Non-zero state isolation — 4 conditions on ANE
########################################################################
print("\n" + "=" * 72)
print("  TEST 3: Non-zero state × batch size — 4 conditions [ANE]")
print("=" * 72)

test_tokens_full = tokens[:bs]
test_tokens_tail = tokens[bs:bs+35]

results_3 = {}

for cond_name, init_block, test_toks, start_pos_override in [
    ("1_zero_full",     None,   test_tokens_full, 0),
    ("2_zero_partial",  None,   test_tokens_tail, 0),
    ("3_nonzero_full",  block1, test_tokens_full, bs),
    ("4_nonzero_partial", block1, test_tokens_tail, bs),
]:
    print(f"\n  ── Condition: {cond_name} [ANE] ──")

    # (a) Batch prefill
    engine._reset_states()
    engine.pos = 0
    if init_block is not None:
        engine._batch_prefill(init_block, 0)
    start_pos = engine.pos if init_block else start_pos_override
    next_batch, logits_batch, hidden_batch = run_batch_prefill_instrumented(
        engine, test_toks, start_pos)
    snap_batch = snapshot_all(engine)

    # (b) Sequential
    engine._reset_states()
    engine.pos = 0
    if init_block is not None:
        engine._batch_prefill(init_block, 0)
    start_pos = engine.pos if init_block else start_pos_override
    next_seq, logits_seq, hidden_seq = run_sequential_tail(engine, test_toks, start_pos)
    snap_seq = snapshot_all(engine)

    # Compare
    rows_c = compare_snaps(snap_batch, snap_seq, "batch", "seq", NC)

    worst = min(rows_c, key=lambda r: r[2])
    first_bad = None
    for r in rows_c:
        if r[2] < 0.99:
            first_bad = r
            break

    results_3[cond_name] = {
        'worst_cos': worst[2],
        'worst_loc': f"chunk{worst[0]}/{worst[1]}",
        'first_bad': f"chunk{first_bad[0]}/{first_bad[1]} cos={first_bad[2]:.4f}" if first_bad else "none",
        'token_match': next_batch == next_seq,
        'next_batch': next_batch,
        'next_seq': next_seq,
    }

    # Print per-chunk summary
    for ci in range(NC):
        conv_row = next((r for r in rows_c if r[0] == ci and r[1] == 'conv'), None)
        rec_row = next((r for r in rows_c if r[0] == ci and r[1] == 'rec'), None)
        kv_rows = [r for r in rows_c if r[0] == ci and r[1] not in ('conv', 'rec')]
        conv_s = f"conv={conv_row[2]:.6f}" if conv_row else "conv=N/A"
        rec_s = f"rec={rec_row[2]:.6f}" if rec_row else "rec=N/A"
        kv_s = "  ".join(f"{r[1]}={r[2]:.6f}" for r in kv_rows)
        print(f"    chunk{ci}: {conv_s}  {rec_s}  {kv_s}")

    # Hidden at last token
    for ci in range(NC):
        hb = hidden_batch.get(ci)
        hs = hidden_seq.get(ci)
        if hb is None or hs is None:
            continue
        if hb.ndim >= 3 and hb.shape[1] > 1:
            hb = hb[:, len(test_toks) - 1:len(test_toks), :]
        if hs.ndim >= 3 and hs.shape[1] > 1:
            hs = hs[:, -1:, :]
        c = cosine(hb, hs)
        if c < 0.999:
            print(f"    chunk{ci} hidden(last): cos={c:.6f}")

    print(f"    Token: batch={next_batch}, seq={next_seq}, match={next_batch == next_seq}")

# Summary table
print("\n  ── TEST 3 Summary [ANE] ──")
print(f"  {'condition':<20}  {'worst_cos':>10}  {'worst_loc':<20}  {'first_bad':<30}  {'tok_match':>10}")
for cn, r in results_3.items():
    print(f"  {cn:<20}  {r['worst_cos']:>10.6f}  {r['worst_loc']:<20}  {r['first_bad']:<30}  {str(r['token_match']):>10}")


########################################################################
# TEST 4: Divergence timeline on ANE
########################################################################
print("\n" + "=" * 72)
print("  TEST 4: Divergence timeline — ANE")
print("=" * 72)

# Case A
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_A4, logits_A4, _ = run_sequential_tail(engine, tail, engine.pos)
snap_A_tail4 = snapshot_all(engine)
next_A42, logits_A42, _ = run_sequential_tail(engine, [next_A4], engine.pos)
snap_A_dec4 = snapshot_all(engine)

# Case B
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_B4, logits_B4, _ = run_batch_prefill_instrumented(engine, tail, engine.pos)
snap_B_tail4 = snapshot_all(engine)
next_B42, logits_B42, _ = run_sequential_tail(engine, [next_B4], engine.pos)
snap_B_dec4 = snapshot_all(engine)

print(f"  After tail:    A={next_A4}, B={next_B4}")
print(f"  After decode1: A={next_A42}, B={next_B42}")

rows_tail4 = compare_snaps(snap_A_tail4, snap_B_tail4, num_chunks=NC)
worst_tail4 = min(rows_tail4, key=lambda r: r[2])
print(f"\n  After tail:    worst cos = {worst_tail4[2]:.6f} at chunk{worst_tail4[0]}/{worst_tail4[1]}")

rows_dec4 = compare_snaps(snap_A_dec4, snap_B_dec4, num_chunks=NC)
worst_dec4 = min(rows_dec4, key=lambda r: r[2])
print(f"  After decode1: worst cos = {worst_dec4[2]:.6f} at chunk{worst_dec4[0]}/{worst_dec4[1]}")

if logits_A4 is not None and logits_B4 is not None:
    print(f"\n  Tail logits cos:    {cosine(logits_A4, logits_B4):.6f}")
if logits_A42 is not None and logits_B42 is not None:
    print(f"  Decode1 logits cos: {cosine(logits_A42, logits_B42):.6f}")


########################################################################
# TEST 5: Per-chunk divergence growth on ANE
########################################################################
print("\n" + "=" * 72)
print("  TEST 5: Per-chunk divergence growth in tail processing — ANE")
print("=" * 72)

# Case A
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
snap_pre_tail5 = snapshot_all(engine)
pos_start5 = engine.pos

next_A5, _, _ = run_sequential_tail(engine, tail, pos_start5)
snap_A5 = snapshot_all(engine)

# Case B
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_B5, _, _ = run_batch_prefill_instrumented(engine, tail, engine.pos)
snap_B5 = snapshot_all(engine)

print("\n  Pre-tail to Post-tail state CHANGE — A vs B [ANE]:")
print(f"  {'chunk':>5}  {'state':<12}  {'A_change':>10}  {'B_change':>10}  {'A_vs_B':>10}")
print(f"  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")
for ci in range(NC):
    for key in snap_A5[ci]:
        pre = snap_pre_tail5[ci][key]
        a_post = snap_A5[ci][key]
        b_post = snap_B5[ci][key]
        delta_a = a_post.astype(np.float64) - pre.astype(np.float64)
        delta_b = b_post.astype(np.float64) - pre.astype(np.float64)
        a_change = float(np.linalg.norm(delta_a))
        b_change = float(np.linalg.norm(delta_b))
        ab_cos = cosine(a_post, b_post)
        if a_change > 1e-6 or b_change > 1e-6:
            print(f"  {ci:>5}  {key:<12}  {a_change:>10.4f}  {b_change:>10.4f}  {ab_cos:>10.6f}")


########################################################################
# TEST 6: Direct padding audit on conv/rec states (ANE)
########################################################################
print("\n" + "=" * 72)
print("  TEST 6: Direct padding effect on conv/rec states — ANE")
print("=" * 72)

# Run 1: zero padding
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
convs_pre_z6 = [c.copy() for c in engine.lin_convs]
recs_pre_z6 = [r.copy() for r in engine.lin_recs]
next_z6, _, _ = run_batch_prefill_instrumented(engine, tail, engine.pos, zero_padding=True)
convs_zero6 = [c.copy() for c in engine.lin_convs]
recs_zero6 = [r.copy() for r in engine.lin_recs]

# Run 2: random padding
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
convs_pre_r6 = [c.copy() for c in engine.lin_convs]
recs_pre_r6 = [r.copy() for r in engine.lin_recs]
next_r6, _, _ = run_batch_prefill_instrumented(engine, tail, engine.pos, random_padding=True)
convs_rand6 = [c.copy() for c in engine.lin_convs]
recs_rand6 = [r.copy() for r in engine.lin_recs]

print(f"\n  Token: ZERO_pad={next_z6}, RAND_pad={next_r6}")
print(f"\n  Conv state: ZERO vs RANDOM padding [ANE]")
for ci in range(NC):
    c = cosine(convs_zero6[ci], convs_rand6[ci])
    mx = maxabs(convs_zero6[ci], convs_rand6[ci])
    change_z = float(np.linalg.norm((convs_zero6[ci] - convs_pre_z6[ci]).astype(np.float64)))
    change_r = float(np.linalg.norm((convs_rand6[ci] - convs_pre_r6[ci]).astype(np.float64)))
    print(f"    chunk{ci}: cos={c:.8f}  max_abs={mx:.8f}  delta_zero={change_z:.4f}  delta_rand={change_r:.4f}")

print(f"\n  Recurrent state: ZERO vs RANDOM padding [ANE]")
for ci in range(NC):
    c = cosine(recs_zero6[ci], recs_rand6[ci])
    mx = maxabs(recs_zero6[ci], recs_rand6[ci])
    change_z = float(np.linalg.norm((recs_zero6[ci] - recs_pre_z6[ci]).astype(np.float64)))
    change_r = float(np.linalg.norm((recs_rand6[ci] - recs_pre_r6[ci]).astype(np.float64)))
    print(f"    chunk{ci}: cos={c:.8f}  max_abs={mx:.8f}  delta_zero={change_z:.4f}  delta_rand={change_r:.4f}")


########################################################################
# BONUS: ANE vs CPU_ONLY comparison for block1 (identical operation)
########################################################################
print("\n" + "=" * 72)
print("  BONUS: Checking if this ANE run differs from CPU_ONLY baseline")
print("  (This is a sanity check — if ANE != CPU, the mechanism may differ)")
print("=" * 72)

# Load CPU_ONLY engine for comparison
print("  Loading CPU_ONLY engine for cross-comparison...")
engine_cpu = ChatEngine(model_dir=MODEL_DIR, hf_path=HF_PATH, ctx=4096, num_chunks=9,
                        compute_unit=ct.ComputeUnit.CPU_ONLY)
engine_cpu.use_combined = False
engine_cpu.combined_dir = None
_cs._find_model = _find_model_mlmodelc_first
engine_cpu.load()
_cs._find_model = _orig_find_model

# Run identical Case B on CPU_ONLY
engine_cpu._reset_states()
engine_cpu.pos = 0
_ = engine_cpu._batch_prefill(block1, 0)
next_B_cpu, logits_B_cpu, hidden_B_cpu = run_batch_prefill_instrumented(engine_cpu, tail, engine_cpu.pos)
snap_B_cpu = snapshot_all(engine_cpu)

# Run identical Case B on ANE (already have snap_B from TEST 1)
# Compare ANE snap_B vs CPU snap_B
print("\n  ANE vs CPU_ONLY — Case B states:")
rows_ane_cpu = compare_snaps(snap_B, snap_B_cpu, "ANE", "CPU", NC)
print_table(rows_ane_cpu, "ANE vs CPU_ONLY — batch tail states")

print(f"\n  Tokens: ANE={next_B}, CPU={next_B_cpu}")
if logits_B is not None and logits_B_cpu is not None:
    print(f"  Logits cos: {cosine(logits_B, logits_B_cpu):.6f}")

# Also compare Case A on CPU
engine_cpu._reset_states()
engine_cpu.pos = 0
_ = engine_cpu._batch_prefill(block1, 0)
next_A_cpu, logits_A_cpu, _ = run_sequential_tail(engine_cpu, tail, engine_cpu.pos)
snap_A_cpu = snapshot_all(engine_cpu)

print("\n  ANE vs CPU_ONLY — Case A states:")
rows_ane_cpu_a = compare_snaps(snap_A, snap_A_cpu, "ANE", "CPU", NC)
non_match = [(ci,k,c,mx,mn) for ci,k,c,mx,mn in rows_ane_cpu_a if c < 0.9999]
if non_match:
    print_table(non_match, "ANE vs CPU_ONLY divergent states (Case A)")
else:
    print("  Case A: ANE ≈ CPU_ONLY (all cos > 0.9999)")

print(f"  Tokens: ANE_A={next_A}, CPU_A={next_A_cpu}")


########################################################################
# SUMMARY
########################################################################
print("\n" + "=" * 72)
print("  ALL ANE TESTS COMPLETE")
print("=" * 72)
