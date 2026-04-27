#!/usr/bin/env python3
"""
Strict evidence-first RCA on the ACTUAL deployed CoreML path.

Compares Case A (256 batch + 35 sequential) vs Case B (256 batch + 35 batch)
with full state dumps at every chunk boundary.

Also isolates:
 - padding contribution (same valid_len, zero vs random padding)
 - non-zero state contribution (4 conditions: zero/nonzero × full/partial)
 - whether inter-layer padding hidden states leak inside a chunk

All comparisons use the actual compiled CoreML models, not PyTorch.
"""

import sys, os, copy, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine

# ─── Configuration ───────────────────────────────────────────────────
MODEL_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3'
HF_PATH   = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'
COMPUTE   = ct.ComputeUnit.CPU_ONLY  # deterministic

# ─── Helpers ─────────────────────────────────────────────────────────
def cosine(a, b):
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    if na < 1e-30 and nb < 1e-30:
        return 1.0  # both zero
    if na < 1e-30 or nb < 1e-30:
        return 0.0  # one zero
    return float(np.dot(a64, b64) / (na * nb))

def maxabs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))

def meanabs(a, b):
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def snapshot_all(engine):
    """Capture all states: KV cache, conv, rec, and return deep copies."""
    snap = {}
    for ci in range(engine.num_chunks):
        snap[ci] = {}
        # KV cache states (CoreML state objects — read via read_state)
        for sn in engine.kv_state_names:
            snap[ci][sn] = engine.states[ci].read_state(name=sn).copy()
        # Linear conv/rec states (regular numpy arrays)
        snap[ci]['conv'] = engine.lin_convs[ci].copy()
        snap[ci]['rec']  = engine.lin_recs[ci].copy()
    return snap


def compare_snaps(snap_a, snap_b, label_a="A", label_b="B", num_chunks=9):
    """Compare two snapshots, report per-chunk per-state metrics."""
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
    """Run batch prefill, returning per-chunk hidden states AND final state snapshot.

    This mirrors _batch_prefill() but captures intermediate hidden states.
    """
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
            # Fill padding with random to test leakage
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

    hiddens_per_chunk = {}  # ci -> hidden AFTER chunk
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

        # Save FULL hidden (all positions, before re-zeroing)
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
    """Run tokens sequentially using _step_kv_only / _step.
    Returns per-chunk hidden states for LAST token only.
    """
    n = len(token_ids)
    hiddens_per_chunk = {}  # ci -> hidden after chunk for last token

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
print("Loading engine...")
engine = ChatEngine(model_dir=MODEL_DIR, hf_path=HF_PATH, ctx=4096, num_chunks=9,
                    compute_unit=COMPUTE)
engine.load()
bs = engine._prefill_bs
NC = engine.num_chunks
print(f"Batch size: {bs}, Chunks: {NC}, KV states: {engine.kv_state_names}")

# Build prompt tokens: exactly bs + 35
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
print(f"Block1: {len(block1)}, Tail: {len(tail)}")


########################################################################
# TEST 1: Case A vs Case B — full state comparison at every boundary
########################################################################
print("\n" + "=" * 72)
print("  TEST 1: Case A (seq tail) vs Case B (batch tail)")
print("=" * 72)

# ─── Case A: 256 batch + 35 sequential ───
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
snap_after_block1_A = snapshot_all(engine)
pos_after_block1_A = engine.pos

next_A, logits_A, hidden_A = run_sequential_tail(engine, tail, pos_after_block1_A)
snap_A = snapshot_all(engine)

# ─── Case B: 256 batch + 35 batch ───
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
snap_after_block1_B = snapshot_all(engine)

next_B, logits_B, hidden_B = run_batch_prefill_instrumented(engine, tail, engine.pos)
snap_B = snapshot_all(engine)

# Verify block1 states are identical
print("\n  Block1 state comparison (should be identical):")
rows_b1 = compare_snaps(snap_after_block1_A, snap_after_block1_B, num_chunks=NC)
divergent_b1 = [r for r in rows_b1 if r[2] < 0.9999]
if divergent_b1:
    print_table(divergent_b1, "Block1 DIVERGENT states")
else:
    print("  All block1 states identical (cos > 0.9999) ✓")

# Compare after tail
print("\n  After-tail state comparison (A seq vs B batch):")
rows = compare_snaps(snap_A, snap_B, num_chunks=NC)
print_table(rows, "A(seq) vs B(batch) — all states after tail")

# Hidden state comparison (last token for A, last valid for B)
print("\n  Hidden states at last valid token position:")
for ci in range(NC):
    h_a = hidden_A.get(ci)
    h_b = hidden_B.get(ci)
    if h_a is None or h_b is None:
        continue
    # A has shape (1, 1, D), B has shape (1, bs, D) — extract last valid
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
print(f"\n  First decode token: A={next_A}, B={next_B}")
if logits_A is not None and logits_B is not None:
    c_logits = cosine(logits_A, logits_B)
    print(f"  Logits cosine: {c_logits:.6f}")
    topk_A = np.argsort(logits_A.ravel())[::-1][:10]
    topk_B = np.argsort(logits_B.ravel())[::-1][:10]
    print(f"  Top-10 A: {topk_A.tolist()}")
    print(f"  Top-10 B: {topk_B.tolist()}")

# Find first chunk with large divergence
first_div = None
for ci, key, c, mx, mn in rows:
    if c < 0.99 and first_div is None:
        first_div = (ci, key, c)
print(f"\n  First large divergence (cos < 0.99): chunk{first_div[0]}/{first_div[1]} cos={first_div[2]:.6f}" if first_div else "\n  No large divergence found")


########################################################################
# TEST 2: Padding leakage — direct CoreML comparison
########################################################################
print("\n" + "=" * 72)
print("  TEST 2: Padding leakage (zero vs random padding, same valid_len)")
print("=" * 72)

# Same block1, then tail with ZERO padding
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
pos_tail = engine.pos
# Save states to restore for second run
snap_before_tail_Z = snapshot_all(engine)
convs_Z = [c.copy() for c in engine.lin_convs]
recs_Z = [r.copy() for r in engine.lin_recs]

next_Z, logits_Z, hidden_Z = run_batch_prefill_instrumented(engine, tail, pos_tail, zero_padding=True)
snap_Z = snapshot_all(engine)

# Restore to pre-tail state and run with RANDOM padding
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
# Verify pre-tail states match
snap_before_tail_R = snapshot_all(engine)
pre_match = all(cosine(snap_before_tail_Z[ci][k], snap_before_tail_R[ci][k]) > 0.9999
                for ci in range(NC) for k in snap_before_tail_Z[ci])
print(f"  Pre-tail states match: {pre_match}")

next_R, logits_R, hidden_R = run_batch_prefill_instrumented(engine, tail, engine.pos, random_padding=True)
snap_R = snapshot_all(engine)

# Compare states
print("\n  Zero-padding vs Random-padding states:")
rows_pad = compare_snaps(snap_Z, snap_R, "ZERO", "RAND", NC)
print_table(rows_pad, "ZERO vs RANDOM padding — state comparison")

# Compare hidden at valid positions
print("\n  Hidden state comparison at VALID token positions:")
for ci in range(NC):
    hz = hidden_Z.get(ci)
    hr = hidden_R.get(ci)
    if hz is None or hr is None:
        continue
    # Compare ONLY valid positions (0..valid_len-1)
    vl = len(tail)
    hz_v = hz[:, :vl, :]
    hr_v = hr[:, :vl, :]
    c_val = cosine(hz_v, hr_v)
    mx_val = maxabs(hz_v, hr_v)
    # Also compare PADDING positions
    hz_p = hz[:, vl:, :]
    hr_p = hr[:, vl:, :]
    c_pad = cosine(hz_p, hr_p)
    print(f"    chunk{ci}: valid cos={c_val:.6f} max_abs={mx_val:.6f}  |  padding cos={c_pad:.6f}")

print(f"\n  Token: ZERO={next_Z}, RANDOM={next_R}")
if logits_Z is not None and logits_R is not None:
    print(f"  Logits cosine: {cosine(logits_Z, logits_R):.6f}")

# ─── Sub-test: Padding hidden state norms inside chunks ───
print("\n  Padding hidden state norms (shows intra-chunk leakage):")
vl = len(tail)
for ci in range(NC):
    hz = hidden_Z.get(ci)
    if hz is None:
        continue
    # With zero padding, how large are padding positions after chunk?
    pad_norm = float(np.linalg.norm(hz[:, vl:, :].astype(np.float64)))
    val_norm = float(np.linalg.norm(hz[:, :vl, :].astype(np.float64)))
    ratio = pad_norm / max(val_norm, 1e-30)
    print(f"    chunk{ci}: valid_norm={val_norm:.2f}  pad_norm={pad_norm:.2f}  ratio={ratio:.6f}")


########################################################################
# TEST 3: Non-zero state isolation — 4 conditions on CoreML
########################################################################
print("\n" + "=" * 72)
print("  TEST 3: Non-zero state × batch size — 4 conditions on CoreML")
print("=" * 72)

# Condition 1: zero state + full batch (256 tokens from position 0)
# Condition 2: zero state + partial batch (35 tokens from position 0)
# Condition 3: non-zero state + full batch (256 tokens from position 256, after block1)
# Condition 4: non-zero state + partial batch (35 tokens from position 256, after block1)
#
# For each condition, run the same tokens through:
#  (a) batch prefill path (prefill model)
#  (b) sequential path (infer model)
# Then compare.

test_tokens_full = tokens[:bs]   # 256 tokens
test_tokens_tail = tokens[bs:bs+35]  # 35 tokens

results_3 = {}

for cond_name, init_block, test_toks, start_pos_override in [
    ("1_zero_full",     None,   test_tokens_full, 0),
    ("2_zero_partial",  None,   test_tokens_tail, 0),
    ("3_nonzero_full",  block1, test_tokens_full, bs),
    ("4_nonzero_partial", block1, test_tokens_tail, bs),
]:
    print(f"\n  ── Condition: {cond_name} ──")

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

    # Find worst state
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

    # Print condensed per-chunk summary (conv and rec only — the linear attention states)
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
print("\n  ── TEST 3 Summary ──")
print(f"  {'condition':<20}  {'worst_cos':>10}  {'worst_loc':<20}  {'first_bad':<30}  {'tok_match':>10}")
for cn, r in results_3.items():
    print(f"  {cn:<20}  {r['worst_cos']:>10.6f}  {r['worst_loc']:<20}  {r['first_bad']:<30}  {str(r['token_match']):>10}")


########################################################################
# TEST 4: Does divergence exist immediately after tail, or grow later?
########################################################################
print("\n" + "=" * 72)
print("  TEST 4: Divergence timeline — immediate vs delayed")
print("=" * 72)

# Run Case A and B, but also check after first decode step
# Case A
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_A, logits_A, _ = run_sequential_tail(engine, tail, engine.pos)
snap_A_tail = snapshot_all(engine)
# First decode step
next_A2, logits_A2, _ = run_sequential_tail(engine, [next_A], engine.pos)
snap_A_dec1 = snapshot_all(engine)

# Case B
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_B, logits_B, _ = run_batch_prefill_instrumented(engine, tail, engine.pos)
snap_B_tail = snapshot_all(engine)
# First decode step
next_B2, logits_B2, _ = run_sequential_tail(engine, [next_B], engine.pos)
snap_B_dec1 = snapshot_all(engine)

print(f"  After tail:    A={next_A}, B={next_B}")
print(f"  After decode1: A={next_A2}, B={next_B2}")

# Compare at tail boundary
rows_tail = compare_snaps(snap_A_tail, snap_B_tail, num_chunks=NC)
worst_tail = min(rows_tail, key=lambda r: r[2])
print(f"\n  After tail:    worst cos = {worst_tail[2]:.6f} at chunk{worst_tail[0]}/{worst_tail[1]}")

# Compare after decode1
rows_dec = compare_snaps(snap_A_dec1, snap_B_dec1, num_chunks=NC)
worst_dec = min(rows_dec, key=lambda r: r[2])
print(f"  After decode1: worst cos = {worst_dec[2]:.6f} at chunk{worst_dec[0]}/{worst_dec[1]}")

if logits_A is not None and logits_B is not None:
    print(f"\n  Tail logits cos:    {cosine(logits_A, logits_B):.6f}")
if logits_A2 is not None and logits_B2 is not None:
    print(f"  Decode1 logits cos: {cosine(logits_A2, logits_B2):.6f}")


########################################################################
# TEST 5: Per-chunk divergence growth — where does it start?
########################################################################
print("\n" + "=" * 72)
print("  TEST 5: Per-chunk divergence growth in tail processing")
print("=" * 72)

# Run Case A and B but dump conv/rec after EACH chunk during tail processing
# Case A: sequential — we can only get final state per chunk
# Case B: batch — we get all states after each chunk

# Case A
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
snap_pre_tail = snapshot_all(engine)
pos_start = engine.pos

# Case A: run sequential, capture per-chunk state after ALL tail tokens
next_A_t5, _, _ = run_sequential_tail(engine, tail, pos_start)
snap_A_t5 = snapshot_all(engine)

# Case B: run batch, capture per-chunk state
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
next_B_t5, _, _ = run_batch_prefill_instrumented(engine, tail, engine.pos)
snap_B_t5 = snapshot_all(engine)

# Now compare per-chunk: pre-tail → post-tail change for A and B
print("\n  Pre-tail to Post-tail state CHANGE — compared between A and B:")
print(f"  {'chunk':>5}  {'state':<12}  {'A_change':>10}  {'B_change':>10}  {'A_vs_B':>10}")
print(f"  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")
for ci in range(NC):
    for key in snap_A_t5[ci]:
        pre = snap_pre_tail[ci][key]
        a_post = snap_A_t5[ci][key]
        b_post = snap_B_t5[ci][key]
        delta_a = a_post.astype(np.float64) - pre.astype(np.float64)
        delta_b = b_post.astype(np.float64) - pre.astype(np.float64)
        a_change = float(np.linalg.norm(delta_a))
        b_change = float(np.linalg.norm(delta_b))
        # Compare the deltas themselves
        c_delta = cosine(delta_a, delta_b) if a_change > 1e-10 and b_change > 1e-10 else 1.0
        ab_cos = cosine(a_post, b_post)
        if a_change > 1e-6 or b_change > 1e-6:
            print(f"  {ci:>5}  {key:<12}  {a_change:>10.4f}  {b_change:>10.4f}  {ab_cos:>10.6f}")


########################################################################
# TEST 6: conv_state + recurrent_state padding audit
########################################################################
print("\n" + "=" * 72)
print("  TEST 6: Direct padding effect audit on conv/rec states")
print("=" * 72)
print("  Running tail batch with valid_len=35 and inspecting whether")
print("  padding positions affect conv_state and recurrent_state.")

# Run 1: zero padding
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
convs_pre_z = [c.copy() for c in engine.lin_convs]
recs_pre_z = [r.copy() for r in engine.lin_recs]
next_z6, _, _ = run_batch_prefill_instrumented(engine, tail, engine.pos, zero_padding=True)
convs_zero = [c.copy() for c in engine.lin_convs]
recs_zero = [r.copy() for r in engine.lin_recs]

# Run 2: random padding (fresh from identical starting state)
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
convs_pre_r = [c.copy() for c in engine.lin_convs]
recs_pre_r = [r.copy() for r in engine.lin_recs]
next_r6, _, _ = run_batch_prefill_instrumented(engine, tail, engine.pos, random_padding=True)
convs_rand = [c.copy() for c in engine.lin_convs]
recs_rand = [r.copy() for r in engine.lin_recs]

print(f"\n  Token: ZERO_pad={next_z6}, RAND_pad={next_r6}")
print(f"\n  Conv state: ZERO vs RANDOM padding")
for ci in range(NC):
    c = cosine(convs_zero[ci], convs_rand[ci])
    mx = maxabs(convs_zero[ci], convs_rand[ci])
    change_z = float(np.linalg.norm((convs_zero[ci] - convs_pre_z[ci]).astype(np.float64)))
    change_r = float(np.linalg.norm((convs_rand[ci] - convs_pre_r[ci]).astype(np.float64)))
    print(f"    chunk{ci}: cos={c:.8f}  max_abs={mx:.8f}  Δ_zero={change_z:.4f}  Δ_rand={change_r:.4f}")

print(f"\n  Recurrent state: ZERO vs RANDOM padding")
for ci in range(NC):
    c = cosine(recs_zero[ci], recs_rand[ci])
    mx = maxabs(recs_zero[ci], recs_rand[ci])
    change_z = float(np.linalg.norm((recs_zero[ci] - recs_pre_z[ci]).astype(np.float64)))
    change_r = float(np.linalg.norm((recs_rand[ci] - recs_pre_r[ci]).astype(np.float64)))
    print(f"    chunk{ci}: cos={c:.8f}  max_abs={mx:.8f}  Δ_zero={change_z:.4f}  Δ_rand={change_r:.4f}")


########################################################################
# SUMMARY
########################################################################
print("\n" + "=" * 72)
print("  ALL TESTS COMPLETE")
print("=" * 72)
