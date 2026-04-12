#!/usr/bin/env python3
"""
3-way KV-cache compaction comparison.

Compares three compaction strategies against the no-compaction baseline:

  A) KV shift + linear keep-as-is     (v1 — inconsistent memory systems)
  B) KV shift + linear rebuild         (v2 — consistent, shift + replay linear only)
  C) Full reset + replay               (v3 — reset everything, replay from scratch)

Strategy B works by:
  1. Shift KV cache arrays right→left via read_state/write_state
  2. Save the shifted KV cache to numpy arrays
  3. Reset ALL states (KV + linear) to zero
  4. Replay kept tokens sequentially (rebuilds KV + linear from scratch)
  5. Restore saved shifted KV cache (overwrite replayed KV)
  => Linear states are now consistent with the kept window
  => KV cache preserves original attention embeddings

Usage:
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/kv_cache_compaction_3way.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = "qwen3_5_stable_lut4ffn_lut6em_fp32"
NUM_CHUNKS = 9
CTX = 2048
GEN_BEFORE = 20     # tokens to generate before compaction
GEN_AFTER  = 30     # tokens to generate after compaction
COMPUTE_UNIT = ct.ComputeUnit.ALL  # Use CPU_ONLY if boot volume is low on space

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
banner("Loading models")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

print("Loading embed/lmhead (CPU_ONLY)...")
# Use separate pre-compiled models to avoid boot-volume space issues
embed_c = os.path.join(MODEL_DIR, "embed_single.mlmodelc")
lmhead_c = os.path.join(MODEL_DIR, "lm_head_nosplit.mlmodelc")
embed_p = os.path.join(MODEL_DIR, "embed_single.mlpackage")
lmhead_p = os.path.join(MODEL_DIR, "lm_head_nosplit.mlpackage")
if os.path.isdir(embed_c) and os.path.isdir(lmhead_c):
    embed  = ct.models.CompiledMLModel(embed_c, compute_units=ct.ComputeUnit.CPU_ONLY)
    lmhead = ct.models.CompiledMLModel(lmhead_c, compute_units=ct.ComputeUnit.CPU_ONLY)
else:
    combined_el = os.path.join(MODEL_DIR, "embed_lmhead_combined.mlpackage")
    embed  = ct.models.MLModel(combined_el, compute_units=ct.ComputeUnit.CPU_ONLY, function_name="embed")
    lmhead = ct.models.MLModel(combined_el, compute_units=ct.ComputeUnit.CPU_ONLY, function_name="lmhead")

ffns = []
for ci in range(NUM_CHUNKS):
    # Prefer pre-compiled .mlmodelc (loads via CompiledMLModel, avoids boot-volume space)
    path_c = os.path.join(MODEL_DIR, f"ffn_LUT4_chunk{ci}.mlmodelc")
    path_p = os.path.join(MODEL_DIR, f"ffn_LUT4_chunk{ci}.mlpackage")
    print(f"  chunk {ci}...", end="", flush=True)
    t0 = time.time()
    if os.path.isdir(path_c):
        ffns.append(ct.models.CompiledMLModel(path_c, compute_units=COMPUTE_UNIT))
    else:
        ffns.append(ct.models.MLModel(path_p, compute_units=COMPUTE_UNIT))
    print(f" {time.time()-t0:.0f}s")

# Detect shapes — try get_spec first, fall back to known shapes
# Known shapes for Qwen3.5-4B [FLLL] 9-chunk partition:
KNOWN_CONV = {0: (3,1024,32), 8: (1,1024,32)}  # chunks 1-7 are (4,1024,32)
KNOWN_REC  = {0: (3,32,128,128), 8: (1,32,128,128)}  # chunks 1-7 are (4,32,128,128)
per_chunk_conv = []
per_chunk_rec  = []
for ci in range(NUM_CHUNKS):
    cs = KNOWN_CONV.get(ci, (4, 1024, 32))
    rs = KNOWN_REC.get(ci, (4, 32, 128, 128))
    try:
        for inp in ffns[ci].get_spec().description.input:
            shp = tuple(inp.type.multiArrayType.shape)
            if inp.name == 'linear_conv_state':    cs = shp
            elif inp.name == 'linear_recurrent_state': rs = shp
    except (AttributeError, Exception):
        pass  # CompiledMLModel has no get_spec — use known defaults
    per_chunk_conv.append(cs)
    per_chunk_rec.append(rs)

# Discover KV state names
KV_NAMES = ['k_cache', 'v_cache']  # Default for Qwen3.5
try:
    for s in ffns[0].get_spec().description.state:
        KV_NAMES.append(s.name)
    KV_NAMES = KV_NAMES[2:]  # Remove defaults if spec worked
except (AttributeError, Exception):
    pass  # Use defaults
print(f"KV state names: {KV_NAMES}")

# Quick probe
s0 = ffns[0].make_state()
kv_sample = s0.read_state(name=KV_NAMES[0])
SEQ_AXIS = kv_sample.ndim - 2
print(f"KV shape per chunk: {kv_sample.shape}, seq_axis={SEQ_AXIS}")

# ---------------------------------------------------------------------------
# Inference primitives
# ---------------------------------------------------------------------------
_tb = np.zeros((1, 1), dtype=np.int32)
_mb = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
_pb = np.zeros(1, dtype=np.int32)
_rb = np.zeros(1, dtype=np.int32)


def fresh():
    """Fresh zero states."""
    sts = [m.make_state() for m in ffns]
    lc  = [np.zeros(per_chunk_conv[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
    lr  = [np.zeros(per_chunk_rec[ci],  dtype=np.float16) for ci in range(NUM_CHUNKS)]
    return sts, lc, lr


def _run_chunks(tok_id, pos, rope_pos, sts, lc, lr, *, with_lmhead=False):
    """Run one token through all FFN chunks. Returns next_id if with_lmhead."""
    _tb[0, 0] = tok_id
    hid = list(embed.predict({"input_ids": _tb}).values())[0]
    _mb[:] = -65504.0
    _mb[:, :, :, :pos+1] = 0
    _pb[0] = pos
    _rb[0] = rope_pos
    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states":         hid.astype(np.float16),
            "position_ids":          _rb,
            "causal_mask":           _mb,
            "current_pos":           _pb,
            "linear_conv_state":     lc[ci],
            "linear_recurrent_state": lr[ci],
        }
        out = ffns[ci].predict(inp, state=sts[ci])
        hid = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lc[ci] = out['linear_conv_state_out']
            lr[ci] = out['linear_recurrent_state_out']
    if with_lmhead:
        lo = lmhead.predict({"hidden_states": hid.astype(np.float16)})
        return int(np.argmax(lo["logits"].flatten().astype(np.float32)))
    return None


def prefill_and_gen(token_ids, n_gen, sts, lc, lr, start_pos=0, rope_off=0):
    """Sequential prefill + greedy decode. Returns (gen_ids, final_pos)."""
    pos = start_pos
    for i, tid in enumerate(token_ids):
        rp = pos + rope_off
        if i == len(token_ids) - 1:
            nxt = _run_chunks(tid, pos, rp, sts, lc, lr, with_lmhead=True)
        else:
            _run_chunks(tid, pos, rp, sts, lc, lr, with_lmhead=False)
        pos += 1
    gen = [nxt]
    for _ in range(n_gen - 1):
        if pos >= CTX:
            break
        rp = pos + rope_off
        nxt = _run_chunks(gen[-1], pos, rp, sts, lc, lr, with_lmhead=True)
        pos += 1
        gen.append(nxt)
    return gen, pos


def read_kv(sts):
    """Read all KV cache arrays from MLState into numpy dict."""
    saved = {}
    for ci in range(NUM_CHUNKS):
        saved[ci] = {}
        for sn in KV_NAMES:
            saved[ci][sn] = sts[ci].read_state(name=sn).copy()
    return saved


def write_kv(sts, saved):
    """Write saved KV arrays back into MLState."""
    for ci in range(NUM_CHUNKS):
        for sn in KV_NAMES:
            sts[ci].write_state(name=sn, value=saved[ci][sn])


def shift_kv(sts, discard, keep, old_phys):
    """Shift KV entries right→left. Returns saved shifted arrays."""
    saved = {}
    for ci in range(NUM_CHUNKS):
        saved[ci] = {}
        for sn in KV_NAMES:
            kv = sts[ci].read_state(name=sn)
            shifted = np.zeros_like(kv)
            src = [slice(None)] * kv.ndim
            dst = [slice(None)] * kv.ndim
            src[SEQ_AXIS] = slice(discard, old_phys)
            dst[SEQ_AXIS] = slice(0, keep)
            shifted[tuple(dst)] = kv[tuple(src)]
            sts[ci].write_state(name=sn, value=shifted)
            saved[ci][sn] = shifted
    return saved


def replay_linear_only(kept_tokens, sts, lc, lr, rope_off):
    """Replay kept tokens to rebuild linear states, then restore saved KV.

    1. Save shifted KV
    2. Reset linear states to zero
    3. Replay tokens (updates both KV and linear)
    4. Restore saved KV (overwriting the replayed KV)
    Linear states are now consistent with the kept window.
    """
    # 1. Save the shifted KV
    saved_kv = read_kv(sts)

    # 2. Zero-out linear states
    for ci in range(NUM_CHUNKS):
        lc[ci] = np.zeros(per_chunk_conv[ci], dtype=np.float16)
        lr[ci] = np.zeros(per_chunk_rec[ci],  dtype=np.float16)

    # 3. Replay kept tokens (rebuilds KV + linear)
    keep = len(kept_tokens)
    # We also need a fresh KV to replay into, so we reset the MLState KV
    for ci in range(NUM_CHUNKS):
        blank = ffns[ci].make_state()
        for sn in KV_NAMES:
            sts[ci].write_state(name=sn,
                                value=blank.read_state(name=sn))
    for i, tid in enumerate(kept_tokens):
        rp = i + rope_off
        _run_chunks(tid, i, rp, sts, lc, lr, with_lmhead=False)

    # 4. Restore shifted KV
    write_kv(sts, saved_kv)


def compare(name_a, ids_a, name_b, ids_b):
    """Compare two token lists, return (match_frac, first_diff_pos)."""
    n = min(len(ids_a), len(ids_b))
    if n == 0:
        return 0.0, -1
    m = sum(1 for a, b in zip(ids_a[:n], ids_b[:n]) if a == b)
    frac = m / n
    fd = -1
    for i in range(n):
        if ids_a[i] != ids_b[i]:
            fd = i
            break
    tag = "✓" if frac >= 0.95 else ("~" if frac >= 0.7 else "✗")
    print(f"  {tag} {name_a} vs {name_b}: {m}/{n} = {frac*100:.1f}%"
          f"  (first_diff@{fd})" if fd >= 0 else
          f"  {tag} {name_a} vs {name_b}: {m}/{n} = {frac*100:.1f}%  (perfect)")
    return frac, fd


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
prompt = ("The quick brown fox jumps over the lazy dog. "
          "Once upon a time in a land far away,")
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
print(f"\nPrompt: {len(prompt_ids)} tokens")
all_before_tokens = list(prompt_ids)  # will extend with GEN_BEFORE tokens

# ---------------------------------------------------------------------------
# BASELINE: no compaction
# ---------------------------------------------------------------------------
banner("BASELINE (no compaction)")
s_b, lc_b, lr_b = fresh()
base_ids, base_pos = prefill_and_gen(prompt_ids, GEN_BEFORE + GEN_AFTER,
                                      s_b, lc_b, lr_b)
print(f"Generated {len(base_ids)} tok, pos={base_pos}")
print(f"  Text: {repr(tokenizer.decode(base_ids, skip_special_tokens=True)[:200])}")
base_before = base_ids[:GEN_BEFORE]
base_after  = base_ids[GEN_BEFORE:GEN_BEFORE + GEN_AFTER]

# ---------------------------------------------------------------------------
# PRE-SHIFT: generate GEN_BEFORE identically for all 3 strategies
# ---------------------------------------------------------------------------
banner("PRE-SHIFT: generate first batch")
s_pre, lc_pre, lr_pre = fresh()
pre_ids, pre_pos = prefill_and_gen(prompt_ids, GEN_BEFORE, s_pre, lc_pre, lr_pre)
assert pre_ids == base_before, "Pre-shift must match baseline prefix!"
print(f"Pre-shift: {len(pre_ids)} tok, pos={pre_pos} — matches baseline ✓")
all_before_tokens.extend(pre_ids)

# Compute shift parameters
total_phys  = pre_pos
keep_count  = total_phys // 2
discard     = total_phys - keep_count
rope_off    = total_phys - keep_count   # old_logical(=total_phys) - keep_count
kept_tokens = all_before_tokens[-keep_count:]
print(f"Compaction: keep={keep_count}, discard={discard}, rope_off={rope_off}")
print(f"  kept_tokens[-5:] = {kept_tokens[-5:]}")

# ---------------------------------------------------------------------------
# Strategy A: KV shift + linear keep-as-is
# ---------------------------------------------------------------------------
banner("STRATEGY A: KV shift + linear keep-as-is")

# Clone state from pre-shift
s_a, lc_a, lr_a = fresh()
# Re-run to get identical state (can't copy MLState directly)
prefill_and_gen(prompt_ids, GEN_BEFORE, s_a, lc_a, lr_a)

t0 = time.time()
shift_kv(s_a, discard, keep_count, total_phys)
t_a_shift = time.time() - t0
# linear states: kept as-is (full prior history)

pos_a = keep_count
ids_a = []
nxt = pre_ids[-1]  # last generated token before compaction
for _ in range(GEN_AFTER):
    if pos_a >= CTX: break
    nxt = _run_chunks(nxt, pos_a, pos_a + rope_off, s_a, lc_a, lr_a, with_lmhead=True)
    pos_a += 1
    ids_a.append(nxt)
t_a = time.time() - t0

text_a = tokenizer.decode(ids_a, skip_special_tokens=True)
print(f"Generated {len(ids_a)} tok in {t_a:.2f}s (shift: {t_a_shift*1000:.0f}ms)")
print(f"  Text: {repr(text_a[:200])}")

# ---------------------------------------------------------------------------
# Strategy B: KV shift + linear rebuild from kept window
# ---------------------------------------------------------------------------
banner("STRATEGY B: KV shift + linear rebuild")

s_b2, lc_b2, lr_b2 = fresh()
prefill_and_gen(prompt_ids, GEN_BEFORE, s_b2, lc_b2, lr_b2)

t0 = time.time()
saved_kv = shift_kv(s_b2, discard, keep_count, total_phys)
t_b_shift = time.time() - t0

# Rebuild linear states from kept tokens
t0_lin = time.time()
replay_linear_only(kept_tokens, s_b2, lc_b2, lr_b2, rope_off)
t_b_linear = time.time() - t0_lin

pos_b = keep_count
ids_b2 = []
nxt = pre_ids[-1]
t0_gen = time.time()
for _ in range(GEN_AFTER):
    if pos_b >= CTX: break
    nxt = _run_chunks(nxt, pos_b, pos_b + rope_off, s_b2, lc_b2, lr_b2, with_lmhead=True)
    pos_b += 1
    ids_b2.append(nxt)
t_b_gen = time.time() - t0_gen

text_b = tokenizer.decode(ids_b2, skip_special_tokens=True)
print(f"Generated {len(ids_b2)} tok "
      f"(shift: {t_b_shift*1000:.0f}ms, linear rebuild: {t_b_linear:.2f}s, gen: {t_b_gen:.2f}s)")
print(f"  Text: {repr(text_b[:200])}")

# ---------------------------------------------------------------------------
# Strategy C: Full reset + replay
# ---------------------------------------------------------------------------
banner("STRATEGY C: Full reset + replay")

s_c, lc_c, lr_c = fresh()
t0 = time.time()
for i, tid in enumerate(kept_tokens):
    rp = i + rope_off
    if i == len(kept_tokens) - 1:
        nxt_c = _run_chunks(tid, i, rp, s_c, lc_c, lr_c, with_lmhead=True)
    else:
        _run_chunks(tid, i, rp, s_c, lc_c, lr_c, with_lmhead=False)
t_c_replay = time.time() - t0

pos_c = keep_count
ids_c = []
nxt = nxt_c
for _ in range(GEN_AFTER):
    if pos_c >= CTX: break
    nxt = _run_chunks(nxt, pos_c, pos_c + rope_off, s_c, lc_c, lr_c, with_lmhead=True)
    pos_c += 1
    ids_c.append(nxt)
t_c = time.time() - t0

text_c = tokenizer.decode(ids_c, skip_special_tokens=True)
print(f"Generated {len(ids_c)} tok in {t_c:.2f}s (replay: {t_c_replay:.2f}s)")
print(f"  Text: {repr(text_c[:200])}")

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
banner("COMPARISON vs BASELINE")

frac_a, _ = compare("A(shift+keep)",    ids_a,  "BASELINE", base_after)
frac_b, _ = compare("B(shift+rebuild)", ids_b2, "BASELINE", base_after)
frac_c, _ = compare("C(reset+replay)",  ids_c,  "BASELINE", base_after)

print()
compare("A(shift+keep)",    ids_a,  "B(shift+rebuild)", ids_b2)
compare("A(shift+keep)",    ids_a,  "C(reset+replay)",  ids_c)
compare("B(shift+rebuild)", ids_b2, "C(reset+replay)",  ids_c)

# KV cache comparison: B shifted vs C replayed
banner("KV CACHE: B(shifted) vs C(replayed)")
for ci in range(NUM_CHUNKS):
    for sn in KV_NAMES:
        kv_b = s_b2[ci].read_state(name=sn)
        kv_c = s_c[ci].read_state(name=sn)
        sl = [slice(None)] * kv_b.ndim
        sl[SEQ_AXIS] = slice(0, pos_c)
        region_b = kv_b[tuple(sl)].astype(np.float32)
        region_c = kv_c[tuple(sl)].astype(np.float32)
        mx = np.abs(region_b - region_c).max()
        mn = np.abs(region_b - region_c).mean()
        cs = (np.sum(region_b * region_c) /
              (np.linalg.norm(region_b) * np.linalg.norm(region_c) + 1e-10))
        if ci < 3 or ci == NUM_CHUNKS - 1:
            print(f"  chunk {ci} {sn}: max={mx:.4f} mean={mn:.6f} cos={cs:.6f}")

# Linear state comparison: B rebuilt vs C replayed
banner("LINEAR STATE: B(rebuilt) vs C(replayed from scratch)")
for ci in range(NUM_CHUNKS):
    cd = np.abs(lc_b2[ci].astype(np.float32) - lc_c[ci].astype(np.float32))
    rd = np.abs(lr_b2[ci].astype(np.float32) - lr_c[ci].astype(np.float32))
    if ci < 3 or ci == NUM_CHUNKS - 1:
        print(f"  chunk {ci}: conv max={cd.max():.6f} mean={cd.mean():.6f}"
              f"  rec max={rd.max():.6f} mean={rd.mean():.6f}")

# ---------------------------------------------------------------------------
# RoPE / position sanity checks
# ---------------------------------------------------------------------------
banner("POSITION SANITY CHECKS")

# Verify: fresh run with same rope_offset produces same first token
s_fresh, lc_fresh, lr_fresh = fresh()
fresh_ids, _ = prefill_and_gen(kept_tokens, 5, s_fresh, lc_fresh, lr_fresh,
                                start_pos=0, rope_off=rope_off)
# And from pos=0 with no offset (different RoPE = SHOULD differ)
s_fresh2, lc_fresh2, lr_fresh2 = fresh()
fresh_no_off, _ = prefill_and_gen(kept_tokens, 5, s_fresh2, lc_fresh2, lr_fresh2,
                                   start_pos=0, rope_off=0)
print(f"Fresh (rope_off={rope_off}): {fresh_ids[:5]}")
print(f"Fresh (rope_off=0):          {fresh_no_off[:5]}")
print(f"C (reset+replay):            {ids_c[:5]}")
print(f"B (shift+rebuild):           {ids_b2[:5]}")
fresh_match_c = (fresh_ids[:5] == ids_c[:5])
print(f"Fresh(offset) == C(replay): {fresh_match_c}  "
      f"{'✓ RoPE correct' if fresh_match_c else '✗ RoPE mismatch!'}")

# ---------------------------------------------------------------------------
# Extended generation stability test
# ---------------------------------------------------------------------------
banner("EXTENDED STABILITY: generate 60 more tokens after compaction")

# Strategy B extended
pos_bx = pos_b
ids_bx = list(ids_b2)
nxt = ids_b2[-1] if ids_b2 else pre_ids[-1]
for _ in range(60):
    if pos_bx >= CTX: break
    nxt = _run_chunks(nxt, pos_bx, pos_bx + rope_off, s_b2, lc_b2, lr_b2, with_lmhead=True)
    pos_bx += 1
    ids_bx.append(nxt)

# Baseline extended (already have GEN_BEFORE+GEN_AFTER, generate 60 more)
s_be, lc_be, lr_be = fresh()
base_ext, _ = prefill_and_gen(prompt_ids, GEN_BEFORE + GEN_AFTER + 60,
                               s_be, lc_be, lr_be)
base_ext_after = base_ext[GEN_BEFORE:GEN_BEFORE + GEN_AFTER + 60]

bx_ext = ids_bx[:len(base_ext_after)]
frac_ext, _ = compare("B_extended", bx_ext, "BASELINE_ext", base_ext_after)

text_bx = tokenizer.decode(ids_bx, skip_special_tokens=True)
text_base_ext = tokenizer.decode(base_ext_after, skip_special_tokens=True)
print(f"\nB extended ({len(ids_bx)} tok): {repr(text_bx[:300])}")
print(f"Baseline   ({len(base_ext_after)} tok): {repr(text_base_ext[:300])}")

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
banner("FINAL VERDICT")

best = max([("A(shift+keep)", frac_a),
            ("B(shift+rebuild)", frac_b),
            ("C(reset+replay)", frac_c)], key=lambda x: x[1])

print(f"  Strategy A (shift+keep):    {frac_a*100:.1f}% match to baseline")
print(f"  Strategy B (shift+rebuild): {frac_b*100:.1f}% match to baseline")
print(f"  Strategy C (reset+replay):  {frac_c*100:.1f}% match to baseline")
print(f"  Extended B stability:       {frac_ext*100:.1f}% match to baseline")
print(f"\n  BEST: {best[0]} at {best[1]*100:.1f}%")

if frac_b >= frac_a and frac_b >= frac_c:
    print("\n  => Strategy B (KV shift + linear rebuild) is the winner.")
    print("     This confirms that making both memory systems consistent")
    print("     with the kept window is the correct approach.")
elif frac_c > frac_b:
    print("\n  => Strategy C (full replay) is better — the original KV")
    print("     values don't help vs freshly-replayed ones.")
else:
    print(f"\n  => {best[0]} wins.")

print("\nDone.")
