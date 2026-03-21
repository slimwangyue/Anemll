#!/usr/bin/env python3
"""Debug parity for batch=256, ctx=1024 — isolated per-chunk + channel analysis.

Uses already-exported models from /tmp/qwen35_4chunk_parity/.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, time
import numpy as np
import coremltools as ct

OUT_DIR = "/tmp/qwen35_4chunk_parity"
BATCH = 256
CTX = 1024
NUM_CHUNKS = 4
CHUNKS = [(0, 8), (8, 16), (16, 24), (24, 32)]


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12)) if d > 1e-12 else 0.0


def analyze_errors(ref, cml, label):
    """Detailed error analysis."""
    ref_f = ref.astype(np.float32)
    cml_f = cml.astype(np.float32)
    diff = np.abs(ref_f - cml_f)
    cos = cosine(ref, cml)

    print(f"\n  [{label}]")
    print(f"  cosine={cos:.10f}  max_abs={diff.max():.6f}  mean_abs={diff.mean():.8f}")

    # Per-channel analysis: (1, seq, hidden) → channel = last dim
    # Mean abs error per channel
    ch_err = diff[0].mean(axis=0)  # (hidden,) — mean over seq positions
    worst_ch = np.argsort(ch_err)[-10:][::-1]
    print(f"  Top-10 worst channels (mean_abs):")
    for c in worst_ch:
        print(f"    ch={c:4d}  mean_abs={ch_err[c]:.6f}  "
              f"ref_mean={ref_f[0,:,c].mean():.4f}  cml_mean={cml_f[0,:,c].mean():.4f}  "
              f"ref_std={ref_f[0,:,c].std():.4f}  cml_std={cml_f[0,:,c].std():.4f}")

    # Per-position analysis: mean error per seq position
    pos_err = diff[0].mean(axis=1)  # (seq,)
    worst_pos = np.argsort(pos_err)[-5:][::-1]
    print(f"  Top-5 worst positions (mean_abs across channels):")
    for p in worst_pos:
        print(f"    pos={p:4d}  mean_abs={pos_err[p]:.6f}")

    # Check if error is systematic bias in specific channels
    bias = (cml_f - ref_f)[0].mean(axis=0)  # mean signed error per channel
    worst_bias = np.argsort(np.abs(bias))[-5:][::-1]
    print(f"  Top-5 channels with largest systematic bias:")
    for c in worst_bias:
        print(f"    ch={c:4d}  bias={bias[c]:+.6f}  (cml is {'higher' if bias[c] > 0 else 'lower'})")

    return cos, float(diff.max()), float(diff.mean())


# Load PyTorch references
print("Loading PyTorch references...")
embed_np = np.load(os.path.join(OUT_DIR, "embed.npy"))
torch_chunks = {i+1: np.load(os.path.join(OUT_DIR, f"torch_chunk{i+1}.npy")) for i in range(NUM_CHUNKS)}


# Build causal mask and position_ids
position_ids = np.arange(BATCH, dtype=np.int32)
causal_mask = np.full((1, 1, BATCH, CTX), -65504.0, dtype=np.float16)
for r in range(BATCH):
    causal_mask[0, 0, r, :r + 1] = 0


# ── ISOLATED PARITY: each chunk gets PyTorch reference input ──
print("\n" + "=" * 64)
print("  ISOLATED PARITY (each chunk gets PyTorch reference input)")
print("=" * 64)

isolated_results = []
for ci in range(NUM_CHUNKS):
    s, e = CHUNKS[ci]
    pkg = os.path.join(OUT_DIR, f"chunk{ci+1}.mlpackage")
    torch_ref = torch_chunks[ci + 1]

    # Input: embed for chunk 1, PyTorch output of previous chunk for others
    if ci == 0:
        hidden_input = embed_np.copy()
        src = "embed (perfect parity)"
    else:
        hidden_input = torch_chunks[ci].copy()  # PyTorch output of chunk ci
        src = f"PyTorch chunk {ci} output"

    print(f"\n{'─'*60}")
    print(f"Chunk {ci+1} (layers {s}-{e-1}) — input: {src}")

    t0 = time.time()
    cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    state = cml.make_state()

    inp = {
        "hidden_states": hidden_input.astype(np.float16),
        "position_ids": position_ids,
        "causal_mask": causal_mask,
        "current_pos": np.zeros((1,), dtype=np.int32),
    }

    try:
        out = cml.predict(inp, state=state)
        cml_out = list(out.values())[0]
        print(f"  Output shape: {cml_out.shape}")
    except Exception as exc:
        print(f"  ❌ PREDICT FAILED: {exc}")
        isolated_results.append((ci + 1, "FAIL", None, None, None))
        del cml; gc.collect()
        continue

    # Handle shape mismatch for last chunk
    ref = torch_ref
    cmp = cml_out
    if cml_out.shape != ref.shape:
        min_seq = min(cml_out.shape[1], ref.shape[1])
        ref = ref[:, :min_seq, :]
        cmp = cml_out[:, :min_seq, :]
        print(f"  Shape mismatch — comparing first {min_seq} token(s)")

    cos, mx, mn = analyze_errors(ref, cmp, f"Chunk {ci+1} ISOLATED")
    isolated_results.append((ci + 1, "OK", mx, mn, cos))

    # Save for comparison
    np.save(os.path.join(OUT_DIR, f"cml_iso_chunk{ci+1}.npy"), cml_out)
    del cml, state; gc.collect()


# ── Summary ──
print(f"\n{'='*70}")
print(f"  ISOLATED PARITY SUMMARY  (batch={BATCH}, ctx={CTX})")
print(f"{'='*70}")
print(f"  {'Chunk':<8} {'Grade':<14} {'max_abs':<12} {'mean_abs':<14} {'cosine':<15}")
print(f"  {'-'*64}")
for ci, status, mx, mn, cos in isolated_results:
    if status == "OK":
        if cos > 0.999 and mx < 0.5:
            grade = "GOOD"
        elif cos > 0.99 and mx < 2.0:
            grade = "ACCEPT"
        else:
            grade = "BAD"
        print(f"  {ci:<8} {grade:<14} {mx:<12.6f} {mn:<14.8f} {cos:<15.10f}")
    else:
        print(f"  {ci:<8} {status}")
print(f"{'='*70}")
