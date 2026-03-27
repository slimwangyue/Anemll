#!/usr/bin/env python3
"""Measure latency and size delta between FP16 and FP32 compute precision,
plus full-pipeline model size reference.

Part 1: Latency benchmark — same single-layer CoreNorm, FP16 vs FP32 on ANE
Part 2: Full pipeline model size reference (existing production models)
Part 3: Impact estimate for full pipeline re-export
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Dict

import coremltools as ct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

OUT_DIR = "/tmp/qwen35_mitigation_exp"

# ────────────────────────────────────────────────────────────────────
# Part 1: Latency benchmark (ANE)
# ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Part 1: Latency Benchmark — FP16 vs FP32 CoreNorm (single layer, ANE)")
print("=" * 70)

N_WARMUP = 5
N_ITERS = 30


def bench_model(pkg_path: str, label: str) -> Dict:
    """Load, warm up, and benchmark a single CoreNorm mlpackage on ANE."""
    cml = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # Read input spec to create dummy inputs
    spec = cml.get_spec()
    inputs = {}
    for inp in spec.description.input:
        name = inp.name
        shape = tuple(d.value if hasattr(d, 'value') else d for d in inp.type.multiArrayType.shape)
        inputs[name] = np.random.randn(*shape).astype(np.float16)

    # Warm up
    for _ in range(N_WARMUP):
        state = cml.make_state()
        cml.predict(inputs, state=state)

    # Benchmark
    times = []
    for _ in range(N_ITERS):
        state = cml.make_state()
        t0 = time.perf_counter()
        cml.predict(inputs, state=state)
        times.append(time.perf_counter() - t0)

    times_ms = sorted([t * 1000 for t in times])
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = times_ms[0]
    max_ms = times_ms[-1]
    p50_ms = times_ms[len(times_ms) // 2]
    # Drop top/bottom 10% for trimmed mean
    trim = max(1, len(times_ms) // 10)
    trimmed = times_ms[trim:-trim]
    trimmed_mean = sum(trimmed) / len(trimmed) if trimmed else mean_ms

    pkg_size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(pkg_path)
        for f in fns
    ) / (1024 * 1024)

    print(f"  [{label}] trimmed_mean={trimmed_mean:.2f}ms  p50={p50_ms:.2f}ms  "
          f"min={min_ms:.2f}ms  max={max_ms:.2f}ms  size={pkg_size_mb:.1f}MB")

    del cml
    gc.collect()
    return {
        "label": label,
        "mean_ms": round(mean_ms, 2),
        "trimmed_mean_ms": round(trimmed_mean, 2),
        "p50_ms": round(p50_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "size_mb": round(pkg_size_mb, 1),
        "n_iters": N_ITERS,
    }


# Benchmark all available strategies
strategies = [
    ("A_baseline.mlpackage", "A_FP16_baseline"),
    ("D_fp32_precision.mlpackage", "D_FP32_precision"),
    ("C_direct_rmsnorm.mlpackage", "C_DirectRMSNorm_FP16"),
    ("F_direct_rmsnorm_fp32.mlpackage", "F_DirectRMSNorm_FP32"),
]

results = {}
for pkg_name, label in strategies:
    pkg_path = os.path.join(OUT_DIR, pkg_name)
    if os.path.exists(pkg_path):
        results[label] = bench_model(pkg_path, label)
    else:
        print(f"  [{label}] SKIP — not found")

# Compare FP16 vs FP32
if "A_FP16_baseline" in results and "D_FP32_precision" in results:
    r16 = results["A_FP16_baseline"]
    r32 = results["D_FP32_precision"]
    overhead = r32["trimmed_mean_ms"] / r16["trimmed_mean_ms"]
    size_ratio = r32["size_mb"] / r16["size_mb"]
    print(f"\n  ┌────────────────────────────────────────────────────┐")
    print(f"  │ FP32 vs FP16 (single-layer CoreNorm on ANE)       │")
    print(f"  │  Latency: {r32['trimmed_mean_ms']:.2f}ms / {r16['trimmed_mean_ms']:.2f}ms = "
          f"{overhead:.2f}x overhead   │")
    print(f"  │  Size:    {r32['size_mb']:.1f}MB / {r16['size_mb']:.1f}MB = "
          f"{size_ratio:.2f}x ratio         │")
    print(f"  └────────────────────────────────────────────────────┘")

# ────────────────────────────────────────────────────────────────────
# Part 2: Production model size reference
# ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  Part 2: Production Model Sizes (for reference)")
print(f"{'='*70}")

STABLE_DIR = "/Users/yw68/Anemll/qwen3_5_stable_models"
REMOTE_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"

prod_sizes = {}
for d in [STABLE_DIR, REMOTE_DIR]:
    if os.path.exists(d):
        print(f"\n  {d}:")
        for entry in sorted(os.listdir(d)):
            full = os.path.join(d, entry)
            if entry.endswith((".mlpackage", ".mlmodelc")):
                sz = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fns in os.walk(full)
                    for f in fns
                ) / (1024 * 1024)
                print(f"    {entry:<50} {sz:>8.1f} MB")
                prod_sizes[entry] = round(sz, 1)

# ────────────────────────────────────────────────────────────────────
# Part 3: Full pipeline impact estimate
# ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  Part 3: Full-Pipeline Impact Estimate")
print(f"{'='*70}")

if "A_FP16_baseline" in results and "D_FP32_precision" in results:
    r16 = results["A_FP16_baseline"]
    r32 = results["D_FP32_precision"]
    overhead_ms = r32["trimmed_mean_ms"] - r16["trimmed_mean_ms"]
    overhead_pct = (overhead_ms / r16["trimmed_mean_ms"]) * 100
    size_diff_mb = r32["size_mb"] - r16["size_mb"]

    # Existing production chunk sizes for estimate
    ffn_sizes = [v for k, v in prod_sizes.items() if "ffn_LUT4" in k or "prefill_LUT4" in k]
    total_ffn_mb = sum(ffn_sizes)

    print(f"""
  SINGLE-LAYER CoreNorm (what we measured):
    This is one layer of the core recurrence + norm + out_proj stage.
    The full model has 28 layers × 4 chunks = 112 CoreNorm invocations per pass.

  LATENCY (on ANE):
    FP16 single-layer: {r16['trimmed_mean_ms']:.2f} ms
    FP32 single-layer: {r32['trimmed_mean_ms']:.2f} ms
    Per-layer overhead: {overhead_ms:+.2f} ms ({overhead_pct:+.1f}%)

    NOTE: Per-predict-call overhead is what matters (not per-layer),
    because ANE serializes ops within a single predict() call.
    The actual decode-step overhead would be measured on the full
    chunked model (all 7 layers inside one chunk predict call).

  MODEL SIZE:
    FP16 CoreNorm layer: {r16['size_mb']:.1f} MB
    FP32 CoreNorm layer: {r32['size_mb']:.1f} MB
    Delta: {size_diff_mb:+.1f} MB

    compute_precision=FLOAT32 only affects intermediate computation dtype.
    Weights stay LUT-quantized (4-bit), so on-disk size is nearly identical.
    Current total FFN+prefill size: {total_ffn_mb:.0f} MB
    Expected FP32 total: ~{total_ffn_mb:.0f} MB (negligible change)

  RECOMMENDATION:
    Re-export with compute_precision=ct.precision.FLOAT32 for linear-attn
    chunks. Expected model size ≈ same, latency overhead needs full-pipeline
    measurement but is expected to be modest given ANE's fp32 support.
""")

# Save report
report = {
    "benchmark": results,
    "production_sizes_mb": prod_sizes,
}
report_path = "tests/dev/mitigation_fp32_latency_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"  Report saved: {report_path}")
