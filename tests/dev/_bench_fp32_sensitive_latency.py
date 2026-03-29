#!/usr/bin/env python3
"""Latency benchmark: measure added cost of FP32-sensitive ops and CPU+GPU mode.

Compares decode latency for one FFN chunk across configurations:
  A) LUT4 baseline (all FP16 on ANE)
  B) FP16 weights (no quant, FP16 on ANE)
  C) FP32-sensitive ops (sensitive ops in FP32, rest FP16 on ANE)
  D) Selective LUT4 (linear-attn weights FP16, MLP LUT4, FP32 sensitive ops)
  E) FP16 on CPU+GPU (no ANE)

Each config is timed over WARMUP + BENCHMARK iterations per-token decode.

Usage:
    cd /Users/yw68/Anemll
    python tests/dev/_bench_fp32_sensitive_latency.py
"""
import os, sys, time, gc
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts_qwen3_5"))

from config import BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, PER_CHANNEL, DEFAULT_HF_MODEL
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape

import coremltools as ct

CHUNK_IDX = 0
WARMUP = 10
ITERS = 50

OUT_BASE = "/tmp/qwen35_parity"

# Mapping label -> (subdir, lut_bits, fp32_sensitive, compute_units_for_predict)
CONFIGS = {
    "A: LUT4 baseline (ANE)":           (f"{OUT_BASE}_lutA",  "lut4", False, ct.ComputeUnit.CPU_AND_NE),
    "B: FP16 weights (ANE)":            (f"{OUT_BASE}_fp16B", "fp16", False, ct.ComputeUnit.CPU_AND_NE),
    "C: FP32-sensitive (ANE)":           (f"{OUT_BASE}_fp32C", "fp16", True,  ct.ComputeUnit.CPU_AND_NE),
    "D: Selective LUT4 (ANE)":           (f"{OUT_BASE}_selD",  "lut4", True,  ct.ComputeUnit.CPU_AND_NE),
    "E: FP16 weights (CPU+GPU)":         (f"{OUT_BASE}_fp16B", "fp16", False, ct.ComputeUnit.CPU_AND_GPU),
    "F: FP32-sensitive (CPU+GPU)":       (f"{OUT_BASE}_fp32C", "fp16", True,  ct.ComputeUnit.CPU_AND_GPU),
}


def get_model_cfg():
    cfg = Qwen35Config.from_json(os.path.join(DEFAULT_HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    return cfg


def find_mlpackage(subdir, tag):
    path = os.path.join(subdir, f"ffn_{tag}_chunk{CHUNK_IDX}.mlpackage")
    if os.path.exists(path):
        return path
    return None


def make_inputs(cfg):
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start = CHUNK_IDX * base + min(CHUNK_IDX, rem)
    end = start + base + (1 if CHUNK_IDX < rem else 0)
    local_num_layers = end - start

    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)

    return {
        "hidden_states": np.random.randn(1, 1, cfg.hidden_size).astype(np.float16) * 0.1,
        "position_ids": np.array([5], dtype=np.int32),
        "causal_mask": np.full((1, 1, 1, CTX), 0.0, dtype=np.float16),
        "current_pos": np.array([5], dtype=np.int32),
        "linear_conv_state": np.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=np.float16),
        "linear_recurrent_state": np.zeros(
            (local_num_layers, cfg.text_config.linear_num_value_heads,
             cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
            dtype=np.float16
        ),
    }


def bench_config(label, mlpackage_path, compute_units, inputs):
    print(f"\n  {label}")
    print(f"    Loading {os.path.basename(os.path.dirname(mlpackage_path))}/...mlpackage with {compute_units}")

    try:
        mlmodel = ct.models.MLModel(mlpackage_path, compute_units=compute_units)
    except Exception as e:
        print(f"    LOAD FAILED: {e}")
        return None

    try:
        state = mlmodel.make_state()
    except Exception as e:
        print(f"    make_state FAILED: {e}")
        return None

    # Warmup
    for _ in range(WARMUP):
        state = mlmodel.make_state()
        mlmodel.predict(inputs, state)

    # Benchmark
    times = []
    for _ in range(ITERS):
        state = mlmodel.make_state()
        t0 = time.perf_counter()
        mlmodel.predict(inputs, state)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    # Remove top/bottom 10% outliers
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    mean_ms = sum(trimmed) / len(trimmed)
    p50 = trimmed[len(trimmed) // 2]
    p95 = trimmed[int(len(trimmed) * 0.95)]
    min_ms = trimmed[0]

    print(f"    mean={mean_ms:.2f}ms  p50={p50:.2f}ms  p95={p95:.2f}ms  min={min_ms:.2f}ms")
    del mlmodel
    gc.collect()
    return {"mean": mean_ms, "p50": p50, "p95": p95, "min": min_ms}


def main():
    cfg = get_model_cfg()
    inputs = make_inputs(cfg)

    print("=" * 70)
    print(f"  LATENCY BENCHMARK — chunk {CHUNK_IDX} decode (single token)")
    print(f"  Warmup: {WARMUP}  Iterations: {ITERS}")
    print("=" * 70)

    results = {}
    for label, (subdir, tag, fp32s, cu) in CONFIGS.items():
        path = find_mlpackage(subdir, tag)
        if path is None:
            print(f"\n  {label}")
            print(f"    SKIPPED — model not found at {subdir}")
            continue
        r = bench_config(label, path, cu, inputs)
        if r:
            results[label] = r

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  LATENCY SUMMARY — chunk {CHUNK_IDX} decode")
    print(f"{'=' * 70}")
    print(f"  {'Config':<35s} {'Mean':>8s} {'P50':>8s} {'P95':>8s} {'Min':>8s}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    baseline_mean = None
    for label, r in results.items():
        if baseline_mean is None:
            baseline_mean = r["mean"]
        print(f"  {label:<35s} {r['mean']:>7.2f}ms {r['p50']:>7.2f}ms {r['p95']:>7.2f}ms {r['min']:>7.2f}ms")

    if baseline_mean and len(results) > 1:
        print(f"\n  {'Config':<35s} {'Overhead vs baseline':>20s}")
        print(f"  {'-' * 35} {'-' * 20}")
        for label, r in results.items():
            overhead = r["mean"] - baseline_mean
            pct = (overhead / baseline_mean * 100) if baseline_mean > 0 else 0
            print(f"  {label:<35s} {overhead:>+8.2f}ms ({pct:>+6.1f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
