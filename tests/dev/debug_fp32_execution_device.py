#!/usr/bin/env python3
"""Diagnostic: Verify WHERE FP32 models actually execute (ANE vs CPU vs GPU).

Tests:
1. Load FP32 chunk with CPU_AND_NE, CPU_AND_GPU, CPU_ONLY — compare latency
2. Inspect MIL op dtypes in FP16 vs FP32 models
3. Check if CoreML silently falls back off ANE when ops are FP32
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

FP16_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"
FP32_DIR = "/tmp/qwen35_fp32_chunks"
TOKENIZER = "/Users/yw68/Anemll/qwen3_5_stable_models"

NEG_INF = np.float16(-65504.0)


def find_model(base, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = Path(base) / f"{name}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Missing {name} in {base}")


def inspect_spec_dtypes(path, label):
    """Load spec and show input/output dtypes."""
    print(f"\n  [{label}] Spec: {Path(path).name}")
    spec = ct.utils.load_spec(path)
    desc = spec.description

    for inp in desc.input:
        try:
            dt = inp.type.multiArrayType.dataType
            shape = list(inp.type.multiArrayType.shape)
            print(f"    Input  {inp.name:<30} dtype={dt} shape={shape}")
        except Exception:
            print(f"    Input  {inp.name:<30} (state or other)")

    for out in desc.output:
        try:
            dt = out.type.multiArrayType.dataType
            shape = list(out.type.multiArrayType.shape)
            print(f"    Output {out.name:<30} dtype={dt} shape={shape}")
        except Exception:
            print(f"    Output {out.name:<30} (state or other)")


def benchmark_compute_unit(path, label, cu_name, cu, n_steps=10):
    """Load model, run n_steps, report latency."""
    print(f"\n  [{label}] Loading with {cu_name}...", end="", flush=True)
    t0 = time.time()
    try:
        model = ct.models.MLModel(path, compute_units=cu)
    except Exception as e:
        print(f" FAILED: {e}")
        return None
    t_load = time.time() - t0
    print(f" loaded in {t_load:.1f}s")

    # Get input shapes
    spec = model.get_spec()
    inputs = {}
    for inp in spec.description.input:
        try:
            shape = tuple(int(x) for x in inp.type.multiArrayType.shape)
            dt = inp.type.multiArrayType.dataType
            # Map protobuf dtype to numpy
            if dt == 65552:  # FLOAT16
                inputs[inp.name] = np.zeros(shape, dtype=np.float16)
            elif dt == 65568:  # FLOAT32
                inputs[inp.name] = np.zeros(shape, dtype=np.float32)
            elif dt == 131104:  # INT32
                inputs[inp.name] = np.zeros(shape, dtype=np.int32)
            else:
                inputs[inp.name] = np.zeros(shape, dtype=np.float16)
        except Exception:
            pass  # state input

    # Always try to create state (FFN chunks have state)
    try:
        state = model.make_state()
    except Exception:
        state = None

    # Warmup
    print(f"    Warmup...", end="", flush=True)
    for _ in range(2):
        if state is not None:
            model.predict(inputs, state=state)
        else:
            model.predict(inputs)
    print(" done")

    # Benchmark
    times = []
    for i in range(n_steps):
        t0 = time.time()
        if state is not None:
            model.predict(inputs, state=state)
        else:
            model.predict(inputs)
        times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"    {cu_name}: avg={avg_ms:.1f}ms ± {std_ms:.1f}ms over {n_steps} steps")

    del model
    gc.collect()
    return avg_ms


def main():
    print("=" * 70)
    print("  FP32 vs FP16 Execution Device Diagnostic")
    print("=" * 70)

    # ── Test 1: Inspect model spec dtypes ──
    print("\n── Test 1: Inspect Model Spec Dtypes ──")
    fp16_path = find_model(FP16_DIR, "ffn_LUT4_chunk0")
    fp32_path = find_model(FP32_DIR, "ffn_LUT4_chunk0")
    inspect_spec_dtypes(fp16_path, "FP16-chunk0")
    inspect_spec_dtypes(fp32_path, "FP32-chunk0")

    # ── Test 2: Check MIL program op types ──
    print("\n── Test 2: MIL Op Dtype Analysis ──")
    for path, label in [(fp16_path, "FP16"), (fp32_path, "FP32")]:
        spec = ct.utils.load_spec(path)
        prog = spec.mlProgram
        dtype_counts = {}
        op_count = 0
        for func in prog.functions.values():
            for blk in func.block_specializations.values():
                for op in blk.operations:
                    op_count += 1
                    for attr_name, attr_val in op.attributes.items():
                        if 'dtype' in attr_name.lower() or attr_name == 'dtype':
                            val = attr_val.immediate.s if attr_val.HasField('immediate') else '?'
                            dtype_counts[val] = dtype_counts.get(val, 0) + 1
        print(f"  [{label}] {op_count} ops, dtype distribution: {dtype_counts}")

    # ── Test 3: Benchmark across compute units ──
    print("\n── Test 3: Latency Benchmark (chunk0) ──")
    print("  Testing FP16 model across compute units:")
    units = [
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
        ("ALL", ct.ComputeUnit.ALL),
    ]

    fp16_times = {}
    for cu_name, cu in units:
        t = benchmark_compute_unit(fp16_path, "FP16", cu_name, cu, n_steps=8)
        fp16_times[cu_name] = t

    print("\n  Testing FP32 model across compute units:")
    fp32_times = {}
    for cu_name, cu in units:
        t = benchmark_compute_unit(fp32_path, "FP32", cu_name, cu, n_steps=8)
        fp32_times[cu_name] = t

    # ── Test 4: Also test embeddings (simpler model) ──
    print("\n── Test 4: Embeddings Benchmark ──")
    embed_path = find_model(FP16_DIR, "embeddings")
    for cu_name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
                         ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)]:
        benchmark_compute_unit(embed_path, "embed", cu_name, cu, n_steps=20)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY: Where does FP32 actually run?")
    print("=" * 70)
    print(f"\n  {'Unit':<16} {'FP16 (ms)':>12} {'FP32 (ms)':>12} {'Ratio':>8}")
    print(f"  {'-'*50}")
    for cu_name, _ in units:
        t16 = fp16_times.get(cu_name)
        t32 = fp32_times.get(cu_name)
        if t16 and t32:
            ratio = t32 / t16
            print(f"  {cu_name:<16} {t16:>10.1f}ms {t32:>10.1f}ms {ratio:>7.2f}x")
    print()
    print("  If FP32(CPU_AND_NE) ≈ FP32(CPU_AND_GPU) >> FP32(CPU_ONLY):")
    print("    → FP32 runs on GPU, not ANE (ANE doesn't support FP32)")
    print("  If FP32(CPU_AND_NE) ≈ FP16(CPU_AND_NE):")
    print("    → ANE is quantizing FP32→FP16 internally")
    print("  If FP32(CPU_AND_NE) ≈ FP32(CPU_ONLY):")
    print("    → Falls back to CPU entirely")


if __name__ == "__main__":
    main()
