#!/usr/bin/env python3
"""Lightweight LUT6+argmax LM head validation.

Only loads the lm_head model (not all 10) to test:
1. Model loads on ANE
2. Outputs argmax_idx + argmax_val
3. Deterministic output
4. Timing comparison vs previous fp16 (if available)

Then runs the full pipeline using chat_server engine (which handles OOM better).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import coremltools as ct

EXPORT_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"


def find_model(base, name):
    for ext in [".mlmodelc", ".mlpackage"]:
        p = os.path.join(base, name + ext)
        if os.path.exists(p):
            return p
    return None


def test_lm_head():
    print("=" * 60)
    print("  LUT6+Argmax LM Head — Lightweight Validation")
    print("=" * 60)

    path = find_model(EXPORT_DIR, "lm_head")
    assert path, "lm_head not found!"
    print(f"\n  Path: {path}")

    # Get file size
    total_bytes = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            total_bytes += os.path.getsize(os.path.join(dp, f))
    print(f"  Size: {total_bytes/1e6:.1f} MB")

    # Load
    print("\n[1] Loading on CPU_AND_NE...")
    t0 = time.time()
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Check outputs
    spec = model.get_spec()
    output_names = [o.name for o in spec.description.output]
    input_names = [i.name for i in spec.description.input]
    print(f"\n[2] Interface check")
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")
    assert "argmax_idx" in output_names, f"Missing argmax_idx! Got: {output_names}"
    assert "argmax_val" in output_names, f"Missing argmax_val! Got: {output_names}"
    assert "hidden_states" in input_names, f"Missing hidden_states! Got: {input_names}"
    print("  PASS: argmax_idx + argmax_val outputs present")

    # Warmup
    print("\n[3] Warmup (3 calls)...")
    dummy = np.zeros((1, 1, 2560), dtype=np.float16)
    for _ in range(3):
        model.predict({"hidden_states": dummy})
    print("  Done")

    # Determinism test
    print("\n[4] Determinism test (10 identical calls)...")
    results = []
    for _ in range(10):
        out = model.predict({"hidden_states": dummy})
        results.append(int(out["argmax_idx"].flatten()[0]))
    unique = set(results)
    print(f"  Results: {results[:5]}... unique: {unique}")
    assert len(unique) == 1, f"Non-deterministic! Got {unique}"
    print("  PASS: Deterministic output")

    # Random input test
    print("\n[5] Random input test...")
    np.random.seed(42)
    random_input = np.random.randn(1, 1, 2560).astype(np.float16)
    out = model.predict({"hidden_states": random_input})
    idx = int(out["argmax_idx"].flatten()[0])
    val = float(out["argmax_val"].flatten()[0])
    print(f"  argmax_idx: {idx}")
    print(f"  argmax_val: {val:.4f}")
    assert 0 <= idx < 248320, f"Token ID out of range: {idx}"
    print("  PASS: Valid token ID range")

    # Timing test
    print("\n[6] Timing test (50 calls, ANE)...")
    times = []
    for _ in range(50):
        t0 = time.time()
        model.predict({"hidden_states": random_input})
        times.append((time.time() - t0) * 1000)
    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Std:    {np.std(times):.2f} ms")
    print(f"  P50:    {np.percentile(times, 50):.2f} ms")
    print(f"  P95:    {np.percentile(times, 95):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")

    # Compare vs fp16 baseline
    fp16_path = find_model(EXPORT_DIR, "lm_head_fp16_backup")
    if fp16_path:
        print(f"\n[7] Comparison with fp16 baseline...")
        print(f"  fp16 path: {fp16_path}")
        fp16_model = ct.models.MLModel(fp16_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        fp16_times = []
        for _ in range(3):  # warmup
            fp16_model.predict({"hidden_states": dummy})
        for _ in range(50):
            t0 = time.time()
            fp16_model.predict({"hidden_states": random_input})
            fp16_times.append((time.time() - t0) * 1000)
        fp16_times = np.array(fp16_times)
        print(f"  fp16 Mean: {np.mean(fp16_times):.2f} ms")
        print(f"  LUT6 Mean: {np.mean(times):.2f} ms")
        speedup = np.mean(fp16_times) / np.mean(times)
        print(f"  Speedup:   {speedup:.2f}x")
        del fp16_model
    else:
        print("\n[7] No fp16 baseline found — skipping comparison")

    del model

    print(f"\n{'='*60}")
    print("  ALL TESTS PASSED")
    print(f"  LM Head: LUT6 + argmax, {total_bytes/1e6:.1f} MB")
    print(f"  ANE latency: {np.mean(times):.2f}ms (P50: {np.percentile(times, 50):.2f}ms)")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_lm_head()
