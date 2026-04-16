#!/usr/bin/env python3
"""Profile per-chunk ANE latency: LUT4 baseline vs FP16-attn-all (D2).

Builds two model variants for selected chunks, compiles to .mlmodelc,
loads on ANE (CPU_AND_NE), and measures repeated inference latency.

Usage:
  python scripts_qwen3_5/bench_fp16_attn_latency.py \
      --model models/Qwen__Qwen3.5-4B \
      --output /tmp/qwen35_latency \
      --chunks 1,4 \
      --warmup 10 --iters 50
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES,
    FFN_PER_CHANNEL,
)

# Reuse tensor family defs and helpers from ablation script
from fp16_ablation import (
    TENSOR_FAMILIES, EXPERIMENT_CONFIGS,
    _load_model, _convert_chunk_fp16,
    _build_selective_lut_config,
)

CONFIGS_TO_BENCH = {
    "A1_all_lut4": [],                                  # baseline: everything LUT4
    "D2_fp16_attn_all": ["attn_q", "attn_kv", "attn_o"],  # FP16 attention
}


def compile_mlpackage(mlpackage_path, output_dir):
    """Compile .mlpackage → .mlmodelc using coremlcompiler."""
    basename = os.path.splitext(os.path.basename(mlpackage_path))[0]
    cmd = [
        "xcrun", "coremlcompiler", "compile",
        mlpackage_path, output_dir,
        "--add-mlprogram-if-eligible", "force",
    ]
    print(f"    Compiling {basename}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"    COMPILE ERROR: {result.stderr[:500]}")
        raise RuntimeError(f"Compilation failed for {mlpackage_path}")
    # Find resulting .mlmodelc
    mlmodelc = os.path.join(output_dir, f"{basename}.mlmodelc")
    if not os.path.isdir(mlmodelc):
        # coremlcompiler sometimes names it differently
        for item in os.listdir(output_dir):
            if item.endswith(".mlmodelc") and basename in item:
                mlmodelc = os.path.join(output_dir, item)
                break
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(mlmodelc)
        for f in fns
    ) / 1e6 if os.path.isdir(mlmodelc) else 0
    print(f"    Compiled in {elapsed:.1f}s → {size_mb:.1f} MB")
    return mlmodelc


def build_random_inputs(mlmodel, sample_idx=0):
    """Build random inputs matching the model's input spec."""
    spec = mlmodel.get_spec()

    # Try multi-function first, then single-function
    fn_inputs = None
    if hasattr(spec.description, 'functions'):
        for fn in spec.description.functions:
            if fn.name == "infer":
                fn_inputs = fn.input
                break
    if fn_inputs is None:
        fn_inputs = spec.description.input

    inputs = {}
    for inp in fn_inputs:
        try:
            shape = tuple(inp.type.multiArrayType.shape)
        except Exception:
            continue
        name = inp.name
        if name in ("position_ids", "current_pos"):
            inputs[name] = np.array([sample_idx % CTX], dtype=np.int32)
        elif name == "causal_mask":
            mask = np.full(shape, -65504.0, dtype=np.float16)
            mask[:, :, :, :sample_idx + 1] = 0
            inputs[name] = mask
        else:
            inputs[name] = np.random.randn(*shape).astype(np.float16) * 0.1
    return inputs


def profile_model(mlmodel, n_warmup=10, n_iters=50, label=""):
    """Run repeated inferences and measure latency statistics.

    Returns dict with p50/p90/p99/mean/std in milliseconds.
    """
    import coremltools as ct

    # Build input once
    np.random.seed(42)
    inputs = build_random_inputs(mlmodel, sample_idx=5)

    # Create state (for KV cache)
    try:
        state = mlmodel.make_state()
    except Exception:
        state = None

    predict_kwargs = {"data": inputs}
    if state is not None:
        predict_kwargs["state"] = state

    # Warmup
    print(f"    [{label}] Warming up ({n_warmup} iters)...")
    for _ in range(n_warmup):
        _ = mlmodel.predict(**predict_kwargs)
        if state is not None:
            # Reset state each iteration to avoid accumulation effects
            state = mlmodel.make_state()
            predict_kwargs["state"] = state

    # Timed iterations
    print(f"    [{label}] Profiling ({n_iters} iters)...")
    latencies_ms = []
    for i in range(n_iters):
        if state is not None:
            state = mlmodel.make_state()
            predict_kwargs["state"] = state

        t0 = time.perf_counter()
        _ = mlmodel.predict(**predict_kwargs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    stats = {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n_iters": n_iters,
    }
    print(f"    [{label}] p50={stats['p50_ms']:.2f}ms  mean={stats['mean_ms']:.2f}ms  "
          f"p90={stats['p90_ms']:.2f}ms  std={stats['std_ms']:.2f}ms")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Bench FP16 attn-all latency on ANE")
    parser.add_argument("--model", required=True, help="HF model path")
    parser.add_argument("--output", default="/Volumes/MySSD/Anemll/qwen35_latency_bench",
                        help="Output dir (use SSD to avoid boot drive space issues)")
    parser.add_argument("--chunks", default="1,4,8", help="Comma-separated chunk indices")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations")
    parser.add_argument("--configs", default="A1_all_lut4,D2_fp16_attn_all",
                        help="Comma-separated config names to benchmark")
    args = parser.parse_args()

    # Redirect temp files to SSD (boot drive often full)
    tmpdir = os.path.join(args.output, ".tmp")
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    import tempfile
    tempfile.tempdir = tmpdir

    import coremltools as ct
    import coremltools.optimize as cto

    chunks = [int(x) for x in args.chunks.split(",")]
    configs = [c.strip() for c in args.configs.split(",")]
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("  ANE Latency Profiling: LUT4 vs FP16-attn-all")
    print("=" * 70)
    print(f"  Model:   {args.model}")
    print(f"  Chunks:  {chunks}")
    print(f"  Configs: {configs}")
    print(f"  Warmup:  {args.warmup}  Iters: {args.iters}")
    print(f"  Output:  {args.output}")

    # Load model once
    print("\n  Loading model weights...")
    t0 = time.time()
    model = _load_model(args.model)
    print(f"  Loaded in {time.time() - t0:.1f}s\n")

    all_results = {}

    for ci in chunks:
        sl, el = CHUNK_RANGES[ci]
        print(f"\n{'='*60}")
        print(f"  Chunk {ci}: layers [{sl}-{el-1}]")
        print(f"{'='*60}")

        # Convert to FP16 (unquantized reference) — lazy, only if needed
        fp16_mlmodel = None

        chunk_results = {}

        for config_name in configs:
            fp16_families = CONFIGS_TO_BENCH.get(config_name,
                                                  EXPERIMENT_CONFIGS.get(config_name, []))
            print(f"\n  Config: {config_name} (FP16: {fp16_families or 'none'})")

            # Check if .mlpackage already exists (skip re-palettization)
            pkg_dir = os.path.join(args.output, f"chunk{ci}_{config_name}")
            pkg_path = pkg_dir + ".mlpackage"

            if os.path.exists(pkg_path):
                print(f"    Using cached {pkg_path}")
                pkg_size_mb = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fns in os.walk(pkg_path)
                    for f in fns
                ) / 1e6
            else:
                # Need FP16 reference for palettization
                if fp16_mlmodel is None:
                    print(f"  Converting FP16 reference...")
                    t0 = time.time()
                    fp16_mlmodel = _convert_chunk_fp16(model, ci)
                    print(f"  Converted in {time.time()-t0:.1f}s")

                # Build palettizer config & quantize
                print(f"    Building config & palettizing...")
                t0 = time.time()
                opt_config = _build_selective_lut_config(
                    fp16_mlmodel, fp16_families,
                    lut_bits=4, per_channel=FFN_PER_CHANNEL,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    quant_mlmodel = cto.coreml.palettize_weights(fp16_mlmodel, opt_config)
                print(f"    Palettized in {time.time()-t0:.1f}s")

                # Save as .mlpackage
                if os.path.exists(pkg_path):
                    shutil.rmtree(pkg_path)
                print(f"    Saving {pkg_path}...")
                quant_mlmodel.save(pkg_path)

                # Get .mlpackage size
                pkg_size_mb = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fns in os.walk(pkg_path)
                    for f in fns
                ) / 1e6
                print(f"    Package size: {pkg_size_mb:.1f} MB")

                del quant_mlmodel
                gc.collect()

            # Load on ANE from .mlpackage (CoreML compiles internally)
            print(f"    Loading on ANE (CPU_AND_NE)...")
            t0 = time.time()
            ane_model = ct.models.MLModel(
                pkg_path,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )
            load_time = time.time() - t0
            print(f"    Loaded in {load_time:.1f}s")

            # Profile
            stats = profile_model(ane_model, n_warmup=args.warmup, n_iters=args.iters,
                                  label=config_name)
            stats["mlpackage_size_mb"] = pkg_size_mb
            stats["load_time_s"] = load_time
            chunk_results[config_name] = stats

            del ane_model
            gc.collect()
            time.sleep(1)  # Let ANE cool down between configs

        # Cleanup FP16 reference
        if fp16_mlmodel is not None:
            del fp16_mlmodel
        gc.collect()

        all_results[f"chunk{ci}"] = chunk_results

        # Print comparison for this chunk
        print(f"\n  --- Chunk {ci} Comparison ---")
        baseline = chunk_results.get("A1_all_lut4", {})
        for cfg_name, stats in chunk_results.items():
            p50 = stats.get("p50_ms", 0)
            size = stats.get("mlmodelc_size_mb", 0)
            delta_ms = p50 - baseline.get("p50_ms", p50)
            delta_pct = (delta_ms / baseline.get("p50_ms", 1)) * 100 if baseline.get("p50_ms") else 0
            print(f"    {cfg_name:<25s}  p50={p50:7.2f}ms  size={size:7.1f}MB  "
                  f"Δp50={delta_ms:+.2f}ms ({delta_pct:+.1f}%)")

    # Save results
    results_path = os.path.join(args.output, "latency_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Final summary table
    print("\n" + "=" * 90)
    print("  FINAL LATENCY COMPARISON")
    print("=" * 90)
    print(f"  {'Chunk':<10s} {'Config':<25s} {'p50 ms':>10s} {'mean ms':>10s} "
          f"{'p90 ms':>10s} {'size MB':>10s} {'Δp50':>10s}")
    print("-" * 90)

    for chunk_key in sorted(all_results.keys()):
        chunk_data = all_results[chunk_key]
        baseline_p50 = chunk_data.get("A1_all_lut4", {}).get("p50_ms", 0)
        for cfg_name in configs:
            if cfg_name not in chunk_data:
                continue
            s = chunk_data[cfg_name]
            delta = s["p50_ms"] - baseline_p50 if baseline_p50 > 0 else 0
            print(f"  {chunk_key:<10s} {cfg_name:<25s} {s['p50_ms']:>10.2f} {s['mean_ms']:>10.2f} "
                  f"{s['p90_ms']:>10.2f} {s.get('mlpackage_size_mb', 0):>10.1f} {delta:>+10.2f}")
        print()

    # Per-step total estimate (9 chunks)
    if "A1_all_lut4" in configs and "D2_fp16_attn_all" in configs:
        # Estimate total step time from profiled chunks
        a1_times = []
        d2_times = []
        for ck, cd in all_results.items():
            if "A1_all_lut4" in cd:
                a1_times.append(cd["A1_all_lut4"]["p50_ms"])
            if "D2_fp16_attn_all" in cd:
                d2_times.append(cd["D2_fp16_attn_all"]["p50_ms"])
        if a1_times and d2_times:
            avg_a1 = np.mean(a1_times)
            avg_d2 = np.mean(d2_times)
            total_a1 = avg_a1 * NUM_CHUNKS
            total_d2 = avg_d2 * NUM_CHUNKS
            print(f"  Estimated total per-step (9 chunks):")
            print(f"    A1_all_lut4:       {total_a1:.1f}ms → {1000/total_a1:.1f} tok/s")
            print(f"    D2_fp16_attn_all:  {total_d2:.1f}ms → {1000/total_d2:.1f} tok/s")
            print(f"    Overhead:          {total_d2-total_a1:+.1f}ms ({(total_d2-total_a1)/total_a1*100:+.1f}%)")


if __name__ == "__main__":
    main()
