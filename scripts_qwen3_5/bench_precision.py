#!/usr/bin/env python3
"""Quick ANE utilization comparison between fp16, fp32, and selective-fp32 exports."""
import sys, os, time, argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from config import CTX

import coremltools as ct


def measure_chunk(model_dir, chunk_idx, label, compute_unit, function_name=None, warmup=5, iters=30):
    combined_dir = os.path.join(model_dir, f"combined_{label}_dedup")
    if os.path.isdir(combined_dir):
        path = os.path.join(combined_dir, f"chunk{chunk_idx}.mlpackage")
        fn = "infer"
    else:
        path = os.path.join(model_dir, f"ffn_{label}_chunk{chunk_idx}.mlpackage")
        fn = function_name

    if not os.path.exists(path):
        return None

    print(f"  Loading {path} ({compute_unit})...")
    kwargs = {"compute_units": compute_unit}
    if fn:
        kwargs["function_name"] = fn
    m = ct.models.MLModel(path, **kwargs)

    state = m.make_state()
    spec = m.get_spec()

    # Get input shapes
    imap = {}
    if fn:
        for f in spec.description.functions:
            if f.name == fn:
                for inp in f.input:
                    try:
                        imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except:
                        pass
                break
    if not imap:
        for inp in spec.description.input:
            try:
                imap[inp.name] = tuple(inp.type.multiArrayType.shape)
            except:
                pass

    # Build test inputs
    hidden = np.random.randn(1, 1, 2560).astype(np.float16)
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :10] = 0
    inp = {
        "hidden_states": hidden,
        "position_ids": np.array([5], dtype=np.int32),
        "causal_mask": mask,
        "current_pos": np.array([5], dtype=np.int32),
    }
    if "linear_conv_state" in imap:
        inp["linear_conv_state"] = np.zeros(imap["linear_conv_state"], dtype=np.float16)
        inp["linear_recurrent_state"] = np.zeros(imap["linear_recurrent_state"], dtype=np.float16)

    # Warmup
    for _ in range(warmup):
        m.predict(inp, state=state)

    # Measure
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        m.predict(inp, state=state)
        times.append((time.perf_counter() - t0) * 1000)

    del m
    import gc; gc.collect()

    med = sorted(times)[len(times) // 2]
    mn = min(times)
    avg = sum(times) / len(times)
    return {"median": med, "min": mn, "avg": avg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default="0,1", help="Chunks to test")
    parser.add_argument("--label", type=str, default="LUT4")
    args = parser.parse_args()

    chunks = [int(x) for x in args.chunks.split(",")]

    dirs = {
        "fp16": "qwen3_5_v4_lut4_fp16",
        "fp32": "qwen3_5_stable_lut4ffn_lut6em_fp32",
        "sel-fp32": "qwen3_5_v4_lut4_selfp32",
    }

    results = {}
    for name, d in dirs.items():
        if not os.path.isdir(d):
            continue
        results[name] = {}
        print(f"\n{'='*60}")
        print(f"  {name} ({d})")
        print(f"{'='*60}")
        for ci in chunks:
            r = measure_chunk(d, ci, args.label, ct.ComputeUnit.CPU_AND_NE)
            if r:
                results[name][ci] = r
                print(f"  Chunk {ci}: median={r['median']:.2f}ms  min={r['min']:.2f}ms  avg={r['avg']:.2f}ms")
            else:
                print(f"  Chunk {ci}: not found")

    # Also measure fp16 on CPU_ONLY for reference
    print(f"\n{'='*60}")
    print(f"  fp16-cpu (CPU_ONLY reference)")
    print(f"{'='*60}")
    results["fp16-cpu"] = {}
    for ci in chunks:
        r = measure_chunk(dirs["fp16"], ci, args.label, ct.ComputeUnit.CPU_ONLY)
        if r:
            results["fp16-cpu"][ci] = r
            print(f"  Chunk {ci}: median={r['median']:.2f}ms  min={r['min']:.2f}ms  avg={r['avg']:.2f}ms")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY (median ms)")
    print(f"{'='*60}")
    print(f"  {'Config':<15s}", end="")
    for ci in chunks:
        print(f"  Chunk {ci:>2d}", end="")
    print()
    for name in results:
        print(f"  {name:<15s}", end="")
        for ci in chunks:
            if ci in results[name]:
                print(f"  {results[name][ci]['median']:>7.2f}", end="")
            else:
                print(f"  {'N/A':>7s}", end="")
        print()


if __name__ == "__main__":
    main()
