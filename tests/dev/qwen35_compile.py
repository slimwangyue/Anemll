#!/usr/bin/env python3
"""Compile Qwen3.5-4B .mlpackage models to .mlmodelc for faster loading.

Usage:
    python tests/dev/qwen35_compile.py [--model-dir /path/to/models] [--output /path/to/output]

Compiles all .mlpackage files (separate + combined dedup) to .mlmodelc using
xcrun coremlcompiler. Compiled models load significantly faster.

Note: Multi-function .mlmodelc may not support function_name on ANE.
      For batch prefill, use separate prefill .mlmodelc files instead.
"""
import os
import sys
import glob
import time
import subprocess
import argparse


def compile_model(mlpackage_path, output_dir):
    """Compile a .mlpackage to .mlmodelc using xcrun coremlcompiler."""
    name = os.path.basename(mlpackage_path)
    if not os.path.exists(mlpackage_path):
        print(f"  SKIP (not found): {name}")
        return False

    # Output will be <output_dir>/<name with .mlmodelc instead of .mlpackage>
    out_name = name.replace(".mlpackage", ".mlmodelc")
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path):
        print(f"  SKIP (exists): {out_name}")
        return True

    print(f"  Compiling {name}...")
    t0 = time.time()
    cmd = ["xcrun", "coremlcompiler", "compile", mlpackage_path, output_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0 and os.path.exists(out_path):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(out_path)
            for f in fns
        ) / (1024 * 1024)
        print(f"    OK: {out_name} ({size_mb:.0f} MB, {elapsed:.1f}s)")
        return True
    else:
        print(f"    FAIL ({elapsed:.1f}s): {result.stderr[:200]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compile Qwen3.5-4B models to .mlmodelc")
    parser.add_argument("--model-dir", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1",
                        help="Directory containing .mlpackage models")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: same as model-dir)")
    parser.add_argument("--separate-only", action="store_true",
                        help="Only compile separate models (skip combined dedup)")
    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output or model_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Qwen3.5-4B Model Compilation (.mlpackage -> .mlmodelc)")
    print("=" * 60)
    print(f"  Input:  {model_dir}")
    print(f"  Output: {output_dir}")

    results = {"ok": 0, "fail": 0, "skip": 0}

    # ── Separate models ──
    print("\n── Separate Models ──")
    separate_models = [
        "embeddings",
        "lm_head",
    ]
    # Find FFN and prefill chunks
    for pattern in ["ffn_LUT4_chunk*.mlpackage", "prefill_LUT4_chunk*.mlpackage"]:
        for p in sorted(glob.glob(os.path.join(model_dir, pattern))):
            name = os.path.basename(p).replace(".mlpackage", "")
            separate_models.append(name)

    for name in separate_models:
        src = os.path.join(model_dir, f"{name}.mlpackage")
        if compile_model(src, output_dir):
            results["ok"] += 1
        else:
            results["fail"] += 1

    # ── Combined dedup models ──
    if not args.separate_only:
        combined_dir = os.path.join(model_dir, "combined_LUT4_dedup")
        if os.path.isdir(combined_dir):
            combined_out = os.path.join(output_dir, "combined_LUT4_dedup")
            os.makedirs(combined_out, exist_ok=True)
            print("\n── Combined Dedup Models ──")
            for p in sorted(glob.glob(os.path.join(combined_dir, "chunk*.mlpackage"))):
                if compile_model(p, combined_out):
                    results["ok"] += 1
                else:
                    results["fail"] += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Done: {results['ok']} compiled, {results['fail']} failed")
    total_mlmodelc = len(glob.glob(os.path.join(output_dir, "*.mlmodelc")))
    combined_mlmodelc = len(glob.glob(os.path.join(output_dir, "combined_LUT4_dedup", "*.mlmodelc")))
    print(f"  Separate .mlmodelc: {total_mlmodelc}")
    print(f"  Combined .mlmodelc: {combined_mlmodelc}")

    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
