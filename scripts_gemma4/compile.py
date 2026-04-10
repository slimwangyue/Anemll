#!/usr/bin/env python3
"""Compile Gemma4 E4B CoreML .mlpackage to .mlmodelc.

Usage:
    cd /path/to/Anemll
    python scripts_gemma4/compile.py [--model-dir gemma4_E4B_lut4ffn_lut6em]
"""
import argparse
import glob
import os
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(1, os.path.dirname(_SCRIPT_DIR))

from config import DEFAULT_OUTPUT, FFN_LABEL


def compile_model(mlpackage_path, output_dir):
    """Compile a single .mlpackage to .mlmodelc."""
    basename = os.path.splitext(os.path.basename(mlpackage_path))[0]
    mlmodelc_path = os.path.join(output_dir, f"{basename}.mlmodelc")

    if os.path.exists(mlmodelc_path):
        print(f"  Skipping (exists): {mlmodelc_path}")
        return True

    print(f"  Compiling: {mlpackage_path}")
    t0 = time.time()
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", mlpackage_path, output_dir],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    # Report size
    if os.path.exists(mlmodelc_path):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(mlmodelc_path) for f in fns
        ) / (1024 * 1024)
        print(f"  OK: {basename}.mlmodelc ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compile Gemma4 CoreML models")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT, help="Model directory")
    parser.add_argument("--output", default=None, help="Output directory (default: same as model-dir)")
    args = parser.parse_args()

    output_dir = args.output or args.model_dir

    # 1. Compile top-level .mlpackage files
    packages = sorted(glob.glob(os.path.join(args.model_dir, "*.mlpackage")))
    if packages:
        print(f"\nCompiling {len(packages)} top-level models...")
        for pkg in packages:
            compile_model(pkg, output_dir)

    # 2. Compile combined chunks
    combined_dir = os.path.join(args.model_dir, f"combined_{FFN_LABEL}_dedup")
    if os.path.isdir(combined_dir):
        combined_packages = sorted(glob.glob(os.path.join(combined_dir, "*.mlpackage")))
        if combined_packages:
            combined_output = os.path.join(output_dir, f"combined_{FFN_LABEL}_dedup")
            os.makedirs(combined_output, exist_ok=True)
            print(f"\nCompiling {len(combined_packages)} combined models...")
            for pkg in combined_packages:
                compile_model(pkg, combined_output)

    print("\nCompile complete.")


if __name__ == "__main__":
    main()
