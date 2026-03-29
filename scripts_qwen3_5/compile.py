#!/usr/bin/env python3
"""Qwen3.5-4B Milestone 2.1 — Step 3: Compile .mlpackage to .mlmodelc.

Usage:
    python scripts_qwen3_5/compile.py --model-dir /path/to/models
    python scripts_qwen3_5/compile.py  # uses default output dir
"""
import os, glob, time, subprocess, argparse, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)  # must be first for config.py

from config import DEFAULT_OUTPUT, FFN_LABEL


def compile_model(mlpackage_path, output_dir):
    name = os.path.basename(mlpackage_path)
    out_name = name.replace(".mlpackage", ".mlmodelc")
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path):
        print(f"  [skip] {out_name}")
        return True

    print(f"  Compiling {name}...")
    t0 = time.time()
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", mlpackage_path, output_dir],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0

    if result.returncode == 0 and os.path.exists(out_path):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(out_path) for f in fns
        ) / (1024 * 1024)
        print(f"    OK: {out_name} ({size_mb:.0f} MB, {elapsed:.1f}s)")
        return True
    else:
        print(f"    FAIL ({elapsed:.1f}s): {result.stderr[:200]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compile Qwen3.5-4B models")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--output", default=None, help="Output dir (default: same as model-dir)")
    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output or model_dir

    print("=" * 70)
    print("  Qwen3.5-4B Model Compilation — Milestone 2.1")
    print(f"  Input:  {model_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    ok = fail = 0

    # Separate models
    print("\n── Separate Models ──")
    for p in sorted(glob.glob(os.path.join(model_dir, "*.mlpackage"))):
        if compile_model(p, output_dir):
            ok += 1
        else:
            fail += 1

    # Combined dedup models
    combined_dir = os.path.join(model_dir, f"combined_{FFN_LABEL}_dedup")
    if os.path.isdir(combined_dir):
        combined_out = os.path.join(output_dir, f"combined_{FFN_LABEL}_dedup")
        os.makedirs(combined_out, exist_ok=True)
        print("\n── Combined Dedup Models ──")
        for p in sorted(glob.glob(os.path.join(combined_dir, "chunk*.mlpackage"))):
            if compile_model(p, combined_out):
                ok += 1
            else:
                fail += 1

    print(f"\nDone: {ok} compiled, {fail} failed")
    print(f"\nNext: python scripts_qwen3_5/chat_server.py --model-dir {output_dir}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
