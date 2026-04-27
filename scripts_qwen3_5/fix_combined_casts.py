#!/usr/bin/env python3
"""Post-combine fix: inject state casts into combined mlpackages.

The combine step re-runs MIL optimization passes that strip the cast ops
injected during export. This script re-applies the fix to the combined
mlpackages, then they must be recompiled.

Usage:
    python scripts_qwen3_5/fix_combined_casts.py --model-dir qwen3_5_4B_milestone_3.4_fix
"""
import os, sys, argparse, time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from config import NUM_CHUNKS, FFN_LABEL, DEFAULT_OUTPUT
from export import _ensure_state_slice_update_casts, CHUNK_RANGES, F_LAYERS

import coremltools as ct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    label = FFN_LABEL
    combined_dir = os.path.join(args.model_dir, f"combined_{label}_dedup")

    print("=" * 70)
    print("  Post-combine state-cast injection")
    print(f"  Dir: {combined_dir}")
    print("=" * 70)

    for ci in range(NUM_CHUNKS):
        sl, el = CHUNK_RANGES[ci]
        has_f = any(li in F_LAYERS for li in range(sl, el))
        if not has_f:
            print(f"  chunk {ci}: L-only, skip")
            continue

        pkg = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if not os.path.exists(pkg):
            print(f"  chunk {ci}: {pkg} not found, skip")
            continue

        print(f"  chunk {ci}: loading...", end="", flush=True)
        t0 = time.time()
        model = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_ONLY,
                                   function_name="prefill")
        print(f" loaded ({time.time()-t0:.1f}s)", end="", flush=True)

        n = _ensure_state_slice_update_casts(model)
        print(f", injected {n} casts", end="", flush=True)

        if n > 0:
            model.save(pkg)
            print(f", saved ({time.time()-t0:.1f}s)")
        else:
            print(f", no changes needed")

    print("\nDone. Re-compile with:")
    print(f"  python scripts_qwen3_5/compile.py --model-dir {args.model_dir}")


if __name__ == "__main__":
    main()
