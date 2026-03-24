#!/usr/bin/env python3
"""Qwen3.5-4B — Exploratory compile: .mlpackage -> .mlmodelc via coremlcompiler.

Usage:
    python scripts_qwen3_5/explore/explore_compile.py --config batch512_ctx1024
    python scripts_qwen3_5/explore/explore_compile.py --config all
"""
import os, time, subprocess, argparse, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts_qwen3_5.explore.explore_config import CONFIGS, get_config, list_configs


def compile_model(mlpackage_path, output_dir):
    name = os.path.basename(mlpackage_path)
    out_name = name.replace(".mlpackage", ".mlmodelc")
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(out_path) for f in fns
        ) / (1024 * 1024)
        print(f"  [skip] {out_name} ({size_mb:.0f} MB)")
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
        print(f"    FAIL ({elapsed:.1f}s): {result.stderr[:500]}")
        return False


def compile_config(cfg):
    name = cfg["name"]
    out = cfg["output_dir"]
    lut = cfg["LUT_BITS"]
    label = f"LUT{lut}"

    print(f"\n{'='*70}")
    print(f"  COMPILE: {name}")
    print(f"{'='*70}")

    t_total = time.time()
    ok = 0
    fail = 0

    # Compile embeddings
    p = os.path.join(out, "embeddings.mlpackage")
    if os.path.exists(p):
        if compile_model(p, out):
            ok += 1
        else:
            fail += 1

    # Compile lm_head
    p = os.path.join(out, "lm_head.mlpackage")
    if os.path.exists(p):
        if compile_model(p, out):
            ok += 1
        else:
            fail += 1

    # Compile combined chunks (if they exist)
    combined_dir = os.path.join(out, f"combined_{label}_dedup")
    if os.path.exists(combined_dir):
        for ci in range(cfg["NUM_CHUNKS"]):
            p = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
            if os.path.exists(p):
                if compile_model(p, combined_dir):
                    ok += 1
                else:
                    fail += 1
    else:
        # Compile individual ffn/prefill chunks
        for ci in range(cfg["NUM_CHUNKS"]):
            for prefix in [f"ffn_{label}_chunk{ci}", f"prefill_{label}_chunk{ci}"]:
                p = os.path.join(out, f"{prefix}.mlpackage")
                if os.path.exists(p):
                    if compile_model(p, out):
                        ok += 1
                    else:
                        fail += 1

    print(f"\n  Results: {ok} OK, {fail} FAIL ({time.time()-t_total:.1f}s)")
    return fail == 0


def main():
    parser = argparse.ArgumentParser(description="Compile exploratory Qwen3.5-4B models")
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_configs()
        return

    if not args.config:
        print("ERROR: --config required")
        sys.exit(1)

    if args.config == "all":
        configs = list(CONFIGS.values())
    else:
        configs = [get_config(args.config)]

    for cfg in configs:
        compile_config(cfg)


if __name__ == "__main__":
    main()
