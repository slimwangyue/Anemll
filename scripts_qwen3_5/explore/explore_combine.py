#!/usr/bin/env python3
"""Qwen3.5-4B — Exploratory combine: merge decode + prefill into dedup models.

Usage:
    python scripts_qwen3_5/explore/explore_combine.py --config batch512_ctx1024
    python scripts_qwen3_5/explore/explore_combine.py --config all
"""
import os, time, argparse, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts_qwen3_5.explore.explore_config import CONFIGS, get_config, list_configs
from anemll.utils.combine_models import _save_multifunction_dedup


def dir_size_mb(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def combine_config(cfg, skip_existing=False):
    name = cfg["name"]
    out = cfg["output_dir"]
    chunks = cfg["NUM_CHUNKS"]
    lut = cfg["LUT_BITS"]
    label = f"LUT{lut}"
    combined_dir = os.path.join(out, f"combined_{label}_dedup")
    os.makedirs(combined_dir, exist_ok=True)

    # Verify sources
    missing = []
    for ci in range(chunks):
        for prefix in [f"ffn_{label}_chunk{ci}", f"prefill_{label}_chunk{ci}"]:
            p = os.path.join(out, f"{prefix}.mlpackage")
            if not os.path.exists(p):
                missing.append(p)
    if missing:
        print(f"ERROR [{name}]: Missing files:")
        for m in missing:
            print(f"  {m}")
        return 1

    print(f"\n{'='*70}")
    print(f"  COMBINE: {name}")
    print(f"{'='*70}")

    t_total = time.time()
    total_size = 0.0
    for ci in range(chunks):
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if skip_existing and os.path.exists(combined_path):
            sz = dir_size_mb(combined_path)
            total_size += sz
            print(f"  [skip] chunk {ci} ({sz:.1f} MB)")
            continue

        dec_path = os.path.join(out, f"ffn_{label}_chunk{ci}.mlpackage")
        pf_path = os.path.join(out, f"prefill_{label}_chunk{ci}.mlpackage")
        sources = [
            (dec_path, "main", "infer"),
            (pf_path, "main", "prefill"),
        ]
        print(f"  Combining chunk {ci}...")
        t0 = time.time()
        _save_multifunction_dedup(sources, combined_path, dedup_weights=True, verbose=False)
        sz = dir_size_mb(combined_path)
        total_size += sz
        print(f"    Done ({time.time()-t0:.1f}s) — {sz:.1f} MB")

    print(f"\n  Total combined: {total_size:.1f} MB")
    print(f"  Elapsed: {time.time()-t_total:.1f}s")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Combine exploratory Qwen3.5-4B chunks")
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--skip-existing", action="store_true")
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
        combine_config(cfg, args.skip_existing)


if __name__ == "__main__":
    main()
