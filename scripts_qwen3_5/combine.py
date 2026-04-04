#!/usr/bin/env python3
"""Qwen3.5-4B Milestone 2.1 — Step 2: Combine chunks with ANEMLL-Dedup.

Combines each decode + prefill into a single multi-function .mlpackage
with shared (deduplicated) weights.

Usage:
    python scripts_qwen3_5/combine.py --input /path/to/exported
    python scripts_qwen3_5/combine.py --skip-existing
"""
import os, time, argparse, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)  # must be first for config.py

from config import BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, FFN_LABEL, DEFAULT_OUTPUT
from anemll.utils.combine_models import _save_multifunction_dedup


def dir_size_mb(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Combine Qwen3.5-4B chunks (Milestone 2.1)")
    parser.add_argument("--input", default=DEFAULT_OUTPUT,
                        help="Directory with exported .mlpackage files")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-chunk", type=int, default=None,
                        help="Combine only the specified chunk index")
    args = parser.parse_args()

    label = FFN_LABEL
    combined_dir = os.path.join(args.input, f"combined_{label}_dedup")
    os.makedirs(combined_dir, exist_ok=True)

    # Verify sources
    # missing = []
    # for ci in range(NUM_CHUNKS):
    #     dec = os.path.join(args.input, f"ffn_{label}_chunk{ci}.mlpackage")
    #     if not os.path.exists(dec):
    #         missing.append(dec)
    #     pf = os.path.join(args.input, f"prefill_{label}_chunk{ci}.mlpackage")
    #     if not os.path.exists(pf):
    #         missing.append(pf)
    # if missing:
    #     print("ERROR: Missing source files:")
    #     for m in missing:
    #         print(f"  {m}")
    #     return 1

    print("=" * 70)
    print("  Qwen3.5-4B ANEMLL-Dedup Combine — Milestone 2.1")
    print(f"  Functions per chunk: infer + prefill")
    if args.only_chunk is not None:
        print(f"  Only chunk: {args.only_chunk}")
    print("=" * 70)

    t_total = time.time()
    total_size = 0.0
    chunk_indices = [args.only_chunk] if args.only_chunk is not None else list(range(NUM_CHUNKS))
    for ci in chunk_indices:
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if args.skip_existing and os.path.exists(combined_path):
            sz = dir_size_mb(combined_path)
            total_size += sz
            print(f"  [skip] chunk {ci} ({sz:.1f} MB)")
            continue

        dec_path = os.path.join(args.input, f"ffn_{label}_chunk{ci}.mlpackage")
        sources = [
            (dec_path, "main", "infer"),
        ]
        # Support both old naming (prefill_{label}_chunk{ci}.mlpackage) and
        # new bucket-based naming (prefill_{label}_chunk{ci}_bs{bucket}.mlpackage).
        pf_path_old = os.path.join(args.input, f"prefill_{label}_chunk{ci}.mlpackage")
        pf_path_bs = os.path.join(args.input, f"prefill_{label}_chunk{ci}_bs{BATCH_SIZE}.mlpackage")
        if os.path.exists(pf_path_bs):
            sources.append((pf_path_bs, "main", "prefill"))
        elif os.path.exists(pf_path_old):
            sources.append((pf_path_old, "main", "prefill"))
        else:
            print(f"  ERROR: No prefill found for chunk {ci}")
            return 1

        print(f"  Combining chunk {ci} (infer, prefill)...")
        t0 = time.time()
        _save_multifunction_dedup(sources, combined_path, dedup_weights=True, verbose=False)
        sz = dir_size_mb(combined_path)
        total_size += sz
        print(f"    Done ({time.time()-t0:.1f}s) — {sz:.1f} MB")

    print(f"\n  Total combined: {total_size:.1f} MB")
    print(f"  Elapsed: {time.time()-t_total:.1f}s")
    print(f"\nNext: python scripts_qwen3_5/compile.py --model-dir {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
