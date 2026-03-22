#!/usr/bin/env python3
"""Qwen3.5-4B Milestone 1.2: Combine decode+prefill chunks with ANEMLL-Dedup.

Takes the separate .mlpackage files from qwen35_export.py and combines
each decode + prefill into a single multi-function .mlpackage with
shared (deduplicated) weights.

Output structure:
  combined_LUT4_dedup/
    chunk0.mlpackage  (functions: infer, prefill)
    chunk1.mlpackage
    chunk2.mlpackage
    chunk3.mlpackage

Usage:
    python tests/dev/qwen35_combine.py --input /path/to/exported
    python tests/dev/qwen35_combine.py --input /path/to/exported --skip-existing
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time, argparse
from anemll.utils.combine_models import _save_multifunction_dedup

BATCH_SIZE = 256   # prefill input length
CTX = 1024         # KV cache / context length
NUM_CHUNKS = 4
LUT_BITS = 4


def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Combine Qwen3.5-4B chunks with ANEMLL-Dedup")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory with exported .mlpackage files from qwen35_export.py")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip combining if combined chunk already exists")
    args = parser.parse_args()

    label = f"LUT{LUT_BITS}"
    combined_dir = os.path.join(args.input, f"combined_{label}_dedup")
    os.makedirs(combined_dir, exist_ok=True)

    print("=" * 70)
    print("  Qwen3.5-4B ANEMLL-Dedup Combine — Milestone 1.2")
    print(f"  Input: {args.input}")
    print(f"  Output: {combined_dir}")
    print(f"  Functions per chunk: infer + prefill")
    print("=" * 70)

    # Verify all source files exist
    missing = []
    for ci in range(NUM_CHUNKS):
        dec = os.path.join(args.input, f"ffn_{label}_chunk{ci}.mlpackage")
        if not os.path.exists(dec):
            missing.append(dec)
        pf = os.path.join(args.input, f"prefill_{label}_chunk{ci}.mlpackage")
        if not os.path.exists(pf):
            missing.append(pf)
    if missing:
        print("\nERROR: Missing source files:")
        for m in missing:
            print(f"  {m}")
        print(f"\nRun qwen35_export.py first:")
        print(f"  python tests/dev/qwen35_export.py --model <HF_MODEL> --output {args.input}")
        sys.exit(1)

    t_total = time.time()
    total_size = 0.0

    for ci in range(NUM_CHUNKS):
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if args.skip_existing and os.path.exists(combined_path):
            sz = dir_size_mb(combined_path)
            total_size += sz
            print(f"  [skip] chunk {ci} ({sz:.1f} MB)")
            continue

        dec_path = os.path.join(args.input, f"ffn_{label}_chunk{ci}.mlpackage")
        pf_path = os.path.join(args.input, f"prefill_{label}_chunk{ci}.mlpackage")

        sources = [
            (dec_path, "main", "infer"),
            (pf_path, "main", "prefill"),
        ]

        t0 = time.time()
        print(f"  Combining chunk {ci} (infer, prefill)...")
        _save_multifunction_dedup(sources, combined_path,
                                  dedup_weights=True, verbose=False)
        sz = dir_size_mb(combined_path)
        total_size += sz
        print(f"    Done ({time.time()-t0:.1f}s) — {sz:.1f} MB")

    # Size comparison
    sep_dec = sum(dir_size_mb(os.path.join(args.input, f"ffn_{label}_chunk{ci}.mlpackage"))
                  for ci in range(NUM_CHUNKS))
    sep_pf = sum(dir_size_mb(os.path.join(args.input, f"prefill_{label}_chunk{ci}.mlpackage"))
                 for ci in range(NUM_CHUNKS))
    embed_sz = dir_size_mb(os.path.join(args.input, "embeddings.mlpackage"))
    lmhead_sz = dir_size_mb(os.path.join(args.input, "lm_head.mlpackage"))

    sep_total = sep_dec + sep_pf + embed_sz + lmhead_sz
    dedup_total = total_size + embed_sz + lmhead_sz

    print(f"\n{'='*70}")
    print("  SIZE COMPARISON")
    print(f"{'='*70}")
    print(f"  Separate decode:    {sep_dec:>8.1f} MB ({NUM_CHUNKS} chunks)")
    print(f"  Separate prefill:   {sep_pf:>8.1f} MB ({NUM_CHUNKS} chunks)")
    print(f"  Embed + LM Head:    {embed_sz + lmhead_sz:>8.1f} MB")
    print(f"  TOTAL (separate):   {sep_total:>8.1f} MB")
    print(f"  TOTAL (dedup):      {dedup_total:>8.1f} MB  ({(1-dedup_total/sep_total)*100:.1f}% saving)")
    print(f"\n  Elapsed: {time.time()-t_total:.1f}s")

    print(f"\nDeployable model set:")
    print(f"  {os.path.join(args.input, 'embeddings.mlpackage')}")
    print(f"  {os.path.join(args.input, 'lm_head.mlpackage')}")
    print(f"  {combined_dir}/chunk{{0..{NUM_CHUNKS-1}}}.mlpackage")
    print(f"\nNext: python tests/dev/qwen35_validate.py --model-dir {args.input}")


if __name__ == "__main__":
    main()
