#!/usr/bin/env python3
"""Re-deduplicate existing combined mlpackages in-place.

The original combine skipped dedup because preflight rejected the I/O
signature mismatch (prefill has valid_len, infer does not).  After fixing
_preflight_check_io_signature to allow extra target inputs, this script
re-applies dedup to the existing combined models without re-exporting.

Usage:
    python scripts_qwen3_5/rededup.py
    python scripts_qwen3_5/rededup.py --input /path/to/models
    python scripts_qwen3_5/rededup.py --only-chunk 0   # single chunk
"""
import os, sys, time, tempfile, shutil, argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct
from config import NUM_CHUNKS, FFN_LABEL, DEFAULT_OUTPUT
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
    parser = argparse.ArgumentParser(description="Re-dedup existing combined mlpackages")
    parser.add_argument("--input", default=DEFAULT_OUTPUT,
                        help="Root model directory")
    parser.add_argument("--only-chunk", type=int, default=None)
    args = parser.parse_args()

    label = FFN_LABEL
    combined_dir = os.path.join(args.input, f"combined_{label}_dedup")

    chunk_indices = [args.only_chunk] if args.only_chunk is not None else list(range(NUM_CHUNKS))

    print("=" * 70)
    print("  Re-dedup: apply weight sharing to existing combined models")
    print(f"  Combined dir: {combined_dir}")
    print(f"  Chunks: {chunk_indices}")
    print("=" * 70)

    t_total = time.time()
    for ci in chunk_indices:
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if not os.path.exists(combined_path):
            print(f"  ERROR: {combined_path} not found")
            return 1

        size_before = dir_size_mb(combined_path)
        print(f"\n  Chunk {ci}: {size_before:.1f} MB (before)")

        tmp = tempfile.mkdtemp(prefix=f"rededup_chunk{ci}_")
        try:
            # Extract each function to a separate mlpackage
            infer_tmp = os.path.join(tmp, "infer.mlpackage")
            prefill_tmp = os.path.join(tmp, "prefill.mlpackage")

            print(f"    Extracting infer → {infer_tmp}")
            m_infer = ct.models.MLModel(combined_path, function_name="infer")
            m_infer.save(infer_tmp)
            del m_infer

            print(f"    Extracting prefill → {prefill_tmp}")
            m_prefill = ct.models.MLModel(combined_path, function_name="prefill")
            m_prefill.save(prefill_tmp)
            del m_prefill

            # Re-combine with dedup
            dedup_out = os.path.join(tmp, f"chunk{ci}_dedup.mlpackage")
            print(f"    Combining with dedup...")
            sources = [
                (infer_tmp, "infer", "infer"),
                (prefill_tmp, "prefill", "prefill"),
            ]
            t0 = time.time()
            _save_multifunction_dedup(sources, dedup_out, dedup_weights=True, verbose=True)
            dt = time.time() - t0

            size_after = dir_size_mb(dedup_out)
            saved = size_before - size_after
            print(f"    Dedup done ({dt:.1f}s): {size_before:.1f} → {size_after:.1f} MB "
                  f"(saved {saved:.1f} MB, {100*saved/size_before:.0f}%)")

            # Replace original
            shutil.rmtree(combined_path)
            shutil.move(dedup_out, combined_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    total_size = sum(dir_size_mb(os.path.join(combined_dir, f"chunk{ci}.mlpackage"))
                     for ci in chunk_indices
                     if os.path.exists(os.path.join(combined_dir, f"chunk{ci}.mlpackage")))
    print(f"\n  Total: {total_size:.1f} MB | Elapsed: {time.time()-t_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
