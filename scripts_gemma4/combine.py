#!/usr/bin/env python3
"""Combine Gemma4 E4B CoreML models with weight deduplication.

Combines decode + prefill chunks into multi-function .mlpackage files.

Usage:
    cd /path/to/Anemll
    python scripts_gemma4/combine.py [--combine-embed-lmhead]
"""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(1, os.path.dirname(_SCRIPT_DIR))

from config import NUM_CHUNKS, FFN_LABEL, DEFAULT_OUTPUT
from anemll.utils.combine_models import _save_multifunction_dedup


def combine_chunks(model_dir, output_dir, label, num_chunks):
    """Combine decode + prefill for each chunk."""
    os.makedirs(output_dir, exist_ok=True)

    for ci in range(num_chunks):
        decode_path = os.path.join(model_dir, f"decode_{label}_chunk{ci:02d}.mlpackage")
        prefill_path = os.path.join(model_dir, f"prefill_{label}_chunk{ci:02d}.mlpackage")
        out_path = os.path.join(output_dir, f"chunk{ci:02d}.mlpackage")

        if os.path.exists(out_path):
            print(f"  Skipping (exists): {out_path}")
            continue

        if not os.path.exists(decode_path):
            print(f"  WARNING: Missing {decode_path}")
            continue
        if not os.path.exists(prefill_path):
            print(f"  WARNING: Missing {prefill_path}")
            continue

        print(f"  Combining chunk {ci}: decode + prefill -> {out_path}")

        sources = [
            (decode_path, "main", "infer"),
            (prefill_path, "main", "prefill"),
        ]

        _save_multifunction_dedup(
            sources=sources,
            output_path=out_path,
            dedup_weights=True,
        )


def combine_embed_lmhead(model_dir, output_dir):
    """Combine embeddings + lm_head into single package."""
    embed_path = os.path.join(model_dir, "embeddings.mlpackage")
    lm_head_paths = [f for f in os.listdir(model_dir) if f.startswith("lm_head_") and f.endswith(".mlpackage")]

    if not lm_head_paths:
        print("  WARNING: No lm_head .mlpackage found")
        return

    lm_head_path = os.path.join(model_dir, lm_head_paths[0])
    out_path = os.path.join(output_dir, "embed_lmhead_combined.mlpackage")

    if os.path.exists(out_path):
        print(f"  Skipping (exists): {out_path}")
        return

    if not os.path.exists(embed_path):
        print(f"  WARNING: Missing {embed_path}")
        return

    print(f"  Combining: {embed_path} + {lm_head_path}")

    sources = [
        (embed_path, "main", "embed"),
        (lm_head_path, "main", "lm_head"),
    ]

    _save_multifunction_dedup(
        sources=sources,
        output_path=out_path,
        dedup_weights=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Combine Gemma4 CoreML models")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT, help="Input model directory")
    parser.add_argument("--label", default=FFN_LABEL, help="FFN label (e.g., LUT4)")
    parser.add_argument("--combine-embed-lmhead", action="store_true", help="Also combine embed+lmhead")
    args = parser.parse_args()

    output_dir = os.path.join(args.model_dir, f"combined_{args.label}_dedup")
    print(f"Input: {args.model_dir}")
    print(f"Output: {output_dir}")

    combine_chunks(args.model_dir, output_dir, args.label, NUM_CHUNKS)

    if args.combine_embed_lmhead:
        combine_embed_lmhead(args.model_dir, output_dir)

    print("\nCombine complete.")


if __name__ == "__main__":
    main()
