#!/usr/bin/env python3
"""Qwen3.5-4B Milestone 1: Export all CoreML model components.

Exports the stable B+E config:
  - embeddings (LUT4)        → embeddings.mlpackage
  - lm_head (fp16)           → lm_head.mlpackage
  - 4 FFN decode chunks (LUT4)  → ffn_LUT4_chunk{0..3}.mlpackage
  - 4 FFN prefill chunks (LUT4) → prefill_LUT4_chunk{0..3}.mlpackage

After export, run qwen35_combine.py to create dedup-combined models,
then qwen35_validate.py for multi-round conversation validation.

Usage:
    python tests/dev/qwen35_export.py --model /path/to/Qwen3.5-4B --output /path/to/output
    python tests/dev/qwen35_export.py --model /path/to/Qwen3.5-4B --output /path/to/output --skip-existing
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import gc, time, argparse
import numpy as np
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

# ── Config ──
BATCH_SIZE = 256   # prefill input length
CTX = 1024         # KV cache / context length
NUM_CHUNKS = 4
LUT_BITS = 4
PER_CHANNEL = 8


def export_embeddings(model, out_dir, skip_existing):
    path = os.path.join(out_dir, "embeddings.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] {path}")
        return
    print("  Exporting embeddings (LUT4)...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS,
                           per_channel=PER_CHANNEL)
    ml = conv.convert_part_1(model)
    ml.save(path)
    del ml, conv; gc.collect()
    print(f"  Saved embeddings ({time.time()-t0:.1f}s)")


def export_lm_head(model, out_dir, skip_existing):
    path = os.path.join(out_dir, "lm_head.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] {path}")
        return
    print("  Exporting lm_head (fp16)...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=None)
    ml = conv.convert_part_3(model, argmax_in_model=False)
    ml.save(path)
    del ml, conv; gc.collect()
    print(f"  Saved lm_head ({time.time()-t0:.1f}s)")


def export_ffn_chunks(model, out_dir, skip_existing):
    label = f"LUT{LUT_BITS}"
    for ci in range(NUM_CHUNKS):
        # Decode chunk
        dec_path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        if skip_existing and os.path.exists(dec_path):
            print(f"  [skip] decode chunk {ci}")
        else:
            print(f"  Exporting decode chunk {ci} ({label})...")
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                   num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS,
                                   per_channel=PER_CHANNEL)
            ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
            ml.save(dec_path)
            del ml, conv; gc.collect()
            print(f"  Saved decode chunk {ci} ({time.time()-t0:.1f}s)")

        # Prefill chunk
        pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage")
        if skip_existing and os.path.exists(pf_path):
            print(f"  [skip] prefill chunk {ci}")
        else:
            print(f"  Exporting prefill chunk {ci} ({label})...")
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                   num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS,
                                   per_channel=PER_CHANNEL)
            ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
            ml.save(pf_path)
            del ml, conv; gc.collect()
            print(f"  Saved prefill chunk {ci} ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5-4B for ANE (Milestone 1)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace Qwen3.5-4B model directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for exported .mlpackage files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if .mlpackage already exists")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("  Qwen3.5-4B ANE Export — Milestone 1 (B+E Config)")
    print(f"  Embed: LUT4 | LM Head: fp16 | FFN: LUT4 × {NUM_CHUNKS} chunks")
    print(f"  Batch: {BATCH_SIZE} | CTX: {CTX}")
    print(f"  per_channel={PER_CHANNEL}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    # Load model
    print("\nLoading model weights...")
    t_load = time.time()
    cfg = Qwen35Config.from_json(os.path.join(args.model, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(args.model), f"Failed to load weights from {args.model}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t_load:.1f}s")

    # Export all parts
    t_total = time.time()

    print("\n[1/3] Embeddings")
    export_embeddings(model, args.output, args.skip_existing)

    print("\n[2/3] LM Head")
    export_lm_head(model, args.output, args.skip_existing)

    print("\n[3/3] FFN Chunks (decode + prefill)")
    export_ffn_chunks(model, args.output, args.skip_existing)

    del model; gc.collect()

    # Summary
    total_mb = 0
    print(f"\n{'='*70}")
    print("  EXPORT SUMMARY")
    print(f"{'='*70}")
    for f in sorted(os.listdir(args.output)):
        full = os.path.join(args.output, f)
        if os.path.isdir(full):
            sz = sum(os.path.getsize(os.path.join(dp, fn))
                     for dp, _, fns in os.walk(full) for fn in fns
                     if not os.path.islink(os.path.join(dp, fn))) / (1024 * 1024)
            total_mb += sz
            print(f"  {f:<45s} {sz:>8.1f} MB")
    print(f"  {'TOTAL':<45s} {total_mb:>8.1f} MB")
    print(f"\n  Elapsed: {time.time()-t_total:.1f}s")
    print(f"\nNext: python tests/dev/qwen35_combine.py --input {args.output}")


if __name__ == "__main__":
    main()
