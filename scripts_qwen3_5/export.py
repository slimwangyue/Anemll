#!/usr/bin/env python3
"""Qwen3.5-4B Milestone 1.2 — Step 1: Export all CoreML model components.

Exports:
  - embeddings (LUT4)             → embeddings.mlpackage
  - lm_head (LUT6)                → lm_head.mlpackage
  - 4 FFN decode chunks (LUT4)    → ffn_LUT4_chunk{0..3}.mlpackage
  - 4 FFN prefill chunks (LUT4)   → prefill_LUT4_chunk{0..3}.mlpackage

Dynamic KV cache slicing via RangeDim — one prefill model per chunk.

Usage:
    python scripts_qwen3_5/export.py --model /path/to/Qwen3.5-4B --output /path/to/output
    python scripts_qwen3_5/export.py --skip-existing
"""
import gc, time, argparse, os
from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, LM_HEAD_LUT,
    PER_CHANNEL, DEFAULT_HF_MODEL, DEFAULT_OUTPUT,
)
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


def export_embeddings(model, out_dir, skip_existing):
    path = os.path.join(out_dir, "embeddings.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] embeddings")
        return
    print("  Exporting embeddings (LUT4)...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=PER_CHANNEL)
    ml = conv.convert_part_1(model)
    ml.save(path)
    del ml, conv; gc.collect()
    print(f"  Saved embeddings ({time.time()-t0:.1f}s)")


def export_lm_head(model, out_dir, skip_existing):
    path = os.path.join(out_dir, "lm_head.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] lm_head")
        return
    print(f"  Exporting lm_head (LUT{LM_HEAD_LUT})...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LM_HEAD_LUT, per_channel=PER_CHANNEL)
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
                                   num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=PER_CHANNEL)
            ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
            ml.save(dec_path)
            del ml, conv; gc.collect()
            print(f"  Saved decode chunk {ci} ({time.time()-t0:.1f}s)")

        # Single prefill chunk (dynamic position via RangeDim)
        pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage")
        if skip_existing and os.path.exists(pf_path):
            print(f"  [skip] prefill chunk {ci}")
        else:
            print(f"  Exporting prefill chunk {ci} ({label})...")
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                   num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=PER_CHANNEL)
            ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
            ml.save(pf_path)
            del ml, conv; gc.collect()
            print(f"  Saved prefill chunk {ci} ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5-4B for ANE (Milestone 1.1)")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL,
                        help="Path to HuggingFace Qwen3.5-4B model directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for exported .mlpackage files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if .mlpackage already exists")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("  Qwen3.5-4B ANE Export — Milestone 1.2 (Dynamic KV Slicing)")
    print(f"  Embed: LUT4 | LM Head: LUT{LM_HEAD_LUT} | FFN: LUT4 × {NUM_CHUNKS} chunks")
    print(f"  Batch: {BATCH_SIZE} | CTX: {CTX}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print("=" * 70)

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

    t_total = time.time()
    print("\n[1/3] Embeddings")
    export_embeddings(model, args.output, args.skip_existing)
    print("\n[2/3] LM Head")
    export_lm_head(model, args.output, args.skip_existing)
    print("\n[3/3] FFN Chunks (decode + prefill)")
    export_ffn_chunks(model, args.output, args.skip_existing)
    del model; gc.collect()

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
            print(f"  {f:<50s} {sz:>8.1f} MB")
    print(f"  {'TOTAL':<50s} {total_mb:>8.1f} MB")
    print(f"\n  Elapsed: {time.time()-t_total:.1f}s")
    print(f"\nNext: python scripts_qwen3_5/combine.py --input {args.output}")


if __name__ == "__main__":
    main()
