#!/usr/bin/env python3
"""Export Gemma4 E4B model to CoreML .mlpackage files.

Usage:
    cd /path/to/Anemll
    python scripts_gemma4/export.py [--nosplit-lmhead] [--lut-bits 4] [--per-channel 4]
"""
import argparse
import os
import sys
import time
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(1, os.path.dirname(_SCRIPT_DIR))

from anemll.ane_converter.gemma4_converter import _apply_fp16_scaling

from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES, LUT_BITS, LM_HEAD_LUT,
    PER_CHANNEL, FFN_PER_CHANNEL, FFN_LABEL,
    DEFAULT_HF_MODEL, DEFAULT_OUTPUT,
    Gemma4ForCausalLM, Gemma4Config, Gemma4Converter,
    MODEL_DTYPE, TEST_DEVICE,
)


def load_model(model_path: str, ctx: int, fp16_scale: str = "auto") -> Gemma4ForCausalLM:
    """Load Gemma4 model for CoreML conversion."""
    os.environ['ANEMLL_ALLOW_MISSING_WEIGHTS'] = '1'
    cfg = Gemma4Config.from_json(os.path.join(model_path, "config.json"))
    cfg.context_length = ctx
    cfg.state_length = ctx

    model = Gemma4ForCausalLM(cfg, enable_coreml=True)
    model.load_pretrained_weights(model_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Apply FP16 residual stream scaling to prevent MLP overflow
    alpha = _apply_fp16_scaling(model, fp16_scale, model_path)
    if alpha:
        print(f"FP16 scaling applied: alpha={alpha}")
    else:
        print("WARNING: No FP16 scaling applied - may produce NaN in FP16!")

    return model


def export_embeddings(model, output_dir, converter):
    """Export embeddings with PLE to mlpackage."""
    print("\n" + "=" * 60)
    print("EXPORTING EMBEDDINGS + PLE")
    print("=" * 60)

    out_path = os.path.join(output_dir, "embeddings.mlpackage")
    if os.path.exists(out_path):
        print(f"  Skipping (exists): {out_path}")
        return

    t0 = time.time()
    mlmodel = converter.convert_embeddings(model)
    mlmodel.save(out_path)
    print(f"  Saved: {out_path} ({time.time() - t0:.1f}s)")


def export_lm_head(model, output_dir, converter, lut_bits=None):
    """Export LM head (Part 3) to mlpackage."""
    print("\n" + "=" * 60)
    print("EXPORTING LM HEAD")
    print("=" * 60)

    label = f"lut{lut_bits}" if lut_bits else "fp16"
    out_path = os.path.join(output_dir, f"lm_head_{label}.mlpackage")
    if os.path.exists(out_path):
        print(f"  Skipping (exists): {out_path}")
        return

    # Temporarily set LUT for lm_head
    old_lut = converter.lut_bits
    converter.lut_bits = lut_bits

    t0 = time.time()
    mlmodel = converter.convert_part_3(model)
    mlmodel.save(out_path)
    print(f"  Saved: {out_path} ({time.time() - t0:.1f}s)")

    converter.lut_bits = old_lut


def export_ffn_chunks(model, output_dir, converter, lut_bits, per_channel, num_chunks, chunk_ranges):
    """Export FFN chunks (Part 2) for decode and prefill."""
    print("\n" + "=" * 60)
    print(f"EXPORTING FFN CHUNKS ({num_chunks} chunks, {FFN_LABEL})")
    print("=" * 60)

    # Set FFN quantization
    converter.lut_bits = lut_bits
    converter.per_channel = per_channel

    for ci, (start, end) in enumerate(chunk_ranges):
        for mode in ["decode", "prefill"]:
            is_prefill = mode == "prefill"
            label = f"{mode}_{FFN_LABEL}_chunk{ci:02d}"
            out_path = os.path.join(output_dir, f"{label}.mlpackage")

            if os.path.exists(out_path):
                print(f"  Skipping (exists): {out_path}")
                continue

            print(f"\n  Chunk {ci}/{num_chunks}: layers {start}-{end-1} ({mode})")
            t0 = time.time()

            if is_prefill:
                mlmodel = converter.convert_part_2_prefill(
                    model, chunk_idx=ci, total_chunks=num_chunks
                )
            else:
                mlmodel = converter.convert_part_2(
                    model, chunk_idx=ci, total_chunks=num_chunks, force_rotation=False
                )

            mlmodel.save(out_path)
            print(f"  Saved: {out_path} ({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Export Gemma4 E4B to CoreML")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL, help="HF model path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--ctx", type=int, default=CTX, help="Context length")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Prefill batch size")
    parser.add_argument("--lut-bits", type=int, default=LUT_BITS, help="FFN LUT bits")
    parser.add_argument("--lm-head-lut", type=int, default=LM_HEAD_LUT, help="LM head LUT bits")
    parser.add_argument("--per-channel", type=int, default=FFN_PER_CHANNEL, help="FFN per-channel")
    parser.add_argument("--num-chunks", type=int, default=NUM_CHUNKS, help="Number of FFN chunks")
    parser.add_argument("--nosplit-lmhead", action="store_true", help="Export non-split LM head")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Context: {args.ctx}, Batch: {args.batch_size}")
    print(f"FFN: LUT{args.lut_bits} pc={args.per_channel}, LM head: LUT{args.lm_head_lut}")
    print(f"Chunks: {args.num_chunks}")

    model = load_model(args.model, args.ctx)
    converter = Gemma4Converter(
        model,
        context_length=args.ctx,
        batch_size=args.batch_size,
        lut_bits=None,  # Set per-component
        per_channel=PER_CHANNEL,
        num_chunks=args.num_chunks,
    )

    # 1. Embeddings + PLE
    export_embeddings(model, args.output, converter)

    # 2. LM Head
    export_lm_head(model, args.output, converter, lut_bits=args.lm_head_lut)

    # 3. FFN chunks (decode + prefill)
    export_ffn_chunks(
        model, args.output, converter,
        lut_bits=args.lut_bits,
        per_channel=args.per_channel,
        num_chunks=args.num_chunks,
        chunk_ranges=CHUNK_RANGES[:args.num_chunks],
    )

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
