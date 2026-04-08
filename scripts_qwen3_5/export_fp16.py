#!/usr/bin/env python3
"""Export Qwen3.5-4B as FP16 CoreML models (NO LUT quantization).

Exports decode-only models for root-cause comparison:
  - embeddings (FP16)
  - lm_head_logits (FP16)
  - 6 FFN decode chunks (FP16) — NO prefill (sequential mode only)

Usage:
    python scripts_qwen3_5/export_fp16.py
    python scripts_qwen3_5/export_fp16.py --output /tmp/qwen35_fp16_models
"""
import gc, time, argparse, os, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from config import BATCH_SIZE, CTX, NUM_CHUNKS, PER_CHANNEL, FFN_PER_CHANNEL, DEFAULT_HF_MODEL
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

DEFAULT_FP16_OUTPUT = os.path.join(_REPO_ROOT, "qwen3_5_fp16_models")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5-4B FP16 (no quantization)")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--output", default=DEFAULT_FP16_OUTPUT)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("  Qwen3.5-4B FP16 Export (NO LUT quantization)")
    print(f"  Chunks: {NUM_CHUNKS} | CTX: {CTX} | BATCH: {BATCH_SIZE}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    print("\nLoading model weights...")
    t_load = time.time()
    cfg = Qwen35Config.from_json(os.path.join(args.model, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(args.model), f"Failed to load from {args.model}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t_load:.1f}s")

    t_total = time.time()

    # 1. FFN decode chunks (FP16 — lut_bits=None)
    print(f"\n[1/3] FFN Decode Chunks (FP16, no LUT)")
    for ci in range(NUM_CHUNKS):
        path = os.path.join(args.output, f"ffn_FP16_chunk{ci}.mlpackage")
        if args.skip_existing and os.path.exists(path):
            print(f"  [skip] chunk {ci}")
            continue
        print(f"  Exporting decode chunk {ci} (FP16)...", end="", flush=True)
        t0 = time.time()
        conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                               num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=FFN_PER_CHANNEL)
        ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
        ml.save(path)
        del ml, conv; gc.collect()
        print(f" {time.time()-t0:.1f}s")

    # 2. Embeddings (FP16)
    print(f"\n[2/3] Embeddings (FP16)")
    path = os.path.join(args.output, "embeddings.mlpackage")
    if args.skip_existing and os.path.exists(path):
        print(f"  [skip] embeddings")
    else:
        print(f"  Exporting embeddings (FP16)...", end="", flush=True)
        t0 = time.time()
        conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                               num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=PER_CHANNEL)
        ml = conv.convert_part_1(model)
        ml.save(path)
        del ml, conv; gc.collect()
        print(f" {time.time()-t0:.1f}s")

    # 3. LM Head (FP16)
    print(f"\n[3/3] LM Head (FP16)")
    path = os.path.join(args.output, "lm_head_logits.mlpackage")
    if args.skip_existing and os.path.exists(path):
        print(f"  [skip] lm_head_logits")
    else:
        print(f"  Exporting lm_head_logits (FP16)...", end="", flush=True)
        t0 = time.time()
        conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                               num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=PER_CHANNEL)
        ml = conv.convert_part_3(model, argmax_in_model=False)
        ml.save(path)
        del ml, conv; gc.collect()
        print(f" {time.time()-t0:.1f}s")

    del model; gc.collect()

    total_mb = 0
    print(f"\n{'='*70}")
    print("  FP16 EXPORT SUMMARY")
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


if __name__ == "__main__":
    main()
