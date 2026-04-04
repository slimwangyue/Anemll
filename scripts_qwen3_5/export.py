#!/usr/bin/env python3
"""Qwen3.5-4B Milestone 2.1 — Step 1: Export all CoreML model components.

Exports:
  - embeddings (LUT6 gs=8)          → embeddings.mlpackage
  - lm_head (LUT6 gs=8)             → lm_head.mlpackage
  - 4 FFN decode chunks (LUT6 gs=4) → ffn_LUT6_chunk{0..3}.mlpackage
  - 4 FFN prefill chunks (LUT6 gs=4)→ prefill_LUT6_chunk{0..3}.mlpackage

Dynamic KV cache slicing via RangeDim — one prefill model per chunk.

Usage:
    python scripts_qwen3_5/export.py --model /path/to/Qwen3.5-4B --output /path/to/output
    python scripts_qwen3_5/export.py --skip-existing
"""
import gc, time, argparse, os, sys, shutil, glob

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)  # must be first for config.py

from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, LM_HEAD_LUT,
    PER_CHANNEL, FFN_PER_CHANNEL, FFN_LABEL, DEFAULT_HF_MODEL, DEFAULT_OUTPUT,
)
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


def export_embeddings(model, out_dir, skip_existing):
    path = os.path.join(out_dir, "embeddings.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] embeddings")
        return
    print(f"  Exporting embeddings (LUT{LUT_BITS} gs={PER_CHANNEL})...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=PER_CHANNEL)
    ml = conv.convert_part_1(model)
    ml.save(path)
    del ml, conv; gc.collect()
    print(f"  Saved embeddings ({time.time()-t0:.1f}s)")


def export_lm_head(model, out_dir, skip_existing):
    path = os.path.join(out_dir, "lm_head_logits.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] lm_head_logits")
        return
    print(f"  Exporting lm_head_logits 16-way split (LUT{LM_HEAD_LUT})...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LM_HEAD_LUT, per_channel=PER_CHANNEL)
    ml = conv.convert_part_3(model, argmax_in_model=False)
    ml.save(path)
    del ml, conv; gc.collect()
    print(f"  Saved lm_head_logits ({time.time()-t0:.1f}s)")


def export_ffn_chunks(model, out_dir, skip_existing, only_chunk=None, static_prefill=False):
    label = FFN_LABEL
    chunk_indices = [only_chunk] if only_chunk is not None else list(range(NUM_CHUNKS))
    for ci in chunk_indices:
        # Decode chunk
        dec_path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        if skip_existing and os.path.exists(dec_path):
            print(f"  [skip] decode chunk {ci}")
        else:
            print(f"  Exporting decode chunk {ci} ({label} gs={FFN_PER_CHANNEL})...")
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                   num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=FFN_PER_CHANNEL)
            ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
            ml.save(dec_path)
            del ml, conv; gc.collect()
            print(f"  Saved decode chunk {ci} ({time.time()-t0:.1f}s)")

        # Prefill chunk
        if static_prefill:
            pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}_bs{BATCH_SIZE}.mlpackage")
            pf_desc = f"prefill chunk {ci} static bs{BATCH_SIZE}"
        else:
            pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage")
            pf_desc = f"prefill chunk {ci}"
        if skip_existing and os.path.exists(pf_path):
            print(f"  [skip] {pf_desc}")
        else:
            print(f"  Exporting {pf_desc} ({label} gs={FFN_PER_CHANNEL})...")
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                   num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=FFN_PER_CHANNEL)
            if static_prefill:
                ml = conv.convert_part_2_prefill_exact(
                    model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                    exact_seq_len=BATCH_SIZE)
            else:
                ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
            ml.save(pf_path)
            del ml, conv; gc.collect()
            print(f"  Saved {pf_desc} ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5-4B for ANE (Milestone 2.1)")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL,
                        help="Path to HuggingFace Qwen3.5-4B model directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for exported .mlpackage files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if .mlpackage already exists")
    parser.add_argument("--only-chunk", type=int, default=None,
                        help="Export only the specified chunk index")
    parser.add_argument("--static-prefill", action="store_true",
                        help="Use static-shape prefill (convert_part_2_prefill_exact) with valid_len")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    prefill_mode = "static" if args.static_prefill else "dynamic"
    print("=" * 70)
    print("  Qwen3.5-4B ANE Export — Milestone 2.1 (LUT6 gs=4 FFN)")
    print(f"  Embed: LUT{LUT_BITS} gs={PER_CHANNEL} | LM Head: LUT{LM_HEAD_LUT} gs={PER_CHANNEL} | FFN: {FFN_LABEL} gs={FFN_PER_CHANNEL} × {NUM_CHUNKS} chunks")
    print(f"  Batch: {BATCH_SIZE} | CTX: {CTX} | Prefill: {prefill_mode}")
    if args.only_chunk is not None:
        print(f"  Only chunk: {args.only_chunk}")
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
    print("\n[1/3] FFN Chunks (decode + prefill)")
    export_ffn_chunks(model, args.output, args.skip_existing,
                      only_chunk=args.only_chunk, static_prefill=args.static_prefill)
    print("\n[2/3] Embeddings")
    export_embeddings(model, args.output, args.skip_existing)
    print("\n[3/3] LM Head")
    export_lm_head(model, args.output, args.skip_existing)
    
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
    # Copy tokenizer files so the output dir is self-contained
    tok_patterns = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "special_tokens_map.json"]
    copied = []
    for pat in tok_patterns:
        for src in glob.glob(os.path.join(args.model, pat)):
            dst = os.path.join(args.output, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(os.path.basename(src))
    if copied:
        print(f"  Copied tokenizer files: {', '.join(copied)}")

    print(f"\n  Elapsed: {time.time()-t_total:.1f}s")
    print(f"\nNext: python scripts_qwen3_5/combine.py --input {args.output}")


if __name__ == "__main__":
    main()
