#!/usr/bin/env python3
"""Re-export fixed-shape embeddings and combine with lm_head into a deduped multifunction model.

Produces:
  embed_single.mlpackage    (seq_len=1)   → function "embedding_decode"
  embed_prefill.mlpackage   (seq_len=512) → function "embedding_prefill"
  lm_head_nosplit.mlpackage (existing)    → function "lmhead"
  → embed_lmhead_combined.mlpackage (3-function, deduped)

Usage:
    python scripts_qwen3_5/reexport_embed_combine.py \
        --model /path/to/Qwen3.5-4B \
        --output /Users/yw68/Anemll/qwen3_5_stable_lut4ffn_lut6em_test
"""
import gc, time, argparse, os, sys, shutil
import numpy as np
import torch
import coremltools as ct

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS,
    PER_CHANNEL, DEFAULT_HF_MODEL,
)
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


def dir_size_mb(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def export_fixed_embeddings(model, out_dir, skip_existing=False):
    """Export embed_single (seq_len=1) and embed_prefill (seq_len=BATCH_SIZE)."""
    for seq_len, suffix in [(1, "embed_single"), (BATCH_SIZE, "embed_prefill")]:
        fpath = os.path.join(out_dir, f"{suffix}.mlpackage")
        if skip_existing and os.path.exists(fpath):
            print(f"  [skip] {suffix}")
            continue
        print(f"  Exporting {suffix} (seq_len={seq_len}, LUT{LUT_BITS} gs={PER_CHANNEL})...")
        t0 = time.time()
        conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=PER_CHANNEL,
                                compute_precision="float16")
        ml = conv.convert_part_1(model, seq_len=seq_len)
        ml.save(fpath)
        del ml, conv; gc.collect()
        print(f"  Saved {suffix} ({time.time()-t0:.1f}s) — {dir_size_mb(fpath):.1f} MB")


def combine_embed_lmhead(input_dir, skip_existing=False):
    """Combine embed_single + embed_prefill + lm_head_nosplit → embed_lmhead_combined.mlpackage."""
    from anemll.utils.dedup_weights import prepare_dedup_sources, dedup_cross_model_blobs

    embed_single_path = os.path.join(input_dir, "embed_single.mlpackage")
    embed_prefill_path = os.path.join(input_dir, "embed_prefill.mlpackage")
    lmhead_path = os.path.join(input_dir, "lm_head_nosplit.mlpackage")
    combined_path = os.path.join(input_dir, "embed_lmhead_combined.mlpackage")

    for p, label in [(embed_single_path, "embed_single"), (embed_prefill_path, "embed_prefill"),
                     (lmhead_path, "lm_head_nosplit")]:
        if not os.path.exists(p):
            print(f"  ERROR: {label} not found at {p}")
            return None

    if skip_existing and os.path.exists(combined_path):
        sz = dir_size_mb(combined_path)
        print(f"  [skip] embed_lmhead_combined ({sz:.1f} MB)")
        return combined_path

    # Remove old combined if it exists
    if os.path.exists(combined_path):
        shutil.rmtree(combined_path)

    sources = [
        (embed_single_path,  "main", "embedding_decode"),
        (embed_prefill_path, "main", "embedding_prefill"),
        (lmhead_path,        "main", "lmhead"),
    ]

    print(f"  Combining 3 functions: embedding_decode, embedding_prefill, lmhead")
    t0 = time.time()

    try:
        with prepare_dedup_sources(sources, verbose=True, preflight=False) as deduped:
            desc = ct.utils.MultiFunctionDescriptor()
            for path, src_fn, tgt_fn in deduped:
                desc.add_function(path, src_fn, tgt_fn)
            desc.default_function_name = "embedding_decode"
            ct.utils.save_multifunction(desc, combined_path)

        print("  Running cross-model blob dedup...")
        fn_names = [tgt for _, _, tgt in sources]
        total_saved = 0
        for i in range(len(fn_names)):
            for j in range(i + 1, len(fn_names)):
                saved = dedup_cross_model_blobs(combined_path, fn_names[i], fn_names[j], verbose=True)
                total_saved += saved
        if total_saved > 0:
            print(f"  Cross-model dedup saved {total_saved / 1e6:.1f} MB total")
        else:
            print("  Cross-model dedup: no savings (blobs may already be shared or differ)")
    except Exception as e:
        print(f"  Dedup failed ({e}), trying without dedup...")
        if os.path.exists(combined_path):
            shutil.rmtree(combined_path)
        desc = ct.utils.MultiFunctionDescriptor()
        for path, src_fn, tgt_fn in sources:
            desc.add_function(path, src_fn, tgt_fn)
        desc.default_function_name = "embedding_decode"
        ct.utils.save_multifunction(desc, combined_path)

    sz = dir_size_mb(combined_path)
    print(f"  Saved embed_lmhead_combined ({time.time()-t0:.1f}s) — {sz:.1f} MB")
    return combined_path


def main():
    parser = argparse.ArgumentParser(description="Re-export embeddings and combine with lm_head")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL,
                        help="Path to HuggingFace Qwen3.5-4B model directory")
    parser.add_argument("--output", default="/Users/yw68/Anemll/qwen3_5_stable_lut4ffn_lut6em_test",
                        help="Output directory (must contain lm_head_nosplit.mlpackage)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip already-exported mlpackages")
    args = parser.parse_args()

    print("=" * 70)
    print("  Re-export Embeddings + Combine with LM Head")
    print(f"  Embed: LUT{LUT_BITS} gs={PER_CHANNEL} | Batch: {BATCH_SIZE}")
    print(f"  Functions: embedding_decode (1), embedding_prefill ({BATCH_SIZE}), lmhead")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    # Step 1: Load model
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

    # Step 2: Export fixed-shape embeddings
    print("\n[1/2] Exporting fixed-shape embeddings...")
    export_fixed_embeddings(model, args.output, skip_existing=args.skip_existing)
    del model; gc.collect()

    # Step 3: Combine
    print("\n[2/2] Combining embedding_decode + embedding_prefill + lmhead...")
    result = combine_embed_lmhead(args.output, skip_existing=args.skip_existing)

    if result:
        print(f"\nDone! Combined model: {result}")
        print(f"  Size: {dir_size_mb(result):.1f} MB")
    else:
        print("\nFailed!")
        return 1


if __name__ == "__main__":
    main()
