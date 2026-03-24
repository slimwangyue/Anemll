#!/usr/bin/env python3
"""Qwen3.5-4B — Exploratory export for alternative batch/ctx configurations.

Exports models to a SEPARATE directory from stable. Never touches stable models.

Usage:
    # Export a specific config:
    python scripts_qwen3_5/explore/explore_export.py --config batch512_ctx1024

    # Export all configs:
    python scripts_qwen3_5/explore/explore_export.py --config all

    # List available configs:
    python scripts_qwen3_5/explore/explore_export.py --list

    # Skip parts that already exist:
    python scripts_qwen3_5/explore/explore_export.py --config batch256_ctx2048 --skip-existing
"""
import gc, time, argparse, os, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts_qwen3_5.explore.explore_config import CONFIGS, get_config, list_configs, HF_MODEL
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


def export_config(cfg, hf_model, skip_existing=False, parts=None):
    """Export all model parts for a given configuration."""
    name = cfg["name"]
    out = cfg["output_dir"]
    batch = cfg["BATCH_SIZE"]
    ctx = cfg["CTX"]
    chunks = cfg["NUM_CHUNKS"]
    lut = cfg["LUT_BITS"]
    lm_lut = cfg["LM_HEAD_LUT"]
    pc = cfg["PER_CHANNEL"]

    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print(f"  EXPLORATORY EXPORT: {name}")
    print(f"  Batch={batch}  CTX={ctx}  Chunks={chunks}  LUT={lut}  LM_HEAD_LUT={lm_lut}")
    print(f"  Output: {out}")
    print(f"  HF Model: {hf_model}")
    print("=" * 70)

    # Load model
    print("\nLoading model weights...")
    t_load = time.time()
    model_cfg = Qwen35Config.from_json(os.path.join(hf_model, "config.json"))
    model_cfg.context_length = ctx
    model_cfg.state_length = ctx
    model = Qwen35ForCausalLM(model_cfg)
    assert model.load_pretrained_weights(hf_model), f"Failed to load weights from {hf_model}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t_load:.1f}s")

    do_all = parts is None
    t_total = time.time()

    # Part 1: Embeddings
    if do_all or "embed" in parts:
        path = os.path.join(out, "embeddings.mlpackage")
        if skip_existing and os.path.exists(path):
            print("\n  [skip] embeddings")
        else:
            print("\n  [1] Exporting embeddings...")
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=ctx, batch_size=batch,
                                   num_chunks=chunks, lut_bits=lut, per_channel=pc)
            ml = conv.convert_part_1(model)
            ml.save(path)
            del ml, conv; gc.collect()
            print(f"      Saved embeddings ({time.time()-t0:.1f}s)")

    # Part 3: LM Head (shared across configs — same for all ctx/batch)
    if do_all or "lmhead" in parts:
        path = os.path.join(out, "lm_head.mlpackage")
        if skip_existing and os.path.exists(path):
            print("\n  [skip] lm_head")
        else:
            print(f"\n  [2] Exporting lm_head (LUT{lm_lut})...")
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=ctx, batch_size=batch,
                                   num_chunks=chunks, lut_bits=lm_lut, per_channel=pc)
            ml = conv.convert_part_3(model, argmax_in_model=False)
            ml.save(path)
            del ml, conv; gc.collect()
            print(f"      Saved lm_head ({time.time()-t0:.1f}s)")

    # Part 2: FFN decode + prefill chunks
    if do_all or "ffn" in parts:
        label = f"LUT{lut}"
        for ci in range(chunks):
            # Decode
            dec_path = os.path.join(out, f"ffn_{label}_chunk{ci}.mlpackage")
            if skip_existing and os.path.exists(dec_path):
                print(f"\n  [skip] decode chunk {ci}")
            else:
                print(f"\n  [3.{ci}a] Exporting decode chunk {ci} ({label})...")
                t0 = time.time()
                conv = Qwen35Converter(model, context_length=ctx, batch_size=batch,
                                       num_chunks=chunks, lut_bits=lut, per_channel=pc)
                ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=chunks)
                ml.save(dec_path)
                del ml, conv; gc.collect()
                print(f"      Saved decode chunk {ci} ({time.time()-t0:.1f}s)")

            # Prefill
            pf_path = os.path.join(out, f"prefill_{label}_chunk{ci}.mlpackage")
            if skip_existing and os.path.exists(pf_path):
                print(f"\n  [skip] prefill chunk {ci}")
            else:
                print(f"\n  [3.{ci}b] Exporting prefill chunk {ci} ({label})...")
                t0 = time.time()
                conv = Qwen35Converter(model, context_length=ctx, batch_size=batch,
                                       num_chunks=chunks, lut_bits=lut, per_channel=pc)
                ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=chunks)
                ml.save(pf_path)
                del ml, conv; gc.collect()
                print(f"      Saved prefill chunk {ci} ({time.time()-t0:.1f}s)")

    del model; gc.collect()

    # Summary
    total_mb = 0
    print(f"\n{'='*70}")
    print(f"  EXPORT SUMMARY: {name}")
    print(f"{'='*70}")
    for f in sorted(os.listdir(out)):
        full = os.path.join(out, f)
        if os.path.isdir(full):
            sz = sum(os.path.getsize(os.path.join(dp, fn))
                     for dp, _, fns in os.walk(full) for fn in fns) / (1024**2)
            total_mb += sz
            print(f"  {f:<45} {sz:>8.1f} MB")
    print(f"  {'TOTAL':<45} {total_mb:>8.1f} MB")
    print(f"  Export time: {time.time()-t_total:.1f}s")

    return out


def main():
    parser = argparse.ArgumentParser(description="Exploratory export for Qwen3.5-4B")
    parser.add_argument("--config", type=str, required=False,
                        help="Config name (e.g. batch512_ctx1024) or 'all'")
    parser.add_argument("--model", type=str, default=HF_MODEL,
                        help="HuggingFace model directory")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--list", action="store_true", help="List available configs")
    parser.add_argument("--parts", nargs="+", choices=["embed", "lmhead", "ffn"],
                        help="Only export specific parts")
    args = parser.parse_args()

    if args.list:
        list_configs()
        return

    if not args.config:
        print("ERROR: --config required. Use --list to see options.")
        sys.exit(1)

    if args.config == "all":
        configs = list(CONFIGS.values())
    else:
        configs = [get_config(args.config)]

    for cfg in configs:
        export_config(cfg, args.model, args.skip_existing, args.parts)


if __name__ == "__main__":
    main()
