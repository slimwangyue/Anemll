#!/usr/bin/env python3
"""Export LUT4 lm_head variants with different group_sizes.

Runs on Linux (no inference needed — just export + palettize).
Results (.mlpackage) are fetched back to Mac for accuracy testing.

Usage:
    python export_lmhead_lut4_variants.py --group-sizes 1 2 4 8
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Anemll"))
import gc, time, argparse

# Suppress coremltools proxy warnings on Linux
import warnings
warnings.filterwarnings("ignore", "Failed to load")

import torch
import coremltools as ct
import coremltools.optimize as cto
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

MODEL_PATH = os.path.expanduser("~/local_llm/models/Qwen__Qwen3.5-4B")
CTX = 256
NUM_CHUNKS = 4


def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def export_lmhead(lut_bits, group_size, out_dir):
    tag = f"lm_head_LUT{lut_bits}_gs{group_size}"
    path = os.path.join(out_dir, f"{tag}.mlpackage")
    if os.path.exists(path):
        sz = dir_size_mb(path)
        print(f"  {tag} already exists ({sz:.1f} MB), skipping.")
        return path

    print(f"\n{'='*60}")
    print(f"  Exporting {tag} (LUT{lut_bits}, group_size={group_size})")
    print(f"{'='*60}")
    t0 = time.time()

    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(MODEL_PATH), "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"  Model loaded ({time.time()-t0:.1f}s)")
    t1 = time.time()

    conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                           num_chunks=NUM_CHUNKS, lut_bits=lut_bits,
                           per_channel=group_size)
    ml = conv.convert_part_3(model, argmax_in_model=False)

    print(f"  Conversion + palettization: {time.time()-t1:.1f}s")

    ml.save(path)
    sz = dir_size_mb(path)
    total_time = time.time() - t0
    print(f"  Saved: {path} ({sz:.1f} MB)")
    print(f"  Total time: {total_time:.1f}s")

    del ml, conv, model
    gc.collect()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--out-dir", default=os.path.expanduser("~/lmhead_lut4_sweep"))
    parser.add_argument("--lut-bits", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"LM Head LUT{args.lut_bits} Group-Size Sweep")
    print(f"Group sizes: {args.group_sizes}")
    print(f"Output: {args.out_dir}")
    print(f"Model: {MODEL_PATH}")

    results = []
    for gs in args.group_sizes:
        try:
            path = export_lmhead(args.lut_bits, gs, args.out_dir)
            sz = dir_size_mb(path)
            results.append((gs, sz, path))
        except Exception as e:
            print(f"\n  ERROR exporting gs={gs}: {e}")
            import traceback
            traceback.print_exc()
            results.append((gs, 0, f"FAILED: {e}"))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Group Size':>12} {'Size (MB)':>10} {'Path'}")
    print(f"  {'-'*60}")
    for gs, sz, path in results:
        print(f"  {gs:>12} {sz:>10.1f} {path}")
    print(f"\nDone. Use scp to fetch .mlpackage dirs to Mac for validation.")


if __name__ == "__main__":
    main()
