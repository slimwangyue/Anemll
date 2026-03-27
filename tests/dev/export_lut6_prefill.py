#!/usr/bin/env python3
"""
Export LUT6 prefill chunks for Qwen3.5-4B.

The decode chunks (ffn_LUT4_chunk{0..3}.mlpackage) are already LUT6 despite
the misleading name. This script exports the matching LUT6 prefill chunks.
"""
import gc, time, os, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

BATCH_SIZE = 256
CTX = 1024
NUM_CHUNKS = 4
LUT6_BITS = 6
PER_CHANNEL = 8

STABLE_DIR = os.path.join(_REPO_ROOT, "qwen3_5_stable_models")
HF_MODEL = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"

from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


def dir_size_mb(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def main():
    print("=" * 70)
    print("  LUT6 Prefill Export for Qwen3.5-4B")
    print(f"  LUT bits: {LUT6_BITS} | Batch: {BATCH_SIZE} | CTX: {CTX}")
    print(f"  Output: {STABLE_DIR}")
    print("=" * 70)

    print("\nLoading model weights...")
    t_load = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), f"Failed to load weights from {HF_MODEL}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t_load:.1f}s")

    for ci in range(NUM_CHUNKS):
        pf_path = os.path.join(STABLE_DIR, f"prefill_LUT6_chunk{ci}.mlpackage")
        if os.path.exists(pf_path):
            print(f"  [skip] prefill LUT6 chunk {ci} already exists ({dir_size_mb(pf_path):.0f} MB)")
            continue
        print(f"  Exporting prefill chunk {ci} (LUT6)...")
        t0 = time.time()
        conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                               num_chunks=NUM_CHUNKS, lut_bits=LUT6_BITS, per_channel=PER_CHANNEL)
        ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
        ml.save(pf_path)
        sz = dir_size_mb(pf_path)
        del ml, conv; gc.collect()
        print(f"  Saved prefill chunk {ci} ({time.time()-t0:.1f}s, {sz:.0f} MB)")

    del model; gc.collect()
    print("\nDone. All LUT6 prefill chunks exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
