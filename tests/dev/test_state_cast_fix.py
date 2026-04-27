#!/usr/bin/env python3
"""Test the _ensure_state_slice_update_casts fix on chunk 3 (broken chunk).

Exports chunk 3 prefill with V4 precision, applies the fix, and verifies
that casts are injected between read_state and slice_update.
"""
import warnings, sys, os
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts_qwen3_5")
# scripts_qwen3_5/config.py must come first to shadow root config.py
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPTS_DIR)
os.chdir(REPO_ROOT)

import torch
torch.set_grad_enabled(False)

from config import CTX, BATCH_SIZE, NUM_CHUNKS, CHUNK_RANGES, FFN_PER_CHANNEL
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from export import _ensure_state_slice_update_casts, get_v4_compute_precision


def check_slice_update_casts(ml, label):
    """Check if slice_update ops on state have casts on their x input."""
    spec = ml.get_spec()
    fn = spec.mlProgram.functions["main"]
    block = list(fn.block_specializations.values())[0]

    results = {}
    for i, op in enumerate(block.operations):
        if op.type != "slice_update":
            continue
        out_name = op.outputs[0].name
        if "cache" not in out_name.lower():
            continue
        x_name = op.inputs["x"].arguments[0].name
        # Find producer of x
        x_src_type = "?"
        for j, op2 in enumerate(block.operations):
            for o in op2.outputs:
                if o.name == x_name:
                    x_src_type = op2.type
                    break

        is_k = "k_cache" in out_name.lower()
        cache_type = "k_cache" if is_k else "v_cache"
        has_cast = x_src_type == "cast"
        results[cache_type] = {"x": x_name, "src": x_src_type, "has_cast": has_cast}
        status = "✓ cast" if has_cast else "✗ DIRECT read_state"
        print(f"  [{label}] {cache_type}: x={x_name} (from {x_src_type}) → {status}")

    return results


# Load model
HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
print(f"Loading model from {HF_MODEL}...")
cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
model.load_pretrained_weights(HF_MODEL)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"  Loaded: {cfg.num_hidden_layers} layers, CTX={CTX}, BS={BATCH_SIZE}")

# Export chunk 3 prefill with V4
ci = 3
sl, el = CHUNK_RANGES[ci]
v4_cp, fp16_ls, fp32_ls = get_v4_compute_precision(model, ci)
print(f"\nChunk {ci}: F-layers={fp32_ls}, L-layers={fp16_ls}")

conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                       num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=FFN_PER_CHANNEL,
                       compute_precision="float32")
conv.compute_precision = v4_cp

print("Converting prefill chunk 3...")
ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                                 override_start_layer=sl, override_end_layer=el)
print("Done converting\n")

# Check BEFORE fix
print("=== BEFORE _ensure_state_slice_update_casts ===")
before = check_slice_update_casts(ml, "BEFORE")

# Apply fix
n = _ensure_state_slice_update_casts(ml)
print(f"\n>>> Injected {n} cast(s)\n")

# Check AFTER fix
print("=== AFTER _ensure_state_slice_update_casts ===")
after = check_slice_update_casts(ml, "AFTER")

# Validate
print("\n=== VALIDATION ===")
all_ok = True
for cache_type in ["k_cache", "v_cache"]:
    if cache_type in after:
        if after[cache_type]["has_cast"]:
            print(f"  {cache_type}: PASS (cast present)")
        else:
            print(f"  {cache_type}: FAIL (no cast)")
            all_ok = False

if all_ok:
    print("\n✓ All state slice_update ops have casts — fix verified!")
else:
    print("\n✗ Some state slice_update ops still missing casts")
    sys.exit(1)
