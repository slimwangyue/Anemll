#!/usr/bin/env python3
"""Test that shift-left-append KV cache pattern exports and runs on ANE.

Exports a single decode chunk (8 layers), runs one token through CoreML on ANE,
and compares vs PyTorch reference.

Usage:
    python tests/dev/_test_decode_ane.py
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, shutil
import numpy as np
import torch
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_decode_ane_test"
CTX = 256

os.makedirs(OUT_DIR, exist_ok=True)

def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


print("=" * 70)
print("  Decode ANE Test (shift-left-append KV cache)")
print("=" * 70)

# ── 1. Load model ──
print("Loading model...")
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
model.eval()
for p in model.parameters():
    p.requires_grad = False

conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)

# ── 2. Export chunk 1 (layers 0-7) as decode model ──
print("\nExporting decode chunk 1 (layers 0-7)...")
converter = Qwen35Converter(
    model=model,
    context_length=CTX,
    batch_size=1,
    lut_bits=None,  # fp16 for testing
    num_chunks=4,
)

try:
    mlmodel = converter.convert_part_2(model, chunk_idx=0, total_chunks=4)
    pkg_path = os.path.join(OUT_DIR, "qwen35_FFN_chunk_01of04.mlpackage")
    if os.path.exists(pkg_path):
        shutil.rmtree(pkg_path)
    mlmodel.save(pkg_path)
    print("  Saved:", pkg_path)
    del mlmodel
    gc.collect()
except Exception as e:
    print("  EXPORT FAILED:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── 3. Test on ANE ──
print("\nLoading CoreML model on ANE...")
ane_ok = False
try:
    cml = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print("  Model loaded on ANE successfully!")
    ane_ok = True
except Exception as e:
    print("  ANE LOAD FAILED:", e)
    print("  Trying CPU_AND_GPU...")
    cml = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    print("  Loaded on CPU_AND_GPU (fallback)")

state = cml.make_state()

# ── 4. Run one token ──
print("\nRunning one decode token...")
hidden = np.random.randn(1, 1, cfg.hidden_size).astype(np.float16) * 0.01
pos_ids = np.array([0], dtype=np.int32)
# For shift-left-append: mask allows the LAST pos+1 positions
# At pos=0, allow only the last 1 position
mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
mask[:, :, :, -1:] = 0  # Allow only last position (where new KV was appended)
cur_pos = np.array([0], dtype=np.int32)

predict_ok = False
try:
    out = cml.predict({
        "hidden_states": hidden,
        "position_ids": pos_ids,
        "causal_mask": mask,
        "current_pos": cur_pos,
    }, state=state)
    cml_out = out["output_hidden_states"]
    print("  SUCCESS! Output shape:", cml_out.shape)
    print("  Output range: min=%.4f max=%.4f mean=%.6f" % (
        cml_out.min(), cml_out.max(), cml_out.mean()))
    predict_ok = True
except RuntimeError as e:
    if "ANEProgram" in str(e):
        print("  ANE PREDICT FAILED:", str(e)[:200])
        print("\n  Retrying with CPU_AND_GPU...")
        del cml, state; gc.collect()
        cml = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        state = cml.make_state()
        out = cml.predict({
            "hidden_states": hidden,
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": cur_pos,
        }, state=state)
        cml_out = out["output_hidden_states"]
        print("  CPU_AND_GPU output shape:", cml_out.shape)
        predict_ok = True
        ane_ok = False
    else:
        raise

# ── 5. Run multiple tokens to verify state updates work ──
if predict_ok:
    print("\nRunning 5 sequential tokens to verify state accumulation...")
    for step in range(5):
        hidden_step = np.random.randn(1, 1, cfg.hidden_size).astype(np.float16) * 0.01
        pos_ids_step = np.array([step], dtype=np.int32)
        # For shift-left-append: mask the last (step+1) positions
        mask_step = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        n_active = min(step + 1, CTX)
        mask_step[:, :, :, -n_active:] = 0
        cur_pos_step = np.array([step], dtype=np.int32)

        try:
            out = cml.predict({
                "hidden_states": hidden_step,
                "position_ids": pos_ids_step,
                "causal_mask": mask_step,
                "current_pos": cur_pos_step,
            }, state=state)
            step_out = out["output_hidden_states"]
            print("  Step %d: OK, shape=%s, range=[%.4f, %.4f]" % (
                step, step_out.shape, step_out.min(), step_out.max()))
        except RuntimeError as e:
            print("  Step %d: FAILED: %s" % (step, str(e)[:150]))
            break

# ── 6. PyTorch reference comparison ──
print("\nComparing single token: PyTorch vs CoreML...")
# Re-run step 0 with fresh state
del cml, state; gc.collect()
compute = ct.ComputeUnit.CPU_AND_NE if ane_ok else ct.ComputeUnit.CPU_AND_GPU
cml = ct.models.MLModel(pkg_path, compute_units=compute)
state = cml.make_state()

# Fixed input for reproducible comparison
torch.manual_seed(42)
np.random.seed(42)
test_hidden = torch.randn(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE) * 0.01
test_pos = torch.tensor([0], dtype=torch.int32)
test_mask = torch.full((1, 1, 1, CTX), float("-inf"), dtype=MODEL_DTYPE)
test_mask[:, :, :, -1:] = 0
test_curpos = torch.tensor([0], dtype=torch.int32)

# PyTorch reference
layers_per_chunk = 8
pt_k = torch.zeros(layers_per_chunk, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
pt_v = torch.zeros(layers_per_chunk, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
pt_conv = torch.zeros(layers_per_chunk, ane_d1, ane_d2, dtype=MODEL_DTYPE)
pt_rec = torch.zeros(layers_per_chunk, cfg.text_config.linear_num_value_heads,
                     cfg.text_config.linear_key_head_dim,
                     cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)

with torch.no_grad():
    pt_out = model.model.process_layers_regular_single_token_export_local_state(
        hidden_states=test_hidden, position_ids=test_pos,
        causal_mask=test_mask, current_pos=test_curpos,
        kv_cache_0=None, k_cache=pt_k, v_cache=pt_v,
        linear_conv_state=pt_conv, linear_recurrent_state=pt_rec,
        start_layer=0, end_layer=8, apply_final_norm=False,
    )

# CoreML
cml_out2 = cml.predict({
    "hidden_states": test_hidden.numpy().astype(np.float16),
    "position_ids": test_pos.numpy(),
    "causal_mask": test_mask.numpy().astype(np.float16),
    "current_pos": test_curpos.numpy(),
}, state=state)
cml_result = cml_out2["output_hidden_states"]

cos = cosine(pt_out.numpy(), cml_result)
diff = np.abs(pt_out.numpy().astype(np.float32) - cml_result.astype(np.float32))
print("  Cosine similarity: %.10f" % cos)
print("  Max abs diff: %.6f" % diff.max())
print("  Mean abs diff: %.8f" % diff.mean())

if cos > 0.99:
    print("\n  PARITY: PASS (cos > 0.99)")
elif cos > 0.95:
    print("\n  PARITY: ACCEPTABLE (cos > 0.95)")
else:
    print("\n  PARITY: FAIL (cos < 0.95)")

del cml, state; gc.collect()
print("\n" + "=" * 70)
print("  ANE: %s" % ("PASS" if ane_ok else "FAIL (CPU_AND_GPU fallback)"))
print("=" * 70)
