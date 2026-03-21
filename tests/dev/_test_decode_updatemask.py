#!/usr/bin/env python3
"""Test update_mask decode pattern on ANE: export, load, run, compare parity.

Exports chunk 1 (layers 0-7) decode model with update_mask input,
tests on CPU_ONLY vs CPU_AND_NE, compares against PyTorch FFNWrapper.
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

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_decode_updatemask"
CTX = 256
os.makedirs(OUT_DIR, exist_ok=True)

def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))

print("=" * 60)
print("  Update-mask decode ANE test")
print("=" * 60)

# ── 1. Load model ──
print("\nLoading model...")
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(MODEL_PATH)
model.eval()
for p in model.parameters():
    p.requires_grad = False

print("Layer types in chunk 0 (layers 0-7):")
for i in range(8):
    print(f"  Layer {i}: {model.model.layers[i].layer_type}")

conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)

# ── 2. Export chunk 1 via converter ──
print("\nExporting decode chunk 1 (layers 0-7)...")
converter = Qwen35Converter(
    model=model,
    context_length=CTX,
    batch_size=1,
    lut_bits=None,
    num_chunks=4,
)
mlmodel = converter.convert_part_2(model, chunk_idx=0, total_chunks=4)
pkg_path = os.path.join(OUT_DIR, "chunk_01of04.mlpackage")
if os.path.exists(pkg_path):
    shutil.rmtree(pkg_path)
mlmodel.save(pkg_path)
print(f"  Saved: {pkg_path}")

# Print model inputs for verification
spec = mlmodel.get_spec()
print("  Inputs:")
for inp in spec.description.input:
    if inp.type.HasField("multiArrayType"):
        print(f"    {inp.name}: shape={list(inp.type.multiArrayType.shape)}")
    elif inp.type.HasField("stateType"):
        print(f"    {inp.name}: [state]")
del mlmodel; gc.collect()

# ── 3. Build PyTorch reference via FFNWrapper pattern ──
print("\nBuilding PyTorch reference...")

class TestWrapper(torch.nn.Module):
    def __init__(self, model, start_layer, end_layer):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.local_num_layers = end_layer - start_layer
        c = model.config
        self.register_buffer("k_cache", torch.zeros(
            self.local_num_layers, c.num_key_value_heads, c.state_length, c.head_dim, dtype=MODEL_DTYPE))
        self.register_buffer("v_cache", torch.zeros(
            self.local_num_layers, c.num_key_value_heads, c.state_length, c.head_dim, dtype=MODEL_DTYPE))
        self.register_buffer("linear_conv_state", torch.zeros(
            self.local_num_layers, ane_d1, ane_d2, dtype=MODEL_DTYPE))
        self.register_buffer("linear_recurrent_state", torch.zeros(
            self.local_num_layers, c.text_config.linear_num_value_heads,
            c.text_config.linear_key_head_dim, c.text_config.linear_value_head_dim, dtype=MODEL_DTYPE))

    def forward(self, hidden_states, position_ids, causal_mask, current_pos, update_mask):
        return self.model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden_states, position_ids=position_ids,
            causal_mask=causal_mask, current_pos=current_pos,
            kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
            linear_conv_state=self.linear_conv_state,
            linear_recurrent_state=self.linear_recurrent_state,
            update_mask=update_mask,
            start_layer=self.start_layer, end_layer=self.end_layer,
            apply_final_norm=False,
        )

# ── 4. Test with 5 sequential tokens ──
print("\nRunning 5-token sequential comparison...")

torch.manual_seed(42)
np.random.seed(42)

for backend_name, compute in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
    print(f"\n--- {backend_name} ---")
    
    # Fresh state for CoreML
    try:
        cml = ct.models.MLModel(pkg_path, compute_units=compute)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        continue
    state = cml.make_state()
    
    # Fresh state for PyTorch
    pt_wrapper = TestWrapper(model, 0, 8).eval()
    
    for step in range(5):
        h = torch.randn(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE) * 0.01
        pos = torch.tensor([step], dtype=torch.int32)
        mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=MODEL_DTYPE)
        mask[:, :, :, :step+1] = 0  # Allow positions 0..step
        cur = torch.tensor([step], dtype=torch.int32)
        # update_mask: 1.0 at position `step`, 0.0 elsewhere
        umask = torch.zeros((1, 1, CTX, 1), dtype=MODEL_DTYPE)
        umask[:, :, step, :] = 1.0
        
        # PyTorch
        with torch.no_grad():
            pt_out = pt_wrapper(h, pos, mask, cur, umask)
        
        # CoreML
        cml_out = cml.predict({
            "hidden_states": h.numpy().astype(np.float16),
            "position_ids": pos.numpy(),
            "causal_mask": mask.numpy().astype(np.float16),
            "current_pos": cur.numpy(),
            "update_mask": umask.numpy().astype(np.float16),
        }, state=state)["output_hidden_states"]
        
        cos = cosine(pt_out.numpy(), cml_out)
        diff = np.abs(pt_out.numpy().astype(np.float32) - cml_out.astype(np.float32))
        print(f"  Step {step}: cos={cos:.8f}  max_abs={diff.max():.4f}  range=[{cml_out.min():.4f}, {cml_out.max():.4f}]")
    
    del cml, state, pt_wrapper; gc.collect()

print("\n" + "=" * 60)
print("  Done")
print("=" * 60)
