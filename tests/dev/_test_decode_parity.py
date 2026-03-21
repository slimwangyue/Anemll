#!/usr/bin/env python3
"""Quick decode parity check: CPU vs ANE vs PyTorch FFNWrapper.

Uses the same FFNWrapper that the converter traces, for apples-to-apples comparison.
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
PKG_PATH = "/tmp/qwen35_decode_ane_test/qwen35_FFN_chunk_01of04.mlpackage"
CTX = 256

def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))

print("Loading model...")
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(MODEL_PATH)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Check which layers in chunk 0 (0-7) are linear vs full attention
for i in range(8):
    print(f"  Layer {i}: {model.model.layers[i].layer_type}")

# Set up test input
torch.manual_seed(42)
np.random.seed(42)
test_hidden = torch.randn(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE) * 0.01
test_pos = torch.tensor([0], dtype=torch.int32)
# Use -65504 (max finite negative fp16) not -inf
test_mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=MODEL_DTYPE)
test_mask[:, :, :, -1:] = 0  # Allow only last position (shift-left-append)
test_curpos = torch.tensor([0], dtype=torch.int32)

# ── PyTorch via FFNWrapper (same as converter traces) ──
print("\n--- PyTorch FFNWrapper ---")
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

# Reconstruct the FFNWrapper inline (same as convert_part_2 creates)
class TestFFNWrapper(torch.nn.Module):
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
        if c.has_linear_attention():
            conv_dim = (c.text_config.linear_num_key_heads * c.text_config.linear_key_head_dim * 2
                        + c.text_config.linear_num_value_heads * c.text_config.linear_value_head_dim)
            conv_kernel = max(1, int(c.text_config.linear_conv_kernel_dim))
            ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)
            self.register_buffer("linear_conv_state", torch.zeros(
                self.local_num_layers, ane_d1, ane_d2, dtype=MODEL_DTYPE))
            self.register_buffer("linear_recurrent_state", torch.zeros(
                self.local_num_layers, c.text_config.linear_num_value_heads,
                c.text_config.linear_key_head_dim, c.text_config.linear_value_head_dim, dtype=MODEL_DTYPE))
        else:
            self.linear_conv_state = None
            self.linear_recurrent_state = None

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        return self.model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden_states, position_ids=position_ids,
            causal_mask=causal_mask, current_pos=current_pos,
            kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
            linear_conv_state=self.linear_conv_state,
            linear_recurrent_state=self.linear_recurrent_state,
            start_layer=self.start_layer, end_layer=self.end_layer,
            apply_final_norm=False,
        )

wrapper = TestFFNWrapper(model, 0, 8).eval()
with torch.no_grad():
    pt_out = wrapper(test_hidden, test_pos, test_mask, test_curpos)
print(f"  shape: {pt_out.shape}, range: [{pt_out.min():.4f}, {pt_out.max():.4f}]")

# ── CoreML CPU_ONLY ──
print("\n--- CoreML CPU_ONLY ---")
cml_cpu = ct.models.MLModel(PKG_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)
state_cpu = cml_cpu.make_state()
cml_cpu_out = cml_cpu.predict({
    "hidden_states": test_hidden.numpy().astype(np.float16),
    "position_ids": test_pos.numpy(),
    "causal_mask": test_mask.numpy().astype(np.float16),
    "current_pos": test_curpos.numpy(),
}, state=state_cpu)["output_hidden_states"]
print(f"  shape: {cml_cpu_out.shape}, range: [{cml_cpu_out.min():.4f}, {cml_cpu_out.max():.4f}]")

cos_pt_cpu = cosine(pt_out.numpy(), cml_cpu_out)
print(f"  cos(PyTorch, CPU_ONLY): {cos_pt_cpu:.10f}")
del cml_cpu, state_cpu; gc.collect()

# ── CoreML CPU_AND_NE ──
print("\n--- CoreML CPU_AND_NE ---")
cml_ane = ct.models.MLModel(PKG_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE)
state_ane = cml_ane.make_state()
cml_ane_out = cml_ane.predict({
    "hidden_states": test_hidden.numpy().astype(np.float16),
    "position_ids": test_pos.numpy(),
    "causal_mask": test_mask.numpy().astype(np.float16),
    "current_pos": test_curpos.numpy(),
}, state=state_ane)["output_hidden_states"]
print(f"  shape: {cml_ane_out.shape}, range: [{cml_ane_out.min():.4f}, {cml_ane_out.max():.4f}]")

cos_pt_ane = cosine(pt_out.numpy(), cml_ane_out)
cos_cpu_ane = cosine(cml_cpu_out, cml_ane_out)
print(f"  cos(PyTorch, ANE):  {cos_pt_ane:.10f}")
print(f"  cos(CPU_ONLY, ANE): {cos_cpu_ane:.10f}")
del cml_ane, state_ane; gc.collect()

print("\n=== Summary ===")
print(f"  PyTorch vs CPU_ONLY: cos={cos_pt_cpu:.6f}")
print(f"  PyTorch vs ANE:     cos={cos_pt_ane:.6f}")
print(f"  CPU_ONLY vs ANE:    cos={cos_cpu_ane:.6f}")
if cos_pt_cpu < 0.95:
    print("  -> EXPORT/TRACE issue (bad parity even on CPU)")
elif cos_cpu_ane < 0.95:
    print("  -> ANE-SPECIFIC issue (CPU OK but ANE diverges)")
else:
    print("  -> ALL GOOD")
