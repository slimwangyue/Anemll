#!/usr/bin/env python3
"""Isolate ANE decode parity: test linear-only vs full-attention layers separately."""
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
OUT_DIR = "/tmp/qwen35_decode_isolate"
CTX = 256
os.makedirs(OUT_DIR, exist_ok=True)

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

conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)

class IsolatedWrapper(torch.nn.Module):
    """Export a specific layer range for decode."""
    def __init__(self, model, start_layer, end_layer):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.local_num_layers = end_layer - start_layer
        c = model.config

        # Check which layer types are present
        has_full = any(model.model.layers[i].layer_type != "linear_attention"
                       for i in range(start_layer, end_layer))
        has_linear = any(model.model.layers[i].layer_type == "linear_attention"
                        for i in range(start_layer, end_layer))

        # Only register states that will be used (unused states cause CoreML error)
        if has_full:
            self.register_buffer("k_cache", torch.zeros(
                self.local_num_layers, c.num_key_value_heads, c.state_length, c.head_dim, dtype=MODEL_DTYPE))
            self.register_buffer("v_cache", torch.zeros(
                self.local_num_layers, c.num_key_value_heads, c.state_length, c.head_dim, dtype=MODEL_DTYPE))
        else:
            self.k_cache = None
            self.v_cache = None

        if has_linear and c.has_linear_attention():
            self.register_buffer("linear_conv_state", torch.zeros(
                self.local_num_layers, ane_d1, ane_d2, dtype=MODEL_DTYPE))
            self.register_buffer("linear_recurrent_state", torch.zeros(
                self.local_num_layers, c.text_config.linear_num_value_heads,
                c.text_config.linear_key_head_dim, c.text_config.linear_value_head_dim, dtype=MODEL_DTYPE))
        else:
            self.linear_conv_state = None
            self.linear_recurrent_state = None

        # Build state list matching only registered buffers
        states = []
        if has_full:
            states.extend([
                ct.StateType(wrapped_type=ct.TensorType(
                    shape=(self.local_num_layers, c.num_key_value_heads, c.state_length, c.head_dim),
                    dtype=np.float16), name="k_cache"),
                ct.StateType(wrapped_type=ct.TensorType(
                    shape=(self.local_num_layers, c.num_key_value_heads, c.state_length, c.head_dim),
                    dtype=np.float16), name="v_cache"),
            ])
        if has_linear and c.has_linear_attention():
            states.extend([
                ct.StateType(wrapped_type=ct.TensorType(
                    shape=(self.local_num_layers, ane_d1, ane_d2),
                    dtype=np.float16), name="linear_conv_state"),
                ct.StateType(wrapped_type=ct.TensorType(
                    shape=(self.local_num_layers, c.text_config.linear_num_value_heads,
                           c.text_config.linear_key_head_dim, c.text_config.linear_value_head_dim),
                    dtype=np.float16), name="linear_recurrent_state"),
            ])
        self.states = states

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


def export_and_test(name, start_layer, end_layer):
    """Export layers start..end, test CPU vs ANE parity."""
    print(f"\n{'='*60}")
    layer_types = [model.model.layers[i].layer_type for i in range(start_layer, end_layer)]
    print(f"  {name}: layers {start_layer}-{end_layer-1}")
    print(f"  Types: {layer_types}")
    print(f"{'='*60}")

    wrapper = IsolatedWrapper(model, start_layer, end_layer).eval()

    # Trace
    h = torch.zeros(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE)
    p = torch.zeros(1, dtype=torch.int32)
    m = torch.zeros(1, 1, 1, CTX, dtype=MODEL_DTYPE)
    c = torch.zeros(1, dtype=torch.int32)

    traced = torch.jit.trace(wrapper, (h, p, m, c))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=h.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=p.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=m.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=c.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    pkg_path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    if os.path.exists(pkg_path):
        shutil.rmtree(pkg_path)
    mlmodel.save(pkg_path)
    del mlmodel, traced
    gc.collect()

    # Test input
    torch.manual_seed(42)
    np.random.seed(42)
    test_h = torch.randn(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE) * 0.01
    test_p = torch.tensor([5], dtype=torch.int32)  # pos=5 to have some RoPE
    test_m = torch.full((1, 1, 1, CTX), -65504.0, dtype=MODEL_DTYPE)
    test_m[:, :, :, -6:] = 0  # Allow last 6 positions (pos 0-5)
    test_c = torch.tensor([5], dtype=torch.int32)

    # PyTorch
    wrapper2 = IsolatedWrapper(model, start_layer, end_layer).eval()
    with torch.no_grad():
        pt_out = wrapper2(test_h, test_p, test_m, test_c)
    del wrapper2

    np_inputs = {
        "hidden_states": test_h.numpy().astype(np.float16),
        "position_ids": test_p.numpy(),
        "causal_mask": test_m.numpy().astype(np.float16),
        "current_pos": test_c.numpy(),
    }

    # CPU
    cml_cpu = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    st_cpu = cml_cpu.make_state()
    # Run 6 steps to fill state with tokens at positions 0-5 
    # (just run step 0 for simplicity - checking single step parity)
    cpu_out = cml_cpu.predict(np_inputs, state=st_cpu)["output_hidden_states"]
    del cml_cpu, st_cpu; gc.collect()

    # ANE
    cml_ane = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    st_ane = cml_ane.make_state()
    ane_out = cml_ane.predict(np_inputs, state=st_ane)["output_hidden_states"]
    del cml_ane, st_ane; gc.collect()

    cos_pt_cpu = cosine(pt_out.numpy(), cpu_out)
    cos_pt_ane = cosine(pt_out.numpy(), ane_out)
    cos_cpu_ane = cosine(cpu_out, ane_out)

    print(f"  PyTorch vs CPU:  cos={cos_pt_cpu:.8f}")
    print(f"  PyTorch vs ANE:  cos={cos_pt_ane:.8f}")
    print(f"  CPU vs ANE:      cos={cos_cpu_ane:.8f}")
    gc.collect()
    return cos_pt_cpu, cos_pt_ane, cos_cpu_ane


# Test individual layers
results = {}

# Test 1: Single full attention layer only (layer 3)
results["full_L3"] = export_and_test("full_L3", 3, 4)

# Test 2: Single linear attention layer (layer 0) 
results["linear_L0"] = export_and_test("linear_L0", 0, 1)

# Test 3: 3 linear attention layers (layers 0-2)
results["linear_L0_L2"] = export_and_test("linear_L0_L2", 0, 3)

# Test 4: Mixed: 3 linear + 1 full (layers 0-3)
results["mixed_L0_L3"] = export_and_test("mixed_L0_L3", 0, 4)

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
for name, (pt_cpu, pt_ane, cpu_ane) in results.items():
    status = "OK" if cpu_ane > 0.95 else "DIVERGE"
    print(f"  {name:20s}: CPU={pt_cpu:.6f}  ANE={pt_ane:.6f}  CPU-ANE={cpu_ane:.6f}  [{status}]")
