#!/usr/bin/env python3
"""Test if CoreML states are causing error -14 by exporting without states."""
import sys
import os
import time
import shutil

sys.path.insert(0, '/Users/yw68/Anemll')
os.environ['QWEN35_NUM_CHUNKS'] = '4'
os.environ['QWEN35_HF_MODEL'] = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'
os.environ['QWEN35_STATIC_PREFILL_CTX'] = '2048'

import torch
import numpy as np
import coremltools as ct

from anemll.ane_converter.qwen3_5_converter import Qwen35Converter, ane_conv_state_shape
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM, MODEL_DTYPE, TEST_DEVICE

model_dir = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'
output_base = '/Users/yw68/Anemll/tests/dev/ane_op_test_models'

print("Loading model...", flush=True)
cfg = Qwen35Config.from_json(os.path.join(model_dir, "config.json"))
cfg.context_length = 2048
cfg.state_length = 2048
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(model_dir), "Failed to load weights"
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Export chunk 0 (9 layers) WITHOUT states
start_layer = 0
end_layer = 9
export_seq_len = 512

class StatelessPrefillWrapper(torch.nn.Module):
    def __init__(self, model, start_layer, end_layer, seq_len):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.seq_len = seq_len
        self.local_num_layers = end_layer - start_layer

        cfg = model.config
        # k_cache and v_cache as regular inputs/outputs, NOT CoreML states
        for layer_idx in range(start_layer, end_layer):
            layer = self.model.model.layers[layer_idx]
            if getattr(layer, "layer_type", None) == "linear_attention":
                layer.self_attn.export_expected_batch_size = 1
                layer.self_attn.export_expected_seq_len = seq_len

    def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                k_cache, v_cache, linear_conv_state, linear_recurrent_state, valid_len):
        out = self.model.model.process_layers_prefill_export_local_state(
            hidden_states=hidden_states, position_ids=position_ids,
            causal_mask=causal_mask, current_pos=current_pos,
            kv_cache_0=None, k_cache=k_cache, v_cache=v_cache,
            linear_conv_state=linear_conv_state,
            linear_recurrent_state=linear_recurrent_state,
            start_layer=self.start_layer, end_layer=self.end_layer,
            apply_final_norm=False, expected_batch_size=1,
            expected_seq_len=self.seq_len, valid_len=valid_len)
        return out, k_cache, v_cache, linear_conv_state, linear_recurrent_state

wrapper = StatelessPrefillWrapper(model, start_layer, end_layer, export_seq_len).eval()

k_cache = torch.zeros((end_layer - start_layer, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                       dtype=MODEL_DTYPE, device=TEST_DEVICE)
v_cache = torch.zeros_like(k_cache)
hidden_states = torch.zeros((1, export_seq_len, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
position_ids = torch.arange(export_seq_len, dtype=torch.int32, device=TEST_DEVICE)
causal_mask = torch.zeros((1, 1, export_seq_len, 2048), dtype=torch.float16, device=TEST_DEVICE)
current_pos = torch.tensor([0], dtype=torch.int32, device=TEST_DEVICE)
valid_len = torch.tensor([export_seq_len], dtype=torch.int32, device=TEST_DEVICE)

conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
lin_conv = torch.zeros((end_layer - start_layer, ane_dim1, ane_dim2), dtype=MODEL_DTYPE, device=TEST_DEVICE)
lin_rec = torch.zeros((end_layer - start_layer,
                        cfg.text_config.linear_num_value_heads,
                        cfg.text_config.linear_key_head_dim,
                        cfg.text_config.linear_value_head_dim), dtype=MODEL_DTYPE, device=TEST_DEVICE)

print("Tracing...", flush=True)
traced = torch.jit.trace(
    wrapper,
    (hidden_states, position_ids, causal_mask, current_pos, k_cache, v_cache, lin_conv, lin_rec, valid_len),
    check_trace=False)

print("Converting (NO states)...", flush=True)
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
        ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ct.TensorType(name="k_cache", shape=k_cache.shape, dtype=np.float16),
        ct.TensorType(name="v_cache", shape=v_cache.shape, dtype=np.float16),
        ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
        ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
        ct.TensorType(name="valid_len", shape=valid_len.shape, dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="output_hidden_states", dtype=np.float16),
        ct.TensorType(name="k_cache_out", dtype=np.float16),
        ct.TensorType(name="v_cache_out", dtype=np.float16),
        ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
        ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
    ],
    # NO states parameter!
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
    convert_to="mlprogram",
)

# Count ops
spec = mlmodel.get_spec()
prog = spec.mlProgram
total_ops = sum(1 for func in prog.functions.values()
                for block in func.block_specializations.values()
                for op in block.operations)
print(f"Total ops: {total_ops}")

# Check if it loads on Mac
print("Checking if model loads on Mac...")
try:
    pred = mlmodel.predict({"hidden_states": np.zeros(hidden_states.shape, dtype=np.float16),
                            "position_ids": np.arange(export_seq_len, dtype=np.int32),
                            "causal_mask": np.zeros(causal_mask.shape, dtype=np.float16),
                            "current_pos": np.array([0], dtype=np.int32),
                            "k_cache": np.zeros(k_cache.shape, dtype=np.float16),
                            "v_cache": np.zeros(v_cache.shape, dtype=np.float16),
                            "linear_conv_state": np.zeros(lin_conv.shape, dtype=np.float16),
                            "linear_recurrent_state": np.zeros(lin_rec.shape, dtype=np.float16),
                            "valid_len": np.array([export_seq_len], dtype=np.int32)})
    print("MAC PREDICT: OK!")
except Exception as e:
    print(f"MAC PREDICT FAIL: {e}")

output_path = os.path.join(output_base, "test_stateless_prefill_chunk0.mlpackage")
if os.path.exists(output_path):
    shutil.rmtree(output_path)
mlmodel.save(output_path)
size_mb = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fns in os.walk(output_path) for f in fns) / 1e6
print(f"Saved: {output_path} ({size_mb:.1f} MB, {total_ops} ops)")
