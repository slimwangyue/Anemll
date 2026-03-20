#!/usr/bin/env python3
"""Export a single Qwen3.5 layer (decode mode) and test on ANE.
This isolates whether the issue is layer count, specific ops, or something else.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch
import numpy as np
import coremltools as ct
import os
import time
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
NUM_LAYERS_TO_TEST = 1  # Just 1 layer

print("Loading model config...")
config = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
config.context_length = 256
config.state_length = 256

print("Creating model...")
model = Qwen35ForCausalLM(config)
ok = model.load_pretrained_weights(MODEL_PATH)
if not ok:
    print("FAILED to load weights!")
    exit(1)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Create a minimal wrapper for layer 0 (linear attention)
print(f"\nExporting layer 0 (type: {model.model.layers[0].layer_type})...")

class SingleLayerWrapper(torch.nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.layer = model.model.layers[layer_idx]
        layer_type = self.layer.layer_type
        cfg = model.config

        if layer_type == "linear_attention":
            conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                       + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
            conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
            self.register_buffer("conv_state", torch.zeros(1, conv_dim, conv_kernel, dtype=torch.float16))
            self.register_buffer("rec_state", torch.zeros(
                1, cfg.text_config.linear_num_value_heads,
                cfg.text_config.linear_key_head_dim,
                cfg.text_config.linear_value_head_dim, dtype=torch.float16))
            self.has_kv = False
            self.layer.self_attn.export_expected_batch_size = 1
            self.layer.self_attn.export_expected_seq_len = 1
        else:
            self.register_buffer("k_cache", torch.zeros(1, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim, dtype=torch.float16))
            self.register_buffer("v_cache", torch.zeros(1, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim, dtype=torch.float16))
            self.has_kv = True

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        layer = self.layer
        if not self.has_kv:
            x = layer.input_layernorm(hidden_states)
            conv_state = self.conv_state
            rec_state = self.rec_state
            attn_out, next_conv, next_rec = layer.self_attn.forward_regular(
                hidden_states=x,
                conv_state=conv_state,
                recurrent_state=rec_state,
                has_previous_state=True,
                causal_mask=causal_mask,
                expected_batch_size=1,
                expected_seq_len=1,
                force_recurrent=True,
            )
            self.conv_state[:] = next_conv
            self.rec_state[:] = next_rec.to(self.rec_state.dtype)
            hidden_states = hidden_states + attn_out
            post = layer.post_attention_layernorm(hidden_states)
            return hidden_states + layer.mlp(post)
        else:
            x = layer.input_layernorm(hidden_states)
            query_states, key_states, value_states, gate = layer.self_attn.get_new_kv_cache(x, current_pos)
            # Static slice: always write at 0
            self.k_cache[:, :, 0:1, :] = key_states
            self.v_cache[:, :, 0:1, :] = value_states
            attn_out = layer.self_attn.forward_regular(
                hidden_states=x,
                query_states=query_states,
                kv_cache_layer=(self.k_cache.squeeze(0), self.v_cache.squeeze(0)),
                causal_mask=causal_mask,
                gate=gate,
            )
            hidden_states = hidden_states + attn_out
            post = layer.post_attention_layernorm(hidden_states)
            return hidden_states + layer.mlp(post)

wrapper = SingleLayerWrapper(model, 0).eval()

hidden = torch.zeros(1, 1, 2560, dtype=torch.float16)
pos_ids = torch.zeros(1, dtype=torch.int32)
mask = torch.zeros(1, 1, 1, 256, dtype=torch.float16)
cpos = torch.zeros(1, dtype=torch.int32)

print("Tracing...")
traced = torch.jit.trace(wrapper, (hidden, pos_ids, mask, cpos))

# Build states
layer_type = model.model.layers[0].layer_type
if layer_type == "linear_attention":
    conv_dim = (config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim * 2
               + config.text_config.linear_num_value_heads * config.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(config.text_config.linear_conv_kernel_dim))
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, conv_dim, conv_kernel), dtype=np.float16), name="conv_state"),
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, config.text_config.linear_num_value_heads,
            config.text_config.linear_key_head_dim, config.text_config.linear_value_head_dim), dtype=np.float16), name="rec_state"),
    ]
else:
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, config.num_key_value_heads, config.state_length, config.head_dim), dtype=np.float16), name="k_cache"),
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, config.num_key_value_heads, config.state_length, config.head_dim), dtype=np.float16), name="v_cache"),
    ]

print("Converting to CoreML...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=pos_ids.shape, dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
        ct.TensorType(name="current_pos", shape=cpos.shape, dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    states=states,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)

path = "/tmp/qwen35_ane_test/single_layer_0.mlpackage"
if os.path.exists(path):
    import shutil
    shutil.rmtree(path)
mlmodel.save(path)

# Check MIL ops
spec = mlmodel.get_spec()
from collections import Counter
op_counts = Counter()
for fn in spec.mlProgram.functions.values():
    for blk in fn.block_specializations.values():
        for op in blk.operations:
            op_counts[op.type] += 1
print(f"\nOp counts ({sum(op_counts.values())} total):")
for t, c in op_counts.most_common(15):
    print(f"  {t}: {c}")
del mlmodel

# Test on ANE
print(f"\nTesting single layer 0 ({layer_type}) on ANE...")
loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
state = loaded.make_state()
inputs = {
    "hidden_states": np.random.randn(1, 1, 2560).astype(np.float16) * 0.01,
    "position_ids": np.zeros(1, dtype=np.int32),
    "causal_mask": np.zeros((1, 1, 1, 256), dtype=np.float16),
    "current_pos": np.zeros(1, dtype=np.int32),
}
try:
    out = loaded.predict(inputs, state=state)
    print("✅ Single layer SUCCESS on ANE!")
except Exception as e:
    if "ANE" in str(e):
        print("❌ Single layer FAILED on ANE")
    else:
        print(f"❌ Error: {str(e)[:200]}")
