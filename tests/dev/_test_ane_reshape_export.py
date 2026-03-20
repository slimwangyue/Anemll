#!/usr/bin/env python3
"""Test the ANE conv_state reshape fix by exporting all 4 prefill chunks
through the production code path and loading each on ANE.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os
import shutil
import torch
import numpy as np
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_ane_test/reshape_test"

# Clean output
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...")
config = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
config.context_length = 256
config.state_length = 256
config.batch_size = 64

model = Qwen35ForCausalLM(config)
ok = model.load_pretrained_weights(MODEL_PATH)
assert ok, "Failed to load weights"
model.eval()
for p in model.parameters():
    p.requires_grad = False

conv_dim = (config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim * 2
           + config.text_config.linear_num_value_heads * config.text_config.linear_value_head_dim)
conv_kernel = max(1, int(config.text_config.linear_conv_kernel_dim))
ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
print(f"conv_dim={conv_dim}, conv_kernel={conv_kernel}")
print(f"ANE-safe shape: ({ane_dim1}, {ane_dim2}) [was ({conv_dim}, {conv_kernel})]")
print(f"Total layers: {config.num_hidden_layers}")

# 4 chunks: 0-7, 8-15, 16-23, 24-31
chunks = [(0, 8), (8, 16), (16, 24), (24, 32)]
batch_size = 64
results = []

for chunk_idx, (start_layer, end_layer) in enumerate(chunks):
    num_layers = end_layer - start_layer
    print(f"\n{'='*60}")
    print(f"CHUNK {chunk_idx+1}/4: layers {start_layer}-{end_layer-1}")
    print(f"{'='*60}")

    class PrefillWrapper(torch.nn.Module):
        def __init__(self, model, start_layer, end_layer, export_seq_len):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.export_seq_len = export_seq_len
            cfg = model.config
            local_layers = end_layer - start_layer

            self.register_buffer("k_cache", torch.zeros(
                (local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE))
            self.register_buffer("v_cache", torch.zeros(
                (local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE))

            if cfg.has_linear_attention():
                cd = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                     + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
                ck = max(1, int(cfg.text_config.linear_conv_kernel_dim))
                ad1, ad2 = ane_conv_state_shape(cd, ck)
                self.register_buffer("linear_conv_state", torch.zeros(
                    (local_layers, ad1, ad2), dtype=MODEL_DTYPE, device=TEST_DEVICE))
                self.register_buffer("linear_recurrent_state", torch.zeros(
                    (local_layers, cfg.text_config.linear_num_value_heads,
                     cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
                    dtype=MODEL_DTYPE, device=TEST_DEVICE))
            else:
                self.linear_conv_state = None
                self.linear_recurrent_state = None

            for layer_idx in range(start_layer, end_layer):
                layer = self.model.model.layers[layer_idx]
                if getattr(layer, "layer_type", None) == "linear_attention":
                    layer.self_attn.export_expected_batch_size = 1
                    layer.self_attn.export_expected_seq_len = export_seq_len

            self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                model, local_layers, prefix="", split_full_attention_kv=True
            )

        def forward(self, hidden_states, position_ids, causal_mask, current_pos):
            out = self.model.model.process_layers_prefill_export_local_state(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                kv_cache_0=None,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                linear_conv_state=self.linear_conv_state,
                linear_recurrent_state=self.linear_recurrent_state,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                apply_final_norm=False,
                expected_batch_size=1,
                expected_seq_len=self.export_seq_len,
            )
            if self.end_layer == len(self.model.model.layers):
                return out[:, 0:1, :]
            return out

    wrapper = PrefillWrapper(model, start_layer, end_layer, batch_size).eval()

    hidden = torch.zeros((1, batch_size, config.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    pos_ids = torch.zeros((batch_size,), dtype=torch.int32, device=TEST_DEVICE)
    mask = torch.zeros((1, 1, batch_size, config.state_length), dtype=torch.float16, device=TEST_DEVICE)
    cpos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

    wrapper.k_cache.zero_()
    wrapper.v_cache.zero_()
    if wrapper.linear_conv_state is not None:
        wrapper.linear_conv_state.zero_()
    if wrapper.linear_recurrent_state is not None:
        wrapper.linear_recurrent_state.zero_()

    print("Tracing...")
    traced = torch.jit.trace(wrapper, (hidden, pos_ids, mask, cpos))

    states = Qwen35Converter.GetChunkLocalTransformerStates(
        model, num_layers, prefix="", split_full_attention_kv=True
    )

    path = os.path.join(OUT_DIR, f"prefill_chunk{chunk_idx+1}.mlpackage")
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
    mlmodel.save(path)
    print(f"Saved to {path}")
    del mlmodel

    # Test on ANE
    print(f"Loading chunk {chunk_idx+1} on ANE...")
    loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = loaded.make_state()
    out_shape = (1, 1, config.hidden_size) if end_layer == 32 else (1, batch_size, config.hidden_size)
    inputs = {
        "hidden_states": np.random.randn(1, batch_size, config.hidden_size).astype(np.float16) * 0.01,
        "position_ids": np.arange(batch_size, dtype=np.int32),
        "causal_mask": np.zeros((1, 1, batch_size, config.state_length), dtype=np.float16),
        "current_pos": np.zeros(1, dtype=np.int32),
    }
    try:
        out = loaded.predict(inputs, state=state)
        shape = list(out.values())[0].shape
        print(f"✅ CHUNK {chunk_idx+1} SUCCESS on ANE! Output shape: {shape}")
        results.append(("PASS", shape))
    except Exception as e:
        err = str(e)[:300]
        print(f"❌ CHUNK {chunk_idx+1} FAILED: {err}")
        results.append(("FAIL", err))
    del loaded

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for i, (status, info) in enumerate(results):
    print(f"  Chunk {i+1}: {status} - {info}")
