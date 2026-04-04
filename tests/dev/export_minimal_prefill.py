#!/usr/bin/env python3
"""Export a minimal 1-layer or 2-layer prefill chunk to find the breaking point."""
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

from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
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
print("Model loaded.", flush=True)

# Try exporting just layers [0, N) for different N values
for num_layers in [1, 2, 4]:
    name = f"test_real_prefill_{num_layers}layer"
    output_path = os.path.join(output_base, f"{name}.mlpackage")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    print(f"\n=== Exporting {num_layers}-layer prefill chunk ===", flush=True)
    t0 = time.time()

    # Use the converter but force specific layer range
    export_seq_len = 512
    context_length = 2048
    start_layer = 0
    end_layer = num_layers

    # Prepare the model layers
    from anemll.ane_converter.qwen3_5_converter import ane_conv_state_shape

    class MinimalPrefillWrapper(torch.nn.Module):
        def __init__(self, model, start_layer, end_layer, seq_len):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.seq_len = seq_len
            self.local_num_layers = end_layer - start_layer

            cfg = model.config
            self.register_buffer("k_cache", torch.zeros(
                (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE))
            self.register_buffer("v_cache", torch.zeros(
                (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE))

            if cfg.has_linear_attention():
                conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
                conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
                ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                self._lin_conv_shape = (self.local_num_layers, ane_dim1, ane_dim2)
                self._lin_rec_shape = (self.local_num_layers,
                    cfg.text_config.linear_num_value_heads,
                    cfg.text_config.linear_key_head_dim,
                    cfg.text_config.linear_value_head_dim)
                self._has_linear = True
            else:
                self._has_linear = False

            for layer_idx in range(start_layer, end_layer):
                layer = self.model.model.layers[layer_idx]
                if getattr(layer, "layer_type", None) == "linear_attention":
                    layer.self_attn.export_expected_batch_size = 1
                    layer.self_attn.export_expected_seq_len = seq_len

            self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                model, self.local_num_layers, prefix="", split_full_attention_kv=True)

        def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                    linear_conv_state, linear_recurrent_state, valid_len):
            out = self.model.model.process_layers_prefill_export_local_state(
                hidden_states=hidden_states, position_ids=position_ids,
                causal_mask=causal_mask, current_pos=current_pos,
                kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
                linear_conv_state=linear_conv_state,
                linear_recurrent_state=linear_recurrent_state,
                start_layer=self.start_layer, end_layer=self.end_layer,
                apply_final_norm=False, expected_batch_size=1,
                expected_seq_len=self.seq_len, valid_len=valid_len)
            return out, linear_conv_state, linear_recurrent_state

    wrapper = MinimalPrefillWrapper(model, start_layer, end_layer, export_seq_len).eval()

    hidden_states = torch.zeros((1, export_seq_len, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    position_ids = torch.arange(export_seq_len, dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, export_seq_len, context_length), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.tensor([0], dtype=torch.int32, device=TEST_DEVICE)
    valid_len = torch.tensor([export_seq_len], dtype=torch.int32, device=TEST_DEVICE)

    if wrapper._has_linear:
        lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
        lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    else:
        lin_conv = torch.zeros((num_layers, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        lin_rec = torch.zeros((num_layers, 1, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)

    # Reset state buffers
    for name_buf, buf in wrapper.named_buffers():
        if 'cache' in name_buf:
            buf.zero_()

    traced = torch.jit.trace(
        wrapper,
        (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec, valid_len),
        check_trace=False)

    for name_buf, buf in wrapper.named_buffers():
        if 'cache' in name_buf:
            buf.zero_()
    for name_buf, buf in traced.named_buffers():
        if 'cache' in name_buf:
            buf.zero_()

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
            ct.TensorType(name="valid_len", shape=valid_len.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
        ],
        states=wrapper.states,
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

    mlmodel.save(output_path)
    dt = time.time() - t0
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fns in os.walk(output_path) for f in fns) / 1e6
    print(f"  {num_layers} layers: {total_ops} ops, {size_mb:.1f} MB, {dt:.1f}s")

print("\nDone. Push to iPhone and test on ANE.", flush=True)
