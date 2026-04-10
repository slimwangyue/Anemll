#!/usr/bin/env python3
"""Dump MIL op names from a real chunk to understand naming patterns."""
import sys, os, warnings, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts_qwen3_5"))

import torch
import numpy as np
import coremltools as ct
from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES, FFN_PER_CHANNEL
from anemll.models.qwen3_5_model import (
    Qwen35Config, Qwen35ForCausalLM, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# Load model
cfg = Qwen35Config.from_json(os.path.join(REPO, "models/Qwen__Qwen3.5-4B/config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
model.load_pretrained_weights(os.path.join(REPO, "models/Qwen__Qwen3.5-4B"))
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Pick chunk 5 (layers 19-22): FLLL pattern, has both full and linear attn
chunk_idx = 5
sl, el = CHUNK_RANGES[chunk_idx]
print(f"Chunk {chunk_idx}: layers {sl}-{el-1}")

conv = Qwen35Converter(
    model, context_length=CTX, batch_size=BATCH_SIZE,
    num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=FFN_PER_CHANNEL,
    compute_precision="float16",
)

class FFNWrapper(torch.nn.Module):
    def __init__(self, model, start_layer, end_layer):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.local_num_layers = end_layer - start_layer
        self.register_buffer("k_cache", torch.zeros(
            (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=MODEL_DTYPE, device=TEST_DEVICE))
        self.register_buffer("v_cache", torch.zeros(
            (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=MODEL_DTYPE, device=TEST_DEVICE))
        conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                    + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
        conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
        ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
        self._lin_conv_shape = (self.local_num_layers, ane_dim1, ane_dim2)
        self._lin_rec_shape = (self.local_num_layers, cfg.text_config.linear_num_value_heads,
                               cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim)
        self.states = Qwen35Converter.GetChunkLocalTransformerStates(
            model, self.local_num_layers, prefix="", split_full_attention_kv=True)

    def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                linear_conv_state, linear_recurrent_state):
        out = self.model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden_states, position_ids=position_ids,
            causal_mask=causal_mask, current_pos=current_pos,
            kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
            linear_conv_state=linear_conv_state, linear_recurrent_state=linear_recurrent_state,
            start_layer=self.start_layer, end_layer=self.end_layer, apply_final_norm=False)
        return out, linear_conv_state, linear_recurrent_state

wrapper = FFNWrapper(model, sl, el).eval()
hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)

conv._reset_state_buffers(wrapper)
traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec), check_trace=False)
conv._reset_state_buffers(wrapper)
conv._reset_state_buffers(traced)

print("Converting to MIL (FP32)...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
        ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
        ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
    ],
    outputs=[
        ct.TensorType(name="output_hidden_states", dtype=np.float16),
        ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
        ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
    ],
    states=wrapper.states,
    compute_precision=ct.precision.FLOAT32,  # Keep FP32 to see all op names
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
    convert_to="mlprogram",
)

# Dump ALL op names, types from MIL
prog = mlmodel._mil_program
for fname, func in prog.functions.items():
    print(f"\n=== Function: {fname} ===")
    op_count = 0
    for op in func.operations:
        out_shapes = []
        for o in op.outputs:
            try:
                out_shapes.append(str(tuple(o.shape)))
            except:
                out_shapes.append("?")
        op_count += 1
        print(f"  {op.name:60s}  {op.op_type:20s}  {','.join(out_shapes)}")
    print(f"\n  Total ops in {fname}: {op_count}")

# Also print summary by op_type
from collections import Counter
type_counts = Counter()
type_names = {}
for fname, func in prog.functions.items():
    for op in func.operations:
        type_counts[op.op_type] += 1
        type_names.setdefault(op.op_type, []).append(op.name)

print("\n=== Op Type Summary ===")
for op_type, count in type_counts.most_common():
    sample = type_names[op_type][:3]
    print(f"  {op_type:20s}: {count:4d}  (e.g. {', '.join(sample)})")

print("\nDone.")
