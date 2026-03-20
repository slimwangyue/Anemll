#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import coremltools as ct

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from tests.dev.test_qwen35_linear_attention_stateful_coreml_vs_hf import (
    StatelessLinearAttentionPrefillBlock,
)


def main() -> None:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    model_path = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
    layer_idx = 30
    seq_len = 32

    cfg = Qwen35Config.from_json(os.path.join(model_path, "config.json"))
    model = Qwen35ForCausalLM(cfg).half().eval()
    if not model.load_pretrained_weights(model_path):
        raise RuntimeError("failed to load weights")
    our_attn = model.model.layers[layer_idx].self_attn

    hf_cfg = Qwen3_5Config.from_pretrained(model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=layer_idx).half().eval()

    torch.manual_seed(7)
    hidden_prefill = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32)
    x_prefill = hf_layer.input_layernorm(hidden_prefill).half()
    conv_state = torch.zeros((1, our_attn.conv_dim, our_attn.linear_conv_kernel_dim), dtype=torch.float16)
    recurrent_state = torch.zeros(
        (1, our_attn.num_v_heads, our_attn.head_k_dim, our_attn.head_v_dim), dtype=torch.float16
    )

    block = StatelessLinearAttentionPrefillBlock(
        our_attn,
        seq_len,
        force_recurrent=True,
        force_fp16_math=True,
        recurrent_state_output_dtype=torch.float16,
    ).eval()
    traced = torch.jit.trace(block, (x_prefill, conv_state, recurrent_state), strict=False, check_trace=False)

    prog = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=x_prefill.shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
            ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="attn_out"),
            ct.TensorType(name="next_conv"),
            ct.TensorType(name="next_rec"),
        ],
        convert_to="milinternal",
        minimum_deployment_target=ct.target.iOS18,
    )

    print("scan cast ops")
    found = 0
    for fn_name, fn in prog.functions.items():
        print("FUNCTION", fn_name)
        for op in fn.operations:
            if op.op_type != "cast":
                continue
            dtype = getattr(op.dtype, "val", op.dtype)
            xval = getattr(op.x, "val", None)
            if xval is None:
                continue
            arr = np.array(xval)
            if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            amin = float(np.min(finite))
            amax = float(np.max(finite))
            if str(dtype) in ("fp16", "float16") and (amax > 65504 or amin < -65504):
                found += 1
                print("OVERFLOW_CAST", op.name, "dtype", dtype, "shape", arr.shape, "min", amin, "max", amax)
            elif str(dtype) in ("fp16", "float16") and (amax > 1e4 or amin < -1e4):
                found += 1
                print("LARGE_CAST", op.name, "dtype", dtype, "shape", arr.shape, "min", amin, "max", amax)
    print("FOUND", found)


if __name__ == "__main__":
    main()
