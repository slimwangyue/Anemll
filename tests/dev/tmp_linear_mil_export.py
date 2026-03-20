#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import coremltools as ct

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from tests.dev.tmp_linear_layout_probe import LayoutStage
from tests.dev.tmp_linear_monolithic_debug_probe import MonolithicLinearDebugBlock
from tests.dev.tmp_linear_proj_probe import CausalConvStage, QKVProjStage
from tests.dev.tmp_linear_stage_probe import NormOutStage, RecurrentCoreStage


def dump_program(name: str, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], specs, dump_dir: Path) -> None:
    traced = torch.jit.trace(module, inputs, strict=False, check_trace=False)
    prog = ct.convert(
        traced,
        inputs=specs,
        convert_to="milinternal",
        minimum_deployment_target=ct.target.iOS18,
    )
    out_path = dump_dir / f"{name}.mil.txt"
    out_path.write_text(str(prog))
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B")
    parser.add_argument("--layer-idx", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--dump-dir", default="/tmp/qwen35_linear_mil")
    args = parser.parse_args()

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    model = Qwen35ForCausalLM(cfg).half().eval()
    if not model.load_pretrained_weights(args.model_path):
        raise RuntimeError("failed to load weights")
    attn = model.model.layers[args.layer_idx].self_attn

    hf_cfg = Qwen3_5Config.from_pretrained(args.model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=args.layer_idx).half().eval()

    torch.manual_seed(7)
    hidden = torch.randn(1, args.seq_len, cfg.hidden_size, dtype=torch.float32)
    x = hf_layer.input_layernorm(hidden).half()
    conv_state = torch.zeros((1, attn.conv_dim, attn.linear_conv_kernel_dim), dtype=torch.float16)
    recurrent_state = torch.zeros((1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim), dtype=torch.float16)

    proj = QKVProjStage(attn).eval()
    mixed_qkv, z, b, a = proj(x)
    conv = CausalConvStage(attn, args.seq_len).eval()
    conv_out, _ = conv(mixed_qkv, conv_state)
    layout = LayoutStage(attn, args.seq_len).eval()
    query, key, value, beta, g, z_out = layout(conv_out, b, a, z)
    core = RecurrentCoreStage(attn, args.seq_len).eval()
    core_out, _ = core(query, key, value, beta, g, recurrent_state)
    norm = NormOutStage(attn, args.seq_len).eval()

    dump_program(
        "monolithic_debug",
        MonolithicLinearDebugBlock(attn, args.seq_len).eval(),
        (x, conv_state, recurrent_state),
        [
            ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
            ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
        ],
        dump_dir,
    )
    dump_program(
        "stage_proj",
        proj,
        (x,),
        [ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16)],
        dump_dir,
    )
    dump_program(
        "stage_conv",
        conv,
        (mixed_qkv, conv_state),
        [
            ct.TensorType(name="mixed_qkv", shape=mixed_qkv.shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
        ],
        dump_dir,
    )
    dump_program(
        "stage_layout",
        layout,
        (conv_out, b, a, z),
        [
            ct.TensorType(name="conv_out", shape=conv_out.shape, dtype=np.float16),
            ct.TensorType(name="b", shape=b.shape, dtype=np.float16),
            ct.TensorType(name="a", shape=a.shape, dtype=np.float16),
            ct.TensorType(name="z", shape=z.shape, dtype=np.float16),
        ],
        dump_dir,
    )
    dump_program(
        "stage_core",
        core,
        (query, key, value, beta, g, recurrent_state),
        [
            ct.TensorType(name="query", shape=query.shape, dtype=np.float16),
            ct.TensorType(name="key", shape=key.shape, dtype=np.float16),
            ct.TensorType(name="value", shape=value.shape, dtype=np.float16),
            ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
            ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
            ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
        ],
        dump_dir,
    )
    dump_program(
        "stage_norm",
        norm,
        (core_out, z_out),
        [
            ct.TensorType(name="core", shape=core_out.shape, dtype=np.float16),
            ct.TensorType(name="z", shape=z_out.shape, dtype=np.float16),
        ],
        dump_dir,
    )


if __name__ == "__main__":
    main()
