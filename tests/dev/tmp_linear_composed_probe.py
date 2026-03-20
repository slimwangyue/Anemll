#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import coremltools as ct

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from tests.dev.tmp_linear_layout_probe import LayoutStage
from tests.dev.tmp_linear_proj_probe import CausalConvStage, QKVProjStage
from tests.dev.tmp_linear_stage_probe import NormOutStage, RecurrentCoreStage, metrics, print_metric


class ComposedLinearBlock(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.proj = QKVProjStage(attn)
        self.conv = CausalConvStage(attn, seq_len)
        self.layout = LayoutStage(attn, seq_len)
        self.core = RecurrentCoreStage(attn, seq_len)
        self.norm = NormOutStage(attn, seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_qkv, z, b, a = self.proj(hidden_states)
        conv_out, next_conv = self.conv(mixed_qkv, conv_state)
        query, key, value, beta, g, z_out = self.layout(conv_out, b, a, z)
        core, next_rec = self.core(query, key, value, beta, g, recurrent_state)
        attn_out = self.norm(core, z_out)
        return attn_out, next_conv, next_rec


def convert_model(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    specs: list[ct.TensorType],
    output_names: list[str],
    package_name: str,
    pass_pipeline=None,
) -> ct.models.MLModel:
    traced = torch.jit.trace(module, inputs, strict=False, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=specs,
        outputs=[ct.TensorType(name=name) for name in output_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=pass_pipeline,
    )
    pkg = os.path.join("/tmp", package_name)
    shutil.rmtree(pkg, ignore_errors=True)
    mlmodel.save(pkg)
    return ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.ALL)


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def main() -> None:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-pipeline", choices=["default", "empty"], default="default")
    args = parser.parse_args()

    model_path = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
    layer_idx = 30
    seq_len = 32

    cfg = Qwen35Config.from_json(os.path.join(model_path, "config.json"))
    model = Qwen35ForCausalLM(cfg).half().eval()
    if not model.load_pretrained_weights(model_path):
        raise RuntimeError("failed to load weights")
    attn = model.model.layers[layer_idx].self_attn

    hf_cfg = Qwen3_5Config.from_pretrained(model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=layer_idx).half().eval()

    torch.manual_seed(7)
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32)
    hidden_decode = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32)
    x = hf_layer.input_layernorm(hidden).half()
    x_decode = hf_layer.input_layernorm(hidden_decode).half()
    conv_state = torch.zeros((1, attn.conv_dim, attn.linear_conv_kernel_dim), dtype=torch.float16)
    recurrent_state = torch.zeros((1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim), dtype=torch.float16)

    prefill = ComposedLinearBlock(attn, seq_len).eval()
    decode = ComposedLinearBlock(attn, 1).eval()
    pass_pipeline = None if args.pass_pipeline == "default" else ct.PassPipeline.EMPTY

    with torch.no_grad():
        t_prefill, t_conv, t_rec = prefill(x, conv_state, recurrent_state)
        t_decode, t_conv_d, t_rec_d = decode(x_decode, t_conv, t_rec)

    m_prefill = convert_model(
        prefill,
        (x, conv_state, recurrent_state),
        [
            ct.TensorType(name="input_hidden_states", shape=x.shape, dtype=np.float16),
            ct.TensorType(name="input_conv_state", shape=conv_state.shape, dtype=np.float16),
            ct.TensorType(name="input_recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
        ],
        ["attn_out", "next_conv", "next_rec"],
        "qwen35_linear_composed_prefill.mlpackage",
        pass_pipeline=pass_pipeline,
    )
    m_decode = convert_model(
        decode,
        (x_decode, t_conv, t_rec),
        [
            ct.TensorType(name="input_hidden_states", shape=x_decode.shape, dtype=np.float16),
            ct.TensorType(name="input_conv_state", shape=t_conv.shape, dtype=np.float16),
            ct.TensorType(name="input_recurrent_state", shape=t_rec.shape, dtype=np.float16),
        ],
        ["attn_out", "next_conv", "next_rec"],
        "qwen35_linear_composed_decode.mlpackage",
        pass_pipeline=pass_pipeline,
    )

    p_prefill = m_prefill.predict(
        {
            "input_hidden_states": to_np(x),
            "input_conv_state": to_np(conv_state),
            "input_recurrent_state": to_np(recurrent_state),
        }
    )
    p_decode = m_decode.predict(
        {
            "input_hidden_states": to_np(x_decode),
            "input_conv_state": p_prefill["next_conv"],
            "input_recurrent_state": p_prefill["next_rec"],
        }
    )

    print("Composed Single-Model vs Torch")
    print_metric("prefill_attn", metrics(p_prefill["attn_out"], t_prefill))
    print_metric("prefill_conv", metrics(p_prefill["next_conv"], t_conv))
    print_metric("prefill_rec", metrics(p_prefill["next_rec"], t_rec))
    print_metric("decode_attn", metrics(p_decode["attn_out"], t_decode))
    print_metric("decode_conv", metrics(p_decode["next_conv"], t_conv_d))
    print_metric("decode_rec", metrics(p_decode["next_rec"], t_rec_d))


if __name__ == "__main__":
    main()
