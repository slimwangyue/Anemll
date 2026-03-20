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


def metrics(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> dict[str, float]:
    if not isinstance(a, torch.Tensor):
        a = torch.from_numpy(np.asarray(a))
    if not isinstance(b, torch.Tensor):
        b = torch.from_numpy(np.asarray(b))
    a32 = a.float().reshape(-1)
    b32 = b.float().reshape(-1)
    diff = (a32 - b32).abs()
    cos = torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0)).item()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a32 - b32) ** 2)).item()),
        "cosine": float(cos),
    }


def print_metric(name: str, m: dict[str, float]) -> None:
    print(
        f"{name:18s} max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
        f"rmse={m['rmse']:.6e} cosine={m['cosine']:.8f}"
    )


class QKVProjStage(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module):
        super().__init__()
        self.attn = attn

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_qkv = self.attn._conv2d_proj(self.attn.in_proj_qkv, hidden_states).transpose(1, 2)
        z = self.attn._conv2d_proj(self.attn.in_proj_z, hidden_states)
        b = self.attn._conv2d_proj(self.attn.in_proj_b, hidden_states)
        a = self.attn._conv2d_proj(self.attn.in_proj_a, hidden_states)
        return mixed_qkv, z, b, a


class CausalConvStage(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.attn = attn
        self.seq_len = int(seq_len)

    def forward(self, mixed_qkv: torch.Tensor, conv_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attn._causal_conv_update(mixed_qkv, conv_state, expected_seq_len=self.seq_len)


def convert_and_run(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    specs: list[ct.TensorType],
    output_names: list[str],
    package_name: str,
) -> dict[str, np.ndarray]:
    traced = torch.jit.trace(module, inputs, strict=False, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=specs,
        outputs=[ct.TensorType(name=name) for name in output_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )
    pkg = os.path.join("/tmp", package_name)
    if os.path.exists(pkg):
        import shutil

        shutil.rmtree(pkg)
    mlmodel.save(pkg)
    runtime = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.ALL)
    feed = {spec.name: tensor.detach().cpu().numpy() for spec, tensor in zip(specs, inputs)}
    return runtime.predict(feed)


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
    attn = model.model.layers[layer_idx].self_attn

    hf_cfg = Qwen3_5Config.from_pretrained(model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=layer_idx).half().eval()

    torch.manual_seed(7)
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32)
    x = hf_layer.input_layernorm(hidden).half()
    conv_state = torch.zeros((1, attn.conv_dim, attn.linear_conv_kernel_dim), dtype=torch.float16)

    with torch.no_grad():
        proj_stage = QKVProjStage(attn).eval()
        conv_stage = CausalConvStage(attn, seq_len).eval()
        tmixed, tz, tb, ta = proj_stage(x)
        tconv_out, tnext_conv = conv_stage(tmixed, conv_state)

    proj_out = convert_and_run(
        proj_stage,
        (x,),
        [ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16)],
        ["mixed_qkv", "z", "b", "a"],
        "qwen35_linear_proj_only.mlpackage",
    )
    print("Projection Only Stage")
    print_metric("mixed_qkv", metrics(proj_out["mixed_qkv"], tmixed))
    print_metric("z", metrics(proj_out["z"], tz))
    print_metric("b", metrics(proj_out["b"], tb))
    print_metric("a", metrics(proj_out["a"], ta))

    conv_out = convert_and_run(
        conv_stage,
        (tmixed, conv_state),
        [
            ct.TensorType(name="mixed_qkv", shape=tmixed.shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
        ],
        ["conv_out", "next_conv"],
        "qwen35_linear_causal_conv_only.mlpackage",
    )
    print("Causal Conv Stage")
    print_metric("conv_out", metrics(conv_out["conv_out"], tconv_out))
    print_metric("next_conv", metrics(conv_out["next_conv"], tnext_conv))


if __name__ == "__main__":
    main()
