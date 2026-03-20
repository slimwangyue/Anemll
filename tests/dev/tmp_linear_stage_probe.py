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
        f"{name:16s} max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
        f"rmse={m['rmse']:.6e} cosine={m['cosine']:.8f}"
    )


class ProjConvStage(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.attn = attn
        self.seq_len = int(seq_len)

    def forward(
        self, hidden_states: torch.Tensor, conv_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_qkv = self.attn._conv2d_proj(self.attn.in_proj_qkv, hidden_states).transpose(1, 2)
        z = self.attn._conv2d_proj(self.attn.in_proj_z, hidden_states).reshape(
            1, self.seq_len, self.attn.num_v_heads, self.attn.head_v_dim
        )
        b = self.attn._conv2d_proj(self.attn.in_proj_b, hidden_states)
        a = self.attn._conv2d_proj(self.attn.in_proj_a, hidden_states)
        conv_out, next_conv = self.attn._causal_conv_update(mixed_qkv, conv_state, expected_seq_len=self.seq_len)
        mixed_qkv = conv_out.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.attn.key_dim, self.attn.key_dim, self.attn.value_dim], dim=-1
        )
        query = query.reshape(1, self.seq_len, self.attn.num_k_heads, self.attn.head_k_dim)
        key = key.reshape(1, self.seq_len, self.attn.num_k_heads, self.attn.head_k_dim)
        value = value.reshape(1, self.seq_len, self.attn.num_v_heads, self.attn.head_v_dim)
        beta = b.sigmoid()
        g = -self.attn.A_log.to(torch.float16).exp() * torch.nn.functional.softplus(
            a.to(torch.float16) + self.attn.dt_bias
        )
        if self.attn.num_v_heads // self.attn.num_k_heads > 1:
            rep = self.attn.num_v_heads // self.attn.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        return query, key, value, beta, g, z, next_conv


class RecurrentCoreStage(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.attn = attn
        self.seq_len = int(seq_len)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attn._recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            recurrent_state=recurrent_state,
            output_final_state=True,
            expected_batch_size=1,
            expected_num_heads=self.attn.num_v_heads,
            expected_seq_len=self.seq_len,
            expected_k_dim=self.attn.head_k_dim,
            expected_v_dim=self.attn.head_v_dim,
            math_dtype=torch.float16,
        )


class NormOutStage(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.attn = attn
        self.seq_len = int(seq_len)

    def forward(self, core: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        core = core.reshape(-1, self.attn.head_v_dim)
        zf = z.reshape(-1, self.attn.head_v_dim)
        core = self.attn.norm(core, zf).reshape(1, self.seq_len, self.attn.value_dim)
        return self.attn.out_proj(core.permute(0, 2, 1).unsqueeze(2)).squeeze(2).transpose(1, 2)


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
    recurrent_state = torch.zeros((1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim), dtype=torch.float16)

    with torch.no_grad():
        proj_stage = ProjConvStage(attn, seq_len).eval()
        core_stage = RecurrentCoreStage(attn, seq_len).eval()
        norm_stage = NormOutStage(attn, seq_len).eval()

        tq, tk, tv, tb, tg, tz, tnext_conv = proj_stage(x, conv_state)
        tcore, tnext_rec = core_stage(tq, tk, tv, tb, tg, recurrent_state)
        tout = norm_stage(tcore, tz)

    proj_out = convert_and_run(
        proj_stage,
        (x, conv_state),
        [
            ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
        ],
        ["query", "key", "value", "beta", "g", "z", "next_conv"],
        "qwen35_linear_stage_proj.mlpackage",
    )
    print("ProjConv Stage")
    print_metric("query", metrics(proj_out["query"], tq))
    print_metric("key", metrics(proj_out["key"], tk))
    print_metric("value", metrics(proj_out["value"], tv))
    print_metric("beta", metrics(proj_out["beta"], tb))
    print_metric("g", metrics(proj_out["g"], tg))
    print_metric("z", metrics(proj_out["z"], tz))
    print_metric("next_conv", metrics(proj_out["next_conv"], tnext_conv))

    core_out = convert_and_run(
        core_stage,
        (tq, tk, tv, tb, tg, recurrent_state),
        [
            ct.TensorType(name="query", shape=tq.shape, dtype=np.float16),
            ct.TensorType(name="key", shape=tk.shape, dtype=np.float16),
            ct.TensorType(name="value", shape=tv.shape, dtype=np.float16),
            ct.TensorType(name="beta", shape=tb.shape, dtype=np.float16),
            ct.TensorType(name="g", shape=tg.shape, dtype=np.float16),
            ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
        ],
        ["core", "next_rec"],
        "qwen35_linear_stage_core.mlpackage",
    )
    print("Recurrent Core Stage")
    print_metric("core", metrics(core_out["core"], tcore))
    print_metric("next_rec", metrics(core_out["next_rec"], tnext_rec))

    norm_out = convert_and_run(
        norm_stage,
        (tcore, tz),
        [
            ct.TensorType(name="core", shape=tcore.shape, dtype=np.float16),
            ct.TensorType(name="z", shape=tz.shape, dtype=np.float16),
        ],
        ["attn_out"],
        "qwen35_linear_stage_norm.mlpackage",
    )
    print("NormOut Stage")
    print_metric("attn_out", metrics(norm_out["attn_out"], tout))


if __name__ == "__main__":
    main()
