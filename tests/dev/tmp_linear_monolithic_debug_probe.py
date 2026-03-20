#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import coremltools as ct

from anemll.models.qwen3_5_model import MODEL_DTYPE, Qwen35Config, Qwen35ForCausalLM


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


class MonolithicLinearDebugBlock(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.attn = attn
        self.seq_len = int(seq_len)
        self.hidden_size = int(attn.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = hidden_states.view(1, self.seq_len, self.hidden_size)
        bsz = 1
        seq_len = self.seq_len

        mixed_qkv = self.attn._conv2d_proj(self.attn.in_proj_qkv, hidden_states).transpose(1, 2)
        z = self.attn._conv2d_proj(self.attn.in_proj_z, hidden_states).reshape(
            bsz, seq_len, self.attn.num_v_heads, self.attn.head_v_dim
        )
        b = self.attn._conv2d_proj(self.attn.in_proj_b, hidden_states)
        a = self.attn._conv2d_proj(self.attn.in_proj_a, hidden_states)
        conv_out, next_conv_state = self.attn._causal_conv_update(
            mixed_qkv, conv_state, expected_seq_len=seq_len
        )
        mixed_qkv_after = conv_out.transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv_after, [self.attn.key_dim, self.attn.key_dim, self.attn.value_dim], dim=-1
        )
        query = query.reshape(bsz, seq_len, self.attn.num_k_heads, self.attn.head_k_dim)
        key = key.reshape(bsz, seq_len, self.attn.num_k_heads, self.attn.head_k_dim)
        value = value.reshape(bsz, seq_len, self.attn.num_v_heads, self.attn.head_v_dim)

        beta = b.sigmoid()
        g = -self.attn.A_log.to(MODEL_DTYPE).exp() * torch.nn.functional.softplus(
            a.to(MODEL_DTYPE) + self.attn.dt_bias
        )
        if self.attn.num_v_heads // self.attn.num_k_heads > 1:
            rep = self.attn.num_v_heads // self.attn.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        core, next_recurrent_state = self.attn._recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            recurrent_state=recurrent_state,
            output_final_state=True,
            expected_batch_size=1,
            expected_num_heads=self.attn.num_v_heads,
            expected_seq_len=seq_len,
            expected_k_dim=self.attn.head_k_dim,
            expected_v_dim=self.attn.head_v_dim,
            math_dtype=MODEL_DTYPE,
        )

        core_flat = core.reshape(-1, self.attn.head_v_dim)
        z_flat = z.reshape(-1, self.attn.head_v_dim)
        normed = self.attn.norm(core_flat, z_flat).reshape(bsz, seq_len, self.attn.value_dim)
        out = self.attn.out_proj(normed.permute(0, 2, 1).unsqueeze(2)).squeeze(2).transpose(1, 2)
        return (
            mixed_qkv,
            conv_out,
            query,
            key,
            value,
            beta,
            g,
            core,
            normed,
            out,
            next_conv_state,
            next_recurrent_state.to(torch.float16),
        )


def main() -> None:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B")
    parser.add_argument("--layer-idx", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--pass-pipeline", choices=["default", "cleanup", "empty"], default="default")
    parser.add_argument("--convert-to", choices=["mlprogram", "milinternal"], default="mlprogram")
    parser.add_argument("--dump-dir", default="/tmp/qwen35_linear_monolithic_debug")
    args = parser.parse_args()

    model_path = args.model_path
    layer_idx = args.layer_idx
    seq_len = args.seq_len

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

    block = MonolithicLinearDebugBlock(attn, seq_len).eval()
    with torch.no_grad():
        torch_out = block(x, conv_state, recurrent_state)

    traced = torch.jit.trace(block, (x, conv_state, recurrent_state), strict=False, check_trace=False)
    pass_pipeline = None
    if args.pass_pipeline == "empty":
        pass_pipeline = ct.PassPipeline.EMPTY
    elif args.pass_pipeline == "cleanup":
        pass_pipeline = getattr(ct.PassPipeline, "CLEANUP", None)
        if pass_pipeline is None:
            raise RuntimeError("coremltools does not expose ct.PassPipeline.CLEANUP in this environment")

    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_hidden_states", shape=x.shape, dtype=np.float16),
            ct.TensorType(name="input_conv_state", shape=conv_state.shape, dtype=np.float16),
            ct.TensorType(name="input_recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="dbg_mixed_qkv"),
            ct.TensorType(name="dbg_conv_out"),
            ct.TensorType(name="dbg_query"),
            ct.TensorType(name="dbg_key"),
            ct.TensorType(name="dbg_value"),
            ct.TensorType(name="dbg_beta"),
            ct.TensorType(name="dbg_g"),
            ct.TensorType(name="dbg_core"),
            ct.TensorType(name="dbg_normed"),
            ct.TensorType(name="dbg_attn_out"),
            ct.TensorType(name="dbg_next_conv"),
            ct.TensorType(name="dbg_next_rec"),
        ],
        convert_to=args.convert_to,
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=pass_pipeline,
    )

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    stem = f"monolithic_{args.pass_pipeline}"
    if args.convert_to == "milinternal":
        out_path = dump_dir / f"{stem}.mil.txt"
        out_path.write_text(str(converted))
        print(f"Saved MIL program: {out_path}")
        return

    mlmodel = converted
    pkg = str(dump_dir / f"{stem}.mlpackage")
    if os.path.exists(pkg):
        import shutil

        shutil.rmtree(pkg)
    mlmodel.save(pkg)
    runtime = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.ALL)
    pred = runtime.predict(
        {
            "input_hidden_states": x.detach().cpu().numpy(),
            "input_conv_state": conv_state.detach().cpu().numpy(),
            "input_recurrent_state": recurrent_state.detach().cpu().numpy(),
        }
    )

    names = [
        ("mixed_qkv", "dbg_mixed_qkv"),
        ("conv_out", "dbg_conv_out"),
        ("query", "dbg_query"),
        ("key", "dbg_key"),
        ("value", "dbg_value"),
        ("beta", "dbg_beta"),
        ("g", "dbg_g"),
        ("core", "dbg_core"),
        ("normed", "dbg_normed"),
        ("attn_out", "dbg_attn_out"),
        ("next_conv", "dbg_next_conv"),
        ("next_rec", "dbg_next_rec"),
    ]
    print(f"Monolithic Debug vs Torch ({args.pass_pipeline})")
    for (label, pred_name), ref in zip(names, torch_out):
        print_metric(label, metrics(pred[pred_name], ref))


if __name__ == "__main__":
    main()
