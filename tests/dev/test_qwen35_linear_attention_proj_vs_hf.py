#!/usr/bin/env python3
"""Projection-level parity test: ANEMLL Qwen3.5 linear_attention vs HF reference.

This compares sub-projection outputs (not full linear-attention state/update math):
- in_proj_qkv
- in_proj_a
- in_proj_b
- in_proj_z
- out_proj
- conv (applied on in_proj_qkv output)

Usage:
  conda run -n qwen_coreml python tests/dev/test_qwen35_linear_attention_proj_vs_hf.py \
    --model-path /home/yue/local_llm/models/Qwen__Qwen3.5-4B --layer-idx 0 --seq-len 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM


def _load_index(model_path: str) -> Dict:
    with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
        return json.load(f)


def _read_tensors(model_path: str, keys: List[str], weight_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    file_to_keys: Dict[str, List[str]] = {}
    for key in keys:
        if key in weight_map:
            file_to_keys.setdefault(weight_map[key], []).append(key)
    out: Dict[str, torch.Tensor] = {}
    for shard, shard_keys in file_to_keys.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for key in shard_keys:
                out[key] = f.get_tensor(key)
    return out


def _metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
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


def _conv2d_proj(conv: torch.nn.Conv2d, x_bsh: torch.Tensor) -> torch.Tensor:
    y = conv(x_bsh.permute(0, 2, 1).unsqueeze(2))
    return y.squeeze(2).transpose(1, 2)


def _print_metric(name: str, m: Dict[str, float]) -> None:
    print(
        f"{name:12s} max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
        f"rmse={m['rmse']:.6e} cosine={m['cosine']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer-idx", type=int, default=0, help="must be a linear_attention layer")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    if cfg.text_config.layer_types[args.layer_idx] != "linear_attention":
        raise ValueError(
            f"layer {args.layer_idx} is {cfg.text_config.layer_types[args.layer_idx]}, expected linear_attention"
        )

    # ANEMLL path
    anemll = Qwen35ForCausalLM(cfg)
    if not anemll.load_pretrained_weights(args.model_path):
        raise RuntimeError("ANEMLL loader failed.")
    our_attn = anemll.model.layers[args.layer_idx].self_attn.half().eval()

    # HF layer path
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    hf_cfg = Qwen3_5Config.from_pretrained(args.model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=args.layer_idx)
    idx = _load_index(args.model_path)
    weight_map = idx["weight_map"]
    prefix = f"model.language_model.layers.{args.layer_idx}."
    layer_keys = [k for k in weight_map.keys() if k.startswith(prefix)]
    tensors = _read_tensors(args.model_path, layer_keys, weight_map)
    hf_state = {k[len(prefix) :]: v for k, v in tensors.items()}
    missing, unexpected = hf_layer.load_state_dict(hf_state, strict=False)
    if missing:
        raise RuntimeError(f"HF layer missing keys: {missing}")
    if unexpected:
        print(f"HF unexpected keys: {unexpected}")
    hf_attn = hf_layer.linear_attn.half().eval()

    # Shared normalized input for projection checks.
    hidden = torch.randn(1, args.seq_len, cfg.hidden_size, dtype=torch.float16)
    x = hf_layer.input_layernorm(hidden.float()).half()

    with torch.no_grad():
        # Projections
        ours_qkv = _conv2d_proj(our_attn.in_proj_qkv, x)
        ours_a = _conv2d_proj(our_attn.in_proj_a, x)
        ours_b = _conv2d_proj(our_attn.in_proj_b, x)
        ours_z = _conv2d_proj(our_attn.in_proj_z, x)

        hf_qkv = hf_attn.in_proj_qkv(x)
        hf_a = hf_attn.in_proj_a(x)
        hf_b = hf_attn.in_proj_b(x)
        hf_z = hf_attn.in_proj_z(x)

        # out_proj expects state_dim input.
        state = torch.randn(1, args.seq_len, our_attn.state_dim, dtype=torch.float16)
        ours_out = _conv2d_proj(our_attn.out_proj, state)
        hf_out = hf_attn.out_proj(state)

        # Conv parity on functional prefill path:
        # - ANEMLL: _causal_conv_update with zero-initial state
        # - HF: silu(conv1d(...)) and crop to input seq_len
        seq_len = x.shape[1]
        zero_state = torch.zeros(
            (x.shape[0], our_attn.conv_dim, our_attn.linear_conv_kernel_dim), dtype=torch.float16
        )
        ours_conv, _ = our_attn._causal_conv_update(ours_qkv.transpose(1, 2), zero_state)
        hf_conv = torch.nn.functional.silu(hf_attn.conv1d(hf_qkv.transpose(1, 2))[:, :, :seq_len])

    print("Qwen3.5 Linear-Attention Projection Parity")
    print(f"layer_idx={args.layer_idx} seq_len={args.seq_len} dtype=fp16")
    _print_metric("in_proj_qkv", _metrics(ours_qkv, hf_qkv))
    _print_metric("in_proj_a", _metrics(ours_a, hf_a))
    _print_metric("in_proj_b", _metrics(ours_b, hf_b))
    _print_metric("in_proj_z", _metrics(ours_z, hf_z))
    _print_metric("out_proj", _metrics(ours_out, hf_out))
    _print_metric("conv", _metrics(ours_conv, hf_conv))


if __name__ == "__main__":
    main()
