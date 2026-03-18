#!/usr/bin/env python3
"""Layer-level parity: ANEMLL Qwen3.5 linear attention vs HF reference.

This validates:
1) no-cache forward parity
2) cache prefill parity
3) cache decode parity

For decode parity on recurrent path, use a layer index that is the last
linear_attention layer in the architecture (for Qwen3.5-4B this is layer 30).
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


def _print_metric(name: str, m: Dict[str, float]) -> None:
    print(
        f"{name:12s} max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
        f"rmse={m['rmse']:.6e} cosine={m['cosine']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer-idx", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    if cfg.text_config.layer_types[args.layer_idx] != "linear_attention":
        raise ValueError(
            f"layer {args.layer_idx} is {cfg.text_config.layer_types[args.layer_idx]}, expected linear_attention"
        )

    anemll = Qwen35ForCausalLM(cfg).half()
    if not anemll.load_pretrained_weights(args.model_path):
        raise RuntimeError("ANEMLL loader failed.")
    our_layer = anemll.model.layers[args.layer_idx]
    our_attn = our_layer.self_attn

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5DynamicCache

    hf_cfg = Qwen3_5Config.from_pretrained(args.model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=args.layer_idx).half().eval()
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

    hidden = torch.randn(1, args.seq_len, cfg.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        x = hf_layer.input_layernorm(hidden).half()
        x1 = x[:, -1:, :]

        # No-cache forward
        hf_out = hf_layer.linear_attn(hidden_states=x, cache_params=None, attention_mask=None)
        our_out = our_attn(hidden_states=x, causal_mask=None, position_ids=torch.arange(args.seq_len))

        # Cache prefill + decode
        hf_cache = Qwen3_5DynamicCache(hf_cfg)
        hf_prefill = hf_layer.linear_attn(hidden_states=x, cache_params=hf_cache, attention_mask=None)
        hf_decode = hf_layer.linear_attn(hidden_states=x1, cache_params=hf_cache, attention_mask=None)

        conv_state = torch.zeros((1, our_attn.conv_dim, our_attn.linear_conv_kernel_dim), dtype=torch.float16)
        rec_state = torch.zeros((1, our_attn.num_v_heads, our_attn.head_k_dim, our_attn.head_v_dim), dtype=torch.float32)
        our_prefill, conv_state, rec_state = our_attn.forward_prefill(
            hidden_states=x,
            conv_state=conv_state,
            recurrent_state=rec_state,
            has_previous_state=False,
        )
        our_decode, _, _ = our_attn.forward_regular(
            hidden_states=x1,
            conv_state=conv_state,
            recurrent_state=rec_state,
            has_previous_state=True,
        )

    print("Qwen3.5 Linear-Attention Parity")
    print(f"layer_idx={args.layer_idx} seq_len={args.seq_len} dtype=fp16")
    _print_metric("forward", _metrics(our_out, hf_out))
    _print_metric("prefill", _metrics(our_prefill, hf_prefill))
    _print_metric("decode", _metrics(our_decode, hf_decode))


if __name__ == "__main__":
    main()

