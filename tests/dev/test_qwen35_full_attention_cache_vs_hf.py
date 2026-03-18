#!/usr/bin/env python3
"""Cache-path parity test: ANEMLL Qwen3.5 full_attention vs HF (prefill/decode)."""

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


def _hf_prefill_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    m = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype)
    for i in range(seq_len):
        if i + 1 < seq_len:
            m[:, :, i, i + 1 :] = float("-inf")
    return m


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
        f"{name:10s} max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
        f"rmse={m['rmse']:.6e} cosine={m['cosine']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    if cfg.text_config.layer_types[args.layer_idx] != "full_attention":
        raise ValueError(
            f"layer {args.layer_idx} is {cfg.text_config.layer_types[args.layer_idx]}, expected full_attention"
        )

    anemll = Qwen35ForCausalLM(cfg).half().eval()
    if not anemll.load_pretrained_weights(args.model_path):
        raise RuntimeError("ANEMLL loader failed.")
    our_attn = anemll.model.layers[args.layer_idx].self_attn

    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    hf_cfg = Qwen3_5Config.from_pretrained(args.model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=args.layer_idx).half().eval()
    hf_rotary = Qwen3_5TextRotaryEmbedding(hf_cfg).half()

    idx = _load_index(args.model_path)
    weight_map = idx["weight_map"]
    prefix = f"model.language_model.layers.{args.layer_idx}."
    layer_keys = [k for k in weight_map.keys() if k.startswith(prefix)]
    tensors = _read_tensors(args.model_path, layer_keys, weight_map)
    hf_state = {k[len(prefix) :]: v for k, v in tensors.items()}
    missing, unexpected = hf_layer.load_state_dict(hf_state, strict=False)
    if missing:
        raise RuntimeError(f"HF missing keys: {missing}")
    if unexpected:
        print(f"HF unexpected keys: {unexpected}")

    seq_len = args.seq_len
    hidden_prefill = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32)
    hidden_decode = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        x_prefill = hf_layer.input_layernorm(hidden_prefill).half()
        x_decode = hf_layer.input_layernorm(hidden_decode).half()

        cache = DynamicCache()
        hf_prefill, _ = hf_layer.self_attn(
            hidden_states=x_prefill,
            position_embeddings=hf_rotary(x_prefill, torch.arange(seq_len).unsqueeze(0)),
            attention_mask=_hf_prefill_mask(seq_len, torch.float16),
            past_key_values=cache,
        )
        hf_decode, _ = hf_layer.self_attn(
            hidden_states=x_decode,
            position_embeddings=hf_rotary(x_decode, torch.tensor([[seq_len]], dtype=torch.long)),
            attention_mask=torch.zeros((1, 1, 1, seq_len + 1), dtype=torch.float16),
            past_key_values=cache,
        )

        kvh, st, hd = cfg.num_key_value_heads, cfg.state_length, cfg.head_dim
        k_cache = torch.zeros((kvh, st, hd), dtype=torch.float16)
        v_cache = torch.zeros((kvh, st, hd), dtype=torch.float16)
        q, k, v, g = our_attn.get_new_kv_cache_prefill(x_prefill, torch.arange(seq_len))
        k_cache[:, :seq_len, :] = k.squeeze(0)
        v_cache[:, :seq_len, :] = v.squeeze(0)

        mask_prefill = torch.full((1, 1, seq_len, st), float("-inf"), dtype=torch.float16)
        for i in range(seq_len):
            mask_prefill[:, :, i, : i + 1] = 0
        our_prefill = our_attn.forward_prefill(
            hidden_states=x_prefill,
            query_states=q,
            kv_cache_layer=(k_cache, v_cache),
            causal_mask=mask_prefill,
            gate=g,
        )

        q1, k1, v1, g1 = our_attn.get_new_kv_cache(x_decode, seq_len)
        k_cache[:, seq_len : seq_len + 1, :] = k1.squeeze(0)
        v_cache[:, seq_len : seq_len + 1, :] = v1.squeeze(0)
        mask_decode = torch.full((1, 1, 1, st), float("-inf"), dtype=torch.float16)
        mask_decode[:, :, :, : seq_len + 1] = 0
        our_decode = our_attn.forward_regular(
            hidden_states=x_decode,
            query_states=q1,
            kv_cache_layer=(k_cache, v_cache),
            causal_mask=mask_decode,
            gate=g1,
        )

    print("Qwen3.5 Full-Attention Cache Parity")
    print(f"layer_idx={args.layer_idx} seq_len={seq_len} dtype=fp16")
    _print_metric("prefill", _metrics(our_prefill, hf_prefill))
    _print_metric("decode", _metrics(our_decode, hf_decode))


if __name__ == "__main__":
    main()

