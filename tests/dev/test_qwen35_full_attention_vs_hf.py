#!/usr/bin/env python3
"""Block-level parity test: ANEMLL Qwen3.5 full_attention vs HF reference.

Usage:
  conda run -n qwen_coreml python tests/dev/test_qwen35_full_attention_vs_hf.py \
    --model-path /home/yue/local_llm/models/Qwen__Qwen3.5-4B --layer-idx 3 --seq-len 16
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

# Ensure local package import works when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM


def _load_index(model_path: str) -> Dict:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        return json.load(f)


def _read_tensors(model_path: str, keys: List[str], weight_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    file_to_keys: Dict[str, List[str]] = {}
    for key in keys:
        if key not in weight_map:
            continue
        file_to_keys.setdefault(weight_map[key], []).append(key)

    out: Dict[str, torch.Tensor] = {}
    for shard, shard_keys in file_to_keys.items():
        full = os.path.join(model_path, shard)
        with safe_open(full, framework="pt", device="cpu") as f:
            for key in shard_keys:
                out[key] = f.get_tensor(key)
    return out


def _causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    for i in range(seq_len):
        if i + 1 < seq_len:
            mask[:, :, i, i + 1 :] = float("-inf")
    return mask


def _metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a32 = a.float().reshape(-1)
    b32 = b.float().reshape(-1)
    diff = (a32 - b32).abs()
    cosine = torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0)).item()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a32 - b32) ** 2)).item()),
        "cosine": float(cosine),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF checkpoint path")
    parser.add_argument("--layer-idx", type=int, default=3, help="full_attention layer index to compare")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=["fp16"], default="fp16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    if cfg.text_config.layer_types[args.layer_idx] != "full_attention":
        raise ValueError(
            f"layer {args.layer_idx} is {cfg.text_config.layer_types[args.layer_idx]}, not full_attention."
        )

    # Build ANEMLL model and load weights via project loader (already handles key remapping).
    anemll_model = Qwen35ForCausalLM(cfg)
    if not anemll_model.load_pretrained_weights(args.model_path):
        raise RuntimeError("ANEMLL loader failed.")

    # Build HF decoder layer and load only that layer's weights.
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5TextRotaryEmbedding,
    )

    hf_cfg = Qwen3_5Config.from_pretrained(args.model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=args.layer_idx)
    hf_rotary = Qwen3_5TextRotaryEmbedding(hf_cfg)

    idx = _load_index(args.model_path)
    weight_map = idx.get("weight_map", {})
    prefix = f"model.language_model.layers.{args.layer_idx}."
    layer_keys = [k for k in weight_map.keys() if k.startswith(prefix)]
    tensors = _read_tensors(args.model_path, layer_keys, weight_map)
    hf_state = {k[len(prefix) :]: v for k, v in tensors.items()}
    missing, unexpected = hf_layer.load_state_dict(hf_state, strict=False)
    if missing:
        raise RuntimeError(f"HF layer missing keys: {missing}")
    if unexpected:
        print(f"HF layer unexpected keys: {unexpected}")

    # Inputs
    seq_len = args.seq_len
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32)
    position_ids_2d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_ids_1d = position_ids_2d.squeeze(0)
    mask = _causal_mask(seq_len, dtype=torch.float32, device=hidden.device)

    # Use HF layernorm output as common attention input to isolate attention parity.
    with torch.no_grad():
        x = hf_layer.input_layernorm(hidden)

    # ANEMLL full-attention currently uses internal fp16 casts; keep both paths in fp16.
    x = x.half()
    mask = mask.half()
    anemll_model = anemll_model.half()
    hf_layer = hf_layer.half()
    hf_rotary = hf_rotary.half()

    anemll_attn = anemll_model.model.layers[args.layer_idx].self_attn

    with torch.no_grad():
        # HF attention path
        pos_emb = hf_rotary(x, position_ids_2d)
        hf_out, _ = hf_layer.self_attn(
            hidden_states=x,
            position_embeddings=pos_emb,
            attention_mask=mask,
            past_key_values=None,
        )

        # ANEMLL attention path
        anemll_out = anemll_attn(
            hidden_states=x,
            causal_mask=mask,
            position_ids=position_ids_1d,
        )

    m = _metrics(anemll_out, hf_out)
    print("Qwen3.5 Full-Attention Parity")
    print(f"layer_idx={args.layer_idx} seq_len={seq_len} dtype={args.dtype}")
    print(f"max_abs={m['max_abs']:.6e}")
    print(f"mean_abs={m['mean_abs']:.6e}")
    print(f"rmse={m['rmse']:.6e}")
    print(f"cosine={m['cosine']:.8f}")


if __name__ == "__main__":
    main()
