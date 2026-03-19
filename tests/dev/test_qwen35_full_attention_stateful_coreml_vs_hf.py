#!/usr/bin/env python3
"""Stateful CoreML parity for Qwen3.5 full-attention block (prefill/decode).

This script does three checks:
1) PyTorch block parity vs HF for prefill/decode.
2) Stateful CoreML export for full-attention block (registers k/v cache as states).
3) CoreML prediction parity vs PyTorch for prefill/decode when runtime is available.

Note: On Linux, coremltools often lacks runtime proxy (`libcoremlpython`), so predict
may be skipped while export/state registration is still validated.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM, MODEL_DTYPE


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
        f"{name:22s} max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
        f"rmse={m['rmse']:.6e} cosine={m['cosine']:.8f}"
    )


def _parse_int_csv(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _make_hidden_states(hidden_size: int, seq_len: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    hidden_prefill = torch.randn(1, seq_len, hidden_size, dtype=torch.float32, generator=gen)
    hidden_decode = torch.randn(1, 1, hidden_size, dtype=torch.float32, generator=gen)
    return hidden_prefill, hidden_decode


def _pad_chunk_hidden(hidden_chunk: torch.Tensor, input_seq_len: int) -> torch.Tensor:
    padded = torch.zeros(
        (1, input_seq_len, hidden_chunk.shape[-1]),
        dtype=hidden_chunk.dtype,
        device=hidden_chunk.device,
    )
    padded[:, : hidden_chunk.shape[1], :] = hidden_chunk
    return padded


def _chunk_slices(total_len: int, input_seq_len: int) -> List[Tuple[int, int]]:
    return [(start, min(start + input_seq_len, total_len)) for start in range(0, total_len, input_seq_len)]


def _hf_chunk_attention_mask(start: int, chunk_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((1, 1, chunk_len, start + chunk_len), dtype=dtype)
    mask[:, :, :, start:] = torch.triu(
        torch.full((chunk_len, chunk_len), float("-inf"), dtype=dtype),
        diagonal=1,
    )
    return mask


class _StatefulFullAttentionBase(torch.nn.Module):
    """Shared state buffers for fixed-shape full-attention wrappers."""

    def __init__(self, attn: torch.nn.Module, cfg: Qwen35Config):
        super().__init__()
        self.attn = attn
        self.num_kv_heads = cfg.num_key_value_heads
        self.state_length = cfg.state_length
        self.head_dim = cfg.head_dim
        self.register_buffer(
            "k_cache",
            torch.zeros((self.num_kv_heads, self.state_length, self.head_dim), dtype=MODEL_DTYPE),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros((self.num_kv_heads, self.state_length, self.head_dim), dtype=MODEL_DTYPE),
        )

    def _build_fixed_cache_mask(self, q_len: int, current_pos: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        q_idx = torch.arange(q_len, device=device).unsqueeze(-1)
        k_idx = torch.arange(self.state_length, device=device).unsqueeze(0)
        allowed = k_idx <= (current_pos + q_idx)
        z = torch.zeros((), dtype=dtype, device=device)
        ninf = torch.full((), float("-inf"), dtype=dtype, device=device)
        return torch.where(allowed, z, ninf).unsqueeze(0).unsqueeze(0)

    def _build_position_mask(self, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        q_pos = position_ids.to(device=device, dtype=torch.long).reshape(-1, 1)
        k_idx = torch.arange(self.state_length, device=device).reshape(1, -1)
        allowed = k_idx <= q_pos
        z = torch.zeros((), dtype=dtype, device=device)
        ninf = torch.full((), float("-inf"), dtype=dtype, device=device)
        return torch.where(allowed, z, ninf).unsqueeze(0).unsqueeze(0)


class StatefulFullAttentionPrefillBlock(_StatefulFullAttentionBase):
    """Stateful prefill-only wrapper with a dedicated ANE-friendly forward graph."""

    def __init__(self, attn: torch.nn.Module, cfg: Qwen35Config, prefill_seq_len: int):
        super().__init__(attn, cfg)
        self.prefill_seq_len = int(prefill_seq_len)
        self.hidden_size = int(cfg.hidden_size)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # hidden_states: [1, S, H], position_ids: [S]
        hidden_states = hidden_states.view(1, self.prefill_seq_len, self.hidden_size)
        position_ids = position_ids.view(self.prefill_seq_len)
        q, k, v, gate = self.attn.get_new_kv_cache_prefill(hidden_states, position_ids)
        self.k_cache[:, : self.prefill_seq_len, :] = k.squeeze(0)
        self.v_cache[:, : self.prefill_seq_len, :] = v.squeeze(0)
        mask = self._build_position_mask(position_ids, hidden_states.dtype, hidden_states.device)
        return self.attn.forward_prefill(
            hidden_states=hidden_states,
            query_states=q,
            kv_cache_layer=(self.k_cache, self.v_cache),
            causal_mask=mask,
            gate=gate,
        )


class StatefulFullAttentionChunkPrefillBlock(_StatefulFullAttentionBase):
    """Stateful chunked prefill block with fixed input width and fixed cache position."""

    def __init__(self, attn: torch.nn.Module, cfg: Qwen35Config, input_seq_len: int, current_pos: int):
        super().__init__(attn, cfg)
        self.input_seq_len = int(input_seq_len)
        self.current_pos = int(current_pos)
        self.hidden_size = int(cfg.hidden_size)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(1, self.input_seq_len, self.hidden_size)
        position_ids = position_ids.view(self.input_seq_len)
        q, k, v, gate = self.attn.get_new_kv_cache_prefill(hidden_states, position_ids)
        chunk_end = self.current_pos + self.input_seq_len
        self.k_cache[:, self.current_pos : chunk_end, :] = k.squeeze(0)
        self.v_cache[:, self.current_pos : chunk_end, :] = v.squeeze(0)
        mask = self._build_fixed_cache_mask(
            q_len=self.input_seq_len,
            current_pos=self.current_pos,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return self.attn.forward_prefill(
            hidden_states=hidden_states,
            query_states=q,
            kv_cache_layer=(self.k_cache, self.v_cache),
            causal_mask=mask,
            gate=gate,
        )


class StatefulFullAttentionDecodeBlock(_StatefulFullAttentionBase):
    """Stateful decode-only wrapper with a fixed single-token forward graph."""

    def __init__(self, attn: torch.nn.Module, cfg: Qwen35Config, current_pos: int):
        super().__init__(attn, cfg)
        self.current_pos = int(current_pos)
        self.register_buffer(
            "current_pos_tensor",
            torch.tensor([current_pos], dtype=torch.long),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [1, 1, H]
        q, k, v, gate = self.attn.get_new_kv_cache(hidden_states, self.current_pos_tensor)
        self.k_cache[:, self.current_pos : self.current_pos + 1, :] = k.squeeze(0)
        self.v_cache[:, self.current_pos : self.current_pos + 1, :] = v.squeeze(0)
        mask = self._build_fixed_cache_mask(1, self.current_pos, hidden_states.dtype, hidden_states.device)
        return self.attn.forward_regular(
            hidden_states=hidden_states,
            query_states=q,
            kv_cache_layer=(self.k_cache, self.v_cache),
            causal_mask=mask,
            gate=gate,
        )


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _run_hf_parity(
    model_path: str,
    layer_idx: int,
    cfg: Qwen35Config,
    our_attn: torch.nn.Module,
    hidden_prefill: torch.Tensor,
    hidden_decode: torch.Tensor,
    *,
    title: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    hf_cfg = Qwen3_5Config.from_pretrained(model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=layer_idx).half().eval()
    hf_rotary = Qwen3_5TextRotaryEmbedding(hf_cfg).half()

    idx = _load_index(model_path)
    weight_map = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    layer_keys = [k for k in weight_map.keys() if k.startswith(prefix)]
    tensors = _read_tensors(model_path, layer_keys, weight_map)
    hf_state = {k[len(prefix) :]: v for k, v in tensors.items()}
    missing, unexpected = hf_layer.load_state_dict(hf_state, strict=False)
    if missing:
        raise RuntimeError(f"HF missing keys: {missing}")
    if unexpected:
        print(f"HF unexpected keys: {unexpected}")

    seq_len = hidden_prefill.shape[1]
    with torch.no_grad():
        x_prefill = hf_layer.input_layernorm(hidden_prefill).half()
        x_decode = hf_layer.input_layernorm(hidden_decode).half()

        cache = DynamicCache()
        hf_prefill, _ = hf_layer.self_attn(
            hidden_states=x_prefill,
            position_embeddings=hf_rotary(x_prefill, torch.arange(seq_len).unsqueeze(0)),
            attention_mask=torch.triu(
                torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float16), diagonal=1
            ),
            past_key_values=cache,
        )
        hf_decode, _ = hf_layer.self_attn(
            hidden_states=x_decode,
            position_embeddings=hf_rotary(x_decode, torch.tensor([[seq_len]], dtype=torch.long)),
            attention_mask=torch.zeros((1, 1, 1, seq_len + 1), dtype=torch.float16),
            past_key_values=cache,
        )

        prefill_block = StatefulFullAttentionPrefillBlock(our_attn, cfg, prefill_seq_len=seq_len).half().eval()
        decode_block = StatefulFullAttentionDecodeBlock(our_attn, cfg, current_pos=seq_len).half().eval()
        our_prefill = prefill_block(
            hidden_states=x_prefill,
            position_ids=torch.arange(seq_len, dtype=torch.long),
        )
        decode_block.k_cache.copy_(prefill_block.k_cache)
        decode_block.v_cache.copy_(prefill_block.v_cache)
        our_decode = decode_block(hidden_states=x_decode)

    print(title or "PyTorch Full-Attn Block Parity vs HF")
    _print_metric("prefill_torch_vs_hf", _metrics(our_prefill, hf_prefill))
    _print_metric("decode_torch_vs_hf", _metrics(our_decode, hf_decode))
    return our_prefill, our_decode


def _run_hf_prefill_only_parity(
    model_path: str,
    layer_idx: int,
    cfg: Qwen35Config,
    our_attn: torch.nn.Module,
    hidden_prefill: torch.Tensor,
    *,
    title: str,
) -> torch.Tensor:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    hf_cfg = Qwen3_5Config.from_pretrained(model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=layer_idx).half().eval()
    hf_rotary = Qwen3_5TextRotaryEmbedding(hf_cfg).half()

    idx = _load_index(model_path)
    weight_map = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    layer_keys = [k for k in weight_map.keys() if k.startswith(prefix)]
    tensors = _read_tensors(model_path, layer_keys, weight_map)
    hf_state = {k[len(prefix) :]: v for k, v in tensors.items()}
    missing, unexpected = hf_layer.load_state_dict(hf_state, strict=False)
    if missing:
        raise RuntimeError(f"HF missing keys: {missing}")
    if unexpected:
        print(f"HF unexpected keys: {unexpected}")

    seq_len = hidden_prefill.shape[1]
    with torch.no_grad():
        x_prefill = hf_layer.input_layernorm(hidden_prefill).half()
        hf_prefill, _ = hf_layer.self_attn(
            hidden_states=x_prefill,
            position_embeddings=hf_rotary(x_prefill, torch.arange(seq_len).unsqueeze(0)),
            attention_mask=torch.triu(
                torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float16), diagonal=1
            ),
            past_key_values=None,
        )

        prefill_block = StatefulFullAttentionPrefillBlock(our_attn, cfg, prefill_seq_len=seq_len).half().eval()
        our_prefill = prefill_block(
            hidden_states=x_prefill,
            position_ids=torch.arange(seq_len, dtype=torch.long),
        )

    print(title)
    _print_metric("prefill_torch_vs_hf", _metrics(our_prefill, hf_prefill))
    return our_prefill


def _load_hf_attention_layer(model_path: str, layer_idx: int):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    hf_cfg = Qwen3_5Config.from_pretrained(model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=layer_idx).half().eval()
    hf_rotary = Qwen3_5TextRotaryEmbedding(hf_cfg).half()

    idx = _load_index(model_path)
    weight_map = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    layer_keys = [k for k in weight_map.keys() if k.startswith(prefix)]
    tensors = _read_tensors(model_path, layer_keys, weight_map)
    hf_state = {k[len(prefix) :]: v for k, v in tensors.items()}
    missing, unexpected = hf_layer.load_state_dict(hf_state, strict=False)
    if missing:
        raise RuntimeError(f"HF missing keys: {missing}")
    if unexpected:
        print(f"HF unexpected keys: {unexpected}")
    return hf_layer, hf_rotary


def _run_torch_chunked_prompt_parity(
    hf_layer: torch.nn.Module,
    hf_rotary: torch.nn.Module,
    cfg: Qwen35Config,
    our_attn: torch.nn.Module,
    hidden_prefill: torch.Tensor,
    hidden_decode: torch.Tensor,
    input_seq_len: int,
    *,
    title: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from transformers.cache_utils import DynamicCache

    prompt_len = hidden_prefill.shape[1]
    if prompt_len >= cfg.state_length:
        raise ValueError(f"prompt_len {prompt_len} must be smaller than state_length {cfg.state_length}")

    with torch.no_grad():
        x_prefill = hf_layer.input_layernorm(hidden_prefill).half()
        x_decode = hf_layer.input_layernorm(hidden_decode).half()

        hf_cache = DynamicCache()
        hf_prefill_chunks: List[torch.Tensor] = []
        for start, end in _chunk_slices(prompt_len, input_seq_len):
            chunk = x_prefill[:, start:end, :]
            chunk_len = end - start
            position_ids = torch.arange(start, end, dtype=torch.long).unsqueeze(0)
            hf_chunk, _ = hf_layer.self_attn(
                hidden_states=chunk,
                position_embeddings=hf_rotary(chunk, position_ids),
                attention_mask=_hf_chunk_attention_mask(start, chunk_len, dtype=torch.float16),
                past_key_values=hf_cache,
            )
            hf_prefill_chunks.append(hf_chunk)
        hf_prefill = torch.cat(hf_prefill_chunks, dim=1)
        hf_decode, _ = hf_layer.self_attn(
            hidden_states=x_decode,
            position_embeddings=hf_rotary(x_decode, torch.tensor([[prompt_len]], dtype=torch.long)),
            attention_mask=torch.zeros((1, 1, 1, prompt_len + 1), dtype=torch.float16),
            past_key_values=hf_cache,
        )

        shared_k = torch.zeros((cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE)
        shared_v = torch.zeros((cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE)
        our_prefill_chunks: List[torch.Tensor] = []
        for start, end in _chunk_slices(prompt_len, input_seq_len):
            chunk = x_prefill[:, start:end, :]
            chunk_len = end - start
            padded_chunk = _pad_chunk_hidden(chunk, input_seq_len)
            position_ids = torch.arange(start, start + input_seq_len, dtype=torch.long)
            prefill_block = StatefulFullAttentionChunkPrefillBlock(
                our_attn, cfg, input_seq_len=input_seq_len, current_pos=start
            ).half().eval()
            prefill_block.k_cache.copy_(shared_k)
            prefill_block.v_cache.copy_(shared_v)
            chunk_out = prefill_block(
                hidden_states=padded_chunk,
                position_ids=position_ids,
            )
            shared_k.copy_(prefill_block.k_cache)
            shared_v.copy_(prefill_block.v_cache)
            our_prefill_chunks.append(chunk_out[:, :chunk_len, :])

        our_prefill = torch.cat(our_prefill_chunks, dim=1)
        decode_block = StatefulFullAttentionDecodeBlock(our_attn, cfg, current_pos=prompt_len).half().eval()
        decode_block.k_cache.copy_(shared_k)
        decode_block.v_cache.copy_(shared_v)
        our_decode = decode_block(hidden_states=x_decode)

    print(title)
    _print_metric("prefill_torch_vs_hf", _metrics(our_prefill, hf_prefill))
    _print_metric("decode_torch_vs_hf", _metrics(our_decode, hf_decode))
    return our_prefill, our_decode


def _load_our_attn(model_path: str, cfg: Qwen35Config, layer_idx: int) -> torch.nn.Module:
    model = Qwen35ForCausalLM(cfg).half().eval()
    if not model.load_pretrained_weights(model_path):
        raise RuntimeError("ANEMLL loader failed.")
    return model.model.layers[layer_idx].self_attn


def _run_chunked_prompt_matrix(
    model_path: str,
    layer_idx: int,
    cfg: Qwen35Config,
    our_attn: torch.nn.Module,
    prompt_lengths: List[int],
    input_seq_len: int,
    seed: int,
) -> None:
    hf_layer, hf_rotary = _load_hf_attention_layer(model_path, layer_idx)
    print("Chunked Full-Attn Validation Matrix")
    print(f"input_seq_len={input_seq_len} state_length={cfg.state_length}")

    for prompt_len in prompt_lengths:
        if prompt_len < 1:
            raise ValueError(f"prompt_len must be >= 1, got {prompt_len}")
        if prompt_len >= cfg.state_length:
            raise ValueError(
                f"prompt_len {prompt_len} must be smaller than state_length {cfg.state_length} for decode parity."
            )
        hidden_prefill, hidden_decode = _make_hidden_states(cfg.hidden_size, prompt_len, seed + prompt_len)
        torch_prefill, torch_decode = _run_torch_chunked_prompt_parity(
            hf_layer=hf_layer,
            hf_rotary=hf_rotary,
            cfg=cfg,
            our_attn=our_attn,
            hidden_prefill=hidden_prefill,
            hidden_decode=hidden_decode,
            input_seq_len=input_seq_len,
            title=f"Chunked PyTorch Full-Attn Parity vs HF [prompt_len={prompt_len}]",
        )
        _run_chunked_coreml_parity(
            cfg=cfg,
            our_attn=our_attn,
            input_seq_len=input_seq_len,
            hidden_prefill=hf_layer.input_layernorm(hidden_prefill).half(),
            hidden_decode=hf_layer.input_layernorm(hidden_decode).half(),
            torch_prefill=torch_prefill,
            torch_decode=torch_decode,
            title=f"Chunked Stateful CoreML Parity vs PyTorch [prompt_len={prompt_len}]",
        )


def _run_validation_matrix(
    model_path: str,
    layer_idx: int,
    base_cfg: Qwen35Config,
    base_attn: torch.nn.Module,
    seq_lens: List[int],
    seed: int,
    long_prefill_seq_lens: List[int],
    long_prefill_state_length: int,
) -> None:
    print("Full-Attn Validation Matrix")
    print(f"base_state_length={base_cfg.state_length}")

    for seq_len in seq_lens:
        hidden_prefill, hidden_decode = _make_hidden_states(base_cfg.hidden_size, seq_len, seed + seq_len)
        if seq_len >= base_cfg.state_length:
            _run_hf_prefill_only_parity(
                model_path=model_path,
                layer_idx=layer_idx,
                cfg=base_cfg,
                our_attn=base_attn,
                hidden_prefill=hidden_prefill,
                title=(
                    "PyTorch Full-Attn Prefill Parity vs HF "
                    f"[seq_len={seq_len}, decode_skipped_at_state_limit={base_cfg.state_length}]"
                ),
            )
        else:
            _run_hf_parity(
                model_path=model_path,
                layer_idx=layer_idx,
                cfg=base_cfg,
                our_attn=base_attn,
                hidden_prefill=hidden_prefill,
                hidden_decode=hidden_decode,
                title=f"PyTorch Full-Attn Block Parity vs HF [seq_len={seq_len}]",
            )

    if not long_prefill_seq_lens:
        return

    if long_prefill_state_length <= base_cfg.state_length:
        raise ValueError(
            "long_prefill_state_length must be greater than the base state_length "
            f"({base_cfg.state_length}), got {long_prefill_state_length}."
        )

    long_cfg = Qwen35Config.from_json(os.path.join(model_path, "config.json"))
    long_cfg.state_length = int(long_prefill_state_length)
    long_cfg.text_config.state_length = int(long_prefill_state_length)
    long_attn = copy.deepcopy(base_attn).half().eval()
    long_attn.config.state_length = int(long_prefill_state_length)
    long_attn.config.text_config.state_length = int(long_prefill_state_length)

    print(
        f"Long prefill stress validation: state_length={long_cfg.state_length} "
        f"(base fixed context={base_cfg.state_length})"
    )
    for seq_len in long_prefill_seq_lens:
        if seq_len <= base_cfg.state_length:
            raise ValueError(
                f"long prefill seq_len must exceed the base state_length {base_cfg.state_length}, got {seq_len}."
            )
        if seq_len >= long_cfg.state_length:
            raise ValueError(
                f"long prefill seq_len must be smaller than long_prefill_state_length {long_cfg.state_length}, got {seq_len}."
            )
        hidden_prefill, _ = _make_hidden_states(long_cfg.hidden_size, seq_len, seed + 1000 + seq_len)
        _run_hf_prefill_only_parity(
            model_path=model_path,
            layer_idx=layer_idx,
            cfg=long_cfg,
            our_attn=long_attn,
            hidden_prefill=hidden_prefill,
            title=(
                "PyTorch Full-Attn Prefill Parity vs HF "
                f"[seq_len={seq_len}, state_length={long_cfg.state_length}]"
            ),
        )


def _export_and_run_coreml(
    block_prefill: torch.nn.Module,
    block_decode: torch.nn.Module,
    cfg: Qwen35Config,
    seq_len: int,
    x_prefill: torch.Tensor,
    x_decode: torch.Tensor,
    save_packages: bool,
) -> None:
    import coremltools as ct

    # Trace separate fixed-shape prefill/decode blocks.
    traced_prefill = torch.jit.trace(
        block_prefill,
        (
            x_prefill,
            torch.arange(seq_len, dtype=torch.long),
        ),
        strict=False,
    )
    traced_decode = torch.jit.trace(
        block_decode,
        (x_decode,),
        strict=False,
    )

    state_k = ct.StateType(
        wrapped_type=ct.TensorType(shape=(cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=np.float16),
        name="k_cache",
    )
    state_v = ct.StateType(
        wrapped_type=ct.TensorType(shape=(cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=np.float16),
        name="v_cache",
    )
    common_inputs_prefill = [
        ct.TensorType(name="hidden_states", shape=x_prefill.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=(seq_len,), dtype=np.int32),
    ]
    common_inputs_decode = [
        ct.TensorType(name="hidden_states", shape=x_decode.shape, dtype=np.float16),
    ]

    ml_prefill = ct.convert(
        traced_prefill,
        inputs=common_inputs_prefill,
        outputs=[ct.TensorType(name="attn_out")],
        states=[state_k, state_v],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )
    ml_decode = ct.convert(
        traced_decode,
        inputs=common_inputs_decode,
        outputs=[ct.TensorType(name="attn_out")],
        states=[state_k, state_v],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )

    if save_packages:
        out_dir = REPO_ROOT / "tests" / "dev"
        prefill_pkg = out_dir / "qwen35_full_attn_stateful_prefill.mlpackage"
        decode_pkg = out_dir / "qwen35_full_attn_stateful_decode.mlpackage"
        ml_prefill.save(str(prefill_pkg))
        ml_decode.save(str(decode_pkg))
        print(f"Saved stateful CoreML prefill block: {prefill_pkg}")
        print(f"Saved stateful CoreML decode block: {decode_pkg}")

    # Attempt runtime parity (may fail on non-macOS without libcoremlpython).
    try:
        st = ml_prefill.make_state()
        coreml_prefill = ml_prefill.predict(
            {
                "hidden_states": _to_np(x_prefill),
                "position_ids": np.arange(seq_len, dtype=np.int32),
            },
            state=st,
        )["attn_out"]
        print("CoreML prefill predict: SUCCESS")
    except Exception as e:
        print(f"CoreML prefill predict: SKIPPED ({e})")
        return

    try:
        # Carry state into decode by copying state tensors.
        st_decode = ml_decode.make_state()
        st_decode.write_state(name="k_cache", value=st.read_state(name="k_cache"))
        st_decode.write_state(name="v_cache", value=st.read_state(name="v_cache"))
        coreml_decode = ml_decode.predict(
            {
                "hidden_states": _to_np(x_decode),
            },
            state=st_decode,
        )["attn_out"]
        print("CoreML decode predict: SUCCESS")
    except Exception as e:
        print(f"CoreML decode predict: SKIPPED ({e})")
        return

    with torch.no_grad():
        t_prefill = block_prefill(
            x_prefill,
            torch.arange(seq_len, dtype=torch.long),
        )
        # Recreate decode baseline from a fresh block with prefilled state.
        t_block_for_decode = block_decode
        t_block_for_decode.k_cache.copy_(block_prefill.k_cache)
        t_block_for_decode.v_cache.copy_(block_prefill.v_cache)
        t_decode = t_block_for_decode(x_decode)

    print("Stateful CoreML Parity vs PyTorch")
    _print_metric("prefill_coreml_vs_torch", _metrics(torch.from_numpy(coreml_prefill), t_prefill))
    _print_metric("decode_coreml_vs_torch", _metrics(torch.from_numpy(coreml_decode), t_decode))


def _convert_coreml_prefill_block(block_prefill: torch.nn.Module, cfg: Qwen35Config, input_seq_len: int):
    import coremltools as ct

    sample_hidden = torch.zeros((1, input_seq_len, cfg.hidden_size), dtype=torch.float16)
    sample_pos = torch.arange(input_seq_len, dtype=torch.long)
    traced_prefill = torch.jit.trace(
        block_prefill,
        (sample_hidden, sample_pos),
        strict=False,
    )
    state_k = ct.StateType(
        wrapped_type=ct.TensorType(shape=(cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=np.float16),
        name="k_cache",
    )
    state_v = ct.StateType(
        wrapped_type=ct.TensorType(shape=(cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=np.float16),
        name="v_cache",
    )
    return ct.convert(
        traced_prefill,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, input_seq_len, cfg.hidden_size), dtype=np.float16),
            ct.TensorType(name="position_ids", shape=(input_seq_len,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="attn_out")],
        states=[state_k, state_v],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )


def _convert_coreml_decode_block(block_decode: torch.nn.Module, cfg: Qwen35Config):
    import coremltools as ct

    sample_hidden = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16)
    traced_decode = torch.jit.trace(
        block_decode,
        (sample_hidden,),
        strict=False,
    )
    state_k = ct.StateType(
        wrapped_type=ct.TensorType(shape=(cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=np.float16),
        name="k_cache",
    )
    state_v = ct.StateType(
        wrapped_type=ct.TensorType(shape=(cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=np.float16),
        name="v_cache",
    )
    return ct.convert(
        traced_decode,
        inputs=[ct.TensorType(name="hidden_states", shape=(1, 1, cfg.hidden_size), dtype=np.float16)],
        outputs=[ct.TensorType(name="attn_out")],
        states=[state_k, state_v],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )


def _run_chunked_coreml_parity(
    cfg: Qwen35Config,
    our_attn: torch.nn.Module,
    input_seq_len: int,
    hidden_prefill: torch.Tensor,
    hidden_decode: torch.Tensor,
    torch_prefill: torch.Tensor,
    torch_decode: torch.Tensor,
    *,
    title: str,
) -> None:
    prompt_len = hidden_prefill.shape[1]
    prefill_models = {}
    decode_model = _convert_coreml_decode_block(
        StatefulFullAttentionDecodeBlock(our_attn, cfg, current_pos=prompt_len).half().eval(),
        cfg,
    )

    try:
        current_state = None
        coreml_prefill_chunks: List[np.ndarray] = []
        for start, end in _chunk_slices(prompt_len, input_seq_len):
            if start not in prefill_models:
                prefill_models[start] = _convert_coreml_prefill_block(
                    StatefulFullAttentionChunkPrefillBlock(
                        our_attn, cfg, input_seq_len=input_seq_len, current_pos=start
                    ).half().eval(),
                    cfg,
                    input_seq_len,
                )
            ml_prefill = prefill_models[start]
            st = ml_prefill.make_state()
            if current_state is not None:
                st.write_state(name="k_cache", value=current_state["k_cache"])
                st.write_state(name="v_cache", value=current_state["v_cache"])

            chunk = hidden_prefill[:, start:end, :]
            chunk_len = end - start
            padded_chunk = _pad_chunk_hidden(chunk, input_seq_len)
            position_ids = np.arange(start, start + input_seq_len, dtype=np.int32)
            out = ml_prefill.predict(
                {"hidden_states": _to_np(padded_chunk), "position_ids": position_ids},
                state=st,
            )["attn_out"]
            coreml_prefill_chunks.append(out[:, :chunk_len, :])
            current_state = {
                "k_cache": st.read_state(name="k_cache"),
                "v_cache": st.read_state(name="v_cache"),
            }

        st_decode = decode_model.make_state()
        if current_state is not None:
            st_decode.write_state(name="k_cache", value=current_state["k_cache"])
            st_decode.write_state(name="v_cache", value=current_state["v_cache"])
        coreml_decode = decode_model.predict(
            {"hidden_states": _to_np(hidden_decode)},
            state=st_decode,
        )["attn_out"]
    except Exception as e:
        print(f"{title}: SKIPPED ({e})")
        return

    coreml_prefill = np.concatenate(coreml_prefill_chunks, axis=1)
    print(title)
    _print_metric("prefill_coreml_vs_torch", _metrics(torch.from_numpy(coreml_prefill), torch_prefill))
    _print_metric("decode_coreml_vs_torch", _metrics(torch.from_numpy(coreml_decode), torch_decode))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--input-seq-len", type=int, default=256)
    parser.add_argument("--cache-size", type=int, default=2048)
    parser.add_argument("--prompt-lengths", type=str, default="128,320,1026")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    if cfg.text_config.layer_types[args.layer_idx] != "full_attention":
        raise ValueError(
            f"layer {args.layer_idx} is {cfg.text_config.layer_types[args.layer_idx]}, expected full_attention"
        )

    cfg.state_length = int(args.cache_size)
    cfg.text_config.state_length = int(args.cache_size)

    our_attn = _load_our_attn(args.model_path, cfg, args.layer_idx)
    prompt_lengths = sorted(set(_parse_int_csv(args.prompt_lengths)))
    _run_chunked_prompt_matrix(
        model_path=args.model_path,
        layer_idx=args.layer_idx,
        cfg=cfg,
        our_attn=our_attn,
        prompt_lengths=prompt_lengths,
        input_seq_len=args.input_seq_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
