"""Qwen 3.5 text-model implementation for ANEMLL.

Scope:
1) robust config parsing for Qwen3.5 nested text_config layouts
2) full-attention + linear-attention text decoder layers
3) fixed-shape cache contracts for ANE-oriented execution
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch.nn.functional as F
import torch
import torch.nn as nn


MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 256
STATE_LENGTH = 256


@dataclass
class Qwen35TextConfig:
    model_type: str = "qwen3_5_text"
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    use_cache: bool = True
    max_position_embeddings: int = 0
    rope_theta: float = 10000000.0
    rope_type: str = "default"
    partial_rotary_factor: float = 1.0
    layer_types: List[str] | None = None
    attn_output_gate: bool = False
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0
    context_length: int = CONTEXT_LENGTH
    state_length: int = STATE_LENGTH

    @classmethod
    def from_raw_dict(cls, data: Dict) -> "Qwen35TextConfig":
        rope = data.get("rope_parameters", {}) or {}
        layer_types = data.get("layer_types", [])
        return cls(
            model_type=data.get("model_type", "qwen3_5_text"),
            hidden_size=int(data.get("hidden_size", 0)),
            intermediate_size=int(data.get("intermediate_size", 0)),
            num_hidden_layers=int(data.get("num_hidden_layers", 0)),
            num_attention_heads=int(data.get("num_attention_heads", 0)),
            num_key_value_heads=int(data.get("num_key_value_heads", 0)),
            head_dim=int(data.get("head_dim", 0)),
            vocab_size=int(data.get("vocab_size", 0)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            attention_bias=bool(data.get("attention_bias", False)),
            use_cache=bool(data.get("use_cache", True)),
            max_position_embeddings=int(data.get("max_position_embeddings", 0)),
            rope_theta=float(rope.get("rope_theta", 10000000.0)),
            rope_type=str(rope.get("rope_type", "default")),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
            layer_types=list(layer_types) if isinstance(layer_types, list) else [],
            attn_output_gate=bool(data.get("attn_output_gate", False)),
            linear_num_key_heads=int(data.get("linear_num_key_heads", 0)),
            linear_num_value_heads=int(data.get("linear_num_value_heads", 0)),
            linear_key_head_dim=int(data.get("linear_key_head_dim", 0)),
            linear_value_head_dim=int(data.get("linear_value_head_dim", 0)),
            linear_conv_kernel_dim=int(data.get("linear_conv_kernel_dim", 0)),
            context_length=int(data.get("context_length", CONTEXT_LENGTH)),
            state_length=int(data.get("state_length", STATE_LENGTH)),
        )


class Qwen35Config:
    """Normalized Qwen3.5 config wrapper.

    Qwen3.5 checkpoints may store text params under top-level `text_config`.
    This class normalizes that shape for ANEMLL internals.
    """

    def __init__(self, raw_config: Dict):
        self.raw_config = raw_config
        self.architectures = raw_config.get("architectures", [])
        self.model_type = raw_config.get("model_type", "")
        self.text_config = Qwen35TextConfig.from_raw_dict(
            raw_config.get("text_config", raw_config)
        )

        # Expose common attributes expected by converter/model code.
        self.hidden_size = self.text_config.hidden_size
        self.intermediate_size = self.text_config.intermediate_size
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.head_dim = self.text_config.head_dim
        self.vocab_size = self.text_config.vocab_size
        self.rms_norm_eps = self.text_config.rms_norm_eps
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.context_length = self.text_config.context_length
        self.state_length = self.text_config.state_length

    @classmethod
    def from_json(cls, json_file: str) -> "Qwen35Config":
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def has_linear_attention(self) -> bool:
        return any(t == "linear_attention" for t in self.text_config.layer_types)

    def has_full_attention(self) -> bool:
        return any(t == "full_attention" for t in self.text_config.layer_types)


def _load_weight_index(model_path: str) -> Dict:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return {}
    with open(index_path, "r") as f:
        return json.load(f)


def _load_weight_keys(model_path: str) -> List[str]:
    """Load checkpoint key list from index file if present.

    This avoids loading all shard tensors during compatibility preflight.
    """
    index_data = _load_weight_index(model_path)
    if index_data:
        weight_map = index_data.get("weight_map", {})
        return list(weight_map.keys())

    # Fallback for unsharded checkpoints without an index.
    return []


def _normalize_language_key(key: str) -> str:
    """Map HF multimodal language keys into text-model-relative keys."""
    if key.startswith("model.language_model."):
        return key[len("model.language_model.") :]
    if key.startswith("language_model."):
        return key[len("language_model.") :]
    if key.startswith("model."):
        return key[len("model.") :]
    return key


def _layer_index_from_key(key: str) -> int | None:
    if not key.startswith("layers."):
        return None
    parts = key.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


class Qwen35RMSNorm(nn.Module):
    """ANE-friendly RMSNorm with Qwen3.5 offset scaling semantics."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        # Qwen3.5 uses output * (1 + weight)
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * self.hidden_size,),
            weight=None,
            bias=None,
            eps=float(self.eps),
        )
        normed = normed[..., : self.hidden_size]
        scale = 1.0 + self.weight.to(normed.dtype, copy=False).to(normed.device, copy=False)
        return normed * scale


class Qwen35RMSNormGated(nn.Module):
    """RMSNorm + SiLU gate used by Qwen3.5 linear attention."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # ANE-oriented RMSNorm via doubled LayerNorm trick:
        # concat([x, -x]) -> zero mean, LayerNorm variance equals mean(x^2).
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * self.hidden_size,),
            weight=None,
            bias=None,
            eps=float(self.eps),
        )
        normed = normed[..., : self.hidden_size]
        out = normed * self.weight.to(hidden_states.dtype)
        out = out * F.silu(gate.to(hidden_states.dtype))
        return out


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Qwen35RotaryEmbedding(nn.Module):
    """RoPE cache with partial-rotary support."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rotary_dim = max(2, int(self.head_dim * config.text_config.partial_rotary_factor))
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1

        inv_freq = 1.0 / (
            config.text_config.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2).float().to(TEST_DEVICE)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(
            max(config.context_length, config.state_length) * 2, device=TEST_DEVICE
        ).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().unsqueeze(0)
        self.sin_cached = emb.sin().unsqueeze(0)

    def get(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() == 1:
            pos_ids = position_ids
        else:
            pos_ids = position_ids.squeeze(0)
        cos = self.cos_cached[:, pos_ids].to(x.dtype)
        sin = self.sin_cached[:, pos_ids].to(x.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor, half_dim: int) -> torch.Tensor:
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_rotary_dim = rotary_dim // 2
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = q[..., :rotary_dim]
    k_rot = k[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_pass = k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (_rotate_half(q_rot, half_rotary_dim) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot, half_rotary_dim) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def apply_rotary_pos_emb_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prefill rotary helper for Qwen3.5 (partial rotary aware)."""
    return _apply_rotary_partial(q, k, cos, sin, rotary_dim)


def apply_rotary_pos_emb_single(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-token rotary helper for Qwen3.5 (partial rotary aware)."""
    return _apply_rotary_partial(q, k, cos, sin, rotary_dim)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1)
    return hidden_states.flatten(1, 2)


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Conv2d(
            self.hidden_size, self.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE
        )
        self.up_proj = nn.Conv2d(
            self.hidden_size, self.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE
        )
        self.down_proj = nn.Conv2d(
            self.intermediate_size, self.hidden_size, 1, bias=False, dtype=MODEL_DTYPE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(gated)
        return out.squeeze(2).permute(0, 2, 1)


class Qwen35FullAttention(nn.Module):
    """Self-attention block for full_attention layers only."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        # Qwen3.5 full-attention q_proj in this checkpoint is 2x wider than k/v head_dim.
        # TODO: replace this heuristic with exact HF reference decomposition.
        self.q_head_dim = config.head_dim * 2
        self.rotary = Qwen35RotaryEmbedding(config)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        q_proj_dim = self.num_heads * self.q_head_dim
        kv_proj_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Conv2d(self.hidden_size, q_proj_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.k_proj = nn.Conv2d(self.hidden_size, kv_proj_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.v_proj = nn.Conv2d(self.hidden_size, kv_proj_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.o_proj = nn.Conv2d(self.num_heads * self.head_dim, self.hidden_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _project_qkvg(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        q_all = (
            self.q_proj(hs)
            .view(1, self.num_heads, self.head_dim * 2, -1)
            .permute(0, 1, 3, 2)
        )  # [B,H,S,2D]
        query_states = q_all[..., : self.head_dim]
        gate = q_all[..., self.head_dim :].permute(0, 2, 1, 3).flatten(2, 3)
        key_states = (
            self.k_proj(hs)
            .view(1, self.num_kv_heads, self.head_dim, -1)
            .permute(0, 1, 3, 2)
        )
        value_states = (
            self.v_proj(hs)
            .view(1, self.num_kv_heads, self.head_dim, -1)
            .permute(0, 1, 3, 2)
        )
        return query_states, key_states, value_states, gate

    def _query_for_scores(self, query_states: torch.Tensor) -> torch.Tensor:
        if query_states.shape[-1] == self.head_dim:
            return query_states
        # TODO: match HF q-splitting logic for q_head_dim > k_head_dim.
        return query_states[..., : self.head_dim]

    def get_new_kv_cache(
        self, hidden_states: torch.Tensor, current_pos: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO(parity): Keep API/shape contract aligned with qwen_model.py::QwenAttention.get_new_kv_cache.
        # Current version is a bring-up approximation for full_attention-only layers.
        query_states, key_states, value_states, gate = self._project_qkvg(hidden_states)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        pos = current_pos.reshape(1).to(device=hidden_states.device, dtype=torch.long)
        cos, sin = self.rotary.get(hidden_states, pos)
        query_states, key_states = apply_rotary_pos_emb_single(
            query_states, key_states, cos, sin, self.rotary.rotary_dim
        )
        return (
            query_states.to(MODEL_DTYPE),
            key_states.to(MODEL_DTYPE),
            value_states.to(MODEL_DTYPE),
            gate.to(MODEL_DTYPE),
        )

    def get_new_kv_cache_prefill(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO(parity): Keep API/shape contract aligned with qwen_model.py::QwenAttention.get_new_kv_cache_prefill.
        # Current version is a bring-up approximation for full_attention-only layers.
        query_states, key_states, value_states, gate = self._project_qkvg(hidden_states)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cos, sin = self.rotary.get(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb_prefill(
            query_states, key_states, cos, sin, self.rotary.rotary_dim
        )
        return (
            query_states.to(MODEL_DTYPE),
            key_states.to(MODEL_DTYPE),
            value_states.to(MODEL_DTYPE),
            gate.to(MODEL_DTYPE),
        )

    def _project_output(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate.to(attn_output.dtype))
        out = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return out.squeeze(2).permute(0, 2, 1)

    def forward_regular(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        kv_cache_layer: Tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TODO(parity): Mirror qwen_model.py::QwenAttention.forward_regular exactly
        # once linear_attention/shared-cache contracts are finalized for Qwen3.5.
        k_cache, v_cache = kv_cache_layer

        # Match qwen_model.py: keep a fixed cache length contract for CoreML.
        k_cache = k_cache[..., : self.config.state_length, :]
        v_cache = v_cache[..., : self.config.state_length, :]

        n_rep = self.num_heads // self.num_kv_heads
        key_states = _repeat_kv(k_cache.unsqueeze(0), n_rep)
        value_states = _repeat_kv(v_cache.unsqueeze(0), n_rep)

        attn_weights = (
            torch.matmul(query_states.to(MODEL_DTYPE), key_states.transpose(-1, -2).to(MODEL_DTYPE))
            * self.scale
        )
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states.to(MODEL_DTYPE))
        attn_output = attn_output.transpose(1, 2).contiguous().flatten(2, 3)
        return self._project_output(attn_output, hidden_states, gate=gate)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        kv_cache_layer: Tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TODO(parity): Mirror qwen_model.py::QwenAttention.forward_prefill exactly
        # once linear_attention/shared-cache contracts are finalized for Qwen3.5.
        k_cache, v_cache = kv_cache_layer

        # Match qwen_model.py: keep a fixed cache length contract for CoreML.
        k_cache = k_cache[..., : self.config.state_length, :]
        v_cache = v_cache[..., : self.config.state_length, :]

        n_rep = self.num_heads // self.num_kv_heads
        key_states = _repeat_kv(k_cache.unsqueeze(0), n_rep)
        value_states = _repeat_kv(v_cache.unsqueeze(0), n_rep)

        attn_weights = (
            torch.matmul(
                query_states.to(MODEL_DTYPE), key_states.transpose(-2, -1).to(MODEL_DTYPE)
            )
            * self.scale
        )
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states.to(MODEL_DTYPE))
        attn_output = attn_output.transpose(1, 2).contiguous().flatten(2, 3)
        return self._project_output(attn_output, hidden_states, gate=gate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        query_states, key_states, value_states, gate = self._project_qkvg(hidden_states)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        n_rep = self.num_heads // self.num_kv_heads
        key_states = _repeat_kv(key_states, n_rep)
        value_states = _repeat_kv(value_states, n_rep)

        cos, sin = self.rotary.get(hidden_states, position_ids)
        query_states, key_states = _apply_rotary_partial(
            query_states, key_states, cos, sin, self.rotary.rotary_dim
        )

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask[:, :, :seq_len, :seq_len].to(attn_weights.dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, -1)

        return self._project_output(attn_output, hidden_states, gate=gate)


class Qwen35LinearAttention(nn.Module):
    """Qwen3.5 linear attention (gated delta net) with fixed-shape cache contract."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.text_config.linear_num_key_heads
        self.num_v_heads = config.text_config.linear_num_value_heads
        self.head_k_dim = config.text_config.linear_key_head_dim
        self.head_v_dim = config.text_config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.state_dim = self.value_dim
        self.linear_conv_kernel_dim = max(1, int(config.text_config.linear_conv_kernel_dim))
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Conv2d(
            self.hidden_size, self.conv_dim, 1, bias=False, dtype=MODEL_DTYPE
        )
        self.in_proj_z = nn.Conv2d(self.hidden_size, self.value_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_proj_b = nn.Conv2d(self.hidden_size, self.num_v_heads, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_proj_a = nn.Conv2d(self.hidden_size, self.num_v_heads, 1, bias=False, dtype=MODEL_DTYPE)
        self.conv2d = nn.Conv2d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=(1, self.linear_conv_kernel_dim),
            padding=0,
            groups=self.conv_dim,
            bias=False,
            dtype=MODEL_DTYPE,
        )
        self.out_proj = nn.Conv2d(self.value_dim, self.hidden_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32, device=TEST_DEVICE))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads, dtype=MODEL_DTYPE, device=TEST_DEVICE))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

    @staticmethod
    def _conv2d_proj(conv: nn.Conv2d, x_bsh: torch.Tensor) -> torch.Tensor:
        y = conv(x_bsh.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE))
        return y.squeeze(2).transpose(1, 2)

    def _causal_conv_update(
        self, mixed_qkv_t: torch.Tensor, conv_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # mixed_qkv_t: [B, conv_dim, seq_len], conv_state: [B, conv_dim, K]
        k = self.linear_conv_kernel_dim
        seq_len = mixed_qkv_t.shape[-1]
        stacked = torch.cat([conv_state.to(mixed_qkv_t.dtype), mixed_qkv_t], dim=-1)
        stacked4 = stacked.unsqueeze(2)  # [B, C, 1, K + seq_len]
        # Use static temporal slicing: keep only the last `seq_len` outputs.
        out4 = self.conv2d(stacked4).squeeze(2)
        out = F.silu(out4[:, :, -seq_len:])
        next_state = stacked[:, :, -k:]
        return out.to(mixed_qkv_t.dtype), next_state.to(mixed_qkv_t.dtype)

    @staticmethod
    def _chunk_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mirror HF torch fallback as closely as possible for parity bring-up.
        initial_dtype = query.dtype
        query = _l2norm(query, dim=-1)
        key = _l2norm(key, dim=-1)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, seq_len, k_dim = key.shape
        v_dim = value.shape[-1]
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = seq_len + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_dim, v_dim).to(value)
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
            v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :seq_len]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    @staticmethod
    def _recurrent_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        output_final_state: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        initial_dtype = query.dtype
        query = _l2norm(query, dim=-1)
        key = _l2norm(key, dim=-1)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]
        bsz, n_heads, seq_len, _ = key.shape
        v_dim = value.shape[-1]
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        out = torch.zeros(bsz, n_heads, seq_len, v_dim, dtype=value.dtype, device=value.device)
        state = recurrent_state.to(value)
        for i in range(seq_len):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        if not output_final_state:
            state = None
        out = out.transpose(1, 2).contiguous().to(initial_dtype)
        return out, state

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        has_previous_state: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        mixed_qkv = self._conv2d_proj(self.in_proj_qkv, hidden_states).transpose(1, 2)
        z = self._conv2d_proj(self.in_proj_z, hidden_states).reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        b = self._conv2d_proj(self.in_proj_b, hidden_states)
        a = self._conv2d_proj(self.in_proj_a, hidden_states)

        if conv_state is None:
            conv_state = torch.zeros(
                (bsz, self.conv_dim, self.linear_conv_kernel_dim),
                dtype=MODEL_DTYPE,
                device=hidden_states.device,
            )
        conv_out, next_conv_state = self._causal_conv_update(mixed_qkv, conv_state)
        mixed_qkv = conv_out.transpose(1, 2)

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        if recurrent_state is None:
            recurrent_state = torch.zeros(
                (bsz, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                dtype=torch.float32,
                device=hidden_states.device,
            )

        if has_previous_state and seq_len == 1:
            core, next_recurrent_state = self._recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, recurrent_state=recurrent_state, output_final_state=True
            )
        else:
            core, next_recurrent_state = self._chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=None, output_final_state=True
            )

        core = core.reshape(-1, self.head_v_dim)
        zf = z.reshape(-1, self.head_v_dim)
        core = self.norm(core, zf).reshape(bsz, seq_len, self.value_dim)
        out = self.out_proj(core.permute(0, 2, 1).unsqueeze(2)).squeeze(2).transpose(1, 2)
        return out, next_conv_state, next_recurrent_state

    def get_new_kv_cache(
        self, hidden_states: torch.Tensor, current_pos: torch.LongTensor | int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compatibility adapter: returns (attn_output, next_conv_state, next_recurrent_state_flat)
        bsz = hidden_states.shape[0]
        conv_state = torch.zeros(
            (bsz, self.conv_dim, self.linear_conv_kernel_dim), dtype=MODEL_DTYPE, device=hidden_states.device
        )
        recurrent_state = torch.zeros(
            (bsz, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        out, next_conv, next_rec = self._forward_impl(
            hidden_states=hidden_states,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            has_previous_state=bool(int(current_pos) > 0) if not isinstance(current_pos, torch.Tensor) else bool(int(current_pos.item()) > 0),
        )
        return out, next_conv, next_rec.reshape(bsz, self.num_v_heads, -1).to(MODEL_DTYPE)

    def get_new_kv_cache_prefill(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = hidden_states.shape[0]
        conv_state = torch.zeros(
            (bsz, self.conv_dim, self.linear_conv_kernel_dim), dtype=MODEL_DTYPE, device=hidden_states.device
        )
        recurrent_state = torch.zeros(
            (bsz, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        out, next_conv, next_rec = self._forward_impl(
            hidden_states=hidden_states,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            has_previous_state=False,
        )
        return out, next_conv, next_rec.reshape(bsz, self.num_v_heads, -1).to(MODEL_DTYPE)

    def forward_regular(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        has_previous_state: bool = True,
        causal_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward_impl(
            hidden_states=hidden_states,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            has_previous_state=has_previous_state,
        )

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        has_previous_state: bool = False,
        causal_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward_impl(
            hidden_states=hidden_states,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            has_previous_state=has_previous_state,
        )

    def forward(
        self, hidden_states: torch.Tensor, causal_mask: torch.Tensor | None, position_ids: torch.LongTensor
    ) -> torch.Tensor:
        out, _, _ = self._forward_impl(
            hidden_states=hidden_states,
            conv_state=None,
            recurrent_state=None,
            has_previous_state=False,
        )
        return out


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = (
            Qwen35FullAttention(config)
            if layer_type == "full_attention"
            else Qwen35LinearAttention(config)
        )
        self.mlp = Qwen35MLP(config)

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor | None, position_ids: torch.LongTensor) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, causal_mask, position_ids)
        hidden_states = residual + x
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        hidden_states = residual + x
        return hidden_states


class Qwen35Model(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(TEST_DEVICE)
        layer_types = config.text_config.layer_types
        if not layer_types:
            layer_types = ["full_attention"] * config.num_hidden_layers
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) does not match num_hidden_layers ({config.num_hidden_layers})."
            )
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(config, layer_type=layer_types[i]) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        cache_size = (
            2 * config.num_hidden_layers,
            config.num_key_value_heads,
            config.state_length,
            config.head_dim,
        )
        self.register_buffer("kv_cache_0", torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE))
        if any(layer.layer_type == "linear_attention" for layer in self.layers):
            conv_dim = (
                config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim * 2
                + config.text_config.linear_num_value_heads * config.text_config.linear_value_head_dim
            )
            conv_kernel = max(1, int(config.text_config.linear_conv_kernel_dim))
            self.register_buffer(
                "linear_conv_state",
                torch.zeros(
                    (config.num_hidden_layers, conv_dim, conv_kernel),
                    dtype=MODEL_DTYPE,
                    device=TEST_DEVICE,
                ),
            )
            self.register_buffer(
                "linear_recurrent_state",
                torch.zeros(
                    (
                        config.num_hidden_layers,
                        config.text_config.linear_num_value_heads,
                        config.text_config.linear_key_head_dim,
                        config.text_config.linear_value_head_dim,
                    ),
                    dtype=torch.float32,
                    device=TEST_DEVICE,
                ),
            )

    def _build_fixed_cache_mask(
        self, q_len: int, current_pos: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Build ANE-friendly fixed-width cache mask [1, 1, q_len, state_length]."""
        cache_len = self.config.state_length
        q_idx = torch.arange(q_len, device=device).unsqueeze(-1)  # [q_len, 1]
        k_idx = torch.arange(cache_len, device=device).unsqueeze(0)  # [1, cache_len]
        allowed = k_idx <= (current_pos + q_idx)  # [q_len, cache_len]
        zeros = torch.zeros((q_len, cache_len), dtype=dtype, device=device)
        neg_inf = torch.full((q_len, cache_len), float("-inf"), dtype=dtype, device=device)
        mask_2d = torch.where(allowed, zeros, neg_inf)
        return mask_2d.unsqueeze(0).unsqueeze(0)

    def _process_layer_prefill(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor | None,
        current_pos: torch.LongTensor | int,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        if layer.layer_type == "linear_attention":
            x = layer.input_layernorm(hidden_states)
            pos = int(current_pos.item()) if isinstance(current_pos, torch.Tensor) else int(current_pos)
            if pos == 0:
                self.linear_conv_state[layer_idx].zero_()
                self.linear_recurrent_state[layer_idx].zero_()
            conv_state = self.linear_conv_state[layer_idx : layer_idx + 1]
            recurrent_state = self.linear_recurrent_state[layer_idx : layer_idx + 1]
            attn_out, next_conv, next_rec = layer.self_attn.forward_prefill(
                hidden_states=x,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                has_previous_state=(pos > 0),
                causal_mask=causal_mask,
            )
            self.linear_conv_state[layer_idx : layer_idx + 1] = next_conv
            self.linear_recurrent_state[layer_idx : layer_idx + 1] = next_rec.to(self.linear_recurrent_state.dtype)
            hidden_states = hidden_states + attn_out
            post = layer.post_attention_layernorm(hidden_states)
            return hidden_states + layer.mlp(post)

        x = layer.input_layernorm(hidden_states)
        query_states, key_states, value_states, gate = layer.self_attn.get_new_kv_cache_prefill(x, position_ids)

        pos = int(current_pos.item()) if isinstance(current_pos, torch.Tensor) else int(current_pos)
        seq_len = key_states.shape[2]
        fixed_mask = self._build_fixed_cache_mask(
            q_len=seq_len, current_pos=pos, dtype=MODEL_DTYPE, device=hidden_states.device
        )
        key_idx = layer_idx
        value_idx = layer_idx + self.config.num_hidden_layers
        self.kv_cache_0[key_idx:key_idx + 1, :, pos:pos + seq_len, :] = key_states
        self.kv_cache_0[value_idx:value_idx + 1, :, pos:pos + seq_len, :] = value_states

        key_cache = self.kv_cache_0[key_idx:key_idx + 1].squeeze(0)
        value_cache = self.kv_cache_0[value_idx:value_idx + 1].squeeze(0)
        attn_out = layer.self_attn.forward_prefill(
            hidden_states=x,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=fixed_mask,
            gate=gate,
        )
        hidden_states = hidden_states + attn_out
        post = layer.post_attention_layernorm(hidden_states)
        return hidden_states + layer.mlp(post)

    def _process_layer_regular(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor | None,
        current_pos: torch.LongTensor | int,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        if layer.layer_type == "linear_attention":
            x = layer.input_layernorm(hidden_states)
            pos = int(current_pos.item()) if isinstance(current_pos, torch.Tensor) else int(current_pos)
            seq_len = hidden_states.shape[1]
            if pos == 0:
                self.linear_conv_state[layer_idx].zero_()
                self.linear_recurrent_state[layer_idx].zero_()
            conv_state = self.linear_conv_state[layer_idx : layer_idx + 1]
            recurrent_state = self.linear_recurrent_state[layer_idx : layer_idx + 1]
            if seq_len == 1:
                attn_out, next_conv, next_rec = layer.self_attn.forward_regular(
                    hidden_states=x,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    has_previous_state=(pos > 0),
                    causal_mask=causal_mask,
                )
            else:
                attn_out, next_conv, next_rec = layer.self_attn.forward_prefill(
                    hidden_states=x,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    has_previous_state=(pos > 0),
                    causal_mask=causal_mask,
                )
            self.linear_conv_state[layer_idx : layer_idx + 1] = next_conv
            self.linear_recurrent_state[layer_idx : layer_idx + 1] = next_rec.to(self.linear_recurrent_state.dtype)
            hidden_states = hidden_states + attn_out
            post = layer.post_attention_layernorm(hidden_states)
            return hidden_states + layer.mlp(post)

        x = layer.input_layernorm(hidden_states)
        seq_len = hidden_states.shape[1]
        if seq_len == 1:
            query_states, key_states, value_states, gate = layer.self_attn.get_new_kv_cache(x, current_pos)
        else:
            query_states, key_states, value_states, gate = layer.self_attn.get_new_kv_cache_prefill(x, position_ids)

        pos = int(current_pos.item()) if isinstance(current_pos, torch.Tensor) else int(current_pos)
        key_idx = layer_idx
        value_idx = layer_idx + self.config.num_hidden_layers
        fixed_mask = self._build_fixed_cache_mask(
            q_len=seq_len, current_pos=pos, dtype=MODEL_DTYPE, device=hidden_states.device
        )

        if seq_len == 1:
            self.kv_cache_0[key_idx:key_idx + 1, :, pos:pos + 1, :] = key_states
            self.kv_cache_0[value_idx:value_idx + 1, :, pos:pos + 1, :] = value_states
        else:
            self.kv_cache_0[key_idx:key_idx + 1, :, pos:pos + seq_len, :] = key_states
            self.kv_cache_0[value_idx:value_idx + 1, :, pos:pos + seq_len, :] = value_states

        key_cache = self.kv_cache_0[key_idx:key_idx + 1].squeeze(0)
        value_cache = self.kv_cache_0[value_idx:value_idx + 1].squeeze(0)

        if seq_len == 1:
            attn_out = layer.self_attn.forward_regular(
                hidden_states=x,
                query_states=query_states,
                kv_cache_layer=(key_cache, value_cache),
                causal_mask=fixed_mask,
                gate=gate,
            )
        else:
            attn_out = layer.self_attn.forward_prefill(
                hidden_states=x,
                query_states=query_states,
                kv_cache_layer=(key_cache, value_cache),
                causal_mask=fixed_mask,
                gate=gate,
            )
        hidden_states = hidden_states + attn_out
        post = layer.post_attention_layernorm(hidden_states)
        return hidden_states + layer.mlp(post)

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor | int | None = None,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        if current_pos is None:
            for layer in self.layers:
                hidden_states = layer(hidden_states, causal_mask, position_ids)
        else:
            for layer_idx in range(len(self.layers)):
                if IN_PREFILL:
                    hidden_states = self._process_layer_prefill(
                        layer_idx, hidden_states, position_ids, causal_mask, current_pos
                    )
                else:
                    hidden_states = self._process_layer_regular(
                        layer_idx, hidden_states, position_ids, causal_mask, current_pos
                    )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen35ForCausalLM(nn.Module):
    """Qwen3.5 CausalLM wrapper with full-attention-only implementation."""

    config_class = Qwen35Config

    def __init__(self, config: Qwen35Config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen35Model(config)
        self.lm_head = nn.Conv2d(
            config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)

    def _compatibility_report(self, model_path: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        keys = _load_weight_keys(model_path)
        if not keys:
            errors.append(
                "No model.safetensors.index.json found; add index-based key preflight support for this checkpoint layout."
            )
            return False, errors

        norm_keys = [_normalize_language_key(k) for k in keys]

        # Check whether this is multimodal checkpoint wrapping language_model.
        if self.config.model_type == "qwen3_5":
            if not any(k.startswith("model.language_model.") for k in keys):
                errors.append("Expected model.language_model.* keys for qwen3_5 checkpoint.")

        # Verify expected full-attention keys exist for at least one layer.
        expected_probe = "layers.3.self_attn.q_proj.weight"
        if not any(expected_probe in k for k in norm_keys):
            errors.append(
                f"Missing expected full-attention probe key `{expected_probe}` after normalization."
            )

        return len(errors) == 0, errors

    def _load_hf_state_dict(self, model_path: str) -> Dict[str, torch.Tensor]:
        try:
            import safetensors.torch
        except Exception as exc:
            raise RuntimeError(
                "safetensors is required for Qwen3.5 weight loading. "
                "Install it in your env (e.g. conda env qwen_coreml)."
            ) from exc

        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                full = os.path.join(model_path, file)
                state_dict.update(safetensors.torch.load_file(full))
        return state_dict

    def _is_implemented_layer_key(self, norm_key: str) -> bool:
        layer_idx = _layer_index_from_key(norm_key)
        if layer_idx is None:
            return True
        if layer_idx >= len(self.model.layers):
            return False
        layer_type = self.model.layers[layer_idx].layer_type
        if layer_type == "full_attention":
            return ".linear_attn." not in norm_key
        if layer_type == "linear_attention":
            return True
        return False

    def _reshape_if_conv_weight(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        if key.endswith(".self_attn.conv2d.weight"):
            if tensor.dim() == 3:
                return tensor.view(tensor.shape[0], tensor.shape[1], 1, tensor.shape[2])
            return tensor
        conv_weight_suffixes = (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".self_attn.gate_proj.weight",
            ".self_attn.in_proj_qkv.weight",
            ".self_attn.in_proj_a.weight",
            ".self_attn.in_proj_b.weight",
            ".self_attn.in_proj_z.weight",
            ".self_attn.out_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
            "lm_head.weight",
        )
        if key.endswith(conv_weight_suffixes):
            return tensor.view(tensor.shape[0], tensor.shape[1], 1, 1)
        return tensor

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)

        ok, errors = self._compatibility_report(model_path)
        if not ok:
            print("Qwen3.5 compatibility preflight failed:")
            for err in errors:
                print(f"  - {err}")
            return False

        raw_state = self._load_hf_state_dict(model_path)
        if "lm_head.weight" not in raw_state:
            embed_key = "model.language_model.embed_tokens.weight"
            if embed_key in raw_state:
                raw_state["lm_head.weight"] = raw_state[embed_key].clone()
        mapped_state: Dict[str, torch.Tensor] = {}
        ignored_unimplemented = 0

        for k, v in raw_state.items():
            nk = _normalize_language_key(k)

            # Ignore non-language branches and MTP branch.
            if nk.startswith("visual.") or nk.startswith("vision_") or nk.startswith("mtp."):
                continue

            if not self._is_implemented_layer_key(nk):
                ignored_unimplemented += 1
                continue

            # HF key -> local module key mapping.
            if nk.startswith("layers.") or nk.startswith("embed_tokens.") or nk.startswith("norm."):
                local_key = f"model.{nk}"
                if ".linear_attn." in local_key:
                    local_key = local_key.replace(".linear_attn.", ".self_attn.")
                if local_key.endswith(".self_attn.conv1d.weight"):
                    local_key = local_key.replace(".self_attn.conv1d.weight", ".self_attn.conv2d.weight")
            elif nk == "lm_head.weight":
                local_key = "lm_head.weight"
            else:
                # Unknown but language-related key, skip for now.
                continue

            mapped_state[local_key] = self._reshape_if_conv_weight(local_key, v)

        missing, unexpected = self.load_state_dict(mapped_state, strict=False)
        missing = [m for m in missing if "rotary.inv_freq" not in m]
        missing = [m for m in missing if m not in {"model.kv_cache_0", "model.linear_conv_state", "model.linear_recurrent_state"}]

        if missing:
            print("Missing keys (implemented path):", missing)
            if unexpected:
                print("Unexpected keys:", unexpected)
            return False

        if ignored_unimplemented > 0:
            print(
                f"Loaded implemented full-attention path; ignored {ignored_unimplemented} "
                "weights from unimplemented branches (linear_attention/vision/mtp)."
            )
        if unexpected:
            print("Unexpected keys:", unexpected)
        return True

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor | None = None,
        current_pos: torch.LongTensor | int | None = None,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            causal_mask=causal_mask,
            position_ids=position_ids,
            current_pos=current_pos,
            IN_PREFILL=IN_PREFILL,
        )
        logits = self.lm_head(hidden_states.permute(0, 2, 1).unsqueeze(2))
        return logits.squeeze(2).permute(0, 2, 1)
