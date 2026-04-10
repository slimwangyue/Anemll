"""Gemma4 model implementation for ANEMLL.

This module provides an ANE-optimized implementation of the Gemma4 (E4B/E2B)
text decoder for Apple Neural Engine. All dense layers are expressed as
``nn.Conv2d`` with ``kernel_size=1`` and weights are loaded from Hugging Face
checkpoints with correct reshaping.

Key Gemma4 architecture features (differences from Gemma3):
- Dual head dimensions: head_dim=256 (local/sliding), global_head_dim=512 (full attention)
- Per-Layer Embeddings (PLE): 256-dim gated per-layer token embedding
- KV cache sharing: layers >= first_kv_shared_layer reuse KV from source layers
- Proportional RoPE: only 25% of dims rotated for global layers (partial_rotary_factor=0.25)
- Logit softcapping: tanh(logits/30) * 30 for numerical stability
- Per-layer learned scalar (layer_scalar) for residual scaling
- HuggingFace weight prefix: model.language_model.* (multimodal text backbone)
- Large vocabulary (262,144 tokens) with 16-way LM head splitting
"""

from __future__ import annotations

import os
import json
import math
from typing import Dict

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Gemma4 model constants
# ---------------------------------------------------------------------------

MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 256

FORCE_UNIFIED_CACHE = False
ENABLE_UNIFIED_CACHE = False
ENABLE_SPLIT_CACHE = True
STATE_LENGTH = CONTEXT_LENGTH
DISABLE_KV_CACHE = False

# LM head configuration
ENABLE_CONV2D = bool(1)
ENABLE_VACAB_SPLIT = bool(1)
ENABLE_VACAB_SPLIT8 = bool(0)
ENABLE_VACAB_SPLIT16 = bool(1)
ENABLE_LOGITS2 = bool(1)
ENABLE_COREML = bool(0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Gemma4Config:
    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["Gemma4ForConditionalGeneration"])
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)

        # Tokenizer / specials
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.bos_token_id = kwargs.get("bos_token_id", 2)
        self.eos_token_id = kwargs.get("eos_token_id", 1)
        self.eos_token_ids = kwargs.get("eos_token_ids", [1, 106])

        # Geometry
        self.hidden_act = kwargs.get("hidden_act", "gelu_pytorch_tanh")
        self.hidden_size = kwargs.get("hidden_size", 2560)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 42)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.head_dim = kwargs.get("head_dim", 256)
        self.global_head_dim = kwargs.get("global_head_dim", 512)
        self.intermediate_size = kwargs.get("intermediate_size", 10240)

        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1_000_000.0)
        self.rope_local_base_freq = kwargs.get("rope_local_base_freq", 10_000.0)
        self.query_pre_attn_scalar = kwargs.get("query_pre_attn_scalar", 256)

        self.model_type = kwargs.get("model_type", "gemma4")
        self.transformers_version = kwargs.get("transformers_version", "5.5.0")

        # Gemma4 specific: proportional RoPE
        self.partial_rotary_factor = kwargs.get("partial_rotary_factor", 0.25)

        # Gemma4 specific: logit softcapping
        self.final_logit_softcapping = kwargs.get("final_logit_softcapping", 30.0)

        # Gemma4 specific: per-layer embeddings (PLE)
        self.hidden_size_per_layer_input = kwargs.get("hidden_size_per_layer_input", 256)
        self.vocab_size_per_layer_input = kwargs.get("vocab_size_per_layer_input", 262_144)

        # Gemma4 specific: KV sharing
        self.num_kv_shared_layers = kwargs.get("num_kv_shared_layers", 18)

        # Context / cache
        self.context_length = kwargs.get("context_length", 256)
        self.state_length = kwargs.get("state_length", self.context_length)

        # FP16 residual clamping
        self.enable_residual_clamp = kwargs.get("enable_residual_clamp", False)
        self.residual_clamp_value = kwargs.get("residual_clamp_value", 65504.0)

        # Interleaved attention
        self.sliding_window = kwargs.get("sliding_window", 512)

        if "layer_types" in kwargs:
            self.layer_types = kwargs["layer_types"]
        else:
            default_layer_types = ["sliding_attention"] * self.num_hidden_layers
            for i in range(5, self.num_hidden_layers, 6):
                default_layer_types[i] = "full_attention"
            # Last layer always full attention
            if default_layer_types[-1] != "full_attention":
                default_layer_types[-1] = "full_attention"
            self.layer_types = default_layer_types
            print(f"WARNING: layer_types not in config, using computed defaults for {self.num_hidden_layers} layers")
            print(f"  Global attention at: {[i for i, t in enumerate(default_layer_types) if t == 'full_attention']}")

        # RoPE parameters from nested config (Gemma4 specific)
        rope_params = kwargs.get("rope_parameters", None)
        if rope_params:
            full_attn_params = rope_params.get("full_attention", {})
            sliding_attn_params = rope_params.get("sliding_attention", {})
            self.rope_theta = full_attn_params.get("rope_theta", self.rope_theta)
            self.partial_rotary_factor = full_attn_params.get("partial_rotary_factor", self.partial_rotary_factor)
            self.rope_local_base_freq = sliding_attn_params.get("rope_theta", self.rope_local_base_freq)

        self.attention_size = kwargs.get("attention_size", self.state_length)

        self.use_split_cache = kwargs.get("use_split_cache", ENABLE_SPLIT_CACHE)
        self.single_cache = kwargs.get("single_cache", False)
        if self.single_cache:
            self.use_split_cache = False

        self.use_right_fill_cache = kwargs.get("use_right_fill_cache", False)
        self.force_rotation_mode = kwargs.get("force_rotation_mode", None)

        self.vocab_size = kwargs.get("vocab_size", 262_144)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.use_cache = kwargs.get("use_cache", True)

        self.batch_size = kwargs.get("batch_size", 64)
        self.prefill_dynamic_slice = kwargs.get("prefill_dynamic_slice", False)

        # KV sharing: compute source layer mapping
        self.first_kv_shared_layer_idx = self.num_hidden_layers - self.num_kv_shared_layers
        self._kv_source_map = self._compute_kv_source_map()

        # Count source (non-shared) layers by type
        source_types = self.layer_types[:self.first_kv_shared_layer_idx]
        self.num_source_local_layers = sum(1 for t in source_types if t == "sliding_attention")
        self.num_source_global_layers = sum(1 for t in source_types if t == "full_attention")

    def _compute_kv_source_map(self) -> dict:
        """Compute mapping from shared layer indices to their KV source layer index.

        For each shared layer (>= first_kv_shared_layer_idx), find the last
        non-shared layer of the same type (sliding or full attention).
        """
        if self.num_kv_shared_layers <= 0:
            return {}
        prev_types = self.layer_types[:self.first_kv_shared_layer_idx]
        source_map = {}
        for layer_idx in range(self.first_kv_shared_layer_idx, self.num_hidden_layers):
            target_type = self.layer_types[layer_idx]
            # Find last occurrence of same type in non-shared layers
            source_idx = len(prev_types) - 1 - prev_types[::-1].index(target_type)
            source_map[layer_idx] = source_idx
        return source_map

    def is_kv_shared_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_kv_shared_layer_idx and self.num_kv_shared_layers > 0

    def get_kv_source_layer(self, layer_idx: int) -> int:
        """For a shared layer, return the source layer index whose KV to reuse."""
        return self._kv_source_map[layer_idx]

    def get_global_layer_indices(self) -> list:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    def get_local_layer_indices(self) -> list:
        return [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]

    def get_num_global_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    def get_num_local_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "sliding_attention")

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        # Gemma4 wraps text params under text_config
        if "text_config" in config_dict:
            text_cfg = dict(config_dict.get("text_config", {}))
            if "hidden_activation" in text_cfg and "hidden_act" not in text_cfg:
                text_cfg["hidden_act"] = text_cfg["hidden_activation"]
            for key in ("bos_token_id", "eos_token_id", "pad_token_id", "eos_token_ids"):
                if key in config_dict and key not in text_cfg:
                    text_cfg[key] = config_dict[key]
            config_dict = text_cfg
        return cls(**config_dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_layer_cache_mapping(layer_idx: int, layer_types: list) -> tuple:
    """Map layer index to (cache_type, cache_index) for split cache.

    Only counts *source* layers (non-shared) for cache index computation.
    Shared layers return the same cache index as their source.
    """
    if layer_types[layer_idx] == "full_attention":
        global_idx = sum(1 for i in range(layer_idx)
                        if layer_types[i] == "full_attention")
        return 'global', global_idx
    else:
        local_idx = sum(1 for i in range(layer_idx)
                       if layer_types[i] == "sliding_attention")
        return 'local', local_idx


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Gemma4RMSNorm(nn.Module):
    """ANE-optimized RMSNorm using the doubled-tensor LayerNorm trick.
    
    Uses weight directly (not 1+weight delta pattern) because Gemma4 HF
    checkpoint stores the actual scale values, not deltas from 1.0.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = hidden_states.shape[-1]
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None, bias=None,
            eps=float(self.variance_epsilon),
        )
        normed = normed[..., :hidden_size]
        return normed * self.weight.to(normed.dtype).to(normed.device)


class Gemma4HeadNorm(nn.Module):
    """ANE-optimized per-head RMSNorm.
    
    Uses weight directly (not 1+weight delta pattern) because Gemma4 HF
    checkpoint stores the actual scale values (~0.984), not deltas from 1.0.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = hidden_states.shape[-1]
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None, bias=None,
            eps=float(self.variance_epsilon),
        )
        normed = normed[..., :hidden_size]
        return normed * self.weight.to(normed.dtype).to(normed.device)


class Gemma4HeadNormNoScale(nn.Module):
    """ANE-optimized per-head RMSNorm WITHOUT learned scale (for value normalization)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = hidden_states.shape[-1]
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None, bias=None,
            eps=float(self.variance_epsilon),
        )
        return normed[..., :hidden_size]


class Gemma4RotaryEmbedding(nn.Module):
    """Rotary positional embedding with support for proportional (partial) rotation.

    Local layers: standard full rotation, theta=10k
    Global layers: proportional rotation (only partial_rotary_factor of dims), theta=1e6
    """

    def __init__(self, dim: int, theta: float, max_seq_len: int, partial_rotary_factor: float = 1.0) -> None:
        super().__init__()
        self.dim = dim  # full head_dim
        self.rotary_dim = int(dim * partial_rotary_factor)  # how many dims to actually rotate
        # Ensure even rotary_dim
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=TEST_DEVICE) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, device=TEST_DEVICE, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [max_seq, rotary_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq, rotary_dim]
        self.cos_cached = emb.cos().unsqueeze(0)  # [1, max_seq, rotary_dim]
        self.sin_cached = emb.sin().unsqueeze(0)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor | None = None):
        if position_ids is not None:
            if position_ids.dim() == 1:
                pos_ids = position_ids
            else:
                pos_ids = position_ids.squeeze(0)
            cos = self.cos_cached[:, pos_ids].to(x.dtype)
            sin = self.sin_cached[:, pos_ids].to(x.dtype)
            return cos, sin
        else:
            seq_len = x.shape[1]
            return (
                self.cos_cached[:, :seq_len].to(x.dtype),
                self.sin_cached[:, :seq_len].to(x.dtype),
            )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_partial(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to only the first `rotary_dim` dimensions, pass-through the rest.

    For single-token: cos/sin shape [1, 1, 1, rotary_dim] (no unsqueeze needed).
    For prefill variant use apply_rotary_pos_emb_partial_prefill.
    """
    head_dim = q.shape[-1]
    if rotary_dim >= head_dim:
        # Full rotation — cos/sin already have correct shape
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    # Partial rotation: split → rotate first part → concat
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


def apply_rotary_pos_emb_partial_prefill(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill variant — cos/sin already have shape [1, 1, seq_len, rotary_dim]."""
    head_dim = q.shape[-1]
    if rotary_dim >= head_dim:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1)
    return hidden_states.view(bsz, n_kv * n_rep, seq_len, head_dim)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.up_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.down_proj = nn.Conv2d(self.intermediate_size, self.hidden_size, 1, bias=False, dtype=MODEL_DTYPE)

        def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
        self.act_fn = gelu_tanh

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        a = self.gate_proj(x)
        b = self.up_proj(x)
        d = self.act_fn(a) * b
        # Compute down_proj in float32 to avoid FP16 overflow (316K+ activations)
        # Clamp to FP16 range before converting back — converter applies weight scaling for ANE
        e = F.conv2d(d.float(), self.down_proj.weight.float())
        e = e.clamp(-65504.0, 65504.0).to(MODEL_DTYPE)
        return e.squeeze(2).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Gemma4Attention(nn.Module):
    """Gemma4 attention with dual head dimensions and KV sharing support."""

    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        # Select head dimension based on layer type
        is_global = config.layer_types[layer_idx] == "full_attention"
        self.head_dim = config.global_head_dim if is_global else config.head_dim
        self.is_global = is_global

        if not hasattr(Gemma4Attention, '_config_printed'):
            print(f"Gemma4Attention: local head_dim={config.head_dim}, global head_dim={config.global_head_dim}")
            Gemma4Attention._config_printed = True

        q_proj_dim = self.num_heads * self.head_dim
        kv_proj_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Conv2d(self.hidden_size, q_proj_dim, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(self.hidden_size, kv_proj_dim, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(self.hidden_size, kv_proj_dim, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.o_proj = nn.Conv2d(q_proj_dim, self.hidden_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)

        self.q_norm = Gemma4HeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4HeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4HeadNormNoScale(self.head_dim, eps=config.rms_norm_eps)
        self.scale = 1.0  # Gemma4 uses scaling=1.0 (norm handles magnitude)

        # KV sharing
        self.is_kv_shared = config.is_kv_shared_layer(layer_idx)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """From [num_kv, seq, dim] to [1, num_heads, seq, dim]."""
        x = x.unsqueeze(1).repeat(1, n_rep, 1, 1)
        return x.view(1, -1, x.size(-2), x.size(-1))

    def get_new_kv_cache(self, hidden_states, current_pos, rotary_emb, rotary_dim):
        """Single-token K/V computation with partial RoPE."""
        bsz, q_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        query_states = self.q_proj(hidden_states).view(1, self.num_heads, 1, self.head_dim).to(MODEL_DTYPE)
        key_states = self.k_proj(hidden_states).view(1, self.num_kv_heads, 1, self.head_dim).to(MODEL_DTYPE)
        value_states = self.v_proj(hidden_states).view(1, self.num_kv_heads, 1, self.head_dim).to(MODEL_DTYPE)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb_partial(
            query_states, key_states, cos, sin, rotary_dim
        )
        return query_states, key_states, value_states

    def get_new_kv_cache_prefill(self, hidden_states, current_pos, rotary_emb, rotary_dim):
        """Batch K/V computation for prefill with partial RoPE."""
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        query_states = self.q_proj(hidden_states).view(1, self.num_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        key_states = self.k_proj(hidden_states).view(1, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        value_states = self.v_proj(hidden_states).view(1, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        value_states = self.v_norm(value_states)
        cos, sin = rotary_emb
        cos = cos.permute(0, 2, 1, 3)  # [1, 1, seq_len, rotary_dim]
        sin = sin.permute(0, 2, 1, 3)

        query_states, key_states = apply_rotary_pos_emb_partial_prefill(
            query_states, key_states, cos, sin, rotary_dim
        )
        return query_states.to(MODEL_DTYPE), key_states.to(MODEL_DTYPE), value_states.to(MODEL_DTYPE)

    def forward_regular(self, hidden_states, query_states, kv_cache_layer=None,
                        causal_mask=None, current_pos=None, layer_idx=None):
        """Forward pass for single-token generation."""
        bsz, q_len, _ = hidden_states.shape
        K_layer_cache, V_layer_cache = kv_cache_layer

        use_split_cache = getattr(self.config, 'use_split_cache', ENABLE_SPLIT_CACHE)
        if use_split_cache and layer_idx is not None:
            if self.config.layer_types[layer_idx] == "full_attention":
                window_size = self.config.attention_size
            else:
                window_size = self.config.sliding_window
        else:
            window_size = self.config.attention_size

        K_window = K_layer_cache[..., :window_size, :]
        V_window = V_layer_cache[..., :window_size, :]

        n_rep = self.num_heads // self.num_kv_heads
        key_states = self._repeat_kv(K_window, n_rep)
        value_states = self._repeat_kv(V_window, n_rep)

        attn_weights = torch.matmul(query_states.to(MODEL_DTYPE), key_states.transpose(-1, -2).to(MODEL_DTYPE)) * self.scale
        if causal_mask is not None:
            q_seq_len = query_states.shape[-2]
            k_seq_len = key_states.shape[-2]
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)[:, :, :q_seq_len, :k_seq_len]

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states.to(MODEL_DTYPE))
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)

    def forward_prefill(self, hidden_states, query_states, kv_cache_layer=None,
                        causal_mask=None, layer_idx=None):
        """Forward pass for prefill mode."""
        bsz, q_len, _ = hidden_states.shape
        K_layer_cache, V_layer_cache = kv_cache_layer

        use_split_cache = getattr(self.config, 'use_split_cache', ENABLE_SPLIT_CACHE)
        if use_split_cache and layer_idx is not None:
            if self.config.layer_types[layer_idx] == "full_attention":
                window_size = self.config.attention_size
            else:
                window_size = self.config.sliding_window
        else:
            window_size = self.config.attention_size

        K_window = K_layer_cache[:, :window_size, :]
        V_window = V_layer_cache[:, :window_size, :]

        n_rep = self.num_heads // self.num_kv_heads
        key_states = self._repeat_kv(K_window, n_rep)
        value_states = self._repeat_kv(V_window, n_rep)

        attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states.to(MODEL_DTYPE), key_states.to(MODEL_DTYPE)) * self.scale
        if causal_mask is not None:
            q_seq_len = query_states.shape[2]
            k_seq_len = key_states.shape[2]
            mask_slice = causal_mask.to(MODEL_DTYPE)[:, :, :q_seq_len, :k_seq_len]
            attn_weights = attn_weights + mask_slice

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value_states.to(MODEL_DTYPE))
        attn_output = attn_output.transpose(1, 2).contiguous()
        actual_bsz, actual_seq_len, num_heads, head_dim = attn_output.shape
        attn_output = attn_output.reshape(actual_bsz, actual_seq_len, num_heads * head_dim)
        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config)

        # 4 norms per block (same as Gemma3)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Per-layer learned scalar
        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=MODEL_DTYPE))

        # Per-Layer Embedding (PLE) components
        ple_dim = config.hidden_size_per_layer_input
        if ple_dim > 0:
            self.act_fn = lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
            self.per_layer_input_gate = nn.Conv2d(config.hidden_size, ple_dim, 1, bias=False, dtype=MODEL_DTYPE)
            self.per_layer_projection = nn.Conv2d(ple_dim, config.hidden_size, 1, bias=False, dtype=MODEL_DTYPE)
            self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def apply_ple(self, hidden_states: torch.Tensor, per_layer_emb: torch.Tensor) -> torch.Tensor:
        """Apply Per-Layer Embedding gating and projection.

        Applied AFTER attention + FFN (not before). Gate has act_fn.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            per_layer_emb: [batch, seq_len, ple_dim] — this layer's 256-dim embedding
        """
        if self.config.hidden_size_per_layer_input <= 0:
            return hidden_states

        residual = hidden_states

        # Gate: project hidden → ple_dim, apply activation, then multiply with per-layer embedding
        hs_conv = hidden_states.permute(0, 2, 1).unsqueeze(2)  # [B, H, 1, S]
        gate = self.per_layer_input_gate(hs_conv)  # [B, ple_dim, 1, S]
        gate = gate.squeeze(2).permute(0, 2, 1)  # [B, S, ple_dim]

        # Apply activation to gate (gelu_pytorch_tanh) before multiplying
        gate = self.act_fn(gate)
        gated = gate * per_layer_emb  # [B, S, ple_dim]

        # Project back to hidden_size
        gated_conv = gated.permute(0, 2, 1).unsqueeze(2)  # [B, ple_dim, 1, S]
        contribution = self.per_layer_projection(gated_conv)  # [B, H, 1, S]
        contribution = contribution.squeeze(2).permute(0, 2, 1)  # [B, S, H]

        # Norm and residual add
        contribution = self.post_per_layer_input_norm(contribution)
        return residual + contribution


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        self.disable_kv_cache = False

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=MODEL_DTYPE)
        self.embedding_scale = config.hidden_size ** 0.5

        # Per-Layer Embeddings (PLE) — global components
        ple_dim = config.hidden_size_per_layer_input
        num_layers = config.num_hidden_layers
        if ple_dim > 0:
            total_ple_dim = ple_dim * num_layers  # 256 * 42 = 10752
            self.embed_tokens_per_layer = nn.Embedding(config.vocab_size, total_ple_dim, dtype=MODEL_DTYPE)
            self.per_layer_model_projection = nn.Conv2d(config.hidden_size, total_ple_dim, 1, bias=False, dtype=MODEL_DTYPE)
            self.per_layer_projection_norm = Gemma4RMSNorm(ple_dim, eps=config.rms_norm_eps)
            self.ple_scale = ple_dim ** 0.5

        # Dual RoPE: global (proportional) and local (full rotation)
        max_pos = max(config.context_length, config.state_length) * 2
        self.rotary_emb_global = Gemma4RotaryEmbedding(
            dim=config.global_head_dim,
            theta=config.rope_theta,
            max_seq_len=max_pos,
            partial_rotary_factor=config.partial_rotary_factor,
        )
        self.rotary_emb_local = Gemma4RotaryEmbedding(
            dim=config.head_dim,
            theta=config.rope_local_base_freq,
            max_seq_len=max_pos,
            partial_rotary_factor=1.0,  # full rotation for local
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # KV cache initialization (split cache with dual head dims)
        # Only allocate cache for source (non-shared) layers
        num_source_local = config.num_source_local_layers
        num_source_global = config.num_source_global_layers

        # Build source layer index lists (only layers < first_kv_shared_layer_idx)
        self._source_local_indices = [
            i for i in range(config.first_kv_shared_layer_idx)
            if config.layer_types[i] == "sliding_attention"
        ]
        self._source_global_indices = [
            i for i in range(config.first_kv_shared_layer_idx)
            if config.layer_types[i] == "full_attention"
        ]

        # Also build full layer lists (including shared) for cache lookup mapping
        self._all_local_indices = [i for i in range(config.num_hidden_layers)
                                   if config.layer_types[i] == "sliding_attention"]
        self._all_global_indices = [i for i in range(config.num_hidden_layers)
                                    if config.layer_types[i] == "full_attention"]

        # Local cache: [2 * num_source_local, kv_heads, sliding_window, head_dim]
        local_cache_shape = (
            2 * num_source_local,
            config.num_key_value_heads,
            config.sliding_window,
            config.head_dim,  # local head_dim = 256
        )
        self.register_buffer("kv_cache_local",
                           torch.zeros(local_cache_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE))

        # Global cache: [2 * num_source_global, kv_heads, state_length, global_head_dim]
        global_cache_shape = (
            2 * num_source_global,
            config.num_key_value_heads,
            config.state_length,
            config.global_head_dim,  # global head_dim = 512
        )
        self.register_buffer("kv_cache_global",
                           torch.zeros(global_cache_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE))

        if not hasattr(Gemma4Model, '_cache_init_printed'):
            print(f"Gemma4 SPLIT KV caches (source layers only):")
            print(f"  kv_cache_local:  {self.kv_cache_local.shape} for {num_source_local} source local layers {self._source_local_indices}")
            print(f"  kv_cache_global: {self.kv_cache_global.shape} for {num_source_global} source global layers {self._source_global_indices}")
            print(f"  KV shared layers ({config.num_kv_shared_layers}): {config.first_kv_shared_layer_idx}-{config.num_hidden_layers-1}")
            print(f"  KV source map sample: {dict(list(config._kv_source_map.items())[:5])}")
            Gemma4Model._cache_init_printed = True

    # ─── RoPE helpers ────────────────────────────────────────────────────

    def get_rotary_embeddings_s(self, current_pos, layer_idx=0):
        """Get rotary embeddings for single-token at current_pos."""
        if self.config.layer_types[layer_idx] == "full_attention":
            rotary_emb = self.rotary_emb_global
        else:
            rotary_emb = self.rotary_emb_local
        sin = rotary_emb.sin_cached[:, current_pos:current_pos + 1].view(1, 1, 1, -1)
        cos = rotary_emb.cos_cached[:, current_pos:current_pos + 1].view(1, 1, 1, -1)
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def get_rotary_embedding_prefill(self, positions, layer_idx=0):
        """Get rotary embeddings for a batch of positions."""
        if self.config.layer_types[layer_idx] == "full_attention":
            rotary_emb = self.rotary_emb_global
        else:
            rotary_emb = self.rotary_emb_local
        if positions.dim() == 2:
            seq_len = positions.size(1)
            pos_indices = positions.squeeze(0)
        else:
            seq_len = positions.size(0)
            pos_indices = positions
        cos = rotary_emb.cos_cached[:, pos_indices].view(1, seq_len, 1, rotary_emb.rotary_dim)
        sin = rotary_emb.sin_cached[:, pos_indices].view(1, seq_len, 1, rotary_emb.rotary_dim)
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def _get_rotary_dim(self, layer_idx: int) -> int:
        """Get the rotary dimension for a layer (partial for global, full for local)."""
        if self.config.layer_types[layer_idx] == "full_attention":
            return self.rotary_emb_global.rotary_dim
        else:
            return self.rotary_emb_local.rotary_dim

    # ─── PLE helpers ─────────────────────────────────────────────────────

    def compute_per_layer_embeddings(self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor = None) -> torch.Tensor | None:
        """Compute all per-layer embeddings from input_ids and inputs_embeds.

        Follows HF reference: combines token-based PLE with hidden-state projection.
        Returns: [batch, seq_len, num_layers, ple_dim] or None if PLE disabled.
        """
        if self.config.hidden_size_per_layer_input <= 0:
            return None
        ple_dim = self.config.hidden_size_per_layer_input
        num_layers = self.config.num_hidden_layers

        # 1. Token-based per-layer embeddings (scaled by sqrt(ple_dim) inside embedding)
        ple = self.embed_tokens_per_layer(input_ids)  # [B, S, total_ple_dim]
        ple = ple * self.ple_scale  # scale by sqrt(256)
        bsz, seq_len, _ = ple.shape
        ple = ple.view(bsz, seq_len, num_layers, ple_dim)  # [B, S, 42, 256]

        # 2. Model projection from inputs_embeds
        if inputs_embeds is not None:
            # Project hidden states to PLE space, scaled by hidden_size^-0.5
            ie_conv = inputs_embeds.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)  # [B, H, 1, S]
            proj = self.per_layer_model_projection(ie_conv)  # [B, total_ple_dim, 1, S]
            proj = proj.squeeze(2).permute(0, 2, 1)  # [B, S, total_ple_dim]
            proj = proj * (self.config.hidden_size ** -0.5)
            proj = proj.view(bsz, seq_len, num_layers, ple_dim)
            proj = self.per_layer_projection_norm(proj)

            # 3. Combine: (projection + token_ple) * rsqrt(2)
            ple = (proj + ple) * (2.0 ** -0.5)
        else:
            # Fallback: just use token-based with norm
            ple = self.per_layer_projection_norm(ple)

        return ple

    def get_per_layer_emb_slice(self, per_layer_emb: torch.Tensor | None, layer_idx: int) -> torch.Tensor | None:
        """Get the PLE slice for a specific layer.

        Returns: [batch, seq_len, ple_dim] or None.
        """
        if per_layer_emb is None:
            return None
        return per_layer_emb[:, :, layer_idx, :]

    # ─── KV cache helpers ────────────────────────────────────────────────

    def _get_source_cache_idx(self, layer_idx: int) -> tuple:
        """Get (cache_type, cache_index) for a layer, resolving KV sharing.

        For source layers: returns (type, index_in_source_list).
        For shared layers: returns (type, index_of_source_in_source_list).
        """
        # Resolve to source layer if shared
        effective_layer = layer_idx
        if self.config.is_kv_shared_layer(layer_idx):
            effective_layer = self.config.get_kv_source_layer(layer_idx)

        if self.config.layer_types[effective_layer] == "full_attention":
            cache_idx = self._source_global_indices.index(effective_layer)
            return 'global', cache_idx
        else:
            cache_idx = self._source_local_indices.index(effective_layer)
            return 'local', cache_idx

    def _apply_update_mask(self, cache_slice, new_states, update_mask):
        mask = update_mask.to(dtype=cache_slice.dtype)
        expanded = new_states.expand(cache_slice.shape[0], new_states.shape[1], cache_slice.shape[2], new_states.shape[3])
        return cache_slice * (1.0 - mask) + expanded * mask

    def _apply_update_mask_batch(self, cache_slice, new_states, update_mask):
        mask = update_mask.to(dtype=cache_slice.dtype)
        mask = mask.expand(cache_slice.shape[0], cache_slice.shape[1], cache_slice.shape[2], mask.shape[-1])
        updates = torch.einsum("bhkd,bhlk->bhld", new_states, mask)
        mask_sum = torch.clamp(mask.sum(dim=-1, keepdim=True), max=1.0)
        return cache_slice * (1.0 - mask_sum) + updates

    def _fill_kv_local(self, layer_idx, key_states, value_states, update_mask_local):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        assert cache_type == 'local'
        num_source_local = len(self._source_local_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_local
        k_slice = self.kv_cache_local[k_idx:k_idx + 1]
        v_slice = self.kv_cache_local[v_idx:v_idx + 1]
        self.kv_cache_local[k_idx:k_idx + 1, :, :, :] = self._apply_update_mask(k_slice, key_states, update_mask_local)
        self.kv_cache_local[v_idx:v_idx + 1, :, :, :] = self._apply_update_mask(v_slice, value_states, update_mask_local)

    def _update_kv_local(self, layer_idx, key_states, value_states):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        assert cache_type == 'local'
        num_source_local = len(self._source_local_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_local
        sw = self.config.sliding_window
        k_slice = self.kv_cache_local[k_idx:k_idx + 1]
        k_tail = torch.narrow(k_slice, 2, 1, sw - 1)
        self.kv_cache_local[k_idx:k_idx + 1, :, :, :] = torch.cat([k_tail, key_states], dim=2)
        v_slice = self.kv_cache_local[v_idx:v_idx + 1]
        v_tail = torch.narrow(v_slice, 2, 1, sw - 1)
        self.kv_cache_local[v_idx:v_idx + 1, :, :, :] = torch.cat([v_tail, value_states], dim=2)

    def _store_kv_local(self, layer_idx, key_states, value_states, current_pos, update_mask=None):
        force_rotation = getattr(self.config, 'force_rotation_mode', None)
        if force_rotation is True:
            self._update_kv_local(layer_idx, key_states, value_states)
        elif force_rotation is False:
            if update_mask is not None:
                update_mask_local = torch.narrow(update_mask, 2, 0, self.config.sliding_window)
                self._fill_kv_local(layer_idx, key_states, value_states, update_mask_local)
            else:
                cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
                num_source_local = len(self._source_local_indices)
                k_idx = cache_idx
                v_idx = cache_idx + num_source_local
                self.kv_cache_local[k_idx:k_idx + 1, :, current_pos:current_pos + 1, :] = key_states
                self.kv_cache_local[v_idx:v_idx + 1, :, current_pos:current_pos + 1, :] = value_states
        else:
            sliding_window = self.config.sliding_window
            if current_pos < sliding_window:
                if update_mask is not None:
                    update_mask_local = torch.narrow(update_mask, 2, 0, self.config.sliding_window)
                    self._fill_kv_local(layer_idx, key_states, value_states, update_mask_local)
                else:
                    cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
                    num_source_local = len(self._source_local_indices)
                    k_idx = cache_idx
                    v_idx = cache_idx + num_source_local
                    self.kv_cache_local[k_idx:k_idx + 1, :, current_pos:current_pos + 1, :] = key_states
                    self.kv_cache_local[v_idx:v_idx + 1, :, current_pos:current_pos + 1, :] = value_states
            else:
                self._update_kv_local(layer_idx, key_states, value_states)

    def _store_kv_local_prefill(self, layer_idx, key_states, value_states, current_pos, seq_len):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        num_source_local = len(self._source_local_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_local
        sw = self.config.sliding_window
        if seq_len > sw:
            start = seq_len - sw
            key_states = torch.narrow(key_states, 1, start, sw)
            value_states = torch.narrow(value_states, 1, start, sw)
            seq_len = sw
        if getattr(self.config, "prefill_dynamic_slice", False):
            self.kv_cache_local[k_idx:k_idx + 1, :, current_pos:current_pos + seq_len, :] = key_states[:, :seq_len, :]
            self.kv_cache_local[v_idx:v_idx + 1, :, current_pos:current_pos + seq_len, :] = value_states[:, :seq_len, :]
            return
        end_pos = min(seq_len, sw)
        self.kv_cache_local[k_idx:k_idx + 1, :, 0:end_pos, :] = key_states[:, :end_pos, :]
        self.kv_cache_local[v_idx:v_idx + 1, :, 0:end_pos, :] = value_states[:, :end_pos, :]

    def _store_kv_local_prefill_masked(self, layer_idx, key_states, value_states, update_mask_local):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        num_source_local = len(self._source_local_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_local
        k_slice = self.kv_cache_local[k_idx:k_idx + 1]
        v_slice = self.kv_cache_local[v_idx:v_idx + 1]
        self.kv_cache_local[k_idx:k_idx + 1, :, :, :] = self._apply_update_mask_batch(k_slice, key_states, update_mask_local)
        self.kv_cache_local[v_idx:v_idx + 1, :, :, :] = self._apply_update_mask_batch(v_slice, value_states, update_mask_local)

    def _update_kv_local_prefill(self, layer_idx, key_states, value_states, seq_len):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        num_source_local = len(self._source_local_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_local
        sw = self.config.sliding_window
        batch_size = self.config.batch_size
        k_slice = self.kv_cache_local[k_idx:k_idx + 1]
        k_tail = k_slice[:, :, batch_size:sw, :]
        self.kv_cache_local[k_idx:k_idx + 1, :, :, :] = torch.cat([k_tail, key_states[:, :, :batch_size, :]], dim=2)
        v_slice = self.kv_cache_local[v_idx:v_idx + 1]
        v_tail = v_slice[:, :, batch_size:sw, :]
        self.kv_cache_local[v_idx:v_idx + 1, :, :, :] = torch.cat([v_tail, value_states[:, :, :batch_size, :]], dim=2)

    def _store_kv_global(self, layer_idx, key_states, value_states, current_pos, update_mask=None):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        assert cache_type == 'global'
        num_source_global = len(self._source_global_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_global
        if update_mask is None:
            pos = current_pos  # ANE fix: min() generates greater_equal+select that block ANE
            self.kv_cache_global[k_idx:k_idx + 1, :, pos:pos + 1, :] = key_states
            self.kv_cache_global[v_idx:v_idx + 1, :, pos:pos + 1, :] = value_states
        else:
            update_mask_global = torch.narrow(update_mask, 2, 0, self.config.state_length)
            k_slice = self.kv_cache_global[k_idx:k_idx + 1]
            v_slice = self.kv_cache_global[v_idx:v_idx + 1]
            self.kv_cache_global[k_idx:k_idx + 1, :, :, :] = self._apply_update_mask(k_slice, key_states, update_mask_global)
            self.kv_cache_global[v_idx:v_idx + 1, :, :, :] = self._apply_update_mask(v_slice, value_states, update_mask_global)

    def _store_kv_global_prefill(self, layer_idx, key_states, value_states, current_pos, seq_len):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        num_source_global = len(self._source_global_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_global
        sl = self.config.state_length
        if seq_len > sl:
            start = seq_len - sl
            key_states = torch.narrow(key_states, 2, start, sl)
            value_states = torch.narrow(value_states, 2, start, sl)
            seq_len = sl
        if getattr(self.config, "prefill_dynamic_slice", False):
            self.kv_cache_global[k_idx:k_idx + 1, :, current_pos:current_pos + seq_len, :] = key_states[:, :seq_len, :]
            self.kv_cache_global[v_idx:v_idx + 1, :, current_pos:current_pos + seq_len, :] = value_states[:, :seq_len, :]
            return
        end_pos = min(seq_len, sl)
        self.kv_cache_global[k_idx:k_idx + 1, :, 0:end_pos, :] = key_states[:, :end_pos, :]
        self.kv_cache_global[v_idx:v_idx + 1, :, 0:end_pos, :] = value_states[:, :end_pos, :]

    def _store_kv_global_prefill_masked(self, layer_idx, key_states, value_states, update_mask_global):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        num_source_global = len(self._source_global_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_global
        k_slice = self.kv_cache_global[k_idx:k_idx + 1]
        v_slice = self.kv_cache_global[v_idx:v_idx + 1]
        self.kv_cache_global[k_idx:k_idx + 1, :, :, :] = self._apply_update_mask_batch(k_slice, key_states, update_mask_global)
        self.kv_cache_global[v_idx:v_idx + 1, :, :, :] = self._apply_update_mask_batch(v_slice, value_states, update_mask_global)

    def _update_kv_global(self, layer_idx, key_states, value_states):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        num_source_global = len(self._source_global_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_global
        sl = self.config.state_length
        k_slice = self.kv_cache_global[k_idx:k_idx + 1]
        k_tail = torch.narrow(k_slice, 2, 1, sl - 1)
        self.kv_cache_global[k_idx:k_idx + 1, :, :, :] = torch.cat([k_tail, key_states], dim=2)
        v_slice = self.kv_cache_global[v_idx:v_idx + 1]
        v_tail = torch.narrow(v_slice, 2, 1, sl - 1)
        self.kv_cache_global[v_idx:v_idx + 1, :, :, :] = torch.cat([v_tail, value_states], dim=2)

    def _update_kv_global_prefill(self, layer_idx, key_states, value_states, seq_len):
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        num_source_global = len(self._source_global_indices)
        k_idx = cache_idx
        v_idx = cache_idx + num_source_global
        sl = self.config.state_length
        actual_store_len = min(seq_len, sl)
        if seq_len >= sl:
            self.kv_cache_global[k_idx:k_idx + 1, :, :, :] = key_states[:, :, -sl:, :]
            self.kv_cache_global[v_idx:v_idx + 1, :, :, :] = value_states[:, :, -sl:, :]
        else:
            k_slice = self.kv_cache_global[k_idx:k_idx + 1]
            k_tail = torch.narrow(k_slice, 2, seq_len, sl - seq_len)
            self.kv_cache_global[k_idx:k_idx + 1, :, :, :] = torch.cat([k_tail, key_states[:, :, :actual_store_len, :]], dim=2)
            v_slice = self.kv_cache_global[v_idx:v_idx + 1]
            v_tail = torch.narrow(v_slice, 2, seq_len, sl - seq_len)
            self.kv_cache_global[v_idx:v_idx + 1, :, :, :] = torch.cat([v_tail, value_states[:, :, :actual_store_len, :]], dim=2)

    def _get_kv_cache_for_layer(self, layer_idx, current_pos=0):
        """Get (key_cache, value_cache) for a layer, resolving KV sharing."""
        cache_type, cache_idx = self._get_source_cache_idx(layer_idx)
        if cache_type == 'global':
            num_source_global = len(self._source_global_indices)
            k = self.kv_cache_global[cache_idx:cache_idx + 1].squeeze(0)
            v = self.kv_cache_global[cache_idx + num_source_global:cache_idx + num_source_global + 1].squeeze(0)
            return k, v
        else:
            num_source_local = len(self._source_local_indices)
            k = self.kv_cache_local[cache_idx:cache_idx + 1].squeeze(0)
            v = self.kv_cache_local[cache_idx + num_source_local:cache_idx + num_source_local + 1].squeeze(0)
            return k, v

    # ─── Layer processing ────────────────────────────────────────────────

    def _process_layer_core(self, layer_idx, hidden_states, per_layer_emb, query_states,
                            key_states, value_states, causal_mask, is_prefill):
        """Core layer processing: PLE → attention → MLP (shared by all modes)."""
        layer = self.layers[layer_idx]

        # Apply Per-Layer Embedding
        ple_slice = self.get_per_layer_emb_slice(per_layer_emb, layer_idx)
        if ple_slice is not None:
            hidden_states = layer.apply_ple(hidden_states, ple_slice)

        # Attention has already been computed (query_states from normalized hidden)
        # The attention output is computed in the caller and passed through
        # ... actually, we do normalization + attention here per the Gemma3 pattern

        return hidden_states

    def process_layer_prefill(self, layer_idx, hidden_states, position_ids, causal_mask,
                              current_pos, per_layer_emb=None, update_mask=None):
        layer = self.layers[layer_idx]

        rotary_emb = self.get_rotary_embedding_prefill(position_ids, layer_idx)
        rotary_dim = self._get_rotary_dim(layer_idx)

        # --- Attention ---
        residual = hidden_states
        normalized_states = layer.input_layernorm(hidden_states)

        is_shared = self.config.is_kv_shared_layer(layer_idx)

        if is_shared:
            query_states, _, _ = layer.self_attn.get_new_kv_cache_prefill(
                normalized_states, current_pos, rotary_emb, rotary_dim
            )
        else:
            query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
                normalized_states, current_pos, rotary_emb, rotary_dim
            )
            seq_length = key_states.shape[2]

            if update_mask is not None:
                if self.config.layer_types[layer_idx] == "full_attention":
                    um = torch.narrow(update_mask, 2, 0, self.config.state_length)
                    self._store_kv_global_prefill_masked(layer_idx, key_states, value_states, um)
                else:
                    um = torch.narrow(update_mask, 2, 0, self.config.sliding_window)
                    self._store_kv_local_prefill_masked(layer_idx, key_states, value_states, um)
            else:
                if self.config.layer_types[layer_idx] == "full_attention":
                    self._store_kv_global_prefill(layer_idx, key_states, value_states, current_pos, seq_length)
                else:
                    self._store_kv_local_prefill(layer_idx, key_states, value_states, current_pos, seq_length)

        key_cache, value_cache = self._get_kv_cache_for_layer(layer_idx, current_pos)

        attn_output = layer.self_attn.forward_prefill(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
            layer_idx=layer_idx,
        )

        attn_output = layer.post_attention_layernorm(attn_output)
        hidden_states = residual + attn_output

        # --- FFN ---
        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # --- PLE ---
        ple_slice = self.get_per_layer_emb_slice(per_layer_emb, layer_idx)
        if ple_slice is not None:
            hidden_states = layer.apply_ple(hidden_states, ple_slice)

        # --- Layer scalar ---
        hidden_states = hidden_states * layer.layer_scalar

        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, -self.config.residual_clamp_value, self.config.residual_clamp_value)

        return hidden_states

    def process_layer_prefill_rotate(self, layer_idx, hidden_states, position_ids, causal_mask,
                                     current_pos, per_layer_emb=None, update_mask=None):
        layer = self.layers[layer_idx]

        rotary_emb = self.get_rotary_embedding_prefill(position_ids, layer_idx)
        rotary_dim = self._get_rotary_dim(layer_idx)

        # --- Attention ---
        residual = hidden_states
        normalized_states = layer.input_layernorm(hidden_states)

        is_shared = self.config.is_kv_shared_layer(layer_idx)
        if is_shared:
            query_states, _, _ = layer.self_attn.get_new_kv_cache_prefill(
                normalized_states, current_pos, rotary_emb, rotary_dim
            )
        else:
            query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
                normalized_states, current_pos, rotary_emb, rotary_dim
            )
            seq_length = key_states.shape[2]
            if self.config.layer_types[layer_idx] == "full_attention":
                self._store_kv_global_prefill(layer_idx, key_states, value_states, current_pos, seq_length)
            else:
                self._update_kv_local_prefill(layer_idx, key_states, value_states, seq_length)

        key_cache, value_cache = self._get_kv_cache_for_layer(layer_idx, current_pos)

        attn_output = layer.self_attn.forward_prefill(
            hidden_states=normalized_states, query_states=query_states,
            kv_cache_layer=(key_cache, value_cache), causal_mask=causal_mask, layer_idx=layer_idx,
        )
        attn_output = layer.post_attention_layernorm(attn_output)
        hidden_states = residual + attn_output

        # --- FFN ---
        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # --- PLE ---
        ple_slice = self.get_per_layer_emb_slice(per_layer_emb, layer_idx)
        if ple_slice is not None:
            hidden_states = layer.apply_ple(hidden_states, ple_slice)

        # --- Layer scalar ---
        hidden_states = hidden_states * layer.layer_scalar

        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, -self.config.residual_clamp_value, self.config.residual_clamp_value)
        return hidden_states

    def process_layer_regular(self, layer_idx, hidden_states, position_ids, causal_mask,
                              current_pos, per_layer_emb=None, update_mask=None):
        layer = self.layers[layer_idx]
        seq_len = hidden_states.shape[1]
        if seq_len == 1:
            rotary_emb = self.get_rotary_embeddings_s(current_pos, layer_idx)
        else:
            rotary_emb = self.get_rotary_embedding_prefill(position_ids, layer_idx)
        rotary_dim = self._get_rotary_dim(layer_idx)

        # --- Attention ---
        residual = hidden_states
        normalized_states = layer.input_layernorm(hidden_states)

        is_shared = self.config.is_kv_shared_layer(layer_idx)

        if is_shared:
            if seq_len == 1:
                query_states, _, _ = layer.self_attn.get_new_kv_cache(
                    normalized_states, current_pos, rotary_emb, rotary_dim
                )
            else:
                query_states, _, _ = layer.self_attn.get_new_kv_cache_prefill(
                    normalized_states, current_pos, rotary_emb, rotary_dim
                )
        else:
            if seq_len == 1:
                query_states, key_states, value_states = layer.self_attn.get_new_kv_cache(
                    normalized_states, current_pos, rotary_emb, rotary_dim
                )
            else:
                query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
                    normalized_states, current_pos, rotary_emb, rotary_dim
                )

            if not self.disable_kv_cache:
                if seq_len == 1:
                    if self.config.layer_types[layer_idx] == "full_attention":
                        self._store_kv_global(layer_idx, key_states, value_states, current_pos, update_mask)
                    else:
                        self._store_kv_local(layer_idx, key_states, value_states, current_pos, update_mask)
                else:
                    if self.config.layer_types[layer_idx] == "full_attention":
                        self._store_kv_global_prefill(layer_idx, key_states, value_states, current_pos, seq_len)
                    else:
                        self._store_kv_local_prefill(layer_idx, key_states, value_states, current_pos, seq_len)

        key_cache, value_cache = self._get_kv_cache_for_layer(layer_idx, current_pos)

        if seq_len == 1:
            attn_output = layer.self_attn.forward_regular(
                hidden_states=normalized_states, query_states=query_states,
                kv_cache_layer=(key_cache, value_cache), causal_mask=causal_mask,
                current_pos=current_pos, layer_idx=layer_idx,
            )
        else:
            attn_output = layer.self_attn.forward_prefill(
                hidden_states=normalized_states, query_states=query_states,
                kv_cache_layer=(key_cache, value_cache), causal_mask=causal_mask,
                layer_idx=layer_idx,
            )

        attn_output = layer.post_attention_layernorm(attn_output)
        hidden_states = residual + attn_output

        # --- FFN ---
        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # --- PLE (applied AFTER attention + FFN, per HF reference) ---
        ple_slice = self.get_per_layer_emb_slice(per_layer_emb, layer_idx)
        if ple_slice is not None:
            hidden_states = layer.apply_ple(hidden_states, ple_slice)

        # --- Layer scalar (multiplies entire output once, no residual add) ---
        hidden_states = hidden_states * layer.layer_scalar
        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, -self.config.residual_clamp_value, self.config.residual_clamp_value)

        return hidden_states

    def process_layer(self, layer_idx, hidden_states, position_ids, causal_mask, current_pos,
                      per_layer_emb=None, IN_PREFILL=False, IN_PREFILL_ROTATE=False, update_mask=None):
        if IN_PREFILL_ROTATE:
            return self.process_layer_prefill_rotate(
                layer_idx, hidden_states, position_ids, causal_mask, current_pos, per_layer_emb, update_mask
            )
        elif IN_PREFILL:
            return self.process_layer_prefill(
                layer_idx, hidden_states, position_ids, causal_mask, current_pos, per_layer_emb, update_mask
            )
        else:
            return self.process_layer_regular(
                layer_idx, hidden_states, position_ids, causal_mask, current_pos, per_layer_emb, update_mask
            )

    def process_layers(self, hidden_states, position_ids, causal_mask, current_pos,
                       input_ids=None, per_layer_emb=None, start_layer=0, end_layer=None,
                       IN_PREFILL=False, IN_PREFILL_ROTATE=False, update_mask=None):
        """Process a range of transformer layers"""
        if end_layer is None:
            end_layer = len(self.layers)

        hidden_states = hidden_states.to(MODEL_DTYPE)

        for i in range(start_layer, end_layer):
            hidden_states = self.process_layer(
                i, hidden_states, position_ids, causal_mask, current_pos,
                per_layer_emb, IN_PREFILL, IN_PREFILL_ROTATE, update_mask
            )
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
        update_mask: torch.Tensor | None = None,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.embedding_scale

        # Compute per-layer embeddings once
        per_layer_emb = self.compute_per_layer_embeddings(input_ids, inputs_embeds=hidden_states)

        hidden_states = self.process_layers(
            hidden_states, position_ids, causal_mask, current_pos,
            per_layer_emb=per_layer_emb,
            start_layer=0, end_layer=None,
            IN_PREFILL=IN_PREFILL, update_mask=update_mask,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def forward_prefill(
        self, hidden_states, position_ids=None, causal_mask=None, current_pos=None,
        per_layer_emb=None, start_layer=None, end_layer=None, update_mask=None,
    ):
        batch_size, seq_length, _ = hidden_states.size()
        if start_layer is not None and end_layer is not None:
            hidden_states = self.process_layers(
                hidden_states, position_ids, causal_mask, current_pos,
                per_layer_emb=per_layer_emb,
                start_layer=start_layer, end_layer=end_layer,
                IN_PREFILL=True, update_mask=update_mask,
            )
        else:
            hidden_states = self.process_layers(
                hidden_states, position_ids, causal_mask, current_pos,
                per_layer_emb=per_layer_emb,
                IN_PREFILL=True, update_mask=update_mask,
            )
        if end_layer is None or end_layer == len(self.layers):
            hidden_states = self.norm(hidden_states)
        return hidden_states

    # ─── Weight loading ──────────────────────────────────────────────────

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(
                    safetensors.torch.load_file(os.path.join(model_path, file))
                )

        conv_state = {}
        for k, v in state_dict.items():
            # Strip multimodal prefix: model.language_model.* → *
            if k.startswith("model.language_model."):
                k = k[len("model.language_model."):]
            elif k.startswith("language_model."):
                k = k[len("language_model."):]
            elif k.startswith("model."):
                k = k[len("model."):]
            else:
                # Skip non-text weights (audio_tower, vision_tower, etc.)
                continue
            if "lm_head.weight" in k:
                continue

            # Conv2d reshape for projection weights
            if any(proj in k for proj in [
                "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                "gate_proj.weight", "up_proj.weight", "down_proj.weight",
                "per_layer_input_gate.weight", "per_layer_projection.weight",
                "per_layer_model_projection.weight",
            ]):
                conv_state[k] = v.view(v.shape[0], v.shape[1], 1, 1)
            else:
                conv_state[k] = v

        missing, unexpected = self.load_state_dict(conv_state, strict=False)
        # Filter expected missing keys
        missing = [m for m in missing if "rotary_emb" not in m and "kv_cache" not in m]

        allow_missing = os.environ.get("ANEMLL_ALLOW_MISSING_WEIGHTS", "").lower() in ("1", "true", "yes")
        if missing:
            print(f"Missing keys ({len(missing)}):", missing[:20])
            if allow_missing:
                print("Continuing despite missing weights (ANEMLL_ALLOW_MISSING_WEIGHTS=1).")
                return True
            raise RuntimeError(f"Failed to load Gemma4 weights: {len(missing)} missing keys.")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}):", unexpected[:10])
        return True


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------

class Gemma4ForCausalLM(nn.Module):
    config_class = Gemma4Config

    def __init__(self, config: Gemma4Config, enable_coreml=False, disable_kv_cache=False, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.enable_coreml = enable_coreml
        self.disable_kv_cache = disable_kv_cache or DISABLE_KV_CACHE

        if enable_coreml:
            global ENABLE_COREML
            ENABLE_COREML = True

        self.model = Gemma4Model(config)
        self.model.disable_kv_cache = self.disable_kv_cache

        # LM head: 16-way split for 262K vocab
        if ENABLE_CONV2D and ENABLE_VACAB_SPLIT16:
            vocab_split = config.vocab_size // 16
            vocab_remainder = config.vocab_size % 16
            self.lm_head_split = 16
            for i in range(16):
                split_size = vocab_split + (1 if i < vocab_remainder else 0)
                setattr(self, f"lm_head16_{i+1}",
                       nn.Conv2d(config.hidden_size, split_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE))
            if not hasattr(Gemma4ForCausalLM, '_lm_head_printed'):
                print("Created lm_head16_1 through lm_head16_16")
                Gemma4ForCausalLM._lm_head_printed = True
        elif ENABLE_CONV2D:
            self.lm_head1 = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
            self.lm_head_split = 1
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=MODEL_DTYPE)
            self.lm_head_split = 1

        # Logit softcapping
        self.final_logit_softcapping = config.final_logit_softcapping

    def _apply_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        if self.final_logit_softcapping is not None and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        update_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        current_pos: torch.LongTensor,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        assert len(input_ids.shape) == 2

        hidden_states = self.model(
            input_ids, causal_mask, position_ids, current_pos, update_mask, IN_PREFILL=IN_PREFILL,
        )

        if not IN_PREFILL and current_pos is not None:
            seq_len = hidden_states.shape[1]
            if seq_len == 1:
                pos_tensor = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
            else:
                if isinstance(current_pos, torch.Tensor):
                    pos_tensor = current_pos if current_pos.dim() > 0 else current_pos.unsqueeze(0)
                else:
                    pos_tensor = torch.tensor([current_pos], device=hidden_states.device, dtype=torch.long)
            hidden_states = torch.index_select(hidden_states, dim=1, index=pos_tensor)

        # LM head projection
        if ENABLE_CONV2D:
            hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

            if ENABLE_VACAB_SPLIT16:
                logits_list = []
                for i in range(16):
                    head = getattr(self, f"lm_head16_{i+1}")
                    logits_list.append(head(hidden_states).squeeze(2).transpose(1, 2))

                # Apply softcapping to each logit chunk, inputs_embeds=hidden_states
                logits_list = [self._apply_softcapping(l) for l in logits_list]

                if self.enable_coreml and ENABLE_LOGITS2:
                    return tuple(logits_list)
                else:
                    logits = torch.cat(logits_list, dim=2)
            else:
                logits = self.lm_head1(hidden_states).squeeze(2).transpose(1, 2)
                logits = self._apply_softcapping(logits)
        else:
            logits = self.lm_head(hidden_states)
            logits = self._apply_softcapping(logits)

        return logits

    def prefill_kv_cache(self, input_ids, position_ids, start_pos, causal_mask):
        batch_size, seq_length = input_ids.shape
        hidden_states = self.model.embed_tokens(input_ids)
        hidden_states = hidden_states * self.model.embedding_scale
        hidden_states = hidden_states.to(MODEL_DTYPE)

        # Compute PLE
        per_layer_emb = self.model.compute_per_layer_embeddings(input_ids, inputs_embeds=hidden_states)

        if causal_mask is not None:
            causal_mask_prefill = causal_mask[:, :, :seq_length, :]
        else:
            causal_mask_prefill = None

        mask_len = max(
            self.model.config.state_length,
            getattr(self.model.config, "sliding_window", 0) or 0,
        )
        update_mask = torch.zeros(
            (1, 1, mask_len, seq_length), dtype=MODEL_DTYPE, device=TEST_DEVICE,
        )
        for i in range(seq_length):
            pos = start_pos + i
            if pos < mask_len:
                update_mask[0, 0, pos, i] = 1.0

        with torch.no_grad():
            self.model.forward_prefill(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask_prefill,
                current_pos=start_pos,
                per_layer_emb=per_layer_emb,
                update_mask=update_mask,
            )

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not self.model.load_pretrained_weights(model_path):
            return False

        # Load lm_head weights
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(
                    safetensors.torch.load_file(os.path.join(model_path, file))
                )

        # Find lm_head weight or fall back to embed_tokens
        lm_head_weight = None
        embed_tokens_key = None
        for k, v in state_dict.items():
            if k == "lm_head.weight":
                lm_head_weight = v
            if "embed_tokens.weight" in k and "per_layer" not in k:
                embed_tokens_key = k

        if lm_head_weight is None:
            if self.config.tie_word_embeddings and embed_tokens_key is not None:
                print(f"lm_head.weight not found, using {embed_tokens_key} (tie_word_embeddings=True)")
                lm_head_weight = state_dict[embed_tokens_key].clone()
            else:
                print("WARNING: lm_head.weight not found")
                return False

        if ENABLE_CONV2D:
            reshaped = lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1)
            if ENABLE_VACAB_SPLIT16:
                vocab_split = self.config.vocab_size // 16
                vocab_remainder = self.config.vocab_size % 16
                split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]
                splits = torch.split(reshaped, split_sizes)
                for i, split in enumerate(splits):
                    getattr(self, f"lm_head16_{i+1}").weight.data.copy_(split)
            else:
                self.lm_head1.weight.data.copy_(reshaped)
        else:
            self.lm_head.weight.data.copy_(lm_head_weight)

        return True
