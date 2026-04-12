#!/usr/bin/env python3
"""
ANE Bisection: Incrementally add F-layer components to find what kills ANE.

Background:
  - Standalone MHA (Conv2d Q/K/V/O + matmul + softmax) achieves 65-95% ANE
    at Qwen3.5-4B dimensions (embed=2560, n_head=16).
  - Full F-layer chunk export achieves 0% ANE.
  - Something between these two kills ANE scheduling.

This script adds F-layer components one at a time to isolate the cause:
  V0: Simple uniform MHA (16 heads, head_dim=160)         — baseline
  V1: GQA dimensions (16Q/4KV heads, head_dim=256)        — dimension change
  V2: V1 + Input RMSNorm (doubled trick)
  V3: V1 + RoPE (on-the-fly, partial 128/256)
  V4: V1 + Q/K per-head RMSNorm
  V5: V1 + Gated Q (sigmoid gate on output)
  V6: V1 + Full layer (RMSNorm + residual + FFN)          — all compute, no state
  V7: V1 + ALL extras (RMSNorm+RoPE+QKnorm+gate+res+FFN)  — full compute
  V8: V7 + KV cache write (I/O tensors, dynamic position)
  V9: V7 + KV cache as CoreML state (ct.StateType)
  V10: V7 + Causal mask input
  V11: V9 + V10 + 2 layers                                — realistic chunk

All test in cross_attn/decode mode: 1 token query × CTX cache.
"""

import argparse
import gc
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

import coremltools as ct

REPO_ROOT = "/Volumes/MySSD/Anemll"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ═══════════════════════════════════════════════════════════════════════
#  Qwen3.5-4B dimensions
# ═══════════════════════════════════════════════════════════════════════
HIDDEN_SIZE = 2560
NUM_Q_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
Q_HEAD_DIM = HEAD_DIM * 2  # gated Q: 512 per head
INTERMEDIATE_SIZE = 9216
ROTARY_DIM = 128  # partial rotary: 128 out of 256
EPS = 1e-6

Q_DIM = NUM_Q_HEADS * HEAD_DIM      # 4096
KV_DIM = NUM_KV_HEADS * HEAD_DIM    # 1024
SCALE = 1.0 / math.sqrt(HEAD_DIM)

# Test config
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "ane_bisection")
NUM_WARMUP = 10
NUM_RUNS = 30


# ═══════════════════════════════════════════════════════════════════════
#  Building blocks
# ═══════════════════════════════════════════════════════════════════════

class RMSNormDoubled(nn.Module):
    """Qwen3.5 ANE-friendly doubled RMSNorm: cat([x, -x]) → layer_norm → scale."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled, (2 * self.dim,), None, None, self.eps)
        normed = normed[..., :self.dim]
        return normed * self.weight


class PerHeadNorm(nn.Module):
    """Per-head RMS normalization for Q/K."""
    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps
        self.head_dim = head_dim

    def forward(self, x):
        # x: (B, H, S, D) → normalize last dim
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled, (2 * self.head_dim,), None, None, self.eps)
        normed = normed[..., :self.head_dim]
        return normed * self.weight


def repeat_kv(x, n_rep):
    """Expand KV heads: (B, H_kv, S, D) → (B, H_q, S, D)."""
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1).flatten(1, 2)


def rotate_half(x, half_dim):
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, position_ids, inv_freq, rotary_dim):
    """On-the-fly partial RoPE application."""
    pos = position_ids.float().unsqueeze(-1)  # (1, 1)
    freqs = pos * inv_freq.unsqueeze(0)       # (1, rotary_dim//2)
    emb = torch.cat([freqs, freqs], dim=-1)   # (1, rotary_dim)
    cos = emb.cos().unsqueeze(0).unsqueeze(0) # (1, 1, 1, rotary_dim)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)

    half = rotary_dim // 2
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_rot = q_rot * cos + rotate_half(q_rot, half) * sin
    k_rot = k_rot * cos + rotate_half(k_rot, half) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  Variant definitions
# ═══════════════════════════════════════════════════════════════════════

class V0_SimpleMHA(nn.Module):
    """Pure MHA: uniform 16 heads, head_dim=160. Matches standalone test."""
    def __init__(self):
        super().__init__()
        self.n_head = 16
        self.head_dim = HIDDEN_SIZE // 16  # 160
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, 1, bias=False)
        self.o_proj = nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, 1, bias=False)

    def forward(self, hidden_states, k_cache, v_cache):
        # hidden_states: (1, 1, HIDDEN) BSC
        h = hidden_states.permute(0, 2, 1).unsqueeze(2)  # (1, H, 1, 1)
        q = self.q_proj(h).view(1, self.n_head, self.head_dim, 1).permute(0, 1, 3, 2)
        # k_cache, v_cache: (1, 16, CTX, 160)
        attn = torch.matmul(q, k_cache.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_cache)
        out = out.transpose(1, 2).flatten(2, 3)  # (1, 1, HIDDEN)
        out = out.permute(0, 2, 1).unsqueeze(2)  # BC1S
        out = self.o_proj(out)
        return out.squeeze(2).permute(0, 2, 1)   # BSC


class V1_GQA(nn.Module):
    """GQA dimensions: 16Q/4KV heads, head_dim=256. Q→4096, K/V→1024, O→2560."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Conv2d(HIDDEN_SIZE, Q_DIM, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.o_proj = nn.Conv2d(Q_DIM, HIDDEN_SIZE, 1, bias=False)

    def forward(self, hidden_states, k_cache, v_cache):
        h = hidden_states.permute(0, 2, 1).unsqueeze(2)
        q = self.q_proj(h).view(1, NUM_Q_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        # k_cache, v_cache: (1, 4, CTX, 256) → expand to (1, 16, CTX, 256)
        k_exp = repeat_kv(k_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        v_exp = repeat_kv(v_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).flatten(2, 3)
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.o_proj(out)
        return out.squeeze(2).permute(0, 2, 1)


class V2_GQA_RMSNorm(nn.Module):
    """V1 + Input RMSNorm (doubled trick)."""
    def __init__(self):
        super().__init__()
        self.norm = RMSNormDoubled(HIDDEN_SIZE)
        self.attn = V1_GQA()

    def forward(self, hidden_states, k_cache, v_cache):
        x = self.norm(hidden_states)
        return self.attn(x, k_cache, v_cache)


class V3_GQA_RoPE(nn.Module):
    """V1 + RoPE (on-the-fly, partial 128/256)."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Conv2d(HIDDEN_SIZE, Q_DIM, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.o_proj = nn.Conv2d(Q_DIM, HIDDEN_SIZE, 1, bias=False)
        # RoPE frequencies
        self.register_buffer("inv_freq", 1.0 / (500000.0 ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)))

    def forward(self, hidden_states, k_cache, v_cache, position_ids):
        h = hidden_states.permute(0, 2, 1).unsqueeze(2)
        q = self.q_proj(h).view(1, NUM_Q_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        # We only project Q for RoPE (K is already in cache, pre-RoPE'd)
        # But for fairness, also project and RoPE a dummy K
        k_new = self.k_proj(h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        v_new = self.v_proj(h)  # unused, just to match projection count

        q, k_new = apply_rope(q, k_new, position_ids, self.inv_freq, ROTARY_DIM)

        k_exp = repeat_kv(k_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        v_exp = repeat_kv(v_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).flatten(2, 3)
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.o_proj(out)
        return out.squeeze(2).permute(0, 2, 1)


class V4_GQA_QKNorm(nn.Module):
    """V1 + per-head Q/K RMSNorm."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Conv2d(HIDDEN_SIZE, Q_DIM, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.o_proj = nn.Conv2d(Q_DIM, HIDDEN_SIZE, 1, bias=False)
        self.q_norm = PerHeadNorm(HEAD_DIM)
        self.k_norm = PerHeadNorm(HEAD_DIM)

    def forward(self, hidden_states, k_cache, v_cache):
        h = hidden_states.permute(0, 2, 1).unsqueeze(2)
        q = self.q_proj(h).view(1, NUM_Q_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        k_new = self.k_proj(h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        q = self.q_norm(q)
        k_new = self.k_norm(k_new)
        k_exp = repeat_kv(k_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        v_exp = repeat_kv(v_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).flatten(2, 3)
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.o_proj(out)
        return out.squeeze(2).permute(0, 2, 1)


class V5_GQA_Gate(nn.Module):
    """V1 + Gated Q (sigmoid gate on attention output)."""
    def __init__(self):
        super().__init__()
        # q_proj outputs 2x: head_dim for Q, head_dim for gate
        self.q_proj = nn.Conv2d(HIDDEN_SIZE, NUM_Q_HEADS * Q_HEAD_DIM, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.o_proj = nn.Conv2d(Q_DIM, HIDDEN_SIZE, 1, bias=False)

    def forward(self, hidden_states, k_cache, v_cache):
        h = hidden_states.permute(0, 2, 1).unsqueeze(2)
        q_all = self.q_proj(h).view(1, NUM_Q_HEADS, Q_HEAD_DIM, 1).permute(0, 1, 3, 2)
        q = q_all[..., :HEAD_DIM]
        gate = q_all[..., HEAD_DIM:].permute(0, 2, 1, 3).flatten(2, 3)  # (1, 1, Q_DIM)

        k_exp = repeat_kv(k_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        v_exp = repeat_kv(v_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).flatten(2, 3)  # (1, 1, Q_DIM)

        # Apply sigmoid gate
        out = out * torch.sigmoid(gate)
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.o_proj(out)
        return out.squeeze(2).permute(0, 2, 1)


class V6_GQA_NormResFFN(nn.Module):
    """V1 + RMSNorm + residual + post-norm + FFN. Full compute, no state."""
    def __init__(self):
        super().__init__()
        self.input_norm = RMSNormDoubled(HIDDEN_SIZE)
        self.post_norm = RMSNormDoubled(HIDDEN_SIZE)
        self.attn = V1_GQA()
        # FFN
        self.gate_proj = nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False)
        self.up_proj = nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False)
        self.down_proj = nn.Conv2d(INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, bias=False)

    def forward(self, hidden_states, k_cache, v_cache):
        # Attention with residual
        x = self.input_norm(hidden_states)
        attn_out = self.attn(x, k_cache, v_cache)
        hidden_states = hidden_states + attn_out

        # FFN with residual
        post = self.post_norm(hidden_states)
        h = post.permute(0, 2, 1).unsqueeze(2)
        ffn = F.silu(self.gate_proj(h)) * self.up_proj(h)
        ffn = self.down_proj(ffn)
        ffn = ffn.squeeze(2).permute(0, 2, 1)
        return hidden_states + ffn


class V7_FullLayer(nn.Module):
    """Full F-layer compute: RMSNorm + RoPE + QKNorm + GatedQ + GQA + residual + FFN.
    Stateless KV (passed as regular inputs). 1 layer."""
    def __init__(self):
        super().__init__()
        self.input_norm = RMSNormDoubled(HIDDEN_SIZE)
        self.post_norm = RMSNormDoubled(HIDDEN_SIZE)
        # Attention projections (gated Q)
        self.q_proj = nn.Conv2d(HIDDEN_SIZE, NUM_Q_HEADS * Q_HEAD_DIM, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.o_proj = nn.Conv2d(Q_DIM, HIDDEN_SIZE, 1, bias=False)
        # Per-head norms
        self.q_norm = PerHeadNorm(HEAD_DIM)
        self.k_norm = PerHeadNorm(HEAD_DIM)
        # RoPE
        self.register_buffer("inv_freq", 1.0 / (500000.0 ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)))
        # FFN
        self.gate_proj = nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False)
        self.up_proj = nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False)
        self.down_proj = nn.Conv2d(INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, bias=False)

    def forward(self, hidden_states, k_cache, v_cache, position_ids):
        # Input norm
        x = self.input_norm(hidden_states)

        # Q/K/V projection
        h = x.permute(0, 2, 1).unsqueeze(2)
        q_all = self.q_proj(h).view(1, NUM_Q_HEADS, Q_HEAD_DIM, 1).permute(0, 1, 3, 2)
        q = q_all[..., :HEAD_DIM]
        gate = q_all[..., HEAD_DIM:].permute(0, 2, 1, 3).flatten(2, 3)

        k_new = self.k_proj(h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        _ = self.v_proj(h)  # project V but don't use (K/V already in cache)

        # Q/K norms
        q = self.q_norm(q)
        k_new = self.k_norm(k_new)

        # RoPE
        q, k_new = apply_rope(q, k_new, position_ids, self.inv_freq, ROTARY_DIM)

        # Attention with GQA expansion
        k_exp = repeat_kv(k_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        v_exp = repeat_kv(v_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).flatten(2, 3)

        # Gated output + O projection
        out = out * torch.sigmoid(gate)
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.o_proj(out)
        attn_out = out.squeeze(2).permute(0, 2, 1)

        # Residual
        hidden_states = hidden_states + attn_out

        # FFN
        post = self.post_norm(hidden_states)
        h2 = post.permute(0, 2, 1).unsqueeze(2)
        ffn = F.silu(self.gate_proj(h2)) * self.up_proj(h2)
        ffn = self.down_proj(ffn)
        ffn = ffn.squeeze(2).permute(0, 2, 1)
        return hidden_states + ffn


class V8_KVWrite(nn.Module):
    """V7 + KV cache write at dynamic position (I/O tensors, NOT CoreML state)."""
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.input_norm = RMSNormDoubled(HIDDEN_SIZE)
        self.post_norm = RMSNormDoubled(HIDDEN_SIZE)
        self.q_proj = nn.Conv2d(HIDDEN_SIZE, NUM_Q_HEADS * Q_HEAD_DIM, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.o_proj = nn.Conv2d(Q_DIM, HIDDEN_SIZE, 1, bias=False)
        self.q_norm = PerHeadNorm(HEAD_DIM)
        self.k_norm = PerHeadNorm(HEAD_DIM)
        self.register_buffer("inv_freq", 1.0 / (500000.0 ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)))
        self.gate_proj = nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False)
        self.up_proj = nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False)
        self.down_proj = nn.Conv2d(INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, bias=False)

    def forward(self, hidden_states, k_cache, v_cache, position_ids, current_pos):
        x = self.input_norm(hidden_states)
        h = x.permute(0, 2, 1).unsqueeze(2)

        q_all = self.q_proj(h).view(1, NUM_Q_HEADS, Q_HEAD_DIM, 1).permute(0, 1, 3, 2)
        q = q_all[..., :HEAD_DIM]
        gate = q_all[..., HEAD_DIM:].permute(0, 2, 1, 3).flatten(2, 3)

        k_new = self.k_proj(h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        v_new = self.v_proj(h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)

        q = self.q_norm(q)
        k_new = self.k_norm(k_new)
        q, k_new = apply_rope(q, k_new, position_ids, self.inv_freq, ROTARY_DIM)

        # KV cache write at dynamic position (THIS IS THE SUSPECT)
        pos = current_pos[0]
        k_cache[:, :, pos:pos+1, :] = k_new.squeeze(0)
        v_cache[:, :, pos:pos+1, :] = v_new.squeeze(0)

        k_exp = repeat_kv(k_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        v_exp = repeat_kv(v_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).flatten(2, 3)
        out = out * torch.sigmoid(gate)
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.o_proj(out)
        attn_out = out.squeeze(2).permute(0, 2, 1)

        hidden_states = hidden_states + attn_out
        post = self.post_norm(hidden_states)
        h2 = post.permute(0, 2, 1).unsqueeze(2)
        ffn = F.silu(self.gate_proj(h2)) * self.up_proj(h2)
        ffn = self.down_proj(ffn)
        ffn = ffn.squeeze(2).permute(0, 2, 1)

        return hidden_states + ffn, k_cache, v_cache


class V9_KVState(nn.Module):
    """V7 + KV cache as CoreML states (ct.StateType). Dynamic position write."""
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.input_norm = RMSNormDoubled(HIDDEN_SIZE)
        self.post_norm = RMSNormDoubled(HIDDEN_SIZE)
        self.q_proj = nn.Conv2d(HIDDEN_SIZE, NUM_Q_HEADS * Q_HEAD_DIM, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False)
        self.o_proj = nn.Conv2d(Q_DIM, HIDDEN_SIZE, 1, bias=False)
        self.q_norm = PerHeadNorm(HEAD_DIM)
        self.k_norm = PerHeadNorm(HEAD_DIM)
        self.register_buffer("inv_freq", 1.0 / (500000.0 ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)))
        self.gate_proj = nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False)
        self.up_proj = nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False)
        self.down_proj = nn.Conv2d(INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, bias=False)
        # KV cache as buffers → CoreML states
        self.register_buffer("k_cache", torch.zeros(
            1, NUM_KV_HEADS, ctx, HEAD_DIM, dtype=torch.float32))
        self.register_buffer("v_cache", torch.zeros(
            1, NUM_KV_HEADS, ctx, HEAD_DIM, dtype=torch.float32))

    def forward(self, hidden_states, position_ids, current_pos):
        x = self.input_norm(hidden_states)
        h = x.permute(0, 2, 1).unsqueeze(2)

        q_all = self.q_proj(h).view(1, NUM_Q_HEADS, Q_HEAD_DIM, 1).permute(0, 1, 3, 2)
        q = q_all[..., :HEAD_DIM]
        gate = q_all[..., HEAD_DIM:].permute(0, 2, 1, 3).flatten(2, 3)

        k_new = self.k_proj(h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        v_new = self.v_proj(h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)

        q = self.q_norm(q)
        k_new = self.k_norm(k_new)
        q, k_new = apply_rope(q, k_new, position_ids, self.inv_freq, ROTARY_DIM)

        # KV state write
        pos = current_pos[0]
        self.k_cache[:, :, pos:pos+1, :] = k_new
        self.v_cache[:, :, pos:pos+1, :] = v_new

        k_exp = repeat_kv(self.k_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        v_exp = repeat_kv(self.v_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).flatten(2, 3)
        out = out * torch.sigmoid(gate)
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.o_proj(out)
        attn_out = out.squeeze(2).permute(0, 2, 1)

        hidden_states = hidden_states + attn_out
        post = self.post_norm(hidden_states)
        h2 = post.permute(0, 2, 1).unsqueeze(2)
        ffn = F.silu(self.gate_proj(h2)) * self.up_proj(h2)
        ffn = self.down_proj(ffn)
        ffn = ffn.squeeze(2).permute(0, 2, 1)
        return hidden_states + ffn


class V10_Mask(nn.Module):
    """V7 + causal mask input."""
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.layer = V7_FullLayer()

    def forward(self, hidden_states, k_cache, v_cache, position_ids, causal_mask):
        # Override the attention to add mask
        x = self.layer.input_norm(hidden_states)
        h = x.permute(0, 2, 1).unsqueeze(2)

        q_all = self.layer.q_proj(h).view(1, NUM_Q_HEADS, Q_HEAD_DIM, 1).permute(0, 1, 3, 2)
        q = q_all[..., :HEAD_DIM]
        gate = q_all[..., HEAD_DIM:].permute(0, 2, 1, 3).flatten(2, 3)

        k_new = self.layer.k_proj(h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        _ = self.layer.v_proj(h)

        q = self.layer.q_norm(q)
        k_new = self.layer.k_norm(k_new)
        q, k_new = apply_rope(q, k_new, position_ids, self.layer.inv_freq, ROTARY_DIM)

        k_exp = repeat_kv(k_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        v_exp = repeat_kv(v_cache, NUM_Q_HEADS // NUM_KV_HEADS)
        attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
        attn = attn + causal_mask  # ADD MASK
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).flatten(2, 3)
        out = out * torch.sigmoid(gate)
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.layer.o_proj(out)
        attn_out = out.squeeze(2).permute(0, 2, 1)

        hidden_states = hidden_states + attn_out

        post = self.layer.post_norm(hidden_states)
        h2 = post.permute(0, 2, 1).unsqueeze(2)
        ffn = F.silu(self.layer.gate_proj(h2)) * self.layer.up_proj(h2)
        ffn = self.layer.down_proj(ffn)
        ffn = ffn.squeeze(2).permute(0, 2, 1)
        return hidden_states + ffn


class V11_TwoLayers(nn.Module):
    """Two full layers stacked with KV state and mask — realistic chunk approximation."""
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.layers = nn.ModuleList([self._make_layer() for _ in range(2)])
        # KV cache: (2, KV_HEADS, CTX, HEAD_DIM) for 2 layers
        self.register_buffer("k_cache", torch.zeros(
            2, NUM_KV_HEADS, ctx, HEAD_DIM, dtype=torch.float32))
        self.register_buffer("v_cache", torch.zeros(
            2, NUM_KV_HEADS, ctx, HEAD_DIM, dtype=torch.float32))

    def _make_layer(self):
        return nn.ModuleDict({
            "input_norm": RMSNormDoubled(HIDDEN_SIZE),
            "post_norm": RMSNormDoubled(HIDDEN_SIZE),
            "q_proj": nn.Conv2d(HIDDEN_SIZE, NUM_Q_HEADS * Q_HEAD_DIM, 1, bias=False),
            "k_proj": nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False),
            "v_proj": nn.Conv2d(HIDDEN_SIZE, KV_DIM, 1, bias=False),
            "o_proj": nn.Conv2d(Q_DIM, HIDDEN_SIZE, 1, bias=False),
            "q_norm": PerHeadNorm(HEAD_DIM),
            "k_norm": PerHeadNorm(HEAD_DIM),
            "gate_proj": nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False),
            "up_proj": nn.Conv2d(HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, bias=False),
            "down_proj": nn.Conv2d(INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, bias=False),
        })

    def _forward_layer(self, layer, hidden_states, k_cache_slice, v_cache_slice,
                       position_ids, current_pos, causal_mask, inv_freq):
        x = layer["input_norm"](hidden_states)
        h = x.permute(0, 2, 1).unsqueeze(2)

        q_all = layer["q_proj"](h).view(1, NUM_Q_HEADS, Q_HEAD_DIM, 1).permute(0, 1, 3, 2)
        q = q_all[..., :HEAD_DIM]
        gate = q_all[..., HEAD_DIM:].permute(0, 2, 1, 3).flatten(2, 3)

        k_new = layer["k_proj"](h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        v_new = layer["v_proj"](h).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)

        q = layer["q_norm"](q)
        k_new = layer["k_norm"](k_new)
        q, k_new = apply_rope(q, k_new, position_ids, inv_freq, ROTARY_DIM)

        pos = current_pos[0]
        k_cache_slice[:, :, pos:pos+1, :] = k_new.squeeze(0)
        v_cache_slice[:, :, pos:pos+1, :] = v_new.squeeze(0)

        k_exp = repeat_kv(k_cache_slice, NUM_Q_HEADS // NUM_KV_HEADS)
        v_exp = repeat_kv(v_cache_slice, NUM_Q_HEADS // NUM_KV_HEADS)
        attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
        attn = attn + causal_mask
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).flatten(2, 3)
        out = out * torch.sigmoid(gate)
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = layer["o_proj"](out)
        attn_out = out.squeeze(2).permute(0, 2, 1)

        hidden_states = hidden_states + attn_out
        post = layer["post_norm"](hidden_states)
        h2 = post.permute(0, 2, 1).unsqueeze(2)
        ffn = F.silu(layer["gate_proj"](h2)) * layer["up_proj"](h2)
        ffn = layer["down_proj"](ffn)
        ffn = ffn.squeeze(2).permute(0, 2, 1)
        return hidden_states + ffn

    def forward(self, hidden_states, position_ids, current_pos, causal_mask):
        inv_freq = 1.0 / (500000.0 ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32,
                         device=hidden_states.device) / ROTARY_DIM))
        for i, layer in enumerate(self.layers):
            hidden_states = self._forward_layer(
                layer, hidden_states,
                self.k_cache[i:i+1],
                self.v_cache[i:i+1],
                position_ids, current_pos, causal_mask, inv_freq,
            )
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════
#  Export and measurement
# ═══════════════════════════════════════════════════════════════════════

VARIANT_SPECS = {
    "V0_simple_mha":    {"has_pos": False, "has_mask": False, "has_kv_write": False, "has_state": False, "kv_heads": 16, "kv_head_dim": HIDDEN_SIZE // 16},
    "V1_gqa":           {"has_pos": False, "has_mask": False, "has_kv_write": False, "has_state": False, "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V2_gqa_rmsnorm":   {"has_pos": False, "has_mask": False, "has_kv_write": False, "has_state": False, "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V3_gqa_rope":      {"has_pos": True,  "has_mask": False, "has_kv_write": False, "has_state": False, "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V4_gqa_qknorm":    {"has_pos": False, "has_mask": False, "has_kv_write": False, "has_state": False, "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V5_gqa_gate":      {"has_pos": False, "has_mask": False, "has_kv_write": False, "has_state": False, "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V6_norm_res_ffn":  {"has_pos": False, "has_mask": False, "has_kv_write": False, "has_state": False, "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V7_full_layer":    {"has_pos": True,  "has_mask": False, "has_kv_write": False, "has_state": False, "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V8_kv_write":      {"has_pos": True,  "has_mask": False, "has_kv_write": True,  "has_state": False, "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V9_kv_state":      {"has_pos": True,  "has_mask": False, "has_kv_write": True,  "has_state": True,  "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V10_mask":         {"has_pos": True,  "has_mask": True,  "has_kv_write": False, "has_state": False, "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM},
    "V11_2layers":      {"has_pos": True,  "has_mask": True,  "has_kv_write": True,  "has_state": True,  "kv_heads": NUM_KV_HEADS, "kv_head_dim": HEAD_DIM, "num_layers": 2},
}


def create_model(variant, ctx):
    if variant == "V0_simple_mha":
        return V0_SimpleMHA()
    elif variant == "V1_gqa":
        return V1_GQA()
    elif variant == "V2_gqa_rmsnorm":
        return V2_GQA_RMSNorm()
    elif variant == "V3_gqa_rope":
        return V3_GQA_RoPE()
    elif variant == "V4_gqa_qknorm":
        return V4_GQA_QKNorm()
    elif variant == "V5_gqa_gate":
        return V5_GQA_Gate()
    elif variant == "V6_norm_res_ffn":
        return V6_GQA_NormResFFN()
    elif variant == "V7_full_layer":
        return V7_FullLayer()
    elif variant == "V8_kv_write":
        return V8_KVWrite(ctx)
    elif variant == "V9_kv_state":
        return V9_KVState(ctx)
    elif variant == "V10_mask":
        return V10_Mask(ctx)
    elif variant == "V11_2layers":
        return V11_TwoLayers(ctx)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def create_inputs(variant, spec, ctx):
    """Create sample inputs for tracing and CoreML export."""
    inputs = {}
    inputs["hidden_states"] = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.float32)

    if not spec["has_state"]:
        kv_h = spec["kv_heads"]
        kv_d = spec["kv_head_dim"]
        if spec.get("num_layers", 1) > 1:
            pass  # V11 uses internal state
        elif spec["has_kv_write"]:
            inputs["k_cache"] = torch.randn(1, kv_h, ctx, kv_d, dtype=torch.float32)
            inputs["v_cache"] = torch.randn(1, kv_h, ctx, kv_d, dtype=torch.float32)
        else:
            inputs["k_cache"] = torch.randn(1, kv_h, ctx, kv_d, dtype=torch.float32)
            inputs["v_cache"] = torch.randn(1, kv_h, ctx, kv_d, dtype=torch.float32)

    if spec["has_pos"]:
        inputs["position_ids"] = torch.tensor([ctx // 2], dtype=torch.long)

    if spec["has_kv_write"] and not spec["has_state"] and spec.get("num_layers", 1) == 1:
        inputs["current_pos"] = torch.tensor([ctx // 2], dtype=torch.int32)

    if spec["has_state"] and spec.get("num_layers", 1) > 1:
        inputs["current_pos"] = torch.tensor([ctx // 2], dtype=torch.int32)

    if spec["has_state"] and spec.get("num_layers", 1) == 1:
        inputs["current_pos"] = torch.tensor([ctx // 2], dtype=torch.int32)

    if spec["has_mask"]:
        inputs["causal_mask"] = torch.zeros(1, 1, 1, ctx, dtype=torch.float32)

    return inputs


def get_forward_args(variant, spec, inputs):
    """Get ordered args for model.forward()."""
    if variant == "V0_simple_mha":
        return (inputs["hidden_states"], inputs["k_cache"], inputs["v_cache"])
    elif variant in ("V1_gqa", "V2_gqa_rmsnorm", "V4_gqa_qknorm", "V5_gqa_gate", "V6_norm_res_ffn"):
        return (inputs["hidden_states"], inputs["k_cache"], inputs["v_cache"])
    elif variant in ("V3_gqa_rope", "V7_full_layer"):
        return (inputs["hidden_states"], inputs["k_cache"], inputs["v_cache"], inputs["position_ids"])
    elif variant == "V8_kv_write":
        return (inputs["hidden_states"], inputs["k_cache"], inputs["v_cache"],
                inputs["position_ids"], inputs["current_pos"])
    elif variant == "V9_kv_state":
        return (inputs["hidden_states"], inputs["position_ids"], inputs["current_pos"])
    elif variant == "V10_mask":
        return (inputs["hidden_states"], inputs["k_cache"], inputs["v_cache"],
                inputs["position_ids"], inputs["causal_mask"])
    elif variant == "V11_2layers":
        return (inputs["hidden_states"], inputs["position_ids"],
                inputs["current_pos"], inputs["causal_mask"])
    else:
        raise ValueError(f"Unknown variant: {variant}")


def export_variant(variant, spec, ctx, skip_existing=False):
    """Export a variant to CoreML."""
    name = f"{variant}_ctx{ctx}"
    out_path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
    if skip_existing and os.path.exists(out_path):
        print(f"  [skip] {name} exists")
        return out_path

    model = create_model(variant, ctx)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    inputs = create_inputs(variant, spec, ctx)
    args = get_forward_args(variant, spec, inputs)

    with torch.no_grad():
        traced = torch.jit.trace(model, args)

    # Build CoreML input types
    ct_inputs = []
    ct_inputs.append(ct.TensorType(name="hidden_states",
                                    shape=inputs["hidden_states"].shape, dtype=np.float16))
    if "k_cache" in inputs:
        ct_inputs.append(ct.TensorType(name="k_cache",
                                        shape=inputs["k_cache"].shape, dtype=np.float16))
        ct_inputs.append(ct.TensorType(name="v_cache",
                                        shape=inputs["v_cache"].shape, dtype=np.float16))
    if "position_ids" in inputs:
        ct_inputs.append(ct.TensorType(name="position_ids",
                                        shape=inputs["position_ids"].shape, dtype=np.int32))
    if "current_pos" in inputs:
        ct_inputs.append(ct.TensorType(name="current_pos",
                                        shape=inputs["current_pos"].shape, dtype=np.int32))
    if "causal_mask" in inputs:
        ct_inputs.append(ct.TensorType(name="causal_mask",
                                        shape=inputs["causal_mask"].shape, dtype=np.float16))

    # Build outputs
    ct_outputs = [ct.TensorType(name="output", dtype=np.float16)]
    if spec["has_kv_write"] and not spec["has_state"] and spec.get("num_layers", 1) == 1:
        ct_outputs.append(ct.TensorType(name="k_cache_out", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name="v_cache_out", dtype=np.float16))

    # Build states
    states = None
    if spec["has_state"]:
        n_layers = spec.get("num_layers", 1)
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(n_layers, NUM_KV_HEADS, ctx, HEAD_DIM),
                    dtype=np.float16),
                name="k_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(n_layers, NUM_KV_HEADS, ctx, HEAD_DIM),
                    dtype=np.float16),
                name="v_cache",
            ),
        ]

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mlmodel.save(out_path)
    del mlmodel, traced, model
    gc.collect()
    return out_path


def analyze_mil(path, name):
    """Count key MIL ops."""
    mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = mlmodel.get_spec()
    prog = spec.mlProgram

    op_counts = {}
    hostile = []
    for fn in prog.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                t = op.type
                op_counts[t] = op_counts.get(t, 0) + 1
                if t in ("gather", "scatter", "greater_equal", "select",
                          "read_state", "coreml_update_state"):
                    hostile.append(t)

    total = sum(op_counts.values())
    conv = op_counts.get("conv", 0)
    matmul = op_counts.get("matmul", 0) + op_counts.get("einsum", 0)
    softmax = op_counts.get("softmax", 0)
    hostile_str = ", ".join(f"{h}({hostile.count(h)})" for h in sorted(set(hostile))) if hostile else "NONE"
    print(f"  {name:45s} ops={total:4d} conv={conv:2d} matmul={matmul:2d} softmax={softmax:2d} hostile={hostile_str}")
    del mlmodel
    return total, hostile


def measure_ane(path, name, spec, ctx, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    """Measure wall time and estimate ANE%."""
    mlmodel = ct.models.MLModel(path, compute_units=compute_unit)

    # Build prediction inputs
    pred = {"hidden_states": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float16)}
    kv_h = spec["kv_heads"]
    kv_d = spec["kv_head_dim"]

    if "k_cache" in create_inputs(name.split("_ctx")[0], spec, ctx):
        pred["k_cache"] = np.random.randn(1, kv_h, ctx, kv_d).astype(np.float16)
        pred["v_cache"] = np.random.randn(1, kv_h, ctx, kv_d).astype(np.float16)
    if spec["has_pos"]:
        pred["position_ids"] = np.array([ctx // 2], dtype=np.int32)
    if spec["has_kv_write"] or spec["has_state"]:
        pred["current_pos"] = np.array([ctx // 2], dtype=np.int32)
    if spec["has_mask"]:
        pred["causal_mask"] = np.zeros((1, 1, 1, ctx), dtype=np.float16)

    # Initialize state if needed
    if spec["has_state"]:
        n_layers = spec.get("num_layers", 1)
        state = mlmodel.make_state()
    else:
        state = None

    # Warmup
    for _ in range(NUM_WARMUP):
        if state is not None:
            mlmodel.predict(pred, state=state)
        else:
            mlmodel.predict(pred)

    # Timed runs
    import resource
    times = []
    cpu_times = []
    for _ in range(NUM_RUNS):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        if state is not None:
            mlmodel.predict(pred, state=state)
        else:
            mlmodel.predict(pred)
        t1 = time.perf_counter()
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        times.append(t1 - t0)
        cpu_times.append((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))

    wall_ms = np.median(times) * 1000
    cpu_ms = np.median(cpu_times) * 1000
    cpu_pct = (cpu_ms / wall_ms * 100) if wall_ms > 0 else 0
    ane_pct = max(0, 100 - cpu_pct)
    tag = "ANE" if compute_unit == ct.ComputeUnit.CPU_AND_NE else "CPU"
    print(f"  [{tag}] {name:42s} wall={wall_ms:8.2f}ms cpu={cpu_ms:8.2f}ms  CPU%={cpu_pct:5.1f}%  ANE%={ane_pct:5.1f}%")
    del mlmodel
    gc.collect()
    return wall_ms, cpu_ms, ane_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated variant names to test (default: all)")
    parser.add_argument("--cpu-ref", action="store_true", help="Also measure CPU_ONLY for reference")
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    variants = list(VARIANT_SPECS.keys())
    if args.variants:
        requested = [v.strip() for v in args.variants.split(",")]
        variants = [v for v in variants if v in requested]

    print("=" * 90)
    print(f"  ANE BISECTION: Finding what kills ANE in F-layer")
    print(f"  Variants: {len(variants)}, CTX: {args.ctx}")
    print("=" * 90)

    # Phase 1: Export
    if not args.skip_export:
        print(f"\n{'='*90}")
        print("  PHASE 1: EXPORT")
        print(f"{'='*90}")
        for variant in variants:
            spec = VARIANT_SPECS[variant]
            for ctx in args.ctx:
                name = f"{variant}_ctx{ctx}"
                t0 = time.time()
                try:
                    path = export_variant(variant, spec, ctx, skip_existing=args.skip_existing)
                    elapsed = time.time() - t0
                    print(f"  {name:50s} {elapsed:6.1f}s  ✓")
                except Exception as e:
                    print(f"  {name:50s} FAILED: {e}")
                gc.collect()

    # Phase 2: MIL analysis
    print(f"\n{'='*90}")
    print("  PHASE 2: MIL OP ANALYSIS")
    print(f"{'='*90}")
    for variant in variants:
        for ctx in args.ctx:
            name = f"{variant}_ctx{ctx}"
            path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
            if os.path.exists(path):
                try:
                    analyze_mil(path, name)
                except Exception as e:
                    print(f"  {name:45s} FAILED: {e}")
            else:
                print(f"  {name:45s} MISSING")

    # Phase 3: ANE measurement
    print(f"\n{'='*90}")
    print(f"  PHASE 3: ANE UTILIZATION ({NUM_RUNS} runs, {NUM_WARMUP} warmup)")
    print(f"{'='*90}")
    results = {}
    for ctx in args.ctx:
        print(f"\n  --- CTX={ctx} ---")
        for variant in variants:
            spec = VARIANT_SPECS[variant]
            name = f"{variant}_ctx{ctx}"
            path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
            if not os.path.exists(path):
                print(f"  {name:50s} MISSING")
                continue
            try:
                wall, cpu, ane = measure_ane(path, name, spec, ctx)
                results[(variant, ctx)] = (wall, cpu, ane)
            except Exception as e:
                print(f"  {name:50s} FAILED: {e}")
            gc.collect()

        if args.cpu_ref:
            print(f"\n  --- CTX={ctx} CPU_ONLY reference ---")
            for variant in variants:
                spec = VARIANT_SPECS[variant]
                name = f"{variant}_ctx{ctx}"
                path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
                if not os.path.exists(path):
                    continue
                try:
                    measure_ane(path, name, spec, ctx, compute_unit=ct.ComputeUnit.CPU_ONLY)
                except Exception as e:
                    print(f"  {name:50s} FAILED: {e}")
                gc.collect()

    # Phase 4: Summary table
    print(f"\n{'='*90}")
    print("  SUMMARY: ANE% BY VARIANT AND CTX")
    print(f"{'='*90}")
    header = f"  {'Variant':<30s}"
    for ctx in args.ctx:
        header += f"  {'CTX='+str(ctx):>12s}"
    print(header)
    print("  " + "-" * (30 + 14 * len(args.ctx)))

    for variant in variants:
        row = f"  {variant:<30s}"
        for ctx in args.ctx:
            key = (variant, ctx)
            if key in results:
                wall, cpu, ane = results[key]
                row += f"  {ane:7.1f}% {wall:5.1f}ms"
            else:
                row += f"  {'---':>12s}"
        print(row)

    # Find the transition
    print(f"\n{'='*90}")
    print("  ANE TRANSITION ANALYSIS")
    print(f"{'='*90}")
    for ctx in args.ctx:
        print(f"\n  CTX={ctx}:")
        prev_ane = None
        for variant in variants:
            key = (variant, ctx)
            if key not in results:
                continue
            _, _, ane = results[key]
            if prev_ane is not None:
                delta = ane - prev_ane
                marker = " <<<< SIGNIFICANT DROP" if delta < -15 else ""
                marker = " <<<< ANE KILLED" if delta < -30 else marker
                print(f"    {variant:<30s}  ANE={ane:5.1f}%  Δ={delta:+6.1f}%{marker}")
            else:
                print(f"    {variant:<30s}  ANE={ane:5.1f}%  (baseline)")
            prev_ane = ane

    print(f"\n{'='*90}")
    print("  Done!")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
