#!/usr/bin/env python3
"""
Experiment: Measure ANE utilization impact of Apple's Principles 2 & 3 on chunk2 (FLLL, layers 7-10).

Variants:
  A) baseline          — current model, no changes
  B) p3_direct_layout  — Principle 3: reduce transposes in L-layer layout stage
                          (BCHW → (B,nH,dH,S) → transpose last 2 instead of BSH intermediate)
  C) p3_fused_proj     — Principle 3: fuse 4 L-layer input projections into 1 Conv2d + split
                          AND fuse MLP gate+up into 1 Conv2d + split (fewer Conv2d dispatches)
  D) p3_combined       — B + C combined (direct layout + fused projections)
  E) p2_perhead_attn   — Principle 2: split F-layer attention into per-head computations
                          (Q/K/V split into list of per-head tensors, matmul per-head)
  F) p2p3_all          — All optimizations combined (D + E)

Each variant exports chunk2 decode + prefill, measures:
  - MIL op counts (transpose, reshape, conv, matmul)
  - Wall time, CPU time, ANE time (wall - cpu)
  - ANE utilization %
  - Correctness vs baseline (cosine similarity)
"""
import sys, os, time, gc, resource, warnings, argparse, json, shutil
from collections import Counter
from typing import Tuple
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Volumes/MySSD/Anemll')
sys.path.insert(0, '/Volumes/MySSD/Anemll/scripts_qwen3_5')
os.chdir('/Volumes/MySSD/Anemll')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_grad_enabled(False)

import coremltools as ct
from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

HIDDEN = 2560
MODEL_DTYPE = torch.float16
HF_MODEL = 'models/Qwen__Qwen3.5-4B'
CHUNK_IDX = 2
CHUNK2_START, CHUNK2_END = CHUNK_RANGES[CHUNK_IDX]
NUM_LAYERS_CHUNK2 = CHUNK2_END - CHUNK2_START  # 4

OUT_DIR = 'artifacts/p2_p3_ane_impact_chunk2'

import anemll.models.qwen3_5_model as qm
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config,
    Qwen35FullAttention, Qwen35LinearAttention,
    Qwen35LinearProjStage, Qwen35LinearLayoutStage,
    Qwen35LinearCoreNormStage, Qwen35MLP,
    Qwen35RMSNorm, Qwen35RMSNormGated,
    _l2norm, _repeat_kv,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

# ============================================================
# Save original methods for restore
# ============================================================
_orig_layout_fwd = Qwen35LinearLayoutStage.forward
_orig_proj_fwd = Qwen35LinearProjStage.forward
_orig_mlp_fwd = Qwen35MLP.forward
_orig_recurrent = Qwen35LinearAttention._recurrent_gated_delta_rule
_orig_full_attn_project_qkvg = Qwen35FullAttention._project_qkvg
_orig_full_attn_forward_regular = Qwen35FullAttention.forward_regular
_orig_full_attn_forward_prefill = Qwen35FullAttention.forward_prefill
_orig_full_attn_forward = Qwen35FullAttention.forward

# ============================================================
# Principle 3: Direct layout (fewer transposes in L-layer)
# ============================================================
def p3_direct_layout_forward(self, conv_out_cf, z_cf, b_cf, a_cf, bsz, seq_len, force_fp16_math=False):
    """Direct BCHW → (B,nH,S,dH) without BSH intermediate. Saves ~4 layout ops per L-layer.

    For decode (seq_len=1): returns Q/K/V as (B,nH,1,dH) — compatible with p3_direct_recurrent.
    For prefill (seq_len>1): returns Q/K/V as (B,S,nH,dH) — compatible with masking + _chunk_gated_delta_rule.
    """
    query_cf, key_cf, value_cf = torch.split(
        conv_out_cf, [self.key_dim, self.key_dim, self.value_dim], dim=1
    )
    # Direct: (B, nH*dH, 1, S) → (B, nH, dH, S) → transpose(-2,-1) → (B, nH, S, dH)
    # Direct: (B, nH*dH, 1, S) → (B, nH, dH, S) → transpose(-2,-1) → (B, nH, S, dH)
    query = query_cf.reshape(bsz, self.num_k_heads, self.head_k_dim, seq_len).transpose(-2, -1)
    key = key_cf.reshape(bsz, self.num_k_heads, self.head_k_dim, seq_len).transpose(-2, -1)
    value = value_cf.reshape(bsz, self.num_v_heads, self.head_v_dim, seq_len).transpose(-2, -1)
    # GQA repeat must happen while still in (B, nH, S, dH) format (dim=1 is heads)
    if self.num_v_heads // self.num_k_heads > 1:
        rep = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(rep, dim=1)
        key = key.repeat_interleave(rep, dim=1)
    if seq_len > 1:
        # Prefill: transpose to (B,S,nH,dH) for masking and _chunk_gated_delta_rule compatibility
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
    # z still needs from_channels_first for the norm
    z = self.from_channels_first_4d(z_cf).reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
    b = self.from_channels_first_4d(b_cf)
    a = self.from_channels_first_4d(a_cf)
    beta = b.sigmoid()
    if force_fp16_math:
        x = a.to(MODEL_DTYPE) + self.dt_bias
        sp = F.relu(x) + torch.log(1.0 + torch.exp(-torch.abs(x)))
        g = -self.A_log.to(MODEL_DTYPE).exp() * sp
    else:
        x = a.float() + self.dt_bias
        sp = F.relu(x) + torch.log(1.0 + torch.exp(-torch.abs(x)))
        g = -self.A_log.float().exp() * sp
    return query, key, value, g, beta, z

@staticmethod
def p3_direct_recurrent(query, key, value, g, beta, recurrent_state,
    output_final_state=True, expected_batch_size=None, expected_num_heads=None,
    expected_seq_len=None, expected_k_dim=None, expected_v_dim=None,
    math_dtype=torch.float32):
    """Recurrence where layout already provides (B,nH,S,d) — skip entry .transpose(1,2)."""
    initial_dtype = query.dtype
    query, key, value = query.contiguous().to(math_dtype), key.contiguous().to(math_dtype), value.contiguous().to(math_dtype)
    beta = beta.transpose(1, 2).contiguous().to(math_dtype)  # beta still BSH
    g = g.transpose(1, 2).contiguous().to(math_dtype)        # g still BSH
    query = _l2norm(query, dim=-1)
    key = _l2norm(key, dim=-1)
    bsz = expected_batch_size or key.shape[0]
    n_heads = expected_num_heads or key.shape[1]
    seq_len = expected_seq_len or key.shape[2]
    k_dim = expected_k_dim or key.shape[-1]
    v_dim = expected_v_dim or value.shape[-1]
    scale = 1 / (k_dim ** 0.5)
    query = query * scale
    out = torch.zeros(bsz, n_heads, seq_len, v_dim, dtype=value.dtype, device=value.device)
    state = recurrent_state.to(value)
    for i in range(seq_len):
        q_t, k_t, v_t = query[:,:,i], key[:,:,i], value[:,:,i]
        g_t = g[:,:,i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:,:,i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:,:,i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state: state = None
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state

# ============================================================
# Principle 3: Fused projections (fewer Conv2d dispatches)
# ============================================================
def prepare_fused_proj(proj_stage):
    """Pre-create fused 4-way projection weight."""
    w_qkv = proj_stage.in_proj_qkv.weight.data
    w_z = proj_stage.in_proj_z.weight.data
    w_b = proj_stage.in_proj_b.weight.data
    w_a = proj_stage.in_proj_a.weight.data
    fused_w = torch.cat([w_qkv, w_z, w_b, w_a], dim=0)
    fused = nn.Conv2d(HIDDEN, fused_w.shape[0], 1, bias=False, dtype=MODEL_DTYPE)
    fused.weight.data.copy_(fused_w)
    proj_stage._fused_proj = fused
    proj_stage._split_sizes = [w_qkv.shape[0], w_z.shape[0], w_b.shape[0], w_a.shape[0]]

def prepare_fused_mlp(mlp):
    """Pre-create fused gate+up weight."""
    w_gate = mlp.gate_proj.weight.data
    w_up = mlp.up_proj.weight.data
    fused_w = torch.cat([w_gate, w_up], dim=0)
    fused = nn.Conv2d(mlp.hidden_size, fused_w.shape[0], 1, bias=False, dtype=MODEL_DTYPE)
    fused.weight.data.copy_(fused_w)
    mlp._fused_gate_up = fused
    mlp._gate_size = w_gate.shape[0]

def p3_fused_proj_forward(self, hidden_states):
    hidden_cf = self.to_channels_first_4d(hidden_states).to(MODEL_DTYPE)
    fused_out = self._fused_proj(hidden_cf)
    qkv, z, b, a = torch.split(fused_out, self._split_sizes, dim=1)
    return qkv, z, b, a

def p3_fused_mlp_forward(self, x):
    x = x.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
    fused = self._fused_gate_up(x)
    gate_out, up_out = torch.split(fused, [self._gate_size, self._gate_size], dim=1)
    gated = F.silu(gate_out) * up_out
    out = self.down_proj(gated)
    return out.squeeze(2).permute(0, 2, 1)

# ============================================================
# Principle 2: Per-head attention (split Q/K/V, matmul per head)
# ============================================================
def p2_perhead_project_qkvg(self, hidden_states):
    """Same as original but output per-head list instead of batched tensor."""
    hs = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
    q_all = (
        self.q_proj(hs)
        .view(1, self.num_heads, self.head_dim * 2, -1)
        .permute(0, 1, 3, 2)
    )
    query_states = q_all[..., :self.head_dim]
    gate = q_all[..., self.head_dim:].permute(0, 2, 1, 3).flatten(2, 3)
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

def p2_perhead_forward_regular(self, hidden_states, query_states, kv_cache_layer, causal_mask=None, gate=None):
    """Per-head attention: iterate over heads, matmul individually for L2 residency."""
    k_cache, v_cache = kv_cache_layer
    k_cache = k_cache[..., :self.config.state_length, :]
    v_cache = v_cache[..., :self.config.state_length, :]

    n_rep = self.num_heads // self.num_kv_heads

    # Per-head computation
    head_outputs = []
    for h in range(self.num_heads):
        kv_h = h // n_rep
        q_h = query_states[:, h:h+1, :, :].to(MODEL_DTYPE)  # [B,1,S,D]
        k_h = k_cache[kv_h:kv_h+1, :, :].unsqueeze(0).to(MODEL_DTYPE)  # [B,1,CTX,D]
        v_h = v_cache[kv_h:kv_h+1, :, :].unsqueeze(0).to(MODEL_DTYPE)  # [B,1,CTX,D]
        attn_w = torch.matmul(q_h, k_h.transpose(-1, -2)) * self.scale
        if causal_mask is not None:
            attn_w = attn_w + causal_mask.to(MODEL_DTYPE)
        attn_w = torch.softmax(attn_w, dim=-1)
        out_h = torch.matmul(attn_w, v_h)  # [B,1,S,D]
        head_outputs.append(out_h)

    attn_output = torch.cat(head_outputs, dim=1)  # [B,H,S,D]
    attn_output = attn_output.transpose(1, 2).contiguous().flatten(2, 3)
    return self._project_output(attn_output, hidden_states, gate=gate)

def p2_perhead_forward_prefill(self, hidden_states, query_states, kv_cache_layer, causal_mask=None, gate=None):
    """Per-head prefill attention."""
    k_cache, v_cache = kv_cache_layer
    k_cache = k_cache[..., :self.config.state_length, :]
    v_cache = v_cache[..., :self.config.state_length, :]

    n_rep = self.num_heads // self.num_kv_heads

    head_outputs = []
    for h in range(self.num_heads):
        kv_h = h // n_rep
        q_h = query_states[:, h:h+1, :, :].to(MODEL_DTYPE)
        k_h = k_cache[kv_h:kv_h+1, :, :].unsqueeze(0).to(MODEL_DTYPE)
        v_h = v_cache[kv_h:kv_h+1, :, :].unsqueeze(0).to(MODEL_DTYPE)
        attn_w = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            attn_w = attn_w + causal_mask.to(MODEL_DTYPE)
        attn_w = torch.softmax(attn_w, dim=-1)
        out_h = torch.matmul(attn_w, v_h)
        head_outputs.append(out_h)

    attn_output = torch.cat(head_outputs, dim=1)
    attn_output = attn_output.transpose(1, 2).contiguous().flatten(2, 3)
    return self._project_output(attn_output, hidden_states, gate=gate)

def p2_perhead_forward(self, hidden_states, causal_mask, position_ids):
    """Per-head standalone forward (for non-cached path)."""
    bsz, seq_len, _ = hidden_states.shape
    query_states, key_states, value_states, gate = self._project_qkvg(hidden_states)
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    cos, sin = self.rotary.get(hidden_states, position_ids)
    from anemll.models.qwen3_5_model import _apply_rotary_partial
    query_states, key_states = _apply_rotary_partial(
        query_states, key_states, cos, sin, self.rotary.rotary_dim
    )

    n_rep = self.num_heads // self.num_kv_heads
    # Per-head
    head_outputs = []
    for h in range(self.num_heads):
        kv_h = h // n_rep
        q_h = query_states[:, h:h+1, :, :]
        k_h = key_states[:, kv_h:kv_h+1, :, :]
        v_h = value_states[:, kv_h:kv_h+1, :, :]
        attn_w = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            attn_w = attn_w + causal_mask[:, :, :seq_len, :seq_len].to(attn_w.dtype)
        attn_w = torch.softmax(attn_w, dim=-1).to(v_h.dtype)
        out_h = torch.matmul(attn_w, v_h)
        head_outputs.append(out_h)

    attn_output = torch.cat(head_outputs, dim=1)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, -1)
    return self._project_output(attn_output, hidden_states, gate=gate)


# ============================================================
# Variant setup/teardown helpers
# ============================================================
def setup_p3_direct_layout():
    Qwen35LinearLayoutStage.forward = p3_direct_layout_forward
    Qwen35LinearAttention._recurrent_gated_delta_rule = p3_direct_recurrent

def teardown_p3_direct_layout():
    Qwen35LinearLayoutStage.forward = _orig_layout_fwd
    Qwen35LinearAttention._recurrent_gated_delta_rule = _orig_recurrent

def setup_p3_fused_proj(model):
    for i in range(CHUNK2_START, CHUNK2_END):
        layer = model.model.layers[i]
        if hasattr(layer.self_attn, 'proj_stage'):
            prepare_fused_proj(layer.self_attn.proj_stage)
        prepare_fused_mlp(layer.mlp)
    Qwen35LinearProjStage.forward = p3_fused_proj_forward
    Qwen35MLP.forward = p3_fused_mlp_forward

def teardown_p3_fused_proj():
    Qwen35LinearProjStage.forward = _orig_proj_fwd
    Qwen35MLP.forward = _orig_mlp_fwd

def setup_p2_perhead():
    Qwen35FullAttention.forward_regular = p2_perhead_forward_regular
    Qwen35FullAttention.forward_prefill = p2_perhead_forward_prefill
    Qwen35FullAttention.forward = p2_perhead_forward

def teardown_p2_perhead():
    Qwen35FullAttention.forward_regular = _orig_full_attn_forward_regular
    Qwen35FullAttention.forward_prefill = _orig_full_attn_forward_prefill
    Qwen35FullAttention.forward = _orig_full_attn_forward


# ============================================================
# Export function
# ============================================================
def export_variant(model, name, setup_fn, teardown_fn, skip_existing=False):
    """Export decode + prefill for one variant."""
    decode_path = os.path.join(OUT_DIR, f'{name}_decode.mlpackage')
    prefill_path = os.path.join(OUT_DIR, f'{name}_prefill.mlpackage')

    if skip_existing and os.path.exists(decode_path) and os.path.exists(prefill_path):
        print(f"  {name}: cached (decode + prefill)")
        return decode_path, prefill_path

    setup_fn()

    # Decode
    if not (skip_existing and os.path.exists(decode_path)):
        print(f"  {name} decode: exporting...")
        t0 = time.time()
        conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                               num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
                               compute_precision="float16")
        ml = conv.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS,
                                 override_start_layer=CHUNK2_START, override_end_layer=CHUNK2_END)
        if os.path.exists(decode_path):
            shutil.rmtree(decode_path)
        ml.save(decode_path)
        del ml, conv; gc.collect()
        print(f"    Saved in {time.time()-t0:.1f}s")
    else:
        print(f"  {name} decode: cached")

    # Prefill
    if not (skip_existing and os.path.exists(prefill_path)):
        print(f"  {name} prefill: exporting...")
        t0 = time.time()
        conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                               num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
                               compute_precision="float16")
        ml = conv.convert_part_2_prefill(model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS,
                                         override_start_layer=CHUNK2_START, override_end_layer=CHUNK2_END)
        if os.path.exists(prefill_path):
            shutil.rmtree(prefill_path)
        ml.save(prefill_path)
        del ml, conv; gc.collect()
        print(f"    Saved in {time.time()-t0:.1f}s")
    else:
        print(f"  {name} prefill: cached")

    teardown_fn()
    return decode_path, prefill_path


# ============================================================
# MIL op analysis
# ============================================================
def analyze_mil_ops(path):
    """Return dict of op type → count for compute ops only."""
    spec = ct.utils.load_spec(path)
    mlprog = spec.mlProgram
    results = {}
    for fn_name in mlprog.functions:
        func = mlprog.functions[fn_name]
        for k in func.block_specializations:
            block = func.block_specializations[k]
            break
        op_counts = Counter()
        for op in block.operations:
            op_counts[op.type] += 1
        total = sum(op_counts.values())
        weight = sum(v for k, v in op_counts.items() if k in ('const', 'constexpr_lut_to_dense'))
        compute = total - weight
        results[fn_name] = {
            'total': total, 'weight': weight, 'compute': compute,
            'transpose': op_counts.get('transpose', 0),
            'reshape': op_counts.get('reshape', 0),
            'squeeze': op_counts.get('squeeze', 0),
            'expand_dims': op_counts.get('expand_dims', 0),
            'conv': op_counts.get('conv', 0),
            'matmul': op_counts.get('matmul', 0),
            'mul': op_counts.get('mul', 0),
            'split': op_counts.get('split', 0),
            'cast': op_counts.get('cast', 0),
            'reduce_sum': op_counts.get('reduce_sum', 0),
            'reduce_mean': op_counts.get('reduce_mean', 0),
        }
    return results


# ============================================================
# Timing measurement
# ============================================================
def measure_decode(path, n_warmup=5, n_runs=30):
    """Measure decode (seq_len=1) latency and ANE utilization."""
    nl = NUM_LAYERS_CHUNK2
    np.random.seed(42)
    inputs = {
        'hidden_states': np.random.randn(1, 1, HIDDEN).astype(np.float16) * 0.01,
        'position_ids': np.array([0], dtype=np.int32),
        'causal_mask': np.zeros((1, 1, 1, CTX), dtype=np.float16),
        'current_pos': np.array([0], dtype=np.int32),
        'linear_conv_state': np.zeros((nl, 1024, 32), dtype=np.float16),
        'linear_recurrent_state': np.zeros((nl, 32, 128, 128), dtype=np.float16),
    }

    ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = ml.make_state()
    for _ in range(n_warmup):
        ml.predict(inputs, state=state)

    times = []
    for _ in range(n_runs):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        ml.predict(inputs, state=state)
        wall = (time.perf_counter() - t0) * 1000
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)) * 1000
        times.append((wall, cpu))

    # Also get first-call output for accuracy
    state2 = ml.make_state()
    out = ml.predict(inputs, state=state2)

    del ml; gc.collect()

    walls = [t[0] for t in times]
    cpus = [t[1] for t in times]
    w = np.median(walls)
    c = np.median(cpus)
    a = max(0, w - c)
    return {
        'wall_ms': float(w),
        'cpu_ms': float(c),
        'ane_ms': float(a),
        'ane_pct': float(a / w * 100) if w > 0 else 0,
        'output': {k: np.asarray(v).flatten().astype(np.float64) for k, v in out.items()
                   if 'hidden_states' in k},
    }


def measure_prefill(path, n_warmup=3, n_runs=10):
    """Measure prefill (seq_len=BATCH_SIZE) latency and ANE utilization."""
    nl = NUM_LAYERS_CHUNK2
    np.random.seed(42)
    inputs = {
        'hidden_states': np.random.randn(1, BATCH_SIZE, HIDDEN).astype(np.float16) * 0.01,
        'position_ids': np.arange(BATCH_SIZE, dtype=np.int32),
        'causal_mask': np.zeros((1, 1, BATCH_SIZE, CTX), dtype=np.float16),
        'current_pos': np.array([BATCH_SIZE - 1], dtype=np.int32),
        'linear_conv_state': np.zeros((nl, 1024, 32), dtype=np.float16),
        'linear_recurrent_state': np.zeros((nl, 32, 128, 128), dtype=np.float16),
        'valid_len': np.array([BATCH_SIZE], dtype=np.int32),
    }

    ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = ml.make_state()

    for _ in range(n_warmup):
        ml.predict(inputs, state=state)

    times = []
    for _ in range(n_runs):
        state_fresh = ml.make_state()
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        ml.predict(inputs, state=state_fresh)
        wall = (time.perf_counter() - t0) * 1000
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)) * 1000
        times.append((wall, cpu))

    del ml; gc.collect()

    walls = [t[0] for t in times]
    cpus = [t[1] for t in times]
    w = np.median(walls)
    c = np.median(cpus)
    a = max(0, w - c)
    return {
        'wall_ms': float(w),
        'cpu_ms': float(c),
        'ane_ms': float(a),
        'ane_pct': float(a / w * 100) if w > 0 else 0,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--skip-export', action='store_true')
    parser.add_argument('--variants', type=str, default=None,
                        help='Comma-separated list of variants to run (default: all)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Define variants
    all_variants = ['A_baseline', 'B_p3_direct_layout', 'C_p3_fused_proj',
                    'D_p3_combined', 'E_p2_perhead_attn', 'F_p2p3_all']

    if args.variants:
        selected = [v.strip() for v in args.variants.split(',')]
    else:
        selected = all_variants

    if not args.skip_export:
        print("=" * 70)
        print(f"  LOADING MODEL for chunk2 export (layers {CHUNK2_START}-{CHUNK2_END-1}, FLLL)")
        print("=" * 70)
        cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
        cfg.context_length = CTX; cfg.state_length = CTX
        model = Qwen35ForCausalLM(cfg)
        assert model.load_pretrained_weights(HF_MODEL)
        model.eval()
        for p in model.parameters(): p.requires_grad = False

        print("\n--- Exporting variants ---")
        variant_setup = {
            'A_baseline': (lambda: None, lambda: None),
            'B_p3_direct_layout': (setup_p3_direct_layout, teardown_p3_direct_layout),
            'C_p3_fused_proj': (lambda: setup_p3_fused_proj(model), teardown_p3_fused_proj),
            'D_p3_combined': (
                lambda: (setup_p3_direct_layout(), setup_p3_fused_proj(model)),
                lambda: (teardown_p3_direct_layout(), teardown_p3_fused_proj()),
            ),
            'E_p2_perhead_attn': (setup_p2_perhead, teardown_p2_perhead),
            'F_p2p3_all': (
                lambda: (setup_p3_direct_layout(), setup_p3_fused_proj(model), setup_p2_perhead()),
                lambda: (teardown_p3_direct_layout(), teardown_p3_fused_proj(), teardown_p2_perhead()),
            ),
        }

        for name in selected:
            if name in variant_setup:
                setup_fn, teardown_fn = variant_setup[name]
                export_variant(model, name, setup_fn, teardown_fn, skip_existing=args.skip_existing)
            else:
                print(f"  Unknown variant: {name}")

        del model; gc.collect()
    else:
        print("Skipping export (--skip-export)")

    # ---- MIL Analysis ----
    print("\n" + "=" * 70)
    print("  MIL OP ANALYSIS")
    print("=" * 70)

    mil_results = {}
    for name in selected:
        decode_path = os.path.join(OUT_DIR, f'{name}_decode.mlpackage')
        prefill_path = os.path.join(OUT_DIR, f'{name}_prefill.mlpackage')
        if os.path.exists(decode_path):
            mil_results[name] = {
                'decode': analyze_mil_ops(decode_path),
                'prefill': analyze_mil_ops(prefill_path) if os.path.exists(prefill_path) else {},
            }

    # Print decode MIL table
    print(f"\n{'Variant':25s} {'compute':>7s} {'trans':>5s} {'rshp':>5s} {'sqz':>5s} {'edim':>5s} "
          f"{'conv':>5s} {'matm':>5s} {'mul':>5s} {'split':>5s} {'cast':>5s} {'layout':>7s}")
    print("-" * 110)
    for name in selected:
        if name not in mil_results:
            continue
        # Use first function (should be 'main' or unnamed for decode)
        fns = mil_results[name]['decode']
        fn_name = list(fns.keys())[0] if fns else None
        if fn_name is None: continue
        d = fns[fn_name]
        layout = d['transpose'] + d['reshape'] + d['squeeze'] + d['expand_dims']
        print(f"{name:25s} {d['compute']:7d} {d['transpose']:5d} {d['reshape']:5d} {d['squeeze']:5d} "
              f"{d['expand_dims']:5d} {d['conv']:5d} {d['matmul']:5d} {d['mul']:5d} {d['split']:5d} "
              f"{d['cast']:5d} {layout:7d}")

    # Print prefill MIL table
    print(f"\n{'[Prefill]':25s} {'compute':>7s} {'trans':>5s} {'rshp':>5s} {'sqz':>5s} {'edim':>5s} "
          f"{'conv':>5s} {'matm':>5s}")
    print("-" * 80)
    for name in selected:
        if name not in mil_results or not mil_results[name].get('prefill'):
            continue
        fns = mil_results[name]['prefill']
        fn_name = list(fns.keys())[0] if fns else None
        if fn_name is None: continue
        d = fns[fn_name]
        layout = d['transpose'] + d['reshape'] + d['squeeze'] + d['expand_dims']
        print(f"{name:25s} {d['compute']:7d} {d['transpose']:5d} {d['reshape']:5d} {d['squeeze']:5d} "
              f"{d['expand_dims']:5d} {d['conv']:5d} {d['matmul']:5d}")

    # ---- Timing ----
    print("\n" + "=" * 70)
    print("  DECODE TIMING (30 runs, median)")
    print("=" * 70)

    timing_results = {}
    baseline_output = None

    for name in selected:
        decode_path = os.path.join(OUT_DIR, f'{name}_decode.mlpackage')
        if not os.path.exists(decode_path):
            print(f"  {name}: MISSING")
            continue

        print(f"  {name}...", end='', flush=True)
        result = measure_decode(decode_path)

        # Accuracy vs baseline
        if name == 'A_baseline' or name == selected[0]:
            baseline_output = result.get('output', {})
            result['cos_vs_baseline'] = 1.0
        elif baseline_output:
            for key in result.get('output', {}):
                if key in baseline_output:
                    a = baseline_output[key]
                    b = result['output'][key]
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
                    result['cos_vs_baseline'] = cos
                    break

        # Strip numpy arrays for JSON
        result_clean = {k: v for k, v in result.items() if k != 'output'}
        timing_results[name] = {'decode': result_clean}

        cos = result.get('cos_vs_baseline', 'N/A')
        cos_str = f"{cos:.6f}" if isinstance(cos, float) else cos
        print(f" wall={result['wall_ms']:.2f}ms cpu={result['cpu_ms']:.2f}ms "
              f"ane={result['ane_ms']:.2f}ms ({result['ane_pct']:.0f}%) cos={cos_str}")

    # Prefill timing
    print("\n" + "=" * 70)
    print(f"  PREFILL TIMING (seq_len={BATCH_SIZE}, 10 runs, median)")
    print("=" * 70)

    for name in selected:
        prefill_path = os.path.join(OUT_DIR, f'{name}_prefill.mlpackage')
        if not os.path.exists(prefill_path):
            print(f"  {name}: MISSING")
            continue

        print(f"  {name}...", end='', flush=True)
        result = measure_prefill(prefill_path)
        if name in timing_results:
            timing_results[name]['prefill'] = result
        else:
            timing_results[name] = {'prefill': result}

        print(f" wall={result['wall_ms']:.1f}ms cpu={result['cpu_ms']:.1f}ms "
              f"ane={result['ane_ms']:.1f}ms ({result['ane_pct']:.0f}%)")

    # ---- Summary Table ----
    print("\n" + "=" * 70)
    print("  SUMMARY: P2/P3 IMPACT ON CHUNK2 ANE UTILIZATION")
    print("=" * 70)
    print(f"\n{'Variant':25s} │ {'Dec Wall':>8s} {'Dec CPU':>8s} {'Dec ANE%':>8s} │ "
          f"{'PF Wall':>8s} {'PF CPU':>8s} {'PF ANE%':>8s} │ {'Cos':>8s}")
    print("─" * 100)
    for name in selected:
        if name not in timing_results:
            continue
        dec = timing_results[name].get('decode', {})
        pf = timing_results[name].get('prefill', {})
        cos = dec.get('cos_vs_baseline', 'N/A')
        cos_str = f"{cos:.6f}" if isinstance(cos, float) else cos
        print(f"{name:25s} │ {dec.get('wall_ms', 0):7.2f}ms {dec.get('cpu_ms', 0):7.2f}ms "
              f"{dec.get('ane_pct', 0):7.1f}% │ "
              f"{pf.get('wall_ms', 0):7.1f}ms {pf.get('cpu_ms', 0):7.1f}ms "
              f"{pf.get('ane_pct', 0):7.1f}% │ {cos_str}")

    # Save results
    report_path = os.path.join(OUT_DIR, 'report.json')
    with open(report_path, 'w') as f:
        json.dump({'mil': {k: {fk: fv for fk, fv in v.items()}
                           for k, v in mil_results.items()},
                   'timing': timing_results}, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
