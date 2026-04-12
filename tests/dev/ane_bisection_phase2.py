#!/usr/bin/env python3
"""
ANE Bisection Phase 2: Multi-layer stacking and actual model classes.

Phase 1 found: ALL single-layer variants achieve 33-93% ANE.
The actual chunk export (3-4 layers) gets 0%.
Hypothesis: multi-layer stacking kills ANE.

This script tests:
  V9_ref:     Single layer, full features + KV state (from Phase 1 = 93% ANE)
  V11_2L_sep: 2 F-layers with SEPARATE per-layer KV state buffers
  V12_4L_sep: 4 F-layers with separate KV state buffers (matches actual chunk size)
  V13_real_1L: Single F-layer using ACTUAL Qwen3.5 model classes
  V14_real_1L_lut: V13 + LUT4 quantization (matches actual pipeline)
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
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
os.chdir(REPO_ROOT)

# Qwen3.5-4B dimensions
HIDDEN_SIZE = 2560
NUM_Q_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
Q_HEAD_DIM = HEAD_DIM * 2
INTERMEDIATE_SIZE = 9216
ROTARY_DIM = 128
EPS = 1e-6
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
SCALE = 1.0 / math.sqrt(HEAD_DIM)

ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "ane_bisection")
NUM_WARMUP = 10
NUM_RUNS = 30

# Import config for actual model tests
from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES


# ═══════════════════════════════════════════════════════════════════════
#  Building blocks (from Phase 1)
# ═══════════════════════════════════════════════════════════════════════

class RMSNormDoubled(nn.Module):
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
    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps
        self.head_dim = head_dim

    def forward(self, x):
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled, (2 * self.head_dim,), None, None, self.eps)
        normed = normed[..., :self.head_dim]
        return normed * self.weight


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1).flatten(1, 2)


def rotate_half(x, half_dim):
    return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)


def apply_rope(q, k, position_ids, inv_freq, rotary_dim):
    pos = position_ids.float().unsqueeze(-1)
    freqs = pos * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    half = rotary_dim // 2
    q_rot = q[..., :rotary_dim] * cos + rotate_half(q[..., :rotary_dim], half) * sin
    k_rot = k[..., :rotary_dim] * cos + rotate_half(k[..., :rotary_dim], half) * sin
    return torch.cat([q_rot, q[..., rotary_dim:]], dim=-1), \
           torch.cat([k_rot, k[..., rotary_dim:]], dim=-1)


def make_single_layer():
    """Create a single F-layer module dict."""
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


def forward_single_layer(layer, hidden_states, k_cache, v_cache, position_ids,
                          current_pos, causal_mask, inv_freq):
    """Forward pass for one F-layer with KV write + gate + residual + FFN."""
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

    # KV cache write at dynamic position
    pos = current_pos[0]
    k_cache[:, :, pos:pos+1, :] = k_new.squeeze(0)
    v_cache[:, :, pos:pos+1, :] = v_new.squeeze(0)

    # Attention with GQA
    k_exp = repeat_kv(k_cache, NUM_Q_HEADS // NUM_KV_HEADS)
    v_exp = repeat_kv(v_cache, NUM_Q_HEADS // NUM_KV_HEADS)
    attn = torch.matmul(q, k_exp.transpose(-1, -2)) * SCALE
    attn = attn + causal_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_exp)
    out = out.transpose(1, 2).flatten(2, 3)

    # Gated output
    out = out * torch.sigmoid(gate)
    out = out.permute(0, 2, 1).unsqueeze(2)
    out = layer["o_proj"](out)
    attn_out = out.squeeze(2).permute(0, 2, 1)

    # Residual
    hidden_states = hidden_states + attn_out

    # FFN
    post = layer["post_norm"](hidden_states)
    h2 = post.permute(0, 2, 1).unsqueeze(2)
    ffn = F.silu(layer["gate_proj"](h2)) * layer["up_proj"](h2)
    ffn = layer["down_proj"](ffn)
    ffn = ffn.squeeze(2).permute(0, 2, 1)
    return hidden_states + ffn


# ═══════════════════════════════════════════════════════════════════════
#  Multi-layer variants: SEPARATE per-layer KV state buffers
# ═══════════════════════════════════════════════════════════════════════

class MultiLayerF(nn.Module):
    """N F-layers with separate per-layer KV state buffers."""
    def __init__(self, num_layers, ctx):
        super().__init__()
        self.num_layers = num_layers
        self.ctx = ctx
        self.layers = nn.ModuleList([make_single_layer() for _ in range(num_layers)])

        # Separate per-layer KV caches as register_buffers
        for i in range(num_layers):
            self.register_buffer(f"k_cache_{i}", torch.zeros(
                1, NUM_KV_HEADS, ctx, HEAD_DIM, dtype=torch.float32))
            self.register_buffer(f"v_cache_{i}", torch.zeros(
                1, NUM_KV_HEADS, ctx, HEAD_DIM, dtype=torch.float32))

        # Pre-compute inv_freq
        self.register_buffer("inv_freq", 1.0 / (500000.0 ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)))

    def forward(self, hidden_states, position_ids, current_pos, causal_mask):
        for i in range(self.num_layers):
            k_cache = getattr(self, f"k_cache_{i}")
            v_cache = getattr(self, f"v_cache_{i}")
            hidden_states = forward_single_layer(
                self.layers[i], hidden_states, k_cache, v_cache,
                position_ids, current_pos, causal_mask, self.inv_freq,
            )
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════
#  V13: Single F-layer using ACTUAL Qwen3.5 model classes
# ═══════════════════════════════════════════════════════════════════════

class RealSingleFLayer(nn.Module):
    """Wrapper around actual Qwen3.5 model for a single F-layer export."""
    def __init__(self, model, layer_idx, ctx):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.ctx = ctx
        cfg = model.config
        self.register_buffer("k_cache", torch.zeros(
            1, cfg.num_key_value_heads, ctx, cfg.head_dim,
            dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(
            1, cfg.num_key_value_heads, ctx, cfg.head_dim,
            dtype=torch.float16))

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        layer = self.model.model.layers[self.layer_idx]
        x = layer.input_layernorm(hidden_states)
        query_states, key_states, value_states, gate = layer.self_attn.get_new_kv_cache(x, position_ids)

        pos = current_pos[0]
        self.k_cache[:, :, pos:pos+1, :] = key_states.squeeze(0)
        self.v_cache[:, :, pos:pos+1, :] = value_states.squeeze(0)

        key_cache = self.k_cache.squeeze(0)
        value_cache = self.v_cache.squeeze(0)

        attn_out = layer.self_attn.forward_regular(
            hidden_states=x,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
            gate=gate,
        )
        hidden_states = hidden_states + attn_out
        post = layer.post_attention_layernorm(hidden_states)
        return hidden_states + layer.mlp(post)


# ═══════════════════════════════════════════════════════════════════════
#  Export and measurement (reused from Phase 1)
# ═══════════════════════════════════════════════════════════════════════

def analyze_mil(path, name):
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
    print(f"  {name:45s} ops={total:4d} conv={conv:2d} matmul={matmul:2d} soft={softmax:2d} hostile={hostile_str}")
    del mlmodel
    return total, hostile


def measure_ane(path, name, has_state, ctx, num_layers=1, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    mlmodel = ct.models.MLModel(path, compute_units=compute_unit)
    pred = {
        "hidden_states": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float16),
        "position_ids": np.array([ctx // 2], dtype=np.int32),
        "current_pos": np.array([ctx // 2], dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, ctx), dtype=np.float16),
    }

    state = mlmodel.make_state() if has_state else None
    for _ in range(NUM_WARMUP):
        if state is not None:
            mlmodel.predict(pred, state=state)
        else:
            mlmodel.predict(pred)

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


def export_multilayer(num_layers, ctx, skip_existing=False):
    name = f"V{9+num_layers}_sep_{num_layers}L_ctx{ctx}"
    out_path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
    if skip_existing and os.path.exists(out_path):
        print(f"  [skip] {name} exists")
        return out_path, name

    model = MultiLayerF(num_layers, ctx)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.float32)
    pos_ids = torch.tensor([ctx // 2], dtype=torch.long)
    cur_pos = torch.tensor([ctx // 2], dtype=torch.int32)
    mask = torch.zeros(1, 1, 1, ctx, dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(model, (hidden, pos_ids, cur_pos, mask))

    # States: separate per-layer
    states = []
    for i in range(num_layers):
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, NUM_KV_HEADS, ctx, HEAD_DIM), dtype=np.float16),
            name=f"k_cache_{i}"))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, NUM_KV_HEADS, ctx, HEAD_DIM), dtype=np.float16),
            name=f"v_cache_{i}"))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=pos_ids.shape, dtype=np.int32),
            ct.TensorType(name="current_pos", shape=cur_pos.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
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
    return out_path, name


def export_real_single_layer(layer_idx, ctx, model_qwen, skip_existing=False, lut_bits=0):
    """Export a single F-layer using actual Qwen3.5 model classes."""
    suffix = f"_lut{lut_bits}" if lut_bits else ""
    name = f"V13_real_1L_layer{layer_idx}{suffix}_ctx{ctx}"
    out_path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
    if skip_existing and os.path.exists(out_path):
        print(f"  [skip] {name} exists")
        return out_path, name

    wrapper = RealSingleFLayer(model_qwen, layer_idx, ctx)
    wrapper.eval()

    hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.float32)
    pos_ids = torch.tensor([ctx // 2], dtype=torch.long)
    mask = torch.zeros(1, 1, 1, ctx, dtype=torch.float32)
    cur_pos = torch.tensor([ctx // 2], dtype=torch.int32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (hidden, pos_ids, mask, cur_pos))

    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, NUM_KV_HEADS, ctx, HEAD_DIM), dtype=np.float16),
            name="k_cache"),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, NUM_KV_HEADS, ctx, HEAD_DIM), dtype=np.float16),
            name="v_cache"),
    ]

    precision = ct.precision.FLOAT16
    if lut_bits:
        from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision
        precision = FP16ComputePrecision(op_selector=lambda op: True)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=pos_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=cur_pos.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=states,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    if lut_bits:
        from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
        conv = Qwen35Converter.__new__(Qwen35Converter)
        conv.lut_bits = lut_bits
        conv.per_channel = 4
        conv.converted_model = mlmodel
        conv.postprocess(num_workers=1)
        mlmodel = conv.converted_model

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mlmodel.save(out_path)
    del mlmodel, traced, wrapper
    gc.collect()
    return out_path, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--max-layers", type=int, default=4, help="Max layers to stack")
    parser.add_argument("--cpu-ref", action="store_true")
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("=" * 90)
    print("  ANE BISECTION PHASE 2: Multi-layer & actual model classes")
    print(f"  CTX: {args.ctx}, max_layers: {args.max_layers}")
    print("=" * 90)

    # Load Qwen3.5 model for V13/V14
    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE
    HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
    print("\nLoading Qwen3.5-4B weights...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    model_qwen = None
    if not args.skip_export:
        cfg.context_length = max(args.ctx)
        cfg.state_length = max(args.ctx)
        model_qwen = Qwen35ForCausalLM(cfg)
        assert model_qwen.load_pretrained_weights(HF_MODEL)
        model_qwen.eval()
        for p in model_qwen.parameters():
            p.requires_grad = False
        print(f"  Loaded in {time.time()-t0:.1f}s")

    # Identify F-layer indices (not linear_attention)
    if model_qwen:
        f_layer_indices = [i for i, layer in enumerate(model_qwen.model.layers)
                          if layer.layer_type == "full_attention"]
        print(f"  F-layer indices: {f_layer_indices}")
    else:
        f_layer_indices = [3]  # typical first F layer

    # Phase 1: Export
    results_map = {}  # (name, ctx) → path

    if not args.skip_export:
        print(f"\n{'='*90}")
        print("  EXPORT: Multi-layer stacking (1, 2, 3, 4 layers)")
        print(f"{'='*90}")
        for num_layers in range(1, args.max_layers + 1):
            for ctx in args.ctx:
                t0 = time.time()
                try:
                    path, name = export_multilayer(num_layers, ctx, skip_existing=args.skip_existing)
                    results_map[(name, ctx)] = path
                    print(f"  {name:50s} {time.time()-t0:6.1f}s  ✓")
                except Exception as e:
                    print(f"  {num_layers}L ctx={ctx}:  FAILED: {e}")
                gc.collect()

        print(f"\n{'='*90}")
        print("  EXPORT: Actual Qwen3.5 model (single F-layer)")
        print(f"{'='*90}")
        for layer_idx in f_layer_indices[:2]:  # first 2 F-layers
            for ctx in args.ctx:
                for lut in [0, 4]:  # without and with LUT4
                    t0 = time.time()
                    try:
                        path, name = export_real_single_layer(
                            layer_idx, ctx, model_qwen,
                            skip_existing=args.skip_existing, lut_bits=lut)
                        results_map[(name, ctx)] = path
                        print(f"  {name:50s} {time.time()-t0:6.1f}s  ✓")
                    except Exception as e:
                        print(f"  layer{layer_idx} lut{lut} ctx={ctx}: FAILED: {e}")
                    gc.collect()

        del model_qwen
        gc.collect()

    # Phase 2: MIL analysis
    print(f"\n{'='*90}")
    print("  MIL OP ANALYSIS")
    print(f"{'='*90}")
    all_names = []

    # Multi-layer variants
    for num_layers in range(1, args.max_layers + 1):
        for ctx in args.ctx:
            name = f"V{9+num_layers}_sep_{num_layers}L_ctx{ctx}"
            path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
            if os.path.exists(path):
                try:
                    analyze_mil(path, name)
                    all_names.append((name, ctx, True, num_layers))
                except Exception as e:
                    print(f"  {name:45s} FAILED: {e}")

    # Real model variants
    for layer_idx in (f_layer_indices[:2] if f_layer_indices else [3]):
        for ctx in args.ctx:
            for lut in [0, 4]:
                suffix = f"_lut{lut}" if lut else ""
                name = f"V13_real_1L_layer{layer_idx}{suffix}_ctx{ctx}"
                path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
                if os.path.exists(path):
                    try:
                        analyze_mil(path, name)
                        all_names.append((name, ctx, True, 1))
                    except Exception as e:
                        print(f"  {name:45s} FAILED: {e}")

    # Phase 3: ANE measurement
    print(f"\n{'='*90}")
    print(f"  ANE UTILIZATION ({NUM_RUNS} runs, {NUM_WARMUP} warmup)")
    print(f"{'='*90}")
    results = {}

    for ctx in args.ctx:
        print(f"\n  --- CTX={ctx} ---")

        # Multi-layer variants
        for num_layers in range(1, args.max_layers + 1):
            name = f"V{9+num_layers}_sep_{num_layers}L_ctx{ctx}"
            path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
            if os.path.exists(path):
                try:
                    wall, cpu, ane = measure_ane(path, name, True, ctx, num_layers)
                    results[(name, ctx)] = (wall, cpu, ane, num_layers)
                except Exception as e:
                    print(f"  {name:50s} FAILED: {e}")
                gc.collect()

        # Real model variants
        for layer_idx in (f_layer_indices[:2] if f_layer_indices else [3]):
            for lut in [0, 4]:
                suffix = f"_lut{lut}" if lut else ""
                name = f"V13_real_1L_layer{layer_idx}{suffix}_ctx{ctx}"
                path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
                if os.path.exists(path):
                    try:
                        wall, cpu, ane = measure_ane(path, name, True, ctx, 1)
                        results[(name, ctx)] = (wall, cpu, ane, 1)
                    except Exception as e:
                        print(f"  {name:50s} FAILED: {e}")
                    gc.collect()

        if args.cpu_ref:
            print(f"\n  --- CTX={ctx} CPU_ONLY reference ---")
            for num_layers in range(1, args.max_layers + 1):
                name = f"V{9+num_layers}_sep_{num_layers}L_ctx{ctx}"
                path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
                if os.path.exists(path):
                    try:
                        measure_ane(path, name, True, ctx, num_layers, ct.ComputeUnit.CPU_ONLY)
                    except Exception as e:
                        pass
                    gc.collect()

    # Summary
    print(f"\n{'='*90}")
    print("  SUMMARY")
    print(f"{'='*90}")

    print(f"\n  Multi-layer stacking (all features + KV state + mask):")
    header = f"  {'Config':<35s}"
    for ctx in args.ctx:
        header += f"  {'CTX='+str(ctx):>16s}"
    print(header)
    print("  " + "-" * (35 + 18 * len(args.ctx)))

    for num_layers in range(1, args.max_layers + 1):
        row = f"  {num_layers} F-layer(s) (separate KV)     "
        for ctx in args.ctx:
            name = f"V{9+num_layers}_sep_{num_layers}L_ctx{ctx}"
            key = (name, ctx)
            if key in results:
                wall, cpu, ane, _ = results[key]
                row += f"  {ane:7.1f}% {wall:6.1f}ms"
            else:
                row += f"  {'---':>16s}"
        print(row)

    print(f"\n  Actual Qwen3.5 model (single F-layer):")
    for layer_idx in (f_layer_indices[:2] if f_layer_indices else [3]):
        for lut in [0, 4]:
            suffix = f"_lut{lut}" if lut else ""
            label = f"  Layer {layer_idx}" + (f" + LUT{lut}" if lut else " (no LUT)")
            row = f"  {label:<35s}"
            for ctx in args.ctx:
                name = f"V13_real_1L_layer{layer_idx}{suffix}_ctx{ctx}"
                key = (name, ctx)
                if key in results:
                    wall, cpu, ane, _ = results[key]
                    row += f"  {ane:7.1f}% {wall:6.1f}ms"
                else:
                    row += f"  {'---':>16s}"
            print(row)

    # Phase 1 reference
    print(f"\n  Phase 1 reference (single layer, synthesized):")
    print(f"  V9_kv_state at CTX=512: 92.5% ANE, 2.4ms")
    print(f"  V9_kv_state at CTX=2048: 93.5% ANE, 3.4ms")

    print(f"\n{'='*90}")
    print("  Done!")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
