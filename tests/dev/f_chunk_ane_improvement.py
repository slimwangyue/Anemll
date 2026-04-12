#!/usr/bin/env python3
"""
F (Full Attention) Chunk ANE Improvement Investigation

Tests variants of a single F-layer (full attention) chunk to eliminate hostile ops
and improve ANE utilization from 0%.

Hostile ops in standard F chunk: 4 read_state + 2 gather + 2 greater_equal + 2 select = 10
Goal: eliminate enough to allow ANE scheduling.

Variants:
  A. Standard F chunk (baseline — 10 hostile ops)
  B. On-the-fly RoPE, keep KV states (removes 6 RoPE hostile ops → 4 read_state)
  C. Stateless KV (I/O tensors), keep RoPE gather (removes 4 read_state → 6 RoPE hostile)
  D. On-the-fly RoPE + Stateless KV (removes all 10 → 0 hostile ops)
"""
import sys, os, time, gc, warnings
import numpy as np

REPO_ROOT = '/Volumes/MySSD/Anemll'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts_qwen3_5'))
os.chdir(REPO_ROOT)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.setdefault('TMPDIR', '/Volumes/MySSD/tmp')

import torch
torch.set_grad_enabled(False)
import coremltools as ct
from collections import Counter

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

HF_MODEL = os.path.join(REPO_ROOT, 'models', 'Qwen__Qwen3.5-4B')
ARTIFACT_DIR = os.path.join(REPO_ROOT, 'artifacts', 'f_chunk_ane_improvement')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
    apply_rotary_pos_emb_single,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


# ═══════════════════════════════════════════════════════════════════════
# Variant A: Standard F chunk (from f_isolated_experiment)
# Already exported — just reference it.
# ═══════════════════════════════════════════════════════════════════════

VARIANT_A_PATH = os.path.join(REPO_ROOT, 'artifacts', 'f_isolated_experiment', 'chunk1.mlpackage')


# ═══════════════════════════════════════════════════════════════════════
# Variant B: On-the-fly RoPE, keep KV cache as CoreML states
# ═══════════════════════════════════════════════════════════════════════

class VariantB_OnTheFlyRoPE_StateKV(torch.nn.Module):
    """Single F (full-attention) layer with on-the-fly RoPE. KV cache stays as states."""

    def __init__(self, model, layer_idx):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.local_num_layers = 1
        cfg = model.config

        # KV cache as buffers → CoreML states
        self.register_buffer("k_cache", torch.zeros(
            (1, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=MODEL_DTYPE, device=TEST_DEVICE))
        self.register_buffer("v_cache", torch.zeros(
            (1, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=MODEL_DTYPE, device=TEST_DEVICE))

        self.states = Qwen35Converter.GetChunkLocalTransformerStates(
            model, 1, prefix="", split_full_attention_kv=True)

        # On-the-fly RoPE
        layer = model.model.layers[layer_idx]
        self.register_buffer("rope_inv_freq", layer.self_attn.rotary.inv_freq.clone())
        self._rotary_dim = layer.self_attn.rotary.rotary_dim

    def _rope_onthefly(self, position_ids, dtype, device):
        pos_ids = position_ids if position_ids.dim() == 1 else position_ids.squeeze(0)
        t = pos_ids.to(dtype=torch.float32).unsqueeze(-1)
        freqs = t * self.rope_inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).to(dtype)
        return cos, sin

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        cos, sin = self._rope_onthefly(position_ids, hidden_states.dtype, hidden_states.device)
        layer = self.model.model.layers[self.layer_idx]
        x = layer.input_layernorm(hidden_states)
        query_states, key_states, value_states, gate = layer.self_attn._project_qkvg(x)
        query_states = layer.self_attn.q_norm(query_states)
        key_states = layer.self_attn.k_norm(key_states)
        query_states, key_states = apply_rotary_pos_emb_single(
            query_states, key_states, cos, sin, self._rotary_dim)
        query_states = query_states.to(MODEL_DTYPE)
        key_states = key_states.to(MODEL_DTYPE)
        value_states = value_states.to(MODEL_DTYPE)
        gate_states = gate.to(MODEL_DTYPE) if gate is not None else None

        pos = current_pos[0]
        self.k_cache[0, :, pos:pos+1, :] = key_states.squeeze(0)
        self.v_cache[0, :, pos:pos+1, :] = value_states.squeeze(0)
        key_cache = self.k_cache[0:1].squeeze(0)
        value_cache = self.v_cache[0:1].squeeze(0)

        attn_out = layer.self_attn.forward_regular(
            hidden_states=x, query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask, gate=gate_states)
        hidden_states = hidden_states + attn_out
        post = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post)
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════
# Variant C: Standard RoPE (gather), Stateless KV (I/O tensors)
# ═══════════════════════════════════════════════════════════════════════

class VariantC_GatherRoPE_StatelessKV(torch.nn.Module):
    """Single F layer with standard RoPE (gather) but stateless KV cache (I/O tensors)."""

    def __init__(self, model, layer_idx):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        cfg = model.config
        self._kv_shape = (1, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim)

    def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                k_cache_in, v_cache_in):
        layer = self.model.model.layers[self.layer_idx]
        x = layer.input_layernorm(hidden_states)
        query_states, key_states, value_states, gate = layer.self_attn.get_new_kv_cache(x, position_ids)

        pos = current_pos[0]
        k_cache_in[0, :, pos:pos+1, :] = key_states.squeeze(0)
        v_cache_in[0, :, pos:pos+1, :] = value_states.squeeze(0)
        key_cache = k_cache_in[0:1].squeeze(0)
        value_cache = v_cache_in[0:1].squeeze(0)

        attn_out = layer.self_attn.forward_regular(
            hidden_states=x, query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask, gate=gate)
        hidden_states = hidden_states + attn_out
        post = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post)
        return hidden_states, k_cache_in, v_cache_in


# ═══════════════════════════════════════════════════════════════════════
# Variant D: On-the-fly RoPE + Stateless KV (0 hostile ops)
# ═══════════════════════════════════════════════════════════════════════

class VariantD_OnTheFlyRoPE_StatelessKV(torch.nn.Module):
    """Single F layer: on-the-fly RoPE + stateless KV cache. Should have 0 hostile ops."""

    def __init__(self, model, layer_idx):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        cfg = model.config
        self._kv_shape = (1, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim)

        layer = model.model.layers[layer_idx]
        self.register_buffer("rope_inv_freq", layer.self_attn.rotary.inv_freq.clone())
        self._rotary_dim = layer.self_attn.rotary.rotary_dim

    def _rope_onthefly(self, position_ids, dtype, device):
        pos_ids = position_ids if position_ids.dim() == 1 else position_ids.squeeze(0)
        t = pos_ids.to(dtype=torch.float32).unsqueeze(-1)
        freqs = t * self.rope_inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).to(dtype)
        return cos, sin

    def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                k_cache_in, v_cache_in):
        cos, sin = self._rope_onthefly(position_ids, hidden_states.dtype, hidden_states.device)
        layer = self.model.model.layers[self.layer_idx]
        x = layer.input_layernorm(hidden_states)
        query_states, key_states, value_states, gate = layer.self_attn._project_qkvg(x)
        query_states = layer.self_attn.q_norm(query_states)
        key_states = layer.self_attn.k_norm(key_states)
        query_states, key_states = apply_rotary_pos_emb_single(
            query_states, key_states, cos, sin, self._rotary_dim)
        query_states = query_states.to(MODEL_DTYPE)
        key_states = key_states.to(MODEL_DTYPE)
        value_states = value_states.to(MODEL_DTYPE)
        gate_states = gate.to(MODEL_DTYPE) if gate is not None else None

        pos = current_pos[0]
        k_cache_in[0, :, pos:pos+1, :] = key_states.squeeze(0)
        v_cache_in[0, :, pos:pos+1, :] = value_states.squeeze(0)
        key_cache = k_cache_in[0:1].squeeze(0)
        value_cache = v_cache_in[0:1].squeeze(0)

        attn_out = layer.self_attn.forward_regular(
            hidden_states=x, query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask, gate=gate_states)
        hidden_states = hidden_states + attn_out
        post = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post)
        return hidden_states, k_cache_in, v_cache_in


# ═══════════════════════════════════════════════════════════════════════
# Export helpers
# ═══════════════════════════════════════════════════════════════════════

def export_variant_b(model, cfg, layer_idx):
    """Export Variant B: on-the-fly RoPE + KV states."""
    wrapper = VariantB_OnTheFlyRoPE_StateKV(model, layer_idx).eval()
    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos),
                              check_trace=False)
    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    traced.k_cache.zero_(); traced.v_cache.zero_()

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    path = os.path.join(ARTIFACT_DIR, f'variant_b_layer{layer_idx}.mlpackage')
    mlmodel.save(path)
    del mlmodel; gc.collect()
    return path


def export_variant_c(model, cfg, layer_idx):
    """Export Variant C: standard RoPE gather + stateless KV."""
    wrapper = VariantC_GatherRoPE_StatelessKV(model, layer_idx).eval()
    cfg_obj = model.config
    hidden_states = torch.zeros((1, 1, cfg_obj.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    kv_shape = wrapper._kv_shape
    k_cache_in = torch.zeros(kv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    v_cache_in = torch.zeros(kv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)

    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos,
                                        k_cache_in, v_cache_in), check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="k_cache_in", shape=k_cache_in.shape, dtype=np.float16),
            ct.TensorType(name="v_cache_in", shape=v_cache_in.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="k_cache_out", dtype=np.float16),
            ct.TensorType(name="v_cache_out", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    path = os.path.join(ARTIFACT_DIR, f'variant_c_layer{layer_idx}.mlpackage')
    mlmodel.save(path)
    del mlmodel; gc.collect()
    return path


def export_variant_d(model, cfg, layer_idx):
    """Export Variant D: on-the-fly RoPE + stateless KV. Target: 0 hostile ops."""
    wrapper = VariantD_OnTheFlyRoPE_StatelessKV(model, layer_idx).eval()
    cfg_obj = model.config
    hidden_states = torch.zeros((1, 1, cfg_obj.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    kv_shape = wrapper._kv_shape
    k_cache_in = torch.zeros(kv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    v_cache_in = torch.zeros(kv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)

    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos,
                                        k_cache_in, v_cache_in), check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="k_cache_in", shape=k_cache_in.shape, dtype=np.float16),
            ct.TensorType(name="v_cache_in", shape=v_cache_in.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="k_cache_out", dtype=np.float16),
            ct.TensorType(name="v_cache_out", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    path = os.path.join(ARTIFACT_DIR, f'variant_d_layer{layer_idx}.mlpackage')
    mlmodel.save(path)
    del mlmodel; gc.collect()
    return path


# ═══════════════════════════════════════════════════════════════════════
# Analysis & Measurement
# ═══════════════════════════════════════════════════════════════════════

def analyze_mil(path, label):
    """Count hostile ops."""
    spec = ct.utils.load_spec(path)
    prog = spec.mlProgram
    for fname in prog.functions.keys():
        func = prog.functions[fname]
        oc = Counter()
        hostile = []
        for _, blk in func.block_specializations.items():
            for op in blk.operations:
                oc[op.type] += 1
                if op.type in ('gather', 'gather_along_axis', 'greater_equal', 'select',
                               'read_state', 'coreml_update_state'):
                    hostile.append(op.type)
        h = dict(Counter(hostile)) if hostile else "NONE"
        print(f"  {label:50s} fn={fname:6s} total={sum(oc.values()):5d}  hostile={h}")
        sys.stdout.flush()


def measure_ane(path, label, func_name=None, cu=ct.ComputeUnit.CPU_AND_NE):
    """Measure ANE utilization via process_time vs wall_time."""
    WARMUP, RUNS = 5, 15
    ml = ct.models.MLModel(path, compute_units=cu, function_name=func_name)
    # Build inputs from MIL spec
    spec = ct.utils.load_spec(path)
    prog = spec.mlProgram
    fn = func_name or list(prog.functions.keys())[0]
    func = prog.functions[fn]
    inputs_dict = {}
    for inp in func.inputs:
        if inp.type.WhichOneof('type') == 'tensorType':
            tt = inp.type.tensorType
            shape = tuple(d.constant.size for d in tt.dimensions)
            dt_map = {1: np.float32, 3: np.float16, 5: np.int32}
            inputs_dict[inp.name] = np.zeros(shape, dtype=dt_map.get(tt.dataType, np.float16))
    state = ml.make_state()
    for _ in range(WARMUP):
        ml.predict(inputs_dict, state=state)
    walls, cpus = [], []
    for _ in range(RUNS):
        tw = time.perf_counter(); tc = time.process_time()
        ml.predict(inputs_dict, state=state)
        cpus.append(time.process_time() - tc); walls.append(time.perf_counter() - tw)
    mw = sorted(walls)[len(walls)//2] * 1000
    mc = sorted(cpus)[len(cpus)//2] * 1000
    cf = mc / mw if mw > 0 else 1.0
    af = max(0, 1 - cf)
    cu_s = 'ANE' if cu == ct.ComputeUnit.CPU_AND_NE else 'CPU'
    print(f"  {label:50s} [{cu_s}] wall={mw:7.2f}ms  cpu={mc:7.2f}ms  CPU%={cf*100:5.1f}%  ANE%={af*100:5.1f}%")
    sys.stdout.flush()
    del ml, state; gc.collect()
    return mw, af


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--layer", type=int, default=3, help="F-layer index to test (default: 3)")
    args = parser.parse_args()

    LAYER_IDX = args.layer

    print("=" * 70)
    print(f"  F CHUNK ANE IMPROVEMENT — Layer {LAYER_IDX} (Full Attention)")
    print("=" * 70)
    sys.stdout.flush()

    paths = {
        "A_standard": VARIANT_A_PATH,
        "B_onthefly_rope_state_kv": os.path.join(ARTIFACT_DIR, f'variant_b_layer{LAYER_IDX}.mlpackage'),
        "C_gather_rope_stateless_kv": os.path.join(ARTIFACT_DIR, f'variant_c_layer{LAYER_IDX}.mlpackage'),
        "D_onthefly_rope_stateless_kv": os.path.join(ARTIFACT_DIR, f'variant_d_layer{LAYER_IDX}.mlpackage'),
    }

    if not args.skip_export:
        print("\n  Loading model weights...")
        sys.stdout.flush()
        cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
        cfg.context_length = CTX
        cfg.state_length = CTX
        model = Qwen35ForCausalLM(cfg)
        assert model.load_pretrained_weights(HF_MODEL)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print(f"\n  Exporting Variant B (on-the-fly RoPE + KV states)...")
        sys.stdout.flush()
        t0 = time.time()
        paths["B_onthefly_rope_state_kv"] = export_variant_b(model, cfg, LAYER_IDX)
        print(f"    Saved in {time.time()-t0:.1f}s")

        print(f"\n  Exporting Variant C (standard RoPE + stateless KV)...")
        sys.stdout.flush()
        t0 = time.time()
        paths["C_gather_rope_stateless_kv"] = export_variant_c(model, cfg, LAYER_IDX)
        print(f"    Saved in {time.time()-t0:.1f}s")

        print(f"\n  Exporting Variant D (on-the-fly RoPE + stateless KV)...")
        sys.stdout.flush()
        t0 = time.time()
        paths["D_onthefly_rope_stateless_kv"] = export_variant_d(model, cfg, LAYER_IDX)
        print(f"    Saved in {time.time()-t0:.1f}s")

        del model; gc.collect()

    # ── Analysis ──
    print("\n" + "=" * 70)
    print("  MIL HOSTILE OP ANALYSIS")
    print("=" * 70)
    for name, path in paths.items():
        if os.path.exists(path):
            analyze_mil(path, name)
        else:
            print(f"  {name:50s} MISSING")

    # ── ANE Measurement ──
    print("\n" + "=" * 70)
    print("  ANE UTILIZATION MEASUREMENT")
    print("=" * 70)
    results = {}
    for name, path in paths.items():
        if os.path.exists(path):
            w, a = measure_ane(path, name)
            results[name] = (w, a)

    # CPU_ONLY reference for first variant
    print()
    for name in ["A_standard", "D_onthefly_rope_stateless_kv"]:
        if name in paths and os.path.exists(paths[name]):
            measure_ane(paths[name], f"{name} (CPU_ONLY ref)", cu=ct.ComputeUnit.CPU_ONLY)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Variant':50s} {'Wall(ms)':>10s} {'ANE%':>8s}")
    print(f"  {'-'*50} {'-'*10} {'-'*8}")
    for name, (w, a) in results.items():
        print(f"  {name:50s} {w:10.2f} {a*100:7.1f}%")

    if "A_standard" in results and "D_onthefly_rope_stateless_kv" in results:
        wa, _ = results["A_standard"]
        wd, ad = results["D_onthefly_rope_stateless_kv"]
        delta = (wd - wa) / wa * 100
        print(f"\n  Variant D vs A: {delta:+.1f}% latency, {ad*100:.1f}% ANE")
        if ad > 0.1 and wd < wa:
            print("  ✅ WINNER: Variant D improves both ANE and latency!")
        elif ad > 0.1:
            print(f"  ⚠️  Variant D enables ANE but costs {delta:+.1f}% latency")
        else:
            print("  ❌ Variant D did not improve ANE utilization")

    print("\n  Done!")
