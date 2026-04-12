#!/usr/bin/env python3
"""
Variant H: Apple ANE-style per-head einsum attention

Rewrites F-layer attention to follow Apple's ml-ane-transformers pattern:
  - Per-head splitting before matmul (dim=1 channel split)
  - einsum('bchq,bkhc->bkhq') for Q·K^T per head
  - Per-head softmax (smaller tensors → better L2 residency)
  - einsum('bkhq,bchk->bchq') for attn·V per head
  - Cat all heads back

Also creates a reference Variant F' with standard matmul (same interface)
to ensure any ANE% difference is ONLY from the attention pattern.

Usage:
  python tests/dev/f_chunk_apple_ane_attn.py --layer 3
  python tests/dev/f_chunk_apple_ane_attn.py --layer 3 --skip-export
  python tests/dev/f_chunk_apple_ane_attn.py --layer 3 --ctx 512
"""
import sys, os, time, gc, math, warnings, argparse
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

from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
    apply_rotary_pos_emb_single, _repeat_kv,
)

HF_MODEL = os.path.join(REPO_ROOT, 'models', 'Qwen__Qwen3.5-4B')
ARTIFACT_DIR = os.path.join(REPO_ROOT, 'artifacts', 'f_chunk_apple_ane_attn')
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Variant F' — Standard matmul attention (reference baseline)
#   Same interface as Variant H for clean comparison.
#   No cache writes. K/V cache passed as input tensors.
# ═══════════════════════════════════════════════════════════════════════

class VariantF_StandardMatmul(torch.nn.Module):
    """Single F layer, standard torch.matmul attention, no cache writes."""

    def __init__(self, model, layer_idx):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        cfg = model.config
        self.num_heads = cfg.num_attention_heads       # 16
        self.num_kv_heads = cfg.num_key_value_heads    # 4
        self.head_dim = cfg.head_dim                   # 256
        self.n_rep = self.num_heads // self.num_kv_heads  # 4
        self.scale = 1.0 / math.sqrt(self.head_dim)

        layer = model.model.layers[layer_idx]
        self.register_buffer("rope_inv_freq", layer.self_attn.rotary.inv_freq.clone())
        self._rotary_dim = layer.self_attn.rotary.rotary_dim

    def _rope_onthefly(self, position_ids, dtype, device):
        pos_ids = position_ids if position_ids.dim() == 1 else position_ids.squeeze(0)
        t = pos_ids.to(dtype=torch.float32).unsqueeze(-1)
        freqs = t * self.rope_inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(0).to(dtype), emb.sin().unsqueeze(0).to(dtype)

    def forward(self, hidden_states, position_ids, causal_mask, k_cache, v_cache):
        """
        hidden_states: (1, 1, hidden_size)
        position_ids: (1,) int32
        causal_mask: (1, 1, 1, CTX) float16
        k_cache: (1, kv_heads, CTX, head_dim) float16
        v_cache: (1, kv_heads, CTX, head_dim) float16
        """
        layer = self.model.model.layers[self.layer_idx]
        x = layer.input_layernorm(hidden_states)

        # Q/K/V projection via Conv2d + norm + RoPE
        query_states, key_new, value_new, gate = layer.self_attn._project_qkvg(x)
        query_states = layer.self_attn.q_norm(query_states)
        cos, sin = self._rope_onthefly(position_ids, hidden_states.dtype, hidden_states.device)
        query_states, _ = apply_rotary_pos_emb_single(
            query_states, key_new, cos, sin, self._rotary_dim)
        query_states = query_states.to(MODEL_DTYPE)

        # GQA expand k/v cache
        key_states = _repeat_kv(k_cache, self.n_rep)        # (1, H, CTX, D)
        value_states = _repeat_kv(v_cache, self.n_rep)      # (1, H, CTX, D)

        # Standard batched matmul attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-1, -2)
        ) * self.scale                                       # (1, H, 1, CTX)
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)  # (1, H, 1, D)

        attn_output = attn_output.transpose(1, 2).contiguous().flatten(2, 3)  # (1, 1, H*D)

        # Gate + o_proj + residual + MLP
        attn_output = layer.self_attn._project_output(attn_output, x, gate=gate)
        hidden_states = hidden_states + attn_output
        post = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post)
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════
# Variant H — Apple ANE per-head einsum attention
#   Follows Apple ml-ane-transformers pattern:
#   - channels-first 4D format (B, C, 1, S)
#   - per-head split via .split(head_dim, dim=1)
#   - einsum('bchq,bkhc->bkhq') per head
#   - per-head softmax
#   - einsum('bkhq,bchk->bchq') per head
#   - cat heads back
# ═══════════════════════════════════════════════════════════════════════

class VariantH_AppleANEEinsum(torch.nn.Module):
    """Single F layer, Apple ANE per-head einsum attention, no cache writes."""

    def __init__(self, model, layer_idx):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        cfg = model.config
        self.num_heads = cfg.num_attention_heads       # 16
        self.num_kv_heads = cfg.num_key_value_heads    # 4
        self.head_dim = cfg.head_dim                   # 256
        self.n_rep = self.num_heads // self.num_kv_heads  # 4
        self.scale = 1.0 / math.sqrt(self.head_dim)

        layer = model.model.layers[layer_idx]
        self.register_buffer("rope_inv_freq", layer.self_attn.rotary.inv_freq.clone())
        self._rotary_dim = layer.self_attn.rotary.rotary_dim

    def _rope_onthefly(self, position_ids, dtype, device):
        pos_ids = position_ids if position_ids.dim() == 1 else position_ids.squeeze(0)
        t = pos_ids.to(dtype=torch.float32).unsqueeze(-1)
        freqs = t * self.rope_inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(0).to(dtype), emb.sin().unsqueeze(0).to(dtype)

    def forward(self, hidden_states, position_ids, causal_mask, k_cache, v_cache):
        """
        hidden_states: (1, 1, hidden_size)
        position_ids: (1,) int32
        causal_mask: (1, 1, 1, CTX) float16
        k_cache: (1, kv_heads, CTX, head_dim) float16
        v_cache: (1, kv_heads, CTX, head_dim) float16
        """
        layer = self.model.model.layers[self.layer_idx]
        x = layer.input_layernorm(hidden_states)

        # Q/K/V projection via Conv2d + norm + RoPE (same as baseline)
        query_states, key_new, value_new, gate = layer.self_attn._project_qkvg(x)
        query_states = layer.self_attn.q_norm(query_states)
        cos, sin = self._rope_onthefly(position_ids, hidden_states.dtype, hidden_states.device)
        query_states, _ = apply_rotary_pos_emb_single(
            query_states, key_new, cos, sin, self._rotary_dim)
        query_states = query_states.to(MODEL_DTYPE)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  Apple ANE per-head einsum attention
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        B = 1
        S_q = query_states.shape[2]   # 1 for decode
        S_k = k_cache.shape[2]        # CTX

        # Convert Q to channels-first: (B,H,S_q,D) → (B,H*D,1,S_q)
        q_cf = query_states.permute(0, 1, 3, 2).reshape(
            B, self.num_heads * self.head_dim, 1, S_q)

        # Convert K cache to channels-first + transpose for einsum
        # (B,Hkv,S_k,D) → (B,Hkv*D,1,S_k) → transpose → (B,S_k,1,Hkv*D)
        k_cf = k_cache.to(MODEL_DTYPE).permute(0, 1, 3, 2).reshape(
            B, self.num_kv_heads * self.head_dim, 1, S_k)
        k_t = k_cf.transpose(1, 3)   # (B, S_k, 1, Hkv*D)

        # Convert V cache to channels-first: (B,Hkv,S_k,D) → (B,Hkv*D,1,S_k)
        v_cf = v_cache.to(MODEL_DTYPE).permute(0, 1, 3, 2).reshape(
            B, self.num_kv_heads * self.head_dim, 1, S_k)

        # Split into per-head chunks
        mh_q = q_cf.split(self.head_dim, dim=1)      # 16 × (B, 256, 1, S_q)
        mh_k = k_t.split(self.head_dim, dim=3)        # 4 × (B, S_k, 1, 256)
        mh_v = v_cf.split(self.head_dim, dim=1)       # 4 × (B, 256, 1, S_k)

        # Adapt mask: (1,1,1,CTX) → (1,CTX,1,1) for ANE format
        # Apple format: attn_weights are (B,S_k,1,S_q), softmax on dim=1 (S_k)
        if causal_mask is not None:
            mask_ane = causal_mask.permute(0, 3, 1, 2)   # (B, S_k, 1, S_q)
        else:
            mask_ane = None

        # Per-head attention via einsum
        attn_heads = []
        for i in range(self.num_heads):
            kv_idx = i // self.n_rep
            qi = mh_q[i]         # (B, head_dim, 1, S_q)
            ki = mh_k[kv_idx]    # (B, S_k, 1, head_dim)
            vi = mh_v[kv_idx]    # (B, head_dim, 1, S_k)

            # Q·K^T per head → (B, S_k, 1, S_q)
            aw = torch.einsum('bchq,bkhc->bkhq', qi, ki) * self.scale

            if mask_ane is not None:
                aw = aw + mask_ane

            # Softmax over keys (dim=1)
            aw = torch.softmax(aw, dim=1)

            # Weighted V → (B, head_dim, 1, S_q)
            out_h = torch.einsum('bkhq,bchk->bchq', aw, vi)
            attn_heads.append(out_h)

        # Cat all heads: (B, num_heads*head_dim, 1, S_q)
        attn_out_cf = torch.cat(attn_heads, dim=1)

        # Convert back to (B, S_q, H*D) for output projection
        attn_output = attn_out_cf.squeeze(2).permute(0, 2, 1)  # (B, S_q, H*D)

        # Gate + o_proj + residual + MLP
        attn_output = layer.self_attn._project_output(attn_output, x, gate=gate)
        hidden_states = hidden_states + attn_output
        post = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post)
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════

def export_variant(wrapper_cls, model, layer_idx, ctx, label):
    """Export a variant for a given CTX length."""
    cfg = model.config
    wrapper = wrapper_cls(model, layer_idx).eval()

    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, ctx), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    k_cache = torch.zeros((1, cfg.num_key_value_heads, ctx, cfg.head_dim),
                           dtype=MODEL_DTYPE, device=TEST_DEVICE)
    v_cache = torch.zeros((1, cfg.num_key_value_heads, ctx, cfg.head_dim),
                           dtype=MODEL_DTYPE, device=TEST_DEVICE)

    traced = torch.jit.trace(wrapper,
                              (hidden_states, position_ids, causal_mask, k_cache, v_cache),
                              check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="k_cache", shape=k_cache.shape, dtype=np.float16),
            ct.TensorType(name="v_cache", shape=v_cache.shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    path = os.path.join(ARTIFACT_DIR, f'{label}_layer{layer_idx}_ctx{ctx}.mlpackage')
    mlmodel.save(path)
    del mlmodel, traced, wrapper; gc.collect()
    return path


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_mil(path, label):
    """Count MIL ops, flag hostile ones."""
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
        # Count matmul and softmax specifically
        mm = oc.get('matmul', 0) + oc.get('linear', 0) + oc.get('einsum', 0)
        sm = oc.get('softmax', 0)
        print(f"  {label:45s} total={sum(oc.values()):5d}  "
              f"matmul={mm:2d}  softmax={sm:2d}  hostile={h}")
        # Print top 10 op types
        for op_type, cnt in oc.most_common(10):
            print(f"    {op_type:30s}: {cnt}")
        sys.stdout.flush()


def measure_ane(path, label, cu=ct.ComputeUnit.CPU_AND_NE, warmup=10, runs=30):
    """Measure ANE utilization via process_time vs wall_time."""
    ml = ct.models.MLModel(path, compute_units=cu)
    spec = ct.utils.load_spec(path)
    fn = list(spec.mlProgram.functions.keys())[0]
    func = spec.mlProgram.functions[fn]
    inputs_dict = {}
    for inp in func.inputs:
        if inp.type.WhichOneof('type') == 'tensorType':
            tt = inp.type.tensorType
            shape = tuple(d.constant.size for d in tt.dimensions)
            dt_map = {1: np.float32, 3: np.float16, 5: np.int32}
            inputs_dict[inp.name] = np.zeros(shape, dtype=dt_map.get(tt.dataType, np.float16))
    for _ in range(warmup):
        ml.predict(inputs_dict)
    walls, cpus = [], []
    for _ in range(runs):
        tw = time.perf_counter(); tc = time.process_time()
        ml.predict(inputs_dict)
        cpus.append(time.process_time() - tc)
        walls.append(time.perf_counter() - tw)
    mw = sorted(walls)[len(walls) // 2] * 1000
    mc = sorted(cpus)[len(cpus) // 2] * 1000
    cf = mc / mw if mw > 0 else 1.0
    af = max(0, 1 - cf)
    cu_s = 'ANE' if cu == ct.ComputeUnit.CPU_AND_NE else 'CPU'
    print(f"  {label:45s} [{cu_s}] wall={mw:7.2f}ms "
          f"cpu={mc:7.2f}ms  CPU%={cf * 100:5.1f}%  ANE%={af * 100:5.1f}%")
    sys.stdout.flush()
    del ml; gc.collect()
    return mw, mc, af


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--layer", type=int, default=3, help="F-layer index (default: 3)")
    parser.add_argument("--ctx", type=int, nargs='+', default=None,
                        help="CTX lengths to test (default: 128 256 512 1024 2048)")
    args = parser.parse_args()

    LAYER_IDX = args.layer
    CTX_LIST = args.ctx or [128, 256, 512, 1024, 2048]

    VARIANTS = {
        'F_std_matmul': VariantF_StandardMatmul,
        'H_apple_einsum': VariantH_AppleANEEinsum,
    }

    print("=" * 80)
    print(f"  APPLE ANE ATTENTION TEST — Layer {LAYER_IDX}")
    print(f"  CTX lengths: {CTX_LIST}")
    print(f"  Variants: {list(VARIANTS.keys())}")
    print("=" * 80)
    sys.stdout.flush()

    # ── Build paths ──
    paths = {}  # (variant, ctx) → path
    for vname in VARIANTS:
        for ctx in CTX_LIST:
            paths[(vname, ctx)] = os.path.join(
                ARTIFACT_DIR, f'{vname}_layer{LAYER_IDX}_ctx{ctx}.mlpackage')

    # ── Export ──
    if not args.skip_export:
        print("\n  Loading model weights...")
        sys.stdout.flush()
        cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
        cfg.context_length = max(CTX_LIST)
        cfg.state_length = max(CTX_LIST)
        model = Qwen35ForCausalLM(cfg)
        assert model.load_pretrained_weights(HF_MODEL)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        for vname, vcls in VARIANTS.items():
            for ctx in CTX_LIST:
                p = paths[(vname, ctx)]
                if os.path.exists(p):
                    print(f"  SKIP (exists): {vname} ctx={ctx}")
                    continue
                print(f"\n  Exporting {vname} ctx={ctx}...")
                sys.stdout.flush()
                t0 = time.time()
                # Need to reconfigure context for each CTX
                cfg.context_length = ctx
                cfg.state_length = ctx
                paths[(vname, ctx)] = export_variant(vcls, model, LAYER_IDX, ctx, vname)
                print(f"    Saved in {time.time() - t0:.1f}s")
                gc.collect()

        del model; gc.collect()

    # ── MIL Analysis (just one CTX per variant) ──
    print("\n" + "=" * 80)
    print("  MIL OP ANALYSIS (representative CTX)")
    print("=" * 80)
    rep_ctx = CTX_LIST[0]
    for vname in VARIANTS:
        p = paths[(vname, rep_ctx)]
        if os.path.exists(p):
            analyze_mil(p, f"{vname} ctx={rep_ctx}")
        else:
            print(f"  {vname:45s} MISSING")
    sys.stdout.flush()

    # ── ANE Measurement ──
    print("\n" + "=" * 80)
    print("  ANE UTILIZATION (CPU_AND_NE, 30 runs, 10 warmup)")
    print("=" * 80)

    results = {}  # (variant, ctx) → (wall, cpu_ms, ane_frac)
    for vname in VARIANTS:
        print(f"\n  --- {vname} ---")
        for ctx in CTX_LIST:
            p = paths[(vname, ctx)]
            if os.path.exists(p):
                w, c, a = measure_ane(p, f"{vname} ctx={ctx}")
                results[(vname, ctx)] = (w, c, a)

    # CPU_ONLY reference
    print(f"\n  --- CPU_ONLY reference ---")
    for vname in VARIANTS:
        for ctx in [CTX_LIST[0], CTX_LIST[-1]]:
            p = paths[(vname, ctx)]
            if os.path.exists(p):
                w, c, a = measure_ane(p, f"{vname} ctx={ctx}",
                                       cu=ct.ComputeUnit.CPU_ONLY)
                results[(vname + '_CPUONLY', ctx)] = (w, c, a)

    # ── Summary Table ──
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE")
    print("=" * 80)
    print(f"  {'CTX':>5s}  ", end='')
    for vname in VARIANTS:
        print(f"{'wall(' + vname[:8] + ')':>14s}  {'ANE%':>6s}  ", end='')
    print(f"{'speedup':>8s}")
    print(f"  {'-' * 5}  ", end='')
    for _ in VARIANTS:
        print(f"{'-' * 14}  {'-' * 6}  ", end='')
    print(f"{'-' * 8}")

    for ctx in CTX_LIST:
        print(f"  {ctx:5d}  ", end='')
        walls = {}
        for vname in VARIANTS:
            key = (vname, ctx)
            if key in results:
                w, c, a = results[key]
                print(f"{w:12.2f}ms  {a * 100:5.1f}%  ", end='')
                walls[vname] = w
            else:
                print(f"{'N/A':>14s}  {'N/A':>6s}  ", end='')
        # Speedup: H vs F
        if 'F_std_matmul' in walls and 'H_apple_einsum' in walls:
            spd = walls['F_std_matmul'] / walls['H_apple_einsum']
            print(f"{spd:6.2f}x", end='')
        print()

    # ── Verdict ──
    print("\n" + "=" * 80)
    ane_h = [results[(k)][2] for k in results
             if k[0] == 'H_apple_einsum' and results[k][2] > 0.05]
    if ane_h:
        print(f"  ✅ Apple ANE einsum achieved ANE execution at {len(ane_h)} CTX lengths!")
        print(f"     ANE fractions: {[f'{a*100:.1f}%' for a in ane_h]}")
    else:
        ane_f = [results[(k)][2] for k in results if k[0] == 'F_std_matmul']
        print(f"  ❌ Apple ANE einsum did NOT improve ANE utilization.")
        print(f"     F (matmul): {[f'{a*100:.1f}%' for a in ane_f]}")
        ane_h_all = [results[(k)][2] for k in results if k[0] == 'H_apple_einsum']
        print(f"     H (einsum): {[f'{a*100:.1f}%' for a in ane_h_all]}")
        # Check latency difference
        for ctx in CTX_LIST:
            kf = ('F_std_matmul', ctx)
            kh = ('H_apple_einsum', ctx)
            if kf in results and kh in results:
                wf = results[kf][0]
                wh = results[kh][0]
                delta = (wh - wf) / wf * 100
                print(f"     CTX={ctx}: H is {delta:+.1f}% vs F wall time")
    print("=" * 80)
    print("  Done!")
