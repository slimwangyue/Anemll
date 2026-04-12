#!/usr/bin/env python3
"""
L-layer ANE investigation: targeted op-level ablation.

Variants:
  1. baseline (current code with direct norms)
  2. l2norm_matmul: replace reduce_sum in l2norm with matmul dot product
  3. l2norm_only_matmul: only l2norm matmul, keep original recurrence
  4. rmsnorm_sum: RMSNorm via reduce_sum/N instead of reduce_mean

Usage:
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/l_layer_cpu_investigation.py
"""
import sys, os, time, gc, resource, warnings, copy
from collections import Counter
warnings.filterwarnings('ignore')

REPO_ROOT = '/Volumes/MySSD/Anemll'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts_qwen3_5'))
os.chdir(REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

torch.set_grad_enabled(False)

HIDDEN = 2560
MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"

# ---- Custom replacements ----

def _l2norm_matmul(x, dim=-1, eps=1e-6):
    """L2 norm using matmul for the squared-norm instead of reduce_sum."""
    sq_sum = torch.matmul(x.unsqueeze(-2), x.unsqueeze(-1)).squeeze(-1)
    sq_sum = torch.clamp(sq_sum, min=eps)
    return x * torch.rsqrt(sq_sum)


def _recurrent_matmul(
    query, key, value, g, beta, recurrent_state,
    output_final_state=True,
    expected_batch_size=None, expected_num_heads=None,
    expected_seq_len=None, expected_k_dim=None, expected_v_dim=None,
    math_dtype=torch.float32,
):
    """Recurrence with explicit matmul + l2norm via matmul."""
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    query = _l2norm_matmul(query, dim=-1)
    key = _l2norm_matmul(key, dim=-1)
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
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        # matmul instead of multiply+reduce_sum
        kv_mem = torch.matmul(state.transpose(-2, -1), k_t.unsqueeze(-1)).squeeze(-1)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = torch.matmul(state.transpose(-2, -1), q_t.unsqueeze(-1)).squeeze(-1)
    if not output_final_state:
        state = None
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state


# ---- Load model ----
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, Qwen35LinearAttention,
    _l2norm, Qwen35RMSNormGated, Qwen35RMSNorm,
)
import anemll.models.qwen3_5_model as qm

HF_MODEL = 'models/Qwen__Qwen3.5-4B'
print("Loading model...")
cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(HF_MODEL)
model.eval()
for p in model.parameters():
    p.requires_grad = False

from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

out_dir = 'artifacts/l_layer_cpu_investigation'
os.makedirs(out_dir, exist_ok=True)

# Save originals
orig_l2 = qm._l2norm
orig_recurrent = Qwen35LinearAttention._recurrent_gated_delta_rule
orig_rmsnorm_fwd = Qwen35RMSNorm.forward
orig_rmsnormg_fwd = Qwen35RMSNormGated.forward

# ---- Correctness check ----
print("\n=== Correctness check ===")
layer = model.model.layers[0].self_attn
bsz, seq_len = 1, 1
hidden = torch.randn(bsz, seq_len, HIDDEN, dtype=MODEL_DTYPE)
conv_state = torch.zeros(bsz, layer.conv_dim, layer.linear_conv_kernel_dim, dtype=MODEL_DTYPE)
rec_state = torch.zeros(bsz, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, dtype=torch.float32)

with torch.no_grad():
    out_base, _, _ = layer._forward_impl(hidden, conv_state, rec_state, True, bsz, seq_len)

# Test matmul variant
qm._l2norm = _l2norm_matmul
Qwen35LinearAttention._recurrent_gated_delta_rule = staticmethod(_recurrent_matmul)
with torch.no_grad():
    out_matmul, _, _ = layer._forward_impl(hidden, conv_state, rec_state, True, bsz, seq_len)
qm._l2norm = orig_l2
Qwen35LinearAttention._recurrent_gated_delta_rule = orig_recurrent

a = out_base.flatten().float().numpy()
b = out_matmul.flatten().float().numpy()
cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
print(f"  l2norm_matmul + recurrence_matmul vs baseline: cos={cos:.8f}")

# ---- Export variants ----
variant_configs = [
    ('baseline', lambda: None, lambda: None),
    ('l2norm_matmul', 
     lambda: (
         setattr(qm, '_l2norm', _l2norm_matmul),
         setattr(Qwen35LinearAttention, '_recurrent_gated_delta_rule', staticmethod(_recurrent_matmul)),
     ),
     lambda: (
         setattr(qm, '_l2norm', orig_l2),
         setattr(Qwen35LinearAttention, '_recurrent_gated_delta_rule', orig_recurrent),
     )),
    ('l2norm_only_matmul',
     lambda: setattr(qm, '_l2norm', _l2norm_matmul),
     lambda: setattr(qm, '_l2norm', orig_l2)),
]

variants = {}

# Baseline: use existing artifact
base_path = 'artifacts/direct_norm_validation/chunk0_decode.mlpackage'
if os.path.exists(base_path):
    variants['baseline'] = base_path
    print(f"\nBaseline: {base_path} (cached)")

for name, setup, teardown in variant_configs:
    if name == 'baseline':
        continue
    path = os.path.join(out_dir, f'chunk0_{name}_decode.mlpackage')
    if os.path.exists(path):
        print(f"\n{name}: {path} (cached)")
        variants[name] = path
        continue
    
    print(f"\nExporting {name}...")
    setup()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=6, per_channel=8,
                           compute_precision="float16")
    t0 = time.time()
    ml = conv.convert_part_2(model, chunk_idx=0, total_chunks=NUM_CHUNKS,
                             override_start_layer=0, override_end_layer=3)
    ml.save(path)
    print(f"  Saved in {time.time()-t0:.1f}s")
    teardown()
    del ml, conv
    gc.collect()
    variants[name] = path

print(f"\nAll variants: {list(variants.keys())}")

# ---- MIL op comparison ----
print("\n\n=== MIL OP COMPARISON ===")
for name, path in sorted(variants.items()):
    spec = ct.utils.load_spec(path)
    mlprog = spec.mlProgram
    for fn in mlprog.functions:
        func = mlprog.functions[fn]
        for k in func.block_specializations:
            block = func.block_specializations[k]
            break
        op_counts = Counter()
        for op in block.operations:
            op_counts[op.type] += 1
        total = sum(op_counts.values())
        weight = sum(v for k, v in op_counts.items() if k in ('const', 'constexpr_lut_to_dense'))
        compute = total - weight
        trans = op_counts.get('transpose', 0)
        rsum = op_counts.get('reduce_sum', 0)
        rmean = op_counts.get('reduce_mean', 0)
        matm = op_counts.get('matmul', 0)
        conv_op = op_counts.get('conv', 0)
        sbi = op_counts.get('slice_by_index', 0)
        mul_op = op_counts.get('mul', 0)
        add_op = op_counts.get('add', 0)
        print(f"  {name:25s}: compute={compute:3d} trans={trans:2d} "
              f"reduce_sum={rsum:2d} reduce_mean={rmean:2d} matmul={matm:2d} "
              f"conv={conv_op:2d} sbi={sbi:2d} mul={mul_op:2d}")

# ---- Timing comparison ----
print("\n\n=== TIMING ===")
nl = 3
np.random.seed(42)
inputs = {
    'hidden_states': np.random.randn(1, 1, HIDDEN).astype(np.float16) * 0.01,
    'position_ids': np.array([0], dtype=np.int32),
    'causal_mask': np.zeros((1, 1, 1, CTX), dtype=np.float16),
    'current_pos': np.array([0], dtype=np.int32),
    'linear_conv_state': np.zeros((nl, 1024, 32), dtype=np.float16),
    'linear_recurrent_state': np.zeros((nl, 32, 128, 128), dtype=np.float16),
}

n_warmup = 5
n_runs = 30
results = {}

for name, path in sorted(variants.items()):
    print(f"  Measuring {name}...")
    ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = ml.make_state()
    
    for i in range(n_warmup):
        ml.predict(inputs, state=state)
    
    times = []
    for i in range(n_runs):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        ml.predict(inputs, state=state)
        wall = (time.perf_counter() - t0) * 1000
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)) * 1000
        times.append((wall, cpu))
    
    w = np.median([t[0] for t in times])
    c = np.median([t[1] for t in times])
    a = max(0, w - c)
    results[name] = (w, c, a, a / w * 100 if w > 0 else 0)
    
    # Also check accuracy against baseline
    if name != 'baseline' and 'baseline' in variants:
        ml_base = ct.models.MLModel(variants['baseline'], compute_units=ct.ComputeUnit.CPU_AND_NE)
        st_base = ml_base.make_state()
        st_var = ml.make_state()
        out_base = ml_base.predict(inputs, state=st_base)
        out_var = ml.predict(inputs, state=st_var)
        a_arr = np.asarray(out_base['output_hidden_states']).flatten().astype(np.float64)
        b_arr = np.asarray(out_var['output_hidden_states']).flatten().astype(np.float64)
        cos_val = float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-30))
        print(f"    accuracy vs baseline: cos={cos_val:.6f}")
        del ml_base
    
    del ml
    gc.collect()

# ---- Results table ----
print("\n\n=== RESULTS ===")
print(f"{'Variant':25s} {'Wall(ms)':>10s} {'CPU(ms)':>10s} {'ANE(ms)':>10s} {'ANE%':>6s}")
print("-" * 65)
for name in sorted(results.keys()):
    w, c, a, pct = results[name]
    print(f"{name:25s} {w:10.2f} {c:10.2f} {a:10.2f} {pct:5.0f}%")

if 'baseline' in results:
    base_w, base_c = results['baseline'][0], results['baseline'][1]
    print(f"\nDelta from baseline:")
    for name in sorted(results.keys()):
        if name != 'baseline':
            w, c, a, pct = results[name]
            print(f"  {name}: wall {w-base_w:+.2f}ms, cpu {c-base_c:+.2f}ms, "
                  f"ane% {pct-results['baseline'][3]:+.1f}%")

print("\nDone.")
