#!/usr/bin/env python3
"""Export + measure 3L direct-layout variant (fewer transposes in L-layer)."""
import sys, os, time, gc, resource, warnings
from collections import Counter
from typing import Tuple
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Volumes/MySSD/Anemll')
sys.path.insert(0, '/Volumes/MySSD/Anemll/scripts_qwen3_5')
os.chdir('/Volumes/MySSD/Anemll')

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct
from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

torch.set_grad_enabled(False)
HIDDEN = 2560
MODEL_DTYPE = torch.float16

import anemll.models.qwen3_5_model as qm
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, Qwen35LinearLayoutStage,
    Qwen35LinearAttention, _l2norm,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

HF_MODEL = 'models/Qwen__Qwen3.5-4B'
out_dir = 'artifacts/l_layer_cpu_investigation'

orig_layout = Qwen35LinearLayoutStage.forward
orig_recur = Qwen35LinearAttention._recurrent_gated_delta_rule

def direct_layout_forward(self, conv_out_cf, z_cf, b_cf, a_cf, bsz, seq_len, force_fp16_math=False):
    query_cf, key_cf, value_cf = torch.split(conv_out_cf, [self.key_dim, self.key_dim, self.value_dim], dim=1)
    query = query_cf.reshape(bsz, self.num_k_heads, self.head_k_dim, seq_len).transpose(-2, -1)
    key = key_cf.reshape(bsz, self.num_k_heads, self.head_k_dim, seq_len).transpose(-2, -1)
    value = value_cf.reshape(bsz, self.num_v_heads, self.head_v_dim, seq_len).transpose(-2, -1)
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
    if self.num_v_heads // self.num_k_heads > 1:
        rep = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(rep, dim=1)
        key = key.repeat_interleave(rep, dim=1)
    return query, key, value, g, beta, z

def direct_recurrent(query, key, value, g, beta, recurrent_state,
    output_final_state=True, expected_batch_size=None, expected_num_heads=None,
    expected_seq_len=None, expected_k_dim=None, expected_v_dim=None,
    math_dtype=torch.float32):
    initial_dtype = query.dtype
    query, key, value = query.contiguous().to(math_dtype), key.contiguous().to(math_dtype), value.contiguous().to(math_dtype)
    beta = beta.transpose(1, 2).contiguous().to(math_dtype)
    g = g.transpose(1, 2).contiguous().to(math_dtype)
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

# ---- Load model ----
print("Loading model...")
cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
cfg.context_length = CTX; cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(HF_MODEL)
model.eval()
for p in model.parameters(): p.requires_grad = False

# ---- Export ----
direct_path = os.path.join(out_dir, '3L_direct_layout_decode.mlpackage')
if os.path.exists(direct_path):
    import shutil; shutil.rmtree(direct_path)

print("Exporting 3L direct layout variant...")
Qwen35LinearLayoutStage.forward = direct_layout_forward
Qwen35LinearAttention._recurrent_gated_delta_rule = staticmethod(direct_recurrent)

conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                       num_chunks=NUM_CHUNKS, lut_bits=6, per_channel=8,
                       compute_precision="float16")
t0 = time.time()
ml = conv.convert_part_2(model, chunk_idx=0, total_chunks=NUM_CHUNKS,
                         override_start_layer=0, override_end_layer=3)
ml.save(direct_path)
print(f"  Saved in {time.time()-t0:.1f}s")

Qwen35LinearLayoutStage.forward = orig_layout
Qwen35LinearAttention._recurrent_gated_delta_rule = orig_recur
del ml, conv, model; gc.collect()

# ---- MIL + Timing ----
print("\n=== MIL OP COMPARISON ===")
base_path = os.path.join(out_dir, '3L_baseline_decode.mlpackage')
for label, path in [('baseline', base_path), ('direct_layout', direct_path)]:
    spec = ct.utils.load_spec(path)
    mlprog = spec.mlProgram
    for fn in mlprog.functions:
        func = mlprog.functions[fn]
        for k in func.block_specializations: block = func.block_specializations[k]; break
        op_counts = Counter()
        for op in block.operations: op_counts[op.type] += 1
        total = sum(op_counts.values())
        weight = sum(v for k,v in op_counts.items() if k in ('const','constexpr_lut_to_dense'))
        compute = total - weight
        trans = op_counts.get('transpose', 0)
        conv_op = op_counts.get('conv', 0)
        rshp = op_counts.get('reshape', 0)
        print(f"  {label:15s}: compute={compute:3d} trans={trans:2d} conv={conv_op:2d} reshape={rshp:2d}")

print("\n=== TIMING ===")
nl = 3
np.random.seed(42)
inputs = {
    'hidden_states': np.random.randn(1,1,HIDDEN).astype(np.float16) * 0.01,
    'position_ids': np.array([0], dtype=np.int32),
    'causal_mask': np.zeros((1,1,1,CTX), dtype=np.float16),
    'current_pos': np.array([0], dtype=np.int32),
    'linear_conv_state': np.zeros((nl, 1024, 32), dtype=np.float16),
    'linear_recurrent_state': np.zeros((nl, 32, 128, 128), dtype=np.float16),
}

for label, path in [('3L_baseline', base_path), ('3L_direct_layout', direct_path)]:
    ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = ml.make_state()
    for i in range(5): ml.predict(inputs, state=state)
    times = []
    for i in range(30):
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
    print(f"  {label:20s}: wall={w:.2f} cpu={c:.2f} ane={a:.2f} ({a/w*100:.0f}%)")
    if 'direct' in label:
        ml_base = ct.models.MLModel(base_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        st_b = ml_base.make_state(); st_d = ml.make_state()
        out_b = ml_base.predict(inputs, state=st_b)
        out_d = ml.predict(inputs, state=st_d)
        ab = np.asarray(out_b['output_hidden_states']).flatten().astype(np.float64)
        ad = np.asarray(out_d['output_hidden_states']).flatten().astype(np.float64)
        cos = float(np.dot(ab,ad)/(np.linalg.norm(ab)*np.linalg.norm(ad)+1e-30))
        print(f"    CoreML accuracy: cos={cos:.6f}")
        del ml_base
    del ml; gc.collect()

print("\nDone.")
