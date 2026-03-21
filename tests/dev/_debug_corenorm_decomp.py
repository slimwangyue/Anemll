#!/usr/bin/env python3
"""Split CoreNorm stage into: (A) recurrence only, (B) norm+projection only.
Determines if the channel-0 error comes from the gated delta rule recurrence
or from the Qwen35RMSNormGated + output projection.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, shutil
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    Qwen35LinearAttention,
    MODEL_DTYPE, TEST_DEVICE,
)
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_corenorm_split"
SEQ_LEN = 256
CTX = 1024

os.makedirs(OUT_DIR, exist_ok=True)

def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))

def report(ref, cml, label, top_n=5):
    rf = ref.astype(np.float32)
    cf = cml.astype(np.float32)
    diff = np.abs(rf - cf)
    cos = cosine(ref, cml)
    print(f"  [{label}]")
    print(f"    cos={cos:.10f}  max_abs={diff.max():.6f}  mean_abs={diff.mean():.8f}")
    if len(diff.shape) >= 2:
        last_dim = diff.shape[-1]
        flat_diff = diff.reshape(-1, last_dim)
        ch_err = flat_diff.mean(axis=0)
        worst = np.argsort(ch_err)[-top_n:][::-1]
        for c in worst:
            bias = (cf.reshape(-1, last_dim)[:,c] - rf.reshape(-1, last_dim)[:,c]).mean()
            print(f"      ch={c:4d} mean_abs={ch_err[c]:.6f} bias={bias:+.6f}")
    return cos

# ── Load model ──
print("=" * 70)
print(f"  CoreNorm Decomposition  (seq={SEQ_LEN})")
print("=" * 70)
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX; cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
model.eval()
for p in model.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
prompt = "Explain stack and heap memory in one paragraph and give one debugging tip."
text = prompt
while True:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    if ids.shape[1] >= SEQ_LEN:
        ids = ids[:, :SEQ_LEN]; break
    text = text + " " + prompt
ids = ids.to(torch.int32)
with torch.no_grad():
    embed = model.model.embed_tokens(ids).to(torch.float16)

attn = model.model.layers[0].self_attn
conv_dim = attn.conv_dim; conv_kernel = attn.linear_conv_kernel_dim
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)
attn.export_expected_batch_size = 1
attn.export_expected_seq_len = SEQ_LEN

# ── PyTorch stage intermediates ──
with torch.no_grad():
    x_norm = model.model.layers[0].input_layernorm(embed)
    conv_s = torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE)
    rec_s = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=MODEL_DTYPE)
    mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(x_norm)
    conv_out, _ = attn.conv_stage(mixed_qkv_pre, conv_s, expected_seq_len=SEQ_LEN)
    query, key, value, g, beta, z = attn.layout_stage(
        conv_out, z_cf, b_cf, a_cf, 1, SEQ_LEN)

    # Recurrence only (before norm + projection)
    core_raw, next_rec = Qwen35LinearAttention._chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=rec_s, output_final_state=True,
        expected_batch_size=1, expected_num_heads=attn.num_v_heads,
        expected_seq_len=SEQ_LEN, expected_k_dim=attn.head_k_dim,
        expected_v_dim=attn.head_v_dim)

    # Norm + projection (applied to recurrence output)
    core_normed = attn.core_norm_stage.norm(
        core_raw.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim)
    ).reshape(1, SEQ_LEN, attn.value_dim)
    core_cf = attn.core_norm_stage.to_channels_first_4d(core_normed)
    final_out = attn.core_norm_stage.from_channels_first_4d(
        attn.core_norm_stage.conv2d_proj_cf(attn.core_norm_stage.out_proj, core_cf))

print(f"  core_raw: {core_raw.shape}")
print(f"  core_normed: {core_normed.shape}")
print(f"  final_out: {final_out.shape}")


# ── TEST A: Recurrence Only (no norm, no projection) ──
print(f"\n{'='*70}")
print("  TEST A: Recurrence Only (gated delta rule)")
print(f"{'='*70}")

class RecurrenceOnlyWrapper(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.register_buffer("rec_state", torch.zeros(
            1, attn_module.num_v_heads, attn_module.head_k_dim, attn_module.head_v_dim, dtype=MODEL_DTYPE))
        self.num_v_heads = attn_module.num_v_heads
        self.head_k_dim = attn_module.head_k_dim
        self.head_v_dim = attn_module.head_v_dim

    def forward(self, query, key, value, g, beta):
        out, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=self.rec_state, output_final_state=True,
            expected_batch_size=1, expected_num_heads=self.num_v_heads,
            expected_seq_len=SEQ_LEN, expected_k_dim=self.head_k_dim,
            expected_v_dim=self.head_v_dim)
        return out

rw = RecurrenceOnlyWrapper(attn).eval()
print("  Tracing...")
traced_r = torch.jit.trace(rw, (query, key, value, g, beta))
for name, buf in traced_r.named_buffers():
    if "rec_state" in name: buf.zero_()

rec_states = [ct.StateType(
    wrapped_type=ct.TensorType(
        shape=(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim), dtype=np.float16),
    name="rec_state")]

pkg_r = os.path.join(OUT_DIR, "recurrence_only.mlpackage")
if os.path.exists(pkg_r): shutil.rmtree(pkg_r)
print("  Converting...")
mlm_r = ct.convert(
    traced_r,
    inputs=[
        ct.TensorType(name="query", shape=query.shape, dtype=np.float16),
        ct.TensorType(name="key", shape=key.shape, dtype=np.float16),
        ct.TensorType(name="value", shape=value.shape, dtype=np.float16),
        ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
        ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
    ],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    states=rec_states,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_r.save(pkg_r)
del mlm_r, traced_r, rw; gc.collect()

print("  Running on ANE...")
cml_r = ct.models.MLModel(pkg_r, compute_units=ct.ComputeUnit.CPU_AND_NE)
state_r = cml_r.make_state()
out_r = cml_r.predict({
    "query": query.numpy().astype(np.float16),
    "key": key.numpy().astype(np.float16),
    "value": value.numpy().astype(np.float16),
    "g": g.numpy().astype(np.float16),
    "beta": beta.numpy().astype(np.float16),
}, state=state_r)
cml_rec = list(out_r.values())[0]
# core_raw shape: (1, num_v_heads, seq_len, head_v_dim) transposed → (1, seq_len, num_v_heads * head_v_dim)
# Let me check shapes
print(f"  CoreML output: {cml_rec.shape}")
print(f"  PyTorch ref:   {core_raw.shape}")
cos_rec = report(core_raw.numpy(), cml_rec, "Recurrence Only (before norm+proj)")
del cml_r, state_r; gc.collect()


# ── TEST B: Norm + Output Projection Only ──
print(f"\n{'='*70}")
print("  TEST B: Norm + Output Projection (using PyTorch recurrence output)")
print(f"{'='*70}")

class NormProjWrapper(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.norm = attn_module.core_norm_stage.norm
        self.out_proj = attn_module.core_norm_stage.out_proj
        self.head_v_dim = attn_module.head_v_dim
        self.value_dim = attn_module.value_dim

    def forward(self, core_raw, z):
        # core_raw: (1, seq_len, value_dim) after transpose
        core_normed = self.norm(
            core_raw.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim)
        ).reshape(1, SEQ_LEN, self.value_dim)
        cf = core_normed.transpose(1, 2).unsqueeze(2)  # to_channels_first_4d
        out = self.out_proj(cf.to(MODEL_DTYPE))
        return out.squeeze(2).transpose(1, 2)  # from_channels_first_4d

npw = NormProjWrapper(attn).eval()
# Need core_raw in the right shape for the norm+proj
# core_raw is already (1, seq_len, num_v_heads, head_v_dim) = (1, 256, 32, 128)
# Flatten last two dims: (1, 256, 4096) — same as core.reshape(-1, head_v_dim) then reshape back
core_for_norm = core_raw.reshape(1, SEQ_LEN, attn.value_dim).contiguous()
z_for_norm = z.reshape(1, SEQ_LEN, attn.value_dim).contiguous()

print("  Tracing...")
traced_np = torch.jit.trace(npw, (core_for_norm, z_for_norm))

pkg_np = os.path.join(OUT_DIR, "norm_proj_only.mlpackage")
if os.path.exists(pkg_np): shutil.rmtree(pkg_np)
print("  Converting...")
mlm_np = ct.convert(
    traced_np,
    inputs=[
        ct.TensorType(name="core_raw", shape=core_for_norm.shape, dtype=np.float16),
        ct.TensorType(name="z", shape=z_for_norm.shape, dtype=np.float16),
    ],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_np.save(pkg_np)
del mlm_np, traced_np, npw; gc.collect()

print("  Running on ANE (with PyTorch recurrence output as input)...")
cml_np = ct.models.MLModel(pkg_np, compute_units=ct.ComputeUnit.CPU_AND_NE)
out_np = cml_np.predict({
    "core_raw": core_for_norm.numpy().astype(np.float16),
    "z": z_for_norm.numpy().astype(np.float16),
})
cml_norm_proj = list(out_np.values())[0]
cos_norm_proj = report(final_out.numpy(), cml_norm_proj, "Norm + Projection (PyTorch recurrence input)")
del cml_np; gc.collect()


# ── TEST C: L2 Norm isolation ──
print(f"\n{'='*70}")
print("  TEST C: L2 Norm in isolation")
print(f"{'='*70}")

class L2NormWrapper(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2.0, dim=-1)

from anemll.models.qwen3_5_model import _l2norm

# Test with actual query/key values
with torch.no_grad():
    torch_q_normed = _l2norm(query.to(torch.float32), dim=-1).to(torch.float16)

l2w = L2NormWrapper().eval()
traced_l2 = torch.jit.trace(l2w, (query.to(torch.float16),))
pkg_l2 = os.path.join(OUT_DIR, "l2norm_only.mlpackage")
if os.path.exists(pkg_l2): shutil.rmtree(pkg_l2)
mlm_l2 = ct.convert(
    traced_l2,
    inputs=[ct.TensorType(name="x", shape=query.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_l2.save(pkg_l2)
del mlm_l2, traced_l2, l2w; gc.collect()

cml_l2 = ct.models.MLModel(pkg_l2, compute_units=ct.ComputeUnit.CPU_AND_NE)
out_l2 = cml_l2.predict({"x": query.numpy().astype(np.float16)})
cml_q_normed = list(out_l2.values())[0]
cos_l2 = report(torch_q_normed.numpy(), cml_q_normed, "L2 Norm (query)")
del cml_l2; gc.collect()


# ── Summary ──
print(f"\n{'='*70}")
print(f"  CoreNorm Decomposition Summary")
print(f"{'='*70}")
print(f"  {'Test':<50} {'cosine':<15}")
print(f"  {'-'*65}")
print(f"  {'A: Recurrence Only (before norm+proj)':<50} {cos_rec:<15.10f}")
print(f"  {'B: Norm+Projection (PyTorch rec input)':<50} {cos_norm_proj:<15.10f}")
print(f"  {'C: L2 Norm (query)':<50} {cos_l2:<15.10f}")
print(f"  ---")
print(f"  {'Full CoreNorm (from prev test)':<50} {'0.9956 (ref)':<15}")
print(f"  {'Full Monolithic (from prev test)':<50} {'0.9940 (ref)':<15}")
print(f"{'='*70}")
