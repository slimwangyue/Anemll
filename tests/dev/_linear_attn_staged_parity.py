#!/usr/bin/env python3
"""Compare monolithic vs 4-staged linear attention CoreML export parity on ANE.

Exports a single linear-attention layer (layer 0) two ways:
  1. Monolithic: one CoreML model with the full forward_prefill_export
  2. 4-Stage: four separate CoreML models (Proj, Conv, Layout, CoreNorm)

Then compares both against the PyTorch fp32 reference.

Usage:
    python tests/dev/_linear_attn_staged_parity.py
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, shutil, time
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_linear_staged_parity"
LAYER_IDX = 0  # first linear-attention layer
SEQ_LEN = 1    # single token (decode-like, uses recurrent path)

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load model ──────────────────────────────────────────────
print("=" * 70)
print("  Linear Attention: Monolithic vs 4-Staged Parity")
print("=" * 70)
print("Loading model …")
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = 256
cfg.state_length = 256
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
model.eval()
for p in model.parameters():
    p.requires_grad = False

layer = model.model.layers[LAYER_IDX]
assert layer.layer_type == "linear_attention", f"Layer {LAYER_IDX} is {layer.layer_type}"
attn = layer.self_attn

conv_dim = attn.conv_dim
conv_kernel = attn.linear_conv_kernel_dim
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)
print(f"  Layer {LAYER_IDX}: linear_attention")
print(f"  conv_dim={conv_dim}  conv_kernel={conv_kernel}  hidden={cfg.hidden_size}")
print(f"  num_k_heads={attn.num_k_heads}  num_v_heads={attn.num_v_heads}")
print(f"  head_k_dim={attn.head_k_dim}  head_v_dim={attn.head_v_dim}")
print(f"  seq_len={SEQ_LEN}")

# ── 2. Create test input ───────────────────────────────────────
torch.manual_seed(42)
# Use layernorm'd hidden state as input to linear attention
hidden_states = torch.randn(1, SEQ_LEN, cfg.hidden_size, dtype=torch.float16) * 0.1
conv_state = torch.randn(1, conv_dim, conv_kernel, dtype=torch.float16) * 0.01
recurrent_state = torch.randn(
    1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim,
    dtype=torch.float16
) * 0.01

np.save(os.path.join(OUT_DIR, "hidden_states.npy"), hidden_states.numpy())
np.save(os.path.join(OUT_DIR, "conv_state.npy"), conv_state.numpy())
np.save(os.path.join(OUT_DIR, "recurrent_state.npy"), recurrent_state.numpy())
print(f"  hidden_states: {hidden_states.shape}  conv_state: {conv_state.shape}")

# ── 3. PyTorch reference ───────────────────────────────────────
print("\n── PyTorch reference ──")
attn.export_expected_batch_size = 1
attn.export_expected_seq_len = SEQ_LEN
with torch.no_grad():
    torch_out, torch_next_conv, torch_next_rec = attn.forward_prefill_export(
        hidden_states=hidden_states.clone(),
        conv_state=conv_state.clone(),
        recurrent_state=recurrent_state.clone(),
        has_previous_state=True,
    )
print(f"  out: {torch_out.shape}  next_conv: {torch_next_conv.shape}  next_rec: {torch_next_rec.shape}")
np.save(os.path.join(OUT_DIR, "torch_out.npy"), torch_out.detach().numpy())
np.save(os.path.join(OUT_DIR, "torch_next_conv.npy"), torch_next_conv.detach().numpy())
np.save(os.path.join(OUT_DIR, "torch_next_rec.npy"), torch_next_rec.detach().numpy())

# Also get intermediate values from staged execution for per-stage comparison
with torch.no_grad():
    mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(hidden_states.clone())
    conv_out_cf, next_conv_st = attn.conv_stage(mixed_qkv_pre, conv_state.clone(), expected_seq_len=SEQ_LEN)
    query, key, value, g, beta, z = attn.layout_stage(
        conv_out_cf, z_cf, b_cf, a_cf, 1, SEQ_LEN, force_fp16_math=False
    )
    core_out, next_rec_st = attn.core_norm_stage(
        query=query, key=key, value=value, g=g, beta=beta, z=z,
        recurrent_state=recurrent_state.clone(),
        has_previous_state=True, bsz=1, seq_len=SEQ_LEN,
        force_recurrent=True, force_fp16_math=False,
    )

torch_intermediates = {
    "proj_qkv": mixed_qkv_pre.detach().numpy(),
    "proj_z": z_cf.detach().numpy(),
    "proj_b": b_cf.detach().numpy(),
    "proj_a": a_cf.detach().numpy(),
    "conv_out": conv_out_cf.detach().numpy(),
    "next_conv": next_conv_st.detach().numpy(),
    "query": query.detach().numpy(),
    "key": key.detach().numpy(),
    "value": value.detach().numpy(),
    "g": g.detach().numpy(),
    "beta": beta.detach().numpy(),
    "z_layout": z.detach().numpy(),
    "core_out": core_out.detach().numpy(),
    "next_rec": next_rec_st.detach().numpy(),
}
for k, v in torch_intermediates.items():
    np.save(os.path.join(OUT_DIR, f"torch_{k}.npy"), v)
print(f"  Saved {len(torch_intermediates)} intermediate tensors")


# ── 4. Export MONOLITHIC CoreML ─────────────────────────────────
print("\n── Exporting MONOLITHIC CoreML ──")

class MonolithicWrapper(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.attn = attn_module
        self.attn.export_expected_batch_size = 1
        self.attn.export_expected_seq_len = SEQ_LEN

    def forward(self, hidden_states, conv_state, recurrent_state):
        out, next_conv, next_rec = self.attn.forward_prefill_export(
            hidden_states=hidden_states,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            has_previous_state=True,
        )
        return out, next_conv, next_rec

mono_wrapper = MonolithicWrapper(attn).eval()
h_in = torch.zeros(1, SEQ_LEN, cfg.hidden_size, dtype=torch.float16)
c_in = torch.zeros(1, conv_dim, conv_kernel, dtype=torch.float16)
r_in = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=torch.float16)

traced_mono = torch.jit.trace(mono_wrapper, (h_in, c_in, r_in))
mono_pkg = os.path.join(OUT_DIR, "monolithic.mlpackage")
if os.path.exists(mono_pkg):
    shutil.rmtree(mono_pkg)

mlm = ct.convert(
    traced_mono,
    inputs=[
        ct.TensorType(name="hidden_states", shape=h_in.shape, dtype=np.float16),
        ct.TensorType(name="conv_state", shape=c_in.shape, dtype=np.float16),
        ct.TensorType(name="recurrent_state", shape=r_in.shape, dtype=np.float16),
    ],
    outputs=[
        ct.TensorType(name="out", dtype=np.float16),
        ct.TensorType(name="next_conv", dtype=np.float16),
        ct.TensorType(name="next_rec", dtype=np.float16),
    ],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm.save(mono_pkg)
print(f"  Saved → {mono_pkg}")
del mlm, traced_mono
gc.collect()


# ── 5. Export 4-STAGED CoreML ───────────────────────────────────
print("\n── Exporting 4-STAGED CoreML ──")

# Stage 1: Proj
class ProjWrapper(nn.Module):
    def __init__(self, proj_stage):
        super().__init__()
        self.proj_stage = proj_stage
    def forward(self, hidden_states):
        return self.proj_stage(hidden_states)

proj_w = ProjWrapper(attn.proj_stage).eval()
traced_proj = torch.jit.trace(proj_w, (h_in,))
proj_pkg = os.path.join(OUT_DIR, "stage1_proj.mlpackage")
if os.path.exists(proj_pkg):
    shutil.rmtree(proj_pkg)
mlm = ct.convert(
    traced_proj,
    inputs=[ct.TensorType(name="hidden_states", shape=h_in.shape, dtype=np.float16)],
    outputs=[
        ct.TensorType(name="mixed_qkv_pre", dtype=np.float16),
        ct.TensorType(name="z_cf", dtype=np.float16),
        ct.TensorType(name="b_cf", dtype=np.float16),
        ct.TensorType(name="a_cf", dtype=np.float16),
    ],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm.save(proj_pkg)
print(f"  Stage 1 (Proj)    → {proj_pkg}")
del mlm, traced_proj
gc.collect()

# Stage 2: Conv
class ConvWrapper(nn.Module):
    def __init__(self, conv_stage, seq_len):
        super().__init__()
        self.conv_stage = conv_stage
        self.seq_len = seq_len
    def forward(self, mixed_qkv_pre, conv_state):
        return self.conv_stage(mixed_qkv_pre, conv_state, expected_seq_len=self.seq_len)

# Get traced shape of mixed_qkv_pre
with torch.no_grad():
    mqkv, _, _, _ = attn.proj_stage(h_in)
conv_w = ConvWrapper(attn.conv_stage, SEQ_LEN).eval()
traced_conv = torch.jit.trace(conv_w, (mqkv, c_in))
conv_pkg = os.path.join(OUT_DIR, "stage2_conv.mlpackage")
if os.path.exists(conv_pkg):
    shutil.rmtree(conv_pkg)
mlm = ct.convert(
    traced_conv,
    inputs=[
        ct.TensorType(name="mixed_qkv_pre", shape=mqkv.shape, dtype=np.float16),
        ct.TensorType(name="conv_state", shape=c_in.shape, dtype=np.float16),
    ],
    outputs=[
        ct.TensorType(name="conv_out", dtype=np.float16),
        ct.TensorType(name="next_conv", dtype=np.float16),
    ],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm.save(conv_pkg)
print(f"  Stage 2 (Conv)    → {conv_pkg}")
del mlm, traced_conv
gc.collect()

# Stage 3: Layout
class LayoutWrapper(nn.Module):
    def __init__(self, layout_stage, bsz, seq_len):
        super().__init__()
        self.layout_stage = layout_stage
        self.bsz = bsz
        self.seq_len = seq_len
    def forward(self, conv_out_cf, z_cf, b_cf, a_cf):
        return self.layout_stage(conv_out_cf, z_cf, b_cf, a_cf,
                                 self.bsz, self.seq_len, force_fp16_math=False)

with torch.no_grad():
    mqkv_t, z_t, b_t, a_t = attn.proj_stage(h_in)
    conv_out_t, _ = attn.conv_stage(mqkv_t, c_in, expected_seq_len=SEQ_LEN)
layout_w = LayoutWrapper(attn.layout_stage, 1, SEQ_LEN).eval()
traced_layout = torch.jit.trace(layout_w, (conv_out_t, z_t, b_t, a_t))
layout_pkg = os.path.join(OUT_DIR, "stage3_layout.mlpackage")
if os.path.exists(layout_pkg):
    shutil.rmtree(layout_pkg)
mlm = ct.convert(
    traced_layout,
    inputs=[
        ct.TensorType(name="conv_out_cf", shape=conv_out_t.shape, dtype=np.float16),
        ct.TensorType(name="z_cf", shape=z_t.shape, dtype=np.float16),
        ct.TensorType(name="b_cf", shape=b_t.shape, dtype=np.float16),
        ct.TensorType(name="a_cf", shape=a_t.shape, dtype=np.float16),
    ],
    outputs=[
        ct.TensorType(name="query", dtype=np.float16),
        ct.TensorType(name="key", dtype=np.float16),
        ct.TensorType(name="value", dtype=np.float16),
        ct.TensorType(name="g", dtype=np.float16),
        ct.TensorType(name="beta", dtype=np.float16),
        ct.TensorType(name="z_out", dtype=np.float16),
    ],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm.save(layout_pkg)
print(f"  Stage 3 (Layout)  → {layout_pkg}")
del mlm, traced_layout
gc.collect()

# Stage 4: CoreNorm
class CoreNormWrapper(nn.Module):
    def __init__(self, core_norm_stage, bsz, seq_len):
        super().__init__()
        self.core_norm_stage = core_norm_stage
        self.bsz = bsz
        self.seq_len = seq_len
    def forward(self, query, key, value, g, beta, z, recurrent_state):
        return self.core_norm_stage(
            query=query, key=key, value=value, g=g, beta=beta, z=z,
            recurrent_state=recurrent_state,
            has_previous_state=True, bsz=self.bsz, seq_len=self.seq_len,
            force_recurrent=True, force_fp16_math=False,
        )

with torch.no_grad():
    q_t, k_t, v_t, g_t, beta_t, z_out_t = attn.layout_stage(
        conv_out_t, z_t, b_t, a_t, 1, SEQ_LEN, force_fp16_math=False
    )
cn_w = CoreNormWrapper(attn.core_norm_stage, 1, SEQ_LEN).eval()
traced_cn = torch.jit.trace(cn_w, (q_t, k_t, v_t, g_t, beta_t, z_out_t, r_in))
cn_pkg = os.path.join(OUT_DIR, "stage4_corenorm.mlpackage")
if os.path.exists(cn_pkg):
    shutil.rmtree(cn_pkg)
mlm = ct.convert(
    traced_cn,
    inputs=[
        ct.TensorType(name="query", shape=q_t.shape, dtype=np.float16),
        ct.TensorType(name="key", shape=k_t.shape, dtype=np.float16),
        ct.TensorType(name="value", shape=v_t.shape, dtype=np.float16),
        ct.TensorType(name="g", shape=g_t.shape, dtype=np.float16),
        ct.TensorType(name="beta", shape=beta_t.shape, dtype=np.float16),
        ct.TensorType(name="z", shape=z_out_t.shape, dtype=np.float16),
        ct.TensorType(name="recurrent_state", shape=r_in.shape, dtype=np.float16),
    ],
    outputs=[
        ct.TensorType(name="core_out", dtype=np.float16),
        ct.TensorType(name="next_rec", dtype=np.float16),
    ],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm.save(cn_pkg)
print(f"  Stage 4 (CoreNorm) → {cn_pkg}")
del mlm, traced_cn
gc.collect()

# Free model
del model, layer, attn, mono_wrapper, proj_w, conv_w, layout_w, cn_w
gc.collect()


# ── 6. ANE parity: MONOLITHIC ──────────────────────────────────
def cosine_sim(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    d = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return float(np.dot(a_f, b_f) / d) if d > 1e-12 else 0.0

def report(name, ref, cml):
    r = ref.astype(np.float32)
    c = cml.astype(np.float32)
    d = np.abs(r - c)
    cos = cosine_sim(ref, cml)
    print(f"  {name:<20s}  cos={cos:.10f}  max={d.max():.6f}  mean={d.mean():.8f}  "
          f"p99={np.percentile(d, 99):.6f}  p95={np.percentile(d, 95):.6f}")
    return d.max(), d.mean(), cos

print("\n" + "=" * 70)
print("  ANE PARITY: MONOLITHIC")
print("=" * 70)

h_np = np.load(os.path.join(OUT_DIR, "hidden_states.npy"))
c_np = np.load(os.path.join(OUT_DIR, "conv_state.npy"))
r_np = np.load(os.path.join(OUT_DIR, "recurrent_state.npy"))
torch_out_np = np.load(os.path.join(OUT_DIR, "torch_out.npy"))

t0 = time.time()
cml_mono = ct.models.MLModel(mono_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
print(f"  Loaded in {time.time()-t0:.1f}s")
mono_pred = cml_mono.predict({
    "hidden_states": h_np.astype(np.float16),
    "conv_state": c_np.astype(np.float16),
    "recurrent_state": r_np.astype(np.float16),
})
mono_out = mono_pred["out"]
mono_conv = mono_pred["next_conv"]
mono_rec = mono_pred["next_rec"]
print(f"  Output shapes: out={mono_out.shape} next_conv={mono_conv.shape} next_rec={mono_rec.shape}")

mono_mx, mono_mn, mono_cos = report("MONOLITHIC output", torch_out_np, mono_out)
report("MONOLITHIC next_conv", np.load(os.path.join(OUT_DIR, "torch_next_conv.npy")), mono_conv)
report("MONOLITHIC next_rec", np.load(os.path.join(OUT_DIR, "torch_next_rec.npy")), mono_rec)

del cml_mono
gc.collect()

# ── 7. ANE parity: 4-STAGED (cascaded through stages) ─────────
print("\n" + "=" * 70)
print("  ANE PARITY: 4-STAGED (cascaded)")
print("=" * 70)

# Stage 1: Proj
t0 = time.time()
cml_proj = ct.models.MLModel(proj_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
print(f"  Stage 1 loaded in {time.time()-t0:.1f}s")
proj_pred = cml_proj.predict({"hidden_states": h_np.astype(np.float16)})
cml_mqkv = proj_pred["mixed_qkv_pre"]
cml_z = proj_pred["z_cf"]
cml_b = proj_pred["b_cf"]
cml_a = proj_pred["a_cf"]
report("Stage1 mixed_qkv", np.load(os.path.join(OUT_DIR, "torch_proj_qkv.npy")), cml_mqkv)
report("Stage1 z_cf", np.load(os.path.join(OUT_DIR, "torch_proj_z.npy")), cml_z)
report("Stage1 b_cf", np.load(os.path.join(OUT_DIR, "torch_proj_b.npy")), cml_b)
report("Stage1 a_cf", np.load(os.path.join(OUT_DIR, "torch_proj_a.npy")), cml_a)
del cml_proj; gc.collect()

# Stage 2: Conv (use CoreML proj output)
t0 = time.time()
cml_conv = ct.models.MLModel(conv_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
print(f"\n  Stage 2 loaded in {time.time()-t0:.1f}s")
conv_pred = cml_conv.predict({
    "mixed_qkv_pre": cml_mqkv.astype(np.float16),
    "conv_state": c_np.astype(np.float16),
})
cml_conv_out = conv_pred["conv_out"]
cml_next_conv = conv_pred["next_conv"]
report("Stage2 conv_out", np.load(os.path.join(OUT_DIR, "torch_conv_out.npy")), cml_conv_out)
report("Stage2 next_conv", np.load(os.path.join(OUT_DIR, "torch_next_conv.npy")), cml_next_conv)
del cml_conv; gc.collect()

# Stage 3: Layout (use CoreML conv output + CoreML proj z/b/a)
t0 = time.time()
cml_layout = ct.models.MLModel(layout_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
print(f"\n  Stage 3 loaded in {time.time()-t0:.1f}s")
layout_pred = cml_layout.predict({
    "conv_out_cf": cml_conv_out.astype(np.float16),
    "z_cf": cml_z.astype(np.float16),
    "b_cf": cml_b.astype(np.float16),
    "a_cf": cml_a.astype(np.float16),
})
cml_query = layout_pred["query"]
cml_key = layout_pred["key"]
cml_value = layout_pred["value"]
cml_g = layout_pred["g"]
cml_beta = layout_pred["beta"]
cml_z_out = layout_pred["z_out"]
report("Stage3 query", np.load(os.path.join(OUT_DIR, "torch_query.npy")), cml_query)
report("Stage3 key", np.load(os.path.join(OUT_DIR, "torch_key.npy")), cml_key)
report("Stage3 value", np.load(os.path.join(OUT_DIR, "torch_value.npy")), cml_value)
report("Stage3 g", np.load(os.path.join(OUT_DIR, "torch_g.npy")), cml_g)
report("Stage3 beta", np.load(os.path.join(OUT_DIR, "torch_beta.npy")), cml_beta)
report("Stage3 z", np.load(os.path.join(OUT_DIR, "torch_z_layout.npy")), cml_z_out)
del cml_layout; gc.collect()

# Stage 4: CoreNorm (use CoreML layout output)
t0 = time.time()
cml_cn = ct.models.MLModel(cn_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
print(f"\n  Stage 4 loaded in {time.time()-t0:.1f}s")
cn_pred = cml_cn.predict({
    "query": cml_query.astype(np.float16),
    "key": cml_key.astype(np.float16),
    "value": cml_value.astype(np.float16),
    "g": cml_g.astype(np.float16),
    "beta": cml_beta.astype(np.float16),
    "z": cml_z_out.astype(np.float16),
    "recurrent_state": r_np.astype(np.float16),
})
cml_core_out = cn_pred["core_out"]
cml_next_rec = cn_pred["next_rec"]
stg4_mx, stg4_mn, stg4_cos = report("Stage4 core_out", np.load(os.path.join(OUT_DIR, "torch_core_out.npy")), cml_core_out)
report("Stage4 next_rec", np.load(os.path.join(OUT_DIR, "torch_next_rec.npy")), cml_next_rec)
del cml_cn; gc.collect()

# Also compare the final staged output vs the full PyTorch end-to-end output
# (core_out is the attention output before residual, same as torch_out)
stg_e2e_mx, stg_e2e_mn, stg_e2e_cos = report("STAGED end-to-end", torch_out_np, cml_core_out)

# ── Summary ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SUMMARY: Linear Attention Layer {LAYER_IDX} (seq_len={SEQ_LEN})")
print(f"{'='*70}")

def grade(cos, mx):
    if cos > 0.999 and mx < 0.5:
        return "✅ GOOD"
    elif cos > 0.99 and mx < 2.0:
        return "⚠️  ACCEPT"
    else:
        return "❌ BAD"

print(f"  {'Method':<20s} {'Grade':<14s} {'cosine':<15s} {'max_abs':<12s} {'mean_abs':<14s}")
print(f"  {'-'*65}")
print(f"  {'MONOLITHIC':<20s} {grade(mono_cos, mono_mx):<14s} {mono_cos:<15.10f} {mono_mx:<12.6f} {mono_mn:<14.8f}")
print(f"  {'4-STAGED (cascade)':<20s} {grade(stg_e2e_cos, stg_e2e_mx):<14s} {stg_e2e_cos:<15.10f} {stg_e2e_mx:<12.6f} {stg_e2e_mn:<14.8f}")
print(f"{'='*70}")
print(f"Outputs saved to {OUT_DIR}/")
