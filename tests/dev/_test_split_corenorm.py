#!/usr/bin/env python3
"""Test split CoreNorm approaches to fix ANE fusion precision loss.

Tests:
  A) Baseline combined CoreNorm (single layer, expected cos≈0.994)
  B) PassPipeline.EMPTY — no MIL optimizations
  C) Selective pass removal — remove fuse/merge passes keeping noop/cast
  D) State-buffer fusion barrier — write/read core_raw via state
  E) Separate models — recurrence + norm_proj as independent models (proven)

Usage:
  python tests/dev/_test_split_corenorm.py
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
    Qwen35LinearAttention, Qwen35LinearCoreNormStage,
    MODEL_DTYPE, TEST_DEVICE,
)
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_split_corenorm"
SEQ_LEN = 256
CTX = 1024

os.makedirs(OUT_DIR, exist_ok=True)

def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))

def report(ref, cml, label):
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
        worst3 = np.argsort(ch_err)[-3:][::-1]
        for c in worst3:
            print(f"      ch={c:4d} mean_abs={ch_err[c]:.6f}")
    return cos

# ────────────────────────────────────────────────────────────────────
# Load model & prepare inputs
# ────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"  Split CoreNorm Test  (seq={SEQ_LEN}, ctx={CTX})")
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

attn = model.model.layers[0].self_attn  # Layer 0 = linear attention
conv_dim = attn.conv_dim
conv_kernel = attn.linear_conv_kernel_dim
attn.export_expected_batch_size = 1
attn.export_expected_seq_len = SEQ_LEN

# ── Generate PyTorch reference (all stages) ──
with torch.no_grad():
    x_norm = model.model.layers[0].input_layernorm(embed)
    conv_s = torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE)
    rec_s = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=MODEL_DTYPE)
    mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(x_norm)
    conv_out, _ = attn.conv_stage(mixed_qkv_pre, conv_s, expected_seq_len=SEQ_LEN)
    query, key, value, g, beta, z = attn.layout_stage(
        conv_out, z_cf, b_cf, a_cf, 1, SEQ_LEN)

    # Full CoreNorm (combined)
    full_out, _ = attn.core_norm_stage(
        query=query, key=key, value=value, g=g, beta=beta, z=z,
        recurrent_state=rec_s, has_previous_state=True,
        bsz=1, seq_len=SEQ_LEN, force_recurrent=False, force_fp16_math=False)

    # Recurrence only
    core_raw, next_rec = Qwen35LinearAttention._chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=rec_s, output_final_state=True,
        expected_batch_size=1, expected_num_heads=attn.num_v_heads,
        expected_seq_len=SEQ_LEN, expected_k_dim=attn.head_k_dim,
        expected_v_dim=attn.head_v_dim)

    # Norm + projection
    core_for_norm = core_raw.reshape(1, SEQ_LEN, attn.value_dim)
    z_for_norm = z.reshape(1, SEQ_LEN, attn.value_dim)
    core_normed = attn.core_norm_stage.norm(
        core_for_norm.reshape(-1, attn.head_v_dim),
        z_for_norm.reshape(-1, attn.head_v_dim)
    ).reshape(1, SEQ_LEN, attn.value_dim)
    cf = core_normed.transpose(1, 2).unsqueeze(2)
    norm_proj_out = attn.core_norm_stage.out_proj(cf.to(MODEL_DTYPE))
    norm_proj_out = norm_proj_out.squeeze(2).transpose(1, 2)

ref_full = full_out.numpy()
ref_core_raw = core_raw.numpy()
ref_norm_proj = norm_proj_out.numpy()

print(f"  Ref shapes: full={ref_full.shape}, core_raw={ref_core_raw.shape}, norm_proj={ref_norm_proj.shape}")

results = {}

# ────────────────────────────────────────────────────────────────────
# TEST A: Baseline Combined CoreNorm (default pipeline)
# ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  TEST A: Baseline Combined CoreNorm (default pipeline)")
print(f"{'='*70}")

class CombinedCoreNormWrapper(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.core_norm_stage = attn_module.core_norm_stage
        self.register_buffer("rec_state", torch.zeros(
            1, attn_module.num_v_heads, attn_module.head_k_dim, attn_module.head_v_dim, dtype=MODEL_DTYPE))

    def forward(self, query, key, value, g, beta, z):
        out, _ = self.core_norm_stage(
            query=query, key=key, value=value, g=g, beta=beta, z=z,
            recurrent_state=self.rec_state, has_previous_state=True,
            bsz=1, seq_len=SEQ_LEN, force_recurrent=False, force_fp16_math=False)
        return out

def export_and_test(wrapper, inputs, input_specs, ref, label, states=None,
                    pass_pipeline=None, output_name="out"):
    pkg = os.path.join(OUT_DIR, f"{label}.mlpackage")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)

    traced = torch.jit.trace(wrapper, inputs)
    # Zero states after trace
    for name, buf in traced.named_buffers():
        if "state" in name or "barrier" in name:
            buf.zero_()

    convert_kwargs = dict(
        inputs=input_specs,
        outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    if states:
        convert_kwargs["states"] = states
    if pass_pipeline is not None:
        convert_kwargs["pass_pipeline"] = pass_pipeline

    print(f"  Converting{' (custom pipeline)' if pass_pipeline else ''}...")
    try:
        mlm = ct.convert(traced, **convert_kwargs)
    except Exception as e:
        print(f"  CONVERT FAILED: {e}")
        return None
    mlm.save(pkg)
    del mlm, traced; gc.collect()

    print(f"  Running on ANE...")
    try:
        cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
        if states:
            state = cml.make_state()
            out = cml.predict(
                {spec.name: inp.numpy().astype(np.float16) if isinstance(inp, torch.Tensor) else inp
                 for spec, inp in zip(input_specs, inputs)},
                state=state)
        else:
            out = cml.predict(
                {spec.name: inp.numpy().astype(np.float16) if isinstance(inp, torch.Tensor) else inp
                 for spec, inp in zip(input_specs, inputs)})
        cml_out = list(out.values())[0]
        cos = report(ref, cml_out, label)
        del cml; gc.collect()
        return cos
    except Exception as e:
        print(f"  ANE PREDICT FAILED: {e}")
        del cml; gc.collect()
        return None

# Test A
w_a = CombinedCoreNormWrapper(attn).eval()
inp_a = (query, key, value, g, beta, z)
specs_a = [
    ct.TensorType(name="query", shape=query.shape, dtype=np.float16),
    ct.TensorType(name="key", shape=key.shape, dtype=np.float16),
    ct.TensorType(name="value", shape=value.shape, dtype=np.float16),
    ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
    ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
    ct.TensorType(name="z", shape=z.shape, dtype=np.float16),
]
states_a = [ct.StateType(
    wrapped_type=ct.TensorType(
        shape=(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim), dtype=np.float16),
    name="rec_state")]

cos_a = export_and_test(w_a, inp_a, specs_a, ref_full, "A_baseline", states=states_a)
results["A_baseline"] = cos_a
del w_a; gc.collect()

# ────────────────────────────────────────────────────────────────────
# TEST B: PassPipeline.EMPTY (no MIL optimizations)
# ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  TEST B: PassPipeline.EMPTY (no MIL optimizations)")
print(f"{'='*70}")

w_b = CombinedCoreNormWrapper(attn).eval()
cos_b = export_and_test(w_b, inp_a, specs_a, ref_full, "B_empty_pipeline",
                        states=states_a, pass_pipeline=ct.PassPipeline.EMPTY)
results["B_empty_pipeline"] = cos_b
del w_b; gc.collect()

# ────────────────────────────────────────────────────────────────────
# TEST C: Selective pass removal (remove fuse/merge, keep essentials)
# ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  TEST C: Selective pass removal (remove fuse/merge passes)")
print(f"{'='*70}")

pipeline_c = ct.PassPipeline()
fuse_merge_passes = [p for p in pipeline_c.passes
                     if 'fuse' in p or 'merge' in p]
print(f"  Removing {len(fuse_merge_passes)} fuse/merge passes")
for p in fuse_merge_passes:
    try:
        pipeline_c.remove_pass(p)
    except Exception:
        pass

w_c = CombinedCoreNormWrapper(attn).eval()
cos_c = export_and_test(w_c, inp_a, specs_a, ref_full, "C_no_fuse_merge",
                        states=states_a, pass_pipeline=pipeline_c)
results["C_no_fuse_merge"] = cos_c
del w_c; gc.collect()

# ────────────────────────────────────────────────────────────────────
# TEST D: State-buffer fusion barrier
# ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  TEST D: State-buffer fusion barrier")
print(f"{'='*70}")

class StateBarrierCoreNormWrapper(nn.Module):
    """CoreNorm with a state buffer between recurrence and norm+proj.
    The state write/read forces MIL to materialize the tensor."""
    def __init__(self, attn_module):
        super().__init__()
        self.core_norm_stage = attn_module.core_norm_stage
        self.num_v_heads = attn_module.num_v_heads
        self.head_k_dim = attn_module.head_k_dim
        self.head_v_dim = attn_module.head_v_dim
        self.value_dim = attn_module.value_dim
        self.register_buffer("rec_state", torch.zeros(
            1, attn_module.num_v_heads, attn_module.head_k_dim, attn_module.head_v_dim, dtype=MODEL_DTYPE))
        # Barrier state: stores recurrence output to break fusion
        # Shape: (1, SEQ_LEN, value_dim) but need ANE-safe dims
        # value_dim=4096 > 1024, so reshape to (SEQ_LEN, value_dim) as (256, 4096)
        # Actually state shape: (num_v_heads, SEQ_LEN, head_v_dim) = (32, 256, 128)
        self.register_buffer("core_barrier", torch.zeros(
            attn_module.num_v_heads, SEQ_LEN, attn_module.head_v_dim, dtype=MODEL_DTYPE))

    def forward(self, query, key, value, g, beta, z):
        # Recurrence
        core, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=self.rec_state, output_final_state=True,
            expected_batch_size=1, expected_num_heads=self.num_v_heads,
            expected_seq_len=SEQ_LEN, expected_k_dim=self.head_k_dim,
            expected_v_dim=self.head_v_dim)
        # core shape: (1, SEQ_LEN, num_v_heads, head_v_dim) or (1, SEQ_LEN, value_dim)
        # Write to state barrier (reshape to match barrier shape)
        core_reshaped = core.reshape(self.num_v_heads, SEQ_LEN, self.head_v_dim)
        self.core_barrier[:] = core_reshaped
        # Read back from state
        core_from_state = self.core_barrier.reshape(1, SEQ_LEN, self.value_dim)

        # Norm + projection
        core_normed = self.core_norm_stage.norm(
            core_from_state.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim)
        ).reshape(1, SEQ_LEN, self.value_dim)
        cf = core_normed.transpose(1, 2).unsqueeze(2)
        out = self.core_norm_stage.out_proj(cf.to(MODEL_DTYPE))
        return out.squeeze(2).transpose(1, 2)

w_d = StateBarrierCoreNormWrapper(attn).eval()
states_d = [
    ct.StateType(
        wrapped_type=ct.TensorType(
            shape=(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim), dtype=np.float16),
        name="rec_state"),
    ct.StateType(
        wrapped_type=ct.TensorType(
            shape=(attn.num_v_heads, SEQ_LEN, attn.head_v_dim), dtype=np.float16),
        name="core_barrier"),
]
cos_d = export_and_test(w_d, inp_a, specs_a, ref_full, "D_state_barrier",
                        states=states_d)
results["D_state_barrier"] = cos_d
del w_d; gc.collect()

# ────────────────────────────────────────────────────────────────────
# TEST E: Separate models (recurrence + norm_proj)
# ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  TEST E: Separate models (proven approach)")
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

class NormProjWrapper(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.norm = attn_module.core_norm_stage.norm
        self.out_proj = attn_module.core_norm_stage.out_proj
        self.head_v_dim = attn_module.head_v_dim
        self.value_dim = attn_module.value_dim

    def forward(self, core_raw, z):
        core_normed = self.norm(
            core_raw.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim)
        ).reshape(1, SEQ_LEN, self.value_dim)
        cf = core_normed.transpose(1, 2).unsqueeze(2)
        out = self.out_proj(cf.to(MODEL_DTYPE))
        return out.squeeze(2).transpose(1, 2)

# E1: Recurrence model
print("  E1: Exporting recurrence model...")
w_e1 = RecurrenceOnlyWrapper(attn).eval()
traced_e1 = torch.jit.trace(w_e1, (query, key, value, g, beta))
for _, buf in traced_e1.named_buffers():
    buf.zero_()
pkg_e1 = os.path.join(OUT_DIR, "E_recurrence.mlpackage")
if os.path.exists(pkg_e1): shutil.rmtree(pkg_e1)
mlm_e1 = ct.convert(
    traced_e1,
    inputs=[
        ct.TensorType(name="query", shape=query.shape, dtype=np.float16),
        ct.TensorType(name="key", shape=key.shape, dtype=np.float16),
        ct.TensorType(name="value", shape=value.shape, dtype=np.float16),
        ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
        ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
    ],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    states=[ct.StateType(
        wrapped_type=ct.TensorType(
            shape=(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim), dtype=np.float16),
        name="rec_state")],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_e1.save(pkg_e1)
del mlm_e1, traced_e1, w_e1; gc.collect()

# E2: Norm+proj model
print("  E2: Exporting norm+proj model...")
core_for_norm = core_raw.reshape(1, SEQ_LEN, attn.value_dim).contiguous()
z_for_norm = z.reshape(1, SEQ_LEN, attn.value_dim).contiguous()
w_e2 = NormProjWrapper(attn).eval()
traced_e2 = torch.jit.trace(w_e2, (core_for_norm, z_for_norm))
pkg_e2 = os.path.join(OUT_DIR, "E_norm_proj.mlpackage")
if os.path.exists(pkg_e2): shutil.rmtree(pkg_e2)
mlm_e2 = ct.convert(
    traced_e2,
    inputs=[
        ct.TensorType(name="core_raw", shape=core_for_norm.shape, dtype=np.float16),
        ct.TensorType(name="z", shape=z_for_norm.shape, dtype=np.float16),
    ],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_e2.save(pkg_e2)
del mlm_e2, traced_e2, w_e2; gc.collect()

# Run E cascaded
print("  Running E cascaded on ANE...")
cml_e1 = ct.models.MLModel(pkg_e1, compute_units=ct.ComputeUnit.CPU_AND_NE)
state_e1 = cml_e1.make_state()
out_e1 = cml_e1.predict({
    "query": query.numpy().astype(np.float16),
    "key": key.numpy().astype(np.float16),
    "value": value.numpy().astype(np.float16),
    "g": g.numpy().astype(np.float16),
    "beta": beta.numpy().astype(np.float16),
}, state=state_e1)
cml_core_raw = list(out_e1.values())[0]
cos_e1 = report(ref_core_raw, cml_core_raw, "E1: Recurrence only")
del cml_e1, state_e1; gc.collect()

# Feed ANE recurrence output into ANE norm+proj
cml_core_for_norm = cml_core_raw.reshape(1, SEQ_LEN, attn.value_dim).astype(np.float16)
z_np = z_for_norm.numpy().astype(np.float16)
cml_e2 = ct.models.MLModel(pkg_e2, compute_units=ct.ComputeUnit.CPU_AND_NE)
out_e2 = cml_e2.predict({
    "core_raw": cml_core_for_norm,
    "z": z_np,
})
cml_norm_proj = list(out_e2.values())[0]
cos_e2_isolated = report(ref_norm_proj, cml_norm_proj, "E2: Norm+proj (ANE recurrence input)")
cos_e_full = report(ref_full, cml_norm_proj, "E:  Full split (recurrence→norm_proj cascaded)")
results["E_full_split"] = cos_e_full
del cml_e2; gc.collect()

# ────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  {'Test':<50} {'cosine':<15}")
print(f"  {'-'*65}")
for name, cos in results.items():
    val = f"{cos:.10f}" if cos is not None else "FAILED"
    print(f"  {name:<50} {val:<15}")
print(f"{'='*70}")
