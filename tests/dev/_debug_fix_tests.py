#!/usr/bin/env python3
"""Test two potential fixes for linear attention ANE parity:
  1. Reduced chunk_size (64 → 32, 16) to reduce op-fusion scope
  2. Staged export (4 separate CoreML models per linear attention layer)

Uses layer 0 with seq=256, ctx=1024.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, shutil, time, copy
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    Qwen35LinearAttention,
    MODEL_DTYPE, TEST_DEVICE,
)

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_fix_tests"
SEQ_LEN = 256
CTX = 1024
LAYER_IDX = 0

os.makedirs(OUT_DIR, exist_ok=True)


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def report(ref, cml, label, top_n=3):
    rf = ref.astype(np.float32)
    cf = cml.astype(np.float32)
    diff = np.abs(rf - cf)
    cos = cosine(ref, cml)
    print(f"  [{label}] cos={cos:.10f}  max_abs={diff.max():.6f}  mean_abs={diff.mean():.8f}")
    if len(diff.shape) >= 2:
        last_dim = diff.shape[-1]
        flat_diff = diff.reshape(-1, last_dim)
        ch_err = flat_diff.mean(axis=0)
        worst = np.argsort(ch_err)[-top_n:][::-1]
        for c in worst:
            print(f"      ch={c:4d} mean_abs={ch_err[c]:.6f}")
    return cos


# ── 1. Load model ──
print("=" * 70)
print(f"  Fix Testing: chunk_size + staged export  (seq={SEQ_LEN})")
print("=" * 70)
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Realistic input
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
prompt = "Explain stack and heap memory in one paragraph and give one debugging tip."
text = prompt
while True:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    if ids.shape[1] >= SEQ_LEN:
        ids = ids[:, :SEQ_LEN]
        break
    text = text + " " + prompt
ids = ids.to(torch.int32)
with torch.no_grad():
    embed = model.model.embed_tokens(ids).to(torch.float16)

layer = model.model.layers[LAYER_IDX]
attn = layer.self_attn
conv_dim = attn.conv_dim
conv_kernel = attn.linear_conv_kernel_dim
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)

# ── 2. PyTorch reference (fp32 math, chunk_size=64, default) ──
print("\n── PyTorch reference (fp32 math, chunk_size=64) ──")
attn.export_expected_batch_size = 1
attn.export_expected_seq_len = SEQ_LEN
with torch.no_grad():
    x_norm = layer.input_layernorm(embed)
    conv_st = torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE)
    rec_st = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=MODEL_DTYPE)
    torch_attn_out, _, _ = attn.forward_prefill_export(
        hidden_states=x_norm, conv_state=conv_st.clone(), recurrent_state=rec_st.clone(),
        has_previous_state=True)
torch_ref = torch_attn_out.detach().numpy()
print(f"  shape={torch_ref.shape}  mean={torch_attn_out.float().mean():.6f} std={torch_attn_out.float().std():.6f}")


# ───────────────────────────────────────────────────────────────
# TEST 1: Monolithic export with different chunk_sizes
# ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  TEST 1: Monolithic export with varying chunk_size")
print(f"{'='*70}")

class AttnOnlyWrapper(nn.Module):
    def __init__(self, mdl, layer_idx, seq_len):
        super().__init__()
        self.norm = mdl.model.layers[layer_idx].input_layernorm
        self.attn = mdl.model.layers[layer_idx].self_attn
        a1, a2 = ane_conv_state_shape(conv_dim, conv_kernel)
        self.register_buffer("conv_state", torch.zeros(1, a1, a2, dtype=MODEL_DTYPE))
        self.register_buffer("rec_state", torch.zeros(
            1, cfg.text_config.linear_num_value_heads, cfg.text_config.linear_key_head_dim,
            cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE))
        self.attn.export_expected_batch_size = 1
        self.attn.export_expected_seq_len = seq_len

    def forward(self, hidden_states):
        x = self.norm(hidden_states)
        cs = self.conv_state.reshape(1, conv_dim, conv_kernel)
        out, nc, nr = self.attn.forward_prefill_export(
            hidden_states=x, conv_state=cs, recurrent_state=self.rec_state,
            has_previous_state=True)
        return out

attn_states = [
    ct.StateType(wrapped_type=ct.TensorType(shape=(1, ane_d1, ane_d2), dtype=np.float16), name="conv_state"),
    ct.StateType(wrapped_type=ct.TensorType(
        shape=(1, cfg.text_config.linear_num_value_heads,
               cfg.text_config.linear_key_head_dim,
               cfg.text_config.linear_value_head_dim), dtype=np.float16),
        name="rec_state"),
]

mono_results = {}
for cs in [64, 32, 16]:
    print(f"\n  chunk_size={cs}:")

    # Generate PyTorch reference with this chunk_size
    # (need to match for fair comparison)
    orig_chunk = Qwen35LinearAttention._chunk_gated_delta_rule.__code__
    # Monkey-patch chunk_size default — can't easily do this.
    # Instead, generate torch reference for this chunk_size directly.
    with torch.no_grad():
        conv_s = torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE)
        rec_s = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=MODEL_DTYPE)

        # Run proj + conv + layout stages
        mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(x_norm)
        conv_out_cf, _ = attn.conv_stage(mixed_qkv_pre, conv_s, expected_seq_len=SEQ_LEN)
        query, key, value, g, beta, z = attn.layout_stage(
            conv_out_cf, z_cf, b_cf, a_cf, 1, SEQ_LEN, force_fp16_math=False)

        # Run core with specified chunk_size
        core_out, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            chunk_size=cs,
            initial_state=rec_s,
            output_final_state=True,
            expected_batch_size=1,
            expected_num_heads=attn.num_v_heads,
            expected_seq_len=SEQ_LEN,
            expected_k_dim=attn.head_k_dim,
            expected_v_dim=attn.head_v_dim,
        )
        # Norm + output proj
        core_normed = attn.core_norm_stage.norm(
            core_out.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim)
        ).reshape(1, SEQ_LEN, attn.value_dim)
        torch_cs_out = attn.core_norm_stage.from_channels_first_4d(
            attn.core_norm_stage.conv2d_proj_cf(attn.core_norm_stage.out_proj,
                                                 attn.core_norm_stage.to_channels_first_4d(core_normed))
        )
    torch_cs_ref = torch_cs_out.detach().numpy()

    # Check if PyTorch reference changes with chunk_size
    cos_vs_default = cosine(torch_ref, torch_cs_ref)
    print(f"    PyTorch cs={cs} vs cs=64: cos={cos_vs_default:.10f}")

    # Now export CoreML with this chunk_size by temporarily patching the default
    import types

    original_method = Qwen35LinearAttention._chunk_gated_delta_rule

    @staticmethod
    def patched_chunk_gated_delta_rule(
        query, key, value, g, beta,
        chunk_size=cs,  # Use our chunk_size
        initial_state=None, output_final_state=True,
        expected_batch_size=None, expected_num_heads=None,
        expected_seq_len=None, expected_k_dim=None, expected_v_dim=None,
        math_dtype=torch.float32,
    ):
        return original_method(
            query, key, value, g, beta,
            chunk_size=cs,
            initial_state=initial_state, output_final_state=output_final_state,
            expected_batch_size=expected_batch_size, expected_num_heads=expected_num_heads,
            expected_seq_len=expected_seq_len, expected_k_dim=expected_k_dim,
            expected_v_dim=expected_v_dim, math_dtype=math_dtype,
        )

    Qwen35LinearAttention._chunk_gated_delta_rule = patched_chunk_gated_delta_rule

    wrapper = AttnOnlyWrapper(model, LAYER_IDX, SEQ_LEN).eval()
    h_in = torch.zeros((1, SEQ_LEN, cfg.hidden_size), dtype=torch.float16)

    traced = torch.jit.trace(wrapper, (h_in,))
    for name, buf in traced.named_buffers():
        if any(k in name for k in ("conv_state", "rec_state")):
            buf.zero_()

    pkg = os.path.join(OUT_DIR, f"layer0_cs{cs}.mlpackage")
    if os.path.exists(pkg): shutil.rmtree(pkg)
    try:
        mlm = ct.convert(
            traced,
            inputs=[ct.TensorType(name="hidden_states", shape=h_in.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="out", dtype=np.float16)],
            states=attn_states,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
        )
        mlm.save(pkg)
        del mlm, traced, wrapper; gc.collect()

        # Run on ANE
        cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = cml.make_state()
        out = cml.predict({"hidden_states": embed.numpy().astype(np.float16)}, state=state)
        cml_out = list(out.values())[0]
        cos_val = report(torch_cs_ref, cml_out, f"chunk_size={cs}")
        mono_results[cs] = cos_val
        del cml, state; gc.collect()
    except Exception as e:
        print(f"    FAILED: {e}")
        mono_results[cs] = None
    finally:
        Qwen35LinearAttention._chunk_gated_delta_rule = original_method


# ───────────────────────────────────────────────────────────────
# TEST 2: 4-Stage export (seq=256) — each stage as separate CoreML model
# ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  TEST 2: 4-Stage export (seq=256)")
print(f"{'='*70}")

# PyTorch per-stage references
with torch.no_grad():
    conv_s = torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE)
    rec_s = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=MODEL_DTYPE)

    mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(x_norm)
    conv_out_cf, next_conv = attn.conv_stage(mixed_qkv_pre, conv_s, expected_seq_len=SEQ_LEN)
    query, key, value, g, beta, z = attn.layout_stage(
        conv_out_cf, z_cf, b_cf, a_cf, 1, SEQ_LEN, force_fp16_math=False)
    core_out, next_rec = attn.core_norm_stage(
        query=query, key=key, value=value, g=g, beta=beta, z=z,
        recurrent_state=rec_s, has_previous_state=True, bsz=1, seq_len=SEQ_LEN,
        force_recurrent=False, force_fp16_math=False)

torch_stage_refs = {
    "proj": (mixed_qkv_pre.numpy(), z_cf.numpy(), b_cf.numpy(), a_cf.numpy()),
    "conv": conv_out_cf.numpy(),
    "layout": (query.numpy(), key.numpy(), value.numpy(), g.numpy(), beta.numpy(), z.numpy()),
    "core": core_out.numpy(),
}

# Stage 1: ProjStage — test with multiple outputs
class ProjWrapper(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.stage = attn_module.proj_stage
        self.norm = model.model.layers[LAYER_IDX].input_layernorm
    def forward(self, hidden_states):
        x = self.norm(hidden_states)
        qkv, z, b, a = self.stage(x)
        # Return only qkv (largest tensor) for simple comparison
        return qkv

print("\n  Stage 1 (Proj — qkv only):")
pw = ProjWrapper(attn).eval()
traced_p = torch.jit.trace(pw, (torch.zeros((1, SEQ_LEN, cfg.hidden_size), dtype=torch.float16),))
pkg_p = os.path.join(OUT_DIR, "stage1_proj.mlpackage")
if os.path.exists(pkg_p): shutil.rmtree(pkg_p)
mlm_p = ct.convert(
    traced_p,
    inputs=[ct.TensorType(name="x", shape=(1, SEQ_LEN, cfg.hidden_size), dtype=np.float16)],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_p.save(pkg_p)
del mlm_p, traced_p, pw; gc.collect()

cml_p = ct.models.MLModel(pkg_p, compute_units=ct.ComputeUnit.CPU_AND_NE)
out_p = cml_p.predict({"x": embed.numpy().astype(np.float16)})
cml_proj_out = list(out_p.values())[0]
# Compare qkv output only
torch_proj_qkv = mixed_qkv_pre.numpy()
cos_proj = report(torch_proj_qkv, cml_proj_out, "Stage 1: Proj + RMSNorm (qkv)")
del cml_p; gc.collect()


# For Stage 4 (CoreNorm) — the bottleneck, test directly
class CoreNormWrapper(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.stage = attn_module.core_norm_stage
        self.register_buffer("rec_state", torch.zeros(
            1, cfg.text_config.linear_num_value_heads,
            cfg.text_config.linear_key_head_dim,
            cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE))
        self.num_v_heads = attn_module.num_v_heads
        self.head_k_dim = attn_module.head_k_dim
        self.head_v_dim = attn_module.head_v_dim

    def forward(self, query, key, value, g, beta, z):
        out, _ = self.stage(
            query=query, key=key, value=value, g=g, beta=beta, z=z,
            recurrent_state=self.rec_state, has_previous_state=True,
            bsz=1, seq_len=SEQ_LEN, force_recurrent=False, force_fp16_math=False)
        return out

print("\n  Stage 4 (CoreNorm):")
cnw = CoreNormWrapper(attn).eval()
traced_cn = torch.jit.trace(cnw, (query, key, value, g, beta, z))
for name, buf in traced_cn.named_buffers():
    if "rec_state" in name:
        buf.zero_()

core_states = [
    ct.StateType(wrapped_type=ct.TensorType(
        shape=(1, cfg.text_config.linear_num_value_heads,
               cfg.text_config.linear_key_head_dim,
               cfg.text_config.linear_value_head_dim), dtype=np.float16),
        name="rec_state"),
]
pkg_cn = os.path.join(OUT_DIR, "stage4_corenorm.mlpackage")
if os.path.exists(pkg_cn): shutil.rmtree(pkg_cn)
mlm_cn = ct.convert(
    traced_cn,
    inputs=[
        ct.TensorType(name="query", shape=query.shape, dtype=np.float16),
        ct.TensorType(name="key", shape=key.shape, dtype=np.float16),
        ct.TensorType(name="value", shape=value.shape, dtype=np.float16),
        ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
        ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
        ct.TensorType(name="z", shape=z.shape, dtype=np.float16),
    ],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    states=core_states,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_cn.save(pkg_cn)
del mlm_cn, traced_cn, cnw; gc.collect()

cml_cn = ct.models.MLModel(pkg_cn, compute_units=ct.ComputeUnit.CPU_AND_NE)
state_cn = cml_cn.make_state()
out_cn = cml_cn.predict({
    "query": query.numpy().astype(np.float16),
    "key": key.numpy().astype(np.float16),
    "value": value.numpy().astype(np.float16),
    "g": g.numpy().astype(np.float16),
    "beta": beta.numpy().astype(np.float16),
    "z": z.numpy().astype(np.float16),
}, state=state_cn)
cml_core = list(out_cn.values())[0]
cos_core = report(torch_stage_refs["core"], cml_core, "Stage 4: CoreNorm (seq=256)")
del cml_cn, state_cn; gc.collect()


# ── Summary ──
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  {'Test':<50} {'cosine':<15}")
print(f"  {'-'*65}")
for cs, cos in mono_results.items():
    label = f"Monolithic chunk_size={cs}"
    if cos is not None:
        print(f"  {label:<50} {cos:<15.10f}")
    else:
        print(f"  {label:<50} FAILED")
print(f"  {'Stage 1: Proj (seq=256)':<50} {cos_proj:<15.10f}")
print(f"  {'Stage 4: CoreNorm (seq=256)':<50} {cos_core:<15.10f}")
print(f"{'='*70}")
