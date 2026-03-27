#!/usr/bin/env python3
"""Mitigation experiment: reduce ANE linear-attention amplification.

Exports layer-0 CoreNorm with five strategies and measures cosine recovery
against the PyTorch fp16 reference.

Strategies:
  A) Baseline (current code, doubled-LayerNorm RMSNormGated)
  B) Pre-norm scaling: scale recurrence output by 1/S before norm, scale back after
  C) Direct RMSNorm: replace doubled-LayerNorm trick with manual mean-of-squares RMSNorm
  D) FP32 compute precision: export with ct.precision.FLOAT32
  E) FP32 recurrence math only: keep the gated-delta-rule kernel in fp32 at trace time

Usage:
    python tests/dev/debug_mitigation_amplification.py
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from anemll.models.qwen3_5_model import (
    MODEL_DTYPE,
    TEST_DEVICE,
    Qwen35Config,
    Qwen35ForCausalLM,
    Qwen35LinearAttention,
    Qwen35LinearCoreNormStage,
    Qwen35RMSNormGated,
    _l2norm,
    ane_conv_state_shape,
)
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_mitigation_exp"
SEQ_LEN = 256
CTX = 1024
REPORT_PATH = "tests/dev/mitigation_amplification_report.json"

os.makedirs(OUT_DIR, exist_ok=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def metrics(ref: np.ndarray, got: np.ndarray) -> Dict[str, float]:
    diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
    return {
        "cosine": cosine(ref, got),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "ch0_mean_abs": float(diff.reshape(-1, diff.shape[-1])[:, 0].mean()) if diff.ndim >= 2 else 0.0,
    }


# ────────────────────────────────────────────────────────────────────
# Load model & prepare inputs
# ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Mitigation Experiment: Linear-Attention Amplification")
print("=" * 70)

cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
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
        ids = ids[:, :SEQ_LEN]
        break
    text = text + " " + prompt
ids = ids.to(torch.int32)

with torch.no_grad():
    embed = model.model.embed_tokens(ids).to(torch.float16)

attn = model.model.layers[0].self_attn
conv_dim_val = attn.conv_dim
conv_kernel_val = attn.linear_conv_kernel_dim
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim_val, conv_kernel_val)
attn.export_expected_batch_size = 1
attn.export_expected_seq_len = SEQ_LEN

# Generate intermediates
with torch.no_grad():
    x_norm = model.model.layers[0].input_layernorm(embed)
    conv_s = torch.zeros(1, conv_dim_val, conv_kernel_val, dtype=MODEL_DTYPE)
    rec_s = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=MODEL_DTYPE)
    mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(x_norm)
    conv_out, _ = attn.conv_stage(mixed_qkv_pre, conv_s, expected_seq_len=SEQ_LEN)
    query, key, value, g, beta, z = attn.layout_stage(conv_out, z_cf, b_cf, a_cf, 1, SEQ_LEN)

    # PyTorch reference: full CoreNorm
    full_ref, _ = attn.core_norm_stage(
        query=query, key=key, value=value, g=g, beta=beta, z=z,
        recurrent_state=rec_s, has_previous_state=True,
        bsz=1, seq_len=SEQ_LEN, force_recurrent=False, force_fp16_math=False,
    )

    # Also get recurrence-only reference for direct comparison
    core_raw, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=rec_s,
        output_final_state=True, expected_batch_size=1,
        expected_num_heads=attn.num_v_heads, expected_seq_len=SEQ_LEN,
        expected_k_dim=attn.head_k_dim, expected_v_dim=attn.head_v_dim,
    )

ref_full = full_ref.numpy()
print(f"  PyTorch reference shape: {ref_full.shape}")


# ────────────────────────────────────────────────────────────────────
# Helper: export CoreNorm wrapper, run on ANE, return metrics
# ────────────────────────────────────────────────────────────────────
def export_and_test(
    wrapper: nn.Module,
    label: str,
    ref: np.ndarray,
    compute_precision=ct.precision.FLOAT16,
) -> Dict[str, object]:
    pkg = os.path.join(OUT_DIR, f"{label}.mlpackage")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)

    # Trace
    traced = torch.jit.trace(wrapper, (query, key, value, g, beta, z))
    for name, buf in traced.named_buffers():
        if "rec_state" in name:
            buf.zero_()

    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim),
                dtype=np.float16,
            ),
            name="rec_state",
        )
    ]

    print(f"\n  [{label}] Converting (precision={compute_precision})...")
    t0 = time.time()
    mlm = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="query", shape=query.shape, dtype=np.float16),
            ct.TensorType(name="key", shape=key.shape, dtype=np.float16),
            ct.TensorType(name="value", shape=value.shape, dtype=np.float16),
            ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
            ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
            ct.TensorType(name="z", shape=z.shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="out", dtype=np.float16)],
        states=states,
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    mlm.save(pkg)
    cvt_time = time.time() - t0
    del mlm, traced
    gc.collect()

    print(f"  [{label}] Running on ANE...")
    try:
        cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = cml.make_state()
        out = cml.predict(
            {
                "query": query.numpy().astype(np.float16),
                "key": key.numpy().astype(np.float16),
                "value": value.numpy().astype(np.float16),
                "g": g.numpy().astype(np.float16),
                "beta": beta.numpy().astype(np.float16),
                "z": z.numpy().astype(np.float16),
            },
            state=state,
        )
        cml_out = list(out.values())[0]
        m = metrics(ref, cml_out)
        print(f"  [{label}] cosine={m['cosine']:.10f}  max_abs={m['max_abs']:.6f}  "
              f"ch0_mean={m['ch0_mean_abs']:.6f}")
        del cml, state
        gc.collect()
        return {"label": label, "status": "ok", "cvt_sec": cvt_time, **m}
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        return {"label": label, "status": "failed", "error": str(e)[:200]}


# ════════════════════════════════════════════════════════════════════
# STRATEGY A: Baseline (current code)
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  A) Baseline (current code)")
print(f"{'='*70}")


class BaselineWrapper(nn.Module):
    def __init__(self, stage: Qwen35LinearCoreNormStage):
        super().__init__()
        self.stage = stage
        self.register_buffer(
            "rec_state",
            torch.zeros(1, stage.num_v_heads, stage.head_k_dim, stage.head_v_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, query, key, value, g, beta, z):
        out, _ = self.stage(
            query=query, key=key, value=value, g=g, beta=beta, z=z,
            recurrent_state=self.rec_state, has_previous_state=True,
            bsz=1, seq_len=SEQ_LEN, force_recurrent=False, force_fp16_math=False,
        )
        return out


result_a = export_and_test(BaselineWrapper(attn.core_norm_stage).eval(), "A_baseline", ref_full)


# ════════════════════════════════════════════════════════════════════
# STRATEGY B: Pre-norm scaling (scale down before norm, scale up after)
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  B) Pre-norm scaling (S=8)")
print(f"{'='*70}")

SCALE_FACTOR = 8.0


class ScaledCoreNormWrapper(nn.Module):
    """Scale recurrence output by 1/S before norm, then scale the final output by S."""

    def __init__(self, stage: Qwen35LinearCoreNormStage, scale: float):
        super().__init__()
        self.num_v_heads = stage.num_v_heads
        self.head_k_dim = stage.head_k_dim
        self.head_v_dim = stage.head_v_dim
        self.value_dim = stage.value_dim
        self.norm = stage.norm
        self.out_proj = stage.out_proj
        self.scale = scale
        self.register_buffer(
            "rec_state",
            torch.zeros(1, stage.num_v_heads, stage.head_k_dim, stage.head_v_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, query, key, value, g, beta, z):
        core, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=self.rec_state, output_final_state=True,
            expected_batch_size=1, expected_num_heads=self.num_v_heads,
            expected_seq_len=SEQ_LEN, expected_k_dim=self.head_k_dim,
            expected_v_dim=self.head_v_dim,
        )
        # Scale down to reduce perturbation magnitude through norm
        core_scaled = core * (1.0 / self.scale)
        z_scaled = z * (1.0 / self.scale)
        core_normed = self.norm(
            core_scaled.reshape(-1, self.head_v_dim),
            z_scaled.reshape(-1, self.head_v_dim),
        ).reshape(1, SEQ_LEN, self.value_dim)
        cf = core_normed.transpose(1, 2).unsqueeze(2)
        out = self.out_proj(cf.to(MODEL_DTYPE)).squeeze(2).transpose(1, 2)
        # Scale back to recover original magnitude
        return out * self.scale


# PyTorch reference for strategy B (should match original due to RMSNorm scale-invariance,
# but the gate path may differ slightly)
with torch.no_grad():
    b_wrapper = ScaledCoreNormWrapper(attn.core_norm_stage, SCALE_FACTOR).eval()
    ref_b = b_wrapper(query, key, value, g, beta, z).numpy()
# Compare ref_b to ref_full to see if PyTorch equivalence holds
cos_b_vs_orig = cosine(ref_full, ref_b)
print(f"  PyTorch scaled ref vs original: cosine={cos_b_vs_orig:.10f}")

result_b = export_and_test(b_wrapper, "B_prescale_S8", ref_b)


# ════════════════════════════════════════════════════════════════════
# STRATEGY C: Direct RMSNorm (replace doubled-LayerNorm with manual)
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  C) Direct RMSNorm (no doubled-LayerNorm trick)")
print(f"{'='*70}")


class DirectRMSNormGated(nn.Module):
    """Standard RMSNorm + SiLU gate — no doubled-LayerNorm trick."""

    def __init__(self, hidden_size: int, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x = hidden_states.float()
        variance = (x * x).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(hidden_states.dtype) * self.weight.to(hidden_states.dtype)
        return x * F.silu(gate.to(hidden_states.dtype))


class DirectRMSNormCoreNormWrapper(nn.Module):
    def __init__(self, stage: Qwen35LinearCoreNormStage):
        super().__init__()
        self.num_v_heads = stage.num_v_heads
        self.head_k_dim = stage.head_k_dim
        self.head_v_dim = stage.head_v_dim
        self.value_dim = stage.value_dim
        self.norm = DirectRMSNormGated(
            stage.head_v_dim, stage.norm.weight, eps=stage.norm.eps
        )
        self.out_proj = stage.out_proj
        self.register_buffer(
            "rec_state",
            torch.zeros(1, stage.num_v_heads, stage.head_k_dim, stage.head_v_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, query, key, value, g, beta, z):
        core, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=self.rec_state, output_final_state=True,
            expected_batch_size=1, expected_num_heads=self.num_v_heads,
            expected_seq_len=SEQ_LEN, expected_k_dim=self.head_k_dim,
            expected_v_dim=self.head_v_dim,
        )
        core_normed = self.norm(
            core.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        ).reshape(1, SEQ_LEN, self.value_dim)
        cf = core_normed.transpose(1, 2).unsqueeze(2)
        out = self.out_proj(cf.to(MODEL_DTYPE)).squeeze(2).transpose(1, 2)
        return out


with torch.no_grad():
    c_wrapper = DirectRMSNormCoreNormWrapper(attn.core_norm_stage).eval()
    ref_c = c_wrapper(query, key, value, g, beta, z).numpy()
cos_c_vs_orig = cosine(ref_full, ref_c)
print(f"  PyTorch direct-RMSNorm ref vs original: cosine={cos_c_vs_orig:.10f}")

result_c = export_and_test(c_wrapper, "C_direct_rmsnorm", ref_c)


# ════════════════════════════════════════════════════════════════════
# STRATEGY D: FP32 compute precision for entire CoreNorm export
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  D) FP32 compute precision")
print(f"{'='*70}")

result_d = export_and_test(
    BaselineWrapper(attn.core_norm_stage).eval(),
    "D_fp32_precision",
    ref_full,
    compute_precision=ct.precision.FLOAT32,
)


# ════════════════════════════════════════════════════════════════════
# STRATEGY E: FP32 recurrence math (force_fp16_math=False at trace)
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  E) FP32 recurrence math (explicit fp32 in gated-delta-rule)")
print(f"{'='*70}")


class FP32RecurrenceWrapper(nn.Module):
    """Use fp32 explicitly inside the recurrence kernel."""

    def __init__(self, stage: Qwen35LinearCoreNormStage):
        super().__init__()
        self.num_v_heads = stage.num_v_heads
        self.head_k_dim = stage.head_k_dim
        self.head_v_dim = stage.head_v_dim
        self.value_dim = stage.value_dim
        self.norm = stage.norm
        self.out_proj = stage.out_proj
        self.register_buffer(
            "rec_state",
            torch.zeros(1, stage.num_v_heads, stage.head_k_dim, stage.head_v_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, query, key, value, g, beta, z):
        core, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=self.rec_state, output_final_state=True,
            expected_batch_size=1, expected_num_heads=self.num_v_heads,
            expected_seq_len=SEQ_LEN, expected_k_dim=self.head_k_dim,
            expected_v_dim=self.head_v_dim,
            math_dtype=torch.float32,  # explicitly fp32
        )
        core_normed = self.norm(
            core.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        ).reshape(1, SEQ_LEN, self.value_dim)
        cf = core_normed.transpose(1, 2).unsqueeze(2)
        out = self.out_proj(cf.to(MODEL_DTYPE)).squeeze(2).transpose(1, 2)
        return out


with torch.no_grad():
    e_wrapper = FP32RecurrenceWrapper(attn.core_norm_stage).eval()
    ref_e = e_wrapper(query, key, value, g, beta, z).numpy()
cos_e_vs_orig = cosine(ref_full, ref_e)
print(f"  PyTorch fp32-rec ref vs original: cosine={cos_e_vs_orig:.10f}")

result_e = export_and_test(e_wrapper, "E_fp32_recurrence", ref_e)


# ════════════════════════════════════════════════════════════════════
# STRATEGY F: Direct RMSNorm + FP32 compute precision (combined)
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  F) Direct RMSNorm + FP32 compute precision (combined)")
print(f"{'='*70}")

result_f = export_and_test(
    DirectRMSNormCoreNormWrapper(attn.core_norm_stage).eval(),
    "F_direct_rmsnorm_fp32",
    ref_c,
    compute_precision=ct.precision.FLOAT32,
)


# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════
results = [r for r in [result_a, result_b, result_c, result_d, result_e, result_f] if r]

print(f"\n{'='*70}")
print(f"  {'Strategy':<40} {'cosine':<14} {'max_abs':<10} {'ch0_mean':<10} {'status'}")
print(f"  {'-'*80}")
for r in results:
    if r.get("status") == "ok":
        print(f"  {r['label']:<40} {r['cosine']:<14.10f} {r['max_abs']:<10.6f} "
              f"{r['ch0_mean_abs']:<10.6f} {r['status']}")
    else:
        print(f"  {r['label']:<40} {'—':<14} {'—':<10} {'—':<10} {r['status']}")
print(f"{'='*70}")

# Identify best
ok_results = [r for r in results if r.get("status") == "ok"]
if ok_results:
    best = max(ok_results, key=lambda r: r["cosine"])
    baseline_cos = result_a.get("cosine", 0) if result_a.get("status") == "ok" else 0
    print(f"\n  Best: {best['label']}  cosine={best['cosine']:.10f}")
    if baseline_cos > 0:
        recovery = (best["cosine"] - baseline_cos) / (1.0 - baseline_cos) * 100
        print(f"  Recovery: {recovery:.1f}% of gap closed (baseline→1.0)")

# Save report
report = {
    "model": MODEL_PATH,
    "seq_len": SEQ_LEN,
    "ctx": CTX,
    "results": results,
}
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Report saved: {REPORT_PATH}")
