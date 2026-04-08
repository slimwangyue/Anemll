#!/usr/bin/env python3
"""Test hypothesis: ANE error in full pipeline comes from .to(float32) casts
being preserved on CPU but stripped on ANE.

If we trace with math_dtype=fp16 (no fp32 casts), ANE should equal CoreML CPU.
If we trace with math_dtype=fp32 (has fp32 casts), ANE != CoreML CPU.
"""
import sys, os, gc, time, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from anemll.models.qwen3_5_model import Qwen35LinearAttention

BATCH = 1
NUM_HEADS = 32
K_DIM = 128
V_DIM = 128
CHUNK_SIZE = 16
SEQ_LEN = 64
SEED = 42
OUT_DIR = "/tmp/diag_cast_hypothesis"


def cosine_sim(a, b):
    a_f = a.flatten().double()
    b_f = b.flatten().double()
    if a_f.norm() < 1e-10 and b_f.norm() < 1e-10:
        return 1.0
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def report(label, ref, test, indent=2):
    diff = (ref.double() - test.double()).abs()
    cos  = cosine_sim(ref, test)
    max_abs  = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = ref.double().abs().max().item()
    rel_max  = max_abs / max(ref_scale, 1e-10)
    prefix = " " * indent
    print(f"{prefix}{label:55s}  cos={cos:.10f}  max_abs={max_abs:.6e}  "
          f"mean_abs={mean_abs:.6e}  rel={rel_max:.4e}")


class ChunkGDR_FP32Math(nn.Module):
    """Wraps _chunk_gated_delta_rule with math_dtype=float32."""
    def forward(self, query, key, value, g, beta, initial_state):
        out, state = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key, value, g, beta,
            chunk_size=CHUNK_SIZE,
            initial_state=initial_state,
            output_final_state=True,
            expected_batch_size=BATCH,
            expected_num_heads=NUM_HEADS,
            expected_seq_len=SEQ_LEN,
            expected_k_dim=K_DIM,
            expected_v_dim=V_DIM,
            math_dtype=torch.float32,  # Has .to(float32) casts
        )
        return out, state


class ChunkGDR_FP16Math(nn.Module):
    """Wraps _chunk_gated_delta_rule with math_dtype=float16."""
    def forward(self, query, key, value, g, beta, initial_state):
        out, state = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key, value, g, beta,
            chunk_size=CHUNK_SIZE,
            initial_state=initial_state,
            output_final_state=True,
            expected_batch_size=BATCH,
            expected_num_heads=NUM_HEADS,
            expected_seq_len=SEQ_LEN,
            expected_k_dim=K_DIM,
            expected_v_dim=V_DIM,
            math_dtype=torch.float16,  # No .to(float32) casts — pure fp16
        )
        return out, state


def export_and_compare(model, inputs, label, ref_output, ref_state):
    """Export model, load ANE + CPU, compare."""
    model.eval()
    names = list(inputs.keys())
    tensors = list(inputs.values())

    print(f"\n--- {label} ---")
    with torch.no_grad():
        traced = torch.jit.trace(model, tensors)

    ct_inputs = [ct.TensorType(name=n, shape=t.shape) for n, t in inputs.items()]
    ml = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=[ct.TensorType(name="output"), ct.TensorType(name="final_state")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    path = os.path.join(OUT_DIR, f"{label}.mlpackage")
    os.makedirs(OUT_DIR, exist_ok=True)
    ml.save(path)
    del ml; gc.collect()

    ane = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    cpu = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)

    np_inputs = {n: t.numpy().astype(np.float16) for n, t in inputs.items()}
    ane_out = ane.predict(np_inputs)
    cpu_out = cpu.predict(np_inputs)

    ane_output = torch.from_numpy(np.array(ane_out["output"]))
    ane_state  = torch.from_numpy(np.array(ane_out["final_state"]))
    cpu_output = torch.from_numpy(np.array(cpu_out["output"]))
    cpu_state  = torch.from_numpy(np.array(cpu_out["final_state"]))

    report(f"  output ANE vs fp32_ref", ref_output, ane_output)
    report(f"  output CoreML_CPU vs fp32_ref", ref_output, cpu_output)
    report(f"  output ANE vs CoreML_CPU", cpu_output, ane_output)
    report(f"  state  ANE vs fp32_ref", ref_state, ane_state)
    report(f"  state  CoreML_CPU vs fp32_ref", ref_state, cpu_state)
    report(f"  state  ANE vs CoreML_CPU", cpu_state, ane_state)

    del ane, cpu; gc.collect()
    return ane_output, cpu_output, ane_state, cpu_state


def main():
    print("=" * 100)
    print("HYPOTHESIS TEST: .to(float32) cast handling on ANE vs CPU")
    print("=" * 100)

    torch.manual_seed(SEED)
    q = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, K_DIM) * 0.1
    k = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, K_DIM) * 0.1
    v = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, V_DIM) * 0.1
    g = -torch.rand(BATCH, SEQ_LEN, NUM_HEADS).abs() * 2.0 - 0.1
    beta = torch.sigmoid(torch.randn(BATCH, SEQ_LEN, NUM_HEADS))
    s = torch.zeros(BATCH, NUM_HEADS, K_DIM, V_DIM)

    inputs = {"query": q, "key": k, "value": v, "g": g, "beta": beta, "initial_state": s}

    # Compute fp32 reference on CPU
    print("\n[REF] Computing fp32 reference on CPU...")
    ref_out, ref_state = Qwen35LinearAttention._chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=CHUNK_SIZE, initial_state=s.clone(),
        output_final_state=True,
        expected_batch_size=BATCH, expected_num_heads=NUM_HEADS,
        expected_seq_len=SEQ_LEN, expected_k_dim=K_DIM, expected_v_dim=V_DIM,
        math_dtype=torch.float32,
    )

    # Test A: math_dtype=float32 (has .to(float32) casts in trace)
    print("\n" + "=" * 100)
    print("TEST A: math_dtype=float32 — graph HAS .to(float32) casts")
    print("  Prediction: ANE will strip fp32 casts → worse than CoreML CPU")
    print("=" * 100)
    export_and_compare(ChunkGDR_FP32Math(), inputs, "fp32_math", ref_out, ref_state)

    # Test B: math_dtype=float16 (no .to(float32) casts)
    print("\n" + "=" * 100)
    print("TEST B: math_dtype=float16 — graph has NO fp32 casts")
    print("  Prediction: ANE = CoreML CPU (no casts to strip)")
    print("=" * 100)
    export_and_compare(ChunkGDR_FP16Math(), inputs, "fp16_math", ref_out, ref_state)

    # Cleanup
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
