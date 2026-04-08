#!/usr/bin/env python3
"""Minimal test: ANE matmul accumulation error for different reduction dims.

Tests the hypothesis that ANE fp16 matmul produces significantly more error
for larger reduction dimensions (128) compared to smaller ones (16, 32).

Also tests a split-matmul mitigation: replacing one 128-dim matmul with
G separate (128/G)-dim matmuls + sum.
"""
import sys, os, gc, time, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

OUT_DIR = "/tmp/diag_matmul_ane"
SEED = 42

# Shapes matching Qwen3.5 inter-chunk computation
B = 1
H = 32
CS = 16       # chunk_size
K_DIM = 128
V_DIM = 128


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
    print(f"{prefix}{label:50s}  cos={cos:.10f}  max_abs={max_abs:.6e}  "
          f"mean_abs={mean_abs:.6e}  rel={rel_max:.4e}")
    return {"cos": cos, "max_abs": max_abs, "mean_abs": mean_abs}


# ── Test modules ──

class MatmulBaseline(nn.Module):
    """Standard matmul: (B,H,CS,K) @ (B,H,K,V) → (B,H,CS,V)
    Tests 128-dim accumulation.
    """
    def forward(self, a, b):
        return a @ b


class MatmulSplit(nn.Module):
    """Split matmul: split K_DIM into G groups, matmul each, sum.
    Each group has K_DIM/G accumulation.
    """
    def __init__(self, G=4):
        super().__init__()
        self.G = G
        self.K_G = K_DIM // G

    def forward(self, a, b):
        # a: (B, H, CS, K), b: (B, H, K, V)
        G = self.G
        K_G = self.K_G
        result = a[:, :, :, :K_G] @ b[:, :, :K_G, :]
        for gi in range(1, G):
            lo = gi * K_G
            hi = lo + K_G
            result = result + a[:, :, :, lo:hi] @ b[:, :, lo:hi, :]
        return result


class MatmulSmall(nn.Module):
    """Small matmul: (B,H,CS,CS) @ (B,H,CS,V) → (B,H,CS,V)
    Tests 16-dim accumulation.
    """
    def forward(self, a, b):
        return a @ b


class InterChunkOutput(nn.Module):
    """Full inter-chunk output computation:
    attn_inter = (q * g_exp) @ state
    output = attn_inter + attn_local @ v_new
    """
    def forward(self, q_g_exp, state_tensor, attn_local, v_new):
        attn_inter = q_g_exp @ state_tensor
        return attn_inter + attn_local @ v_new


class InterChunkOutputSplit(nn.Module):
    """Same but with split matmul for the 128-dim @ state operation."""
    def __init__(self, G=4):
        super().__init__()
        self.G = G
        self.K_G = K_DIM // G

    def forward(self, q_g_exp, state_tensor, attn_local, v_new):
        # Split the 128-dim matmul into G groups
        G = self.G
        K_G = self.K_G
        attn_inter = q_g_exp[:, :, :, :K_G] @ state_tensor[:, :, :K_G, :]
        for gi in range(1, G):
            lo = gi * K_G
            hi = lo + K_G
            attn_inter = attn_inter + q_g_exp[:, :, :, lo:hi] @ state_tensor[:, :, lo:hi, :]
        return attn_inter + attn_local @ v_new


def export_and_test(model, inputs_dict, label, ref_output):
    """Export, load on ANE + CPU, compare."""
    model.eval()
    input_tensors = list(inputs_dict.values())
    input_names = list(inputs_dict.keys())

    with torch.no_grad():
        traced = torch.jit.trace(model, input_tensors)

    ct_inputs = [ct.TensorType(name=n, shape=t.shape) for n, t in inputs_dict.items()]
    ml = ct.convert(
        traced,
        inputs=ct_inputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    path = os.path.join(OUT_DIR, f"{label}.mlpackage")
    ml.save(path)
    del ml; gc.collect()

    ane_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    cpu_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)

    np_inputs = {n: t.numpy().astype(np.float16) for n, t in inputs_dict.items()}
    ane_out = ane_model.predict(np_inputs)
    cpu_out = cpu_model.predict(np_inputs)

    # Get output (first key)
    out_key = list(ane_out.keys())[0]
    ane_result = torch.from_numpy(np.array(ane_out[out_key]))
    cpu_result = torch.from_numpy(np.array(cpu_out[out_key]))

    report(f"{label} ANE vs fp32_ref", ref_output, ane_result)
    report(f"{label} CoreML_CPU vs fp32_ref", ref_output, cpu_result)
    report(f"{label} ANE vs CoreML_CPU", cpu_result, ane_result)

    del ane_model, cpu_model; gc.collect()
    return ane_result, cpu_result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 100)
    print("MATMUL ACCUMULATION ERROR TEST")
    print(f"B={B} H={H} CS={CS} K_DIM={K_DIM} V_DIM={V_DIM}")
    print("=" * 100)

    torch.manual_seed(SEED)

    # ── Test 1: Large matmul (128-dim reduction) ──
    print("\n" + "-" * 100)
    print("TEST 1: Large matmul (B,H,16,128) @ (B,H,128,128) — 128-dim reduction")
    print("-" * 100)
    a = torch.randn(B, H, CS, K_DIM) * 0.1
    state = torch.randn(B, H, K_DIM, V_DIM) * 0.01
    ref_large = (a @ state).float()

    export_and_test(MatmulBaseline(), {"a": a, "b": state}, "matmul_128dim", ref_large)

    # ── Test 2: Split matmul (4x32-dim reduction) ──
    for G in [2, 4, 8]:
        print(f"\n" + "-" * 100)
        print(f"TEST 2: Split matmul G={G} → {K_DIM//G}-dim reductions")
        print("-" * 100)

        # Reference: same math, just computed differently
        model_split = MatmulSplit(G=G)
        with torch.no_grad():
            ref_split = model_split(a, state).float()
        # Verify split matches baseline on CPU
        report(f"  CPU: split G={G} vs baseline", ref_large, ref_split.float())

        export_and_test(model_split, {"a": a, "b": state}, f"matmul_split_{G}", ref_large)

    # ── Test 3: Small matmul (16-dim reduction) ──
    print(f"\n" + "-" * 100)
    print(f"TEST 3: Small matmul (B,H,16,16) @ (B,H,16,128) — 16-dim reduction")
    print("-" * 100)
    attn_local = torch.randn(B, H, CS, CS) * 0.01
    v_new = torch.randn(B, H, CS, V_DIM) * 0.1
    ref_small = (attn_local @ v_new).float()

    export_and_test(MatmulSmall(), {"a": attn_local, "b": v_new}, "matmul_16dim", ref_small)

    # ── Test 4: Full inter-chunk output (baseline vs split) ──
    print(f"\n" + "-" * 100)
    print(f"TEST 4: Full inter-chunk output (baseline)")
    print("-" * 100)
    q_g_exp = a.clone()  # reuse similar input
    ref_interchunk = (q_g_exp @ state + attn_local @ v_new).float()

    export_and_test(
        InterChunkOutput(),
        {"q_g_exp": q_g_exp, "state_tensor": state, "attn_local": attn_local, "v_new": v_new},
        "interchunk_baseline",
        ref_interchunk,
    )

    for G in [4, 8]:
        print(f"\n" + "-" * 100)
        print(f"TEST 5: Full inter-chunk output (split G={G})")
        print("-" * 100)

        export_and_test(
            InterChunkOutputSplit(G=G),
            {"q_g_exp": q_g_exp, "state_tensor": state, "attn_local": attn_local, "v_new": v_new},
            f"interchunk_split_{G}",
            ref_interchunk,
        )

    # ── Test 6: Realistic magnitudes (matching actual model) ──
    print(f"\n" + "-" * 100)
    print(f"TEST 6: Realistic q*exp(g) magnitudes")
    print("-" * 100)
    # In the real model, q is l2-normed and scaled, g is large negative
    # q * exp(g) has very small values
    q_real = torch.randn(B, H, CS, K_DIM) * 0.01  # l2normed and scaled
    g_real = -torch.rand(B, H, CS, 1) * 15.0 - 2.0  # large negative
    q_g_real = q_real * g_real.exp()  # Very small values
    state_real = torch.randn(B, H, K_DIM, V_DIM) * 0.05
    ref_real = (q_g_real @ state_real).float()

    print(f"  q_g_real range: [{q_g_real.min().item():.6e}, {q_g_real.max().item():.6e}]")
    print(f"  state range: [{state_real.min().item():.6e}, {state_real.max().item():.6e}]")
    print(f"  result range: [{ref_real.min().item():.6e}, {ref_real.max().item():.6e}]")

    export_and_test(MatmulBaseline(), {"a": q_g_real, "b": state_real},
                    "matmul_realistic", ref_real)

    for G in [4, 8]:
        export_and_test(MatmulSplit(G=G), {"a": q_g_real, "b": state_real},
                        f"matmul_realistic_split_{G}", ref_real)

    # Cleanup
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
