#!/usr/bin/env python3
"""Test: Split the 128-dim @state matmul inside the inter-chunk loop.

Since barriers don't work (MIL sees through them), try changing the
actual computation structure. The hypothesis is that ANE handles
multiple smaller matmuls differently from one large matmul in the
context of a multi-iteration unrolled loop.

Also test: what if we compute the output differently?
"""
import sys, os, gc, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from anemll.models.qwen3_5_model import _l2norm

B = 1
H = 32
K = 128
V = 128
CS = 16
SEQ = 64
NC = SEQ // CS
SEED = 42
OUT_DIR = "/tmp/diag_split_test"


def cosine_sim(a, b):
    a_f = a.flatten().double()
    b_f = b.flatten().double()
    if a_f.norm() < 1e-10 and b_f.norm() < 1e-10:
        return 1.0
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def report(label, ref, test):
    diff = (ref.double() - test.double()).abs()
    cos  = cosine_sim(ref, test)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = ref.double().abs().max().item()
    print(f"  {label:55s}  cos={cos:.10f}  max_abs={max_abs:.6e}  mean={mean_abs:.6e}")
    return {"cos": cos, "max_abs": max_abs}


class Baseline(nn.Module):
    """Original inter-chunk loop."""
    def __init__(self):
        super().__init__()
        self.register_buffer("tril_diag", torch.tril(torch.ones(CS, CS)))

    def forward(self, query, key, value, k_cumdecay, decay_mask, g, initial_state):
        state = initial_state.clone()
        chunks = []
        for ci in range(NC):
            q_i = query[:, :, ci]
            k_i = key[:, :, ci]
            v_i = value[:, :, ci]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, ci]) * self.tril_diag
            v_prime = k_cumdecay[:, :, ci] @ state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, ci, :, None].exp()) @ state
            chunks.append((attn_inter + attn @ v_new).unsqueeze(2))
            state = (
                state * g[:, :, ci, -1, None, None].exp()
                + (k_i * (g[:, :, ci, -1, None] - g[:, :, ci]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
        core = torch.cat(chunks, dim=2).reshape(B, H, SEQ, V)
        return core, state


class SplitAttnInterG4(nn.Module):
    """Split the attn_inter @state matmul into 4 groups (32-dim each)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("tril_diag", torch.tril(torch.ones(CS, CS)))
        self.G = 4
        self.KG = K // 4

    def forward(self, query, key, value, k_cumdecay, decay_mask, g, initial_state):
        state = initial_state.clone()
        chunks = []
        G = self.G
        KG = self.KG
        for ci in range(NC):
            q_i = query[:, :, ci]
            k_i = key[:, :, ci]
            v_i = value[:, :, ci]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, ci]) * self.tril_diag
            v_prime = k_cumdecay[:, :, ci] @ state
            v_new = v_i - v_prime
            # SPLIT: attn_inter = (q_i * g_exp) @ state → 4 x 32-dim matmuls
            q_g_exp = q_i * g[:, :, ci, :, None].exp()
            attn_inter = q_g_exp[:,:,:,:KG] @ state[:,:,:KG,:]
            attn_inter = attn_inter + q_g_exp[:,:,:,KG:2*KG] @ state[:,:,KG:2*KG,:]
            attn_inter = attn_inter + q_g_exp[:,:,:,2*KG:3*KG] @ state[:,:,2*KG:3*KG,:]
            attn_inter = attn_inter + q_g_exp[:,:,:,3*KG:4*KG] @ state[:,:,3*KG:4*KG,:]
            chunks.append((attn_inter + attn @ v_new).unsqueeze(2))
            state = (
                state * g[:, :, ci, -1, None, None].exp()
                + (k_i * (g[:, :, ci, -1, None] - g[:, :, ci]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
        core = torch.cat(chunks, dim=2).reshape(B, H, SEQ, V)
        return core, state


class SplitBothG4(nn.Module):
    """Split BOTH @state matmuls (attn_inter AND v_prime)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("tril_diag", torch.tril(torch.ones(CS, CS)))
        self.KG = K // 4

    def forward(self, query, key, value, k_cumdecay, decay_mask, g, initial_state):
        state = initial_state.clone()
        chunks = []
        KG = self.KG
        for ci in range(NC):
            q_i = query[:, :, ci]
            k_i = key[:, :, ci]
            v_i = value[:, :, ci]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, ci]) * self.tril_diag
            # SPLIT v_prime
            kcd = k_cumdecay[:, :, ci]
            v_prime = kcd[:,:,:,:KG] @ state[:,:,:KG,:]
            v_prime = v_prime + kcd[:,:,:,KG:2*KG] @ state[:,:,KG:2*KG,:]
            v_prime = v_prime + kcd[:,:,:,2*KG:3*KG] @ state[:,:,2*KG:3*KG,:]
            v_prime = v_prime + kcd[:,:,:,3*KG:4*KG] @ state[:,:,3*KG:4*KG,:]
            v_new = v_i - v_prime
            # SPLIT attn_inter
            q_g_exp = q_i * g[:, :, ci, :, None].exp()
            attn_inter = q_g_exp[:,:,:,:KG] @ state[:,:,:KG,:]
            attn_inter = attn_inter + q_g_exp[:,:,:,KG:2*KG] @ state[:,:,KG:2*KG,:]
            attn_inter = attn_inter + q_g_exp[:,:,:,2*KG:3*KG] @ state[:,:,2*KG:3*KG,:]
            attn_inter = attn_inter + q_g_exp[:,:,:,3*KG:4*KG] @ state[:,:,3*KG:4*KG,:]
            chunks.append((attn_inter + attn @ v_new).unsqueeze(2))
            state = (
                state * g[:, :, ci, -1, None, None].exp()
                + (k_i * (g[:, :, ci, -1, None] - g[:, :, ci]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
        core = torch.cat(chunks, dim=2).reshape(B, H, SEQ, V)
        return core, state


class ReorderOutput(nn.Module):
    """Reorder: compute (attn @ v_new) first, then add attn_inter."""
    def __init__(self):
        super().__init__()
        self.register_buffer("tril_diag", torch.tril(torch.ones(CS, CS)))

    def forward(self, query, key, value, k_cumdecay, decay_mask, g, initial_state):
        state = initial_state.clone()
        chunks = []
        for ci in range(NC):
            q_i = query[:, :, ci]
            k_i = key[:, :, ci]
            v_i = value[:, :, ci]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, ci]) * self.tril_diag
            v_prime = k_cumdecay[:, :, ci] @ state
            v_new = v_i - v_prime
            # Reorder: compute local part first
            local_part = attn @ v_new
            attn_inter = (q_i * g[:, :, ci, :, None].exp()) @ state
            # Different add order
            chunks.append((local_part + attn_inter).unsqueeze(2))
            state = (
                state * g[:, :, ci, -1, None, None].exp()
                + (k_i * (g[:, :, ci, -1, None] - g[:, :, ci]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
        core = torch.cat(chunks, dim=2).reshape(B, H, SEQ, V)
        return core, state


class UseEinsumForState(nn.Module):
    """Replace @ state with einsum (different MIL lowering path)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("tril_diag", torch.tril(torch.ones(CS, CS)))

    def forward(self, query, key, value, k_cumdecay, decay_mask, g, initial_state):
        state = initial_state.clone()
        chunks = []
        for ci in range(NC):
            q_i = query[:, :, ci]
            k_i = key[:, :, ci]
            v_i = value[:, :, ci]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, ci]) * self.tril_diag
            # Use einsum instead of @ for state matmuls
            v_prime = torch.einsum('bhsk,bhkv->bhsv', k_cumdecay[:, :, ci], state)
            v_new = v_i - v_prime
            q_g_exp = q_i * g[:, :, ci, :, None].exp()
            attn_inter = torch.einsum('bhsk,bhkv->bhsv', q_g_exp, state)
            chunks.append((attn_inter + attn @ v_new).unsqueeze(2))
            state = (
                state * g[:, :, ci, -1, None, None].exp()
                + (k_i * (g[:, :, ci, -1, None] - g[:, :, ci]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
        core = torch.cat(chunks, dim=2).reshape(B, H, SEQ, V)
        return core, state


def export_test(model, inputs, label, ref_output, ref_state):
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, list(inputs.values()))

    ct_inputs = [ct.TensorType(name=n, shape=t.shape) for n, t in inputs.items()]
    ml = ct.convert(
        traced, inputs=ct_inputs,
        outputs=[ct.TensorType(name="output"), ct.TensorType(name="final_state")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    path = os.path.join(OUT_DIR, f"{label}.mlpackage")
    ml.save(path)
    del ml; gc.collect()

    ane = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    cpu = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    np_inp = {n: t.numpy().astype(np.float16) for n, t in inputs.items()}
    ane_out = ane.predict(np_inp)
    cpu_out = cpu.predict(np_inp)

    ane_out_t = torch.from_numpy(np.array(ane_out["output"]))
    cpu_out_t = torch.from_numpy(np.array(cpu_out["output"]))
    ane_st = torch.from_numpy(np.array(ane_out["final_state"]))
    cpu_st = torch.from_numpy(np.array(cpu_out["final_state"]))

    report(f"output ANE vs fp32_ref", ref_output, ane_out_t)
    report(f"output CoreML_CPU vs fp32_ref", ref_output, cpu_out_t)
    r = report(f"output ANE vs CoreML_CPU", cpu_out_t, ane_out_t)
    report(f"state  ANE vs CoreML_CPU", cpu_st, ane_st)

    del ane, cpu; gc.collect()
    return r


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 100)
    print("SPLIT MATMUL TEST: Structural changes to inter-chunk loop")
    print("=" * 100)

    torch.manual_seed(SEED)
    q = torch.randn(B, SEQ, H, K) * 0.1
    k = torch.randn(B, SEQ, H, K) * 0.1
    v = torch.randn(B, SEQ, H, V) * 0.1
    g_raw = -torch.rand(B, SEQ, H).abs() * 2.0 - 0.1
    beta = torch.sigmoid(torch.randn(B, SEQ, H))

    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    b_t = beta.transpose(1, 2).contiguous()
    g_t = g_raw.transpose(1, 2).contiguous()
    q_n = _l2norm(q_t, dim=-1) / K**0.5
    k_n = _l2norm(k_t, dim=-1)
    vb = v_t * b_t.unsqueeze(-1)
    kb = k_n * b_t.unsqueeze(-1)

    q_c = q_n.reshape(B, H, NC, CS, K)
    k_c = k_n.reshape(B, H, NC, CS, K)
    kb_c = kb.reshape(B, H, NC, CS, K)
    vb_c = vb.reshape(B, H, NC, CS, V)
    g_c = g_t.reshape(B, H, NC, CS)

    tril = torch.tril(torch.ones(CS, CS))
    strict = torch.tril(torch.ones(CS, CS), diagonal=-1)
    g_cum = (tril @ g_c.unsqueeze(-1)).squeeze(-1)
    dr = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)) * tril
    dm = dr.exp() * tril

    attn_base = -((kb_c @ k_c.transpose(-1, -2)) * dm) * strict
    attn_rows = [attn_base[..., 0:1, :]]
    for i in range(1, CS):
        row = attn_base[..., i, :i].clone()
        sub = torch.cat([pr[..., :i] for pr in attn_rows[:i]], dim=-2)
        updated = row + (row.unsqueeze(-1) * sub).sum(-2)
        tail = attn_base[..., i:i+1, i:]
        attn_rows.append(torch.cat([updated.unsqueeze(-2), tail], dim=-1))
    attn_w = torch.cat(attn_rows, dim=-2)
    attn_w = attn_w + torch.eye(CS, dtype=attn_w.dtype)
    val_post = attn_w @ vb_c
    k_cumdecay = attn_w @ (kb_c * g_cum.exp().unsqueeze(-1))

    s0 = torch.zeros(B, H, K, V)
    inputs = {
        "query": q_c, "key": k_c, "value": val_post,
        "k_cumdecay": k_cumdecay, "decay_mask": dm,
        "g": g_cum, "initial_state": s0,
    }

    m_ref = Baseline()
    with torch.no_grad():
        ref_out, ref_state = m_ref(**{kk: vv.clone() for kk, vv in inputs.items()})

    tests = [
        ("baseline", Baseline()),
        ("split_attn_inter_G4", SplitAttnInterG4()),
        ("split_both_G4", SplitBothG4()),
        ("reorder_output", ReorderOutput()),
        ("einsum_state", UseEinsumForState()),
    ]

    results = {}
    for name, model in tests:
        print(f"\n{'='*80}")
        print(f"  {name}")
        print(f"{'='*80}")
        r = export_test(model, inputs, name, ref_out, ref_state)
        results[name] = r

    print(f"\n{'='*100}")
    print("SUMMARY: ANE vs CoreML_CPU output divergence")
    print(f"{'='*100}")
    best = min(rr["max_abs"] for rr in results.values())
    for name, r in results.items():
        flag = " ← BEST" if r["max_abs"] == best else ""
        imp = (results["baseline"]["max_abs"] - r["max_abs"]) / results["baseline"]["max_abs"] * 100
        print(f"  {name:30s}  cos={r['cos']:.10f}  max_abs={r['max_abs']:.6e}  improve={imp:+.1f}%{flag}")

    shutil.rmtree(OUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
