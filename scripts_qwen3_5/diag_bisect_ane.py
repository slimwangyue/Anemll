#!/usr/bin/env python3
"""Bisection: find the SMALLEST sub-graph that triggers ANE vs CPU divergence.

Individual ops show ANE = CPU. Full pipeline (2105 ops) shows cos=0.9983.
This script tests intermediate-size graphs to find where the divergence starts.
"""
import sys, os, gc, time, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from anemll.models.qwen3_5_model import _l2norm

BATCH = 1
H = 32
K = 128
V = 128
CS = 16
SEQ = 64
NC = SEQ // CS
SEED = 42
OUT_DIR = "/tmp/diag_bisect_ane"


def cosine_sim(a, b):
    a_f = a.flatten().double()
    b_f = b.flatten().double()
    if a_f.norm() < 1e-10 and b_f.norm() < 1e-10:
        return 1.0
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def report(label, ref, test):
    diff = (ref.double() - test.double()).abs()
    cos  = cosine_sim(ref, test)
    max_abs  = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = ref.double().abs().max().item()
    rel_max  = max_abs / max(ref_scale, 1e-10)
    print(f"  {label:55s}  cos={cos:.10f}  max_abs={max_abs:.6e}  rel={rel_max:.4e}")
    return max_abs


def export_test(model, inputs, label, ref_outputs):
    """Export, test ANE vs CPU, return whether they diverge."""
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, list(inputs.values()))

    ct_inputs = [ct.TensorType(name=n, shape=t.shape) for n, t in inputs.items()]
    out_names = [f"out_{i}" for i in range(len(ref_outputs))]
    ct_outputs = [ct.TensorType(name=n) for n in out_names]

    ml = ct.convert(
        traced, inputs=ct_inputs, outputs=ct_outputs,
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

    max_div = 0
    for i, (name, ref) in enumerate(zip(out_names, ref_outputs)):
        ane_t = torch.from_numpy(np.array(ane_out[name]))
        cpu_t = torch.from_numpy(np.array(cpu_out[name]))
        report(f"[{label}] {name} ANE vs fp32_ref", ref, ane_t)
        report(f"[{label}] {name} CoreML_CPU vs fp32_ref", ref, cpu_t)
        d = report(f"[{label}] {name} ANE vs CoreML_CPU", cpu_t, ane_t)
        max_div = max(max_div, d)

    del ane, cpu; gc.collect()
    return max_div


# ── Test Models ──

class Test1_PrepOnly(nn.Module):
    """Input prep: transpose, l2norm, scale, pad, reshape, compute v_beta/k_beta."""
    def forward(self, query, key, value, g, beta):
        query = query.transpose(1, 2).contiguous()
        key   = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        beta  = beta.transpose(1, 2).contiguous()
        g     = g.transpose(1, 2).contiguous()
        query = _l2norm(query, dim=-1) * (1.0 / K**0.5)
        key   = _l2norm(key, dim=-1)
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        return query, key, v_beta, k_beta, g


class Test2_PrepAndDecay(nn.Module):
    """Prep + cumulative g + decay mask."""
    def __init__(self):
        super().__init__()
        self.register_buffer("tril_ones", torch.tril(torch.ones(CS, CS)))
        self.register_buffer("strict_lower", torch.tril(torch.ones(CS, CS), diagonal=-1))

    def forward(self, query, key, value, g, beta):
        query = query.transpose(1, 2).contiguous()
        key   = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        beta  = beta.transpose(1, 2).contiguous()
        g     = g.transpose(1, 2).contiguous()
        query = _l2norm(query, dim=-1) * (1.0 / K**0.5)
        key   = _l2norm(key, dim=-1)
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        # Reshape to chunks
        query  = query.reshape(BATCH, H, NC, CS, K)
        key    = key.reshape(BATCH, H, NC, CS, K)
        value  = value.reshape(BATCH, H, NC, CS, V)
        k_beta = k_beta.reshape(BATCH, H, NC, CS, K)
        v_beta = v_beta.reshape(BATCH, H, NC, CS, V)
        g      = g.reshape(BATCH, H, NC, CS)

        # Cumulative g + decay
        g = (self.tril_ones @ g.unsqueeze(-1)).squeeze(-1)
        decay_raw = (g.unsqueeze(-1) - g.unsqueeze(-2)) * self.tril_ones
        decay_mask = decay_raw.exp() * self.tril_ones
        return g, decay_mask


class Test3_PrepDecayWoodbury(nn.Module):
    """Prep + decay + intra-chunk Woodbury attention."""
    def __init__(self):
        super().__init__()
        self.register_buffer("tril_ones", torch.tril(torch.ones(CS, CS)))
        self.register_buffer("strict_lower", torch.tril(torch.ones(CS, CS), diagonal=-1))

    def forward(self, query, key, value, g, beta):
        query = query.transpose(1, 2).contiguous()
        key   = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        beta  = beta.transpose(1, 2).contiguous()
        g     = g.transpose(1, 2).contiguous()
        query = _l2norm(query, dim=-1) * (1.0 / K**0.5)
        key   = _l2norm(key, dim=-1)
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        query  = query.reshape(BATCH, H, NC, CS, K)
        key    = key.reshape(BATCH, H, NC, CS, K)
        value  = value.reshape(BATCH, H, NC, CS, V)
        k_beta = k_beta.reshape(BATCH, H, NC, CS, K)
        v_beta = v_beta.reshape(BATCH, H, NC, CS, V)
        g      = g.reshape(BATCH, H, NC, CS)

        g = (self.tril_ones @ g.unsqueeze(-1)).squeeze(-1)
        decay_raw = (g.unsqueeze(-1) - g.unsqueeze(-2)) * self.tril_ones
        decay_mask = decay_raw.exp() * self.tril_ones

        # Intra-chunk attention (Woodbury)
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * self.strict_lower
        attn_rows = [attn[..., 0:1, :]]
        for i in range(1, CS):
            row = attn[..., i, :i].clone()
            sub = torch.cat([prev_row[..., :i] for prev_row in attn_rows[:i]], dim=-2)
            updated_row = row + (row.unsqueeze(-1) * sub).sum(-2)
            tail = attn[..., i:i+1, i:]
            full_row = torch.cat([updated_row.unsqueeze(-2), tail], dim=-1)
            attn_rows.append(full_row)
        attn = torch.cat(attn_rows, dim=-2)
        attn = attn + torch.eye(CS, dtype=attn.dtype, device=attn.device)

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        return value, k_cumdecay


class Test4_InterChunkOnly(nn.Module):
    """Just the inter-chunk loop (state accumulation + output).
    Takes pre-computed inputs.
    """
    def __init__(self, n_chunks):
        super().__init__()
        self.n_chunks = n_chunks
        self.register_buffer("tril_diag", torch.tril(torch.ones(CS, CS)))

    def forward(self, query, key, value, k_cumdecay, decay_mask, g, initial_state):
        # All inputs are already chunked: (B, H, NC, CS, D) or (B, H, NC, CS)
        state = initial_state.clone()
        chunks = []
        for ci in range(self.n_chunks):
            q_i = query[:, :, ci]
            k_i = key[:, :, ci]
            v_i = value[:, :, ci]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, ci]) * self.tril_diag
            v_prime = k_cumdecay[:, :, ci] @ state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, ci, :, None].exp()) @ state
            output_i = attn_inter + attn @ v_new
            chunks.append(output_i.unsqueeze(2))
            state = (
                state * g[:, :, ci, -1, None, None].exp()
                + (k_i * (g[:, :, ci, -1, None] - g[:, :, ci]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
        core = torch.cat(chunks, dim=2)
        total_len = self.n_chunks * CS
        core = core.reshape(BATCH, H, total_len, V)
        return core, state


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 100)
    print("BISECTION: Finding minimal graph that triggers ANE vs CPU divergence")
    print("=" * 100)

    torch.manual_seed(SEED)
    q = torch.randn(BATCH, SEQ, H, K) * 0.1
    k = torch.randn(BATCH, SEQ, H, K) * 0.1
    v = torch.randn(BATCH, SEQ, H, V) * 0.1
    g = -torch.rand(BATCH, SEQ, H).abs() * 2.0 - 0.1
    beta = torch.sigmoid(torch.randn(BATCH, SEQ, H))
    s0 = torch.zeros(BATCH, H, K, V)

    inputs_base = {"query": q, "key": k, "value": v, "g": g, "beta": beta}

    # ── Test 1: Prep only (~50 ops) ──
    print("\n" + "=" * 80)
    print("TEST 1: Input preparation only (~50 ops)")
    print("=" * 80)
    m1 = Test1_PrepOnly()
    with torch.no_grad():
        refs1 = m1(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone())
    export_test(m1, inputs_base, "t1_prep", list(refs1))

    # ── Test 2: Prep + Decay (~100 ops) ──
    print("\n" + "=" * 80)
    print("TEST 2: Prep + cumG + decay_mask (~100 ops)")
    print("=" * 80)
    m2 = Test2_PrepAndDecay()
    with torch.no_grad():
        refs2 = m2(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone())
    export_test(m2, inputs_base, "t2_prepdecay", list(refs2))

    # ── Test 3: Prep + Decay + Woodbury (~700 ops) ──
    print("\n" + "=" * 80)
    print("TEST 3: Prep + decay + Woodbury (~700 ops)")
    print("=" * 80)
    m3 = Test3_PrepDecayWoodbury()
    with torch.no_grad():
        refs3 = m3(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone())
    export_test(m3, inputs_base, "t3_woodbury", list(refs3))

    # ── Test 4: Inter-chunk loop only (pre-computed inputs) ──
    # Prepare proper chunked inputs from CPU fp32
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    b_t = beta.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous()
    q_n = _l2norm(q_t, dim=-1) / K**0.5
    k_n = _l2norm(k_t, dim=-1)
    v_beta = v_t * b_t.unsqueeze(-1)
    k_beta = k_n * b_t.unsqueeze(-1)

    q_c = q_n.reshape(BATCH, H, NC, CS, K)
    k_c = k_n.reshape(BATCH, H, NC, CS, K)
    v_c = v_t.reshape(BATCH, H, NC, CS, V)
    kb_c = k_beta.reshape(BATCH, H, NC, CS, K)
    vb_c = v_beta.reshape(BATCH, H, NC, CS, V)
    g_c = g_t.reshape(BATCH, H, NC, CS)

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

    val_updated = attn_w @ vb_c
    k_cumdecay = attn_w @ (kb_c * g_cum.exp().unsqueeze(-1))

    for nc_test in [1, 2, 4]:
        print(f"\n" + "=" * 80)
        print(f"TEST 4-{nc_test}: Inter-chunk loop ONLY, {nc_test} chunk(s)")
        print("=" * 80)

        m4 = Test4_InterChunkOnly(nc_test)
        inp4 = {
            "query": q_c[:,:,:nc_test],
            "key": k_c[:,:,:nc_test],
            "value": val_updated[:,:,:nc_test],
            "k_cumdecay": k_cumdecay[:,:,:nc_test],
            "decay_mask": dm[:,:,:nc_test],
            "g": g_cum[:,:,:nc_test],
            "initial_state": s0,
        }
        with torch.no_grad():
            refs4 = m4(
                q_c[:,:,:nc_test].clone(),
                k_c[:,:,:nc_test].clone(),
                val_updated[:,:,:nc_test].clone(),
                k_cumdecay[:,:,:nc_test].clone(),
                dm[:,:,:nc_test].clone(),
                g_cum[:,:,:nc_test].clone(),
                s0.clone(),
            )
        export_test(m4, inp4, f"t4_interchunk_{nc_test}", list(refs4))

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print("\n" + "=" * 100)
    print("BISECTION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
