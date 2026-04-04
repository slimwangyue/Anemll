#!/usr/bin/env python3
"""Diagnostic: compare _chunk_gated_delta_rule vs _recurrent_gated_delta_rule parity.

Tests with various seq_len, math_dtype, and initial_state configurations.
Reports cosine similarity, max/mean abs diff, relative error.

Usage:
    python3 scripts_qwen3_5/diag_chunk_vs_recurrent.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from anemll.models.qwen3_5_model import Qwen35LinearAttention

_chunk = Qwen35LinearAttention._chunk_gated_delta_rule
_recurrent = Qwen35LinearAttention._recurrent_gated_delta_rule


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.flatten().float(), b.flatten().float()
    denom = a_f.abs().clamp(min=1e-8)
    return ((a_f - b_f).abs() / denom).max().item()


def report(label: str, a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    cos = cosine_sim(a, b)
    scale = a.float().abs().max().item()
    relerr = rel_err(a, b)
    print(f"  {label:30s}  cos={cos:.10f}  max_abs={diff.max().item():.6e}  "
          f"rel_err={relerr:.6e}  scale={scale:.4e}")


def run_test(
    batch: int,
    num_heads: int,
    seq_len: int,
    k_dim: int,
    v_dim: int,
    chunk_size: int,
    math_dtype: torch.dtype,
    zero_state: bool,
    realistic_g: bool = True,
    seed: int = 42,
):
    dtype_name = "fp32" if math_dtype == torch.float32 else "fp16"
    state_name = "zero" if zero_state else "nonzero"
    g_name = "realistic" if realistic_g else "random"
    print(f"\n{'='*90}")
    print(f"seq_len={seq_len:4d}  dtype={dtype_name}  state={state_name}  g={g_name}  "
          f"chunk={chunk_size}  heads={num_heads}  k={k_dim}  v={v_dim}")
    print(f"{'='*90}")

    torch.manual_seed(seed)
    query = torch.randn(batch, seq_len, num_heads, k_dim)
    key = torch.randn(batch, seq_len, num_heads, k_dim)
    value = torch.randn(batch, seq_len, num_heads, v_dim)
    if realistic_g:
        g = -torch.rand(batch, seq_len, num_heads).abs() * 1.5 - 0.1
    else:
        g = torch.randn(batch, seq_len, num_heads) * 0.5
    beta = torch.rand(batch, seq_len, num_heads)

    if zero_state:
        initial_state = torch.zeros(batch, num_heads, k_dim, v_dim)
    else:
        initial_state = torch.randn(batch, num_heads, k_dim, v_dim) * 0.01

    out_rec, state_rec = _recurrent(
        query.clone(), key.clone(), value.clone(), g.clone(), beta.clone(),
        recurrent_state=initial_state.clone(),
        output_final_state=True,
        expected_batch_size=batch,
        expected_num_heads=num_heads,
        expected_seq_len=seq_len,
        expected_k_dim=k_dim,
        expected_v_dim=v_dim,
        math_dtype=math_dtype,
    )

    try:
        out_chk, state_chk = _chunk(
            query.clone(), key.clone(), value.clone(), g.clone(), beta.clone(),
            chunk_size=chunk_size,
            initial_state=initial_state.clone(),
            output_final_state=True,
            expected_batch_size=batch,
            expected_num_heads=num_heads,
            expected_seq_len=seq_len,
            expected_k_dim=k_dim,
            expected_v_dim=v_dim,
            math_dtype=math_dtype,
        )
    except Exception as e:
        print(f"  ** CHUNK ERROR: {e}")
        return None, None, None, None

    report("output", out_rec, out_chk)
    report("final_state", state_rec, state_chk)

    n_chunks = (seq_len + chunk_size - 1) // chunk_size
    for ci in range(min(n_chunks, 8)):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, seq_len)
        a = out_rec[:, lo:hi]
        b = out_chk[:, lo:hi]
        diff = (a.float() - b.float()).abs()
        cos = cosine_sim(a, b)
        print(f"    chunk {ci:2d} [{lo:4d}:{hi:4d}]  cos={cos:.10f}  "
              f"max_abs={diff.max().item():.6e}")

    return out_rec, out_chk, state_rec, state_chk


def main():
    B, NH, KD, VD, CS = 1, 32, 128, 128, 16

    test_cases = [
        (16,  "single chunk"),
        (48,  "3 chunks"),
        (50,  "3+2 padding"),
        (512, "32 chunks"),
    ]
    dtypes = [
        (torch.float32, "fp32"),
        (torch.float16, "fp16"),
    ]
    states = [(True, "zero"), (False, "nonzero")]
    g_types = [(True, "realistic"), (False, "random")]

    results = []
    for seq_len, _ in test_cases:
        for dtype, dt_desc in dtypes:
            for zero, st_desc in states:
                for realistic, g_desc in g_types:
                    o_r, o_c, s_r, s_c = run_test(
                        B, NH, seq_len, KD, VD, CS, dtype, zero, realistic
                    )
                    if o_r is not None and o_c is not None:
                        oc = cosine_sim(o_r, o_c)
                        sc = cosine_sim(s_r, s_c)
                        om = (o_r.float() - o_c.float()).abs().max().item()
                        sm = (s_r.float() - s_c.float()).abs().max().item()
                        orl = rel_err(o_r, o_c)
                        srl = rel_err(s_r, s_c)
                    else:
                        oc = sc = om = sm = orl = srl = float('nan')
                    results.append((seq_len, dt_desc, st_desc, g_desc,
                                    oc, om, orl, sc, sm, srl))

    print(f"\n\n{'='*130}")
    print("SUMMARY")
    print(f"{'='*130}")
    print(f"{'seq':>5s}  {'dt':>4s}  {'state':>6s}  {'g':>9s}  "
          f"{'out_cos':>14s}  {'out_maxabs':>11s}  {'out_relmax':>11s}  "
          f"{'st_cos':>14s}  {'st_maxabs':>11s}  {'st_relmax':>11s}")
    print("-" * 130)
    for sl, dt, st, gd, oc, om, orl, sc, sm, srl in results:
        print(f"{sl:>5d}  {dt:>4s}  {st:>6s}  {gd:>9s}  "
              f"{oc:>14.10f}  {om:>11.4e}  {orl:>11.4e}  "
              f"{sc:>14.10f}  {sm:>11.4e}  {srl:>11.4e}")


if __name__ == "__main__":
    main()
