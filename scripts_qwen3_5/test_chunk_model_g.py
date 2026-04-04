#!/usr/bin/env python3
"""Definitive parity test with model-realistic g values.

Qwen3.5 computes: g = -A_log.exp() * softplus(a + dt_bias)
g is ALWAYS NEGATIVE (pure decay). Test with realistic negative g distributions.
"""
import sys, os, torch, math
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anemll.models.qwen3_5_model import Qwen35LinearAttention


def compare(name, a, b, indent=2):
    a_f, b_f = a.float().flatten(), b.float().flatten()
    max_abs = (a_f - b_f).abs().max().item()
    mean_abs = (a_f - b_f).abs().mean().item()
    cos = F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
    print(f"{' '*indent}{name:60s} max={max_abs:.6e} mean={mean_abs:.6e}")
    return max_abs


def run(seq_len, chunk_size, n_heads, k_dim, v_dim, g_mode, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(1, seq_len, n_heads, k_dim, dtype=torch.float16)
    k = torch.randn(1, seq_len, n_heads, k_dim, dtype=torch.float16)
    v = torch.randn(1, seq_len, n_heads, v_dim, dtype=torch.float16)
    
    if g_mode == "model_typical":
        # Simulate g = -A_log.exp() * softplus(a + dt_bias)
        # A_log ~ U(-2, 0) => A_log.exp() ~ U(0.14, 1.0) 
        # a ~ N(0, 1), dt_bias ~ 0.5 => softplus(a+0.5) ~ 0.5-3.0
        A_log_exp = 0.14 + 0.86 * torch.rand(1, 1, n_heads, dtype=torch.float16)
        a = torch.randn(1, seq_len, n_heads, dtype=torch.float16)
        g = -(A_log_exp * F.softplus(a + 0.5)).to(torch.float16)
    elif g_mode == "model_small_decay":
        A_log_exp = 0.05 + torch.rand(1, 1, n_heads, dtype=torch.float16) * 0.1
        a = torch.randn(1, seq_len, n_heads, dtype=torch.float16) * 0.5
        g = -(A_log_exp * F.softplus(a + 0.3)).to(torch.float16)
    elif g_mode == "model_strong_decay":
        A_log_exp = 0.5 + torch.rand(1, 1, n_heads, dtype=torch.float16) * 0.5
        a = torch.randn(1, seq_len, n_heads, dtype=torch.float16)
        g = -(A_log_exp * F.softplus(a + 1.0)).to(torch.float16)
    elif g_mode == "near_zero":
        g = -torch.rand(1, seq_len, n_heads, dtype=torch.float16) * 0.01
    else:
        raise ValueError(f"Unknown g_mode: {g_mode}")
    
    beta = torch.sigmoid(torch.randn(1, seq_len, n_heads, dtype=torch.float16))
    init = torch.randn(1, n_heads, k_dim, v_dim, dtype=torch.float32) * 0.1
    
    # Print g statistics
    g_f = g.float()
    print(f"    g stats: mean={g_f.mean().item():.4f} std={g_f.std().item():.4f} "
          f"min={g_f.min().item():.4f} max={g_f.max().item():.4f}")
    # Cumulative g per chunk (sum of 16 g values)
    if seq_len >= chunk_size:
        g_reshape = g_f[0, :seq_len//chunk_size*chunk_size].reshape(-1, chunk_size, n_heads)
        g_cum_last = g_reshape.sum(dim=1)  # sum per chunk
        exp_g_cum = g_cum_last.exp()
        print(f"    exp(g_cum) per chunk: mean={exp_g_cum.mean().item():.4f} "
              f"max={exp_g_cum.max().item():.4f} min={exp_g_cum.min().item():.6f}")

    out_rec, st_rec = Qwen35LinearAttention._recurrent_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        recurrent_state=init.clone(), output_final_state=True
    )
    out_chu, st_chu = Qwen35LinearAttention._chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=chunk_size, initial_state=init.clone(), output_final_state=True
    )
    m = compare(f"output  ({g_mode})", out_chu, out_rec)
    s = compare(f"state   ({g_mode})", st_chu.float(), st_rec.float())
    return m, s


if __name__ == "__main__":
    NH, KD, VD = 16, 128, 128  # Qwen3.5-4B dims
    
    print("="*90)
    print("MODEL-REALISTIC g VALUES (always negative)")
    print("="*90)
    
    for g_mode in ["model_typical", "model_small_decay", "model_strong_decay", "near_zero"]:
        print(f"\n{'─'*90}")
        print(f"g_mode = {g_mode}")
        print(f"{'─'*90}")
        for sl in [64, 128, 256, 512]:
            print(f"\n  seq_len={sl} ({sl//16} chunks):")
            run(sl, 16, NH, KD, VD, g_mode)
    
    print(f"\n{'='*90}")
    print("EXTREME STRESS TEST: seq=1024, model-typical g")
    print(f"{'='*90}")
    run(1024, 16, NH, KD, VD, "model_typical")
    
    print(f"\n{'='*90}")
    print("COMPARISON: fp32 output (no fp16 quantization), seq=512, model_typical")
    print(f"{'='*90}")
    from anemll.models.qwen3_5_model import _l2norm
    torch.manual_seed(42)
    sl, cs = 512, 16
    q = torch.randn(1, sl, NH, KD, dtype=torch.float16)
    k = torch.randn(1, sl, NH, KD, dtype=torch.float16)
    v = torch.randn(1, sl, NH, VD, dtype=torch.float16)
    A_log_exp = 0.14 + 0.86 * torch.rand(1, 1, NH, dtype=torch.float16)
    a = torch.randn(1, sl, NH, dtype=torch.float16)
    g = -(A_log_exp * F.softplus(a + 0.5)).to(torch.float16)
    beta = torch.sigmoid(torch.randn(1, sl, NH, dtype=torch.float16))
    init = torch.randn(1, NH, KD, VD, dtype=torch.float32) * 0.1
    
    # Get fp32 internal results (before to(initial_dtype))
    # Run recurrent
    qr, kr, vr, br, gr = [x.transpose(1,2).contiguous().float() for x in (q.clone(), k.clone(), v.clone(), beta.clone(), g.clone())]
    qr = _l2norm(qr, dim=-1) * (1/KD**0.5); kr = _l2norm(kr, dim=-1)
    state = init.clone()
    out_rec_fp32 = torch.zeros(1, NH, sl, VD)
    for i in range(sl):
        qt, kt, vt = qr[:,:,i], kr[:,:,i], vr[:,:,i]
        gt = gr[:,:,i].exp().unsqueeze(-1).unsqueeze(-1)
        bt = br[:,:,i].unsqueeze(-1)
        state = state * gt
        kv_mem = (state * kt.unsqueeze(-1)).sum(dim=-2)
        delta = (vt - kv_mem) * bt
        state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        out_rec_fp32[:,:,i] = (state * qt.unsqueeze(-1)).sum(dim=-2)
    out_rec_fp32 = out_rec_fp32.transpose(1,2).contiguous()
    
    # Run chunked (get fp16 output from the class method)
    out_chu, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=cs, initial_state=init.clone(), output_final_state=True
    )
    
    compare("chunked(fp16) vs recurrent(fp32)", out_chu, out_rec_fp32)
    # Also compare fp16 of recurrent vs fp32 of recurrent to see quantization effect
    out_rec_fp16, _ = Qwen35LinearAttention._recurrent_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        recurrent_state=init.clone(), output_final_state=True
    )
    compare("recurrent(fp16) vs recurrent(fp32) [quantization]", out_rec_fp16, out_rec_fp32)
    compare("chunked(fp16) vs recurrent(fp16)", out_chu, out_rec_fp16)
