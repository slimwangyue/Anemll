#!/usr/bin/env python3
"""Check if the max_abs error is dominated by fp16 output quantization.
Compare fp32 internals before the to(initial_dtype) cast.
"""
import sys, os, torch, math
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anemll.models.qwen3_5_model import _l2norm


def compare(name, a, b, indent=2):
    a_f, b_f = a.float().flatten(), b.float().flatten()
    max_abs = (a_f - b_f).abs().max().item()
    mean_abs = (a_f - b_f).abs().mean().item()
    cos = F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
    print(f"{' '*indent}{name:50s} max={max_abs:.6e} mean={mean_abs:.6e} cos={cos:.8f}")
    return max_abs


def make_inputs(batch=1, n_heads=16, seq_len=64, k_dim=128, v_dim=128, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, n_heads, k_dim, dtype=torch.float16)
    k = torch.randn(batch, seq_len, n_heads, k_dim, dtype=torch.float16)
    v = torch.randn(batch, seq_len, n_heads, v_dim, dtype=torch.float16)
    g = torch.randn(batch, seq_len, n_heads, dtype=torch.float16) * 0.5
    beta = torch.sigmoid(torch.randn(batch, seq_len, n_heads, dtype=torch.float16))
    init_state = torch.randn(batch, n_heads, k_dim, v_dim, dtype=torch.float32) * 0.1
    return q, k, v, g, beta, init_state


def recurrent_fp32_out(query, key, value, g, beta, init_state, math_dtype=torch.float32):
    """Return output in fp32 (skip the to(initial_dtype) conversion)."""
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    query = _l2norm(query, dim=-1)
    key = _l2norm(key, dim=-1)
    bsz, n_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = 1 / (k_dim ** 0.5)
    query = query * scale
    state = init_state.to(query)
    out = torch.zeros(bsz, n_heads, seq_len, v_dim, dtype=query.dtype)
    for i in range(seq_len):
        q_t, k_t, v_t = query[:,:,i], key[:,:,i], value[:,:,i]
        g_t = g[:,:,i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:,:,i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:,:,i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    # Return in fp32, not converted to initial_dtype
    return out.transpose(1, 2).contiguous(), state


def chunked_fp32_out(query, key, value, g_raw, beta_raw, init_state,
                      chunk_size=16, math_dtype=torch.float32):
    """Return output in fp32 (skip the to(initial_dtype) conversion)."""
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta_raw, g_raw)
    ]
    query = _l2norm(query, dim=-1)
    key = _l2norm(key, dim=-1)
    B, H, S, k_dim = key.shape
    v_dim = value.shape[-1]
    chunk_size = min(chunk_size, S)
    pad_size = (chunk_size - S % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    T = S + pad_size
    scale = 1 / (k_dim ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    NC = T // chunk_size
    query = query.reshape(B, H, NC, chunk_size, k_dim)
    key = key.reshape(B, H, NC, chunk_size, k_dim)
    k_beta = k_beta.reshape(B, H, NC, chunk_size, k_dim)
    v_beta = v_beta.reshape(B, H, NC, chunk_size, v_dim)
    g = g.reshape(B, H, NC, chunk_size)

    tril_ones = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))
    strict_lower = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype), diagonal=-1)
    g = (tril_ones @ g.unsqueeze(-1)).squeeze(-1)
    decay_raw = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_ones
    decay_mask = decay_raw.exp() * tril_ones

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower
    attn_rows = [attn[..., 0:1, :]]
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = torch.cat([prev_row[..., :i] for prev_row in attn_rows[:i]], dim=-2)
        updated_row = row + (row.unsqueeze(-1) * sub).sum(-2)
        tail = attn[..., i : i + 1, i:]
        full_row = torch.cat([updated_row.unsqueeze(-2), tail], dim=-1)
        attn_rows.append(full_row)
    attn = torch.cat(attn_rows, dim=-2)
    attn = attn + torch.eye(chunk_size, dtype=math_dtype)
    value_resolved = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    lower_diag = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))

    state = init_state.to(math_dtype) if init_state is not None else torch.zeros(B, H, k_dim, v_dim, dtype=math_dtype)
    chunks_out = []
    for ci in range(NC):
        q_c, k_c, v_c = query[:,:,ci], key[:,:,ci], value_resolved[:,:,ci]
        kcd_c = k_cumdecay[:,:,ci]
        intra_attn = (q_c @ k_c.transpose(-1, -2) * decay_mask[:,:,ci]) * lower_diag
        v_prime = kcd_c @ state
        v_new = v_c - v_prime
        attn_inter = (q_c * g[:,:,ci,:,None].exp()) @ state
        chunks_out.append((attn_inter + intra_attn @ v_new).unsqueeze(2))
        g_last = g[:,:,ci,-1,None,None]
        k_decayed = k_c * (g[:,:,ci,-1,None] - g[:,:,ci]).exp()[..., None]
        state = state * g_last.exp() + k_decayed.transpose(-1, -2) @ v_new

    out = torch.cat(chunks_out, dim=2).reshape(B, H, T, v_dim)[:,:,:S]
    # Return in fp32
    return out.transpose(1, 2).contiguous(), state


if __name__ == "__main__":
    print("="*90)
    print("TEST 1: fp32 comparison (no fp16 output quantization)")
    print("="*90)
    
    for sl, nh, kd in [(64, 16, 128), (128, 16, 128), (256, 16, 128)]:
        q, k, v, g, beta, init = make_inputs(seq_len=sl, n_heads=nh, k_dim=kd, v_dim=kd)
        out_rec, st_rec = recurrent_fp32_out(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone())
        out_chu, st_chu = chunked_fp32_out(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone())
        
        print(f"\n  seq_len={sl}, n_heads={nh}, k_dim={kd}:")
        compare("fp32 output", out_chu, out_rec)
        compare("fp32 state", st_chu, st_rec)

    print(f"\n{'='*90}")
    print("TEST 2: fp16 output (with quantization)")
    print(f"{'='*90}")
    
    from anemll.models.qwen3_5_model import Qwen35LinearAttention
    for sl, nh, kd in [(64, 16, 128), (128, 16, 128)]:
        q, k, v, g, beta, init = make_inputs(seq_len=sl, n_heads=nh, k_dim=kd, v_dim=kd)
        out_rec, st_rec = Qwen35LinearAttention._recurrent_gated_delta_rule(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            recurrent_state=init.clone(), output_final_state=True
        )
        out_chu, st_chu = Qwen35LinearAttention._chunk_gated_delta_rule(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            chunk_size=16, initial_state=init.clone(), output_final_state=True
        )
        print(f"\n  seq_len={sl}, n_heads={nh}, k_dim={kd}:")
        compare("fp16 output", out_chu, out_rec)
        compare("fp32 state", st_chu.float(), st_rec.float())

    print(f"\n{'='*90}")
    print("TEST 3: Where does the max error token occur? (fp16)")
    print(f"{'='*90}")
    
    q, k, v, g, beta, init = make_inputs(seq_len=64, n_heads=16, k_dim=128, v_dim=128)
    out_rec, _ = Qwen35LinearAttention._recurrent_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        recurrent_state=init.clone(), output_final_state=True
    )
    out_chu, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=16, initial_state=init.clone(), output_final_state=True
    )
    diff = (out_rec.float() - out_chu.float()).abs()
    max_val, max_idx = diff.view(-1).max(0)
    # Decode index
    total = out_rec.shape
    idx = max_idx.item()
    v_idx = idx % total[3]; idx //= total[3]
    s_idx = idx % total[2]; idx //= total[2]
    h_idx = idx % total[1]; idx //= total[1]
    b_idx = idx
    print(f"\n  Max error location: batch={b_idx}, head={h_idx}, seq={s_idx}, v_dim={v_idx}")
    print(f"  Max error: {max_val.item():.6e}")
    print(f"  Recurrent value: {out_rec[b_idx, s_idx, h_idx, v_idx].item():.6f}")
    print(f"  Chunked value:   {out_chu[b_idx, s_idx, h_idx, v_idx].item():.6f}")
    
    # Per-token error histogram
    per_token_max = diff.max(dim=-1).values.max(dim=-1).values  # (B, S)
    print(f"\n  Per-token max error (first 64 tokens):")
    for t in range(min(64, per_token_max.shape[1])):
        bar = "#" * min(int(per_token_max[0, t].item() * 3200), 50)
        print(f"    t={t:3d}: {per_token_max[0,t].item():.6e}  {bar}")
