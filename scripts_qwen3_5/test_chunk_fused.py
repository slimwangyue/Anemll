#!/usr/bin/env python3
"""Test the fused-RHS reformulation for better numerical parity.

Current code:  v_new = (inv @ v_beta) - (inv @ K_bg) @ state  (cancellation AFTER inv amplification)
Proposed:      v_new = inv @ (v_beta - K_bg @ state)           (cancellation BEFORE inv amplification)

Also test: solving (I+L) @ delta = rhs by forward substitution instead of explicit inverse.
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
    denom = b_f.abs().mean().item() + 1e-10
    rel = mean_abs / denom
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


def recurrent_reference(query, key, value, g, beta, init_state, math_dtype=torch.float32):
    """The token-by-token ground truth."""
    initial_dtype = query.dtype
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
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state


def chunked_original(query, key, value, g_raw, beta_raw, init_state,
                     chunk_size=16, math_dtype=torch.float32):
    """Current implementation (precomputed inv @ v_beta and inv @ K_bg)."""
    initial_dtype = query.dtype
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
    value_orig = value.reshape(B, H, NC, chunk_size, v_dim)
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

    # ORIGINAL: precompute resolved terms
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
    return out.transpose(1, 2).contiguous().to(initial_dtype), state


def chunked_fused_rhs(query, key, value, g_raw, beta_raw, init_state,
                       chunk_size=16, math_dtype=torch.float32):
    """PROPOSED FIX: Fuse v_beta - K_bg @ state BEFORE applying inverse.
    
    Instead of precomputing inv @ v_beta and inv @ K_bg separately,
    compute delta = inv @ (v_beta - K_bg @ state) per chunk.
    This puts cancellation BEFORE inv amplification.
    """
    initial_dtype = query.dtype
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
    inv_mat = torch.cat(attn_rows, dim=-2)
    inv_mat = inv_mat + torch.eye(chunk_size, dtype=math_dtype)

    # Precompute K_beta_exp_g (NOT multiplied by inv)
    K_bg = k_beta * g.exp().unsqueeze(-1)  # (B, H, NC, C, k_dim)
    lower_diag = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))

    state = init_state.to(math_dtype) if init_state is not None else torch.zeros(B, H, k_dim, v_dim, dtype=math_dtype)
    chunks_out = []
    for ci in range(NC):
        q_c, k_c = query[:,:,ci], key[:,:,ci]
        vb_c = v_beta[:,:,ci]            # (B, H, C, v_dim)
        kbg_c = K_bg[:,:,ci]             # (B, H, C, k_dim)

        # FUSED RHS: cancellation happens here, BEFORE inv amplification
        rhs = vb_c - kbg_c @ state       # (B, H, C, v_dim)
        v_new = inv_mat[:,:,ci] @ rhs     # (B, H, C, v_dim)

        # Intra-chunk attention (original, unchanged)
        intra_attn = (q_c @ k_c.transpose(-1, -2) * decay_mask[:,:,ci]) * lower_diag
        # Inter-chunk output
        attn_inter = (q_c * g[:,:,ci,:,None].exp()) @ state
        chunks_out.append((attn_inter + intra_attn @ v_new).unsqueeze(2))

        # State update
        g_last = g[:,:,ci,-1,None,None]
        k_decayed = k_c * (g[:,:,ci,-1,None] - g[:,:,ci]).exp()[..., None]
        state = state * g_last.exp() + k_decayed.transpose(-1, -2) @ v_new

    out = torch.cat(chunks_out, dim=2).reshape(B, H, T, v_dim)[:,:,:S]
    return out.transpose(1, 2).contiguous().to(initial_dtype), state


def chunked_forward_sub(query, key, value, g_raw, beta_raw, init_state,
                         chunk_size=16, math_dtype=torch.float32):
    """PROPOSED FIX 2: Forward substitution solve instead of explicit inverse.
    
    Instead of computing inv matrix and multiplying, solve (I+L) @ delta = rhs
    directly by forward substitution. This avoids materializing the full inverse.
    """
    initial_dtype = query.dtype
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

    # Build the L matrix: L[j,i] = (k_beta_j . k_i) * decay(j,i), strict lower
    L = ((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower
    # K_beta_exp_g  
    K_bg = k_beta * g.exp().unsqueeze(-1)
    lower_diag = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))

    state = init_state.to(math_dtype) if init_state is not None else torch.zeros(B, H, k_dim, v_dim, dtype=math_dtype)
    chunks_out = []
    for ci in range(NC):
        q_c, k_c = query[:,:,ci], key[:,:,ci]
        vb_c = v_beta[:,:,ci]
        kbg_c = K_bg[:,:,ci]
        L_c = L[:,:,ci]              # (B, H, C, C)

        # Fused RHS
        rhs = vb_c - kbg_c @ state   # (B, H, C, v_dim)

        # Forward substitution: solve (I + L_c) @ delta = rhs
        # For each row j: delta[j] = rhs[j] - sum_{i<j} L_c[j,i] * delta[i]
        delta_rows = []
        for j in range(chunk_size):
            if j == 0:
                delta_j = rhs[:,:,0:1]  # (B, H, 1, v_dim)
            else:
                # L_c[:,:,j,:j] @ delta_stack[:,:,:j] — but delta_stack is (B,H,j,v_dim)
                delta_stack = torch.cat(delta_rows, dim=2)  # (B, H, j, v_dim)  
                correction = (L_c[:,:,j:j+1,:j] @ delta_stack)  # (B, H, 1, v_dim)
                delta_j = rhs[:,:,j:j+1] - correction
            delta_rows.append(delta_j)
        v_new = torch.cat(delta_rows, dim=2)  # (B, H, C, v_dim)

        # Intra-chunk attention
        intra_attn = (q_c @ k_c.transpose(-1, -2) * decay_mask[:,:,ci]) * lower_diag
        attn_inter = (q_c * g[:,:,ci,:,None].exp()) @ state
        chunks_out.append((attn_inter + intra_attn @ v_new).unsqueeze(2))

        # State update
        g_last = g[:,:,ci,-1,None,None]
        k_decayed = k_c * (g[:,:,ci,-1,None] - g[:,:,ci]).exp()[..., None]
        state = state * g_last.exp() + k_decayed.transpose(-1, -2) @ v_new

    out = torch.cat(chunks_out, dim=2).reshape(B, H, T, v_dim)[:,:,:S]
    return out.transpose(1, 2).contiguous().to(initial_dtype), state


if __name__ == "__main__":
    from anemll.models.qwen3_5_model import Qwen35LinearAttention

    configs = [
        ("small", dict(n_heads=4, k_dim=64, v_dim=64, seq_len=64)),
        ("realistic", dict(n_heads=16, k_dim=128, v_dim=128, seq_len=64)),
        ("long_real", dict(n_heads=16, k_dim=128, v_dim=128, seq_len=128)),
    ]

    for name, cfg in configs:
        print(f"\n{'='*90}")
        print(f"CONFIG: {name} — n_heads={cfg['n_heads']}, k={cfg['k_dim']}, v={cfg['v_dim']}, seq={cfg['seq_len']}")
        print(f"{'='*90}")

        q, k, v, g, beta, init = make_inputs(
            batch=1, n_heads=cfg["n_heads"], seq_len=cfg["seq_len"],
            k_dim=cfg["k_dim"], v_dim=cfg["v_dim"]
        )

        out_rec, st_rec = recurrent_reference(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone()
        )
        out_orig, st_orig = chunked_original(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone()
        )
        out_fused, st_fused = chunked_fused_rhs(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone()
        )
        out_fsub, st_fsub = chunked_forward_sub(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone()
        )

        print("\n  --- Output vs Recurrent ---")
        m1 = compare("ORIGINAL chunked", out_orig, out_rec)
        m2 = compare("FUSED RHS chunked", out_fused, out_rec)
        m3 = compare("FORWARD SUB chunked", out_fsub, out_rec)
        print(f"\n  IMPROVEMENT: fused={m1/m2:.2f}x, fsub={m1/m3:.2f}x (higher is better)")

        print("\n  --- Final State vs Recurrent ---")
        s1 = compare("ORIGINAL state", st_orig.float(), st_rec.float())
        s2 = compare("FUSED RHS state", st_fused.float(), st_rec.float())
        s3 = compare("FORWARD SUB state", st_fsub.float(), st_rec.float())
        print(f"\n  IMPROVEMENT: fused={s1/s2:.2f}x, fsub={s1/s3:.2f}x (higher is better)")
