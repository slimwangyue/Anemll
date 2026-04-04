#!/usr/bin/env python3
"""Deep diagnostic: compare each intermediate value of _chunk_gated_delta_rule
against the ground truth computed step-by-step from the recurrent formula.

Goal: Find which computation introduces the most error per chunk.
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
    print(f"{' '*indent}{name:45s} max={max_abs:.6e} mean={mean_abs:.6e} cos={cos:.8f} rel={rel:.6e}")
    return max_abs


def make_inputs(batch=1, n_heads=4, seq_len=16, k_dim=64, v_dim=64, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, n_heads, k_dim, dtype=torch.float16)
    k = torch.randn(batch, seq_len, n_heads, k_dim, dtype=torch.float16)
    v = torch.randn(batch, seq_len, n_heads, v_dim, dtype=torch.float16)
    g = torch.randn(batch, seq_len, n_heads, dtype=torch.float16) * 0.5
    beta = torch.sigmoid(torch.randn(batch, seq_len, n_heads, dtype=torch.float16))
    init_state = torch.randn(batch, n_heads, k_dim, v_dim, dtype=torch.float32) * 0.1
    return q, k, v, g, beta, init_state


def recurrent_step_by_step(query, key, value, g, beta, init_state, math_dtype=torch.float32):
    """Token-by-token processing, return all intermediate states."""
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
    states = [state.clone()]  # state before any processing
    deltas = []
    outputs = []

    for i in range(seq_len):
        q_t = query[:, :, i]       # (B, H, k_dim)
        k_t = key[:, :, i]         # (B, H, k_dim)
        v_t = value[:, :, i]       # (B, H, v_dim)
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        beta_t = beta[:, :, i].unsqueeze(-1)                 # (B, H, 1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)     # (B, H, v_dim)
        delta = (v_t - kv_mem) * beta_t                       # (B, H, v_dim)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        deltas.append(delta)
        outputs.append(out)
        states.append(state.clone())

    return {
        "query": query, "key": key, "value": value, "g": g, "beta": beta,
        "states": states, "deltas": deltas, "outputs": outputs,
    }


def chunked_step_by_step(query, key, value, g_raw, beta_raw, init_state,
                          chunk_size=16, math_dtype=torch.float32):
    """Reproduce _chunk_gated_delta_rule internals and return all intermediates."""
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta_raw, g_raw)
    ]
    query = _l2norm(query, dim=-1)
    key = _l2norm(key, dim=-1)
    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]

    chunk_size = min(chunk_size, seq_len)
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_len = seq_len + pad_size
    scale = 1 / (k_dim ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    n_chunks = total_len // chunk_size
    query = query.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim)
    key = key.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim)
    value_orig = value.reshape(batch_size, num_heads, n_chunks, chunk_size, v_dim)
    k_beta = k_beta.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim)
    v_beta = v_beta.reshape(batch_size, num_heads, n_chunks, chunk_size, v_dim)
    g = g.reshape(batch_size, num_heads, n_chunks, chunk_size)

    # Masks
    tril_ones = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))
    strict_lower = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype), diagonal=-1)

    # Cumulative g
    g_cum = (tril_ones @ g.unsqueeze(-1)).squeeze(-1)
    decay_raw = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)) * tril_ones
    decay_mask = decay_raw.exp() * tril_ones

    # A matrix
    A_raw = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower
    # Woodbury forward substitution
    attn_rows = [A_raw[..., 0:1, :]]
    for i in range(1, chunk_size):
        row = A_raw[..., i, :i].clone()
        sub = torch.cat([prev_row[..., :i] for prev_row in attn_rows[:i]], dim=-2)
        updated_row = row + (row.unsqueeze(-1) * sub).sum(-2)
        tail = A_raw[..., i : i + 1, i:]
        full_row = torch.cat([updated_row.unsqueeze(-2), tail], dim=-1)
        attn_rows.append(full_row)
    woodbury = torch.cat(attn_rows, dim=-2)
    inv_IpA = woodbury + torch.eye(chunk_size, dtype=math_dtype)

    # Resolved values
    value_resolved = inv_IpA @ v_beta    # (I+A)^{-1} @ v_beta
    k_cumdecay = inv_IpA @ (k_beta * g_cum.exp().unsqueeze(-1))
    lower_diag = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))

    state = init_state.to(math_dtype) if init_state is not None else torch.zeros(batch_size, num_heads, k_dim, v_dim, dtype=math_dtype)
    chunk_outputs = []
    chunk_states = [state.clone()]
    chunk_v_news = []

    for ci in range(n_chunks):
        q_c = query[:, :, ci]          # (B, H, C, k_dim)
        k_c = key[:, :, ci]            # (B, H, C, k_dim)
        v_c = value_resolved[:, :, ci] # (B, H, C, v_dim) resolved
        kcd_c = k_cumdecay[:, :, ci]   # (B, H, C, k_dim)

        # Intra-chunk attention
        intra_attn = (q_c @ k_c.transpose(-1, -2) * decay_mask[:, :, ci]) * lower_diag

        # Inter-chunk: state contribution to resolved deltas
        v_prime = kcd_c @ state  # (B, H, C, v_dim)
        v_new = v_c - v_prime

        # Inter-chunk: state contribution to output
        attn_inter = (q_c * g_cum[:, :, ci, :, None].exp()) @ state

        # Output for this chunk
        out_c = attn_inter + intra_attn @ v_new
        chunk_outputs.append(out_c)
        chunk_v_news.append(v_new)

        # State update
        g_last = g_cum[:, :, ci, -1, None, None]  # (B, H, 1, 1)
        k_decayed = k_c * (g_cum[:, :, ci, -1, None] - g_cum[:, :, ci]).exp()[..., None]  # (B, H, C, k_dim)
        state = state * g_last.exp() + k_decayed.transpose(-1, -2) @ v_new
        chunk_states.append(state.clone())

    return {
        "g_cum": g_cum,
        "decay_mask": decay_mask,
        "A_raw": A_raw,
        "woodbury": woodbury,
        "inv_IpA": inv_IpA,
        "value_resolved": value_resolved,
        "k_cumdecay": k_cumdecay,
        "chunk_outputs": chunk_outputs,
        "chunk_states": chunk_states,
        "chunk_v_news": chunk_v_news,
        "query": query,
        "key": key,
        "v_beta": v_beta,
        "k_beta": k_beta,
    }


def diagnose_single_chunk(seq_len=16, chunk_size=16, n_heads=4, k_dim=64, v_dim=64):
    """Compare intermediates for a single chunk (seq_len == chunk_size)."""
    print(f"\n{'='*80}")
    print(f"SINGLE CHUNK DIAGNOSTIC: seq={seq_len}, chunk={chunk_size}, heads={n_heads}, k={k_dim}, v={v_dim}")
    print(f"{'='*80}")

    q, k, v, g, beta, init = make_inputs(1, n_heads, seq_len, k_dim, v_dim)
    rec = recurrent_step_by_step(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone())
    chu = chunked_step_by_step(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone(), chunk_size)

    # 1. Verify shared preprocessing
    compare("query (after l2norm, scale)", chu["query"][:,:,0], rec["query"])
    compare("key (after l2norm)", chu["key"][:,:,0], rec["key"])

    # 2. Compare the A matrix against ground truth
    # Ground truth A[j,i] = (k_beta_j . k_i) * exp(g_cum_j - g_cum_i) for j > i
    k_rec = rec["key"]       # (B, H, S, k_dim)
    beta_rec = rec["beta"]   # depends on what recurrent returns...
    # Actually the recurrent dict doesn't store beta separately after transpose.
    # Let me just compute from the chunked quantities.

    # 3. Compare resolved deltas (v_new) against recurrent deltas
    print("\n  --- Resolved deltas per token vs recurrent deltas ---")
    rec_deltas = torch.stack(rec["deltas"], dim=2)  # (B, H, S, v_dim)
    chu_v_new = chu["chunk_v_news"][0]               # (B, H, C, v_dim)
    compare("resolved deltas (v_new vs rec deltas)", chu_v_new, rec_deltas)

    # Token-by-token delta comparison
    print("\n  --- Per-token delta ---")
    for t in range(min(seq_len, 8)):
        compare(f"  delta[{t}]", chu_v_new[:,:,t], rec_deltas[:,:,t])

    # 4. Compare outputs
    print("\n  --- Per-token output ---")
    rec_outs = torch.stack(rec["outputs"], dim=2)  # (B, H, S, v_dim)
    chu_outs = chu["chunk_outputs"][0]               # (B, H, C, v_dim)
    for t in range(min(seq_len, 8)):
        compare(f"  output[{t}]", chu_outs[:,:,t], rec_outs[:,:,t])

    # 5. Final state
    print("\n  --- Final state ---")
    compare("final_state", chu["chunk_states"][-1], rec["states"][-1])

    # 6. Woodbury accuracy: check (I+A) @ inv_IpA ≈ I
    A_raw = chu["A_raw"][:,:,0]  # (B, H, C, C)
    # A_raw = -((k_beta @ key.T) * decay_mask) * strict_lower = -A_lower
    # Woodbury computes (I - (-A_lower))^{-1} = (I + A_lower)^{-1}

    # Check inv_IpA accuracy
    inv_mat = chu["inv_IpA"][:,:,0]  # (B, H, C, C)
    eye = torch.eye(chunk_size, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    IpA_mat = eye - A_raw  # I + A_lower = I - (-A_lower) = I - A_raw
    product = inv_mat @ IpA_mat
    identity = eye
    print("\n  --- Woodbury inversion accuracy ---")
    compare("inv(I+A) @ (I+A) vs I", product, identity.expand_as(product))


def diagnose_multi_chunk(seq_len=64, chunk_size=16, n_heads=4, k_dim=64, v_dim=64):
    """Compare per-chunk state error accumulation."""
    print(f"\n{'='*80}")
    print(f"MULTI-CHUNK DIAGNOSTIC: seq={seq_len}, chunk={chunk_size}, heads={n_heads}, k={k_dim}, v={v_dim}")
    print(f"{'='*80}")

    q, k, v, g, beta, init = make_inputs(1, n_heads, seq_len, k_dim, v_dim)
    rec = recurrent_step_by_step(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone())
    chu = chunked_step_by_step(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone(), chunk_size)

    n_chunks = seq_len // chunk_size

    # Compare state after each chunk boundary
    print("\n  --- State error after each chunk ---")
    for ci in range(n_chunks):
        token_idx = (ci + 1) * chunk_size  # state after processing chunk ci
        rec_state = rec["states"][token_idx]
        chu_state = chu["chunk_states"][ci + 1]
        compare(f"state after chunk {ci} (token {token_idx})", chu_state, rec_state)

    # Compare outputs per chunk
    print("\n  --- Output error per chunk ---")
    rec_outs = torch.stack(rec["outputs"], dim=2)
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = start + chunk_size
        chu_out = chu["chunk_outputs"][ci]
        rec_out = rec_outs[:, :, start:end]
        compare(f"output chunk {ci} (tokens {start}-{end-1})", chu_out, rec_out)

    # Compare v_new (resolved deltas) per chunk
    print("\n  --- v_new (resolved deltas) error per chunk ---")
    rec_deltas = torch.stack(rec["deltas"], dim=2)
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = start + chunk_size
        chu_vnew = chu["chunk_v_news"][ci]
        rec_delta = rec_deltas[:, :, start:end]
        compare(f"v_new chunk {ci} (tokens {start}-{end-1})", chu_vnew, rec_delta)

    # Check: what if we use the recurrent state for each chunk?
    # This isolates whether error comes from state propagation vs intra-chunk computation
    print("\n  --- Intra-chunk error with CORRECT state (isolate state prop) ---")
    for ci in range(n_chunks):
        token_idx = ci * chunk_size  # state BEFORE this chunk
        rec_state_before = rec["states"][token_idx]
        # Recompute chunked output using the correct recurrent state
        q_c = chu["query"][:, :, ci]
        k_c = chu["key"][:, :, ci]
        v_c = chu["value_resolved"][:, :, ci]
        kcd_c = chu["k_cumdecay"][:, :, ci]
        g_cum = chu["g_cum"][:, :, ci]
        dm = chu["decay_mask"][:, :, ci]
        lower_diag = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32))

        intra_attn = (q_c @ k_c.transpose(-1, -2) * dm) * lower_diag
        v_prime = kcd_c @ rec_state_before.to(torch.float32)  # USE CORRECT STATE
        v_new = v_c - v_prime
        attn_inter = (q_c * g_cum[:, :, :, None].exp()) @ rec_state_before.to(torch.float32)
        out_corrected = attn_inter + intra_attn @ v_new

        start = ci * chunk_size
        end = start + chunk_size
        rec_out = rec_outs[:, :, start:end]
        compare(f"corrected_out chunk {ci} (tokens {start}-{end-1})", out_corrected, rec_out)


def diagnose_realistic(seq_len=64, chunk_size=16, n_heads=16, k_dim=128, v_dim=128):
    """Realistic Qwen3.5-4B dimensions."""
    print(f"\n{'='*80}")
    print(f"REALISTIC DIAGNOSTIC: seq={seq_len}, chunk={chunk_size}, heads={n_heads}, k={k_dim}, v={v_dim}")
    print(f"{'='*80}")

    q, k, v, g, beta, init = make_inputs(1, n_heads, seq_len, k_dim, v_dim)
    rec = recurrent_step_by_step(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone())
    chu = chunked_step_by_step(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), init.clone(), chunk_size)

    n_chunks = seq_len // chunk_size

    print("\n  --- State error after each chunk ---")
    for ci in range(n_chunks):
        token_idx = (ci + 1) * chunk_size
        compare(f"state after chunk {ci}", chu["chunk_states"][ci + 1], rec["states"][token_idx])

    print("\n  --- Output error per chunk ---")
    rec_outs = torch.stack(rec["outputs"], dim=2)
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = start + chunk_size
        compare(f"output chunk {ci}", chu["chunk_outputs"][ci], rec_outs[:, :, start:end])

    # g magnitude analysis
    g_cum = chu["g_cum"]
    print(f"\n  --- g_cum statistics ---")
    print(f"    g_cum range: [{g_cum.min().item():.4f}, {g_cum.max().item():.4f}]")
    print(f"    g_cum[-1] per chunk (state amplification):")
    for ci in range(n_chunks):
        g_last = g_cum[:, :, ci, -1]  # (B, H)
        print(f"      chunk {ci}: mean={g_last.mean().item():.4f} max={g_last.max().item():.4f} min={g_last.min().item():.4f}")
        print(f"        exp(g_last): mean={g_last.exp().mean().item():.4f} max={g_last.exp().max().item():.4f}")


if __name__ == "__main__":
    diagnose_single_chunk(seq_len=16, chunk_size=16, n_heads=4, k_dim=64, v_dim=64)
    diagnose_multi_chunk(seq_len=64, chunk_size=16, n_heads=4, k_dim=64, v_dim=64)
    diagnose_realistic(seq_len=64, chunk_size=16, n_heads=16, k_dim=128, v_dim=128)
