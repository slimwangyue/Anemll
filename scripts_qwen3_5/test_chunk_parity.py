#!/usr/bin/env python3
"""Minimal parity test: _chunk_gated_delta_rule vs _recurrent_gated_delta_rule.

Tests at the function level with synthetic inputs. No model loading, no ANE.
"""
import sys, os, torch, math
import torch.nn.functional as F
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anemll.models.qwen3_5_model import Qwen35LinearAttention

# ── Helpers ──────────────────────────────────────────────────────
def make_inputs(batch=1, n_heads=2, seq_len=16, k_dim=16, v_dim=16, seed=42):
    """Create random inputs in the shape expected by both functions.
    Input layout: (batch, seq_len, n_heads * dim) for q/k/v/g/beta
    But the functions expect (batch, seq_len, n_heads, dim) after internal reshape.
    Actually looking at the code, inputs are (batch, seq_len, n_heads, dim) already
    transposed to (batch, n_heads, seq_len, dim) inside. Let me match what comes in.
    """
    torch.manual_seed(seed)
    # Both functions do x.transpose(1,2) internally, so input is (B, S, H, D)
    # Actually, looking more carefully: q/k/v are (B, S, H, D), g/beta are (B, S, H, 1) or (B, S, H)
    # Let me trace the shapes...
    # After transpose(1,2): (B, H, S, D) for q/k/v, (B, H, S) for g (squeezed), (B, H, S) for beta
    
    # The simplest: provide (B, H, S, D) and skip the transpose by calling after internal reshape
    # Actually the functions do the transpose themselves. So inputs should be (B, seq_len, n_heads, dim).
    # But g and beta... let me check: g is same shape as query in args.
    
    # From the code:
    # query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(math_dtype) for x in ...]
    # So all 5 inputs have the same first 3 dims: (B, S, H, ...)
    # After transpose: (B, H, S, ...)
    
    # query: (B, S, H, k_dim) -> (B, H, S, k_dim)
    # key: (B, S, H, k_dim) -> (B, H, S, k_dim) 
    # value: (B, S, H, v_dim) -> (B, H, S, v_dim)
    # g: (B, S, H, 1)? No... after transpose g is used as g[..., i] giving (B, H) per token
    # Wait, g is reshaped to (B, H, n_chunks, chunk_size) -- so after transpose it's (B, H, S)
    # That means g input must be (B, S, H) -- 3D tensor
    
    # beta: same pattern as g, (B, S, H) 3D
    
    # Let me check: in _recurrent, after transpose, g[:, :, i] gives (B, H) shape.
    # g[:,:,i].exp().unsqueeze(-1).unsqueeze(-1) -> (B, H, 1, 1) -- yes, g is (B, H, S) after transpose
    # So g input is (B, S, H) 3D.
    
    # Similarly beta[:,:,i].unsqueeze(-1) -> (B, H, 1) -- beta is (B, H, S) after transpose
    # So beta input is (B, S, H) 3D.
    
    # But wait -- the transpose is x.transpose(1,2) for ALL of them.
    # If g is (B, S, H), transpose(1,2) gives (B, H, S) -- correct!
    # If query is (B, S, H, k_dim), transpose(1,2) gives (B, H, S, k_dim) -- correct!
    
    # So g and beta are 3D: (B, S, H)
    # query, key are 4D: (B, S, H, k_dim) 
    # value is 4D: (B, S, H, v_dim)
    
    query = torch.randn(batch, seq_len, n_heads, k_dim, dtype=torch.float16)
    key = torch.randn(batch, seq_len, n_heads, k_dim, dtype=torch.float16)
    value = torch.randn(batch, seq_len, n_heads, v_dim, dtype=torch.float16)
    # g should be moderate-sized for realistic gating
    g = torch.randn(batch, seq_len, n_heads, dtype=torch.float16) * 0.5
    # beta in [0, 1]
    beta = torch.sigmoid(torch.randn(batch, seq_len, n_heads, dtype=torch.float16))
    
    initial_state = torch.randn(batch, n_heads, k_dim, v_dim, dtype=torch.float32) * 0.1
    
    return query, key, value, g, beta, initial_state


def compare(name, a, b):
    """Compare two tensors, print metrics."""
    a_f, b_f = a.float(), b.float()
    max_abs = (a_f - b_f).abs().max().item()
    mean_abs = (a_f - b_f).abs().mean().item()
    # cosine similarity (flatten)
    a_flat = a_f.flatten()
    b_flat = b_f.flatten()
    cos = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    # relative error
    denom = b_f.abs().mean().item() + 1e-10
    rel_err = mean_abs / denom
    print(f"  {name:30s}  cos={cos:.8f}  max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}  rel={rel_err:.6e}")
    return max_abs, mean_abs, cos


def run_test(seq_len, chunk_size, n_heads=2, k_dim=16, v_dim=16, 
             with_initial_state=True, math_dtype=torch.float32, seed=42):
    """Run one comparison."""
    q, k, v, g, beta, init_state = make_inputs(
        batch=1, n_heads=n_heads, seq_len=seq_len, k_dim=k_dim, v_dim=v_dim, seed=seed
    )
    
    state_for_rec = init_state.clone() if with_initial_state else torch.zeros(1, n_heads, k_dim, v_dim)
    state_for_chunk = init_state.clone() if with_initial_state else None
    
    # Recurrent (reference)
    out_rec, state_rec = Qwen35LinearAttention._recurrent_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        recurrent_state=state_for_rec.clone(),
        output_final_state=True,
        math_dtype=math_dtype,
    )
    
    # Chunked
    out_chunk, state_chunk = Qwen35LinearAttention._chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=chunk_size,
        initial_state=state_for_chunk.clone() if state_for_chunk is not None else None,
        output_final_state=True,
        math_dtype=math_dtype,
    )
    
    print(f"\n{'='*80}")
    print(f"seq_len={seq_len}, chunk_size={chunk_size}, init_state={with_initial_state}, "
          f"dtype={math_dtype}, n_heads={n_heads}, k_dim={k_dim}, v_dim={v_dim}")
    print(f"{'='*80}")
    
    compare("output", out_chunk, out_rec)
    compare("final_state", state_chunk.float(), state_rec.float())
    
    return out_rec, out_chunk, state_rec, state_chunk


# ── Also build a "reference chunked" that does the SAME algorithm as recurrent ──
# but structured in chunks, to isolate whether the issue is chunking math vs implementation
def reference_chunked(query, key, value, g, beta, chunk_size, initial_state=None, math_dtype=torch.float32):
    """Process tokens in chunks but use the recurrent formula within each chunk.
    This should be BIT-IDENTICAL to the recurrent version.
    Used to confirm the chunked framing itself doesn't cause issues.
    """
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    from anemll.models.qwen3_5_model import _l2norm
    query = _l2norm(query, dim=-1)
    key = _l2norm(key, dim=-1)
    
    bsz, n_heads, seq_len = query.shape[:3]
    k_dim = key.shape[-1]
    v_dim = value.shape[-1]
    scale = 1 / (k_dim ** 0.5)
    query = query * scale
    
    state = initial_state.to(query) if initial_state is not None else torch.zeros(bsz, n_heads, k_dim, v_dim, dtype=query.dtype, device=query.device)
    out = torch.zeros(bsz, n_heads, seq_len, v_dim, dtype=query.dtype, device=query.device)
    
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        for t in range(chunk_start, chunk_end):
            q_t = query[:, :, t]
            k_t = key[:, :, t]
            v_t = value[:, :, t]
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, t].unsqueeze(-1)
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*80)
    print("PHASE 1: Sanity — reference_chunked vs recurrent (should be ~identical)")
    print("="*80)
    
    q, k, v, g, beta, init = make_inputs(seq_len=16)
    out_rec, st_rec = Qwen35LinearAttention._recurrent_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        recurrent_state=init.clone(), output_final_state=True, math_dtype=torch.float32
    )
    out_ref, st_ref = reference_chunked(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=4, initial_state=init.clone(), math_dtype=torch.float32
    )
    compare("ref_chunked vs recurrent OUT", out_ref, out_rec)
    compare("ref_chunked vs recurrent STATE", st_ref, st_rec)

    print("\n" + "="*80)
    print("PHASE 2: Chunked vs Recurrent — varying seq_len and chunk_size")
    print("="*80)
    
    for seq_len in [1, 2, 4, 8, 16, 32, 64]:
        for chunk_size in [4, 8, 16]:
            if chunk_size > seq_len:
                continue
            run_test(seq_len, chunk_size, with_initial_state=True, math_dtype=torch.float32)
    
    print("\n" + "="*80)
    print("PHASE 3: Effect of no initial state")
    print("="*80)
    
    for seq_len in [16, 32]:
        run_test(seq_len, 16, with_initial_state=False, math_dtype=torch.float32)

    print("\n" + "="*80)
    print("PHASE 4: Realistic dims (Qwen3.5-4B: k_dim=128, v_dim=128, n_heads=16)")
    print("="*80)
    
    for seq_len in [16, 32, 64]:
        run_test(seq_len, 16, n_heads=16, k_dim=128, v_dim=128, 
                 with_initial_state=True, math_dtype=torch.float32)

    print("\n" + "="*80)
    print("PHASE 5: fp16 math (as used on ANE)")
    print("="*80)
    
    for seq_len in [16, 32]:
        run_test(seq_len, 16, n_heads=2, k_dim=16, v_dim=16,
                 with_initial_state=True, math_dtype=torch.float16)
