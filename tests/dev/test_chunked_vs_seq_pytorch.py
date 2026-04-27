#!/usr/bin/env python3
"""
Reproduce chunked-vs-sequential divergence of the gated delta rule in PyTorch.

Goal: Show that the SAME recurrence, computed via:
  (A) chunked parallel path  (prefill model)
  (B) sequential step-by-step (infer model)
...produces different outputs in float16 even though they are mathematically identical.

Uses the exact formulas from anemll/models/qwen3_5_model.py:
  _chunk_gated_delta_rule   (chunked/prefill)
  _recurrent_gated_delta_rule (sequential/infer)

We simulate 2 blocks of 256 tokens each (512 total), where block 2 is a "tail"
of 164 real tokens + 92 padding. Then compare:
  Test A: block1(chunked) → tail(sequential)   ← should match reference
  Test B: all 420 tokens sequential             ← reference
  Test C: block1(chunked) → tail(chunked)       ← expected to diverge in fp16
"""

import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

# ── Qwen3.5-4B linear attention dimensions ──
NUM_HEADS = 32      # v_heads
K_DIM     = 128     # head_k_dim (after l2norm)
V_DIM     = 128     # head_v_dim
CHUNK_SZ  = 32      # internal chunk size for chunked path
BS        = 256     # batch/block size for prefill
TAIL_LEN  = 164     # partial tail
TOTAL     = BS + TAIL_LEN  # 420 tokens


def _l2norm(x, dim=-1):
    return F.normalize(x, p=2, dim=dim)


def chunked_delta_rule(query, key, value, g, beta, chunk_size, initial_state, math_dtype):
    """Exact copy of _chunk_gated_delta_rule from qwen3_5_model.py"""
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    query = _l2norm(query, dim=-1)
    key   = _l2norm(key, dim=-1)

    batch_size, num_heads, seq_len, k_dim = query.shape
    v_dim = value.shape[-1]
    chunk_size = min(chunk_size, seq_len)
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key   = F.pad(key,   (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta  = F.pad(beta,  (0, pad_size))
    g     = F.pad(g,     (0, pad_size))
    total_seq = seq_len + pad_size
    scale = 1 / (k_dim ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key   * beta.unsqueeze(-1)
    n_chunks = total_seq // chunk_size
    query  = query.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim)
    key    = key.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim)
    value  = value.reshape(batch_size, num_heads, n_chunks, chunk_size, v_dim)
    k_beta = k_beta.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim)
    v_beta = v_beta.reshape(batch_size, num_heads, n_chunks, chunk_size, v_dim)
    g      = g.reshape(batch_size, num_heads, n_chunks, chunk_size)

    tril_ones    = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device, dtype=math_dtype))
    strict_lower = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device, dtype=math_dtype), diagonal=-1)

    g = (tril_ones @ g.unsqueeze(-1)).squeeze(-1)
    decay_raw  = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_ones
    decay_mask = decay_raw.exp() * tril_ones
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower

    # Block forward-substitution
    bc = min(16, chunk_size)
    n_blks = chunk_size // bc
    _blks = {}
    for _r in range(n_blks):
        for _c in range(_r + 1):
            _blks[(_r, _c)] = attn[..., _r*bc:(_r+1)*bc, _c*bc:(_c+1)*bc]
    _inv_d = {}
    for _b in range(n_blks):
        _A = _blks[(_b, _b)]
        _rows = [_A[..., 0:1, :]]
        for _i in range(1, bc):
            _row = _A[..., _i, :_i].clone()
            _sub = torch.cat([_pr[..., :_i] for _pr in _rows[:_i]], dim=-2)
            _urow = _row + (_row.unsqueeze(-1) * _sub).sum(-2)
            _tail = _A[..., _i:_i+1, _i:]
            _rows.append(torch.cat([_urow.unsqueeze(-2), _tail], dim=-1))
        _inv_d[_b] = torch.cat(_rows, dim=-2) + torch.eye(bc, dtype=attn.dtype, device=attn.device)
    _inv_f = {}
    for _b in range(n_blks):
        _inv_f[(_b, _b)] = _inv_d[_b]
    for _c in range(n_blks):
        for _r in range(_c + 1, n_blks):
            _acc = torch.zeros_like(_blks[(_r, _c)])
            for _m in range(_c, _r):
                _acc = _acc + _blks[(_r, _m)] @ _inv_f[(_m, _c)]
            _inv_f[(_r, _c)] = _inv_d[_r] @ _acc
    _result_rows = []
    for _r in range(n_blks):
        _rblks = []
        for _c in range(n_blks):
            if _c <= _r:
                _rblks.append(_inv_f[(_r, _c)])
            else:
                _rblks.append(torch.zeros_like(_blks[(_r, _r)]))
            _result_rows.append(torch.cat(_rblks, dim=-1))
    # Fix: collect per-row
    _result_rows_fixed = []
    for _r in range(n_blks):
        _rblks = []
        for _c in range(n_blks):
            if _c <= _r:
                _rblks.append(_inv_f[(_r, _c)])
            else:
                _rblks.append(torch.zeros_like(_blks[(_r, _r)]))
        _result_rows_fixed.append(torch.cat(_rblks, dim=-1))
    attn = torch.cat(_result_rows_fixed, dim=-2)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_state = initial_state.to(value) if initial_state is not None else torch.zeros(
        batch_size, num_heads, k_dim, v_dim, device=value.device, dtype=value.dtype
    )

    strict_lower_diag1 = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device, dtype=math_dtype))
    core_chunks = []
    for i in range(n_chunks):
        q_i, k_i, v_i = query[:,:,i], key[:,:,i], value[:,:,i]
        attn_i = (q_i @ k_i.transpose(-1,-2) * decay_mask[:,:,i]) * strict_lower_diag1
        v_prime = k_cumdecay[:,:,i] @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:,:,i,:,None].exp()) @ last_state
        core_chunks.append((attn_inter + attn_i @ v_new).unsqueeze(2))
        last_state = (
            last_state * g[:,:,i,-1,None,None].exp()
            + (k_i * (g[:,:,i,-1,None] - g[:,:,i]).exp()[...,None]).transpose(-1,-2) @ v_new
        )

    out = torch.cat(core_chunks, dim=2).reshape(batch_size, num_heads, total_seq, v_dim)
    out = out[:,:,:seq_len]
    out = out.transpose(1,2).contiguous().to(initial_dtype)
    return out, last_state


def recurrent_delta_rule(query, key, value, g, beta, recurrent_state, math_dtype):
    """Exact copy of _recurrent_gated_delta_rule from qwen3_5_model.py"""
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    query = _l2norm(query, dim=-1)
    key   = _l2norm(key, dim=-1)

    bsz, n_heads, seq_len, k_dim = query.shape
    v_dim = value.shape[-1]
    scale = 1 / (k_dim ** 0.5)
    query = query * scale

    out = torch.zeros(bsz, n_heads, seq_len, v_dim, dtype=value.dtype, device=value.device)
    state = recurrent_state.to(value)

    for i in range(seq_len):
        q_t    = query[:,:,i]
        k_t    = key[:,:,i]
        v_t    = value[:,:,i]
        g_t    = g[:,:,i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:,:,i].unsqueeze(-1)

        state  = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta  = (v_t - kv_mem) * beta_t
        state  = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:,:,i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    out = out.transpose(1,2).contiguous().to(initial_dtype)
    return out, state


def make_inputs(seq_len, seed=42):
    """Generate random Q, K, V, g, beta inputs for one linear attention layer."""
    torch.manual_seed(seed)
    # Shapes match what the layout stage produces:
    # query, key: (B=1, S, H, K_DIM)
    # value:      (B=1, S, H, V_DIM)
    # beta, g:    (B=1, S, H)   — 3D, per-head scalar
    q = torch.randn(1, seq_len, NUM_HEADS, K_DIM, dtype=torch.float16)
    k = torch.randn(1, seq_len, NUM_HEADS, K_DIM, dtype=torch.float16)
    v = torch.randn(1, seq_len, NUM_HEADS, V_DIM, dtype=torch.float16)
    beta = torch.sigmoid(torch.randn(1, seq_len, NUM_HEADS, dtype=torch.float16))
    # g should be negative (decay)
    g = -torch.abs(torch.randn(1, seq_len, NUM_HEADS, dtype=torch.float16)) * 0.1
    return q, k, v, g, beta


def run_test(math_dtype, label):
    """Run comparison test at given math precision."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  math_dtype={math_dtype}, inputs=float16, heads={NUM_HEADS}, k={K_DIM}, v={V_DIM}")
    print(f"  block1={BS} tokens, tail={TAIL_LEN} tokens, total={TOTAL}")
    print(f"{'='*70}")

    q, k, v, g, beta = make_inputs(TOTAL)
    S0 = torch.zeros(1, NUM_HEADS, K_DIM, V_DIM, dtype=math_dtype)

    # Split inputs
    q1, q2 = q[:, :BS], q[:, BS:]
    k1, k2 = k[:, :BS], k[:, BS:]
    v1, v2 = v[:, :BS], v[:, BS:]
    g1, g2 = g[:, :BS], g[:, BS:]
    b1, b2 = beta[:, :BS], beta[:, BS:]

    # ─── Test B: fully sequential (reference) ───
    out_B, state_B = recurrent_delta_rule(q, k, v, g, beta, S0.clone(), math_dtype)
    last_out_B = out_B[0, -1]  # last token output

    # ─── Block 1 via chunked ───
    out_c1, state_c1 = chunked_delta_rule(q1, k1, v1, g1, b1, CHUNK_SZ, S0.clone(), math_dtype)

    # ─── Block 1 via sequential ───
    out_s1, state_s1 = recurrent_delta_rule(q1, k1, v1, g1, b1, S0.clone(), math_dtype)

    # Compare block1 states
    state_cos = F.cosine_similarity(
        state_c1.reshape(1, -1).float(), state_s1.reshape(1, -1).float()
    ).item()
    state_mse = ((state_c1.float() - state_s1.float()) ** 2).mean().item()
    print(f"\nBlock1 state comparison (chunked vs sequential):")
    print(f"  cosine similarity: {state_cos:.8f}")
    print(f"  MSE:               {state_mse:.2e}")

    # ─── Test A: block1(chunked) → tail(sequential) ───
    out_A, state_A = recurrent_delta_rule(q2, k2, v2, g2, b2, state_c1.clone(), math_dtype)
    last_out_A = out_A[0, -1]

    # ─── Test C: block1(chunked) → tail(chunked) ───
    out_C, state_C = chunked_delta_rule(q2, k2, v2, g2, b2, CHUNK_SZ, state_c1.clone(), math_dtype)
    last_out_C = out_C[0, -1]

    # ─── Compare final outputs ───
    cos_AB = F.cosine_similarity(last_out_A.reshape(1,-1).float(), last_out_B.reshape(1,-1).float()).item()
    cos_CB = F.cosine_similarity(last_out_C.reshape(1,-1).float(), last_out_B.reshape(1,-1).float()).item()
    cos_AC = F.cosine_similarity(last_out_A.reshape(1,-1).float(), last_out_C.reshape(1,-1).float()).item()

    mse_AB = ((last_out_A.float() - last_out_B.float()) ** 2).mean().item()
    mse_CB = ((last_out_C.float() - last_out_B.float()) ** 2).mean().item()
    mse_AC = ((last_out_A.float() - last_out_C.float()) ** 2).mean().item()

    # State comparisons
    state_cos_AB = F.cosine_similarity(state_A.reshape(1,-1).float(), state_B.reshape(1,-1).float()).item()
    state_cos_CB = F.cosine_similarity(state_C.reshape(1,-1).float(), state_B.reshape(1,-1).float()).item()

    print(f"\nFinal output comparison (last token, {NUM_HEADS}x{V_DIM}={NUM_HEADS*V_DIM} dims):")
    print(f"  A (chunk+seq)  vs B (seq):     cos={cos_AB:.8f}  MSE={mse_AB:.2e}")
    print(f"  C (chunk+chunk) vs B (seq):    cos={cos_CB:.8f}  MSE={mse_CB:.2e}")
    print(f"  A (chunk+seq)  vs C (chunk+chunk): cos={cos_AC:.8f}  MSE={mse_AC:.2e}")

    print(f"\nFinal state comparison:")
    print(f"  A (chunk+seq)  vs B (seq):     cos={state_cos_AB:.8f}")
    print(f"  C (chunk+chunk) vs B (seq):    cos={state_cos_CB:.8f}")

    # Simulate logit decision: dot product with random "lm_head" weight
    torch.manual_seed(99)
    lm_head = torch.randn(100, NUM_HEADS * V_DIM, dtype=torch.float16)  # 100 "vocab" tokens
    logits_A = (lm_head.float() @ last_out_A.reshape(-1).float()).numpy()
    logits_B = (lm_head.float() @ last_out_B.reshape(-1).float()).numpy()
    logits_C = (lm_head.float() @ last_out_C.reshape(-1).float()).numpy()
    top_A, top_B, top_C = logits_A.argmax(), logits_B.argmax(), logits_C.argmax()

    print(f"\nSimulated top token (100-vocab toy LM head):")
    print(f"  B (fully seq):      token {top_B}  logit={logits_B[top_B]:.4f}")
    print(f"  A (chunk+seq tail): token {top_A}  logit={logits_A[top_A]:.4f}  match_B={top_A==top_B}")
    print(f"  C (chunk+chunk tail): token {top_C}  logit={logits_C[top_C]:.4f}  match_B={top_C==top_B}")

    # Show logit margin
    sorted_B = np.sort(logits_B)[::-1]
    margin_B = sorted_B[0] - sorted_B[1]
    diff_AC_top = abs(logits_A[top_B] - logits_C[top_B])
    print(f"\n  Top-1 vs Top-2 margin in B: {margin_B:.4f}")
    print(f"  |logit_A - logit_C| at B's top token: {diff_AC_top:.4f}")
    if diff_AC_top > margin_B * 0.5:
        print(f"  ⚠️  A-vs-C logit diff is >{margin_B*0.5:.4f} (50% of margin) → flip risk is HIGH")


def run_multilayer_test(math_dtype, n_layers, label):
    """Chain multiple linear attention layers to show compounding drift."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {n_layers} layers, math_dtype={math_dtype}, heads={NUM_HEADS}, k={K_DIM}, v={V_DIM}")
    print(f"  block1={BS} tokens, tail={TAIL_LEN} tokens, total={TOTAL}")
    print(f"{'='*70}")

    # Generate per-layer inputs (different random projections per layer)
    layer_inputs = []
    for layer_idx in range(n_layers):
        layer_inputs.append(make_inputs(TOTAL, seed=42 + layer_idx * 1000))

    # For multi-layer: output of layer N becomes "hidden" that gets projected into
    # layer N+1's Q/K/V. In a real model, there are dense projections between layers.
    # We simulate this by: output_hidden = prev_hidden + layer_output (residual),
    # then use output_hidden as a perturbation to the next layer's pre-generated Q/K/V.
    # This captures the key mechanism: drift in layer N's output amplifies through
    # residual connections into layer N+1's inputs.

    S0 = torch.zeros(1, NUM_HEADS, K_DIM, V_DIM, dtype=math_dtype)

    # ─── Path B: all sequential (reference) ───
    hidden_B = torch.zeros(1, TOTAL, NUM_HEADS * V_DIM, dtype=torch.float16)
    states_B = []
    for li in range(n_layers):
        q, k, v, g, beta = layer_inputs[li]
        # Perturb inputs by accumulated hidden state (simulates residual projection)
        q_pert = q + hidden_B.reshape(1, TOTAL, NUM_HEADS, V_DIM)[..., :K_DIM] * 0.1
        k_pert = k + hidden_B.reshape(1, TOTAL, NUM_HEADS, V_DIM)[..., :K_DIM] * 0.1
        v_pert = v + hidden_B.reshape(1, TOTAL, NUM_HEADS, V_DIM) * 0.1
        out, state = recurrent_delta_rule(q_pert, k_pert, v_pert, g, beta, S0.clone(), math_dtype)
        hidden_B = hidden_B + out.reshape(1, TOTAL, -1)  # residual
        states_B.append(state)
    last_B = hidden_B[0, -1]

    # ─── Path A: block1(chunked) → tail(sequential) ───
    hidden_A = torch.zeros(1, TOTAL, NUM_HEADS * V_DIM, dtype=torch.float16)
    for li in range(n_layers):
        q, k, v, g, beta = layer_inputs[li]
        q_pert = q + hidden_A.reshape(1, TOTAL, NUM_HEADS, V_DIM)[..., :K_DIM] * 0.1
        k_pert = k + hidden_A.reshape(1, TOTAL, NUM_HEADS, V_DIM)[..., :K_DIM] * 0.1
        v_pert = v + hidden_A.reshape(1, TOTAL, NUM_HEADS, V_DIM) * 0.1
        # Block 1: chunked
        out1, st1 = chunked_delta_rule(
            q_pert[:,:BS], k_pert[:,:BS], v_pert[:,:BS],
            g[:,:BS], beta[:,:BS], CHUNK_SZ, S0.clone(), math_dtype
        )
        # Tail: sequential
        out2, st2 = recurrent_delta_rule(
            q_pert[:,BS:], k_pert[:,BS:], v_pert[:,BS:],
            g[:,BS:], beta[:,BS:], st1.clone(), math_dtype
        )
        out_full = torch.cat([out1, out2], dim=1)
        hidden_A = hidden_A + out_full.reshape(1, TOTAL, -1)
    last_A = hidden_A[0, -1]

    # ─── Path C: block1(chunked) → tail(chunked) ───
    hidden_C = torch.zeros(1, TOTAL, NUM_HEADS * V_DIM, dtype=torch.float16)
    for li in range(n_layers):
        q, k, v, g, beta = layer_inputs[li]
        q_pert = q + hidden_C.reshape(1, TOTAL, NUM_HEADS, V_DIM)[..., :K_DIM] * 0.1
        k_pert = k + hidden_C.reshape(1, TOTAL, NUM_HEADS, V_DIM)[..., :K_DIM] * 0.1
        v_pert = v + hidden_C.reshape(1, TOTAL, NUM_HEADS, V_DIM) * 0.1
        # Block 1: chunked
        out1, st1 = chunked_delta_rule(
            q_pert[:,:BS], k_pert[:,:BS], v_pert[:,:BS],
            g[:,:BS], beta[:,:BS], CHUNK_SZ, S0.clone(), math_dtype
        )
        # Tail: chunked
        out2, st2 = chunked_delta_rule(
            q_pert[:,BS:], k_pert[:,BS:], v_pert[:,BS:],
            g[:,BS:], beta[:,BS:], CHUNK_SZ, st1.clone(), math_dtype
        )
        out_full = torch.cat([out1, out2], dim=1)
        hidden_C = hidden_C + out_full.reshape(1, TOTAL, -1)
    last_C = hidden_C[0, -1]

    # Compare
    cos_AB = F.cosine_similarity(last_A.reshape(1,-1).float(), last_B.reshape(1,-1).float()).item()
    cos_CB = F.cosine_similarity(last_C.reshape(1,-1).float(), last_B.reshape(1,-1).float()).item()
    cos_AC = F.cosine_similarity(last_A.reshape(1,-1).float(), last_C.reshape(1,-1).float()).item()
    mse_AB = ((last_A.float() - last_B.float()) ** 2).mean().item()
    mse_CB = ((last_C.float() - last_B.float()) ** 2).mean().item()

    print(f"\nAfter {n_layers} layers with residual connections:")
    print(f"  A (chunk+seq)   vs B (seq):       cos={cos_AB:.8f}  MSE={mse_AB:.2e}")
    print(f"  C (chunk+chunk) vs B (seq):       cos={cos_CB:.8f}  MSE={mse_CB:.2e}")
    print(f"  A (chunk+seq)   vs C (chunk+chunk): cos={cos_AC:.8f}")

    # Simulate top-token
    torch.manual_seed(99)
    lm_head = torch.randn(1000, NUM_HEADS * V_DIM, dtype=torch.float16)
    logits_A = (lm_head.float() @ last_A.reshape(-1).float()).numpy()
    logits_B = (lm_head.float() @ last_B.reshape(-1).float()).numpy()
    logits_C = (lm_head.float() @ last_C.reshape(-1).float()).numpy()
    top_A, top_B, top_C = logits_A.argmax(), logits_B.argmax(), logits_C.argmax()

    sorted_B = np.sort(logits_B)[::-1]
    margin = sorted_B[0] - sorted_B[1]

    print(f"\nSimulated top token (1000-vocab LM head):")
    print(f"  B (fully seq):        token {top_B:3d}  logit={logits_B[top_B]:.4f}  (margin={margin:.4f})")
    print(f"  A (chunk+seq tail):   token {top_A:3d}  logit={logits_A[top_A]:.4f}  match_B={top_A==top_B}")
    print(f"  C (chunk+chunk tail): token {top_C:3d}  logit={logits_C[top_C]:.4f}  match_B={top_C==top_B}")

    if top_C != top_B:
        print(f"\n  ⚠️  TOKEN FLIP DETECTED: C picked {top_C} instead of {top_B}")
        print(f"      logit_B[{top_B}]={logits_B[top_B]:.6f}  logit_C[{top_B}]={logits_C[top_B]:.6f}")
        print(f"      logit_B[{top_C}]={logits_B[top_C]:.6f}  logit_C[{top_C}]={logits_C[top_C]:.6f}")


# ─── Run at different precisions ───
print("Testing whether chunked vs sequential delta rule diverges in PyTorch")
print(f"This is PURE PyTorch — no CoreML, no ANE, no padding artifacts.")
print(f"If divergence appears in float16 math but not float32, it proves")
print(f"the drift is inherent to the numerical path, not a bug.\n")

run_test(torch.float16, "TEST 1: Single layer, float16 math")
run_test(torch.float32, "TEST 2: Single layer, float32 math")

# Multi-layer tests: compound drift through residual connections
# Qwen3.5-4B has 24 linear attention layers
run_multilayer_test(torch.float16, 8,  "TEST 3: 8 layers, float16 math")
run_multilayer_test(torch.float16, 24, "TEST 4: 24 layers, float16 math (real model)")
run_multilayer_test(torch.float32, 24, "TEST 5: 24 layers, float32 math (control)")
