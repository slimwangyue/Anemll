#!/usr/bin/env python3
"""Verify the Neumann series replacement produces identical results to the loop."""
import sys, os
sys.path.insert(0, '/Users/yw68/Anemll')
os.environ['QWEN35_NUM_CHUNKS'] = '4'
os.environ['QWEN35_HF_MODEL'] = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'
os.environ['QWEN35_STATIC_PREFILL_CTX'] = '2048'

import torch
import math
import numpy as np

# Test the Neumann series vs the original loop on random matrices
torch.manual_seed(42)

for cs in [4, 8, 16, 32, 64]:
    # Create a random strictly lower triangular matrix
    batch = 1
    heads = 4
    n_chunks = 8
    A_raw = torch.randn(batch, heads, n_chunks, cs, cs, dtype=torch.float32) * 0.1
    strict_lower = torch.tril(torch.ones(cs, cs), diagonal=-1)
    A = A_raw * strict_lower  # strictly lower triangular
    
    # Method 1: Original loop
    attn_rows = [A[..., 0:1, :]]
    for i in range(1, cs):
        row = A[..., i, :i].clone()
        sub = torch.cat([prev_row[..., :i] for prev_row in attn_rows[:i]], dim=-2)
        updated_row = row + (row.unsqueeze(-1) * sub).sum(-2)
        tail = A[..., i : i + 1, i:]
        full_row = torch.cat([updated_row.unsqueeze(-2), tail], dim=-1)
        attn_rows.append(full_row)
    loop_result = torch.cat(attn_rows, dim=-2)
    loop_result = loop_result + torch.eye(cs, dtype=A.dtype, device=A.device)
    
    # Method 2: Neumann series via repeated doubling
    S = A.clone()
    P = A.clone()
    n_steps = max(1, int(math.ceil(math.log2(cs))))
    for _ in range(n_steps - 1):
        S = S + P @ S
        P = P @ P
    neumann_result = S + torch.eye(cs, dtype=A.dtype, device=A.device)
    
    # Method 3: Direct (I - A)^{-1} computation for reference
    I = torch.eye(cs, dtype=A.dtype, device=A.device)
    I_minus_A = I.expand_as(A) - A
    direct_result = torch.linalg.solve_triangular(I_minus_A, I.expand_as(A), upper=False)
    
    # Compare
    loop_flat = loop_result.flatten()
    neumann_flat = neumann_result.flatten()
    direct_flat = direct_result.flatten()
    
    cos_ln = torch.nn.functional.cosine_similarity(loop_flat.unsqueeze(0), neumann_flat.unsqueeze(0)).item()
    cos_ld = torch.nn.functional.cosine_similarity(loop_flat.unsqueeze(0), direct_flat.unsqueeze(0)).item()
    max_diff_ln = (loop_flat - neumann_flat).abs().max().item()
    max_diff_ld = (loop_flat - direct_flat).abs().max().item()
    
    print(f"chunk_size={cs:3d}: loop vs neumann: cos={cos_ln:.10f} max_diff={max_diff_ln:.2e}"
          f"  |  loop vs direct: cos={cos_ld:.10f} max_diff={max_diff_ld:.2e}")

print("\nDone.")
