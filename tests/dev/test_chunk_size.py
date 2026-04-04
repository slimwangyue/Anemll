#!/usr/bin/env python3
"""Verify chunk_size=16 vs chunk_size=64 produce equivalent results."""
import torch
from anemll.models.qwen3_5_model import Qwen35LinearAttention

torch.manual_seed(42)
B, H, S, K, V = 1, 4, 64, 128, 128
q = torch.randn(B, S, H, K, dtype=torch.float16)
k = torch.randn(B, S, H, K, dtype=torch.float16)
v = torch.randn(B, S, H, V, dtype=torch.float16)
g = torch.randn(B, S, H, dtype=torch.float16)
beta = torch.randn(B, S, H, dtype=torch.float16)

out64, state64 = Qwen35LinearAttention._chunk_gated_delta_rule(
    q, k, v, g, beta, chunk_size=64,
    expected_batch_size=B, expected_num_heads=H, expected_seq_len=S,
    expected_k_dim=K, expected_v_dim=V)
out16, state16 = Qwen35LinearAttention._chunk_gated_delta_rule(
    q, k, v, g, beta, chunk_size=16,
    expected_batch_size=B, expected_num_heads=H, expected_seq_len=S,
    expected_k_dim=K, expected_v_dim=V)

cosine = torch.nn.functional.cosine_similarity(
    out64.flatten().float(), out16.flatten().float(), dim=0).item()
maxerr = (out64 - out16).abs().max().item()
print(f"Output cosine (64 vs 16): {cosine:.6f}")
print(f"Max abs error: {maxerr:.6f}")
print(f"State max diff: {(state64 - state16).abs().max().item():.6f}")

# Also test with seq_len=512 (the actual prefill size)
torch.manual_seed(42)
S2 = 512
q2 = torch.randn(B, S2, H, K, dtype=torch.float16) * 0.1
k2 = torch.randn(B, S2, H, K, dtype=torch.float16) * 0.1
v2 = torch.randn(B, S2, H, V, dtype=torch.float16) * 0.1
g2 = torch.randn(B, S2, H, dtype=torch.float16) * 0.1  # small gate values to avoid exp overflow
beta2 = torch.randn(B, S2, H, dtype=torch.float16) * 0.1

out64b, state64b = Qwen35LinearAttention._chunk_gated_delta_rule(
    q2, k2, v2, g2, beta2, chunk_size=64,
    expected_batch_size=B, expected_num_heads=H, expected_seq_len=S2,
    expected_k_dim=K, expected_v_dim=V)
out16b, state16b = Qwen35LinearAttention._chunk_gated_delta_rule(
    q2, k2, v2, g2, beta2, chunk_size=16,
    expected_batch_size=B, expected_num_heads=H, expected_seq_len=S2,
    expected_k_dim=K, expected_v_dim=V)

cosine2 = torch.nn.functional.cosine_similarity(
    out64b.flatten().float(), out16b.flatten().float(), dim=0).item()
maxerr2 = (out64b - out16b).abs().max().item()
print(f"\nSeq=512 cosine (64 vs 16): {cosine2:.6f}")
print(f"Seq=512 max abs error: {maxerr2:.6f}")
print(f"Seq=512 state max diff: {(state64b - state16b).abs().max().item():.6f}")
