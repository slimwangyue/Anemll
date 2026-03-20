#!/usr/bin/env python3
"""Verify ANE-legal replacements for cumsum, masked_fill, tril."""
import torch

torch.manual_seed(42)
chunk_size = 64
math_dtype = torch.float32

# --- cumsum replacement ---
g = torch.randn(1, 32, 4, chunk_size)
g_orig = g.clone().cumsum(dim=-1)
tril_ones = torch.tril(torch.ones(chunk_size, chunk_size, dtype=g.dtype))
g_new = (tril_ones @ g.unsqueeze(-1)).squeeze(-1)
diff = (g_orig - g_new).abs()
print(f"cumsum: max_diff={diff.max():.8f} mean={diff.mean():.8f}")
assert diff.max() < 1e-4, f"FAIL: {diff.max()}"
print("  PASS")

# --- masked_fill(triu(diagonal=0), 0) → * strict_lower ---
mask_bool = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
data = torch.randn(1, 32, 4, chunk_size, chunk_size)
orig = data.clone().masked_fill(mask_bool, 0)
strict_lower = torch.tril(torch.ones(chunk_size, chunk_size), diagonal=-1)
new = data * strict_lower
diff2 = (orig - new).abs()
print(f"masked_fill(diag=0): max_diff={diff2.max():.8f}")
assert diff2.max() < 1e-6, f"FAIL: {diff2.max()}"
print("  PASS")

# --- masked_fill(triu(diagonal=1), 0) → * tril(diagonal=0) ---
mask_bool2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
orig2 = data.clone().masked_fill(mask_bool2, 0)
tril_diag0 = torch.tril(torch.ones(chunk_size, chunk_size))
new2 = data * tril_diag0
diff3 = (orig2 - new2).abs()
print(f"masked_fill(diag=1): max_diff={diff3.max():.8f}")
assert diff3.max() < 1e-6, f"FAIL: {diff3.max()}"
print("  PASS")

# --- .tril() → * tril_mask ---
data2 = torch.randn(1, 32, 4, chunk_size, chunk_size)
orig_tril = data2.tril()
new_tril = data2 * tril_diag0
diff4 = (orig_tril - new_tril).abs()
print(f"tril: max_diff={diff4.max():.8f}")
assert diff4.max() < 1e-6
print("  PASS")

print("\nAll ANE-legal replacements verified!")
