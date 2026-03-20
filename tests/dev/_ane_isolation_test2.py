#!/usr/bin/env python3
"""Test Qwen3.5 decode with ALL compute units and build progressively larger models.
"""
import coremltools as ct
import numpy as np
import os
import time

DECODE_PATH = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export/qwen35_FFN_chunk_01of04.mlpackage"

# Test 1: ALL compute unit
print("=" * 60)
print("Test 1: Decode chunk on ALL compute units")
print("=" * 60)

model = ct.models.MLModel(DECODE_PATH, compute_units=ct.ComputeUnit.ALL)
state = model.make_state()
hidden = np.random.randn(1, 1, 2560).astype(np.float16) * 0.01
position_ids = np.zeros((1,), dtype=np.int32)
mask = np.zeros((1, 1, 1, 256), dtype=np.float16)
current_pos = np.zeros((1,), dtype=np.int32)

try:
    out = model.predict({
        "hidden_states": hidden,
        "position_ids": position_ids,
        "causal_mask": mask,
        "current_pos": current_pos,
    }, state=state)
    print("  ✅ SUCCESS on ALL")
    for k, v in out.items():
        arr = np.array(v)
        print(f"    {k}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f}")
except Exception as e:
    err = str(e)
    if "ANE" in err:
        print("  ❌ FAILED (ANE error even with ALL)")
    else:
        print(f"  ❌ FAILED: {err[:300]}")
del model, state

# Test 2: CPU_ONLY as baseline
print("\n" + "=" * 60)
print("Test 2: Decode chunk on CPU_ONLY")
print("=" * 60)

model = ct.models.MLModel(DECODE_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)
state = model.make_state()
try:
    out = model.predict({
        "hidden_states": hidden,
        "position_ids": position_ids,
        "causal_mask": mask,
        "current_pos": current_pos,
    }, state=state)
    print("  ✅ SUCCESS on CPU_ONLY")
except Exception as e:
    print(f"  ❌ FAILED: {str(e)[:300]}")
del model, state

# Test 3: Build a model that replicates key patterns
print("\n" + "=" * 60)
print("Test 3: Minimal linear attention pattern on ANE")
print("=" * 60)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalLinearAttn(nn.Module):
    """Minimal model replicating core linear attention ops."""
    def __init__(self):
        super().__init__()
        # Mimic linear attention: conv + exp + sigmoid + matmul + reduce_sum
        self.q_proj = nn.Conv2d(2560, 2048, 1, bias=False)  # 16*128
        self.k_proj = nn.Conv2d(2560, 2048, 1, bias=False)  # 16*128
        self.v_proj = nn.Conv2d(2560, 4096, 1, bias=False)  # 32*128
        self.o_proj = nn.Conv2d(4096, 2560, 1, bias=False)
        self.register_buffer("rec_state", torch.zeros(1, 32, 128, 128, dtype=torch.float16))

    def forward(self, x):
        # x: (1, 2560, 1, 1)
        q = self.q_proj(x)   # (1, 2048, 1, 1)
        k = self.k_proj(x)   # (1, 2048, 1, 1)
        v = self.v_proj(x)   # (1, 4096, 1, 1)

        # Reshape for attention
        q = q.view(1, 16, 128, 1)
        k = k.view(1, 16, 128, 1)
        v = v.view(1, 32, 128, 1)

        # Simplified linear attention ops
        q = q * torch.sigmoid(q)  # gate
        k = torch.exp(k.clamp(-5, 5))  # ELU-like

        # Outer product update to rec_state (simplified)
        # kv = k[:,:8] outer v[:,:8]  -- just use first 8 heads
        k8 = k[:, :8, :, :].squeeze(-1)  # (1, 8, 128)
        v8 = v[:, :8, :, :].squeeze(-1)  # (1, 8, 128)
        outer = k8.unsqueeze(-1) * v8.unsqueeze(-2)  # (1, 8, 128, 128)
        rec = self.rec_state[:, :8, :, :] + outer
        self.rec_state[:, :8, :, :] = rec

        # Read from rec_state
        q8 = q[:, :8, :, :].squeeze(-1)  # (1, 8, 128)
        out8 = torch.matmul(q8.unsqueeze(-2), rec).squeeze(-2)  # (1, 8, 128)

        out = out8.view(1, 1024, 1, 1)
        out = F.pad(out, (0, 0, 0, 0, 0, 4096 - 1024))  # pad to 4096
        return self.o_proj(out)

m = MinimalLinearAttn().eval().half()
x = torch.randn(1, 2560, 1, 1, dtype=torch.float16)
traced = torch.jit.trace(m, x)
states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, 32, 128, 128), dtype=np.float16),
        name="rec_state",
    )
]
mlm = ct.convert(
    traced,
    inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="y", dtype=np.float16)],
    states=states,
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
)
test_path = "/tmp/qwen35_ane_test/minimal_linear_attn.mlpackage"
mlm.save(test_path)
loaded = ct.models.MLModel(test_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
state = loaded.make_state()
try:
    out = loaded.predict({"x": np.random.randn(1, 2560, 1, 1).astype(np.float16)}, state=state)
    print("  ✅ Minimal linear attn SUCCESS on ANE")
except Exception as e:
    err = str(e)
    if "ANE" in err:
        print("  ❌ Minimal linear attn FAILED on ANE")
    else:
        print(f"  ❌ Error: {err[:200]}")
del loaded
