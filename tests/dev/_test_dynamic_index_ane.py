#!/usr/bin/env python3
"""Test the 4 DYNAMIC approaches on ANE (CPU_AND_NE) to verify they actually run.

From round 2, these preserve dynamic position:
  F2: scatter (non-inplace) → scatter_along_axis
  G:  scatter + gather → scatter + gather_along_axis
  H:  one_hot mask from int pos → one_hot + mul + add
  I:  stateful scatter → scatter_along_axis

We test each on ANE with the STATEFUL model pattern (register_buffer cache).
Also test with larger dimensions closer to real model: head_dim=128, ctx=256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
import os, tempfile, time

# Realistic sizes
CTX = 256
NUM_KV_HEADS = 4
HEAD_DIM = 128
HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 512
TMPDIR = tempfile.mkdtemp(prefix="ane_dynidx_")

def test_on_ane(name, model_class, trace_inputs_fn, predict_fn, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    print(f"\n{'='*60}")
    print(f"  {name} on {compute_unit}")
    print(f"{'='*60}")
    
    model = model_class()
    model.eval()
    trace_inputs = trace_inputs_fn()
    
    with torch.no_grad():
        traced = torch.jit.trace(model, trace_inputs)
    
    # Build CoreML inputs
    ct_inputs = []
    for i, t in enumerate(trace_inputs):
        if t.dtype in (torch.float16, torch.float32):
            ct_inputs.append(ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16))
        elif t.dtype in (torch.int32, torch.int64):
            ct_inputs.append(ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.int32))
    
    t0 = time.time()
    try:
        mlmodel = ct.convert(
            traced,
            inputs=ct_inputs,
            outputs=[ct.TensorType(name="output", dtype=np.float16)],
            compute_units=compute_unit,
            minimum_deployment_target=ct.target.iOS18,
        )
    except Exception as e:
        print(f"  CONVERT FAILED: {e}")
        return None
    t1 = time.time()
    print(f"  Converted in {t1-t0:.1f}s")
    
    mlpath = os.path.join(TMPDIR, f"{name}.mlpackage")
    mlmodel.save(mlpath)
    
    # Test predictions
    try:
        results = predict_fn(mlmodel)
        return results
    except Exception as e:
        print(f"  PREDICT FAILED: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================
# F2: Non-inplace scatter (scatter_along_axis)
# ============================================================
class ScatterModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Shape: (1, num_kv_heads, ctx, head_dim) — like real KV cache
        self.register_buffer('cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, value, pos):
        # value: (1, num_kv_heads, 1, head_dim)
        # pos: (1,) int32
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, NUM_KV_HEADS, 1, HEAD_DIM)
        self.cache = self.cache.scatter(2, idx, value.to(self.cache.dtype))
        return self.cache

def trace_scatter():
    val = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
    pos = torch.tensor([0], dtype=torch.int32)
    return (val, pos)

def test_scatter(ml):
    dynamic = True
    all_ok = True
    for p in [0, 50, 100, 200]:
        val = np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16)
        pos = np.array([p], dtype=np.int32)
        t0 = time.time()
        out = ml.predict({"input_0": val, "input_1": pos})["output"]
        t1 = time.time()
        # Check: value should appear at position p
        written = out[0, 0, p, :4]
        expected = val[0, 0, 0, :4]
        match = np.allclose(written, expected, atol=0.01)
        # Check position 0 (should be zero for p>0)
        pos0 = out[0, 0, 0, :4]
        print(f"  pos={p:3d}: match={match}, time={1000*(t1-t0):.1f}ms, "
              f"written[:4]={written}, expected[:4]={expected}, pos0[:4]={pos0}")
        if not match:
            all_ok = False
        if p > 0 and np.allclose(pos0, expected, atol=0.01):
            dynamic = False
    return {"dynamic": dynamic, "correct": all_ok}


# ============================================================
# G: Scatter write + Gather read
# ============================================================
class ScatterGatherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, value, pos):
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, NUM_KV_HEADS, 1, HEAD_DIM)
        self.cache = self.cache.scatter(2, idx, value.to(self.cache.dtype))
        # Read back from pos
        read = self.cache.gather(2, idx)
        return read  # (1, num_kv_heads, 1, head_dim)

def trace_sg():
    val = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
    pos = torch.tensor([0], dtype=torch.int32)
    return (val, pos)

def test_sg(ml):
    dynamic = True
    for p in [0, 50, 100, 200]:
        val = np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16)
        pos = np.array([p], dtype=np.int32)
        out = ml.predict({"input_0": val, "input_1": pos})["output"]
        match = np.allclose(out, val, atol=0.01)
        print(f"  pos={p:3d}: read_back_match={match}, out[:4]={out[0,0,0,:4]}, val[:4]={val[0,0,0,:4]}")
        if not match:
            dynamic = False
    return {"dynamic": dynamic}


# ============================================================
# H: one_hot from pos (compute mask inside model)
# ============================================================
class OneHotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, value, pos):
        # one_hot: (1,) → (1, CTX) → (1, 1, CTX, 1)
        mask = F.one_hot(pos.long(), num_classes=CTX).to(self.cache.dtype)
        mask = mask.view(1, 1, CTX, 1)
        expanded = value.expand_as(self.cache)
        self.cache = self.cache * (1.0 - mask) + expanded * mask
        return self.cache

def trace_oh():
    val = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
    pos = torch.tensor([0], dtype=torch.int32)
    return (val, pos)

def test_oh(ml):
    dynamic = True
    for p in [0, 50, 100, 200]:
        val = np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16)
        pos = np.array([p], dtype=np.int32)
        t0 = time.time()
        out = ml.predict({"input_0": val, "input_1": pos})["output"]
        t1 = time.time()
        written = out[0, 0, p, :4]
        expected = val[0, 0, 0, :4]
        match = np.allclose(written, expected, atol=0.01)
        print(f"  pos={p:3d}: match={match}, time={1000*(t1-t0):.1f}ms, "
              f"written[:4]={written}, expected[:4]={expected}")
        if not match:
            dynamic = False
    return {"dynamic": dynamic}


# ============================================================
# I: Stateful scatter (sequential writes)
# ============================================================
class StatefulScatterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, value, pos):
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, NUM_KV_HEADS, 1, HEAD_DIM)
        self.cache = self.cache.scatter(2, idx, value.to(self.cache.dtype))
        return self.cache

def trace_ss():
    val = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
    pos = torch.tensor([0], dtype=torch.int32)
    return (val, pos)

def test_ss(ml):
    """Sequential writes: verify accumulation across calls."""
    vals = {}
    for step, p in enumerate([10, 50, 100]):
        val = np.ones((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float16) * (step + 1)
        pos = np.array([p], dtype=np.int32)
        out = ml.predict({"input_0": val, "input_1": pos})["output"]
        vals[p] = step + 1
    
    # After 3 writes, check all positions are correct
    print(f"  After 3 sequential writes (pos=10,50,100):")
    all_ok = True
    for p, expected_val in vals.items():
        actual = out[0, 0, p, 0]
        ok = abs(float(actual) - expected_val) < 0.1
        print(f"    pos={p:3d}: expected={expected_val}, got={actual:.1f}, ok={ok}")
        if not ok:
            all_ok = False
    
    # Check that pos=0 (never written) is still 0
    zero_check = float(out[0, 0, 0, 0])
    print(f"    pos=0 (never written): {zero_check:.1f}, ok={abs(zero_check) < 0.1}")
    
    return {"stateful_correct": all_ok}


if __name__ == "__main__":
    print(f"Temp dir: {TMPDIR}")
    print(f"Cache shape: (1, {NUM_KV_HEADS}, {CTX}, {HEAD_DIM}) = {1*NUM_KV_HEADS*CTX*HEAD_DIM*2/1024:.0f} KB")
    print(f"Testing on CPU_AND_NE (ANE)")
    
    tests = [
        ("F2_scatter", ScatterModel, trace_scatter, test_scatter),
        ("G_scatter_gather", ScatterGatherModel, trace_sg, test_sg),
        ("H_onehot", OneHotModel, trace_oh, test_oh),
        ("I_stateful", StatefulScatterModel, trace_ss, test_ss),
    ]
    
    results = {}
    for name, cls, trace_fn, test_fn in tests:
        try:
            r = test_on_ane(name, cls, trace_fn, test_fn, ct.ComputeUnit.CPU_AND_NE)
            results[name] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}
    
    print(f"\n{'='*60}")
    print(f"  ANE SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name}: {r}")
