#!/usr/bin/env python3
"""Test scatter/gather with FIXED dtype issues from round 1.

Round 1 failures:
  B: scatter_ needed int64 index → fix: use .long()
  C: index_put_ dtype mismatch → fix: cast value to match cache dtype
  F: scatter dtype mismatch → fix: ensure src matches data dtype

Also test:
  G: gather for READ + scatter for WRITE (both tensor-indexed)
  H: one_hot from pos (compute mask dynamically inside the model from int pos)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
import os, tempfile

CTX = 8
HIDDEN = 4
TMPDIR = tempfile.mkdtemp(prefix="dynidx2_")

def test_approach(name, model_class, trace_inputs_fn, test_fn):
    print(f"\n{'='*60}")
    print(f"  APPROACH {name}")
    print(f"{'='*60}")
    
    model = model_class()
    model.eval()
    trace_inputs = trace_inputs_fn()
    
    with torch.no_grad():
        traced = torch.jit.trace(model, trace_inputs)
    
    # Key ops in JIT graph
    graph_str = str(traced.graph)
    print(f"\nJIT Graph (key ops):")
    for line in graph_str.split('\n'):
        ls = line.strip()
        if any(k in ls for k in ['scatter', 'gather', 'index_select', 'slice', 'Int(', 'one_hot', 'expand', 'mul', 'add(%', 'narrow', 'index_put']):
            print(f"  {ls}")
    
    # Convert
    ct_inputs = []
    for i, t in enumerate(trace_inputs):
        if t.dtype in (torch.float16, torch.float32):
            ct_inputs.append(ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16))
        elif t.dtype in (torch.int32, torch.int64):
            ct_inputs.append(ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.int32))
    
    try:
        mlmodel = ct.convert(
            traced,
            inputs=ct_inputs,
            outputs=[ct.TensorType(name="output", dtype=np.float16)],
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS18,
        )
    except Exception as e:
        print(f"  CoreML conversion FAILED: {e}")
        return "CONVERT_FAIL"
    
    mlpath = os.path.join(TMPDIR, f"test_{name}.mlpackage")
    mlmodel.save(mlpath)
    
    # Check MIL for scatter/gather ops
    spec = mlmodel.get_spec()
    mil_str = str(spec)
    for line in mil_str.split('\n'):
        ls = line.strip()
        if any(k in ls for k in ['scatter', 'gather', 'slice_update', 'one_hot']):
            print(f"  MIL: {ls}")
    
    # Test
    dynamic = test_fn(mlmodel, name)
    return "DYNAMIC" if dynamic else "FROZEN"


# ============================================================
# B_fixed: scatter_ with int64 index 
# ============================================================
class ModelB2(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, pos):
        # pos: [1] int32 → int64 for scatter
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, 1, 1, HIDDEN)
        # Ensure value is same dtype as cache
        val = value.to(self.cache.dtype)
        self.cache.scatter_(2, idx, val)
        return self.cache.clone()

def trace_b2():
    return (torch.ones(1,1,1,HIDDEN, dtype=torch.float16)*5, torch.tensor([0], dtype=torch.int32))

def test_b2(ml, name):
    dynamic = True
    for p in [0, 3, 5]:
        val = np.ones((1,1,1,HIDDEN), dtype=np.float16) * (p+1)
        pos = np.array([p], dtype=np.int32)
        out = ml.predict({"input_0": val, "input_1": pos})["output"]
        wp = np.argmax(np.abs(out).sum(axis=-1).flatten())
        print(f"  pos={p}, val={p+1}: written_to={wp}, cache[0]={out[0,0,0,:2]}, cache[{p}]={out[0,0,min(p,CTX-1),:2]}")
        if p > 0 and wp == 0:
            dynamic = False
    return dynamic


# ============================================================
# F_fixed: non-inplace scatter with dtype fix
# ============================================================
class ModelF2(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, pos):
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, 1, 1, HIDDEN)
        val = value.to(self.cache.dtype)
        self.cache = self.cache.scatter(2, idx, val)
        return self.cache

def trace_f2():
    return (torch.ones(1,1,1,HIDDEN, dtype=torch.float16)*5, torch.tensor([0], dtype=torch.int32))

def test_f2(ml, name):
    dynamic = True
    for p in [0, 3, 5]:
        val = np.ones((1,1,1,HIDDEN), dtype=np.float16) * (p+1)
        pos = np.array([p], dtype=np.int32)
        out = ml.predict({"input_0": val, "input_1": pos})["output"]
        wp = np.argmax(np.abs(out).sum(axis=-1).flatten())
        print(f"  pos={p}, val={p+1}: written_to={wp}, cache[0]={out[0,0,0,:2]}, cache[{p}]={out[0,0,min(p,CTX-1),:2]}")
        if p > 0 and wp == 0:
            dynamic = False
    return dynamic


# ============================================================
# G: gather READ + scatter WRITE (both tensor-indexed)
# ============================================================
class ModelG(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.arange(CTX*HIDDEN, dtype=torch.float16).view(1,1,CTX,HIDDEN))
    
    def forward(self, value, pos):
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, 1, 1, HIDDEN)
        # Write
        self.cache = self.cache.scatter(2, idx, value.to(self.cache.dtype))
        # Read back from written position
        read = self.cache.gather(2, idx)  # [1,1,1,H]
        return read

def trace_g():
    return (torch.ones(1,1,1,HIDDEN, dtype=torch.float16)*99, torch.tensor([0], dtype=torch.int32))

def test_g(ml, name):
    dynamic = True
    for p in [0, 3, 5]:
        val = np.ones((1,1,1,HIDDEN), dtype=np.float16) * (p+1)
        pos = np.array([p], dtype=np.int32)
        out = ml.predict({"input_0": val, "input_1": pos})["output"]
        expected = p + 1
        actual = out[0,0,0,0]
        match = abs(float(actual) - expected) < 0.1
        print(f"  pos={p}: read_back={out[0,0,0,:]}, expected={expected}, match={match}")
        if not match:
            dynamic = False
    return dynamic


# ============================================================
# H: compute one_hot mask from int pos INSIDE the model
# ============================================================
class ModelH(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, pos):
        # pos: [1] int32 → one_hot → mask
        # one_hot: [1, CTX] → [1, 1, CTX, 1]
        mask = F.one_hot(pos.long(), num_classes=CTX).to(self.cache.dtype)  # [1, CTX]
        mask = mask.view(1, 1, CTX, 1)  # [1, 1, CTX, 1]
        expanded = value.expand_as(self.cache)
        self.cache = self.cache * (1.0 - mask) + expanded * mask
        return self.cache

def trace_h():
    return (torch.ones(1,1,1,HIDDEN, dtype=torch.float16)*5, torch.tensor([0], dtype=torch.int32))

def test_h(ml, name):
    dynamic = True
    for p in [0, 3, 5]:
        val = np.ones((1,1,1,HIDDEN), dtype=np.float16) * (p+1)
        pos = np.array([p], dtype=np.int32)
        out = ml.predict({"input_0": val, "input_1": pos})["output"]
        wp = np.argmax(np.abs(out).sum(axis=-1).flatten())
        print(f"  pos={p}, val={p+1}: written_to={wp}, cache[0]={out[0,0,0,:2]}, cache[{p}]={out[0,0,min(p,CTX-1),:2]}")
        if p > 0 and wp == 0:
            dynamic = False
    return dynamic


# ============================================================
# I: stateful cache with scatter (real scenario)
# ============================================================
class ModelI(nn.Module):
    """Stateful KV cache write using scatter — mimics real decode."""
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, pos):
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, 1, 1, HIDDEN)
        val = value.to(self.cache.dtype)
        # Write new value at pos
        new_cache = self.cache.scatter(2, idx, val)
        self.cache = new_cache
        # Return full cache (attention would use this)
        return new_cache

def trace_i():
    return (torch.ones(1,1,1,HIDDEN, dtype=torch.float16)*5, torch.tensor([0], dtype=torch.int32))

def test_i(ml, name):
    """Test 3 sequential writes to verify statefulness + dynamic position."""
    dynamic = True
    for p in [0, 3, 5]:
        val = np.ones((1,1,1,HIDDEN), dtype=np.float16) * (p+1)
        pos = np.array([p], dtype=np.int32)
        out = ml.predict({"input_0": val, "input_1": pos})["output"]
        wp = np.argmax(np.abs(out).sum(axis=-1).flatten())
        nonzero = np.count_nonzero(out)
        print(f"  pos={p}, val={p+1}: written_to={wp}, nonzero={nonzero}, row0={out[0,0,0,:2]}, row{p}={out[0,0,min(p,CTX-1),:2]}")
        if p > 0 and wp == 0:
            dynamic = False
    return dynamic


if __name__ == "__main__":
    print(f"Temp dir: {TMPDIR}")
    print(f"Round 2: dtype-fixed scatter + new approaches")
    
    approaches = [
        ("B2_scatter_int64", ModelB2, trace_b2, test_b2),
        ("F2_scatter_noninplace", ModelF2, trace_f2, test_f2),
        ("G_scatter_write_gather_read", ModelG, trace_g, test_g),
        ("H_onehot_from_pos", ModelH, trace_h, test_h),
        ("I_stateful_scatter", ModelI, trace_i, test_i),
    ]
    
    results = {}
    for name, cls, trace_fn, test_fn in approaches:
        try:
            result = test_approach(name, cls, trace_fn, test_fn)
            results[name] = result
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = f"ERROR: {e}"
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        marker = "✓" if status == "DYNAMIC" else "✗" if status == "FROZEN" else "!"
        print(f"  [{marker}] {name}: {status}")
    print(f"\nDYNAMIC = pos actually changes write location at runtime")
    print(f"FROZEN = all writes go to pos=0 regardless of input")
