#!/usr/bin/env python3
"""Test different approaches to keep position index DYNAMIC through jit.trace + coremltools.

The problem: cache[:, :, pos:pos+1, :] calls aten::Int(pos) which constant-folds at trace time.
Hypothesis: ops that take TENSOR indices (scatter, gather, index_select) stay dynamic.

We test 6 approaches:
  A) Baseline: slice with int32 scalar pos (known to freeze)
  B) scatter_ / gather with int32 tensor index
  C) index_select for read + scatter_ for write
  D) one_hot mask approach (the update_mask pattern, known to work)
  E) pos as float16 (does it prevent Int conversion?)
  F) pos as int32 shape [1] (non-scalar)
"""

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
import os, tempfile

CTX = 8
HIDDEN = 4
TMPDIR = tempfile.mkdtemp(prefix="dynidx_")

def test_approach(name, model_class, trace_inputs_fn, test_fn):
    """Generic test runner: trace → convert → test with different pos values."""
    print(f"\n{'='*60}")
    print(f"  APPROACH {name}")
    print(f"{'='*60}")
    
    model = model_class()
    model.eval()
    
    trace_inputs = trace_inputs_fn()
    
    # Trace
    with torch.no_grad():
        traced = torch.jit.trace(model, trace_inputs)
    
    # Print JIT graph
    print(f"\nJIT Graph (key ops):")
    graph_str = str(traced.graph)
    for line in graph_str.split('\n'):
        line_stripped = line.strip()
        if any(k in line_stripped for k in ['scatter', 'gather', 'index_select', 'slice', 'Int(', 'one_hot', 'expand', 'mul', 'add', 'narrow']):
            print(f"  {line_stripped}")
    
    # Convert to CoreML
    ct_inputs = []
    for i, t in enumerate(trace_inputs):
        if t.dtype == torch.float16 or t.dtype == torch.float32:
            ct_inputs.append(ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16))
        elif t.dtype == torch.int32 or t.dtype == torch.int64:
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
        return False
    
    # Save
    mlpath = os.path.join(TMPDIR, f"test_{name}.mlpackage")
    mlmodel.save(mlpath)
    
    # Print MIL ops
    spec = mlmodel.get_spec()
    mil_str = str(spec)
    print(f"\nMIL spec length: {len(mil_str)} chars")
    for line in mil_str.split('\n'):
        line_stripped = line.strip()
        if any(k in line_stripped for k in ['scatter', 'gather', 'index_select', 'slice_update', 'slice_by', 'one_hot']):
            print(f"  MIL: {line_stripped}")
    
    # Test with different pos values
    test_fn(mlmodel, name)
    return True


# ============================================================
# APPROACH A: Baseline slice (known to freeze)
# ============================================================
class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, pos):
        # pos is int32 scalar → aten::Int → frozen
        p = pos.item()  # or pos[0].item() — becomes aten::Int
        self.cache[:, :, p:p+1, :] = value
        return self.cache.clone()

def trace_inputs_a():
    val = torch.ones(1, 1, 1, HIDDEN, dtype=torch.float16) * 5.0
    pos = torch.tensor([0], dtype=torch.int32)
    return (val, pos)

def test_a(mlmodel, name):
    for p in [0, 3, 5]:
        val = np.ones((1, 1, 1, HIDDEN), dtype=np.float16) * (p + 1)
        pos = np.array([p], dtype=np.int32)
        out = mlmodel.predict({"input_0": val, "input_1": pos})["output"]
        nonzero = np.count_nonzero(out)
        written_pos = np.argmax(np.abs(out).sum(axis=-1).flatten())
        print(f"  pos={p}, val={p+1}: written_to_pos={written_pos}, nonzero_cells={nonzero}, cache[0]={out[0,0,0,:2]}, cache[{p}]={out[0,0,min(p,CTX-1),:2]}")


# ============================================================
# APPROACH B: scatter_ for write + gather for read
# ============================================================
class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, pos):
        # pos: shape [1] int32 → expand to scatter index shape [1,1,1,HIDDEN]
        idx = pos.view(1, 1, 1, 1).expand(1, 1, 1, HIDDEN)  # [1,1,1,H]
        # Write: scatter along dim=2
        self.cache.scatter_(2, idx, value)
        return self.cache.clone()

def trace_inputs_b():
    val = torch.ones(1, 1, 1, HIDDEN, dtype=torch.float16) * 5.0
    pos = torch.tensor([0], dtype=torch.int32)
    return (val, pos)

def test_b(mlmodel, name):
    for p in [0, 3, 5]:
        val = np.ones((1, 1, 1, HIDDEN), dtype=np.float16) * (p + 1)
        pos = np.array([p], dtype=np.int32)
        out = mlmodel.predict({"input_0": val, "input_1": pos})["output"]
        nonzero = np.count_nonzero(out)
        written_pos = np.argmax(np.abs(out).sum(axis=-1).flatten())
        print(f"  pos={p}, val={p+1}: written_to_pos={written_pos}, nonzero_cells={nonzero}, cache[0]={out[0,0,0,:2]}, cache[{p}]={out[0,0,min(p,CTX-1),:2]}")


# ============================================================
# APPROACH C: index_put with tensor index
# ============================================================
class ModelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, pos):
        # Use index_put_ with tensor indices
        # cache[:, :, pos, :] = value[:, :, 0, :]
        idx = pos.long()  # ensure int64
        self.cache[0, 0, idx, :] = value[0, 0, 0, :]
        return self.cache.clone()

def trace_inputs_c():
    val = torch.ones(1, 1, 1, HIDDEN, dtype=torch.float16) * 5.0
    pos = torch.tensor([0], dtype=torch.int32)
    return (val, pos)

def test_c(mlmodel, name):
    for p in [0, 3, 5]:
        val = np.ones((1, 1, 1, HIDDEN), dtype=np.float16) * (p + 1)
        pos = np.array([p], dtype=np.int32)
        out = mlmodel.predict({"input_0": val, "input_1": pos})["output"]
        nonzero = np.count_nonzero(out)
        written_pos = np.argmax(np.abs(out).sum(axis=-1).flatten())
        print(f"  pos={p}, val={p+1}: written_to_pos={written_pos}, nonzero_cells={nonzero}, cache[0]={out[0,0,0,:2]}, cache[{p}]={out[0,0,min(p,CTX-1),:2]}")


# ============================================================
# APPROACH D: one_hot mask (update_mask baseline — known to work)
# ============================================================
class ModelD(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, update_mask):
        # update_mask: (1,1,CTX,1) float16 with 1.0 at write position
        expanded = value.expand_as(self.cache)
        self.cache = self.cache * (1.0 - update_mask) + expanded * update_mask
        return self.cache.clone()

def trace_inputs_d():
    val = torch.ones(1, 1, 1, HIDDEN, dtype=torch.float16) * 5.0
    mask = torch.zeros(1, 1, CTX, 1, dtype=torch.float16)
    mask[:, :, 0, :] = 1.0
    return (val, mask)

def test_d(mlmodel, name):
    for p in [0, 3, 5]:
        val = np.ones((1, 1, 1, HIDDEN), dtype=np.float16) * (p + 1)
        mask = np.zeros((1, 1, CTX, 1), dtype=np.float16)
        mask[:, :, p, :] = 1.0
        out = mlmodel.predict({"input_0": val, "input_1": mask})["output"]
        nonzero = np.count_nonzero(out)
        written_pos = np.argmax(np.abs(out).sum(axis=-1).flatten())
        print(f"  pos={p}, val={p+1}: written_to_pos={written_pos}, nonzero_cells={nonzero}, cache[0]={out[0,0,0,:2]}, cache[{p}]={out[0,0,min(p,CTX-1),:2]}")


# ============================================================
# APPROACH E: pos as float16 (avoid aten::Int?)
# ============================================================
class ModelE(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, pos_f16):
        # pos as float16 → cast to int for slicing
        p = pos_f16.to(torch.int32).item()
        self.cache[:, :, p:p+1, :] = value
        return self.cache.clone()

def trace_inputs_e():
    val = torch.ones(1, 1, 1, HIDDEN, dtype=torch.float16) * 5.0
    pos = torch.tensor([0.0], dtype=torch.float16)
    return (val, pos)

def test_e(mlmodel, name):
    for p in [0, 3, 5]:
        val = np.ones((1, 1, 1, HIDDEN), dtype=np.float16) * (p + 1)
        pos = np.array([float(p)], dtype=np.float16)
        out = mlmodel.predict({"input_0": val, "input_1": pos})["output"]
        nonzero = np.count_nonzero(out)
        written_pos = np.argmax(np.abs(out).sum(axis=-1).flatten())
        print(f"  pos={p}, val={p+1}: written_to_pos={written_pos}, nonzero_cells={nonzero}, cache[0]={out[0,0,0,:2]}, cache[{p}]={out[0,0,min(p,CTX-1),:2]}")


# ============================================================
# APPROACH F: scatter_ with int64 (native gather/scatter dtype)
# ============================================================
class ModelF(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, 1, CTX, HIDDEN, dtype=torch.float16))
    
    def forward(self, value, pos):
        # pos: [1] int32 → int64, then scatter
        idx = pos.long().view(1, 1, 1, 1).expand(1, 1, 1, HIDDEN)
        # Scatter along dim=2: write value at position pos
        self.cache = self.cache.scatter(2, idx, value)  # non-inplace returns new tensor
        return self.cache

def trace_inputs_f():
    val = torch.ones(1, 1, 1, HIDDEN, dtype=torch.float16) * 5.0
    pos = torch.tensor([0], dtype=torch.int32)
    return (val, pos)

def test_f(mlmodel, name):
    for p in [0, 3, 5]:
        val = np.ones((1, 1, 1, HIDDEN), dtype=np.float16) * (p + 1)
        pos = np.array([p], dtype=np.int32)
        out = mlmodel.predict({"input_0": val, "input_1": pos})["output"]
        nonzero = np.count_nonzero(out)
        written_pos = np.argmax(np.abs(out).sum(axis=-1).flatten())
        print(f"  pos={p}, val={p+1}: written_to_pos={written_pos}, nonzero_cells={nonzero}, cache[0]={out[0,0,0,:2]}, cache[{p}]={out[0,0,min(p,CTX-1),:2]}")


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print(f"Temp dir: {TMPDIR}")
    print(f"Testing dynamic index preservation through jit.trace + coremltools")
    print(f"Cache shape: (1,1,{CTX},{HIDDEN})")
    print(f"If approach works: different pos values should write to DIFFERENT positions")
    print(f"If approach fails: all writes go to position 0 (frozen)")
    
    approaches = [
        ("A_slice_baseline", ModelA, trace_inputs_a, test_a),
        ("B_scatter_inplace", ModelB, trace_inputs_b, test_b),
        ("C_index_put", ModelC, trace_inputs_c, test_c),
        ("D_update_mask", ModelD, trace_inputs_d, test_d),
        ("E_float16_pos", ModelE, trace_inputs_e, test_e),
        ("F_scatter_noninplace", ModelF, trace_inputs_f, test_f),
    ]
    
    results = {}
    for name, cls, trace_fn, test_fn in approaches:
        try:
            ok = test_approach(name, cls, trace_fn, test_fn)
            results[name] = "OK" if ok else "CONVERT_FAIL"
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            results[name] = f"ERROR: {e}"
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print(f"\nKey: If written_to_pos changes with pos input → DYNAMIC (success)")
    print(f"     If written_to_pos always = 0 → FROZEN (failure)")
