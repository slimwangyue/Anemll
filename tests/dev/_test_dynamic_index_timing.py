#!/usr/bin/env python3
"""Measure scatter vs one_hot timing at CTX=1024 with realistic Qwen3.5-4B dimensions.

Qwen3.5-4B: num_kv_heads=4, head_dim=128, context_length=1024
KV cache shape per layer: (1, 4, 1024, 128) = 1 MB (fp16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
import os, tempfile, time

# Qwen3.5-4B actual dimensions
CTX = 1024
NUM_KV_HEADS = 4
HEAD_DIM = 128
NUM_LAYERS_PER_CHUNK = 8  # typical chunk size

TMPDIR = tempfile.mkdtemp(prefix="timing_")

def build_and_test(name, model_class, trace_inputs_fn, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    print(f"\n{'='*60}")
    print(f"  {name}  (CTX={CTX})")
    print(f"{'='*60}")
    
    model = model_class()
    model.eval()
    trace_inputs = trace_inputs_fn()
    
    with torch.no_grad():
        traced = torch.jit.trace(model, trace_inputs)
    
    ct_inputs = []
    for i, t in enumerate(trace_inputs):
        if t.dtype in (torch.float16, torch.float32):
            ct_inputs.append(ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16))
        elif t.dtype in (torch.int32, torch.int64):
            ct_inputs.append(ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.int32))
    
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_units=compute_unit,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  Convert: {time.time()-t0:.1f}s")
    
    mlpath = os.path.join(TMPDIR, f"{name}.mlpackage")
    mlmodel.save(mlpath)
    
    cache_bytes = 1 * NUM_KV_HEADS * CTX * HEAD_DIM * 2
    print(f"  Cache: ({1},{NUM_KV_HEADS},{CTX},{HEAD_DIM}) = {cache_bytes/1024:.0f} KB")
    
    return mlmodel


def benchmark(mlmodel, name, make_inputs_fn, n_warmup=5, n_iter=50):
    """Benchmark with warmup + timing."""
    # Warmup
    for _ in range(n_warmup):
        inputs = make_inputs_fn(0)
        mlmodel.predict(inputs)
    
    # Correctness check at different positions
    print(f"\n  Correctness:")
    for p in [0, 100, 500, 1023]:
        inputs = make_inputs_fn(p)
        out = mlmodel.predict(inputs)["output"]
        print(f"    pos={p:4d}: shape={out.shape}, max={np.abs(out).max():.4f}")
    
    # Timing
    times = []
    for i in range(n_iter):
        p = i % CTX
        inputs = make_inputs_fn(p)
        t0 = time.perf_counter()
        mlmodel.predict(inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    
    times = np.array(times)
    print(f"\n  Timing ({n_iter} iters):")
    print(f"    mean:   {times.mean():.3f} ms")
    print(f"    median: {np.median(times):.3f} ms")
    print(f"    min:    {times.min():.3f} ms")
    print(f"    max:    {times.max():.3f} ms")
    print(f"    p95:    {np.percentile(times, 95):.3f} ms")
    return times


# ============================================================
# 1. Scatter write (single cache)
# ============================================================
class ScatterSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, value, pos):
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, NUM_KV_HEADS, 1, HEAD_DIM)
        self.cache = self.cache.scatter(2, idx, value.to(self.cache.dtype))
        return self.cache

def trace_scatter_single():
    return (torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.tensor([0], dtype=torch.int32))

def inputs_scatter_single(p):
    return {"input_0": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_1": np.array([p], dtype=np.int32)}


# ============================================================
# 2. One_hot write (single cache)
# ============================================================
class OneHotSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, value, pos):
        mask = F.one_hot(pos.long(), num_classes=CTX).to(self.cache.dtype)
        mask = mask.view(1, 1, CTX, 1)
        expanded = value.expand_as(self.cache)
        self.cache = self.cache * (1.0 - mask) + expanded * mask
        return self.cache

def trace_onehot_single():
    return (torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.tensor([0], dtype=torch.int32))

def inputs_onehot_single(p):
    return {"input_0": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_1": np.array([p], dtype=np.int32)}


# ============================================================
# 3. Update_mask write (existing approach — mask as input)
# ============================================================
class UpdateMaskSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, value, update_mask):
        expanded = value.expand_as(self.cache)
        self.cache = self.cache * (1.0 - update_mask) + expanded * update_mask
        return self.cache

def trace_updatemask_single():
    mask = torch.zeros(1, 1, CTX, 1, dtype=torch.float16)
    mask[:, :, 0, :] = 1.0
    return (torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16), mask)

def inputs_updatemask_single(p):
    mask = np.zeros((1, 1, CTX, 1), dtype=np.float16)
    mask[:, :, p, :] = 1.0
    return {"input_0": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_1": mask}


# ============================================================
# 4. Scatter with K+V caches (realistic: 2 scatter ops per layer)
# ============================================================
class ScatterKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('k_cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        self.register_buffer('v_cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, key, value, pos):
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, NUM_KV_HEADS, 1, HEAD_DIM)
        self.k_cache = self.k_cache.scatter(2, idx, key.to(self.k_cache.dtype))
        self.v_cache = self.v_cache.scatter(2, idx, value.to(self.v_cache.dtype))
        return self.k_cache + self.v_cache  # dummy output to keep both alive

def trace_scatter_kv():
    return (torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.tensor([0], dtype=torch.int32))

def inputs_scatter_kv(p):
    return {"input_0": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_1": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_2": np.array([p], dtype=np.int32)}


# ============================================================
# 5. One_hot with K+V caches (realistic: 2 mask ops per layer)
# ============================================================
class OneHotKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('k_cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        self.register_buffer('v_cache', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, key, value, pos):
        mask = F.one_hot(pos.long(), num_classes=CTX).to(self.k_cache.dtype)
        mask = mask.view(1, 1, CTX, 1)
        k_exp = key.expand_as(self.k_cache)
        v_exp = value.expand_as(self.v_cache)
        self.k_cache = self.k_cache * (1.0 - mask) + k_exp * mask
        self.v_cache = self.v_cache * (1.0 - mask) + v_exp * mask
        return self.k_cache + self.v_cache

def trace_onehot_kv():
    return (torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.tensor([0], dtype=torch.int32))

def inputs_onehot_kv(p):
    return {"input_0": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_1": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_2": np.array([p], dtype=np.int32)}


# ============================================================
# 6. Scatter with 8-layer KV (mimics full chunk)
# ============================================================
class ScatterChunk(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 layers × K+V = 16 cache buffers
        for i in range(NUM_LAYERS_PER_CHUNK):
            self.register_buffer(f'k_cache_{i}', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
            self.register_buffer(f'v_cache_{i}', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, key, value, pos):
        idx = pos.to(torch.int64).view(1, 1, 1, 1).expand(1, NUM_KV_HEADS, 1, HEAD_DIM)
        out = torch.zeros_like(key)
        for i in range(NUM_LAYERS_PER_CHUNK):
            k_cache = getattr(self, f'k_cache_{i}')
            v_cache = getattr(self, f'v_cache_{i}')
            new_k = k_cache.scatter(2, idx, key.to(k_cache.dtype))
            new_v = v_cache.scatter(2, idx, value.to(v_cache.dtype))
            setattr(self, f'k_cache_{i}', new_k)
            setattr(self, f'v_cache_{i}', new_v)
            out = out + new_k[:,:,0:1,:] + new_v[:,:,0:1,:]  # dummy to keep alive
        return out

def trace_scatter_chunk():
    return (torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.tensor([0], dtype=torch.int32))

def inputs_scatter_chunk(p):
    return {"input_0": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_1": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_2": np.array([p], dtype=np.int32)}


# ============================================================
# 7. One_hot with 8-layer KV (mimics full chunk)
# ============================================================
class OneHotChunk(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(NUM_LAYERS_PER_CHUNK):
            self.register_buffer(f'k_cache_{i}', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
            self.register_buffer(f'v_cache_{i}', torch.zeros(1, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
    
    def forward(self, key, value, pos):
        mask = F.one_hot(pos.long(), num_classes=CTX).to(torch.float16)
        mask = mask.view(1, 1, CTX, 1)
        inv_mask = 1.0 - mask
        out = torch.zeros_like(key)
        for i in range(NUM_LAYERS_PER_CHUNK):
            k_cache = getattr(self, f'k_cache_{i}')
            v_cache = getattr(self, f'v_cache_{i}')
            k_exp = key.expand_as(k_cache)
            v_exp = value.expand_as(v_cache)
            new_k = k_cache * inv_mask + k_exp * mask
            new_v = v_cache * inv_mask + v_exp * mask
            setattr(self, f'k_cache_{i}', new_k)
            setattr(self, f'v_cache_{i}', new_v)
            out = out + new_k[:,:,0:1,:] + new_v[:,:,0:1,:]
        return out

def trace_onehot_chunk():
    return (torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16),
            torch.tensor([0], dtype=torch.int32))

def inputs_onehot_chunk(p):
    return {"input_0": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_1": np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16),
            "input_2": np.array([p], dtype=np.int32)}


if __name__ == "__main__":
    print(f"Temp dir: {TMPDIR}")
    print(f"CTX={CTX}, heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"Single cache: {1*NUM_KV_HEADS*CTX*HEAD_DIM*2/1024:.0f} KB")
    print(f"8-layer K+V: {2*NUM_LAYERS_PER_CHUNK*NUM_KV_HEADS*CTX*HEAD_DIM*2/1024/1024:.1f} MB")
    
    tests = [
        ("1_scatter_single",    ScatterSingle,    trace_scatter_single,    inputs_scatter_single),
        ("2_onehot_single",     OneHotSingle,     trace_onehot_single,     inputs_onehot_single),
        ("3_updatemask_single", UpdateMaskSingle,  trace_updatemask_single, inputs_updatemask_single),
        ("4_scatter_kv",        ScatterKV,         trace_scatter_kv,        inputs_scatter_kv),
        ("5_onehot_kv",         OneHotKV,          trace_onehot_kv,         inputs_onehot_kv),
        ("6_scatter_8layer",    ScatterChunk,      trace_scatter_chunk,     inputs_scatter_chunk),
        ("7_onehot_8layer",     OneHotChunk,       trace_onehot_chunk,      inputs_onehot_chunk),
    ]
    
    all_results = {}
    for name, cls, trace_fn, inputs_fn in tests:
        try:
            ml = build_and_test(name, cls, trace_fn)
            times = benchmark(ml, name, inputs_fn)
            all_results[name] = np.median(times)
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results[name] = f"ERROR: {e}"
    
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON (CTX={CTX})")
    print(f"{'='*60}")
    print(f"  {'Test':<25} {'Median ms':>10}  {'Cache Size':>12}")
    print(f"  {'-'*25} {'-'*10}  {'-'*12}")
    for name, t in all_results.items():
        cache_kb = NUM_KV_HEADS * CTX * HEAD_DIM * 2 / 1024
        if "kv" in name:
            cache_kb *= 2
        elif "8layer" in name:
            cache_kb *= 2 * NUM_LAYERS_PER_CHUNK
        if isinstance(t, float):
            print(f"  {name:<25} {t:>10.3f}  {cache_kb:>10.0f} KB")
        else:
            print(f"  {name:<25} {str(t):>10}  {cache_kb:>10.0f} KB")
