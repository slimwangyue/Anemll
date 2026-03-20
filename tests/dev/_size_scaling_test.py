#!/usr/bin/env python3
"""Test if a 1.7GB model can run on ANE (with no states).
This isolates whether model SIZE is the bottleneck.
"""
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import os
import time

# Each Conv2d(2560, 2560, 1) has 2560*2560*2 = 13.1MB of weights
# For 1.7GB, we need ~130 layers. Let's test progressively.

class Conv2dStack(nn.Module):
    def __init__(self, num_layers, dim=2560):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(dim, dim, 1, bias=False) for _ in range(num_layers)])
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

def test_size(num_layers, dim=2560):
    m = Conv2dStack(num_layers, dim).eval().half()
    size_gb = sum(p.numel() * 2 for p in m.parameters()) / (1024**3)
    x = torch.randn(1, dim, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(m, x)
    mlm = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    path = "/tmp/qwen35_ane_test/size_test.mlpackage"
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    mlm.save(path)
    del mlm

    t0 = time.time()
    loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_t = time.time() - t0

    t0 = time.time()
    try:
        out = loaded.predict({"x": np.random.randn(1, dim, 1, 1).astype(np.float16)})
        pred_t = time.time() - t0
        print(f"  {num_layers} layers ({size_gb:.2f}GB): ✅ load={load_t:.1f}s pred={pred_t:.3f}s")
        del loaded
        return True
    except Exception as e:
        pred_t = time.time() - t0
        if "ANE" in str(e):
            print(f"  {num_layers} layers ({size_gb:.2f}GB): ❌ ANE FAIL load={load_t:.1f}s")
        else:
            print(f"  {num_layers} layers ({size_gb:.2f}GB): ❌ {str(e)[:100]}")
        del loaded
        return False

print("=" * 60)
print("Model size scaling test (Conv2d only, no states)")
print("=" * 60)

for n in [10, 30, 50, 80, 100, 130]:
    test_size(n)
