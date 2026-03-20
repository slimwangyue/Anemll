#!/usr/bin/env python3
"""Narrow down conv_state ANE failure: is it 3D shape, 8192 dim, or what?"""
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import os
import shutil

def test_state(label, name, shape, conv_channels=256):
    """Test a single state on ANE."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(conv_channels, conv_channels, 1, bias=False)
            self.register_buffer(name, torch.zeros(*shape, dtype=torch.float16))
        def forward(self, x):
            h = self.conv(x)
            buf = getattr(self, name)
            sl = tuple([slice(0, 1)] * len(shape))
            buf[sl] = torch.zeros(*([1]*len(shape)), dtype=torch.float16)
            return h

    m = TestModel().eval().half()
    x = torch.randn(1, conv_channels, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(m, x)
    ct_states = [ct.StateType(wrapped_type=ct.TensorType(shape=shape, dtype=np.float16), name=name)]
    try:
        mlm = ct.convert(traced,
            inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            states=ct_states, compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18)
    except Exception as e:
        print(f"  {label}: CONVERT ERROR")
        return

    path = f"/tmp/qwen35_ane_test/conv_test.mlpackage"
    if os.path.exists(path):
        shutil.rmtree(path)
    mlm.save(path)
    try:
        loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = loaded.make_state()
        out = loaded.predict({"x": np.random.randn(1, conv_channels, 1, 1).astype(np.float16)}, state=state)
        print(f"  {label}: ✅ SUCCESS")
        del loaded
    except Exception as e:
        err = str(e)
        if "ANE" in err or "execution plan" in err or "not loaded" in err:
            print(f"  {label}: ❌ FAIL")
        else:
            print(f"  {label}: ❌ {err[:100]}")

print("=" * 60)
print("conv_state shape investigation")
print("=" * 60)

# Test 3D states with different sizes
print("\n3D states (varying second dim):")
test_state("3D (1, 64, 4)", "cs", (1, 64, 4))
test_state("3D (1, 256, 4)", "cs", (1, 256, 4))
test_state("3D (1, 1024, 4)", "cs", (1, 1024, 4))
test_state("3D (1, 4096, 4)", "cs", (1, 4096, 4))
test_state("3D (1, 8192, 4)", "cs", (1, 8192, 4))
test_state("3D (1, 16384, 4)", "cs", (1, 16384, 4))

# Test 3D states with different last dim
print("\n3D states (varying last dim):")
test_state("3D (1, 64, 1)", "cs", (1, 64, 1))
test_state("3D (1, 64, 2)", "cs", (1, 64, 2))
test_state("3D (1, 64, 8)", "cs", (1, 64, 8))
test_state("3D (1, 64, 16)", "cs", (1, 64, 16))

# Test 4D version of conv_state (reshape)
print("\n4D states (reshape of conv_state):")
test_state("4D (1, 8192, 4, 1)", "cs", (1, 8192, 4, 1))
test_state("4D (1, 8192, 1, 4)", "cs", (1, 8192, 1, 4))
test_state("4D (1, 4, 8192, 1)", "cs", (1, 4, 8192, 1))

# Test 2D state
print("\n2D states:")
test_state("2D (64, 4)", "cs", (64, 4))
test_state("2D (8192, 4)", "cs", (8192, 4))
