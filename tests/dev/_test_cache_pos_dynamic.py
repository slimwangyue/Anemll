#!/usr/bin/env python3
"""Definitive test: does pos:pos+1 slice_update ACTUALLY write to different positions on ANE?"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch
import numpy as np
import coremltools as ct
import shutil, os

class CacheModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Small cache: (1, 2, 8, 4) — 2 heads, 8 positions, dim 4
        self.register_buffer("cache", torch.zeros(1, 2, 8, 4, dtype=torch.float16))

    def forward(self, x, pos):
        # Write x at position pos
        self.cache[:, :, pos:pos+1, :] = x
        # Read back entire cache and return it
        return self.cache.clone()

model = CacheModel().eval()
x = torch.ones(1, 2, 1, 4, dtype=torch.float16)
pos = torch.tensor([0], dtype=torch.int32)

traced = torch.jit.trace(model, (x, pos))

states = [ct.StateType(wrapped_type=ct.TensorType(shape=(1, 2, 8, 4), dtype=np.float16), name="cache")]

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="x", shape=(1, 2, 1, 4), dtype=np.float16),
        ct.TensorType(name="pos", shape=(1,), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="output", dtype=np.float16)],
    states=states,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
    convert_to="mlprogram",
)

pkg = "/tmp/test_cache_pos.mlpackage"
if os.path.exists(pkg): shutil.rmtree(pkg)
mlmodel.save(pkg)
del mlmodel

def test_backend(name, compute_unit):
    print(f"\n=== {name} ===")
    cml = ct.models.MLModel(pkg, compute_units=compute_unit)
    state = cml.make_state()
    
    # Write 1.0 at pos=0
    x_val = np.ones((1, 2, 1, 4), dtype=np.float16)
    out = cml.predict({"x": x_val, "pos": np.array([0], dtype=np.int32)}, state=state)
    cache_after_pos0 = out["output"]
    print(f"After write at pos=0:")
    print(f"  cache[0,0,:,0] = {cache_after_pos0[0,0,:,0]}")  # Should be [1,0,0,0,0,0,0,0]
    
    # Write 2.0 at pos=3
    x_val2 = np.ones((1, 2, 1, 4), dtype=np.float16) * 2.0
    out2 = cml.predict({"x": x_val2, "pos": np.array([3], dtype=np.int32)}, state=state)
    cache_after_pos3 = out2["output"]
    print(f"After write at pos=3:")
    print(f"  cache[0,0,:,0] = {cache_after_pos3[0,0,:,0]}")  # Should be [1,0,0,2,0,0,0,0]
    
    # Write 3.0 at pos=7
    x_val3 = np.ones((1, 2, 1, 4), dtype=np.float16) * 3.0
    out3 = cml.predict({"x": x_val3, "pos": np.array([7], dtype=np.int32)}, state=state)
    cache_after_pos7 = out3["output"]
    print(f"After write at pos=7:")
    print(f"  cache[0,0,:,0] = {cache_after_pos7[0,0,:,0]}")  # Should be [1,0,0,2,0,0,0,3]
    
    expected = np.array([1, 0, 0, 2, 0, 0, 0, 3], dtype=np.float16)
    actual = cache_after_pos7[0, 0, :, 0]
    if np.allclose(actual, expected, atol=0.01):
        print(f"  RESULT: Dynamic pos writes WORK correctly!")
    else:
        # Check if pos is always stuck at 0
        always_pos0 = np.array([3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float16)
        if np.allclose(actual, always_pos0, atol=0.01):
            print(f"  RESULT: pos is FROZEN at 0 (trace-time value). All writes go to pos=0.")
        else:
            print(f"  RESULT: Unexpected behavior!")
            print(f"    Expected: {expected}")
            print(f"    Got:      {actual}")
    del cml, state

test_backend("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)
test_backend("CPU_AND_NE (ANE)", ct.ComputeUnit.CPU_AND_NE)

print("\nDone.")
