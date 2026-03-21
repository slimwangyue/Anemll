#!/usr/bin/env python3
"""Check if pos:pos+1 slice_update is static or dynamic after JIT trace + CoreML convert."""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch
import numpy as np
import coremltools as ct

# Minimal model with dynamic pos:pos+1 cache write
class DynSliceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 4, 8, 16, dtype=torch.float16))

    def forward(self, x, pos):
        # x: (1, 4, 1, 16), pos: (1,) int32
        self.cache[:, :, pos:pos+1, :] = x
        out = self.cache[:, :, :, :].sum(dim=-1, keepdim=True)
        return out

model = DynSliceModel().eval()
x = torch.randn(1, 4, 1, 16, dtype=torch.float16)
pos = torch.tensor([0], dtype=torch.int32)

traced = torch.jit.trace(model, (x, pos))

# Check the traced graph
print("=== JIT Trace Graph ===")
print(traced.graph)
print()

# Convert to CoreML
states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, 4, 8, 16), dtype=np.float16),
        name="cache",
    )
]

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="x", shape=(1, 4, 1, 16), dtype=np.float16),
        ct.TensorType(name="pos", shape=(1,), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="output", dtype=np.float16)],
    states=states,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
    convert_to="mlprogram",
)

# Examine MIL ops
print("\n=== MIL Program (looking for slice_update / scatter) ===")
spec = mlmodel.get_spec()
mil_prog_str = str(spec)

# Search for key ops
for keyword in ["slice_update", "scatter", "slice_by_index", "slice_by_size"]:
    count = mil_prog_str.count(keyword)
    if count > 0:
        print(f"  Found '{keyword}': {count} occurrences")

# Save and test
pkg = "/tmp/test_dynslice.mlpackage"
import shutil, os
if os.path.exists(pkg):
    shutil.rmtree(pkg)
mlmodel.save(pkg)

# Test on CPU
print("\n=== CPU Test ===")
cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_ONLY)
state = cml.make_state()
# pos=0
out0 = cml.predict({"x": x.numpy(), "pos": np.array([0], dtype=np.int32)}, state=state)
print(f"  pos=0: OK, shape={list(out0.values())[0].shape}")
# pos=3
out3 = cml.predict({"x": x.numpy(), "pos": np.array([3], dtype=np.int32)}, state=state)
print(f"  pos=3: OK, shape={list(out3.values())[0].shape}")
del cml, state

# Test on ANE
print("\n=== ANE Test ===")
try:
    cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = cml.make_state()
    out0 = cml.predict({"x": x.numpy(), "pos": np.array([0], dtype=np.int32)}, state=state)
    print(f"  pos=0: OK")
    out3 = cml.predict({"x": x.numpy(), "pos": np.array([3], dtype=np.int32)}, state=state)
    print(f"  pos=3: OK")
    # Check if pos actually matters - compare with different pos values
    state2 = cml.make_state()
    x_val = np.ones((1, 4, 1, 16), dtype=np.float16) * 99
    cml.predict({"x": x_val, "pos": np.array([0], dtype=np.int32)}, state=state2)
    out_a = cml.predict({"x": np.zeros((1, 4, 1, 16), dtype=np.float16), "pos": np.array([0], dtype=np.int32)}, state=state2)
    
    state3 = cml.make_state()
    cml.predict({"x": x_val, "pos": np.array([5], dtype=np.int32)}, state=state3)
    out_b = cml.predict({"x": np.zeros((1, 4, 1, 16), dtype=np.float16), "pos": np.array([0], dtype=np.int32)}, state=state3)
    
    val_a = list(out_a.values())[0]
    val_b = list(out_b.values())[0]
    print(f"  Write at pos=0 then read sum: {val_a.flatten()[:4]}")
    print(f"  Write at pos=5 then read sum: {val_b.flatten()[:4]}")
    if np.allclose(val_a, val_b, atol=0.01):
        print(f"  RESULT: pos argument is IGNORED by ANE (writes always go to same position)")
    else:
        print(f"  RESULT: pos argument WORKS - different positions produce different sums")
    del cml
except Exception as e:
    print(f"  ANE FAILED: {e}")

print("\nDone.")
