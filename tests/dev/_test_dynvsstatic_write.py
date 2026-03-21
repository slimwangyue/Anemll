#!/usr/bin/env python3
"""Compare DYNAMIC vs STATIC slice_update on CoreML state buffers.

Answers: what exactly happens when cache[:, :, pos:pos+1, :] = x
where pos comes from an input tensor?
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch, numpy as np, coremltools as ct, os, shutil, gc

class DynWriteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 2, 8, 4, dtype=torch.float16))
    def forward(self, x, pos):
        self.cache[:, :, pos:pos+1, :] = x  # DYNAMIC write
        return self.cache.clone()

class StaticWriteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 2, 8, 4, dtype=torch.float16))
    def forward(self, x, pos):
        self.cache[:, :, 0:1, :] = x         # STATIC write (always pos 0)
        return self.cache.clone()

pos = torch.tensor([0], dtype=torch.int32)
x = torch.ones(1, 2, 1, 4, dtype=torch.float16)

for name, ModelClass in [("DYNAMIC", DynWriteModel), ("STATIC", StaticWriteModel)]:
    print(f"\n{'='*50}")
    print(f"  {name} write test")
    print(f"{'='*50}")
    model = ModelClass().eval()
    traced = torch.jit.trace(model, (x, pos))

    # Print graph lines related to slicing
    print("\nJIT Graph (key ops):")
    for line in str(traced.graph).split("\n"):
        s = line.strip()
        if any(k in s for k in ["aten::Int", "aten::add", "aten::slice", "aten::copy_"]):
            print(f"  {s}")

    states = [ct.StateType(wrapped_type=ct.TensorType(shape=(1, 2, 8, 4), dtype=np.float16), name="cache")]
    pkg = f"/tmp/test_{name.lower()}_write.mlpackage"
    if os.path.exists(pkg): shutil.rmtree(pkg)
    mlmodel = ct.convert(traced,
        inputs=[ct.TensorType(name="x", shape=(1, 2, 1, 4), dtype=np.float16),
                ct.TensorType(name="pos", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="out", dtype=np.float16)],
        states=states, compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18, convert_to="mlprogram")
    mlmodel.save(pkg)
    del mlmodel, traced, model; gc.collect()

    # Test on CPU — write different values to different positions
    print("\nCPU_ONLY results:")
    cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_ONLY)
    state = cml.make_state()
    for write_pos, write_val in [(0, 1.0), (3, 2.0), (5, 3.0)]:
        xv = np.ones((1, 2, 1, 4), dtype=np.float16) * write_val
        out = cml.predict({"x": xv, "pos": np.array([write_pos], dtype=np.int32)}, state=state)
        row = out["out"][0, 0, :, 0]
        print(f"  write val={write_val:.0f} at pos={write_pos}: cache[:] = {row}")
    del cml, state; gc.collect()

print("\n\nExpected for DYNAMIC if pos worked: [1, 0, 0, 2, 0, 3, 0, 0]")
print("Expected for DYNAMIC if pos frozen:  [3, 0, 0, 0, 0, 0, 0, 0] (all writes go to pos=0)")
print("Expected for STATIC:                 [3, 0, 0, 0, 0, 0, 0, 0] (always pos=0)")
