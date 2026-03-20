#!/usr/bin/env python3
"""The issue is the state shape. (1, 8192, 4) fails while (1, 32, 128, 128) passes.
Testing dimension thresholds and alternative reshaping strategies.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
import os
import shutil

OUT_DIR = "/tmp/qwen35_ane_test/op_tests7"
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN = 2560
CONV_DIM = 8192

results = []

def test_model(name, model, input_shapes, state_shapes):
    path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    if os.path.exists(path):
        shutil.rmtree(path)
    
    example_inputs = []
    ct_inputs = []
    for iname, shape, dtype in input_shapes:
        t = torch.randn(*shape, dtype=torch.float16) * 0.01 if dtype == np.float16 else torch.zeros(*shape, dtype=torch.int32)
        example_inputs.append(t)
        ct_inputs.append(ct.TensorType(name=iname, shape=shape, dtype=dtype))
    
    ct_states = [ct.StateType(wrapped_type=ct.TensorType(shape=s, dtype=d), name=n) for n, s, d in state_shapes]
    
    traced = torch.jit.trace(model, tuple(example_inputs))
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=ct_states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    
    spec = mlmodel.get_spec()
    from collections import Counter
    op_counts = Counter()
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                if op.type != "const":
                    op_counts[op.type] += 1
    print(f"  Ops: {dict(op_counts)}")
    
    mlmodel.save(path)
    del mlmodel
    
    try:
        loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        inputs_dict = {}
        for iname, shape, dtype in input_shapes:
            inputs_dict[iname] = np.random.randn(*shape).astype(np.float16) * 0.01 if dtype == np.float16 else np.zeros(shape, dtype=np.int32)
        state = loaded.make_state()
        out = loaded.predict(inputs_dict, state=state)
        print(f"  ✅ {name}: PASSED")
        results.append((name, True))
        del loaded
        return True
    except Exception as e:
        msg = str(e)[:200]
        if "not loaded" in msg:
            print(f"  ❌ {name}: FAILED TO LOAD")
        else:
            print(f"  ❌ {name}: FAILED")
        results.append((name, False))
        return False


# Test different state shapings for same total size = 8192*4 = 32768 elements
def make_test_model(state_shape, name_suffix=""):
    class TestModel(nn.Module):
        def __init__(self, sshape):
            super().__init__()
            self.sshape = sshape
            dim1 = sshape[1]
            self.proj = nn.Conv2d(HIDDEN, dim1, 1, bias=False, dtype=torch.float16)
            self.register_buffer("state", torch.zeros(*sshape, dtype=torch.float16))
        def forward(self, x):
            y = self.proj(x)  # [1, dim1, 1, 1]
            s = self.state
            ndim = len(self.sshape)
            if ndim == 3:
                expand = y.squeeze(3).squeeze(2)  # [1, dim1]
                new = s + expand.unsqueeze(-1) * 0.001
            elif ndim == 4:
                expand = y.squeeze(3).squeeze(2)
                new = s + expand.unsqueeze(-1).unsqueeze(-1) * 0.001
            else:
                new = s + 0.001
            self.state[:] = new.to(torch.float16)
            return y
    return TestModel(state_shape).eval()


# Different reshapings of 8192*4 = 32768 elements
shapes_to_test = [
    ("V1_8192x4", (1, 8192, 4)),          # Original failing shape
    ("V2_4096x8", (1, 4096, 8)),          # Halved first dim
    ("V3_2048x16", (1, 2048, 16)),        # Quarter first dim
    ("V4_1024x32", (1, 1024, 32)),        # 1/8th
    ("V5_512x64", (1, 512, 64)),          # 1/16th
    ("V6_256x128", (1, 256, 128)),
    ("V7_128x256", (1, 128, 256)),
    ("V8_64x512", (1, 64, 512)),
    ("V9_32x1024", (1, 32, 1024)),
    # 4D reshapings
    ("V10_64x128x4", (1, 64, 128, 4)),    # 4D: 64*128*4 = 32768
    ("V11_128x64x4", (1, 128, 64, 4)),
    ("V12_32x256x4", (1, 32, 256, 4)),
    ("V13_256x32x4", (1, 256, 32, 4)),
    # Specific test: the actual shape we want for conv_state
    ("V14_8192x1x4", (1, 8192, 1, 4)),
]

for name, shape in shapes_to_test:
    print(f"\n=== {name}: {shape} ===")
    m = make_test_model(shape)
    test_model(name, m,
        [("x", (1, HIDDEN, 1, 1), np.float16)],
        [("state", shape, np.float16)])


# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, passed in results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
