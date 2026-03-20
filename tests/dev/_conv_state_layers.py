#!/usr/bin/env python3
"""Test conv_state with increasing first dim (number of layers)."""
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import os
import shutil

def test_state(label, name, shape, conv_channels=256):
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

    path = f"/tmp/qwen35_ane_test/cs_layer_test.mlpackage"
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

print("conv_state layer scaling (N, 8192, 4):")
for n in [1, 2, 3, 4, 5, 6, 7, 8]:
    test_state(f"({n}, 8192, 4)", "cs", (n, 8192, 4))

print("\nconv_state with 2 states (N, 8192, 4) + k_cache (N, 4, 256, 256):")
# Previous test showed "small all 4" with (2,8192,4) failed
# Let's check if it's the combo or the size
class MultiTestModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, 1, bias=False)
        self.register_buffer("cs", torch.zeros(n, 8192, 4, dtype=torch.float16))
        self.register_buffer("kc", torch.zeros(n, 4, 256, 256, dtype=torch.float16))
    def forward(self, x):
        h = self.conv(x)
        self.cs[0:1, 0:1, 0:1] = torch.zeros(1, 1, 1, dtype=torch.float16)
        self.kc[0:1, 0:1, 0:1, 0:1] = torch.zeros(1, 1, 1, 1, dtype=torch.float16)
        return h

for n in [1, 2, 4, 8]:
    m = MultiTestModel(n).eval().half()
    x = torch.randn(1, 256, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(m, x)
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=(n, 8192, 4), dtype=np.float16), name="cs"),
        ct.StateType(wrapped_type=ct.TensorType(shape=(n, 4, 256, 256), dtype=np.float16), name="kc"),
    ]
    mlm = ct.convert(traced,
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        states=states, compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18)
    path = "/tmp/qwen35_ane_test/multi_test.mlpackage"
    if os.path.exists(path):
        shutil.rmtree(path)
    mlm.save(path)
    try:
        loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = loaded.make_state()
        out = loaded.predict({"x": np.random.randn(1, 256, 1, 1).astype(np.float16)}, state=state)
        print(f"  cs({n},8192,4)+kc({n},4,256,256): ✅")
        del loaded
    except:
        print(f"  cs({n},8192,4)+kc({n},4,256,256): ❌")
