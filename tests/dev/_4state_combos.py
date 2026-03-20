#!/usr/bin/env python3
"""Test exact 4-state configurations for Qwen3.5 on ANE.
Test progressively adding states to find the breaking combination.
"""
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import os, shutil

def test_config(label, states_config, conv_channels=256):
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(conv_channels, conv_channels, 1, bias=False)
            for name, shape in states_config:
                self.register_buffer(name, torch.zeros(*shape, dtype=torch.float16))
        def forward(self, x):
            h = self.conv(x)
            for name, shape in states_config:
                buf = getattr(self, name)
                sl = tuple([slice(0, 1)] * len(shape))
                val = torch.zeros(*([1]*len(shape)), dtype=torch.float16)
                buf[sl] = val
            return h

    m = TestModel().eval().half()
    x = torch.randn(1, conv_channels, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(m, x)
    ct_states = [ct.StateType(wrapped_type=ct.TensorType(shape=s, dtype=np.float16), name=n)
                 for n, s in states_config]
    mlm = ct.convert(traced,
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        states=ct_states, compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18)
    path = "/tmp/qwen35_ane_test/_test4.mlpackage"
    if os.path.exists(path): shutil.rmtree(path)
    mlm.save(path)
    try:
        loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = loaded.make_state()
        out = loaded.predict({"x": np.random.randn(1, conv_channels, 1, 1).astype(np.float16)}, state=state)
        print(f"  {label}: ✅ SUCCESS")
        del loaded
        return True
    except Exception as e:
        err = str(e)
        if "execution plan" in err:
            print(f"  {label}: ❌ COMPILE FAIL")
        elif "ANE" in err:
            print(f"  {label}: ❌ ANE INFERENCE FAIL")
        elif "not loaded" in err:
            print(f"  {label}: ❌ LOAD FAIL")
        else:
            print(f"  {label}: ❌ {err[:100]}")
        return False

# Full-size Qwen3.5 state shapes
K = ("k_cache", (8, 4, 256, 256))
V = ("v_cache", (8, 4, 256, 256))
C = ("conv_state", (8, 8192, 4))
R = ("rec_state", (8, 32, 128, 128))

print("Full-size 4-state combos:")
test_config("k only", [K])
test_config("k+v", [K, V])
test_config("k+v+c", [K, V, C])
test_config("k+v+c+r (all 4)", [K, V, C, R])
test_config("k+v+r (no conv)", [K, V, R])
test_config("k+c (no v, no rec)", [K, C])
test_config("c+r (conv+rec only)", [C, R])
test_config("v+c+r (no k)", [V, C, R])

# Small-size 4-state
Ks = ("k_cache", (2, 4, 256, 256))
Vs = ("v_cache", (2, 4, 256, 256))
Cs = ("conv_state", (2, 8192, 4))
Rs = ("rec_state", (2, 32, 128, 128))

print("\nSmall (2 layer) 4-state combos:")
test_config("small all 4", [Ks, Vs, Cs, Rs])
test_config("small k+v+c", [Ks, Vs, Cs])
test_config("small k+v+r", [Ks, Vs, Rs])
test_config("small c+r", [Cs, Rs])
