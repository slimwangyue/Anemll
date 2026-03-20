#!/usr/bin/env python3
"""Systematically test which StateType configurations break ANE.
Test each state individually, then combinations.
"""
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import os

def test_model_with_states(label, states_config, conv_channels=2560):
    """Create a conv model with specific states and test on ANE.
    states_config: list of (name, shape) tuples
    """
    print(f"\n  {label}:")

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(conv_channels, conv_channels, 1, bias=False)
            for name, shape in states_config:
                self.register_buffer(name, torch.zeros(*shape, dtype=torch.float16))

        def forward(self, x):
            h = self.conv(x)
            # Write to each state with static bounds
            for name, shape in states_config:
                buf = getattr(self, name)
                # Write a small slice - just [0:1] on first dim
                sl = [slice(0, 1)] + [slice(0, 1)] * (len(shape) - 1)
                val_shape = [1] * len(shape)
                buf[tuple(sl)] = torch.zeros(*val_shape, dtype=torch.float16)
            return h

    m = TestModel().eval().half()
    x = torch.randn(1, conv_channels, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(m, x)

    ct_states = []
    for name, shape in states_config:
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=shape, dtype=np.float16),
            name=name,
        ))

    try:
        mlm = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            states=ct_states,
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
        )
    except Exception as e:
        print(f"    CONVERT ERROR: {str(e)[:200]}")
        return

    path = f"/tmp/qwen35_ane_test/state_test_{label.replace(' ', '_')}.mlpackage"
    mlm.save(path)

    try:
        loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = loaded.make_state()
        out = loaded.predict({"x": np.random.randn(1, conv_channels, 1, 1).astype(np.float16)}, state=state)
        print(f"    ✅ SUCCESS on ANE")
        del loaded
    except Exception as e:
        err = str(e)
        if "execution plan" in err or "error code" in err:
            print(f"    ❌ COMPILE FAIL (execution plan error)")
        elif "ANE" in err:
            print(f"    ❌ ANE INFERENCE FAIL")
        elif "not loaded" in err:
            print(f"    ❌ LOAD FAIL (cannot make state)")
        else:
            print(f"    ❌ ERROR: {err[:200]}")

    # Cleanup
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


print("=" * 60)
print("Testing StateType configurations on ANE")
print("=" * 60)

# Qwen3.5 state shapes
K_CACHE = (8, 4, 256, 256)    # 8 layers, 4 kv_heads, 256 seq, 256 head_dim
V_CACHE = (8, 4, 256, 256)
CONV_STATE = (8, 8192, 4)      # 8 layers, 8192 conv_dim, 4 kernel
REC_STATE = (8, 32, 128, 128)  # 8 layers, 32 v_heads, 128 key_dim, 128 val_dim

# 1. Individual states
test_model_with_states("k_cache only", [("k_cache", K_CACHE)])
test_model_with_states("v_cache only", [("v_cache", V_CACHE)])
test_model_with_states("conv_state only", [("conv_state", CONV_STATE)])
test_model_with_states("rec_state only", [("rec_state", REC_STATE)])

# 2. Two states
test_model_with_states("k+v cache", [("k_cache", K_CACHE), ("v_cache", V_CACHE)])
test_model_with_states("conv+rec state", [("conv_state", CONV_STATE), ("rec_state", REC_STATE)])

# 3. Three states
test_model_with_states("k+v+conv", [("k_cache", K_CACHE), ("v_cache", V_CACHE), ("conv_state", CONV_STATE)])

# 4. All four
test_model_with_states("all 4 states", [("k_cache", K_CACHE), ("v_cache", V_CACHE), ("conv_state", CONV_STATE), ("rec_state", REC_STATE)])

# 5. Try smaller state sizes
print("\n" + "=" * 60)
print("Testing smaller state sizes")
print("=" * 60)
test_model_with_states("small k_cache (2,4,256,256)", [("k_cache", (2, 4, 256, 256))])
test_model_with_states("small rec_state (2,32,128,128)", [("rec_state", (2, 32, 128, 128))])
test_model_with_states("tiny rec_state (1,32,128,128)", [("rec_state", (1, 32, 128, 128))])
test_model_with_states("small all 4", [
    ("k_cache", (2, 4, 256, 256)),
    ("v_cache", (2, 4, 256, 256)),
    ("conv_state", (2, 8192, 4)),
    ("rec_state", (2, 32, 128, 128)),
])
