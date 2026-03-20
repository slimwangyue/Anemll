#!/usr/bin/env python3
"""Test if continuation_validation models work on ANE.
Also test a large stateless model to check if model size matters.
"""
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import os

# Test 1: continuation_validation decode model
print("=" * 60)
print("Test 1: continuation_validation decode_step_0_24_embeds")
print("=" * 60)

cv_path = "/Users/yw68/local_llm/dev_model_assets/continuation_validation/decode_step_0_24_embeds.mlpackage"
if os.path.exists(cv_path):
    model = ct.models.MLModel(cv_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    print(f"  Inputs:")
    for inp in spec.description.input:
        if inp.type.HasField("multiArrayType"):
            print(f"    {inp.name}: shape={tuple(inp.type.multiArrayType.shape)}")
        elif inp.type.HasField("stateType"):
            print(f"    {inp.name}: STATE shape={tuple(inp.type.stateType.multiArrayType.shape)}")
    print(f"  States ({len(spec.description.state)}):")
    for st in spec.description.state:
        print(f"    {st.name}")
    from collections import Counter
    op_counts = Counter()
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                op_counts[op.type] += 1
    print(f"  Ops ({sum(op_counts.values())} total): top 5:")
    for t, c in op_counts.most_common(5):
        print(f"    {t}: {c}")
    del model

    # Try on ANE
    try:
        model2 = ct.models.MLModel(cv_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = model2.make_state()
        print("  ✅ Loaded on CPU_AND_NE (has states)")
        # Can't easily predict without knowing exact input format
        del model2
    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:200]}")
else:
    print(f"  SKIP: {cv_path} not found")

# Test 2: Large model (similar weight size) WITHOUT states
print("\n" + "=" * 60)
print("Test 2: Large stateless model (~200MB weights)")
print("=" * 60)

class LargeModel(nn.Module):
    """~200MB model: 30 Conv2d layers of dim 2560"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(30):
            self.layers.append(nn.Conv2d(2560, 2560, 1, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

print("  Creating ~200MB conv model...")
m = LargeModel().eval().half()
param_bytes = sum(p.numel() * 2 for p in m.parameters())
print(f"  Weight size: {param_bytes/1024/1024:.0f}MB")

x = torch.randn(1, 2560, 1, 1, dtype=torch.float16)
traced = torch.jit.trace(m, x)
mlm = ct.convert(
    traced,
    inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="y", dtype=np.float16)],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
)
large_path = "/tmp/qwen35_ane_test/large_stateless.mlpackage"
mlm.save(large_path)

loaded = ct.models.MLModel(large_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
try:
    out = loaded.predict({"x": np.random.randn(1, 2560, 1, 1).astype(np.float16)})
    print(f"  ✅ {param_bytes/1024/1024:.0f}MB stateless model SUCCESS on ANE")
except Exception as e:
    if "ANE" in str(e):
        print(f"  ❌ {param_bytes/1024/1024:.0f}MB stateless model FAILED on ANE") 
    else:
        print(f"  ❌ Error: {str(e)[:200]}")
del loaded

# Test 3: Model with 4 states and LARGE recurrent state
print("\n" + "=" * 60)
print("Test 3: Model with 4 Qwen3.5-sized states + conv layers")
print("=" * 60)

class StatefulConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2560, 2560, 1, bias=False)
        self.register_buffer("k_cache", torch.zeros(8, 4, 256, 256, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(8, 4, 256, 256, dtype=torch.float16))
        self.register_buffer("conv_state", torch.zeros(8, 8192, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(8, 32, 128, 128, dtype=torch.float16))

    def forward(self, x):
        h = self.conv(x)
        v = h.view(1, 1, 2560)
        # Static writes to states
        self.k_cache[0:1, 0:1, 0:1, 0:256] = v[:, :, :256].view(1, 1, 1, 256)
        self.v_cache[0:1, 0:1, 0:1, 0:256] = v[:, :, :256].view(1, 1, 1, 256)
        self.conv_state[0:1, 0:4, 0:1] = v[:, :, :4].view(1, 4, 1)
        self.rec_state[0:1, 0:1, 0:1, 0:128] = v[:, :, :128].view(1, 1, 1, 128)
        return h

ms = StatefulConvModel().eval().half()
x = torch.randn(1, 2560, 1, 1, dtype=torch.float16)
traced = torch.jit.trace(ms, x)

states_list = [
    ct.StateType(wrapped_type=ct.TensorType(shape=(8, 4, 256, 256), dtype=np.float16), name="k_cache"),
    ct.StateType(wrapped_type=ct.TensorType(shape=(8, 4, 256, 256), dtype=np.float16), name="v_cache"),
    ct.StateType(wrapped_type=ct.TensorType(shape=(8, 8192, 4), dtype=np.float16), name="conv_state"),
    ct.StateType(wrapped_type=ct.TensorType(shape=(8, 32, 128, 128), dtype=np.float16), name="rec_state"),
]
mlm = ct.convert(
    traced,
    inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="y", dtype=np.float16)],
    states=states_list,
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
)
ms_path = "/tmp/qwen35_ane_test/stateful_conv_test.mlpackage"
mlm.save(ms_path)
loaded = ct.models.MLModel(ms_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
state = loaded.make_state()
try:
    out = loaded.predict({"x": np.random.randn(1, 2560, 1, 1).astype(np.float16)}, state=state)
    print("  ✅ 4-state + conv model SUCCESS on ANE")
except Exception as e:
    if "ANE" in str(e):
        print("  ❌ 4-state + conv model FAILED on ANE")
    else:
        print(f"  ❌ Error: {str(e)[:200]}")
