#!/usr/bin/env python3
"""Test various models on ANE to isolate the failure point.
1. Test embeddings model (simplest)
2. Test minimal torch model on ANE
3. Test decode chunk with CPU_AND_GPU (confirm works off ANE)
"""
import coremltools as ct
import numpy as np
import torch
import os
import time

print("=" * 60)
print("Test 1: Qwen3.5 Embeddings on ANE")
print("=" * 60)

embed_path = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export/qwen35_embeddings.mlpackage"
if os.path.exists(embed_path):
    model = ct.models.MLModel(embed_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    inp = {"input_ids": np.array([[42]], dtype=np.int32)}
    try:
        out = model.predict(inp)
        print("  ✅ Embeddings SUCCESS on CPU_AND_NE")
    except Exception as e:
        err = str(e)
        if "ANE" in err:
            print("  ❌ Embeddings FAILED on ANE")
        else:
            print(f"  ❌ Error: {err[:200]}")
    del model
else:
    print("  SKIP: not found")

print("\n" + "=" * 60)
print("Test 2: Minimal Conv2d model on ANE")
print("=" * 60)

class MinimalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2560, 2560, 1, bias=False)
    def forward(self, x):
        # x: (1, 2560, 1, 1)
        return self.conv(x)

m = MinimalModel().eval().half()
x = torch.randn(1, 2560, 1, 1, dtype=torch.float16)
traced = torch.jit.trace(m, x)
mlm = ct.convert(
    traced,
    inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="y", dtype=np.float16)],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_path = "/tmp/qwen35_ane_test/minimal_test.mlpackage"
mlm.save(mlm_path)
loaded = ct.models.MLModel(mlm_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
try:
    out = loaded.predict({"x": np.random.randn(1, 2560, 1, 1).astype(np.float16)})
    print("  ✅ Minimal Conv2d SUCCESS on ANE")
except Exception as e:
    print(f"  ❌ Minimal Conv2d FAILED: {str(e)[:200]}")
del loaded

print("\n" + "=" * 60)
print("Test 3: Minimal model with StateType on ANE")
print("=" * 60)

class MinimalStateful(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 4, 256, 128, dtype=torch.float16))

    def forward(self, x):
        # x: (1, 128) - single vector
        self.cache[:, :, 0:1, :] = x.view(1, 1, 1, 128).expand(1, 4, 1, 128)
        return self.cache[:, :, 0:1, :].sum(dim=-1)

ms = MinimalStateful().eval().half()
x_in = torch.randn(1, 128, dtype=torch.float16)
traced_s = torch.jit.trace(ms, x_in)

states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, 4, 256, 128), dtype=np.float16),
        name="cache",
    )
]
mlm_s = ct.convert(
    traced_s,
    inputs=[ct.TensorType(name="x", shape=x_in.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="y", dtype=np.float16)],
    states=states,
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
)
ms_path = "/tmp/qwen35_ane_test/minimal_state_test.mlpackage"
mlm_s.save(ms_path)
loaded_s = ct.models.MLModel(ms_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
state = loaded_s.make_state()
try:
    out = loaded_s.predict({"x": np.random.randn(1, 128).astype(np.float16)}, state=state)
    print("  ✅ Minimal Stateful SUCCESS on ANE")
except Exception as e:
    err = str(e)
    if "ANE" in err:
        print("  ❌ Minimal Stateful FAILED on ANE")
    else:
        print(f"  ❌ Error: {err[:200]}")
del loaded_s

print("\n" + "=" * 60)
print("Test 4: Qwen3.5 Decode chunk on CPU_AND_GPU (sanity)")
print("=" * 60)

decode_path = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export/qwen35_FFN_chunk_01of04.mlpackage"
if os.path.exists(decode_path):
    model = ct.models.MLModel(decode_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    state = model.make_state()
    hidden = np.zeros((1, 1, 2560), dtype=np.float16)
    position_ids = np.zeros((1,), dtype=np.int32)
    mask = np.zeros((1, 1, 1, 256), dtype=np.float16)
    current_pos = np.zeros((1,), dtype=np.int32)
    try:
        out = model.predict({
            "hidden_states": hidden,
            "position_ids": position_ids,
            "causal_mask": mask,
            "current_pos": current_pos,
        }, state=state)
        print("  ✅ Decode SUCCESS on CPU_AND_GPU")
    except Exception as e:
        print(f"  ❌ Decode FAILED on CPU_AND_GPU: {str(e)[:200]}")
    del model
else:
    print("  SKIP: not found")
