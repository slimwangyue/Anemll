#!/usr/bin/env python3
"""Test 1: Check if decode model's slice_updates are dynamic.
Test 2: Test with 4 states matching Qwen3.5 sizes on ANE.
Test 3: Test reduce_sum + exp + sigmoid + softplus pattern on ANE.
"""
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# ===== Test 1: Check decode model slice_update deps =====
print("=" * 60)
print("Test 1: Decode model slice_update dependencies")
print("=" * 60)

DECODE_PATH = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export/qwen35_FFN_chunk_01of04.mlpackage"
model = ct.models.MLModel(DECODE_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)
spec = model.get_spec()
prog = spec.mlProgram

model_inputs = set()
for inp in spec.description.input:
    model_inputs.add(inp.name)

producer = {}
for fn in prog.functions.values():
    for blk in fn.block_specializations.values():
        for op in blk.operations:
            for out in op.outputs:
                producer[out.name] = (op.type, op)

def is_dynamic(name, visited=None, depth=0):
    if visited is None:
        visited = set()
    if name in visited or depth > 10:
        return False
    visited.add(name)
    if name in model_inputs:
        return True
    if name not in producer:
        return False
    op_type, op = producer[name]
    if op_type == "const":
        return False
    for k, v in op.inputs.items():
        for arg in v.arguments:
            if hasattr(arg, 'name') and is_dynamic(arg.name, visited, depth+1):
                return True
    return False

su_idx = 0
dynamic_count = 0
for fn in prog.functions.values():
    for blk in fn.block_specializations.values():
        for op in blk.operations:
            if op.type == "slice_update":
                begin_name = None
                end_name = None
                for k, v in op.inputs.items():
                    for arg in v.arguments:
                        if hasattr(arg, 'name'):
                            if k == "begin":
                                begin_name = arg.name
                            elif k == "end":
                                end_name = arg.name
                begin_dyn = is_dynamic(begin_name) if begin_name else False
                end_dyn = is_dynamic(end_name) if end_name else False
                status = "STATIC"
                if begin_dyn or end_dyn:
                    status = "DYNAMIC"
                    dynamic_count += 1
                print(f"  slice_update #{su_idx}: {status} (begin={begin_name}, end={end_name})")
                su_idx += 1

print(f"\n  Total: {su_idx} slice_updates, {dynamic_count} DYNAMIC")
del model

# ===== Test 2: Multi-state model on ANE =====
print("\n" + "=" * 60)
print("Test 2: 4-state model matching Qwen3.5 sizes on ANE")
print("=" * 60)

class MultiStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2560, 2560, 1, bias=False)
        self.conv2 = nn.Conv2d(2560, 2560, 1, bias=False)
        # States
        self.register_buffer("k_cache", torch.zeros(8, 4, 256, 256, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(8, 4, 256, 256, dtype=torch.float16))
        self.register_buffer("conv_state", torch.zeros(8, 8192, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(8, 32, 128, 128, dtype=torch.float16))

    def forward(self, x):
        # x: (1, 2560, 1, 1)
        h = self.conv1(x)
        # Write to k_cache[0] at position 0
        kv = h.view(1, 4, 1, -1)[:, :, :, :256]
        self.k_cache[0:1, :, 0:1, :] = kv
        self.v_cache[0:1, :, 0:1, :] = kv
        # Write to conv_state[0]
        cs = h.view(1, -1)[:, :8192].unsqueeze(-1)[:, :, :1]
        # Shift conv_state left
        shifted = self.conv_state[0:1, :, 1:]
        new_col = cs.view(1, 8192, 1)
        self.conv_state[0:1] = torch.cat([shifted, new_col], dim=2)
        # Simple operation on rec_state
        rs = self.rec_state[0:1, 0:1, :, :]
        self.rec_state[0:1, 0:1, :, :] = rs * 0.99
        h2 = self.conv2(h)
        return h2

ms = MultiStateModel().eval().half()
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
mpath = "/tmp/qwen35_ane_test/multi_state_test.mlpackage"
mlm.save(mpath)
loaded = ct.models.MLModel(mpath, compute_units=ct.ComputeUnit.CPU_AND_NE)
state = loaded.make_state()
try:
    out = loaded.predict({"x": np.random.randn(1, 2560, 1, 1).astype(np.float16)}, state=state)
    print("  ✅ 4-state model SUCCESS on ANE")
except Exception as e:
    if "ANE" in str(e):
        print("  ❌ 4-state model FAILED on ANE")
    else:
        print(f"  ❌ Error: {str(e)[:200]}")
del loaded
