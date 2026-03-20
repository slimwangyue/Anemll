#!/usr/bin/env python3
"""Final narrowing. The issue might be the number of slice_update ops
or how conv (projection) interacts with state.

Key finding from T2 vs T3:
  T2 (roll, no proj): PASS — ops: read_state, gather, slice_update, write_state
  T10 (matmul shift, no proj): PASS 
  T3 (matmul + proj + slice_update x2): FAIL

And T1: adding proj + math to state causes the FAIL.
But D8: adding proj + math (different code path) to state: PASS.

Let me check: is it the SIZE? (1, 8192, 4) vs (1, 32, 128, 128)?
Or the SHAPE (3D vs 4D)?

Tests:
  U1: (1, 8192, 4) state + conv proj + full write (same as T1 but verify)
  U2: (1, 8192, 1, 4) state (4D) + conv proj + full write
  U3: (1, 32, 128, 128) state + conv proj + full write (same shape as rec_state)
  U4: (1, 8192, 4) state + matmul shift + write back as SliceUpdate entire state
  U5: (1, 8192, 4) but state *= 0.9 only (no conv input, minimal ops)
  U6: Replicate T10 but with conv proj to see if proj is the trigger
  U7: conv_state shift via roll + conv proj (roll worked in T2)
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
import os
import shutil

OUT_DIR = "/tmp/qwen35_ane_test/op_tests6"
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


# ============================================================
print("\n=== U1: (1, 8192, 4) + conv proj + full write ===")
class TestU1(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        new = self.state + y.unsqueeze(-1) * 0.1
        self.state[:] = new.to(torch.float16)
        return new[:, :32, :].reshape(1, 32, 1, 4).to(torch.float16)

test_model("U1_8192x4_conv_full", TestU1().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("state", (1, CONV_DIM, 4), np.float16)])

# ============================================================
print("\n=== U2: (1, 8192, 1, 4) 4D + conv + full ===")
class TestU2(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("state", torch.zeros(1, CONV_DIM, 1, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x)  # [1, 8192, 1, 1]
        new = self.state + y * 0.1
        self.state[:] = new.to(torch.float16)
        return new[:, :32, :, :].to(torch.float16)

test_model("U2_8192x1x4_conv_full", TestU2().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("state", (1, CONV_DIM, 1, 4), np.float16)])

# ============================================================
print("\n=== U3: (1, 32, 128, 128) + conv + full ===")
class TestU3(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, 32, 1, bias=False, dtype=torch.float16)
        self.register_buffer("state", torch.zeros(1, 32, 128, 128, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)  # [1, 32]
        new = self.state + y.unsqueeze(-1).unsqueeze(-1) * 0.001
        self.state[:] = new.to(torch.float16)
        return new[:, :, 0, :].reshape(1, 32, 1, 128).to(torch.float16)

test_model("U3_32x128x128_conv_full", TestU3().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("state", (1, 32, 128, 128), np.float16)])

# ============================================================
print("\n=== U4: (1, 8192, 4) state + NO proj, just scale + write ===")
class TestU4(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        new = self.state * 0.9 + x[:, :CONV_DIM, :, :].squeeze(2) * 0.1
        self.state[:] = new.to(torch.float16)
        return new[:, :32, :].reshape(1, 32, 1, 4).to(torch.float16)

test_model("U4_8192x4_no_proj", TestU4().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("state", (1, CONV_DIM, 4), np.float16)])

# ============================================================
print("\n=== U5: (1, 8192, 4) state * 0.9 only (minimal) ===")
class TestU5(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        new = self.state * 0.9
        self.state[:] = new.to(torch.float16)
        return new[:, :32, :].reshape(1, 32, 1, 4).to(torch.float16)

test_model("U5_8192x4_scale_only", TestU5().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("state", (1, CONV_DIM, 4), np.float16)])

# ============================================================
print("\n=== U6: T10 clone + conv proj ===")
class TestU6(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        shift_mat = torch.zeros(4, 4, dtype=torch.float16)
        for i in range(3):
            shift_mat[i + 1, i] = 1.0
        self.register_buffer("shift_mat", shift_mat)
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)  # [1, 8192]
        shifted = torch.matmul(self.state, self.shift_mat)  # [1, 8192, 4]
        self.state[:] = shifted.to(torch.float16)
        return shifted[:, :32, :].reshape(1, 32, 1, 4).to(torch.float16)

test_model("U6_matmul_shift_conv", TestU6().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("state", (1, CONV_DIM, 4), np.float16)])

# ============================================================
print("\n=== U7: conv_state roll + conv proj ===")
class TestU7(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        rolled = self.state.roll(-1, dims=2)
        self.state[:] = rolled
        return self.state[:, :32, :].reshape(1, 32, 1, 4).to(torch.float16)

test_model("U7_roll_conv", TestU7().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("state", (1, CONV_DIM, 4), np.float16)])

# ============================================================  
print("\n=== U8: (1, 8192, 4) + conv proj + reduce output ===")
class TestU8(nn.Module):
    """Same as U1 but return reduce_sum of state (mirroring real usage)"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        new = self.state + y.unsqueeze(-1) * 0.1
        self.state[:] = new.to(torch.float16)
        return new.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("U8_8192x4_conv_reduce", TestU8().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("state", (1, CONV_DIM, 4), np.float16)])

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, passed in results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
