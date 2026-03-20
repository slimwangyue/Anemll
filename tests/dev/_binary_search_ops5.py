#!/usr/bin/env python3
"""The conv_state shift fails even with narrow+cat+full_write.
But D8 (full read + math + full write) passed.

Hypothesis: it's the slice_by_index on the state that breaks ANE.
Let me test patterns that avoid slicing the state entirely.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
import os
import shutil

OUT_DIR = "/tmp/qwen35_ane_test/op_tests5"
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
            print(f"  ❌ {name}: FAILED - {msg}")
        results.append((name, False))
        return False


# ============================================================
# T1: conv_state full read + full write (no slicing)
# ============================================================
print("\n=== T1: conv_state full read + math + full write (no slice) ===")
class TestT1(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)  # [1, 8192]
        # Full read, multiply, full write — same as D8 pattern
        new_state = self.conv_state * 0.9 + y.unsqueeze(-1) * 0.1
        self.conv_state[:] = new_state.to(torch.float16)
        return new_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("T1_no_slice", TestT1().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# T2: conv_state with roll (known FAIL control)
# ============================================================
print("\n=== T2: conv_state with roll (FAIL control) ===")
class TestT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        rolled = self.conv_state.roll(-1, dims=2)
        self.conv_state[:] = rolled
        return self.conv_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("T2_roll", TestT2().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# T3: conv_state via matmul with shift matrix (no slicing)
# This implements shift-left as: state @ shift_matrix
# ============================================================
print("\n=== T3: conv_state shift via matmul (no slice_by_index) ===")
class TestT3(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        # Shift-left matrix: [[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        shift = torch.zeros(4, 4, dtype=torch.float16)
        shift[1, 0] = 1.0
        shift[2, 1] = 1.0
        shift[3, 2] = 1.0
        self.register_buffer("shift_matrix", shift)
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)  # [1, 8192]
        # Shift left via matmul: state @ shift_matrix
        shifted = torch.matmul(self.conv_state, self.shift_matrix)  # [1, 8192, 4]
        # Write new value at position 3
        shifted[:, :, 3] = y
        self.conv_state[:] = shifted.to(torch.float16)
        return shifted.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("T3_matmul_shift", TestT3().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# T4: conv_state shift via gather with static indices
# ============================================================
print("\n=== T4: conv_state shift via index_select ===")
class TestT4(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        self.register_buffer("shift_idx", torch.tensor([1, 2, 3, 0], dtype=torch.long))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        # Gather-based shift
        shifted = torch.index_select(self.conv_state, 2, self.shift_idx)  # [1, 8192, 4]
        shifted[:, :, 3] = y
        self.conv_state[:] = shifted.to(torch.float16)
        return shifted.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("T4_index_select", TestT4().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# T5: Use 4 separate 2D states instead of 1 3D state
# Shift = just swap references
# ============================================================
print("\n=== T5: 4 separate states (no shift needed) ===")
class TestT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("s0", torch.zeros(1, CONV_DIM, dtype=torch.float16))
        self.register_buffer("s1", torch.zeros(1, CONV_DIM, dtype=torch.float16))
        self.register_buffer("s2", torch.zeros(1, CONV_DIM, dtype=torch.float16))
        self.register_buffer("s3", torch.zeros(1, CONV_DIM, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)  # [1, 8192]
        # Rotate: s0 = old s1, s1 = old s2, s2 = old s3, s3 = new
        new_s0 = self.s1.clone()
        new_s1 = self.s2.clone()
        new_s2 = self.s3.clone()
        self.s0[:] = new_s0
        self.s1[:] = new_s1
        self.s2[:] = new_s2
        self.s3[:] = y
        return (self.s0 + self.s1 + self.s2 + self.s3).unsqueeze(-1).unsqueeze(-1).to(torch.float16)

test_model("T5_4_separate_states", TestT5().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("s0", (1, CONV_DIM), np.float16),
     ("s1", (1, CONV_DIM), np.float16),
     ("s2", (1, CONV_DIM), np.float16),
     ("s3", (1, CONV_DIM), np.float16)])


# ============================================================
# T6: conv_state (1, 8192, 4) with depthwise conv to shift
# Use a depthwise 1D conv with kernel [0, 0, 0, 1] to shift
# ============================================================
print("\n=== T6: conv_state shift via depthwise conv ===")
class TestT6(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        # 1D conv kernel for shift-left: output[i] = input[i+1]
        # With kernel_size=2, pad=0: conv(state, [0, 1]) gives state[:, :, 1:4] effectively
        # But we need to preserve size... let's use a matrix multiply approach
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)  # [1, 8192]
        # Simple approach: manually construct the shifted state
        # Use indexing without slicing
        c0 = self.conv_state[:, :, 1:2]  # col 1
        c1 = self.conv_state[:, :, 2:3]  # col 2
        c2 = self.conv_state[:, :, 3:4]  # col 3
        c3 = y.unsqueeze(-1)
        new_state = torch.cat([c0, c1, c2, c3], dim=2)
        self.conv_state[:] = new_state.to(torch.float16)
        return new_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("T6_multi_slice_cat", TestT6().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# T7: Minimal reproducer — just slice_by_index on a 3D state
# ============================================================
print("\n=== T7: just slice_by_index on state ===")
class TestT7(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        # Read a slice of the state and return it
        sliced = self.conv_state[:, :, 1:4]  # slice_by_index on state
        self.conv_state[:] = (self.conv_state + 0.001).to(torch.float16)
        return sliced.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("T7_slice_state", TestT7().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# T8: Minimal — just read full state + write (no slice)
# ============================================================
print("\n=== T8: read + write full, no slice (control PASS) ===")
class TestT8(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        val = self.conv_state.sum(dim=-1, keepdim=True)
        self.conv_state[:] = (self.conv_state + 0.001).to(torch.float16)
        return val.unsqueeze(2).to(torch.float16)

test_model("T8_full_read_write", TestT8().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# T9: Does it work with smaller 3D state? e.g. (1, 32, 4)
# ============================================================
print("\n=== T9: small state (1, 32, 4) with slice shift ===")
class TestT9(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("conv_state", torch.zeros(1, 32, 4, dtype=torch.float16))
    def forward(self, x):
        sliced = self.conv_state[:, :, 1:4]
        zeros = torch.zeros(1, 32, 1, dtype=torch.float16, device=x.device)
        new_state = torch.cat([sliced, zeros], dim=2)
        self.conv_state[:] = new_state
        return new_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("T9_small_slice_shift", TestT9().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, 32, 4), np.float16)])


# ============================================================
# T10: (1, 8192, 4) — Try the actual Qwen conv_stage approach but avoid
# slicing the state: pass conv_state columns as separate states
# ============================================================
print("\n=== T10: state (1, CONV_DIM, 4) slice via known working pattern ===")
class TestT10(nn.Module):
    """Avoid slicing state by using matmul to extract and shift"""
    def __init__(self):
        super().__init__()
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        # Selection matrix: to get last 3 cols and append a zero col
        # [[0, 0, 0, 0],
        #  [1, 0, 0, 0],
        #  [0, 1, 0, 0],
        #  [0, 0, 1, 0]]
        shift_mat = torch.zeros(4, 4, dtype=torch.float16)
        for i in range(3):
            shift_mat[i + 1, i] = 1.0
        self.register_buffer("shift_mat", shift_mat)
    def forward(self, x):
        # Shift via matmul: no slice_by_index on state
        shifted = torch.matmul(self.conv_state, self.shift_mat)  # [1, 8192, 4]
        self.conv_state[:] = shifted.to(torch.float16)
        return shifted.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("T10_matmul_shift_only", TestT10().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, passed in results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
