#!/usr/bin/env python3
"""Test D failed: 2 states (conv_state + rec_state) at production shapes.
But previous session's combo tests passed with smaller shapes.

Now test:
  D1: conv_state (1, 8192, 4) alone — is this shape even ANE-OK?
  D2: conv_state (1, 8192, 4) + tiny rec_state (1, 2, 2, 2) 
  D3: tiny conv_state (1, 32, 4) + rec_state (1, 32, 128, 128)
  D4: conv_state 4D (1, 8192, 1, 4) + rec_state — rank mismatch fix?
  D5: 2 states both 4D at production shapes
  D6: conv_state as slice_update only (no clone)
  D7: conv_state with narrow+cat instead of slice assign
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
import os
import shutil

OUT_DIR = "/tmp/qwen35_ane_test/op_tests3"
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN = 2560
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_DIM = 8192

results = []

def test_model(name, model, input_shapes, state_shapes=None):
    path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    if os.path.exists(path):
        shutil.rmtree(path)
    
    example_inputs = []
    ct_inputs = []
    for iname, shape, dtype in input_shapes:
        if dtype == np.float16:
            t = torch.randn(*shape, dtype=torch.float16) * 0.01
        else:
            t = torch.zeros(*shape, dtype=torch.int32)
        example_inputs.append(t)
        ct_inputs.append(ct.TensorType(name=iname, shape=shape, dtype=dtype))
    
    ct_states = None
    if state_shapes:
        ct_states = []
        for sname, shape, dtype in state_shapes:
            ct_states.append(ct.StateType(
                wrapped_type=ct.TensorType(shape=shape, dtype=dtype),
                name=sname
            ))
    
    traced = torch.jit.trace(model, tuple(example_inputs))
    
    convert_kwargs = dict(
        inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    if ct_states:
        convert_kwargs["states"] = ct_states
    
    mlmodel = ct.convert(traced, **convert_kwargs)
    
    spec = mlmodel.get_spec()
    from collections import Counter
    op_counts = Counter()
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                op_counts[op.type] += 1
    non_const = {k: v for k, v in op_counts.items() if k != "const"}
    print(f"  Ops: {dict(non_const)}")
    
    mlmodel.save(path)
    del mlmodel
    
    try:
        loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        inputs_dict = {}
        for iname, shape, dtype in input_shapes:
            if dtype == np.float16:
                inputs_dict[iname] = np.random.randn(*shape).astype(np.float16) * 0.01
            else:
                inputs_dict[iname] = np.zeros(shape, dtype=np.int32)
        state = loaded.make_state() if ct_states else None
        kwargs = {"state": state} if state else {}
        out = loaded.predict(inputs_dict, **kwargs)
        print(f"  ✅ {name}: PASSED on ANE")
        results.append((name, True))
        del loaded
        return True
    except Exception as e:
        err_msg = str(e)[:200]
        # Check if it's a loading issue vs prediction issue
        if "not loaded" in err_msg:
            print(f"  ❌ {name}: FAILED TO LOAD (model execution plan error)")
        else:
            print(f"  ❌ {name}: FAILED on ANE - {err_msg}")
        results.append((name, False))
        return False


# ============================================================
# Test D1: conv_state alone
# ============================================================
print("\n=== Test D1: conv_state (1, 8192, 4) alone ===")
class TestD1(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)  # [1, 8192]
        # Shift left and write new value
        self.conv_state[:, :, 0:3] = self.conv_state[:, :, 1:4].clone()
        self.conv_state[:, :, 3:4] = y.unsqueeze(-1)
        return self.conv_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("D1_conv_alone", TestD1().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================  
# Test D2: conv_state + tiny rec_state
# ============================================================
print("\n=== Test D2: conv_state (1, 8192, 4) + tiny rec_state (1, 2, 2, 2) ===")
class TestD2(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, 2, 2, 2, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        self.conv_state[:, :, 0:3] = self.conv_state[:, :, 1:4].clone()
        self.conv_state[:, :, 3:4] = y.unsqueeze(-1)
        self.rec_state[:] = self.rec_state + 0.01
        return self.conv_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("D2_conv_tiny_rec", TestD2().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16),
     ("rec_state", (1, 2, 2, 2), np.float16)])


# ============================================================
# Test D3: tiny conv_state + full rec_state
# ============================================================
print("\n=== Test D3: tiny conv_state (1, 32, 4) + full rec_state ===")
class TestD3(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, 32, 1, bias=False, dtype=torch.float16)
        self.proj2 = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, 32, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        self.conv_state[:, :, 0:3] = self.conv_state[:, :, 1:4].clone()
        self.conv_state[:, :, 3:4] = y.unsqueeze(-1)
        k = self.proj2(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        self.rec_state[:] = (self.rec_state + k.unsqueeze(-1) * 0.01).to(torch.float16)
        return self.rec_state[:, :, 0, :].reshape(1, NUM_V_HEADS * HEAD_V_DIM, 1, 1).to(torch.float16)

test_model("D3_tiny_conv_full_rec", TestD3().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, 32, 4), np.float16),
     ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test D4: conv_state as 4D (1, 8192, 1, 4) + rec_state
# ============================================================
print("\n=== Test D4: conv_state 4D (1, 8192, 1, 4) + rec_state ===")
class TestD4(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_V_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 1, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        q = self.proj_q(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        state = self.rec_state
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = v - kv_mem
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        # Conv state: static shift (4D)
        self.conv_state[:, :, :, 0:3] = self.conv_state[:, :, :, 1:4].clone()
        self.conv_state[:, :, :, 3:4] = 0.0
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, NUM_V_HEADS * HEAD_V_DIM, 1, 1).to(torch.float16)

test_model("D4_conv4d_rec4d", TestD4().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 1, 4), np.float16),
     ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test D5: conv_state narrow+cat instead of slice assign
# ============================================================
print("\n=== Test D5: conv_state narrow+cat update ===")
class TestD5(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_V_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        q = self.proj_q(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        state = self.rec_state
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = v - kv_mem
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        # Conv state: narrow + cat (shift left, append zero)
        shifted = torch.narrow(self.conv_state, 2, 1, 3)
        zeros = torch.zeros(1, CONV_DIM, 1, dtype=torch.float16, device=x.device)
        new_conv_state = torch.cat([shifted, zeros], dim=2)
        self.conv_state[:] = new_conv_state
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, NUM_V_HEADS * HEAD_V_DIM, 1, 1).to(torch.float16)

test_model("D5_narrow_cat", TestD5().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16),
     ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test D6: Does total state SIZE matter? Try smaller rec_state
# Both states 3D, small
# ============================================================
print("\n=== Test D6: 2 small states (1, 64, 4) and (1, 4, 16, 16) ===")
class TestD6(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, 64, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, 64, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, 4, 16, 16, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        self.conv_state[:, :, 0:3] = self.conv_state[:, :, 1:4].clone()
        self.conv_state[:, :, 3:4] = y.unsqueeze(-1)[:, :64]
        self.rec_state[:] = (self.rec_state + 0.001).to(torch.float16)
        return self.conv_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("D6_small_2states", TestD6().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, 64, 4), np.float16),
     ("rec_state", (1, 4, 16, 16), np.float16)])


# ============================================================
# Test D7: Total state bytes test  
# rec_state alone: 1*32*128*128*2 = 1MB — works (tests A-F)
# conv_state alone at full size: 1*8192*4*2 = 64KB
# combined = ~1.06MB
# Test with 2 states at combined size around 1MB but different split

print("\n=== Test D7: 2 states totalling ~512KB ===")
class TestD7(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, 256, 1, bias=False, dtype=torch.float16)
        self.register_buffer("s1", torch.zeros(1, 256, 512, dtype=torch.float16))  # 256KB
        self.register_buffer("s2", torch.zeros(1, 256, 512, dtype=torch.float16))  # 256KB
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)  # [1, 256]
        self.s1[:, :, 0:1] = y.unsqueeze(-1)
        self.s2[:, :, 0:1] = y.unsqueeze(-1)
        return (self.s1.sum(dim=-1, keepdim=True) + self.s2.sum(dim=-1, keepdim=True)).unsqueeze(2).to(torch.float16)

test_model("D7_512k_2states", TestD7().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("s1", (1, 256, 512), np.float16),
     ("s2", (1, 256, 512), np.float16)])


# ============================================================
# Test D8: The EXACT shapes from previous session's working combo tests  
# Check if (1, 32, 128, 128) + (1, 8192, 4) combo works with trivial ops
# ============================================================
print("\n=== Test D8: production shapes, trivial ops ===")
class TestD8(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        # x: [1, HIDDEN, 1, 1]
        # Just read, add small values, write back
        self.rec_state[:] = (self.rec_state + x[:, :1, :1, :1].unsqueeze(-1) * 0.001).to(torch.float16)
        self.conv_state[:] = (self.conv_state + 0.001).to(torch.float16)
        return self.rec_state[:, :, 0, :].reshape(1, NUM_V_HEADS * HEAD_V_DIM, 1, 1).to(torch.float16)

test_model("D8_prod_shapes_trivial", TestD8().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16),
     ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, passed in results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
print()
total_state_bytes = {
    "D1_conv_alone": 1*CONV_DIM*4*2,
    "D2_conv_tiny_rec": 1*CONV_DIM*4*2 + 1*2*2*2*2,
    "D3_tiny_conv_full_rec": 1*32*4*2 + 1*NUM_V_HEADS*HEAD_K_DIM*HEAD_V_DIM*2,
    "D4_conv4d_rec4d": 1*CONV_DIM*1*4*2 + 1*NUM_V_HEADS*HEAD_K_DIM*HEAD_V_DIM*2,
    "D5_narrow_cat": 1*CONV_DIM*4*2 + 1*NUM_V_HEADS*HEAD_K_DIM*HEAD_V_DIM*2,
    "D6_small_2states": 1*64*4*2 + 1*4*16*16*2,
    "D7_512k_2states": 2*1*256*512*2,
    "D8_prod_shapes_trivial": 1*CONV_DIM*4*2 + 1*NUM_V_HEADS*HEAD_K_DIM*HEAD_V_DIM*2,
}
print("State sizes:")
for name, size in total_state_bytes.items():
    print(f"  {name}: {size/1024:.0f} KB")
