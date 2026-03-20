#!/usr/bin/env python3
"""Root cause confirmed: the conv_state shift-left pattern fails.
D1: shift via slice assign → FAIL
D8: full write → PASS

Now test specific conv_state update patterns to find what works:
  S1: self.conv_state[:] = shifted_result (full overwrite, no overlap)
  S2: torch.narrow + cat (compute new, then full write)
  S3: concat shifted + new (compute new, then full write)
  S4: The actual conv pattern from Qwen35LinearConvStage
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
import os
import shutil

OUT_DIR = "/tmp/qwen35_ane_test/op_tests4"
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
# S1: Full overwrite with computed shift
# ============================================================
print("\n=== S1: narrow+cat → full overwrite ===")
class TestS1(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)  # [1, 8192]
        # Compute new state: shift left + append new value
        shifted = torch.narrow(self.conv_state, 2, 1, 3)  # [1, 8192, 3]
        new_val = y.unsqueeze(-1)  # [1, 8192, 1]
        new_state = torch.cat([shifted, new_val], dim=2)  # [1, 8192, 4]
        self.conv_state[:] = new_state
        return new_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("S1_narrow_cat_full", TestS1().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# S2: Slice assign shift (the FAILING pattern from D1)
# ============================================================
print("\n=== S2: slice assign shift (CONTROL - expected FAIL) ===")
class TestS2(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        self.conv_state[:, :, 0:3] = self.conv_state[:, :, 1:4].clone()
        self.conv_state[:, :, 3:4] = y.unsqueeze(-1)
        return self.conv_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("S2_slice_assign", TestS2().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# S3: Index_put_ style (same as slice assign but explicit)
# ============================================================
print("\n=== S3: full overwrite via cat without narrow ===")
class TestS3(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2)
        # Use indexing to get columns 1,2,3 then cat with new
        cols123 = self.conv_state[:, :, 1:4]  # static slice
        new_state = torch.cat([cols123, y.unsqueeze(-1)], dim=2)
        self.conv_state[:] = new_state
        return new_state.sum(dim=-1, keepdim=True).unsqueeze(2).to(torch.float16)

test_model("S3_slice_cat_full", TestS3().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16)])


# ============================================================
# S4: Full overwrite with 2 states (the real use case)
# ============================================================
print("\n=== S4: 2 states, conv via narrow+cat ===")
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
class TestS4(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, VALUE_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_conv = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        q = self.proj_q(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        
        # Recurrence
        state = self.rec_state
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = v - kv_mem
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        
        # Conv state: narrow+cat → full overwrite 
        conv_in = self.proj_conv(x).squeeze(3).squeeze(2)  # [1, 8192]
        shifted = torch.narrow(self.conv_state, 2, 1, 3)
        new_conv = torch.cat([shifted, conv_in.unsqueeze(-1)], dim=2)
        self.conv_state[:] = new_conv
        
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("S4_2states_narrow_cat", TestS4().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16),
     ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# S5: 2 states + all recurrence ops, conv via narrow+cat
# ============================================================
print("\n=== S5: 2 states + all ops, conv via narrow+cat ===")
import torch.nn.functional as F
class TestS5(nn.Module):
    def __init__(self):
        super().__init__()
        KEY_DIM = 16 * 128
        self.proj_qkv = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_a = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.proj_b = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.a_log = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def _l2norm(self, x):
        sq_sum = (x * x).sum(dim=-1, keepdim=True)
        sq_sum = torch.clamp(sq_sum, min=1e-6)
        return x * torch.rsqrt(sq_sum)
    def forward(self, x):
        KEY_DIM = 16 * 128
        qkv = self.proj_qkv(x)
        q_cf, k_cf, v_cf = torch.split(qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=1)
        q = q_cf.squeeze(3).squeeze(2).reshape(1, 1, 16, HEAD_K_DIM)
        k = k_cf.squeeze(3).squeeze(2).reshape(1, 1, 16, HEAD_K_DIM)
        v = v_cf.squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        q = q.repeat_interleave(2, dim=2)
        k = k.repeat_interleave(2, dim=2)
        q = q.transpose(1, 2).squeeze(2)
        k = k.transpose(1, 2).squeeze(2)
        q = self._l2norm(q)
        k = self._l2norm(k)
        a = self.proj_a(x).squeeze(3).squeeze(2)
        b = self.proj_b(x).squeeze(3).squeeze(2).sigmoid()
        g = -self.a_log.exp() * F.softplus(a + self.dt_bias)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        # Recurrence
        state = self.rec_state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * b.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        # Conv state: narrow+cat → full overwrite
        shifted = torch.narrow(self.conv_state, 2, 1, 3)
        new_conv = torch.cat([shifted, torch.zeros(1, CONV_DIM, 1, dtype=torch.float16, device=x.device)], dim=2)
        self.conv_state[:] = new_conv
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("S5_full_combo_narrow_cat", TestS5().eval(),
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
