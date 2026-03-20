#!/usr/bin/env python3
"""Narrowing down: Test 7 failed but Test 8 passed.
Test 7 = recurrence + L2norm + gating + split + tile + conv_state + rec_state
Test 8 = recurrence only + rec_state

Now systematically add components from Test 8 toward Test 7:
  A: rec_state + L2 norm on q,k
  B: rec_state + gating (exp, softplus, sigmoid)
  C: rec_state + L2 norm + gating  
  D: rec_state + conv_state (2 states, no complex ops)
  E: rec_state + tile (GQA expansion)
  F: rec_state + all ops but no conv_state (single state)
  G: rec_state + conv_state + all ops but replace roll with static shift
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
import os
import shutil

OUT_DIR = "/tmp/qwen35_ane_test/op_tests2"
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN = 2560
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_K_HEADS = 16
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM

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
        print(f"  ❌ {name}: FAILED on ANE - {str(e)[:200]}")
        results.append((name, False))
        return False


# ============================================================
# Test A: rec_state + L2 norm on q,k
# ============================================================
print("\n=== Test A: rec_state + L2 norm ===")
class TestA(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, VALUE_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def _l2norm(self, x):
        sq_sum = (x * x).sum(dim=-1, keepdim=True)
        sq_sum = torch.clamp(sq_sum, min=1e-6)
        return x * torch.rsqrt(sq_sum)
    def forward(self, x):
        q = self.proj_q(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        q = self._l2norm(q)
        k = self._l2norm(k)
        state = self.rec_state
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = v - kv_mem
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("A_rec_l2norm", TestA().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test B: rec_state + gating
# ============================================================
print("\n=== Test B: rec_state + gating ===")
class TestB(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, VALUE_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_a = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.proj_b = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.a_log = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        q = self.proj_q(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        a = self.proj_a(x).squeeze(3).squeeze(2)
        b = self.proj_b(x).squeeze(3).squeeze(2).sigmoid()
        g = -self.a_log.exp() * F.softplus(a + self.dt_bias)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = self.rec_state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * b.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("B_rec_gating", TestB().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test C: rec_state + L2 norm + gating
# ============================================================
print("\n=== Test C: rec_state + L2norm + gating ===")
class TestC(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, VALUE_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_a = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.proj_b = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.a_log = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def _l2norm(self, x):
        sq_sum = (x * x).sum(dim=-1, keepdim=True)
        sq_sum = torch.clamp(sq_sum, min=1e-6)
        return x * torch.rsqrt(sq_sum)
    def forward(self, x):
        q = self.proj_q(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        q = self._l2norm(q)
        k = self._l2norm(k)
        a = self.proj_a(x).squeeze(3).squeeze(2)
        b = self.proj_b(x).squeeze(3).squeeze(2).sigmoid()
        g = -self.a_log.exp() * F.softplus(a + self.dt_bias)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = self.rec_state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * b.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("C_rec_l2norm_gating", TestC().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test D: rec_state + conv_state (2 states, minimal ops)
# ============================================================
print("\n=== Test D: 2 states minimal ===")
class TestD(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, VALUE_DIM, 1, bias=False, dtype=torch.float16)
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
        # Conv state: simple static shift (not roll)
        self.conv_state[:, :, 0:3] = self.conv_state[:, :, 1:4].clone()
        self.conv_state[:, :, 3:4] = 0.0
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("D_2states_minimal", TestD().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16),
     ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test E: rec_state + GQA tile expansion
# ============================================================
print("\n=== Test E: rec_state + GQA tile ===")
class TestE(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, KEY_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, KEY_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, VALUE_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        q = self.proj_q(x).squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        # GQA expansion
        q = q.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        k = k.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        q = q.transpose(1, 2).squeeze(2)  # [1, 32, 128]
        k = k.transpose(1, 2).squeeze(2)
        state = self.rec_state
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = v - kv_mem
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("E_rec_tile", TestE().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test F: rec_state ONLY + all ops (L2norm + gating + tile) — single state
# ============================================================
print("\n=== Test F: single rec_state + ALL ops ===")
class TestF(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_qkv = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_a = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.proj_b = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.a_log = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def _l2norm(self, x):
        sq_sum = (x * x).sum(dim=-1, keepdim=True)
        sq_sum = torch.clamp(sq_sum, min=1e-6)
        return x * torch.rsqrt(sq_sum)
    def forward(self, x):
        qkv = self.proj_qkv(x)
        q_cf, k_cf, v_cf = torch.split(qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=1)
        q = q_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v_cf.squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        q = q.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        k = k.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        q = q.transpose(1, 2).squeeze(2)
        k = k.transpose(1, 2).squeeze(2)
        q = self._l2norm(q)
        k = self._l2norm(k)
        a = self.proj_a(x).squeeze(3).squeeze(2)
        b = self.proj_b(x).squeeze(3).squeeze(2).sigmoid()
        g = -self.a_log.exp() * F.softplus(a + self.dt_bias)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = self.rec_state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * b.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("F_single_state_all_ops", TestF().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test G: rec_state + conv_state + all ops, static shift (no roll)
# ============================================================
print("\n=== Test G: 2 states + all ops, static shift ===")
class TestG(nn.Module):
    def __init__(self):
        super().__init__()
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
        qkv = self.proj_qkv(x)
        q_cf, k_cf, v_cf = torch.split(qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=1)
        q = q_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v_cf.squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        q = q.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        k = k.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        q = q.transpose(1, 2).squeeze(2)
        k = k.transpose(1, 2).squeeze(2)
        q = self._l2norm(q)
        k = self._l2norm(k)
        a = self.proj_a(x).squeeze(3).squeeze(2)
        b = self.proj_b(x).squeeze(3).squeeze(2).sigmoid()
        g = -self.a_log.exp() * F.softplus(a + self.dt_bias)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = self.rec_state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * b.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        # Conv state: static shift
        self.conv_state[:, :, 0:3] = self.conv_state[:, :, 1:4].clone()
        self.conv_state[:, :, 3:4] = 0.0
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("G_2states_all_ops_static", TestG().eval(),
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_DIM, 4), np.float16),
     ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16)])


# ============================================================
# Test H: Same as Test 7 (from original), but replace roll with static shift
# ============================================================
print("\n=== Test H: Test 7 clone with static conv shift ===")
class TestH(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_qkv = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_a = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.proj_b = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.a_log = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.register_buffer("conv_state", torch.zeros(1, CONV_DIM, 4, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        qkv = self.proj_qkv(x)
        q_cf, k_cf, v_cf = torch.split(qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=1)
        q = q_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v_cf.squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        q = q.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        k = k.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        q = q.transpose(1, 2).squeeze(2)
        k = k.transpose(1, 2).squeeze(2)
        # L2 norm
        q_sq = (q * q).sum(dim=-1, keepdim=True)
        q_sq = torch.clamp(q_sq, min=1e-6)
        q = q * torch.rsqrt(q_sq)
        k_sq = (k * k).sum(dim=-1, keepdim=True)
        k_sq = torch.clamp(k_sq, min=1e-6)
        k = k * torch.rsqrt(k_sq)
        # Gating
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
        # Conv state: ROLL (as in original Test 7)
        self.conv_state[:] = self.conv_state.roll(-1, dims=2)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

test_model("H_test7_clone_roll", TestH().eval(),
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
