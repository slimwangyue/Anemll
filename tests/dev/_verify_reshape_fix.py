#!/usr/bin/env python3
"""Verify the reshape fix: store conv_state as (1, 1024, 32) in the state,
reshape to (1, 8192, 4) when needed inside forward().
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

OUT_DIR = "/tmp/qwen35_ane_test/reshape_verify"
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN = 2560
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_K_HEADS = 16
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM   # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM  # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM    # 8192
CONV_KERNEL = 4
# Reshape: keep dim[1] <= 1024
CONV_STATE_GROUP = CONV_DIM // 1024  # 8
CONV_STATE_DIM1 = 1024
CONV_STATE_DIM2 = CONV_KERNEL * CONV_STATE_GROUP  # 32

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
        print(f"  ✅ {name}: PASSED on ANE")
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
# Test W1: Full recurrence + reshaped conv_state, single rec_state
# ============================================================
print("\n=== W1: Full layer with reshaped conv_state ===")
class ReshapedConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_qkv = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_a = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.proj_b = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.a_log = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        # conv2d for causal conv
        self.conv2d = nn.Conv2d(CONV_DIM, CONV_DIM, kernel_size=(1, CONV_KERNEL),
                                padding=0, groups=CONV_DIM, bias=False, dtype=torch.float16)
        # Output projection
        self.out_proj = nn.Conv2d(VALUE_DIM, HIDDEN, 1, bias=False, dtype=torch.float16)
        # States: conv_state RESHAPED to (1, 1024, 32)
        self.register_buffer("conv_state", torch.zeros(1, CONV_STATE_DIM1, CONV_STATE_DIM2, dtype=torch.float16))
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    
    def _l2norm(self, x):
        sq_sum = (x * x).sum(dim=-1, keepdim=True)
        sq_sum = torch.clamp(sq_sum, min=1e-6)
        return x * torch.rsqrt(sq_sum)
    
    def forward(self, x):
        # x: [1, HIDDEN, 1, 1]
        # Projections
        qkv = self.proj_qkv(x)  # [1, 8192, 1, 1]
        
        # ---- Conv stage ----
        # Reshape conv_state back to (1, 8192, 4) for computation
        conv_state_real = self.conv_state.reshape(1, CONV_DIM, CONV_KERNEL)
        # cat state with input: [1, 8192, 1, 4+1]
        stacked = torch.cat([conv_state_real.unsqueeze(2), qkv], dim=-1)
        conv_out = self.conv2d(stacked)
        conv_out = F.silu(conv_out[:, :, :, -1:])
        # Save next conv state
        next_conv_state = stacked[:, :, :, -CONV_KERNEL:].squeeze(2)  # [1, 8192, 4]
        # Reshape and write back to ANE-friendly shape
        self.conv_state[:] = next_conv_state.reshape(1, CONV_STATE_DIM1, CONV_STATE_DIM2).to(torch.float16)
        
        # ---- Layout stage (split QKV, GQA expand, L2 norm, gating) ----
        q_cf, k_cf, v_cf = torch.split(conv_out, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=1)
        q = q_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v_cf.squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        q = q.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        k = k.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        q = q.transpose(1, 2).squeeze(2)  # [1, 32, 128]
        k = k.transpose(1, 2).squeeze(2)
        q = self._l2norm(q)
        k = self._l2norm(k)
        
        a = self.proj_a(x).squeeze(3).squeeze(2)
        b = self.proj_b(x).squeeze(3).squeeze(2).sigmoid()
        g = -self.a_log.exp() * F.softplus(a + self.dt_bias)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        
        # ---- Recurrence ----
        state = self.rec_state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * b.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)  # [1, 32, 128]
        
        # ---- Output ----
        out_cf = out.reshape(1, VALUE_DIM, 1, 1)
        result = self.out_proj(out_cf)
        return (result + x).to(torch.float16)

m_w1 = ReshapedConvModel().eval()
test_model("W1_reshaped_conv_state", m_w1,
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    [("conv_state", (1, CONV_STATE_DIM1, CONV_STATE_DIM2), np.float16),
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
print(f"\nConv state shape: (1, {CONV_STATE_DIM1}, {CONV_STATE_DIM2}) = {CONV_STATE_DIM1 * CONV_STATE_DIM2 * 2} bytes")
print(f"Original shape:   (1, {CONV_DIM}, {CONV_KERNEL}) = {CONV_DIM * CONV_KERNEL * 2} bytes")
