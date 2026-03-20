#!/usr/bin/env python3
"""Binary-search: test which specific op pattern from the Qwen3.5 linear attention 
graph causes ANE failure. We build minimal stateful models that replicate increasingly
complex subsets of the op graph.

Test progression:
  1. reduce_sum alone
  2. reduce_sum + rsqrt + clip (L2 norm pattern)
  3. softplus + exp (gating)
  4. tile (GQA expansion)
  5. split
  6. All of the above combined (mini recurrence)
  7. Full recurrence with states
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
import time

OUT_DIR = "/tmp/qwen35_ane_test/op_tests"
os.makedirs(OUT_DIR, exist_ok=True)

# Qwen3.5 dimensions
HIDDEN = 2560
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_K_HEADS = 16
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM   # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM  # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM    # 8192

results = []

def test_model(name, model, input_shapes, state_shapes=None, timeout=30):
    """Export and test a model on ANE. Returns True if ANE succeeds."""
    path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Trace
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
    
    # Dump op counts
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
    
    # Test ANE
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
        err_str = str(e)[:200]
        print(f"  ❌ {name}: FAILED on ANE - {err_str}")
        results.append((name, False))
        return False


# ============================================================
# Test 1: reduce_sum alone
# ============================================================
print("\n=== Test 1: reduce_sum ===")
class ReduceSumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, 256, 1, bias=False, dtype=torch.float16)
    def forward(self, x):
        # x: [1, HIDDEN, 1, 1]
        y = self.proj(x)  # [1, 256, 1, 1]
        sq = y * y
        s = sq.sum(dim=1, keepdim=True)  # reduce_sum along channels
        return s.to(torch.float16)

m1 = ReduceSumModel().eval()
test_model("01_reduce_sum", m1,
    [("x", (1, HIDDEN, 1, 1), np.float16)])


# ============================================================
# Test 2: L2 norm pattern (reduce_sum + clamp + rsqrt)
# ============================================================
print("\n=== Test 2: L2 norm (reduce_sum + clamp + rsqrt) ===")
class L2NormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, 256, 1, bias=False, dtype=torch.float16)
    def forward(self, x):
        y = self.proj(x)  # [1, 256, 1, 1]
        sq_sum = (y * y).sum(dim=1, keepdim=True)
        sq_sum = torch.clamp(sq_sum, min=1e-6)
        normed = y * torch.rsqrt(sq_sum)
        return normed.to(torch.float16)

m2 = L2NormModel().eval()
test_model("02_l2norm", m2,
    [("x", (1, HIDDEN, 1, 1), np.float16)])


# ============================================================
# Test 3: softplus + exp (gating ops)
# ============================================================
print("\n=== Test 3: softplus + exp (gating) ===")
class GatingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, 32, 1, bias=False, dtype=torch.float16)
        self.bias = nn.Parameter(torch.zeros(32, dtype=torch.float16))
        self.a_log = nn.Parameter(torch.zeros(32, dtype=torch.float16))
    def forward(self, x):
        a = self.proj(x).squeeze(3).squeeze(2)  # [1, 32]
        g = -self.a_log.exp() * F.softplus(a + self.bias)
        g_exp = g.exp()
        return g_exp.unsqueeze(2).unsqueeze(3).to(torch.float16)

m3 = GatingModel().eval()
test_model("03_gating", m3,
    [("x", (1, HIDDEN, 1, 1), np.float16)])


# ============================================================
# Test 4: tile (GQA expansion via repeat_interleave)
# ============================================================
print("\n=== Test 4: tile (repeat_interleave GQA) ===")
class TileModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, KEY_DIM, 1, bias=False, dtype=torch.float16)
    def forward(self, x):
        y = self.proj(x)  # [1, 2048, 1, 1]
        y = y.squeeze(3).squeeze(2)  # [1, 2048]
        y = y.reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)  # [1, 1, 16, 128]
        y = y.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)  # [1, 1, 32, 128]
        return y.reshape(1, NUM_V_HEADS * HEAD_K_DIM, 1, 1).to(torch.float16)

m4 = TileModel().eval()
test_model("04_tile_repeat", m4,
    [("x", (1, HIDDEN, 1, 1), np.float16)])


# ============================================================
# Test 5: split
# ============================================================
print("\n=== Test 5: split ===")
class SplitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
    def forward(self, x):
        y = self.proj(x)  # [1, 8192, 1, 1]
        q, k, v = torch.split(y, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=1)
        return (q + k + v[:, :KEY_DIM]).to(torch.float16)

m5 = SplitModel().eval()
test_model("05_split", m5,
    [("x", (1, HIDDEN, 1, 1), np.float16)])


# ============================================================
# Test 6: L2 norm + gating + tile combined (no states)
# ============================================================
print("\n=== Test 6: L2norm + gating + tile combined ===")
class CombinedOpsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_qkv = nn.Conv2d(HIDDEN, CONV_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_a = nn.Conv2d(HIDDEN, NUM_V_HEADS, 1, bias=False, dtype=torch.float16)
        self.a_log = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(NUM_V_HEADS, dtype=torch.float16))
    
    def forward(self, x):
        # x: [1, HIDDEN, 1, 1]
        qkv = self.proj_qkv(x)  # [1, 8192, 1, 1]
        q_cf, k_cf, v_cf = torch.split(qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=1)
        
        # Reshape to attention dims
        q = q_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM)
        
        # GQA expansion
        q = q.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        k = k.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        
        # L2 norm on Q and K
        q = q.transpose(1, 2)  # [1, 32, 1, 128]
        k = k.transpose(1, 2)
        
        q_sq = (q * q).sum(dim=-1, keepdim=True)
        q_sq = torch.clamp(q_sq, min=1e-6)
        q = q * torch.rsqrt(q_sq)
        
        k_sq = (k * k).sum(dim=-1, keepdim=True)
        k_sq = torch.clamp(k_sq, min=1e-6)
        k = k * torch.rsqrt(k_sq)
        
        # Gating
        a = self.proj_a(x).squeeze(3).squeeze(2)  # [1, 32]
        g = -self.a_log.exp() * F.softplus(a + self.dt_bias)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)  # [1, 32, 1, 1]
        
        # Simple output = q * k * g_t summed
        out = (q * k * g_t).sum(dim=-1, keepdim=True)  # reduce_sum
        return out.reshape(1, NUM_V_HEADS, 1, 1).to(torch.float16)

m6 = CombinedOpsModel().eval()
test_model("06_combined_no_state", m6,
    [("x", (1, HIDDEN, 1, 1), np.float16)])


# ============================================================
# Test 7: Full recurrence with states (minimal)
# ============================================================
print("\n=== Test 7: Full recurrence with 2 states ===")
class RecurrenceModel(nn.Module):
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
        # x: [1, HIDDEN, 1, 1]
        qkv = self.proj_qkv(x)  # [1, 8192, 1, 1]
        q_cf, k_cf, v_cf = torch.split(qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=1)
        
        q = q_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v_cf.squeeze(3).squeeze(2).reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM)
        
        # GQA expansion
        q = q.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        k = k.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        
        q = q.transpose(1, 2).squeeze(2)  # [1, 32, 128]
        k = k.transpose(1, 2).squeeze(2)
        v = v.transpose(1, 2).squeeze(2)  # [1, 32, 128]
        
        # L2 norm
        q_sq = (q * q).sum(dim=-1, keepdim=True)
        q_sq = torch.clamp(q_sq, min=1e-6)
        q = q * torch.rsqrt(q_sq)
        k_sq = (k * k).sum(dim=-1, keepdim=True)
        k_sq = torch.clamp(k_sq, min=1e-6)
        k = k * torch.rsqrt(k_sq)
        
        # Gating
        a = self.proj_a(x).squeeze(3).squeeze(2)  # [1, 32]
        b = self.proj_b(x).squeeze(3).squeeze(2).sigmoid()
        g = -self.a_log.exp() * F.softplus(a + self.dt_bias)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)  # [1, 32, 1, 1]
        
        # Recurrence
        state = self.rec_state
        state = state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)  # reduce_sum
        delta = (v - kv_mem) * b.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        
        # Output
        out = (state * q.unsqueeze(-1)).sum(dim=-2)  # reduce_sum
        
        # Conv state update (simple shift)
        self.conv_state[:] = self.conv_state.roll(-1, dims=2)
        
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

m7 = RecurrenceModel().eval()
test_model("07_recurrence_states", m7,
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    state_shapes=[
        ("conv_state", (1, CONV_DIM, 4), np.float16),
        ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16),
    ])


# ============================================================
# Test 8: Just the recurrence with state (no L2norm, no gating)
# ============================================================
print("\n=== Test 8: Recurrence only (no L2norm, no gating) ===")
class RecurrenceOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, VALUE_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    
    def forward(self, x):
        q = self.proj_q(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        
        state = self.rec_state
        # kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)  # [1, 32, 128]
        delta = v - kv_mem  # [1, 32, 128]
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)  # outer product update
        self.rec_state[:] = state.to(torch.float16)
        
        out = (state * q.unsqueeze(-1)).sum(dim=-2)  # [1, 32, 128]
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

m8 = RecurrenceOnlyModel().eval()
test_model("08_recurrence_only", m8,
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    state_shapes=[
        ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16),
    ])


# ============================================================
# Test 9: Just reduce_sum on a 4D tensor (matching the actual shapes)
# ============================================================
print("\n=== Test 9: reduce_sum on [1, 32, 128, 128] along dim=-2 ===")
class ReduceSum4DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        k = self.proj(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        # Multiply state by k and reduce_sum along dim=-2 (head_k_dim)
        s = self.state * k.unsqueeze(-1)  # [1, 32, 128, 128]
        out = s.sum(dim=-2)  # [1, 32, 128] — reduce_sum
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

m9 = ReduceSum4DModel().eval()
test_model("09_reduce_sum_4d", m9,
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    state_shapes=[
        ("state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16),
    ])


# ============================================================
# Test 10: outer product + state write (no reduce_sum)
# ============================================================
print("\n=== Test 10: outer product + state write (no reduce_sum) ===")
class OuterProductModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, VALUE_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    def forward(self, x):
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        # outer product: k.unsqueeze(-1) * v.unsqueeze(-2) -> [1, 32, 128, 128]
        update = k.unsqueeze(-1) * v.unsqueeze(-2)
        state = self.rec_state + update
        self.rec_state[:] = state.to(torch.float16)
        # Just return a slice to prove it ran
        return state[:, :, 0, :].reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

m10 = OuterProductModel().eval()
test_model("10_outer_product", m10,
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    state_shapes=[
        ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16),
    ])


# ============================================================
# Test 11: reduce_sum on rec_state shape with matmul replacement
# ============================================================
print("\n=== Test 11: matmul instead of reduce_sum ===")
class MatmulInsteadModel(nn.Module):
    """Replace (state * k.unsqueeze(-1)).sum(dim=-2) with einsum/matmul"""
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_k = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
        self.proj_v = nn.Conv2d(HIDDEN, VALUE_DIM, 1, bias=False, dtype=torch.float16)
        self.register_buffer("rec_state", torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    
    def forward(self, x):
        q = self.proj_q(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        k = self.proj_k(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        v = self.proj_v(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
        
        state = self.rec_state
        
        # Replace: (state * k.unsqueeze(-1)).sum(dim=-2) 
        # = state[b,h,k,v] * key[b,h,k] summed over k
        # = key @ state (after proper reshape)
        # state: [1, 32, 128, 128], k: [1, 32, 128]
        # k.unsqueeze(-2) @ state = [1, 32, 1, 128] -> squeeze
        kv_mem = torch.matmul(k.unsqueeze(-2), state).squeeze(-2)  # [1, 32, 128]
        
        delta = v - kv_mem
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.rec_state[:] = state.to(torch.float16)
        
        # Replace output reduce_sum with matmul too
        out = torch.matmul(q.unsqueeze(-2), state).squeeze(-2)  # [1, 32, 128]
        return out.reshape(1, VALUE_DIM, 1, 1).to(torch.float16)

m11 = MatmulInsteadModel().eval()
test_model("11_matmul_instead", m11,
    [("x", (1, HIDDEN, 1, 1), np.float16)],
    state_shapes=[
        ("rec_state", (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), np.float16),
    ])


# ============================================================
# Test 12: L2norm via layer_norm trick (ANE-proven pattern)
# ============================================================
print("\n=== Test 12: L2norm via layer_norm trick ===")
class L2NormLayerNormModel(nn.Module):
    """Replace reduce_sum+rsqrt L2norm with the doubled-layernorm trick"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(HIDDEN, NUM_V_HEADS * HEAD_K_DIM, 1, bias=False, dtype=torch.float16)
    
    def forward(self, x):
        y = self.proj(x).squeeze(3).squeeze(2).reshape(1, NUM_V_HEADS, HEAD_K_DIM)
        # L2 norm via doubled concat + layer_norm trick
        y_neg = y * (-1)
        y_cat = torch.cat([y, y_neg], dim=-1)
        y_normed = F.layer_norm(y_cat, [HEAD_K_DIM * 2])
        y_out = y_normed[..., :HEAD_K_DIM]
        return y_out.reshape(1, NUM_V_HEADS * HEAD_K_DIM, 1, 1).to(torch.float16)

m12 = L2NormLayerNormModel().eval()
test_model("12_l2norm_layernorm", m12,
    [("x", (1, HIDDEN, 1, 1), np.float16)])


# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, passed in results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
