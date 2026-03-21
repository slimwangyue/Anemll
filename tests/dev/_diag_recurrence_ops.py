#!/usr/bin/env python3
"""Diagnose linear attention recurrence: find which op causes the ANE error.

The recurrence has these steps per token:
  1. g_t = g.exp()                     — elementwise exp
  2. state = state * g_t               — elementwise multiply (decay)
  3. kv_mem = (state * k).sum(dim=-2)  — reduce_sum over k_dim=128
  4. delta = (v - kv_mem) * beta       — elementwise
  5. state += k * delta                — outer product + accumulate
  6. out = (state * q).sum(dim=-2)     — reduce_sum over k_dim=128

Hypothesis: steps 3 and 6 (reduce_sum over 128 dimensions in fp16) are the error source.

Strategy: Export minimal models isolating each step, compare CPU vs ANE.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, tempfile, shutil
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

TMPDIR = tempfile.mkdtemp(prefix="recur_diag_")

# Qwen3.5 dimensions
NUM_V_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128

def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))

def test_model(name, model, inputs, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    """Export, predict on CPU and ANE, compare."""
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs)
    
    ct_inputs = []
    for i, t in enumerate(inputs):
        ct_inputs.append(ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16))
    
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    
    # Get CPU baseline
    np_inputs = {f"input_{i}": t.numpy().astype(np.float16) for i, t in enumerate(inputs)}
    cpu_out = mlmodel.predict(np_inputs)["output"]
    
    # Save and reload for ANE
    path = os.path.join(TMPDIR, f"{name}.mlpackage")
    if os.path.exists(path):
        shutil.rmtree(path)
    mlmodel.save(path)
    del mlmodel; gc.collect()
    
    ane_model = ct.models.MLModel(path, compute_units=compute_unit)
    ane_out = ane_model.predict(np_inputs)["output"]
    del ane_model; gc.collect()
    
    cos = cosine(cpu_out, ane_out)
    diff = np.abs(cpu_out.astype(np.float32) - ane_out.astype(np.float32))
    
    return cos, diff.max(), diff.mean()


# Also run PyTorch fp16 reference
def pytorch_fp16_ref(model, inputs):
    """Run model in pure fp16 PyTorch and compare to fp32."""
    model.eval()
    with torch.no_grad():
        fp16_out = model(*inputs)
        fp32_inputs = tuple(t.float() for t in inputs)
        # Create fp32 copy
        import copy
        model32 = copy.deepcopy(model).float()
        fp32_out = model32(*fp32_inputs)
    
    a = fp16_out.numpy().flatten().astype(np.float64)
    b = fp32_out.numpy().flatten().astype(np.float64)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    diff = np.abs(a - b)
    return float(cos), float(diff.max()), float(diff.mean())


# ================================================================
# Test 1: exp() alone
# ================================================================
class ExpModel(nn.Module):
    def forward(self, g):
        return g.exp()

# ================================================================
# Test 2: state * g (elementwise multiply)
# ================================================================
class DecayModel(nn.Module):
    def forward(self, state, g):
        return state * g.exp().unsqueeze(-1).unsqueeze(-1)

# ================================================================
# Test 3: reduce_sum over k_dim=128 (THE SUSPECT)
# ================================================================
class ReduceSumModel(nn.Module):
    def forward(self, state, k):
        # state: [1, heads, k_dim, v_dim], k: [1, heads, k_dim]
        return (state * k.unsqueeze(-1)).sum(dim=-2)

# ================================================================
# Test 4: Full recurrence body (1 step)
# ================================================================
class SingleRecurStep(nn.Module):
    def forward(self, state, q, k, v, g, beta):
        # All fp16 inputs
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state_decayed = state * g_t
        kv_mem = (state_decayed * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        new_state = state_decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (new_state * q.unsqueeze(-1)).sum(dim=-2)
        return out

# ================================================================
# Test 5: RMSNormGated alone (norm + gate)
# ================================================================
class NormGatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HEAD_V_DIM, dtype=torch.float16))
        self.eps = 1e-6
    
    def forward(self, x, z):
        doubled = torch.cat([x, -x], dim=-1)
        normed = torch.nn.functional.layer_norm(
            doubled, (2 * HEAD_V_DIM,), None, None, self.eps
        )
        normed = normed[..., :HEAD_V_DIM]
        out = normed * self.weight
        out = out * torch.nn.functional.silu(z)
        return out

# ================================================================
# Test 6: Recurrence + Norm (the amplification chain)
# ================================================================
class RecurPlusNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HEAD_V_DIM, dtype=torch.float16))
        self.eps = 1e-6
    
    def forward(self, state, q, k, v, g, beta, z):
        # Recurrence step
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state_decayed = state * g_t
        kv_mem = (state_decayed * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        new_state = state_decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)
        core = (new_state * q.unsqueeze(-1)).sum(dim=-2)
        # RMSNormGated
        doubled = torch.cat([core, -core], dim=-1)
        normed = torch.nn.functional.layer_norm(
            doubled, (2 * HEAD_V_DIM,), None, None, self.eps
        )
        normed = normed[..., :HEAD_V_DIM]
        out = normed * self.weight
        out = out * torch.nn.functional.silu(z)
        return out

# ================================================================
# Test 7: reduce_sum with l2-normed inputs (as in actual recurrence)
# ================================================================
class ReduceSumL2Normed(nn.Module):
    def forward(self, state, q):
        # q is already l2-normed and scaled, state has typical recurrence magnitudes
        # This mimics: out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return (state * q.unsqueeze(-1)).sum(dim=-2)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Temp: {TMPDIR}")
    print(f"Dims: heads={NUM_V_HEADS}, k_dim={HEAD_K_DIM}, v_dim={HEAD_V_DIM}")
    print()

    # Generate realistic inputs
    state = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16) * 0.1
    q = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.05
    k = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.05
    v = torch.randn(1, NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float16) * 0.1
    g = torch.randn(1, NUM_V_HEADS, dtype=torch.float16) * 0.5  # gate values
    beta = torch.randn(1, NUM_V_HEADS, dtype=torch.float16).sigmoid()
    z = torch.randn(1, NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float16) * 0.5
    # Core output (simulated)
    core_sim = torch.randn(NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float16) * 0.05

    tests = [
        ("1_exp", ExpModel(), (g,)),
        ("2_decay", DecayModel(), (state, g)),
        ("3_reducesum", ReduceSumModel(), (state, k)),
        ("4_single_recur", SingleRecurStep(), (state, q, k, v, g, beta)),
        ("5_normgated", NormGatedModel(), (core_sim, z.reshape(NUM_V_HEADS, HEAD_V_DIM))),
        ("6_recur_plus_norm", RecurPlusNorm(), (state, q, k, v, g, beta, z)),
        ("7_reducesum_l2", ReduceSumL2Normed(), (state, q)),
    ]
    
    print(f"{'Test':<25} {'CPU-ANE cos':>12} {'max_abs':>10} {'mean_abs':>10}")
    print("-" * 60)
    
    for name, model, inputs in tests:
        try:
            cos, maxd, meand = test_model(name, model, inputs)
            print(f"{name:<25} {cos:>12.8f} {maxd:>10.6f} {meand:>10.6f}")
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")
            import traceback; traceback.print_exc()

    # Also test fp16 vs fp32 PyTorch to separate ANE error from fp16 error
    print()
    print(f"{'Test (fp16 vs fp32)':<25} {'cos':>12} {'max_abs':>10} {'mean_abs':>10}")
    print("-" * 60)
    for name, model, inputs in tests:
        try:
            cos, maxd, meand = pytorch_fp16_ref(model, inputs)
            print(f"{name:<25} {cos:>12.8f} {maxd:>10.6f} {meand:>10.6f}")
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")
