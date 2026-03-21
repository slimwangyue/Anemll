#!/usr/bin/env python3
"""Definitive test: PyTorch fp16 STATE vs CoreML fp16 STATE.

Previous tests showed: fp32 vs fp16 COMPUTE didn't matter.
The error comes from STATE storage in fp16.

Test: PyTorch with math_dtype=float16 (fp16 STATE + fp32 ALU on CPU)
vs CoreML fp16 (fp16 STATE + fp16 compute).

If they match better than fp32 PyTorch, we confirm:
  - Using force_fp16_math=True in the export reference
    gives meaningful parity numbers.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, tempfile
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

TMPDIR = tempfile.mkdtemp(prefix="state_match_")
NUM_V_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_STEPS = 30
STATE_SHAPE = (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def _l2norm(x):
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-4))


class CMLModel(nn.Module):
    """Model for CoreML export."""
    def __init__(self):
        super().__init__()
        self.register_buffer('state', torch.zeros(*STATE_SHAPE, dtype=torch.float16))
    
    def forward(self, q, k, v, g, beta):
        q = _l2norm(q) * (1.0 / (HEAD_K_DIM ** 0.5))
        k = _l2norm(k)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = self.state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.state = state
        return (state * q.unsqueeze(-1)).sum(dim=-2)


class PyTorchFP16State:
    """PyTorch reference with fp16 STATE (simulates force_fp16_math=True).
    
    Computation uses fp32 ALUs on CPU, but state stored in fp16.
    This matches the real model when traced with force_fp16_math=True.
    """
    def __init__(self):
        self.state = torch.zeros(*STATE_SHAPE, dtype=torch.float16)
    
    def step(self, q, k, v, g, beta):
        # Cast to fp16 for computation (on CPU, uses fp32 ALUs internally)
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        g = g.to(torch.float16)
        beta = beta.to(torch.float16)
        
        q = _l2norm(q) * (1.0 / (HEAD_K_DIM ** 0.5))
        k = _l2norm(k)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        self.state = self.state * g_t  # state stays fp16
        kv_mem = (self.state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        self.state = self.state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        return (self.state * q.unsqueeze(-1)).sum(dim=-2)


class PyTorchFP32State:
    """PyTorch reference with fp32 STATE (current default).
    
    Everything runs in fp32. This is what force_fp16_math=False gives.
    """
    def __init__(self):
        self.state = torch.zeros(*STATE_SHAPE, dtype=torch.float32)
    
    def step(self, q, k, v, g, beta):
        q, k, v, g, beta = [x.float() for x in (q, k, v, g, beta)]
        q = q / (q.norm(dim=-1, keepdim=True).clamp(min=1e-4)) * (1.0 / (HEAD_K_DIM ** 0.5))
        k = k / (k.norm(dim=-1, keepdim=True).clamp(min=1e-4))
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        self.state = self.state * g_t
        kv_mem = (self.state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        self.state = self.state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        return (self.state * q.unsqueeze(-1)).sum(dim=-2).to(torch.float16)


def make_inputs(step):
    torch.manual_seed(42 + step)
    q = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.1
    k = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.1
    v = torch.randn(1, NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float16) * 0.1
    g = -torch.abs(torch.randn(1, NUM_V_HEADS, dtype=torch.float16)) * 0.5
    beta = torch.randn(1, NUM_V_HEADS, dtype=torch.float16).sigmoid()
    return (q, k, v, g, beta)


if __name__ == "__main__":
    print(f"Temp: {TMPDIR}")
    print(f"State Precision Parity Test ({NUM_STEPS} steps, decay gates)\n")
    
    # Export CoreML
    print("Exporting CoreML model...")
    model = CMLModel()
    model.eval()
    inputs_0 = make_inputs(0)
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs_0, check_trace=False)
    
    ct_inputs = [ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16)
                 for i, t in enumerate(inputs_0)]
    states = [ct.StateType(
        wrapped_type=ct.TensorType(shape=STATE_SHAPE, dtype=np.float16), name="state")]
    
    mlmodel = ct.convert(traced, inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=states, compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16)
    path = os.path.join(TMPDIR, "model.mlpackage")
    mlmodel.save(path)
    del mlmodel, model, traced; gc.collect()
    
    # Run CoreML
    cml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    cml_state = cml.make_state()
    cml_outs = []
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        np_inputs = {f"input_{i}": t.numpy().astype(np.float16) for i, t in enumerate(inputs)}
        cml_outs.append(cml.predict(np_inputs, state=cml_state)["output"])
    del cml, cml_state; gc.collect()
    
    # Run PyTorch references
    print("Running PyTorch fp16-state reference...")
    pt_fp16 = PyTorchFP16State()
    fp16_outs = []
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        with torch.no_grad():
            fp16_outs.append(pt_fp16.step(*inputs).numpy())
    
    print("Running PyTorch fp32-state reference...")
    pt_fp32 = PyTorchFP32State()
    fp32_outs = []
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        with torch.no_grad():
            fp32_outs.append(pt_fp32.step(*inputs).numpy())
    
    # Compare
    print(f"\n{'Step':>4} {'CML vs PT-fp16':>15} {'CML vs PT-fp32':>15} {'PT-fp16 vs fp32':>16}")
    print("-" * 55)
    
    cos_cml_fp16 = []
    cos_cml_fp32 = []
    cos_fp16_fp32 = []
    
    for step in range(NUM_STEPS):
        c1 = cosine(cml_outs[step], fp16_outs[step])
        c2 = cosine(cml_outs[step], fp32_outs[step])
        c3 = cosine(fp16_outs[step], fp32_outs[step])
        cos_cml_fp16.append(c1)
        cos_cml_fp32.append(c2)
        cos_fp16_fp32.append(c3)
        marker = " ←" if c1 > c2 + 0.01 else ""
        print(f"{step:4d} {c1:15.8f} {c2:15.8f} {c3:16.8f}{marker}")
    
    print(f"\n{'Metric':<25} {'CML vs PT-fp16':>15} {'CML vs PT-fp32':>15} {'PT-fp16 vs fp32':>16}")
    print("-" * 75)
    print(f"{'Average cosine':<25} {np.mean(cos_cml_fp16):15.6f} {np.mean(cos_cml_fp32):15.6f} {np.mean(cos_fp16_fp32):16.6f}")
    print(f"{'Min cosine':<25} {np.min(cos_cml_fp16):15.6f} {np.min(cos_cml_fp32):15.6f} {np.min(cos_fp16_fp32):16.6f}")
    print(f"{'Last-5 avg':<25} {np.mean(cos_cml_fp16[-5:]):15.6f} {np.mean(cos_cml_fp32[-5:]):15.6f} {np.mean(cos_fp16_fp32[-5:]):16.6f}")
    
    better_count = sum(1 for c1, c2 in zip(cos_cml_fp16, cos_cml_fp32) if c1 > c2)
    print(f"\nSteps where CML-vs-fp16 > CML-vs-fp32: {better_count}/{NUM_STEPS}")
    
    if np.mean(cos_cml_fp16) > np.mean(cos_cml_fp32):
        improvement = np.mean(cos_cml_fp16) - np.mean(cos_cml_fp32)
        print(f"\n✅ CONFIRMED: fp16 state reference gives BETTER parity (+{improvement:.6f} avg)")
        print("   → Recommendation: use force_fp16_math=True in export path")
    else:
        print(f"\n❌ fp16 state reference does NOT improve parity")
        print("   → State truncation alone doesn't explain the divergence")
    
    print(f"\nCleanup: rm -rf {TMPDIR}")
