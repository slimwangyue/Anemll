#!/usr/bin/env python3
"""Verify: is the recurrence error from fp16 precision or from CoreML conversion?

KEY FINDING from previous test:
  CPU-ANE cos = 1.000000 (ANE == CPU CoreML, identically)
  PT-CPU cos = 0.01-0.37 (CoreML diverges from PyTorch after 5+ steps)

Hypothesis: PyTorch runs recurrence in fp32, CoreML runs in fp16.
Test: Compare PyTorch-fp16 vs CoreML-fp16 — they should match.
If they match → the fix is to keep recurrence in fp32 (mixed precision).
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, tempfile, shutil
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

TMPDIR = tempfile.mkdtemp(prefix="fp16_verify_")

NUM_V_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_STEPS = 20

def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


class StatefulRecurrence(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('state', torch.zeros(
            1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    
    def forward(self, q, k, v, g, beta):
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = self.state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        self.state = state
        return out


class StatefulRecurrenceFP16(nn.Module):
    """Same recurrence but EXPLICITLY in fp16 math — no fp32 casts."""
    def __init__(self):
        super().__init__()
        self.state = torch.zeros(
            1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16)
    
    def forward_step(self, q, k, v, g, beta):
        """Manual step — no tracing, pure fp16."""
        g_t = g.to(torch.float16).exp().unsqueeze(-1).unsqueeze(-1)
        self.state = (self.state.to(torch.float16) * g_t).to(torch.float16)
        kv_mem = (self.state * k.unsqueeze(-1)).to(torch.float16).sum(dim=-2).to(torch.float16)
        delta = ((v - kv_mem) * beta.unsqueeze(-1)).to(torch.float16)
        self.state = (self.state + k.unsqueeze(-1) * delta.unsqueeze(-2)).to(torch.float16)
        out = (self.state * q.unsqueeze(-1)).to(torch.float16).sum(dim=-2).to(torch.float16)
        return out


class StatefulRecurrenceFP32(nn.Module):
    """Same recurrence in fp32 math — the HF reference."""
    def __init__(self):
        super().__init__()
        self.state = torch.zeros(
            1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float32)
    
    def forward_step(self, q, k, v, g, beta):
        q, k, v, g, beta = [x.float() for x in (q, k, v, g, beta)]
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        self.state = self.state * g_t
        kv_mem = (self.state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        self.state = self.state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (self.state * q.unsqueeze(-1)).sum(dim=-2)
        return out.to(torch.float16)


def make_inputs(step):
    torch.manual_seed(42 + step)
    q = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.05
    k = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.05
    v = torch.randn(1, NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float16) * 0.1
    g = torch.randn(1, NUM_V_HEADS, dtype=torch.float16) * 0.5
    beta = torch.randn(1, NUM_V_HEADS, dtype=torch.float16).sigmoid()
    return (q, k, v, g, beta)


if __name__ == "__main__":
    print(f"Temp: {TMPDIR}")
    print(f"Comparing fp16 vs fp32 recurrence across {NUM_STEPS} steps\n")
    
    # Export CoreML model
    model = StatefulRecurrence()
    model.eval()
    inputs_0 = make_inputs(0)
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs_0, check_trace=False)
    
    ct_inputs = [ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16) for i, t in enumerate(inputs_0)]
    states = [ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float16), name="state")]
    
    mlmodel = ct.convert(
        traced, inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=states, compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    path = os.path.join(TMPDIR, "recurrence.mlpackage")
    mlmodel.save(path)
    del mlmodel; gc.collect()
    
    cml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    cml_state = cml.make_state()
    
    # Manual fp16 and fp32 references
    fp16_ref = StatefulRecurrenceFP16()
    fp32_ref = StatefulRecurrenceFP32()
    
    print(f"{'Step':>4} {'CML-fp16 cos':>13} {'CML-fp32 cos':>13} {'fp16-fp32 cos':>14} "
          f"{'fp32 max_abs':>12} {'fp16 max_abs':>12}")
    print("-" * 80)
    
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        np_inputs = {f"input_{i}": t.numpy().astype(np.float16) for i, t in enumerate(inputs)}
        
        # CoreML
        cml_out = cml.predict(np_inputs, state=cml_state)["output"]
        
        # PyTorch fp16
        with torch.no_grad():
            fp16_out = fp16_ref.forward_step(*inputs).numpy()
        
        # PyTorch fp32
        with torch.no_grad():
            fp32_out = fp32_ref.forward_step(*inputs).numpy()
        
        cos_cml_fp16 = cosine(cml_out, fp16_out)
        cos_cml_fp32 = cosine(cml_out, fp32_out)
        cos_fp16_fp32 = cosine(fp16_out, fp32_out)
        
        # Check state divergence
        cml_state_arr = cml_state  # not directly accessible
        fp32_vs_fp16 = np.abs(fp32_out.astype(np.float32) - fp16_out.astype(np.float32)).max()
        fp32_vs_cml = np.abs(fp32_out.astype(np.float32) - cml_out.astype(np.float32)).max()
        
        print(f"{step:4d} {cos_cml_fp16:13.8f} {cos_cml_fp32:13.8f} {cos_fp16_fp32:14.8f} "
              f"{fp32_vs_cml:12.6f} {fp32_vs_fp16:12.6f}")
    
    del cml, cml_state; gc.collect()
    
    print()
    print("=" * 60)
    print("If CML-fp16 cos ≈ 1.0: CoreML matches PyTorch fp16")
    print("If fp16-fp32 cos drops: fp16 precision is the root cause")
    print("Fix: keep recurrence state in fp32 (mixed precision)")
