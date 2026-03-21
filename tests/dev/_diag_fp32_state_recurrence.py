#!/usr/bin/env python3
"""Test fp32 state storage with fp16 compute for recurrence stability.

Strategy:
  - State tensor stored in fp32 (via ct.StateType dtype=np.float32)
  - Compute runs in fp16 (compute_precision=FLOAT16)
  - State read: fp32 → fp16 cast before computation
  - State write: fp16 → fp32 cast after update

This keeps the accumulation precise (fp32 storage prevents drift)
while compute runs on ANE in fp16.

Also tests: split-state approach (state_hi + state_lo in two fp16 buffers)
to simulate fp32 precision within fp16 storage.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, tempfile
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

TMPDIR = tempfile.mkdtemp(prefix="fp32state_")
NUM_V_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_STEPS = 30


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def _l2norm(x):
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-4))

STATE_SHAPE = (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)


class BaselineFP16State(nn.Module):
    """Baseline: state in fp16 (current design)."""
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


class FP32StateRecurrence(nn.Module):
    """State stored in fp32, compute in model dtype (fp16)."""
    def __init__(self):
        super().__init__()
        # fp32 state for precise accumulation
        self.register_buffer('state', torch.zeros(*STATE_SHAPE, dtype=torch.float32))
    
    def forward(self, q, k, v, g, beta):
        q = _l2norm(q) * (1.0 / (HEAD_K_DIM ** 0.5))
        k = _l2norm(k)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        # Read state: fp32 → fp16 for compute
        state = self.state.to(torch.float16) * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        # Write state: fp16 → fp32 for storage
        self.state = state.to(torch.float32)
        return (state * q.unsqueeze(-1)).sum(dim=-2)


class SplitStateRecurrence(nn.Module):
    """Split state into two fp16 tensors: state_hi and state_lo.
    
    To prevent MIL from optimizing away the error tracking:
    - state_hi: main state (as normal)
    - state_lo: captures lost precision via explicit computation
    - Uses multiply-by-one barriers to prevent algebraic simplification
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('state_hi', torch.zeros(*STATE_SHAPE, dtype=torch.float16))
        self.register_buffer('state_lo', torch.zeros(*STATE_SHAPE, dtype=torch.float16))
    
    def forward(self, q, k, v, g, beta):
        q = _l2norm(q) * (1.0 / (HEAD_K_DIM ** 0.5))
        k = _l2norm(k)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        
        # Decay both parts
        state_hi = self.state_hi * g_t
        state_lo = self.state_lo * g_t
        
        # Read from full state (hi + lo)
        kv_mem_hi = (state_hi * k.unsqueeze(-1)).sum(dim=-2)
        kv_mem_lo = (state_lo * k.unsqueeze(-1)).sum(dim=-2)
        kv_mem = kv_mem_hi + kv_mem_lo
        
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        update = k.unsqueeze(-1) * delta.unsqueeze(-2)
        
        # Add update to hi, compute true error for lo
        # Use clamp as an optimization barrier (does nothing numerically)
        new_hi = state_hi + update
        # Error = (new_hi - state_hi) - update
        #       = what was actually added minus what should have been added
        # But MIL will simplify this. Use clamp to break the chain.
        new_hi_clamped = new_hi.clamp(-65504, 65504)  # barrier
        error = (new_hi_clamped - state_hi) - update
        state_lo = state_lo - error  # accumulate compensation
        
        self.state_hi = new_hi_clamped
        self.state_lo = state_lo
        
        # Output from full state
        out = ((new_hi_clamped + state_lo) * q.unsqueeze(-1)).sum(dim=-2)
        return out


class Fp32Reference:
    """Full fp32 reference."""
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
    g = -torch.abs(torch.randn(1, NUM_V_HEADS, dtype=torch.float16)) * 0.5  # decay
    beta = torch.randn(1, NUM_V_HEADS, dtype=torch.float16).sigmoid()
    return (q, k, v, g, beta)


def export_and_run(model, name):
    """Export to CoreML, run NUM_STEPS, return outputs."""
    model.eval()
    inputs_0 = make_inputs(0)
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs_0, check_trace=False)
    
    ct_inputs = [ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16) 
                 for i, t in enumerate(inputs_0)]
    
    states = []
    for buf_name, buf in model.named_buffers():
        dt = np.float32 if buf.dtype == torch.float32 else np.float16
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=buf.shape, dtype=dt),
            name=buf_name
        ))
    
    mlmodel = ct.convert(
        traced, inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=states, compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    path = os.path.join(TMPDIR, f"{name}.mlpackage")
    mlmodel.save(path)
    del mlmodel; gc.collect()
    
    cml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    cml_state = cml.make_state()
    
    outputs = []
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        np_inputs = {f"input_{i}": t.numpy().astype(np.float16) for i, t in enumerate(inputs)}
        out = cml.predict(np_inputs, state=cml_state)["output"]
        outputs.append(out)
    
    del cml, cml_state; gc.collect()
    return outputs


if __name__ == "__main__":
    print(f"Temp: {TMPDIR}")
    print(f"FP32 State Recurrence Test ({NUM_STEPS} steps, decay gates)")
    print(f"State shape: {STATE_SHAPE}\n")
    
    # Test all three approaches
    approaches = [
        ("baseline_fp16", BaselineFP16State()),
        ("fp32_state", FP32StateRecurrence()),
        ("split_state", SplitStateRecurrence()),
    ]
    
    results = {}
    for name, model in approaches:
        print(f"Exporting {name}...")
        try:
            results[name] = export_and_run(model, name)
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = None
        del model
    
    # fp32 reference
    print("Running fp32 reference...")
    ref = Fp32Reference()
    ref_outs = []
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        with torch.no_grad():
            ref_outs.append(ref.step(*inputs).numpy())
    
    # Compare
    print(f"\n{'Step':>4}", end="")
    for name in results:
        if results[name] is not None:
            print(f" {name:>14}", end="")
    print()
    print("-" * (4 + 15 * sum(1 for v in results.values() if v is not None)))
    
    for step in range(NUM_STEPS):
        print(f"{step:4d}", end="")
        for name in results:
            if results[name] is not None:
                cos = cosine(results[name][step], ref_outs[step])
                print(f" {cos:14.8f}", end="")
        print()
    
    # Summary
    print(f"\n{'Approach':<20} {'Avg cos':>10} {'Min cos':>10}")
    print("-" * 45)
    for name in results:
        if results[name] is not None:
            coss = [cosine(results[name][s], ref_outs[s]) for s in range(NUM_STEPS)]
            print(f"{name:<20} {np.mean(coss):10.6f} {np.min(coss):10.6f}")
    
    print(f"\nCleanup: rm -rf {TMPDIR}")
