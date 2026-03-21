#!/usr/bin/env python3
"""Test stateful recurrence: CPU vs ANE across multiple sequential calls.

Finding from previous test: ALL individual ops have PERFECT CPU-ANE parity.
Hypothesis: the error comes from STATE ACCUMULATION across calls.

Test: run N sequential steps with stateful models, measure how error grows.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, tempfile, shutil
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

TMPDIR = tempfile.mkdtemp(prefix="stateful_recur_")

# Qwen3.5 dimensions
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
    """Single recurrence step with persistent state."""
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


class StatefulRecurrencePlusNorm(nn.Module):
    """Recurrence + RMSNormGated + out_proj with persistent state."""
    def __init__(self):
        super().__init__()
        self.register_buffer('state', torch.zeros(
            1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
        self.norm_weight = nn.Parameter(torch.ones(HEAD_V_DIM, dtype=torch.float16))
        self.out_proj = nn.Conv2d(NUM_V_HEADS * HEAD_V_DIM, 2560, 1, bias=False, dtype=torch.float16)
        self.eps = 1e-6
    
    def forward(self, q, k, v, g, beta, z):
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = self.state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        core = (state * q.unsqueeze(-1)).sum(dim=-2)
        self.state = state
        
        # RMSNormGated
        core_flat = core.reshape(-1, HEAD_V_DIM)
        z_flat = z.reshape(-1, HEAD_V_DIM)
        doubled = torch.cat([core_flat, -core_flat], dim=-1)
        normed = torch.nn.functional.layer_norm(
            doubled, (2 * HEAD_V_DIM,), None, None, self.eps
        )
        normed = normed[..., :HEAD_V_DIM]
        out = normed * self.norm_weight
        out = out * torch.nn.functional.silu(z_flat)
        
        # out_proj
        out_bsh = out.reshape(1, 1, NUM_V_HEADS * HEAD_V_DIM)
        out_cf = out_bsh.transpose(1, 2).unsqueeze(2)
        out_proj = self.out_proj(out_cf)
        result = out_proj.squeeze(2).transpose(1, 2)
        return result


def test_stateful(name, model, make_inputs_fn, n_steps=NUM_STEPS):
    print(f"\n{'='*60}")
    print(f"  {name} ({n_steps} sequential steps)")
    print(f"{'='*60}")
    
    model.eval()
    inputs_0 = make_inputs_fn(0)
    
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs_0, check_trace=False)
    
    ct_inputs = [ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16) for i, t in enumerate(inputs_0)]
    
    # Collect states from traced model
    states = []
    for name, buf in traced.named_buffers():
        if name == "state":
            states.append(ct.StateType(
                wrapped_type=ct.TensorType(shape=buf.shape, dtype=np.float16),
                name=name,
            ))
    
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=states if states else None,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    
    path = os.path.join(TMPDIR, f"{name}.mlpackage")
    if os.path.exists(path):
        shutil.rmtree(path)
    mlmodel.save(path)
    del mlmodel; gc.collect()
    
    # Load CPU and ANE versions
    cpu_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    ane_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    cpu_state = cpu_model.make_state()
    ane_state = ane_model.make_state()
    
    # Fresh PyTorch reference
    pt_model = type(model)()
    pt_model.load_state_dict(model.state_dict())
    pt_model.eval()
    
    print(f"{'Step':>4} {'CPU-ANE cos':>12} {'PT-CPU cos':>12} {'PT-ANE cos':>12} "
          f"{'CPU max_abs':>10} {'ANE max_abs':>10}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    for step in range(n_steps):
        inputs = make_inputs_fn(step)
        np_inputs = {f"input_{i}": t.numpy().astype(np.float16) for i, t in enumerate(inputs)}
        
        # PyTorch reference
        with torch.no_grad():
            pt_out = pt_model(*inputs).numpy().astype(np.float16)
        
        # CoreML CPU
        cpu_out = cpu_model.predict(np_inputs, state=cpu_state)["output"]
        
        # CoreML ANE
        ane_out = ane_model.predict(np_inputs, state=ane_state)["output"]
        
        cos_cpu_ane = cosine(cpu_out, ane_out)
        cos_pt_cpu = cosine(pt_out, cpu_out)
        cos_pt_ane = cosine(pt_out, ane_out)
        diff_cpu = np.abs(pt_out.astype(np.float32) - cpu_out.astype(np.float32)).max()
        diff_ane = np.abs(pt_out.astype(np.float32) - ane_out.astype(np.float32)).max()
        
        print(f"{step:4d} {cos_cpu_ane:12.8f} {cos_pt_cpu:12.8f} {cos_pt_ane:12.8f} "
              f"{diff_cpu:10.6f} {diff_ane:10.6f}")
    
    del cpu_model, ane_model, cpu_state, ane_state
    gc.collect()


def make_recurrence_inputs(step):
    torch.manual_seed(42 + step)
    q = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.05
    k = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.05
    v = torch.randn(1, NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float16) * 0.1
    g = torch.randn(1, NUM_V_HEADS, dtype=torch.float16) * 0.5
    beta = torch.randn(1, NUM_V_HEADS, dtype=torch.float16).sigmoid()
    return (q, k, v, g, beta)


def make_full_inputs(step):
    torch.manual_seed(42 + step)
    q = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.05
    k = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.05
    v = torch.randn(1, NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float16) * 0.1
    g = torch.randn(1, NUM_V_HEADS, dtype=torch.float16) * 0.5
    beta = torch.randn(1, NUM_V_HEADS, dtype=torch.float16).sigmoid()
    z = torch.randn(1, NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float16) * 0.5
    return (q, k, v, g, beta, z)


if __name__ == "__main__":
    print(f"Temp: {TMPDIR}")
    print(f"Testing stateful recurrence across {NUM_STEPS} sequential steps")
    
    test_stateful("recurrence_only", StatefulRecurrence(), make_recurrence_inputs)
    test_stateful("recurrence_plus_norm", StatefulRecurrencePlusNorm(), make_full_inputs)
