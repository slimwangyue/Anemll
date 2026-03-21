#!/usr/bin/env python3
"""Test Kahan compensated summation for fp16 recurrence stability.

Root cause recap:
  - Recurrence: state += k * delta (additive update)
  - In fp16, small updates get lost when |state| >> |update|
  - Error compounds: after 20 steps, PT-CML cos → 0.03

Fix: Kahan summation stores a compensation tensor that captures
the lost low-order bits from each addition. This doubles the
effective precision of the state accumulation.

Test: Compare 3 approaches at realistic dims (4 heads, 128 K, 128 V):
  - baseline: standard recurrence (current code)
  - kahan: Kahan compensated summation
  - kahan_full: Kahan + compensated decay
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, tempfile
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

TMPDIR = tempfile.mkdtemp(prefix="kahan_recurrence_")
NUM_V_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_STEPS = 30

def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def _l2norm_fp16(x):
    """L2 normalize like the model does."""
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-4))


class BaselineRecurrence(nn.Module):
    """Standard recurrence (current code) — NO compensation."""
    def __init__(self):
        super().__init__()
        self.register_buffer('state', torch.zeros(
            1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    
    def forward(self, q, k, v, g, beta):
        q = _l2norm_fp16(q)
        k = _l2norm_fp16(k)
        scale = 1.0 / (HEAD_K_DIM ** 0.5)
        q = q * scale
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = self.state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        self.state = state
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out


class KahanRecurrence(nn.Module):
    """Kahan compensated summation for the additive state update."""
    def __init__(self):
        super().__init__()
        self.register_buffer('state', torch.zeros(
            1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
        self.register_buffer('compensation', torch.zeros(
            1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
    
    def forward(self, q, k, v, g, beta):
        q = _l2norm_fp16(q)
        k = _l2norm_fp16(k)
        scale = 1.0 / (HEAD_K_DIM ** 0.5)
        q = q * scale
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        
        # Multiplicative decay (applied to both state and compensation)
        state = self.state * g_t
        comp = self.compensation * g_t
        
        # Read
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        update = k.unsqueeze(-1) * delta.unsqueeze(-2)
        
        # Kahan compensated addition: state += update
        y = update - comp           # correction from previous step
        t = state + y               # tentative new state
        comp = (t - state) - y      # new compensation (lost bits)
        state = t
        
        self.state = state
        self.compensation = comp
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out


class Fp32Reference(nn.Module):
    """Full fp32 reference — ground truth."""
    def __init__(self):
        super().__init__()
        self.state = torch.zeros(
            1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float32)
    
    def forward_step(self, q, k, v, g, beta):
        q, k, v, g, beta = [x.float() for x in (q, k, v, g, beta)]
        q = q / (q.norm(dim=-1, keepdim=True).clamp(min=1e-4))
        k = k / (k.norm(dim=-1, keepdim=True).clamp(min=1e-4))
        scale = 1.0 / (HEAD_K_DIM ** 0.5)
        q = q * scale
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        self.state = self.state * g_t
        kv_mem = (self.state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        self.state = self.state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (self.state * q.unsqueeze(-1)).sum(dim=-2)
        return out.to(torch.float16)


def make_inputs(step, realistic_gates=True):
    """Generate inputs. If realistic_gates=True, use negative g (decay < 1)."""
    torch.manual_seed(42 + step)
    q = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.1
    k = torch.randn(1, NUM_V_HEADS, HEAD_K_DIM, dtype=torch.float16) * 0.1
    v = torch.randn(1, NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float16) * 0.1
    if realistic_gates:
        # Model uses: g = -A_log.exp() * softplus(a + dt_bias)
        # This gives negative values (decay < 1), typically in [-3, -0.01]
        g = -torch.abs(torch.randn(1, NUM_V_HEADS, dtype=torch.float16)) * 0.5
    else:
        g = torch.randn(1, NUM_V_HEADS, dtype=torch.float16) * 0.5
    beta = torch.randn(1, NUM_V_HEADS, dtype=torch.float16).sigmoid()
    return (q, k, v, g, beta)


def export_and_run(model, name, realistic_gates=True):
    """Export CoreML, run NUM_STEPS steps, return per-step outputs."""
    model.eval()
    inputs_0 = make_inputs(0, realistic_gates)
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs_0, check_trace=False)
    
    ct_inputs = [ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16) 
                 for i, t in enumerate(inputs_0)]
    
    # Collect state types from buffers
    states = []
    for buf_name, buf in model.named_buffers():
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=buf.shape, dtype=np.float16),
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
        inputs = make_inputs(step, realistic_gates)
        np_inputs = {f"input_{i}": t.numpy().astype(np.float16) for i, t in enumerate(inputs)}
        out = cml.predict(np_inputs, state=cml_state)["output"]
        outputs.append(out)
    
    del cml, cml_state; gc.collect()
    return outputs


if __name__ == "__main__":
    print(f"Temp: {TMPDIR}")
    print(f"Kahan Compensated Recurrence Test ({NUM_STEPS} steps)")
    print(f"State shape: (1, {NUM_V_HEADS}, {HEAD_K_DIM}, {HEAD_V_DIM})")
    print()
    
    for gate_mode, realistic_gates in [("decay (realistic)", True), ("mixed (stress)", False)]:
        print(f"\n{'='*80}")
        print(f"  Gate mode: {gate_mode}")
        print(f"{'='*80}")
        
        # Export baseline and kahan to CoreML
        print("\nExporting baseline...")
        baseline = BaselineRecurrence()
        baseline_outs = export_and_run(baseline, f"baseline_{gate_mode[:5]}", realistic_gates)
        del baseline
        
        print("Exporting Kahan...")
        kahan = KahanRecurrence()
        kahan_outs = export_and_run(kahan, f"kahan_{gate_mode[:5]}", realistic_gates)
        del kahan
        
        # fp32 reference
        print("Running fp32 reference...")
        fp32_ref = Fp32Reference()
        fp32_outs = []
        for step in range(NUM_STEPS):
            inputs = make_inputs(step, realistic_gates)
            with torch.no_grad():
                out = fp32_ref.forward_step(*inputs).numpy()
            fp32_outs.append(out)
        del fp32_ref
        
        # Compare
        print(f"\n{'Step':>4} {'baseline-fp32':>14} {'kahan-fp32':>14} {'improvement':>12} "
              f"{'base max_err':>12} {'kahan max_err':>14}")
        print("-" * 80)
        
        improvements = []
        for step in range(NUM_STEPS):
            cos_base = cosine(baseline_outs[step], fp32_outs[step])
            cos_kahan = cosine(kahan_outs[step], fp32_outs[step])
            err_base = np.abs(baseline_outs[step].astype(np.float32) - fp32_outs[step].astype(np.float32)).max()
            err_kahan = np.abs(kahan_outs[step].astype(np.float32) - fp32_outs[step].astype(np.float32)).max()
            improvement = cos_kahan - cos_base
            improvements.append(improvement)
            
            marker = " ✓" if improvement > 0.01 else ""
            print(f"{step:4d} {cos_base:14.8f} {cos_kahan:14.8f} {improvement:+12.8f} "
                  f"{err_base:12.6f} {err_kahan:14.6f}{marker}")
        
        avg_imp = np.mean(improvements)
        print(f"\nAvg improvement: {avg_imp:+.6f}")
        if avg_imp > 0:
            print(f"  → Kahan summation HELPS (avg +{avg_imp:.6f})")
        else:
            print(f"  → Kahan summation does NOT help")
    
    print(f"\nCleanup: rm -rf {TMPDIR}")
