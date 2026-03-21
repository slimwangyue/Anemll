#!/usr/bin/env python3
"""Test: does stateless fp32 I/O improve recurrence parity vs fp32 PyTorch?

The stateless approach passes recurrent_state as fp32 input/output tensors.
CoreML still computes in fp16, but state is STORED in fp32 between calls.
The question: does fp32 storage prevent the drift we see with fp16 state?
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, tempfile
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

TMPDIR = tempfile.mkdtemp(prefix="stateless_parity_")
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


# ── Stateful model (current design, fp16 state) ──
class StatefulRecurrence(nn.Module):
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


# ── Stateless model (fp32 I/O, fp16 compute) ──
class StatelessRecurrence(nn.Module):
    def forward(self, q, k, v, g, beta, state_in):
        q = _l2norm(q) * (1.0 / (HEAD_K_DIM ** 0.5))
        k = _l2norm(k)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = state_in * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out, state


# ── fp32 Reference ──
class Fp32Reference:
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
    print(f"Stateless fp32 Parity Test ({NUM_STEPS} steps)")
    print(f"State shape: {STATE_SHAPE}\n")
    
    # ── 1. Export stateful (current design) ──
    print("Exporting stateful model (fp16 state)...")
    sf = StatefulRecurrence()
    sf.eval()
    inputs_0 = make_inputs(0)
    with torch.no_grad():
        traced_sf = torch.jit.trace(sf, inputs_0, check_trace=False)
    ct_inputs = [ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16)
                 for i, t in enumerate(inputs_0)]
    sf_ml = ct.convert(traced_sf, inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=[ct.StateType(
            wrapped_type=ct.TensorType(shape=STATE_SHAPE, dtype=np.float16),
            name="state")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18)
    sf_path = os.path.join(TMPDIR, "stateful.mlpackage")
    sf_ml.save(sf_path)
    del sf, sf_ml, traced_sf; gc.collect()
    
    # ── 2. Export stateless fp16 ──
    print("Exporting stateless model (fp16 I/O)...")
    sl = StatelessRecurrence()
    sl.eval()
    state_in = torch.zeros(*STATE_SHAPE, dtype=torch.float16)
    with torch.no_grad():
        traced_sl = torch.jit.trace(sl, (*inputs_0, state_in), check_trace=False)
    sl16_ml = ct.convert(traced_sl,
        inputs=ct_inputs + [ct.TensorType(name="state_in", shape=STATE_SHAPE, dtype=np.float16)],
        outputs=[ct.TensorType(name="output", dtype=np.float16),
                 ct.TensorType(name="state_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18)
    sl16_path = os.path.join(TMPDIR, "stateless_fp16.mlpackage")
    sl16_ml.save(sl16_path)
    del sl16_ml; gc.collect()
    
    # ── 3. Export stateless fp32 ──
    print("Exporting stateless model (fp32 I/O)...")
    sl32_ml = ct.convert(traced_sl,
        inputs=ct_inputs + [ct.TensorType(name="state_in", shape=STATE_SHAPE, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float16),
                 ct.TensorType(name="state_out", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18)
    sl32_path = os.path.join(TMPDIR, "stateless_fp32.mlpackage")
    sl32_ml.save(sl32_path)
    del sl, traced_sl, sl32_ml; gc.collect()
    
    # ── 4. Run all three ──
    # Stateful
    cml_sf = ct.models.MLModel(sf_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    sf_state = cml_sf.make_state()
    sf_outs = []
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        np_in = {f"input_{i}": t.numpy().astype(np.float16) for i, t in enumerate(inputs)}
        sf_outs.append(cml_sf.predict(np_in, state=sf_state)["output"])
    del cml_sf, sf_state; gc.collect()
    
    # Stateless fp16
    cml_sl16 = ct.models.MLModel(sl16_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    sl16_outs = []
    sl16_state = np.zeros(STATE_SHAPE, dtype=np.float16)
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        np_in = {f"input_{i}": t.numpy().astype(np.float16) for i, t in enumerate(inputs)}
        np_in["state_in"] = sl16_state
        result = cml_sl16.predict(np_in)
        sl16_outs.append(result["output"])
        sl16_state = result["state_out"]
    del cml_sl16; gc.collect()
    
    # Stateless fp32
    cml_sl32 = ct.models.MLModel(sl32_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    sl32_outs = []
    sl32_state = np.zeros(STATE_SHAPE, dtype=np.float32)
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        np_in = {f"input_{i}": t.numpy().astype(np.float16) for i, t in enumerate(inputs)}
        np_in["state_in"] = sl32_state
        result = cml_sl32.predict(np_in)
        sl32_outs.append(result["output"])
        sl32_state = result["state_out"]
    del cml_sl32; gc.collect()
    
    # fp32 reference
    ref = Fp32Reference()
    ref_outs = []
    for step in range(NUM_STEPS):
        inputs = make_inputs(step)
        with torch.no_grad():
            ref_outs.append(ref.step(*inputs).numpy())
    
    # ── 5. Compare ──
    print(f"\n{'Step':>4} {'stateful':>14} {'stateless16':>14} {'stateless32':>14} {'sf==sl16':>10}")
    print("-" * 60)
    
    for step in range(NUM_STEPS):
        cos_sf = cosine(sf_outs[step], ref_outs[step])
        cos_sl16 = cosine(sl16_outs[step], ref_outs[step])
        cos_sl32 = cosine(sl32_outs[step], ref_outs[step])
        cos_sf_sl16 = cosine(sf_outs[step], sl16_outs[step])
        marker = " ✓" if cos_sl32 > cos_sf + 0.01 else ""
        print(f"{step:4d} {cos_sf:14.8f} {cos_sl16:14.8f} {cos_sl32:14.8f} {cos_sf_sl16:10.6f}{marker}")
    
    # Summary
    print(f"\n{'Approach':<20} {'Avg cos':>10} {'Min cos':>10}")
    print("-" * 45)
    for name, outs in [("stateful (fp16)", sf_outs), ("stateless fp16", sl16_outs), ("stateless fp32", sl32_outs)]:
        coss = [cosine(outs[s], ref_outs[s]) for s in range(NUM_STEPS)]
        print(f"{name:<20} {np.mean(coss):10.6f} {np.min(coss):10.6f}")
    
    # State comparison at step 29
    print(f"\nFinal state comparison (step {NUM_STEPS-1}):")
    print(f"  sl16 state max_abs: {np.abs(sl16_state).max():.6f}")
    print(f"  sl32 state max_abs: {np.abs(sl32_state).max():.6f}")
    print(f"  sl32-sl16 state max_diff: {np.abs(sl32_state.astype(np.float32) - sl16_state.astype(np.float32)).max():.8f}")
    
    print(f"\nCleanup: rm -rf {TMPDIR}")
