#!/usr/bin/env python3
"""Test compute_precision=FLOAT32 for recurrence.

Since CoreML state must be fp16, but compute can optionally be fp32.
With compute_precision=FLOAT32:
  - State is fp16 (CoreML limitation)
  - Intermediate ops run in fp32
  - This may still help: the per-step fp32 compute reduces rounding
    even though state storage truncates to fp16

Also test: what cosine do we see between CoreML fp16 vs fp32 compute?
If fp32 compute significantly helps, we can offer it as a quality option
(CPU/GPU mode for higher quality, ANE mode for speed).
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, tempfile
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

TMPDIR = tempfile.mkdtemp(prefix="fp32compute_")
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


class Recurrence(nn.Module):
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


def export_and_run(model, name, compute_precision, compute_units):
    model.eval()
    inputs_0 = make_inputs(0)
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs_0, check_trace=False)
    
    ct_inputs = [ct.TensorType(name=f"input_{i}", shape=t.shape, dtype=np.float16)
                 for i, t in enumerate(inputs_0)]
    states = [ct.StateType(
        wrapped_type=ct.TensorType(shape=STATE_SHAPE, dtype=np.float16),
        name="state"
    )]
    
    mlmodel = ct.convert(
        traced, inputs=ct_inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=states, compute_units=compute_units,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=compute_precision,
    )
    path = os.path.join(TMPDIR, f"{name}.mlpackage")
    mlmodel.save(path)
    del mlmodel; gc.collect()
    
    cml = ct.models.MLModel(path, compute_units=compute_units)
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
    print(f"FP32 Compute vs FP16 Compute Recurrence Test ({NUM_STEPS} steps)\n")
    
    configs = [
        ("fp16_cpu", ct.precision.FLOAT16, ct.ComputeUnit.CPU_ONLY),
        ("fp32_cpu", ct.precision.FLOAT32, ct.ComputeUnit.CPU_ONLY),
        ("fp16_ane", ct.precision.FLOAT16, ct.ComputeUnit.CPU_AND_NE),
    ]
    
    results = {}
    for name, prec, cu in configs:
        print(f"Exporting {name} (precision={prec}, compute={cu})...")
        model = Recurrence()
        try:
            results[name] = export_and_run(model, name, prec, cu)
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
    
    # Compare all against fp32 reference
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
    
    # Cross-compare: fp16 vs fp32 compute (CoreML internal)
    if results.get("fp16_cpu") is not None and results.get("fp32_cpu") is not None:
        print(f"\n--- CoreML fp16 vs fp32 compute (internal comparison) ---")
        for step in [0, 1, 5, 10, 20, 29]:
            if step < NUM_STEPS:
                cos = cosine(results["fp16_cpu"][step], results["fp32_cpu"][step])
                print(f"  Step {step:2d}: cos(fp16_cpu, fp32_cpu) = {cos:.10f}")
    
    # Summary
    print(f"\n{'Approach':<20} {'Avg cos':>10} {'Min cos':>10} {'Last-5 avg':>12}")
    print("-" * 55)
    for name in results:
        if results[name] is not None:
            coss = [cosine(results[name][s], ref_outs[s]) for s in range(NUM_STEPS)]
            last5 = [cosine(results[name][s], ref_outs[s]) for s in range(NUM_STEPS-5, NUM_STEPS)]
            print(f"{name:<20} {np.mean(coss):10.6f} {np.min(coss):10.6f} {np.mean(last5):12.6f}")
    
    print(f"\nCleanup: rm -rf {TMPDIR}")
