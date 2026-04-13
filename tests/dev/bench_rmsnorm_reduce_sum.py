#!/usr/bin/env python3
"""Benchmark: reduce_mean vs reduce_sum RMSNorm for ANE compatibility.

The current RMSNorm uses reduce_mean which causes ANECCompile FAILED(11)
on iPhone A16 for large prefill graphs. This test verifies that replacing
reduce_mean with reduce_sum/H is:
  1. Bit-identical in output
  2. Uses different MIL ops (reduce_sum instead of reduce_mean)
  3. Loads and runs on ANE (CPU_AND_NE)

Usage:
    python tests/dev/bench_rmsnorm_reduce_sum.py
"""

import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import coremltools as ct

HIDDEN = 2560
CTX = 2048
BATCH = 512
EPS = 1e-6

# ── RMSNorm variants ──

class RMSNorm_ReduceMean(nn.Module):
    """Current: uses .mean() → reduce_mean MIL op."""
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H))
        self.eps = eps
    def forward(self, x):
        variance = (x * x).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * (1.0 + self.weight)

class RMSNorm_ReduceSum(nn.Module):
    """Proposed: uses .sum()/H → reduce_sum MIL op (avoids reduce_mean)."""
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.H = H
        self.weight = nn.Parameter(torch.zeros(H))
        self.eps = eps
        self._inv_H = 1.0 / H
    def forward(self, x):
        variance = (x * x).sum(-1, keepdim=True) * self._inv_H
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * (1.0 + self.weight)

class RMSNormGated_ReduceSum(nn.Module):
    """Gated variant with reduce_sum."""
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.H = H
        self.weight = nn.Parameter(torch.ones(H))
        self.eps = eps
        self._inv_H = 1.0 / H
    def forward(self, x, gate):
        variance = (x * x).sum(-1, keepdim=True) * self._inv_H
        normed = x * torch.rsqrt(variance + self.eps)
        out = normed * self.weight
        return out * F.silu(gate)


# ── Wrapper for single-token export ──

class NormWrapper(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
    def forward(self, hidden_states):
        return self.norm(hidden_states)


def export_and_test(name, module, input_shape=(1, 1, 1, HIDDEN)):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False

    wrapper = NormWrapper(module)
    wrapper.eval()

    # Test PyTorch output
    x = torch.randn(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        pt_out = wrapper(x)

    # Trace
    traced = torch.jit.trace(wrapper, x)

    # Convert to CoreML
    ct_input = ct.TensorType(name="hidden_states", shape=input_shape, dtype=np.float16)
    mlmodel = ct.convert(
        traced,
        inputs=[ct_input],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    # Check MIL ops
    spec = mlmodel.get_spec()
    mil_str = str(spec)

    has_reduce_mean = "reduce_mean" in mil_str
    has_reduce_sum = "reduce_sum" in mil_str
    has_rsqrt = "rsqrt" in mil_str

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  reduce_mean: {'YES ⚠️' if has_reduce_mean else 'NO ✓'}")
    print(f"  reduce_sum:  {'YES ✓' if has_reduce_sum else 'NO'}")
    print(f"  rsqrt:       {'YES' if has_rsqrt else 'NO'}")

    # Count specific ops from MIL program
    prog = mlmodel._mil_program
    op_counts = {}
    for fn in prog.functions.values():
        for op in fn.operations:
            op_type = op.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

    print(f"  MIL ops: {dict(sorted(op_counts.items()))}")

    # Save and test ANE loading
    tmp_path = f"/tmp/rmsnorm_test_{name}.mlpackage"
    mlmodel.save(tmp_path)

    # Test CPU_AND_NE loading
    for cu_name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)]:
        try:
            loaded = ct.models.MLModel(tmp_path, compute_units=cu)
            result = loaded.predict({"hidden_states": x.numpy().astype(np.float16)})
            out = list(result.values())[0]
            cos = float(np.dot(pt_out.numpy().flatten(), out.flatten()) /
                       (np.linalg.norm(pt_out.numpy().flatten()) * np.linalg.norm(out.flatten()) + 1e-12))
            print(f"  {cu_name}: LOAD OK, cos={cos:.6f}")
            del loaded
        except Exception as e:
            print(f"  {cu_name}: FAIL — {e}")

    # Timing (CPU_AND_NE)
    loaded = ct.models.MLModel(tmp_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    inp = {"hidden_states": x.numpy().astype(np.float16)}
    # Warmup
    for _ in range(5):
        loaded.predict(inp)
    # Measure
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        loaded.predict(inp)
        times.append((time.perf_counter() - t0) * 1000)
    med = sorted(times)[len(times)//2]
    print(f"  Timing (CPU_AND_NE, 30 runs): {med:.3f} ms median")
    del loaded

    return has_reduce_mean, has_reduce_sum, pt_out


def test_numerical_parity():
    """Verify reduce_mean and reduce_sum are bit-identical."""
    print(f"\n{'='*60}")
    print(f"  Numerical Parity Test")
    print(f"{'='*60}")

    norm_mean = RMSNorm_ReduceMean(HIDDEN)
    norm_sum = RMSNorm_ReduceSum(HIDDEN)

    # Copy weights
    norm_sum.weight.data.copy_(norm_mean.weight.data)

    x = torch.randn(1, 1, 1, HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        out_mean = norm_mean(x)
        out_sum = norm_sum(x)

    diff = (out_mean - out_sum).abs()
    max_diff = diff.max().item()
    cos = F.cosine_similarity(out_mean.flatten().unsqueeze(0),
                               out_sum.flatten().unsqueeze(0)).item()

    print(f"  max_diff (float32): {max_diff:.2e}")
    print(f"  cosine:             {cos:.10f}")
    print(f"  bit-identical:      {max_diff == 0.0}")

    # Test fp16
    x16 = x.half()
    with torch.no_grad():
        out_mean16 = norm_mean(x16)
        out_sum16 = norm_sum(x16)
    diff16 = (out_mean16 - out_sum16).abs()
    max_diff16 = diff16.max().item()
    print(f"  max_diff (float16): {max_diff16:.2e}")

    return max_diff == 0.0


def test_prefill_shape():
    """Test with prefill-sized input (BATCH tokens) — this is where A16 fails."""
    print(f"\n{'='*60}")
    print(f"  Prefill Shape Test (batch={BATCH})")
    print(f"{'='*60}")

    for name, cls in [("reduce_mean", RMSNorm_ReduceMean), ("reduce_sum", RMSNorm_ReduceSum)]:
        module = cls(HIDDEN)
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

        wrapper = NormWrapper(module)
        wrapper.eval()

        x = torch.randn(1, 1, BATCH, HIDDEN, dtype=torch.float32)
        traced = torch.jit.trace(wrapper, x)

        ct_input = ct.TensorType(name="hidden_states", shape=(1, 1, BATCH, HIDDEN), dtype=np.float16)
        mlmodel = ct.convert(
            traced,
            inputs=[ct_input],
            outputs=[ct.TensorType(name="output", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
        )

        # Check MIL ops
        prog = mlmodel._mil_program
        op_counts = {}
        for fn in prog.functions.values():
            for op in fn.operations:
                op_counts[op.op_type] = op_counts.get(op.op_type, 0) + 1

        has_mean = "reduce_mean" in op_counts
        has_sum = "reduce_sum" in op_counts
        print(f"  {name} (batch={BATCH}): reduce_mean={has_mean}, reduce_sum={has_sum}")
        print(f"    MIL ops: {dict(sorted(op_counts.items()))}")

        # Save and test
        tmp_path = f"/tmp/rmsnorm_prefill_{name}.mlpackage"
        mlmodel.save(tmp_path)
        try:
            loaded = ct.models.MLModel(tmp_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            print(f"    CPU_AND_NE: LOAD OK ✓")
            del loaded
        except Exception as e:
            print(f"    CPU_AND_NE: FAIL ⚠️ — {e}")


if __name__ == "__main__":
    print("Qwen3.5-4B RMSNorm ANE Compatibility Test")
    print(f"H={HIDDEN}, BATCH={BATCH}, CTX={CTX}")

    # 1. Numerical parity
    test_numerical_parity()

    # 2. Single-token export & MIL op check
    export_and_test("reduce_mean", RMSNorm_ReduceMean(HIDDEN))
    export_and_test("reduce_sum", RMSNorm_ReduceSum(HIDDEN))

    # 3. Prefill-sized test
    test_prefill_shape()

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
