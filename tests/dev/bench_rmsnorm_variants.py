#!/usr/bin/env python3
"""Test RMSNorm variants: which ones avoid reduce_mean in MIL?

The iPhone A16 ANECCompile fails on reduce_mean in large prefill graphs.
We need an ANE-friendly RMSNorm that:
  1. Avoids reduce_mean MIL ops
  2. Is mathematically identical to RMSNorm (no quality loss)
  3. Runs on ANE

Approaches tested:
  A. reduce_sum * inv_H  (baseline: MIL folds to reduce_mean)
  B. matmul with ones/H vector
  C. einsum mean
  D. doubled-concat layer_norm(2H)
  E. Conv2d(H,1,1) with weights=1/H
"""

import sys, os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
import coremltools as ct

H = 2560
BATCH = 512  # prefill batch

# ── Reference: current (broken on A16 prefill) ──
class RMSNorm_A_ReduceMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H))
        self.eps = 1e-6
    def forward(self, x):
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * (1.0 + self.weight)

# ── B: matmul with ones/H ──
class RMSNorm_B_Matmul(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H))
        self.eps = 1e-6
        self.register_buffer('ones_over_h', torch.ones(H, 1) / H)
    def forward(self, x):
        sq = x * x
        var = torch.matmul(sq, self.ones_over_h)  # (..., H) @ (H, 1) → (..., 1)
        return x * torch.rsqrt(var + self.eps) * (1.0 + self.weight)

# ── C: einsum ──
class RMSNorm_C_Einsum(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H))
        self.eps = 1e-6
        self._inv_H = 1.0 / H
    def forward(self, x):
        sq = x * x
        s = torch.einsum('...h,...h->...', sq, sq.new_ones(sq.shape[-1:])).unsqueeze(-1) * self._inv_H
        return x * torch.rsqrt(s + self.eps) * (1.0 + self.weight)

# ── D: doubled-concat F.layer_norm(2H) ──
class RMSNorm_D_DoubledConcat(nn.Module):
    """Concatenate [x, -x] → 2H dim, apply layer_norm. Mean=0, so LN=RMSNorm."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H))
        self.eps = 1e-6
    def forward(self, x):
        # [x, -x] has mean=0, so layer_norm([x,-x]) = RMSNorm
        x_cat = torch.cat([x, -x], dim=-1)  # (..., 2H)
        normed = F.layer_norm(x_cat, (2 * H,), weight=None, bias=None, eps=self.eps)
        normed = normed[..., :H]  # take first H
        return normed * (1.0 + self.weight)

# ── E: Conv2d(H, 1, 1) with weights=1/H ──
class RMSNorm_E_Conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H))
        self.eps = 1e-6
        # Conv2d expects (B, C, H, W). We'll have (B, H, 1, S) after transpose.
        self.mean_conv = nn.Conv2d(H, 1, kernel_size=1, bias=False)
        self.mean_conv.weight.data.fill_(1.0 / H)
        self.mean_conv.weight.requires_grad = False
    def forward(self, x):
        # x: (B, 1, S, H) → need (B, H, S, 1) for conv
        sq = x * x
        sq_t = sq.permute(0, 3, 2, 1)  # (B, H, S, 1)
        var = self.mean_conv(sq_t)  # (B, 1, S, 1)
        var = var.permute(0, 3, 2, 1)  # (B, 1, S, 1)
        return x * torch.rsqrt(var + self.eps) * (1.0 + self.weight)

# ── F: sum via conv1d ──
class RMSNorm_F_Conv1d(nn.Module):
    """Use a 1x1 Conv2d as dot-product to compute sum(x²)/H."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H))
        self.eps = 1e-6
        # Reshape to (B*S, H, 1, 1) → Conv2d(H, 1, 1) → (B*S, 1, 1, 1)
        self.var_proj = nn.Conv2d(H, 1, kernel_size=1, bias=False)
        self.var_proj.weight.data.fill_(1.0 / H)
        self.var_proj.weight.requires_grad = False
    def forward(self, x):
        # x: (B, 1, S, H)
        B, _, S, _ = x.shape
        sq = x * x  # (B, 1, S, H)
        # reshape to (B, H, 1, S) for Conv2d channel processing
        sq_ch = sq.permute(0, 3, 1, 2)  # (B, H, 1, S)
        var = self.var_proj(sq_ch)  # (B, 1, 1, S)
        var = var.permute(0, 2, 3, 1)  # (B, 1, S, 1)
        return x * torch.rsqrt(var + self.eps) * (1.0 + self.weight)


def test_variant(name, module, shape):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False

    x = torch.randn(*shape)
    with torch.no_grad():
        pt_out = module(x)

    # Reference output
    ref = RMSNorm_A_ReduceMean()
    ref.weight.data.copy_(module.weight.data)
    ref.eval()
    with torch.no_grad():
        ref_out = ref(x)

    cos = F.cosine_similarity(pt_out.flatten().unsqueeze(0),
                               ref_out.flatten().unsqueeze(0)).item()
    max_diff = (pt_out - ref_out).abs().max().item()

    # Trace and convert
    traced = torch.jit.trace(module, x)
    ct_input = ct.TensorType(name="x", shape=shape, dtype=np.float16)
    mlmodel = ct.convert(
        traced,
        inputs=[ct_input],
        outputs=[ct.TensorType(name="out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    # Extract MIL ops
    prog = mlmodel._mil_program
    op_counts = {}
    for fn in prog.functions.values():
        for op in fn.operations:
            op_counts[op.op_type] = op_counts.get(op.op_type, 0) + 1

    has_reduce_mean = op_counts.get("reduce_mean", 0) > 0
    has_layer_norm = op_counts.get("layer_norm", 0) > 0
    has_reduce_sum = op_counts.get("reduce_sum", 0) > 0
    has_matmul = op_counts.get("matmul", 0) > 0 or op_counts.get("linear", 0) > 0
    has_conv = op_counts.get("conv", 0) > 0

    status = "⚠️ has reduce_mean" if has_reduce_mean else "✓ NO reduce_mean"

    print(f"\n  {name}: {status}")
    print(f"    cos vs ref: {cos:.10f}, max_diff: {max_diff:.2e}")
    key_ops = {k: v for k, v in sorted(op_counts.items())
               if k in ('reduce_mean', 'reduce_sum', 'rsqrt', 'layer_norm', 'matmul', 'linear', 'conv', 'concat', 'mul', 'add', 'slice_by_index')}
    print(f"    key MIL ops: {key_ops}")
    print(f"    all MIL ops: {dict(sorted(op_counts.items()))}")

    # ANE load test
    tmp = f"/tmp/rmsnorm_{name}_{shape[2]}.mlpackage"
    mlmodel.save(tmp)
    try:
        loaded = ct.models.MLModel(tmp, compute_units=ct.ComputeUnit.CPU_AND_NE)
        result = loaded.predict({"x": x.numpy().astype(np.float16)})
        out = list(result.values())[0]
        ane_cos = float(np.dot(ref_out.numpy().flatten(), out.flatten()) /
                       (np.linalg.norm(ref_out.numpy().flatten()) * np.linalg.norm(out.flatten()) + 1e-12))
        print(f"    ANE load: OK, cos vs ref: {ane_cos:.6f}")

        # Timing
        inp = {"x": x.numpy().astype(np.float16)}
        for _ in range(3):
            loaded.predict(inp)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            loaded.predict(inp)
            times.append((time.perf_counter() - t0) * 1000)
        med = sorted(times)[len(times)//2]
        print(f"    timing: {med:.3f} ms median")
        del loaded
    except Exception as e:
        print(f"    ANE load: FAIL — {e}")

    return not has_reduce_mean, cos


if __name__ == "__main__":
    shapes = [(1, 1, 1, H), (1, 1, BATCH, H)]
    shape_names = ["decode(1)", f"prefill({BATCH})"]

    variants = [
        ("A_reduce_mean", RMSNorm_A_ReduceMean()),
        ("B_matmul", RMSNorm_B_Matmul()),
        ("D_doubled_concat_LN", RMSNorm_D_DoubledConcat()),
        ("E_conv2d", RMSNorm_E_Conv2d()),
        ("F_conv1d_ch", RMSNorm_F_Conv1d()),
    ]

    for shape, sname in zip(shapes, shape_names):
        print(f"\n{'='*60}")
        print(f"  Shape: {sname} → {shape}")
        print(f"{'='*60}")
        for vname, module in variants:
            # Re-create for each shape test (avoid traced cache issues)
            cls = type(module)
            test_variant(vname, cls(), shape)
