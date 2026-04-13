#!/usr/bin/env python3
"""Diagnose exactly what aspect of reduce_mean blocks ANE on iPhone A16.

Tests whether the blocker is:
  1. Negative axis index: mean(-1) vs mean(3)
  2. The reduce_mean op itself vs reduce_sum
  3. The combination of reduce_mean + rsqrt
  4. The hidden dimension size (H=2560)

Each variant is exported as a standalone model and inspected for MIL op
differences. The actual A16 failure can only be tested on-device, but
we can see if positive indexing changes the MIL lowering.

Usage:
    python tests/dev/diag_rmsnorm_blocker.py
"""

import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
import coremltools as ct

H = 2560
SHAPES = {
    "decode":  (1, 1, 1, H),
    "prefill": (1, 1, 512, H),
}

# ── Test models ──

class V0_MeanNeg1(nn.Module):
    """Current: .mean(-1, keepdim=True)"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(H))
    def forward(self, x):
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * (1.0 + self.w)

class V1_MeanPos3(nn.Module):
    """Hypothesis: .mean(3, keepdim=True) — positive axis"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(H))
    def forward(self, x):
        var = (x * x).mean(3, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * (1.0 + self.w)

class V2_SumDivH(nn.Module):
    """sum(-1) / H — explicit division"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(H))
    def forward(self, x):
        var = (x * x).sum(-1, keepdim=True) / H
        return x * torch.rsqrt(var + 1e-6) * (1.0 + self.w)

class V3_SumMulInvH(nn.Module):
    """sum(-1) * (1/H) — multiply by constant"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(H))
    def forward(self, x):
        var = (x * x).sum(-1, keepdim=True) * (1.0 / H)
        return x * torch.rsqrt(var + 1e-6) * (1.0 + self.w)

class V4_SumPos3(nn.Module):
    """.sum(3) — positive axis"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(H))
    def forward(self, x):
        var = (x * x).sum(3, keepdim=True) * (1.0 / H)
        return x * torch.rsqrt(var + 1e-6) * (1.0 + self.w)

class V5_MeanNeg1_NoRsqrt(nn.Module):
    """mean(-1) WITHOUT rsqrt — isolate the reduce_mean op"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return (x * x).mean(-1, keepdim=True)

class V6_ReduceMean_Only(nn.Module):
    """Just reduce_mean alone — simplest possible"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean(-1, keepdim=True)

class V7_ReduceMean_PosOnly(nn.Module):
    """Just reduce_mean with positive axis"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean(3, keepdim=True)

class V8_ReduceSum_DivH(nn.Module):
    """reduce_sum / H — explicit"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.sum(-1, keepdim=True) / H

class V9_LayerNorm_Natural(nn.Module):
    """F.layer_norm(H) — NO mean subtraction, with gamma=1+w"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(H))
    def forward(self, x):
        return F.layer_norm(x, (H,), 1.0 + self.w, bias=None, eps=1e-6)

class V10_DoubledConcat(nn.Module):
    """Doubled concat: [x, -x] → layer_norm(2H) → keep first H"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(H))
    def forward(self, x):
        x_cat = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(x_cat, (2 * H,), weight=None, bias=None, eps=1e-6)
        normed = normed[..., :H]
        return normed * (1.0 + self.w)


def analyze_mil(mlmodel, name):
    """Extract and display MIL op details."""
    prog = mlmodel._mil_program
    ops = {}
    reduce_ops_detail = []
    for fn_name, fn in prog.functions.items():
        for op in fn.operations:
            t = op.op_type
            ops[t] = ops.get(t, 0) + 1
            if t.startswith("reduce_"):
                # Get axis info
                axes = None
                if hasattr(op, 'axes') and op.axes is not None:
                    axes = op.axes.val if hasattr(op.axes, 'val') else op.axes
                elif hasattr(op, 'axis') and op.axis is not None:
                    axes = op.axis.val if hasattr(op.axis, 'val') else op.axis

                keep = None
                if hasattr(op, 'keep_dims'):
                    keep = op.keep_dims.val if hasattr(op.keep_dims, 'val') else op.keep_dims

                reduce_ops_detail.append({
                    'type': t,
                    'axes': axes,
                    'keep_dims': keep,
                    'name': op.name,
                })

    return ops, reduce_ops_detail


def test_variant(name, module, shape):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False

    x = torch.randn(*shape)
    with torch.no_grad():
        pt_out = module(x)

    traced = torch.jit.trace(module, x)

    ct_input = ct.TensorType(name="x", shape=shape, dtype=np.float16)
    mlmodel = ct.convert(
        traced,
        inputs=[ct_input],
        outputs=[ct.TensorType(name="out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    ops, reduce_detail = analyze_mil(mlmodel, name)

    has_reduce_mean = ops.get("reduce_mean", 0) > 0
    has_reduce_sum = ops.get("reduce_sum", 0) > 0
    has_layer_norm = ops.get("layer_norm", 0) > 0
    has_rsqrt = ops.get("rsqrt", 0) > 0

    status_parts = []
    if has_reduce_mean:
        status_parts.append("reduce_mean⚠️")
    if has_reduce_sum:
        status_parts.append("reduce_sum✓")
    if has_layer_norm:
        status_parts.append("layer_norm✓")
    if has_rsqrt:
        status_parts.append("rsqrt")
    status = " + ".join(status_parts) if status_parts else "none"

    # Only show key ops
    key_ops = {k: v for k, v in sorted(ops.items())
               if k in ('reduce_mean', 'reduce_sum', 'rsqrt', 'layer_norm', 'matmul',
                        'linear', 'conv', 'concat', 'mul', 'add', 'real_div',
                        'slice_by_index')}

    print(f"  {name:30s} │ {status:30s} │ {key_ops}")
    if reduce_detail:
        for rd in reduce_detail:
            print(f"  {'':30s} │   {rd['type']}(axes={rd['axes']}, keep_dims={rd['keep_dims']}, name={rd['name'][:40]})")

    return ops, reduce_detail, mlmodel


if __name__ == "__main__":
    variants = [
        ("V0_mean_neg1 (CURRENT)", V0_MeanNeg1()),
        ("V1_mean_pos3", V1_MeanPos3()),
        ("V2_sum_div_H", V2_SumDivH()),
        ("V3_sum_mul_invH", V3_SumMulInvH()),
        ("V4_sum_pos3_mul_invH", V4_SumPos3()),
        ("V5_mean_neg1_no_rsqrt", V5_MeanNeg1_NoRsqrt()),
        ("V6_reduce_mean_only", V6_ReduceMean_Only()),
        ("V7_reduce_mean_pos_only", V7_ReduceMean_PosOnly()),
        ("V8_reduce_sum_div_H", V8_ReduceSum_DivH()),
        ("V9_layer_norm_natural", V9_LayerNorm_Natural()),
        ("V10_doubled_concat", V10_DoubledConcat()),
    ]

    for shape_name, shape in SHAPES.items():
        print(f"\n{'='*100}")
        print(f"  Shape: {shape_name} → {shape}")
        print(f"{'='*100}")
        print(f"  {'Variant':30s} │ {'MIL reduce ops':30s} │ Key ops")
        print(f"  {'-'*30} │ {'-'*30} │ {'-'*30}")

        for vname, module in variants:
            cls = type(module)
            test_variant(vname, cls(), shape)

    # Summary
    print(f"\n{'='*100}")
    print("  ANALYSIS")
    print(f"{'='*100}")
    print("""
  Key questions:
  1. Does mean(-1) vs mean(3) produce different MIL ops?
     → If axes differ in the MIL spec, positive indexing may help.
  2. Does coremltools fold sum/H back into reduce_mean?
     → If yes, the op itself is the issue, not the axis.
  3. Is layer_norm an alternative that avoids reduce_mean?
     → layer_norm is a single fused op, known to work on ANE.
  4. Is the issue the NUMBER of reduce_mean ops in large graphs?
     → One reduce_mean per RMSNorm × layers × prefill stages.
""")
