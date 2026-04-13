#!/usr/bin/env python3
"""Diagnose whether reduce_mean with fewer elements works on ANE.

H=2560 can be factored: 2560 = 256*10 = 128*20 = 64*40 = 32*80 = 16*160 = 10*256 = 8*320

Strategy: reshape [B,1,S,H] → [B,1,S*G, H/G], mean over last dim (H/G elements),
then reshape back and mean over groups (G elements).
Mathematically identical to mean(-1) over H elements.

We test:
  - Full mean (2560 elements) — baseline
  - Chunked mean with various group sizes
  - Also test ANE load (compile on Mac ANE) for each variant
"""

import torch
import torch.nn as nn
import coremltools as ct
from collections import Counter
import time
import numpy as np

H = 2560
# group_size → number of elements in each reduce_mean call
# Must divide H evenly
GROUP_SIZES = [2560, 1280, 640, 320, 256, 128, 64, 32, 16, 10, 8, 5]

SHAPES = {
    "decode":  (1, 1, 1, H),
    "prefill": (1, 1, 512, H),
}


class RMSNorm_FullMean(nn.Module):
    """Baseline: reduce_mean over all H=2560 elements."""
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.zeros(H))

    def forward(self, x):
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * (1.0 + self.weight)


class RMSNorm_ChunkedMean(nn.Module):
    """Chunked: reshape → mean(group_size) → mean(n_groups)."""
    def __init__(self, group_size):
        super().__init__()
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.zeros(H))
        self.group_size = group_size
        self.n_groups = H // group_size

    def forward(self, x):
        B, C, S, _ = x.shape
        # reshape to [..., n_groups, group_size]
        x_sq = (x * x).reshape(B, C, S, self.n_groups, self.group_size)
        # mean over group_size elements (small reduce_mean)
        group_means = x_sq.mean(-1)              # [B,C,S, n_groups]
        # mean over n_groups (also small reduce_mean)
        var = group_means.mean(-1, keepdim=True)  # [B,C,S, 1]
        return x * torch.rsqrt(var + self.eps) * (1.0 + self.weight)


class RMSNorm_Matmul(nn.Module):
    """Reference: matmul approach (known to avoid reduce_mean)."""
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.zeros(H))
        self._ones_over_h = torch.ones(H, 1) / H

    def forward(self, x):
        ones = self._ones_over_h.to(dtype=x.dtype, device=x.device)
        var = torch.matmul(x * x, ones)
        return x * torch.rsqrt(var + self.eps) * (1.0 + self.weight)


def analyze_mil(mlmodel, label):
    """Return op counts and details about reduce_mean."""
    ops = Counter()
    reduce_details = []
    for op in mlmodel._mil_program.functions['main'].operations:
        ops[op.op_type] += 1
        if op.op_type == 'reduce_mean':
            axes = op.inputs.get('axes', None)
            if axes is not None:
                axes_val = axes.val
            else:
                axes_val = '?'
            reduce_details.append(f"reduce_mean(axes={axes_val})")
    return ops, reduce_details


def test_ane_load(mlmodel, label, shape):
    """Try loading on ANE and do a predict."""
    try:
        model_ane = ct.models.MLModel(
            mlmodel.package_spec if hasattr(mlmodel, 'package_spec') else mlmodel._spec,
            compute_units=ct.ComputeUnit.CPU_AND_NE
        )
    except Exception:
        # Try saving and reloading
        import tempfile, os
        tmpdir = os.environ.get('TMPDIR', '/tmp')
        path = os.path.join(tmpdir, f"_rmsnorm_test_{label}.mlpackage")
        mlmodel.save(path)
        try:
            model_ane = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        except Exception as e:
            return f"LOAD FAIL: {e}"
        finally:
            import shutil
            shutil.rmtree(path, ignore_errors=True)

    try:
        inp = {"x": np.random.randn(*shape).astype(np.float16)}
        t0 = time.time()
        for _ in range(5):
            model_ane.predict(inp)
        dt = (time.time() - t0) / 5 * 1000
        return f"OK ({dt:.1f}ms)"
    except Exception as e:
        return f"PREDICT FAIL: {e}"


def main():
    print("=" * 100)
    print("  RMSNorm chunked-mean diagnostic: testing reduce_mean with varying element counts")
    print("=" * 100)

    for shape_name, shape in SHAPES.items():
        print(f"\n{'=' * 100}")
        print(f"  Shape: {shape_name} → {shape}")
        print(f"{'=' * 100}")
        print(f"  {'Variant':<35} │ {'Group':<8} │ {'#rm':<4} │ {'reduce_mean axes':<30} │ {'ANE test':<20} │ Key ops")
        print(f"  {'-'*35} │ {'-'*8} │ {'-'*4} │ {'-'*30} │ {'-'*20} │ {'-'*30}")

        x = torch.randn(*shape)

        # 1. Full mean baseline
        model = RMSNorm_FullMean().eval()
        ref_out = model(x)
        traced = torch.jit.trace(model, x)
        mlmodel = ct.convert(traced, inputs=[ct.TensorType(shape=shape)],
                             compute_precision=ct.precision.FLOAT16)
        ops, details = analyze_mil(mlmodel, "full")
        n_rm = ops.get('reduce_mean', 0)
        ane = test_ane_load(mlmodel, f"full_{shape_name}", shape)
        print(f"  {'BASELINE full mean (H=2560)':<35} │ {'2560':<8} │ {n_rm:<4} │ {'; '.join(details):<30} │ {ane:<20} │ {dict(ops)}")

        # 2. Matmul reference
        model_mm = RMSNorm_Matmul().eval()
        mm_out = model_mm(x)
        cos_mm = torch.nn.functional.cosine_similarity(mm_out.flatten(), ref_out.flatten(), dim=0)
        traced_mm = torch.jit.trace(model_mm, x)
        mlmodel_mm = ct.convert(traced_mm, inputs=[ct.TensorType(shape=shape)],
                                compute_precision=ct.precision.FLOAT16)
        ops_mm, details_mm = analyze_mil(mlmodel_mm, "matmul")
        n_rm_mm = ops_mm.get('reduce_mean', 0)
        ane_mm = test_ane_load(mlmodel_mm, f"matmul_{shape_name}", shape)
        print(f"  {'MATMUL ref (no reduce_mean)':<35} │ {'N/A':<8} │ {n_rm_mm:<4} │ {'; '.join(details_mm) or 'none':<30} │ {ane_mm:<20} │ {dict(ops_mm)}")

        # 3. Chunked variants
        for gs in GROUP_SIZES:
            if H % gs != 0:
                continue
            if gs == H:
                continue  # same as full mean
            n_groups = H // gs
            label = f"chunk gs={gs} ng={n_groups}"

            model_c = RMSNorm_ChunkedMean(gs).eval()
            c_out = model_c(x)
            cos = torch.nn.functional.cosine_similarity(c_out.flatten(), ref_out.flatten(), dim=0)

            traced_c = torch.jit.trace(model_c, x)
            mlmodel_c = ct.convert(traced_c, inputs=[ct.TensorType(shape=shape)],
                                   compute_precision=ct.precision.FLOAT16)
            ops_c, details_c = analyze_mil(mlmodel_c, f"gs{gs}")
            n_rm_c = ops_c.get('reduce_mean', 0)
            ane_c = test_ane_load(mlmodel_c, f"gs{gs}_{shape_name}", shape)

            flag = "⚠️" if n_rm_c > 0 else "✓"
            print(f"  {label:<35} │ {gs:<8} │ {n_rm_c:<4} │ {'; '.join(details_c):<30} │ {ane_c:<20} │ {dict(ops_c)} cos={cos:.6f}")

    print(f"\n{'=' * 100}")
    print("  CONCLUSION")
    print(f"{'=' * 100}")
    print("  If chunked variants with small group_size still show reduce_mean,")
    print("  the MIL optimizer is NOT folding them. Check if ANE test passes")
    print("  for small reduce_mean. If so, we can use chunked mean in production.")
    print("  The key question: does ANE handle reduce_mean over 8-32 elements OK?")


if __name__ == "__main__":
    main()
