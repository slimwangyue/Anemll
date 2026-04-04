#!/usr/bin/env python3
"""Quick test: Qwen35LinearConvStage with valid_len — correctness + ANE export.

Tests:
1. Correctness: valid_len one-hot path matches naive gather reference
2. CoreML export: traces with valid_len input, converts, predicts on ANE
"""
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, repo_root)

from anemll.models.qwen3_5_model import Qwen35LinearConvStage, MODEL_DTYPE

# ── Config matching Qwen3.5-4B ──
CONV_DIM = 8192
CONV_KERNEL = 4
SEQ_LEN = 256       # fixed bucket size
VALID_LEN = 200     # < SEQ_LEN — 200 real tokens, 56 padding


def make_conv_stage():
    """Build a Qwen35LinearConvStage with random weights."""
    conv2d = nn.Conv2d(
        CONV_DIM, CONV_DIM,
        kernel_size=(1, CONV_KERNEL),
        padding=0,
        groups=CONV_DIM,
        bias=False,
        dtype=MODEL_DTYPE,
    )
    return Qwen35LinearConvStage(conv2d, CONV_KERNEL)


def reference_gather_state(stacked_3d, valid_len_val, k):
    """Naive reference: gather next_state from stacked using Python indexing."""
    # stacked_3d: [B, C, k + S]
    start = int(valid_len_val)
    return stacked_3d[:, :, start:start + k]


# ═══════════════════════════════════════════════════════════════════
# Test 1: Correctness
# ═══════════════════════════════════════════════════════════════════
def test_correctness():
    print("=" * 60)
    print("Test 1: Correctness — one-hot vs naive gather")
    print("=" * 60)

    stage = make_conv_stage()
    stage.eval()

    torch.manual_seed(42)
    mixed_qkv = torch.randn(1, CONV_DIM, 1, SEQ_LEN, dtype=MODEL_DTYPE)
    conv_state = torch.randn(1, CONV_DIM, CONV_KERNEL, dtype=MODEL_DTYPE)
    valid_len_t = torch.tensor([VALID_LEN], dtype=torch.int32)

    with torch.no_grad():
        out_vl, next_state_vl = stage(
            mixed_qkv, conv_state,
            expected_seq_len=SEQ_LEN,
            valid_len=valid_len_t,
        )

    # Build reference
    with torch.no_grad():
        stacked = torch.cat([conv_state.unsqueeze(2), mixed_qkv], dim=-1)
        stacked_3d = stacked.squeeze(2)
        ref_state = reference_gather_state(stacked_3d, VALID_LEN, CONV_KERNEL)

    diff = (next_state_vl.float() - ref_state.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Also test valid_len=None (full-sequence, should take last-k)
    with torch.no_grad():
        out_no_vl, next_state_no_vl = stage(
            mixed_qkv, conv_state,
            expected_seq_len=SEQ_LEN,
            valid_len=None,
        )
    ref_last_k = stacked_3d[:, :, -CONV_KERNEL:]
    diff_no_vl = (next_state_no_vl.float() - ref_last_k.float()).abs()

    print(f"  valid_len={VALID_LEN}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    print(f"  valid_len=None (last-k): max_diff={diff_no_vl.max().item():.6e}")
    print(f"  out shape: {out_vl.shape}")
    print(f"  next_state shape: {next_state_vl.shape}")

    # Outputs should be identical (valid_len only affects state extraction)
    out_diff = (out_vl.float() - out_no_vl.float()).abs().max().item()
    print(f"  out diff (vl vs no_vl): {out_diff:.6e} (should be 0)")

    ok = max_diff < 1e-4 and out_diff < 1e-6
    print(f"  PASS: {ok}")
    return ok


# ═══════════════════════════════════════════════════════════════════
# Test 2: CoreML export + ANE load
# ═══════════════════════════════════════════════════════════════════
def test_ane_export():
    print()
    print("=" * 60)
    print("Test 2: CoreML export with valid_len → ANE prediction")
    print("=" * 60)

    try:
        import coremltools as ct
    except ImportError:
        print("  SKIP: coremltools not installed")
        return True

    stage = make_conv_stage()
    stage.eval()

    # Wrapper for tracing: takes (mixed_qkv, conv_state, valid_len) → (out, next_state)
    class ConvStageWrapper(nn.Module):
        def __init__(self, conv_stage, seq_len):
            super().__init__()
            self.conv_stage = conv_stage
            self.seq_len = seq_len

        def forward(self, mixed_qkv, conv_state, valid_len):
            out, next_state = self.conv_stage(
                mixed_qkv, conv_state,
                expected_seq_len=self.seq_len,
                valid_len=valid_len,
            )
            return out, next_state

    wrapper = ConvStageWrapper(stage, SEQ_LEN)
    wrapper.eval()

    # Trace inputs
    torch.manual_seed(42)
    mixed_qkv = torch.randn(1, CONV_DIM, 1, SEQ_LEN, dtype=MODEL_DTYPE)
    conv_state = torch.randn(1, CONV_DIM, CONV_KERNEL, dtype=MODEL_DTYPE)
    valid_len_t = torch.tensor([VALID_LEN], dtype=torch.int32)

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (mixed_qkv, conv_state, valid_len_t))

    # Get PyTorch reference
    with torch.no_grad():
        ref_out, ref_state = wrapper(mixed_qkv, conv_state, valid_len_t)

    print("  Converting to CoreML...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mixed_qkv", shape=(1, CONV_DIM, 1, SEQ_LEN)),
            ct.TensorType(name="conv_state", shape=(1, CONV_DIM, CONV_KERNEL)),
            ct.TensorType(name="valid_len", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="out"),
            ct.TensorType(name="next_state"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  Converted in {time.time() - t0:.1f}s")

    # Count ops
    spec = mlmodel.get_spec()
    ml_prog = spec.mlProgram
    ops_by_type = {}
    for fn in ml_prog.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                op_type = op.type
                ops_by_type[op_type] = ops_by_type.get(op_type, 0) + 1
    total_ops = sum(ops_by_type.values())
    print(f"  MIL ops: {total_ops}")
    for op_type in ["gather_along_axis", "gather", "less", "one_hot"]:
        if op_type in ops_by_type:
            print(f"    {op_type}: {ops_by_type[op_type]}")

    gather_count = ops_by_type.get("gather_along_axis", 0) + ops_by_type.get("gather", 0)
    less_count = ops_by_type.get("less", 0)
    print(f"  ANE-hostile ops: gather={gather_count}, less={less_count}")

    # Save tmp and predict
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="conv_stage_ane_")
    pkg_path = os.path.join(tmp_dir, "conv_stage.mlpackage")
    mlmodel.save(pkg_path)
    print(f"  Saved: {pkg_path}")

    print("  Running prediction (CPU_AND_NE)...")
    t0 = time.time()
    try:
        pred = mlmodel.predict({
            "mixed_qkv": mixed_qkv.float().numpy(),
            "conv_state": conv_state.float().numpy(),
            "valid_len": np.array([VALID_LEN], dtype=np.int32),
        })
        print(f"  Predicted in {time.time() - t0:.3f}s")

        pred_out = torch.from_numpy(pred["out"]).to(MODEL_DTYPE)
        pred_state = torch.from_numpy(pred["next_state"]).to(MODEL_DTYPE)

        out_cos = F.cosine_similarity(
            pred_out.float().flatten(), ref_out.float().flatten(), dim=0
        ).item()
        state_cos = F.cosine_similarity(
            pred_state.float().flatten(), ref_state.float().flatten(), dim=0
        ).item()
        out_max = (pred_out.float() - ref_out.float()).abs().max().item()
        state_max = (pred_state.float() - ref_state.float()).abs().max().item()

        print(f"  out:   cos={out_cos:.6f}, max_abs_diff={out_max:.4e}")
        print(f"  state: cos={state_cos:.6f}, max_abs_diff={state_max:.4e}")

        ok = out_cos > 0.99 and state_cos > 0.99
        print(f"  PASS: {ok}")
        return ok

    except Exception as e:
        err_str = str(e)
        print(f"  Prediction failed: {err_str}")
        if "ANE" in err_str or "-14" in err_str or "espresso" in err_str:
            print("  This looks like an ANE compilation failure!")
        return False


# ═══════════════════════════════════════════════════════════════════
# Test 3: Verify different valid_len values
# ═══════════════════════════════════════════════════════════════════
def test_multiple_valid_lens():
    print()
    print("=" * 60)
    print("Test 3: Correctness across multiple valid_len values")
    print("=" * 60)

    stage = make_conv_stage()
    stage.eval()

    torch.manual_seed(42)
    mixed_qkv = torch.randn(1, CONV_DIM, 1, SEQ_LEN, dtype=MODEL_DTYPE)
    conv_state = torch.randn(1, CONV_DIM, CONV_KERNEL, dtype=MODEL_DTYPE)

    stacked = torch.cat([conv_state.unsqueeze(2), mixed_qkv], dim=-1)
    stacked_3d = stacked.squeeze(2)

    all_ok = True
    for vl in [1, 32, 64, 128, 200, 255, SEQ_LEN]:
        valid_len_t = torch.tensor([vl], dtype=torch.int32)
        with torch.no_grad():
            _, next_state = stage(mixed_qkv, conv_state, expected_seq_len=SEQ_LEN, valid_len=valid_len_t)
        ref = reference_gather_state(stacked_3d, vl, CONV_KERNEL)
        diff = (next_state.float() - ref.float()).abs().max().item()
        ok = diff < 1e-4
        status = "OK" if ok else "FAIL"
        print(f"  valid_len={vl:>3d}: max_diff={diff:.6e} [{status}]")
        all_ok = all_ok and ok

    # Edge case: valid_len == SEQ_LEN should match the no-valid_len path
    with torch.no_grad():
        _, state_full = stage(mixed_qkv, conv_state, expected_seq_len=SEQ_LEN, valid_len=torch.tensor([SEQ_LEN], dtype=torch.int32))
        _, state_none = stage(mixed_qkv, conv_state, expected_seq_len=SEQ_LEN, valid_len=None)
    diff_edge = (state_full.float() - state_none.float()).abs().max().item()
    ok = diff_edge < 1e-4
    status = "OK" if ok else "FAIL"
    print(f"  valid_len=SEQ_LEN vs None: max_diff={diff_edge:.6e} [{status}]")
    all_ok = all_ok and ok

    print(f"  PASS: {all_ok}")
    return all_ok


if __name__ == "__main__":
    ok1 = test_correctness()
    ok3 = test_multiple_valid_lens()
    ok2 = test_ane_export()

    print()
    print("=" * 60)
    summary = "ALL PASS" if (ok1 and ok2 and ok3) else "SOME FAILURES"
    print(f"Summary: {summary}")
    print("=" * 60)
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)
