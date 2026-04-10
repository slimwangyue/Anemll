#!/usr/bin/env python3
"""Minimal test: Does CoreML state API work with ANE?

Creates a tiny model (single Conv + state read/write) and tests
whether it loads on ANE with and without the state API.
"""
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import tempfile, os, time

# ---- Minimal model WITH state ----
class TinyModelWithState(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 1, bias=False, dtype=torch.float16)
        # State buffer: [2, 4, 16, 8] (small KV cache)
        self.register_buffer("cache", torch.zeros(2, 4, 16, 8, dtype=torch.float16))

    def forward(self, x, pos):
        # x: [1, 8, 1, 1], pos: [1] (int32)
        out = self.conv(x)
        # Read cache
        k = self.cache[0:1]  # [1, 4, 16, 8]
        v = self.cache[1:2]  # [1, 4, 16, 8]
        # Write to cache at pos (dynamic slice)
        self.cache[0:1, :, pos:pos+1, :] = out[:, :4, :, :]
        self.cache[1:2, :, pos:pos+1, :] = out[:, 4:, :, :]
        # Simple attention-like operation (dummy - just use residual)
        return out + x  # residual


class TinyModelNoState(nn.Module):
    """Same model but cache passed as regular input/output."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 1, bias=False, dtype=torch.float16)

    def forward(self, x, cache, pos):
        out = self.conv(x)
        cache[0:1, :, pos:pos+1, :] = out[:, :4, :, :]
        cache[1:2, :, pos:pos+1, :] = out[:, 4:, :, :]
        return out + x, cache


def test_with_state():
    print("=== Test 1: Model WITH State API ===")
    model = TinyModelWithState()
    model.eval()

    x = torch.zeros(1, 8, 1, 1, dtype=torch.float16)
    pos = torch.zeros(1, dtype=torch.int32)

    traced = torch.jit.trace(model, (x, pos))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=(1, 8, 1, 1), dtype=np.float16),
            ct.TensorType(name="pos", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(2, 4, 16, 8), dtype=np.float16),
                name="cache",
            ),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    # Count ops
    spec = mlmodel.get_spec()
    op_counts = {}
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                op_counts[op.type] = op_counts.get(op.type, 0) + 1
    print(f"  Ops: {op_counts}")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_state.mlpackage")
        mlmodel.save(path)
        print(f"  Saved to {path}")

        print("  Loading on CPU_AND_NE...")
        t0 = time.time()
        try:
            m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            print(f"  Loaded in {time.time()-t0:.1f}s")
            print("  PASS: Model with state loads on ANE")
        except Exception as e:
            print(f"  FAIL: {e}")


def test_two_states():
    print("\n=== Test 2: Model WITH TWO States (like Gemma4) ===")

    class TinyTwoState(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 8, 1, bias=False, dtype=torch.float16)
            self.register_buffer("local_cache", torch.zeros(10, 2, 16, 8, dtype=torch.float16))
            self.register_buffer("global_cache", torch.zeros(4, 2, 16, 16, dtype=torch.float16))

        def forward(self, x, pos):
            out = self.conv(x)  # [1, 8, 1, 1]
            # Write to local_cache: need [1, 2, 1, 8]
            local_val = out[:, :2, :, :].expand(1, 2, 1, 8)  # [1, 2, 1, 8]
            self.local_cache[0:1, :, pos:pos+1, :] = local_val
            # Write to global_cache: need [1, 2, 1, 16]
            global_val = out[:, :2, :, :].expand(1, 2, 1, 16)  # [1, 2, 1, 16]
            self.global_cache[0:1, :, pos:pos+1, :] = global_val
            # Read both caches
            lk = self.local_cache[0:1]
            gk = self.global_cache[0:1]
            # dummy touch
            dummy = self.local_cache[0,0,0,0] * 0 + self.global_cache[0,0,0,0] * 0
            return out + x + dummy.view(1, 1, 1, 1)

    model = TinyTwoState()
    model.eval()
    x = torch.zeros(1, 8, 1, 1, dtype=torch.float16)
    pos = torch.zeros(1, dtype=torch.int32)
    traced = torch.jit.trace(model, (x, pos))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=(1, 8, 1, 1), dtype=np.float16),
            ct.TensorType(name="pos", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(10, 2, 16, 8), dtype=np.float16),
                name="local_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(4, 2, 16, 16), dtype=np.float16),
                name="global_cache",
            ),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    spec = mlmodel.get_spec()
    op_counts = {}
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                op_counts[op.type] = op_counts.get(op.type, 0) + 1
    print(f"  Ops: {op_counts}")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_two_state.mlpackage")
        mlmodel.save(path)
        print("  Loading on CPU_AND_NE...")
        t0 = time.time()
        try:
            m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            print(f"  Loaded in {time.time()-t0:.1f}s")
            print("  PASS: Two-state model loads on ANE")
        except Exception as e:
            print(f"  FAIL: {e}")


def test_large_state():
    """Test with Gemma4-sized state tensors."""
    print("\n=== Test 3: Large State (Gemma4 size) ===")

    class TinyLargeState(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(256, 256, 1, bias=False, dtype=torch.float16)
            # Gemma4-sized: [40, 2, 512, 256] local, [8, 2, 512, 512] global
            self.register_buffer("local_cache", torch.zeros(40, 2, 512, 256, dtype=torch.float16))
            self.register_buffer("global_cache", torch.zeros(8, 2, 512, 512, dtype=torch.float16))

        def forward(self, x, pos):
            out = self.conv(x)  # [1, 256, 1, 1]
            # Write to local_cache: need [1, 2, 1, 256]
            local_val = out[:, :2, :, :].expand(1, 2, 1, 256)
            self.local_cache[0:1, :, pos:pos+1, :] = local_val
            # Write to global_cache: need [1, 2, 1, 512]
            global_val = out[:, :2, :, :].expand(1, 2, 1, 512)
            self.global_cache[0:1, :, pos:pos+1, :] = global_val
            dummy = self.local_cache[0,0,0,0] * 0 + self.global_cache[0,0,0,0] * 0
            return out + x + dummy.view(1, 1, 1, 1)

    model = TinyLargeState()
    model.eval()
    x = torch.zeros(1, 256, 1, 1, dtype=torch.float16)
    pos = torch.zeros(1, dtype=torch.int32)
    traced = torch.jit.trace(model, (x, pos))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=(1, 256, 1, 1), dtype=np.float16),
            ct.TensorType(name="pos", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(40, 2, 512, 256), dtype=np.float16),
                name="local_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(8, 2, 512, 512), dtype=np.float16),
                name="global_cache",
            ),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    spec = mlmodel.get_spec()
    op_counts = {}
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                op_counts[op.type] = op_counts.get(op.type, 0) + 1
    print(f"  Ops: {op_counts}")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_large_state.mlpackage")
        mlmodel.save(path)
        print("  Loading on CPU_AND_NE...")
        t0 = time.time()
        try:
            m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            print(f"  Loaded in {time.time()-t0:.1f}s")
            print("  PASS: Large-state model loads on ANE")
        except Exception as e:
            print(f"  FAIL: {e}")


if __name__ == "__main__":
    test_with_state()
    test_two_states()
    test_large_state()
