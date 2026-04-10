#!/usr/bin/env python3
"""Binary search: find minimum number of layers that triggers ANE error -14.

Tests with 1 layer, then 2, then 3, etc. to find the breaking point.
"""
import os, sys, time, tempfile
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, os.path.join(_REPO, "scripts_gemma4"))
sys.path.insert(1, _REPO)

import torch
import numpy as np
import coremltools as ct

from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES,
    DEFAULT_HF_MODEL,
    Gemma4ForCausalLM, Gemma4Config, Gemma4Converter,
    MODEL_DTYPE, TEST_DEVICE, PER_CHANNEL,
)
from export import load_model


def test_chunk(model, num_layers, tmpdir):
    """Export a chunk with the given number of layers and test on ANE."""
    print(f"\n--- Testing {num_layers} layer(s): 0-{num_layers-1} ---")

    converter = Gemma4Converter(
        model=model,
        batch_size=BATCH_SIZE,
        context_length=CTX,
        lut_bits=None,  # No LUT for speed
        per_channel=PER_CHANNEL,
        num_chunks=1,
    )

    # Calculate total_chunks to get exactly num_layers in chunk 0
    # 42 layers / N = total_chunks (works when 42 is divisible by N)
    total_layers = model.config.num_hidden_layers  # 42
    total_chunks = total_layers // num_layers

    t0 = time.time()
    mlmodel = converter.convert_part_2(
        model, chunk_idx=0, total_chunks=total_chunks,
        force_rotation=False,
    )
    export_time = time.time() - t0

    # Count ops
    spec = mlmodel.get_spec()
    op_counts = {}
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                op_counts[op.type] = op_counts.get(op.type, 0) + 1

    total_ops = sum(op_counts.values())
    print(f"  Total MIL ops: {total_ops} (exported in {export_time:.1f}s)")
    for key in ['conv', 'matmul', 'layer_norm', 'softmax', 'gelu', 'pow', 'tanh', 'cast', 'read_state', 'write_state', 'slice_update']:
        if key in op_counts:
            print(f"    {key}: {op_counts[key]}")

    path = os.path.join(tmpdir, f"test_{num_layers}layers.mlpackage")
    mlmodel.save(path)

    # Test on ANE
    print(f"  Loading on CPU_AND_NE...")
    t0 = time.time()
    try:
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    print("=== ANE Layer Binary Search ===")
    print(f"Model: {DEFAULT_HF_MODEL}")

    model = load_model(os.path.expanduser(DEFAULT_HF_MODEL), CTX)
    model.model.config.force_rotation_mode = False

    tmpdir = os.path.join(_REPO, "tests", "dev", "gemma4_layer_search")
    os.makedirs(tmpdir, exist_ok=True)

    # Test: 1 layer (local), 1 global layer, 2, 3, 6
    results = {}
    for n in [1, 2, 3, 6]:
        try:
            ok = test_chunk(model, n, tmpdir)
            results[n] = ok
            if not ok:
                print(f"  ** ANE fails with {n} layer(s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[n] = False

    print("\n=== Summary ===")
    for n, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {n} layers: {status}")


if __name__ == "__main__":
    main()
