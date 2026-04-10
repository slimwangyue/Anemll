#!/usr/bin/env python3
"""Diagnose Gemma4 ANE error -14 using CoreML compute plan API.

Exports decode chunk 0, saves it, then uses MLComputePlan to find
which operations ANE rejects (scheduled to CPU instead).
"""
import os, sys, time
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, os.path.join(_REPO, "scripts_gemma4"))
sys.path.insert(1, _REPO)

import torch
import numpy as np
import coremltools as ct

from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES,
    LUT_BITS, FFN_PER_CHANNEL, PER_CHANNEL,
    DEFAULT_HF_MODEL, DEFAULT_OUTPUT,
    Gemma4ForCausalLM, Gemma4Config, Gemma4Converter,
    MODEL_DTYPE, TEST_DEVICE,
)
from export import load_model


def count_mil_ops(mlmodel):
    """Count MIL ops in an MLModel."""
    spec = mlmodel.get_spec()
    op_counts = {}
    def _count_block(block):
        for op in block.operations:
            name = op.type
            op_counts[name] = op_counts.get(name, 0) + 1
            for blk in op.blocks:
                _count_block(blk)
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            _count_block(blk)
    return op_counts


def main():
    chunk_idx = 0
    start, end = CHUNK_RANGES[chunk_idx]
    output_dir = os.path.join(_REPO, "tests", "dev", "gemma4_ane_debug")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "decode_chunk00.mlpackage")

    print(f"=== Gemma4 ANE Compute Plan Analysis ===")
    print(f"Chunk 0: layers {start}-{end-1}")
    print(f"Output: {output_dir}")
    print()

    # Skip export if already exists
    if os.path.exists(model_path):
        print(f"Model already exists, loading...")
        mlmodel = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    else:
        # Load model
        print("Loading HF model...")
        t0 = time.time()
        model = load_model(os.path.expanduser(DEFAULT_HF_MODEL), CTX)
        model.model.config.force_rotation_mode = False
        print(f"  Loaded in {time.time()-t0:.1f}s")

        print("\nExporting decode chunk 0 (no LUT for speed)...")
        t0 = time.time()
        converter = Gemma4Converter(
            model=model,
            batch_size=BATCH_SIZE,
            context_length=CTX,
            lut_bits=None,  # No LUT for fast export
            per_channel=PER_CHANNEL,
            num_chunks=NUM_CHUNKS,
        )
        mlmodel = converter.convert_part_2(
            model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS, force_rotation=False
        )
        print(f"  Exported in {time.time()-t0:.1f}s")

        # Count ops
        ops = count_mil_ops(mlmodel)
        print("\n--- MIL Op Counts ---")
        for name, count in sorted(ops.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}")

        blocking = [op for op in ['gather', 'select', 'greater_equal'] if op in ops]
        if blocking:
            print(f"\nBLOCKING OPS FOUND: {blocking}")
        else:
            print(f"\nNo gather/select/greater_equal ops.")

        # Save
        print(f"\nSaving to {model_path}...")
        mlmodel.save(model_path)
        print("  Saved.")

    # --- Compute Plan Analysis ---
    print("\n=== CoreML Compute Plan Analysis ===")

    # Method 1: Try MLComputePlan (macOS 15+)
    try:
        import CoreML
        print("Trying CoreML framework directly...")
    except ImportError:
        pass

    # Method 2: Use coremltools API
    try:
        # Compile the model first
        compiled_path = os.path.join(output_dir, "decode_chunk00.mlmodelc")
        if not os.path.exists(compiled_path):
            print("Compiling model...")
            compiled_path = ct.utils.compile_model(model_path)
            print(f"  Compiled to: {compiled_path}")

        # Try loading on ANE
        print("\nLoading on CPU_AND_NE...")
        try:
            m_ane = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            print("  Loaded successfully (may still fallback to CPU)")
        except Exception as e:
            print(f"  Load error: {e}")

        # Try loading on ALL
        print("\nLoading on ALL compute units...")
        try:
            m_all = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
            print("  Loaded successfully")
        except Exception as e:
            print(f"  Load error: {e}")

        # Try loading on CPU_ONLY
        print("\nLoading on CPU_ONLY...")
        try:
            m_cpu = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
            print("  Loaded successfully")
        except Exception as e:
            print(f"  Load error: {e}")

    except Exception as e:
        print(f"Error: {e}")

    # Method 3: Check model spec for state details
    print("\n=== State Tensor Analysis ===")
    spec = mlmodel.get_spec()
    for st in spec.description.state:
        print(f"  State: {st.name}")
        if st.type.HasField('multiArrayType'):
            mt = st.type.multiArrayType
            shape = list(mt.shape)
            dtype = mt.dataType
            elements = 1
            for s in shape:
                elements *= s
            bytes_per_elem = 2  # FP16
            total_mb = elements * bytes_per_elem / (1024*1024)
            print(f"    Shape: {shape}, dtype: {dtype}")
            print(f"    Elements: {elements:,}, Size: {total_mb:.1f} MB")

    total_state_mb = 0
    for st in spec.description.state:
        if st.type.HasField('multiArrayType'):
            mt = st.type.multiArrayType
            shape = list(mt.shape)
            elements = 1
            for s in shape:
                elements *= s
            total_state_mb += elements * 2 / (1024*1024)
    print(f"\n  Total state size: {total_state_mb:.1f} MB")

    # Method 4: Analyze specific op patterns
    print("\n=== Suspicious Op Analysis ===")
    spec = mlmodel.get_spec()
    for fn_name, fn in spec.mlProgram.functions.items():
        for blk_name, blk in fn.block_specializations.items():
            _analyze_ops(blk, fn_name, blk_name)

    print("\n=== Done ===")


def _analyze_ops(block, fn_name, blk_name, depth=0):
    """Analyze operations for ANE-suspicious patterns."""
    for op in block.operations:
        # Check for cast ops - what types do they cast between?
        if op.type == "cast":
            # Try to find the dtype attribute
            for attr_name in op.attributes:
                attr = op.attributes[attr_name]
                print(f"  cast op: {attr_name} = {attr}")

        # Check for tile ops - what dimensions do they tile?
        if op.type == "tile":
            for attr_name in op.attributes:
                attr = op.attributes[attr_name]
                if attr_name == "reps":
                    print(f"  tile op reps: {attr}")

        # Check for slice_update ops (KV cache writes)
        if op.type == "slice_update":
            input_names = [inp for inp in op.inputs]
            print(f"  slice_update inputs: {input_names}")

        # Recurse into blocks
        for blk in op.blocks:
            _analyze_ops(blk, fn_name, blk_name, depth+1)


if __name__ == "__main__":
    main()
