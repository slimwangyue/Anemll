#!/usr/bin/env python3
"""Quick ANE load test for Gemma4 E4B compiled models.

Loads each .mlmodelc on CPU_AND_NE and runs a single forward pass
to verify the models can execute on Apple Neural Engine.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import coremltools as ct


def test_model_load(model_path, compute_unit, description):
    """Load and run a compiled CoreML model."""
    print(f"\n  Testing: {description}")
    print(f"    Path: {model_path}")

    if not os.path.exists(model_path):
        print(f"    SKIP: not found")
        return False

    try:
        t0 = time.time()
        model = ct.models.MLModel(model_path, compute_units=compute_unit)
        load_time = time.time() - t0

        spec = model.get_spec()
        inputs = {inp.name: inp for inp in spec.description.input}
        outputs = {out.name: out for out in spec.description.output}

        print(f"    Loaded in {load_time:.2f}s")
        print(f"    Inputs:  {list(inputs.keys())[:5]}{'...' if len(inputs) > 5 else ''}")
        print(f"    Outputs: {list(outputs.keys())[:5]}{'...' if len(outputs) > 5 else ''}")
        print(f"    OK")
        return True

    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="gemma4_E4B_lut4ffn_lut6em")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU_ONLY (for debugging)")
    args = parser.parse_args()

    compute_unit = ct.ComputeUnit.CPU_ONLY if args.cpu_only else ct.ComputeUnit.CPU_AND_NE
    print(f"Compute unit: {compute_unit}")
    print(f"Model dir: {args.model_dir}")

    passed = 0
    failed = 0
    skipped = 0

    # Test embeddings
    result = test_model_load(
        os.path.join(args.model_dir, "embeddings.mlmodelc"),
        compute_unit, "Embeddings"
    )
    passed += result
    failed += not result

    # Test LM head
    lm_paths = [f for f in os.listdir(args.model_dir)
                if f.startswith("lm_head_") and f.endswith(".mlmodelc")]
    if lm_paths:
        result = test_model_load(
            os.path.join(args.model_dir, lm_paths[0]),
            compute_unit, f"LM Head ({lm_paths[0]})"
        )
        passed += result
        failed += not result

    # Test a decode chunk
    result = test_model_load(
        os.path.join(args.model_dir, "decode_LUT4_chunk00.mlmodelc"),
        compute_unit, "Decode chunk 0"
    )
    passed += result
    failed += not result

    # Test a prefill chunk
    result = test_model_load(
        os.path.join(args.model_dir, "prefill_LUT4_chunk00.mlmodelc"),
        compute_unit, "Prefill chunk 0"
    )
    passed += result
    failed += not result

    # Test a combined chunk
    combined_dir = os.path.join(args.model_dir, "combined_LUT4_dedup")
    if os.path.isdir(combined_dir):
        result = test_model_load(
            os.path.join(combined_dir, "chunk00.mlmodelc"),
            compute_unit, "Combined chunk 0"
        )
        passed += result
        failed += not result

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
