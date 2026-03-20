#!/usr/bin/env python3
"""Test ANE-fixed prefill chunks on Neural Engine.
Loads compiled mlmodelc from /tmp/qwen35_ane_test/ and checks if ANE accepts them.
"""
import coremltools as ct
import numpy as np
import os
import time

EXPORT_DIR = "/tmp/qwen35_ane_test"
HIDDEN_SIZE = 2560
SEQ_LEN = 256

def test_chunk(chunk_idx, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    """Load and test a single prefill chunk on the specified compute unit."""
    name = f"qwen35_prefill_chunk_{chunk_idx:02d}of04"
    # Use .mlpackage for coremltools Python API
    package_path = os.path.join(EXPORT_DIR, f"{name}.mlpackage")
    path = package_path

    if not os.path.exists(path):
        print(f"  SKIP chunk {chunk_idx}: not found")
        return None

    print(f"\n{'='*60}")
    print(f"Chunk {chunk_idx}: {os.path.basename(path)}")
    print(f"Compute unit: {compute_unit}")
    print(f"{'='*60}")

    t0 = time.time()
    model = ct.models.MLModel(path, compute_units=compute_unit)
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s")

    # Inspect I/O
    spec = model.get_spec()
    for inp in spec.description.input:
        if inp.type.HasField("multiArrayType"):
            print(f"  input: {inp.name} shape={tuple(inp.type.multiArrayType.shape)}")
        elif inp.type.HasField("stateType"):
            print(f"  state: {inp.name} shape={tuple(inp.type.stateType.multiArrayType.shape)}")

    # Build inputs
    hidden = np.random.randn(1, SEQ_LEN, HIDDEN_SIZE).astype(np.float16) * 0.01
    position_ids = np.arange(SEQ_LEN, dtype=np.int32)
    mask = np.full((1, 1, SEQ_LEN, SEQ_LEN), -65504.0, dtype=np.float16)
    for r in range(SEQ_LEN):
        mask[..., r, :r + 1] = 0
    current_pos = np.zeros((1,), dtype=np.int32)

    state = model.make_state()

    inputs = {
        "hidden_states": hidden,
        "position_ids": position_ids,
        "causal_mask": mask,
        "current_pos": current_pos,
    }

    # Run prediction
    t0 = time.time()
    try:
        out = model.predict(inputs, state=state)
        pred_time = time.time() - t0
        print(f"  Predict time: {pred_time:.3f}s")
        for k, v in out.items():
            arr = np.array(v)
            print(f"  output {k}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
        print(f"  RESULT: SUCCESS on {compute_unit}")
        return True
    except Exception as e:
        pred_time = time.time() - t0
        print(f"  Predict time: {pred_time:.3f}s")
        print(f"  RESULT: FAILED on {compute_unit}")
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("ANE Prefill Test - Testing ANE-fixed prefill chunks")
    print("=" * 60)

    # Test chunks 1 and 2 (the ones we have)
    results = {}
    for chunk_idx in [1, 2, 3, 4]:
        result = test_chunk(chunk_idx, ct.ComputeUnit.CPU_AND_NE)
        if result is not None:
            results[chunk_idx] = result

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for chunk_idx, success in results.items():
        status = "PASS (ANE)" if success else "FAIL (rejected)"
        print(f"  Chunk {chunk_idx}: {status}")

    if not results:
        print("  No chunks found to test!")
    elif all(results.values()):
        print(f"\n  ALL {len(results)} chunks passed on ANE!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  {len(failed)} chunk(s) failed: {failed}")


if __name__ == "__main__":
    main()
