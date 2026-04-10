#!/usr/bin/env python3
"""Test ANE fix: re-export decode chunk 0, check MIL ops, load on ANE.

Verifies that removing min() from _store_kv_global and using slice indexing
for rotary embeddings eliminates gather/select/greater_equal ops.
"""
import os
import sys
import time
import shutil
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, os.path.join(_REPO, "scripts_gemma4"))
sys.path.insert(1, _REPO)

import torch
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
    """Count MIL ops in an MLModel by inspecting the protobuf spec."""
    spec = mlmodel.get_spec()
    op_counts = {}

    def _count_block(block):
        for op in block.operations:
            name = op.type
            op_counts[name] = op_counts.get(name, 0) + 1
            # blocks is a repeated field, iterate directly
            for blk in op.blocks:
                _count_block(blk)

    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            _count_block(blk)

    return op_counts


def main():
    chunk_idx = 0
    start, end = CHUNK_RANGES[chunk_idx]

    print(f"=== ANE Fix Verification ===")
    print(f"Chunk {chunk_idx}: layers {start}-{end-1}")
    print(f"Model: {DEFAULT_HF_MODEL}")
    print()

    # --- Step 1: Load model ---
    print("Loading model...")
    t0 = time.time()
    model = load_model(DEFAULT_HF_MODEL, CTX)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    converter = Gemma4Converter(
        model,
        context_length=CTX,
        batch_size=BATCH_SIZE,
        lut_bits=LUT_BITS,
        per_channel=PER_CHANNEL,
        num_chunks=NUM_CHUNKS,
    )
    converter.lut_bits = LUT_BITS
    converter.per_channel = FFN_PER_CHANNEL

    # --- Step 2: Export decode chunk 0 ---
    print(f"\nExporting decode chunk {chunk_idx}...")
    t0 = time.time()
    mlmodel = converter.convert_part_2(
        model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS, force_rotation=False
    )
    print(f"  Exported in {time.time()-t0:.1f}s")

    # --- Step 3: Count MIL ops ---
    print("\n--- MIL Op Analysis ---")
    ops = count_mil_ops(mlmodel)

    # Print all ops sorted by count
    for name, cnt in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"  {name}: {cnt}")

    # Check for ANE-blocking ops
    blockers = {k: v for k, v in ops.items() if k in ('gather', 'select', 'greater_equal')}
    print()
    if blockers:
        print(f"FAIL: ANE-blocking ops still present: {blockers}")
        print("The fix did not eliminate all problematic ops.")
    else:
        print("PASS: No gather/select/greater_equal ops found!")

    # --- Step 4: Save and test on ANE ---
    tmp_dir = tempfile.mkdtemp(prefix="gemma4_ane_test_")
    pkg_path = os.path.join(tmp_dir, f"decode_chunk{chunk_idx:02d}.mlpackage")
    print(f"\nSaving to {pkg_path}...")
    mlmodel.save(pkg_path)

    print("\nLoading on ANE (CPU_AND_NE)...")
    t0 = time.time()
    try:
        loaded = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f"  SUCCESS: Loaded on ANE in {time.time()-t0:.1f}s")

        # Quick inference test
        print("\nRunning inference test...")
        state = loaded.make_state()
        hidden = {"hidden_states": torch.randn(1, 1, 2560, dtype=torch.float16).numpy()}
        hidden["position_ids"] = torch.zeros((1,), dtype=int).numpy()
        hidden["causal_mask"] = torch.zeros((1, 1, 1, CTX), dtype=torch.float16).numpy()
        hidden["current_pos"] = torch.zeros((1,), dtype=int).numpy()

        # Check if PLE input is needed
        spec = loaded.get_spec()
        input_names = [inp.name for inp in spec.description.input]
        if "per_layer_emb" in input_names:
            ple_dim = model.config.hidden_size_per_layer_input * model.config.num_hidden_layers
            hidden["per_layer_emb"] = torch.zeros((1, 1, ple_dim), dtype=torch.float16).numpy()

        out = loaded.predict(hidden, state=state)
        print(f"  Inference OK, output keys: {list(out.keys())}")
        for k, v in out.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

    except Exception as e:
        print(f"  FAIL: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
