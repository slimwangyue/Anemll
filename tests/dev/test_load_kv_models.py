#!/usr/bin/env python3
"""Quick test: can prefill and decode models with kv_write_end load and predict?"""
import coremltools as ct
import numpy as np

MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_2"

# Test decode model
print("Loading ffn_LUT4_chunk0 with CPU_AND_NE...")
try:
    m2 = ct.models.MLModel(
        f"{MODEL_DIR}/ffn_LUT4_chunk0.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    s2 = m2.make_state()
    print("  Loaded! State created OK")
    inp2 = {
        "hidden_states": np.zeros((1, 1, 2560), dtype=np.float16),
        "position_ids": np.zeros((1,), dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, 1024), dtype=np.float16),
        "current_pos": np.zeros((1,), dtype=np.int32),
        "linear_conv_state": np.zeros((8, 1024, 32), dtype=np.float16),
        "linear_recurrent_state": np.zeros((8, 32, 128, 128), dtype=np.float16),
        "kv_write_end": np.zeros((1,), dtype=np.int32),
    }
    out2 = m2.predict(inp2, state=s2)
    print("  Prediction OK!")
    print("  Output keys:", list(out2.keys()))
except Exception as e:
    print(f"  Error: {e}")

# Test prefill model with ALL
print()
print("Loading prefill_LUT4_chunk0 with ALL...")
try:
    m = ct.models.MLModel(
        f"{MODEL_DIR}/prefill_LUT4_chunk0.mlpackage",
        compute_units=ct.ComputeUnit.ALL,
    )
    s = m.make_state()
    print("  Loaded with ALL! State created OK")
    inp = {
        "hidden_states": np.zeros((1, 256, 2560), dtype=np.float16),
        "position_ids": np.arange(256, dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 256, 1024), dtype=np.float16),
        "current_pos": np.zeros((1,), dtype=np.int32),
        "linear_conv_state": np.zeros((8, 1024, 32), dtype=np.float16),
        "linear_recurrent_state": np.zeros((8, 32, 128, 128), dtype=np.float16),
        "kv_write_end": np.zeros((256,), dtype=np.int32),
    }
    out = m.predict(inp, state=s)
    print("  Prediction OK!")
    print("  Output keys:", list(out.keys()))
except Exception as e:
    print(f"  Error: {e}")

# Test prefill model with CPU_AND_NE
print()
print("Loading prefill_LUT4_chunk0 with CPU_AND_NE...")
try:
    m3 = ct.models.MLModel(
        f"{MODEL_DIR}/prefill_LUT4_chunk0.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    s3 = m3.make_state()
    print("  Loaded with CPU_AND_NE! State created OK")
except Exception as e:
    print(f"  Error: {e}")
