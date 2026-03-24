#!/usr/bin/env python3
"""Quick test: load a combined model and try make_state()."""
import coremltools as ct
import os

base = "qwen3_5_stable_models/combined_LUT4_dedup"
path = os.path.join(base, "chunk0.mlpackage")

print(f"Loading {path} with function_name='infer'...")
try:
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE,
                          function_name="infer")
    print("  Loaded OK")
    print("  Trying make_state()...")
    s = m.make_state()
    print(f"  State OK: {type(s)}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nLoading without function_name...")
try:
    m2 = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print("  Loaded OK")
    print("  Trying make_state()...")
    s2 = m2.make_state()
    print(f"  State OK: {type(s2)}")
except Exception as e:
    print(f"  FAILED: {e}")

# Try separate ffn
print("\nLoading separate ffn_LUT4_chunk0.mlpackage...")
ffn_path = "qwen3_5_stable_models/ffn_LUT4_chunk0.mlpackage"
try:
    m3 = ct.models.MLModel(ffn_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print("  Loaded OK")
    print("  Trying make_state()...")
    s3 = m3.make_state()
    print(f"  State OK: {type(s3)}")
except Exception as e:
    print(f"  FAILED: {e}")
