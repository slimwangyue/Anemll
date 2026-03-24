#!/usr/bin/env python3
"""Minimal ANE loading test — just embed + 1 FFN chunk, no lm_head."""
import sys, os, time
sys.path.insert(0, "/Users/yw68/Anemll")
import coremltools as ct

STABLE_DIR = "/Users/yw68/Anemll/qwen3_5_stable_models"
cu = ct.ComputeUnit.CPU_AND_NE

print("Test 1: Load embed on ANE...", flush=True)
t0 = time.time()
embed = ct.models.MLModel(os.path.join(STABLE_DIR, "embeddings.mlpackage"), compute_units=cu)
print(f"  OK in {time.time()-t0:.0f}s", flush=True)

combined_dir = os.path.join(STABLE_DIR, "combined_LUT4_dedup")
path = os.path.join(combined_dir, "chunk0.mlpackage")

print("Test 2: Load chunk0 infer on ANE...", flush=True)
t0 = time.time()
ffn = ct.models.MLModel(path, compute_units=cu, function_name="infer")
print(f"  OK in {time.time()-t0:.0f}s", flush=True)

print("Test 3: Load chunk0 prefill on ANE...", flush=True)
t0 = time.time()
pfill = ct.models.MLModel(path, compute_units=cu, function_name="prefill")
print(f"  OK in {time.time()-t0:.0f}s", flush=True)

print("ALL TESTS PASSED", flush=True)
