#!/usr/bin/env python3
"""Test: can state be shared between infer and prefill function instances?"""
import coremltools as ct
import numpy as np
import time

cu = ct.ComputeUnit.CPU_AND_NE
path = "qwen3_5_stable_models/combined_LUT4_dedup/chunk0.mlpackage"

print("Loading infer instance...")
t0 = time.time()
m_infer = ct.models.MLModel(path, compute_units=cu, function_name="infer")
print(f"  Loaded in {time.time()-t0:.1f}s")

print("Loading prefill instance...")
t0 = time.time()
m_prefill = ct.models.MLModel(path, compute_units=cu, function_name="prefill")
print(f"  Loaded in {time.time()-t0:.1f}s")

print()
print("Infer inputs:", m_infer._model_input_names_set)
print("Prefill inputs:", m_prefill._model_input_names_set)

# Create states
state_infer = m_infer.make_state()
state_prefill = m_prefill.make_state()

# -- Test 1: infer with own state --
print()
hs = np.zeros((1, 1, 2560), dtype=np.float16)
pos = np.array([0], dtype=np.int32)
mask = np.full((1, 1, 1, 1024), -65504.0, dtype=np.float16)
mask[0, 0, 0, 0] = 0
conv = np.zeros((8, 1024, 32), dtype=np.float16)
rec = np.zeros((8, 32, 128, 128), dtype=np.float16)

inp = {
    "hidden_states": hs,
    "position_ids": pos,
    "causal_mask": mask,
    "current_pos": pos,
    "linear_conv_state": conv,
    "linear_recurrent_state": rec,
}

print("Test 1: infer predict with own state...", end=" ", flush=True)
t0 = time.time()
out = m_infer.predict(inp, state=state_infer)
print(f"OK ({time.time()-t0:.2f}s), keys={list(out.keys())}")

# -- Test 2: prefill with infer's state --
print("Test 2: prefill predict with infer state...", end=" ", flush=True)
hs_pf = np.zeros((1, 256, 2560), dtype=np.float16)
pos_pf = np.arange(0, 256, dtype=np.int32)
mask_pf = np.full((1, 1, 256, 1024), -65504.0, dtype=np.float16)
for i in range(256):
    mask_pf[0, 0, i, :i + 1] = 0

inp_pf = {
    "hidden_states": hs_pf,
    "position_ids": pos_pf,
    "causal_mask": mask_pf,
    "current_pos": np.array([0], dtype=np.int32),
    "linear_conv_state": conv.copy(),
    "linear_recurrent_state": rec.copy(),
}
try:
    t0 = time.time()
    out = m_prefill.predict(inp_pf, state=state_infer)
    print(f"OK ({time.time()-t0:.2f}s), keys={list(out.keys())}")
    print("  STATE SHARING WORKS!")
except Exception as e:
    print(f"FAILED: {e}")

# -- Test 3: prefill with own state --
print("Test 3: prefill predict with own state...", end=" ", flush=True)
try:
    t0 = time.time()
    out = m_prefill.predict(inp_pf, state=state_prefill)
    print(f"OK ({time.time()-t0:.2f}s), keys={list(out.keys())}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nDone.")
