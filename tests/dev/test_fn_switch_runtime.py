#!/usr/bin/env python3
"""Test: can we switch function_name on a single MLModel instance at runtime?"""
import coremltools as ct
import numpy as np
import time

cu = ct.ComputeUnit.CPU_AND_NE
path = "qwen3_5_stable_models/combined_LUT4_dedup/chunk0.mlpackage"

# Load WITHOUT explicit function_name -> uses defaultFunctionName
print("Loading model (default function)...")
t0 = time.time()
m = ct.models.MLModel(path, compute_units=cu)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  default function_name: {m.function_name}")
print(f"  input_names: {m._model_input_names_set}")

# Create state
state = m.make_state()

# Test 1: predict with default (should be infer-shaped inputs)
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

print(f"\nTest 1: predict with default function ({m.function_name})...", end=" ", flush=True)
t0 = time.time()
out = m.predict(inp, state=state)
print(f"OK ({time.time()-t0:.2f}s)")

# Now switch to prefill function
print("\nSwitching to prefill function...")
m.function_name = "prefill"
# Get the prefill function description to update input names
f_desc = m._get_function_description("prefill")
m._model_input_names_set = set([i.name for i in f_desc.input])
print(f"  function_name: {m.function_name}")
print(f"  input_names: {m._model_input_names_set}")

# Test 2: predict with prefill-shaped inputs
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

print(f"\nTest 2: predict with switched function (prefill)...", end=" ", flush=True)
try:
    t0 = time.time()
    out = m.predict(inp_pf, state=state)
    print(f"OK ({time.time()-t0:.2f}s)")
    print("  RUNTIME SWITCH WORKS!")
except Exception as e:
    print(f"FAILED: {e}")

# Switch back to infer
print("\nSwitching back to infer...")
m.function_name = "infer"
f_desc = m._get_function_description("infer")
m._model_input_names_set = set([i.name for i in f_desc.input])

print(f"\nTest 3: predict with infer again...", end=" ", flush=True)
try:
    t0 = time.time()
    out = m.predict(inp, state=state)
    print(f"OK ({time.time()-t0:.2f}s)")
    print("  SWITCH BACK WORKS!")
except Exception as e:
    print(f"FAILED: {e}")

print("\nDone.")
