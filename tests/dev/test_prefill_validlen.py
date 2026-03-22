#!/usr/bin/env python3
"""Quick smoke test: prefill with valid_len on ANE."""
import coremltools as ct
import numpy as np
import time

cu = ct.ComputeUnit.CPU_AND_NE
path = "qwen3_5_stable_models/combined_LUT4_dedup/chunk3.mlpackage"

print("Loading infer + prefill for chunk 3...")
m_infer = ct.models.MLModel(path, compute_units=cu, function_name="infer")
m_prefill = ct.models.MLModel(path, compute_units=cu, function_name="prefill")
state = m_infer.make_state()

hs = np.zeros((1, 256, 2560), dtype=np.float16)
pos = np.arange(0, 256, dtype=np.int32)
mask = np.full((1, 1, 256, 1024), -65504.0, dtype=np.float16)
for i in range(100):
    mask[0, 0, i, :i + 1] = 0
conv = np.zeros((8, 1024, 32), dtype=np.float16)
rec = np.zeros((8, 32, 128, 128), dtype=np.float16)

inp = {
    "hidden_states": hs,
    "position_ids": pos,
    "causal_mask": mask,
    "current_pos": np.array([0], dtype=np.int32),
    "linear_conv_state": conv,
    "linear_recurrent_state": rec,
    "valid_len": np.array([100], dtype=np.int32),
}

print("Running prefill with valid_len=100...")
t0 = time.time()
out = m_prefill.predict(inp, state=state)
elapsed = time.time() - t0
print(f"  OK ({elapsed:.2f}s)")
hs_out = out["output_hidden_states"]
print(f"  Output shape: {hs_out.shape}")
print(f"  Output non-zero: {np.any(hs_out != 0)}")

# Test with valid_len=256 (full block)
inp["valid_len"] = np.array([256], dtype=np.int32)
mask2 = np.full((1, 1, 256, 1024), -65504.0, dtype=np.float16)
for i in range(256):
    mask2[0, 0, i, :i + 1] = 0
inp["causal_mask"] = mask2
state2 = m_infer.make_state()

print("Running prefill with valid_len=256...")
t0 = time.time()
out2 = m_prefill.predict(inp, state=state2)
elapsed = time.time() - t0
print(f"  OK ({elapsed:.2f}s)")
hs_out2 = out2["output_hidden_states"]
print(f"  Output shape: {hs_out2.shape}")

print("\nSUCCESS: valid_len works on ANE!")
