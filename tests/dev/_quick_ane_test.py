#!/usr/bin/env python3
"""Run a quick ANE test on prefill chunk 1, then save the error detail."""
import coremltools as ct
import numpy as np
import time

path = "/tmp/qwen35_ane_test/qwen35_prefill_chunk_01of04.mlpackage"
print("Loading...")
model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
state = model.make_state()
hidden = np.random.randn(1, 256, 2560).astype(np.float16) * 0.01
position_ids = np.arange(256, dtype=np.int32)
mask = np.full((1, 1, 256, 256), -65504.0, dtype=np.float16)
for r in range(256):
    mask[..., r, :r+1] = 0
current_pos = np.zeros((1,), dtype=np.int32)

inputs = {"hidden_states": hidden, "position_ids": position_ids,
          "causal_mask": mask, "current_pos": current_pos}

# Print timestamp for log correlation
ts = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"Running prediction at {ts}...")
try:
    out = model.predict(inputs, state=state)
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")
