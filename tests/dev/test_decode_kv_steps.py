#!/usr/bin/env python3
"""Minimal test: run a few decode steps with kv_write_end."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import coremltools as ct

MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_2"

print("Loading decode chunk 0...")
m = ct.models.MLModel(f"{MODEL_DIR}/ffn_LUT4_chunk0.mlpackage",
                       compute_units=ct.ComputeUnit.CPU_AND_NE)
state = m.make_state()

# Get lin state shapes from spec
spec = m.get_spec()
shapes = {}
for inp in spec.description.input:
    try:
        shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
    except Exception:
        pass
print("Input shapes:", shapes)

lin_conv = np.zeros(shapes['linear_conv_state'], dtype=np.float16)
lin_rec = np.zeros(shapes['linear_recurrent_state'], dtype=np.float16)

print("\nRunning decode steps...")
for pos in range(20):
    kv_end_size = pos + 1
    inp = {
        "hidden_states": np.zeros((1, 1, 2560), dtype=np.float16),
        "position_ids": np.array([pos], dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, 1024), dtype=np.float16),
        "current_pos": np.array([pos], dtype=np.int32),
        "linear_conv_state": lin_conv,
        "linear_recurrent_state": lin_rec,
        "kv_write_end": np.zeros((kv_end_size,), dtype=np.int32),
    }
    print(f"  pos={pos}, kv_write_end shape=({kv_end_size},)...", end=" ", flush=True)
    try:
        out = m.predict(inp, state=state)
        if 'linear_conv_state_out' in out:
            lin_conv = out['linear_conv_state_out']
            lin_rec = out['linear_recurrent_state_out']
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")
        break

print("\nDone!")
