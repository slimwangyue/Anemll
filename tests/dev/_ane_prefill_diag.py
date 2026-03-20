#!/usr/bin/env python3
"""Diagnose ANE rejection of prefill chunk 1."""
import coremltools as ct
import numpy as np

export_dir = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export"
path = f"{export_dir}/qwen35_prefill_chunk_01of04.mlpackage"

print("Loading model...")
model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)

spec = model.get_spec()
print("\n=== Model I/O ===")
for inp in spec.description.input:
    if inp.type.HasField("multiArrayType"):
        print(f"  input: {inp.name} shape={tuple(inp.type.multiArrayType.shape)} dtype={inp.type.multiArrayType.dataType}")
    elif inp.type.HasField("stateType"):
        print(f"  state: {inp.name} shape={tuple(inp.type.stateType.multiArrayType.shape)} dtype={inp.type.stateType.multiArrayType.dataType}")
for out in spec.description.output:
    if out.type.HasField("multiArrayType"):
        print(f"  output: {out.name} shape={tuple(out.type.multiArrayType.shape)} dtype={out.type.multiArrayType.dataType}")

seq_len = 256
hidden = np.zeros((1, seq_len, 2560), dtype=np.float16)
position_ids = np.arange(seq_len, dtype=np.int32)
mask = np.full((1, 1, seq_len, seq_len), -65504.0, dtype=np.float16)
for r in range(seq_len):
    mask[..., r, :r + 1] = 0
current_pos = np.zeros((1,), dtype=np.int32)

state = model.make_state()

print("\n=== Trying CPU_AND_NE ===")
try:
    out = model.predict(
        {"hidden_states": hidden, "position_ids": position_ids,
         "causal_mask": mask, "current_pos": current_pos},
        state=state,
    )
    print("SUCCESS on CPU_AND_NE")
    for k, v in out.items():
        print(f"  {k}: shape={v.shape}")
except Exception as e:
    print(f"FAILED on CPU_AND_NE:\n{e}")

print("\n=== Trying ALL ===")
del model, state
model2 = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
state2 = model2.make_state()
try:
    out = model2.predict(
        {"hidden_states": hidden, "position_ids": position_ids,
         "causal_mask": mask, "current_pos": current_pos},
        state=state2,
    )
    print("SUCCESS on ALL")
except Exception as e:
    print(f"FAILED on ALL:\n{e}")
