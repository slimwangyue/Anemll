#!/usr/bin/env python3
"""Stress test: run 200 sequential single-token decode steps to find where segfault occurs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import coremltools as ct

MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_2"
CTX = 1024
NUM_CHUNKS = 4
N_STEPS = 200

def load_model(name, cu):
    p = os.path.join(MODEL_DIR, name + ".mlpackage")
    return ct.models.MLModel(p, compute_units=cu)

def get_input_shapes(model):
    spec = model.get_spec()
    shapes = {}
    for inp in spec.description.input:
        try:
            shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
        except Exception:
            pass
    return shapes

print("Loading embed...")
embed = load_model("embeddings", ct.ComputeUnit.CPU_AND_NE)
print("Loading lm_head...")
lmhead = load_model("lm_head", ct.ComputeUnit.CPU_AND_NE)
print("Loading decode chunks (CPU_AND_GPU to avoid ANE error -14)...")
ffns = [load_model(f"ffn_LUT4_chunk{i}", ct.ComputeUnit.CPU_AND_GPU) for i in range(NUM_CHUNKS)]

inp_map = get_input_shapes(ffns[0])
states = [m.make_state() for m in ffns]
lin_convs = [np.zeros(inp_map['linear_conv_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
lin_recs = [np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]

print(f"\nRunning {N_STEPS} sequential decode steps...")
sys.stdout.flush()

tok_id = 151644  # arbitrary start token
for pos in range(N_STEPS):
    sys.stdout.write(f"\rStep {pos}/{N_STEPS}...")
    sys.stdout.flush()

    tok = np.array([[tok_id]], dtype=np.int32)
    hidden = list(embed.predict({"input_ids": tok}).values())[0]

    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :pos + 1] = 0
    kv_write_end = np.zeros((pos + 1,), dtype=np.int32)

    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
            "linear_conv_state": lin_convs[ci],
            "linear_recurrent_state": lin_recs[ci],
            "kv_write_end": kv_write_end,
        }
        out = ffns[ci].predict(inp, state=states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_convs[ci] = out['linear_conv_state_out']
            lin_recs[ci] = out['linear_recurrent_state_out']

    lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if "logits" in lm_out:
        tok_id = int(np.argmax(lm_out["logits"].flatten()))
    else:
        tok_id = int(lm_out["argmax_idx"].flatten()[0])

print(f"\n\n✅ All {N_STEPS} steps completed successfully!")
