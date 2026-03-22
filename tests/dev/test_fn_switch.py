#!/usr/bin/env python3
"""Test if function_name can be changed on a loaded MLModel."""
import coremltools as ct
import numpy as np

path = 'qwen3_5_stable_models/combined_LUT4_dedup/chunk0.mlpackage'
print('Loading model...', flush=True)
m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
print(f'Default function_name: {m.function_name}', flush=True)

# Test 1: Try to change function_name
try:
    m.function_name = 'prefill'
    print(f'Changed to: {m.function_name}', flush=True)
except Exception as e:
    print(f'Cannot change function_name: {e}', flush=True)

# Test 2: Try making a state and running prefill predict
print('\nCreating state...', flush=True)
state = m.make_state()

# Build prefill inputs
hidden = np.zeros((1, 256, 2560), dtype=np.float16)
pos_ids = np.arange(0, 256, dtype=np.int32)
mask = np.full((1, 1, 256, 1024), -65504.0, dtype=np.float16)
for i in range(256):
    mask[0, 0, i, :i+1] = 0
cur_pos = np.array([0], dtype=np.int32)
lin_conv = np.zeros((8, 1024, 32), dtype=np.float16)
lin_rec = np.zeros((8, 32, 128, 128), dtype=np.float16)

inp = {
    "hidden_states": hidden,
    "position_ids": pos_ids,
    "causal_mask": mask,
    "current_pos": cur_pos,
    "linear_conv_state": lin_conv,
    "linear_recurrent_state": lin_rec,
}
print('Running predict with prefill inputs...', flush=True)
try:
    out = m.predict(inp, state=state)
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: {v.shape}', flush=True)
    print('Prefill predict succeeded!', flush=True)
except Exception as e:
    print(f'Prefill predict failed: {e}', flush=True)

    # Fall back to testing infer inputs
    print('\nReverting to infer function_name...', flush=True)
    m.function_name = 'infer'
    state2 = m.make_state()
    hidden_s = np.zeros((1, 1, 2560), dtype=np.float16)
    pos_s = np.array([0], dtype=np.int32)
    mask_s = np.full((1, 1, 1, 1024), -65504.0, dtype=np.float16)
    mask_s[0, 0, 0, 0] = 0
    inp_s = {
        "hidden_states": hidden_s,
        "position_ids": pos_s,
        "causal_mask": mask_s,
        "current_pos": pos_s,
        "linear_conv_state": lin_conv,
        "linear_recurrent_state": lin_rec,
    }
    out_s = m.predict(inp_s, state=state2)
    print('Infer after revert succeeded!', flush=True)
    for k, v in out_s.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: {v.shape}', flush=True)
