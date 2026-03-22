#!/usr/bin/env python3
"""Test which dynamic KV chunks can run on ANE."""
import coremltools as ct
import numpy as np
import os

MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_2"
CTX = 1024

for i in range(4):
    p = os.path.join(MODEL_DIR, f"ffn_LUT4_chunk{i}.mlpackage")
    m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = m.make_state()
    spec = m.get_spec()
    inp_map = {}
    for inp in spec.description.input:
        try:
            inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
        except Exception:
            pass

    hidden = np.random.randn(*inp_map["hidden_states"]).astype(np.float16)
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :1] = 0
    kv_write_end = np.zeros((1,), dtype=np.int32)

    feed = {
        "hidden_states": hidden,
        "position_ids": np.array([0], dtype=np.int32),
        "causal_mask": mask,
        "current_pos": np.array([0], dtype=np.int32),
        "linear_conv_state": np.zeros(inp_map["linear_conv_state"], dtype=np.float16),
        "linear_recurrent_state": np.zeros(inp_map["linear_recurrent_state"], dtype=np.float16),
        "kv_write_end": kv_write_end,
    }
    try:
        out = m.predict(feed, state=state)
        print(f"chunk{i}: ANE predict OK")
    except Exception as e:
        print(f"chunk{i}: ANE predict FAILED: {str(e)[:150]}")
    del m, state
