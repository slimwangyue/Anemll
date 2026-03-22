#!/usr/bin/env python3
"""Test loading all 10 models (embed + lmhead + 4 infer + 4 prefill) and running predict on each.
This will determine if the segfault is from model count or from a different bug."""
import coremltools as ct
import numpy as np
import time
import gc

cu = ct.ComputeUnit.CPU_AND_NE
mdir = "qwen3_5_stable_models"
cdir = "qwen3_5_stable_models/combined_LUT4_dedup"

print("=" * 60)
print("Loading ALL 10 models to test ANE limits")
print("=" * 60)

# 1. Embed
print("\n[1/10] embed...")
t0 = time.time()
embed = ct.models.MLModel(f"{mdir}/embeddings.mlpackage", compute_units=cu)
print(f"  OK ({time.time()-t0:.1f}s)")

# 2. LM head
print("[2/10] lm_head...")
t0 = time.time()
lmhead = ct.models.MLModel(f"{mdir}/lm_head.mlpackage", compute_units=cu)
print(f"  OK ({time.time()-t0:.1f}s)")

# 3-6. Infer (decode) chunks
infer_models = []
for ci in range(4):
    print(f"[{3+ci}/10] chunk{ci} (infer)...")
    t0 = time.time()
    m = ct.models.MLModel(f"{cdir}/chunk{ci}.mlpackage", compute_units=cu, function_name="infer")
    infer_models.append(m)
    print(f"  OK ({time.time()-t0:.1f}s)")

gc.collect()
print(f"\n  --- 6 models loaded, testing predict... ---")

# Quick infer predict test
states = [m.make_state() for m in infer_models]
lin_conv = [np.zeros((8, 1024, 32), dtype=np.float16) for _ in range(4)]
lin_rec  = [np.zeros((8, 32, 128, 128), dtype=np.float16) for _ in range(4)]

tok = np.array([[42]], dtype=np.int32)
hidden = list(embed.predict({"input_ids": tok}).values())[0]
mask = np.full((1, 1, 1, 1024), -65504.0, dtype=np.float16)
mask[:, :, :, :1] = 0

for ci in range(4):
    inp = {
        "hidden_states": hidden.astype(np.float16),
        "position_ids": np.array([0], dtype=np.int32),
        "causal_mask": mask,
        "current_pos": np.array([0], dtype=np.int32),
        "linear_conv_state": lin_conv[ci],
        "linear_recurrent_state": lin_rec[ci],
    }
    out = infer_models[ci].predict(inp, state=states[ci])
    hidden = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        lin_conv[ci] = out['linear_conv_state_out']
        lin_rec[ci] = out['linear_recurrent_state_out']

lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
print(f"  Infer predict OK! lm_out keys: {list(lm_out.keys())}")

# 7-10. Prefill chunks
print(f"\n  --- Now loading 4 prefill models (7-10)... ---\n")
prefill_models = []
for ci in range(4):
    print(f"[{7+ci}/10] chunk{ci} (prefill)...")
    t0 = time.time()
    m = ct.models.MLModel(f"{cdir}/chunk{ci}.mlpackage", compute_units=cu, function_name="prefill")
    prefill_models.append(m)
    print(f"  OK ({time.time()-t0:.1f}s)")
    gc.collect()

print(f"\n  --- All 10 models loaded! Testing prefill predict... ---")

# Prefill predict test
pf_states = [m.make_state() for m in prefill_models]
pf_lin_conv = [np.zeros((8, 1024, 32), dtype=np.float16) for _ in range(4)]
pf_lin_rec  = [np.zeros((8, 32, 128, 128), dtype=np.float16) for _ in range(4)]

tok256 = np.array([list(range(100, 356))], dtype=np.int32)
hidden256 = list(embed.predict({"input_ids": tok256}).values())[0]
mask256 = np.full((1, 1, 256, 1024), -65504.0, dtype=np.float16)
for i in range(256):
    mask256[0, 0, i, :i+1] = 0

print("  Running prefill predict through all 4 chunks...")
for ci in range(4):
    inp = {
        "hidden_states": hidden256.astype(np.float16),
        "position_ids": np.arange(0, 256, dtype=np.int32),
        "causal_mask": mask256,
        "current_pos": np.array([0], dtype=np.int32),
        "linear_conv_state": pf_lin_conv[ci],
        "linear_recurrent_state": pf_lin_rec[ci],
    }
    t0 = time.time()
    out = prefill_models[ci].predict(inp, state=pf_states[ci])
    print(f"  chunk{ci} prefill predict OK ({time.time()-t0:.3f}s)")
    hidden256 = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        pf_lin_conv[ci] = out['linear_conv_state_out']
        pf_lin_rec[ci] = out['linear_recurrent_state_out']

lm_out2 = lmhead.predict({"hidden_states": hidden256.astype(np.float16)})
print(f"\n  Prefill predict OK! lm_out keys: {list(lm_out2.keys())}")

print("\n" + "=" * 60)
print("SUCCESS: All 10 models loaded and predict works!")
print("=" * 60)
