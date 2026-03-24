#!/usr/bin/env python3
"""Quick test: load compiled embed + combined chunk, verify make_state."""
import coremltools as ct
import os, time

cu = ct.ComputeUnit.CPU_AND_NE
base = "qwen3_5_stable_models"

# Load compiled embed
t0 = time.time()
embed = ct.models.CompiledMLModel(os.path.join(base, "embeddings.mlmodelc"), cu)
print(f"embed .mlmodelc loaded in {time.time()-t0:.1f}s", flush=True)

# Load compiled lm_head
t0 = time.time()
lmhead = ct.models.CompiledMLModel(os.path.join(base, "lm_head_logits.mlmodelc"), cu)
print(f"lm_head .mlmodelc loaded in {time.time()-t0:.1f}s", flush=True)

# Load combined chunk with function_name
combined = os.path.join(base, "combined_LUT4_dedup")
t0 = time.time()
m = ct.models.MLModel(os.path.join(combined, "chunk0.mlpackage"),
                       compute_units=cu, function_name="infer")
print(f"chunk0 infer loaded in {time.time()-t0:.1f}s", flush=True)

# Test make_state
t0 = time.time()
s = m.make_state()
print(f"make_state OK in {time.time()-t0:.1f}s", flush=True)

# Test predict on embed
import numpy as np
ids = np.array([[1]], dtype=np.int32)
out = embed.predict({"input_ids": ids})
print(f"embed predict OK, output shape: {list(out.values())[0].shape}", flush=True)

# Test predict on lm_head
hidden = np.zeros((1, 1, 2560), dtype=np.float16)
out = lmhead.predict({"hidden_states": hidden})
print(f"lm_head predict OK, keys: {list(out.keys())}", flush=True)

print("ALL GOOD!", flush=True)
