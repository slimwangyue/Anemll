#!/usr/bin/env python3
"""Debug: load stable models one by one to find which fails make_state()."""
import coremltools as ct
import os, time, gc

cu = ct.ComputeUnit.CPU_AND_NE
base = "qwen3_5_stable_models"
combined = os.path.join(base, "combined_LUT4_dedup")

# Load embed - prefer .mlmodelc
print("Loading embeddings...", flush=True)
t0 = time.time()
embed_path = os.path.join(base, "embeddings.mlmodelc")
if os.path.exists(embed_path):
    embed = ct.models.CompiledMLModel(embed_path, cu)
else:
    embed = ct.models.MLModel(os.path.join(base, "embeddings.mlpackage"), compute_units=cu)
print(f"  embed loaded in {time.time()-t0:.0f}s", flush=True)

# Load lm_head - prefer .mlmodelc
print("Loading lm_head...", flush=True)
t0 = time.time()
for name in ["lm_head_logits", "lm_head"]:
    lm_path = os.path.join(base, name + ".mlmodelc")
    if os.path.exists(lm_path):
        lmhead = ct.models.CompiledMLModel(lm_path, cu)
        break
    lm_path = os.path.join(base, name + ".mlpackage")
    if os.path.exists(lm_path):
        lmhead = ct.models.MLModel(lm_path, compute_units=cu)
        break
print(f"  lm_head loaded in {time.time()-t0:.0f}s", flush=True)

# Load combined chunks (infer only)
ffns = []
for ci in range(4):
    path = os.path.join(combined, f"chunk{ci}.mlpackage")
    print(f"Loading chunk {ci}...", flush=True)
    t0 = time.time()
    m = ct.models.MLModel(path, compute_units=cu, function_name="infer")
    print(f"  chunk {ci} loaded in {time.time()-t0:.0f}s", flush=True)
    ffns.append(m)

# Try make_state on each
for ci, m in enumerate(ffns):
    print(f"Trying make_state() on chunk {ci}...", flush=True)
    try:
        s = m.make_state()
        print(f"  chunk {ci} state OK: {type(s)}", flush=True)
    except Exception as e:
        print(f"  chunk {ci} FAILED: {e}", flush=True)

print("\nDone!", flush=True)
