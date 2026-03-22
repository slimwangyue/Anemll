#!/usr/bin/env python3
"""Test if a single model loaded without function_name can run both infer and prefill."""
import coremltools as ct
import numpy as np
import time

cu = ct.ComputeUnit.CPU_AND_NE
path = "qwen3_5_stable_models/combined_LUT4_dedup/chunk0.mlpackage"
embed_path = "qwen3_5_stable_models/embeddings.mlpackage"

print("Loading embed model...")
embed = ct.models.MLModel(embed_path, compute_units=cu)

print("Loading chunk0 WITHOUT function_name...")
t0 = time.time()
m = ct.models.MLModel(path, compute_units=cu)
print(f"  Loaded in {time.time()-t0:.1f}s")
state = m.make_state()

# Linear states
lin_conv = np.zeros((8, 1024, 32), dtype=np.float16)
lin_rec  = np.zeros((8, 32, 128, 128), dtype=np.float16)

# === Test infer (single token) ===
print("\n=== Test infer (single token) ===")
tok = np.array([[42]], dtype=np.int32)
hidden = list(embed.predict({"input_ids": tok}).values())[0]
print(f"  Hidden shape: {hidden.shape}")

mask = np.full((1, 1, 1, 1024), -65504.0, dtype=np.float16)
mask[:, :, :, :1] = 0

inp_infer = {
    "hidden_states": hidden.astype(np.float16),
    "position_ids": np.array([0], dtype=np.int32),
    "causal_mask": mask,
    "current_pos": np.array([0], dtype=np.int32),
    "linear_conv_state": lin_conv,
    "linear_recurrent_state": lin_rec,
}

try:
    t0 = time.time()
    out = m.predict(inp_infer, state=state)
    print(f"  predict() OK in {time.time()-t0:.3f}s")
    print(f"  Output keys: {list(out.keys())}")
    print(f"  output_hidden_states shape: {out['output_hidden_states'].shape}")
except Exception as e:
    print(f"  predict failed: {e}")

# Try with explicit function_name="infer"
try:
    t0 = time.time()
    out2 = m.predict(inp_infer, state=state, function_name="infer")
    print(f"  predict(function_name='infer') OK in {time.time()-t0:.3f}s")
except Exception as e:
    print(f"  predict(function_name='infer') failed: {e}")

# === Test prefill (256 tokens) ===
print("\n=== Test prefill (256 tokens) ===")
tok256 = np.array([list(range(100, 356))], dtype=np.int32)  # 256 tokens
hidden256 = list(embed.predict({"input_ids": tok256}).values())[0]
print(f"  Hidden shape: {hidden256.shape}")

mask256 = np.full((1, 1, 256, 1024), -65504.0, dtype=np.float16)
for i in range(256):
    mask256[0, 0, i, :i+1] = 0

inp_prefill = {
    "hidden_states": hidden256.astype(np.float16),
    "position_ids": np.arange(0, 256, dtype=np.int32),
    "causal_mask": mask256,
    "current_pos": np.array([0], dtype=np.int32),
    "linear_conv_state": lin_conv,
    "linear_recurrent_state": lin_rec,
}

# Reset state for clean test
state2 = m.make_state()

try:
    t0 = time.time()
    out3 = m.predict(inp_prefill, state=state2, function_name="prefill")
    elapsed = time.time() - t0
    print(f"  predict(function_name='prefill') OK in {elapsed:.3f}s")
    print(f"  Output keys: {list(out3.keys())}")
    print(f"  output_hidden_states shape: {out3['output_hidden_states'].shape}")
except Exception as e:
    print(f"  predict(function_name='prefill') failed: {e}")
    import traceback
    traceback.print_exc()

# Try without function_name (default to first which is infer - will fail with wrong shapes)
try:
    t0 = time.time()
    out4 = m.predict(inp_prefill, state=state2)
    print(f"  predict() with 256-token input (no fn name) OK in {time.time()-t0:.3f}s")
except Exception as e:
    print(f"  predict() with 256-token input (no fn name) failed: {type(e).__name__}: {e}")

print("\nDone.")
