#!/usr/bin/env python3
"""Apply LUT6 quantization to the fp16+argmax LM head intermediate.

Usage:
    python tests/dev/qwen35_quantize_lmhead.py
"""
import sys, os, time, shutil, warnings, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import coremltools as ct
import coremltools.optimize as cto
try:
    from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
except Exception:
    SklearnConvergenceWarning = None

BASE = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"
FP16_PATH = os.path.join(BASE, "_lm_head_fp16_argmax_tmp.mlpackage")
OUT_PATH = os.path.join(BASE, "lm_head.mlpackage")
LUT_BITS = 6
PER_CHANNEL = 8

print(f"Loading fp16+argmax from {FP16_PATH}...")
t0 = time.time()
mlmodel = ct.models.MLModel(FP16_PATH)
print(f"  Loaded in {time.time()-t0:.1f}s")

# Verify outputs
spec = mlmodel.get_spec()
output_names = [o.name for o in spec.description.output]
print(f"  Outputs: {output_names}")

print(f"\nApplying LUT{LUT_BITS} quantization (per_channel={PER_CHANNEL})...")
print("  This will take ~30 minutes for the 248320×2560 weight matrix...")
t0 = time.time()

with warnings.catch_warnings():
    if SklearnConvergenceWarning is not None:
        warnings.simplefilter("ignore", SklearnConvergenceWarning)
    warnings.simplefilter("ignore", UserWarning)
    from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig
    cfg = OpPalettizerConfig(
        mode="kmeans",
        nbits=LUT_BITS,
        granularity="per_grouped_channel",
        group_size=PER_CHANNEL,
        num_kmeans_workers=1,
    )
    config = OptimizationConfig(global_config=cfg)
    mlmodel = cto.coreml.palettize_weights(mlmodel, config)

print(f"  Quantized in {time.time()-t0:.1f}s")

# Save
if os.path.exists(OUT_PATH):
    shutil.rmtree(OUT_PATH)
mlmodel.save(OUT_PATH)
del mlmodel; gc.collect()

# Print size
total_bytes = 0
for dirpath, _, filenames in os.walk(OUT_PATH):
    for fn in filenames:
        total_bytes += os.path.getsize(os.path.join(dirpath, fn))
print(f"\n  Saved LUT{LUT_BITS}+argmax lm_head ({total_bytes/1e6:.1f} MB)")

# Quick load test
print("\nQuick load test (CPU_AND_NE)...")
try:
    loaded = ct.models.MLModel(OUT_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = loaded.get_spec()
    output_names = [o.name for o in spec.description.output]
    print(f"  Outputs: {output_names}")

    dummy = np.zeros((1, 1, 2560), dtype=np.float16)
    result = loaded.predict({"hidden_states": dummy})
    print(f"  argmax_idx: {result['argmax_idx'].flatten()[:3]}")
    print(f"  argmax_val: {result['argmax_val'].flatten()[:3]}")
    print("  Load test PASSED")
    del loaded
except Exception as e:
    print(f"  Load test FAILED: {e}")

# Clean up intermediate
print("\nCleaning up fp16 intermediate...")
shutil.rmtree(FP16_PATH)
print("  Done")

print(f"\n{'='*60}")
print(f"  lm_head.mlpackage — LUT{LUT_BITS} + argmax COMPLETE")
print(f"{'='*60}")
