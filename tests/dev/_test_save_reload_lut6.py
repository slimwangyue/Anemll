#!/usr/bin/env python3
"""Test: save and reload LUT6 chunk, verify make_state() works."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import coremltools as ct
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from anemll.models.qwen3_5_model import Qwen35Model

TMP = "/tmp/test_lut6_chunk0.mlpackage"

print("Loading model...")
model = Qwen35Model.from_pretrained("models/Qwen__Qwen3.5-4B")

print("Converting chunk 0 LUT6 (no ANE safe numerics)...")
conv = Qwen35Converter(model, context_length=1024, batch_size=256,
                       num_chunks=4, lut_bits=6, per_channel=8,
                       ane_safe_numerics=False)
ml = conv.convert_part_2(model, chunk_idx=0, total_chunks=4)

print("Saving...")
ml.save(TMP)
del ml, conv; gc.collect()

for cu_name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
                     ("ALL", ct.ComputeUnit.ALL),
                     ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU)]:
    print(f"\nReloading with {cu_name}...")
    m = ct.models.MLModel(TMP, compute_units=cu)
    try:
        s = m.make_state()
        print(f"  make_state() SUCCESS with {cu_name}")
    except Exception as e:
        print(f"  make_state() FAILED with {cu_name}: {e}")
    del m; gc.collect()
