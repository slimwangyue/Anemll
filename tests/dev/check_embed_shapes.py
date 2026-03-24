#!/usr/bin/env python3
"""Check if stable embeddings can handle batch=512 input."""
import coremltools as ct
import numpy as np

m = ct.models.MLModel('qwen3_5_stable_models/embeddings.mlpackage',
                       compute_units=ct.ComputeUnit.CPU_ONLY)
spec = m.get_spec()
for inp in spec.description.input:
    print(f"Input: {inp.name}")
    t = inp.type.multiArrayType
    print(f"  default shape: {list(t.shape)}")
    if hasattr(t, 'enumeratedShapes') and t.enumeratedShapes.shapes:
        for s in t.enumeratedShapes.shapes:
            print(f"  enumerated: {list(s.shape)}")

# Test batch=512
try:
    ids_512 = np.ones((1, 512), dtype=np.int32)
    out = m.predict({"input_ids": ids_512})
    val = list(out.values())[0]
    print(f"  batch=512 test: output shape = {val.shape} -- OK")
except Exception as e:
    print(f"  batch=512 test FAILED: {e}")

# Test batch=256 (baseline)
ids_256 = np.ones((1, 256), dtype=np.int32)
out = m.predict({"input_ids": ids_256})
val = list(out.values())[0]
print(f"  batch=256 test: output shape = {val.shape} -- OK")

del m
