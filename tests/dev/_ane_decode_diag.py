#!/usr/bin/env python3
"""Quick test: do decode chunks run on ANE?"""
import coremltools as ct
import numpy as np
import sys

export_dir = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export"
chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 1
path = f"{export_dir}/qwen35_FFN_chunk_{chunk:02d}of04.mlpackage"
print(f"Testing decode chunk {chunk}: {path}")

model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
spec = model.get_spec()
inp_dict = {}
for inp in spec.description.input:
    if not inp.type.HasField("multiArrayType"):
        continue
    name = inp.name
    shape = tuple(inp.type.multiArrayType.shape)
    dt = inp.type.multiArrayType.dataType
    print(f"  {name}: shape={shape} dtype={dt}")
    if dt == 131104:  # int32
        inp_dict[name] = np.zeros(shape, dtype=np.int32)
    else:
        inp_dict[name] = np.zeros(shape, dtype=np.float16)

state = model.make_state()
try:
    out = model.predict(inp_dict, state=state)
    print(f"SUCCESS on CPU_AND_NE! output keys: {list(out.keys())}")
except Exception as e:
    msg = str(e)[:300]
    print(f"FAILED on CPU_AND_NE: {msg}")
