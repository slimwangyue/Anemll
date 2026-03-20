#!/usr/bin/env python3
"""Check if existing FFN (decode) chunks work on ANE.
And compare op counts/state sizes between working and failing models.
"""
import coremltools as ct
import numpy as np
import os
import time

EXPORT_DIR = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export"

# Check decode chunk 1
path = f"{EXPORT_DIR}/qwen35_FFN_chunk_01of04.mlpackage"
print(f"Testing decode chunk: {os.path.basename(path)}")

if not os.path.exists(path):
    print("  Not found!")
    exit(1)

t0 = time.time()
model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
print(f"  Load time: {time.time()-t0:.1f}s")

spec = model.get_spec()
print("\n  I/O:")
for inp in spec.description.input:
    if inp.type.HasField("multiArrayType"):
        print(f"    input: {inp.name} shape={tuple(inp.type.multiArrayType.shape)}")
    elif inp.type.HasField("stateType"):
        shape = tuple(inp.type.stateType.multiArrayType.shape)
        size_mb = 2
        for d in shape:
            size_mb *= d
        size_mb /= 1024 * 1024
        print(f"    state: {inp.name} shape={shape} ~{size_mb:.1f}MB")

# Count ops
from collections import Counter
op_counts = Counter()
prog = spec.mlProgram
for fn in prog.functions.values():
    for blk_name, blk in fn.block_specializations.items():
        for op in blk.operations:
            op_counts[op.type] += 1

print(f"\n  Op counts ({sum(op_counts.values())} total):")
for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"    {op_type:25s} {count:5d}")

# Build minimal inputs
state = model.make_state()
hidden = np.zeros((1, 1, 2560), dtype=np.float16)
position_ids = np.zeros((1,), dtype=np.int32)
mask = np.zeros((1, 1, 1, 256), dtype=np.float16)
current_pos = np.zeros((1,), dtype=np.int32)

inputs = {
    "hidden_states": hidden,
    "position_ids": position_ids,
    "causal_mask": mask,
    "current_pos": current_pos,
}

print(f"\n  Running prediction on CPU_AND_NE...")
t0 = time.time()
try:
    out = model.predict(inputs, state=state)
    print(f"  Predict time: {time.time()-t0:.3f}s  ✅ SUCCESS on ANE")
except Exception as e:
    print(f"  Predict time: {time.time()-t0:.3f}s  ❌ FAILED on ANE")
    err_str = str(e)
    if "ANEProgramProcessRequestDirect" in err_str:
        print("  ANE Program Inference error (same as prefill)")
    else:
        print(f"  Error: {err_str[:200]}")
