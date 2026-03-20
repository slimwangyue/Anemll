#!/usr/bin/env python3
"""Deep MIL analysis of ANE-fixed prefill chunks.
Lists ALL op types and counts, flags known ANE-illegal ops.
"""
import coremltools as ct
from collections import Counter

EXPORT_DIR = "/tmp/qwen35_ane_test"
CHUNK = 1

# Known ANE-illegal ops (from Apple docs and empirical testing)
ANE_ILLEGAL = {
    "cumsum", "topk", "argsort", "sort",
    "non_zero", "scatter", "scatter_nd",
    "while_loop", "cond",
    "select",  # from torch.where
    "nms",
}

# Potentially problematic ops (may work in some configurations)
ANE_SUSPECT = {
    "gather", "gather_nd", "gather_along_axis",
    "band_part",
    "einsum",
    "tile", "dynamic_reshape",
}

path = f"{EXPORT_DIR}/qwen35_prefill_chunk_{CHUNK:02d}of04.mlpackage"
print(f"Loading {path}...")
model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)

spec = model.get_spec()
prog = spec.mlProgram

op_counts = Counter()
op_details = {}  # op_type -> list of (block, op_name, inputs)

for fn in prog.functions.values():
    for blk_name, blk in fn.block_specializations.items():
        for op in blk.operations:
            op_type = op.type
            op_counts[op_type] += 1
            if op_type not in op_details:
                op_details[op_type] = []
            # Collect input info for suspicious ops
            if op_type in ANE_ILLEGAL | ANE_SUSPECT:
                input_names = [f"{k}={v.arguments[0].name if v.arguments else '?'}"
                               for k, v in op.inputs.items()]
                op_details[op_type].append((blk_name, op.outputs[0].name if op.outputs else "?",
                                            ", ".join(input_names[:4])))

print(f"\n{'='*60}")
print(f"Op type counts (total {sum(op_counts.values())} ops)")
print(f"{'='*60}")
for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    flag = ""
    if op_type in ANE_ILLEGAL:
        flag = " *** ANE-ILLEGAL ***"
    elif op_type in ANE_SUSPECT:
        flag = " (suspect)"
    print(f"  {op_type:30s} {count:5d}{flag}")

print(f"\n{'='*60}")
print(f"ANE-ILLEGAL ops detail")
print(f"{'='*60}")
illegal_found = False
for op_type in sorted(ANE_ILLEGAL):
    if op_type in op_counts:
        illegal_found = True
        print(f"\n  {op_type} ({op_counts[op_type]} occurrences):")
        for blk, name, inputs in op_details.get(op_type, [])[:5]:
            print(f"    block={blk} output={name} inputs=({inputs})")

if not illegal_found:
    print("  None found!")

print(f"\n{'='*60}")
print(f"Suspect ops detail")
print(f"{'='*60}")
suspect_found = False
for op_type in sorted(ANE_SUSPECT):
    if op_type in op_counts:
        suspect_found = True
        print(f"\n  {op_type} ({op_counts[op_type]} occurrences):")
        for blk, name, inputs in op_details.get(op_type, [])[:5]:
            print(f"    block={blk} output={name} inputs=({inputs})")

if not suspect_found:
    print("  None found!")

# Also check for very large tensors that might exceed ANE capacity
print(f"\n{'='*60}")
print(f"State/Buffer analysis")
print(f"{'='*60}")
for inp in spec.description.input:
    if inp.type.HasField("stateType"):
        shape = tuple(inp.type.stateType.multiArrayType.shape)
        dtype = inp.type.stateType.multiArrayType.dataType
        size_bytes = 1
        for d in shape:
            size_bytes *= d
        size_bytes *= 2  # fp16
        print(f"  state: {inp.name} shape={shape} ~{size_bytes/1024/1024:.1f}MB")
    elif inp.type.HasField("multiArrayType"):
        shape = tuple(inp.type.multiArrayType.shape)
        print(f"  input: {inp.name} shape={shape}")
