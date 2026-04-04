#!/usr/bin/env python3
"""Parse the model.mlmodel protobuf spec directly to analyze ops without loading weights."""
import collections
import re
import sys

sys.path.insert(0, '/Users/yw68/Anemll/.venv/lib/python3.9/site-packages')

from coremltools.proto import Model_pb2

print("Loading spec protobuf...", flush=True)
spec = Model_pb2.Model()
spec_path = '/Users/yw68/Anemll/qwen3_5_4chunk_lut6_bs512_ctx2048/prefill_LUT6_chunk0_bs512.mlpackage/Data/com.apple.CoreML/model.mlmodel'
with open(spec_path, 'rb') as f:
    spec.ParseFromString(f.read())
print("Spec loaded.", flush=True)

prog = spec.mlProgram

slice_names = []
const_names = []
all_op_counts = collections.Counter()

for func in prog.functions.values():
    for block in func.block_specializations.values():
        for op in block.operations:
            all_op_counts[op.type] += 1
            names = [o.name for o in op.outputs]
            if op.type == 'slice_by_index':
                slice_names.extend(names)
            elif op.type == 'const':
                const_names.extend(names)

total = sum(all_op_counts.values())
print(f"\nTotal ops: {total}")
for op, cnt in all_op_counts.most_common(15):
    print(f"  {op}: {cnt}")

print(f"\nslice_by_index count: {len(slice_names)}")
print(f"const count: {len(const_names)}")

# Analyze slice patterns
print("\n=== SLICE_BY_INDEX PATTERNS (top 20) ===")
slice_patterns = collections.Counter()
for name in slice_names:
    pat = re.sub(r'\d+', 'N', name)
    slice_patterns[pat] += 1
for p, c in slice_patterns.most_common(20):
    print(f"  {c:5d}  {p}")

# Show sample names
print("\n=== SAMPLE SLICE NAMES (first 30) ===")
for n in slice_names[:30]:
    print(f"  {n}")

# Analyze const patterns
print("\n=== CONST PATTERNS (top 20) ===")
const_patterns = collections.Counter()
for name in const_names:
    pat = re.sub(r'\d+', 'N', name)
    const_patterns[pat] += 1
for p, c in const_patterns.most_common(20):
    print(f"  {c:5d}  {p}")

# Show sample names
print("\n=== SAMPLE CONST NAMES (first 30) ===")
for n in const_names[:30]:
    print(f"  {n}")

print("\nDone.", flush=True)
