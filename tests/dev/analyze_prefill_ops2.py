#!/usr/bin/env python3
"""Quick analysis of slice_by_index and const naming patterns in prefill chunk."""
import coremltools as ct
import collections
import re
import sys

print("Loading model...", flush=True)
model_path = '/Users/yw68/Anemll/qwen3_5_4chunk_lut6_bs512_ctx2048/prefill_LUT6_chunk0_bs512.mlpackage'
model = ct.models.MLModel(model_path)
spec = model.get_spec()
prog = spec.mlProgram
print("Model loaded.", flush=True)

slice_names = []
const_names = []

for func in prog.functions.values():
    for block in func.block_specializations.values():
        for op in block.operations:
            names = [o.name for o in op.outputs]
            if op.type == 'slice_by_index':
                slice_names.extend(names)
            elif op.type == 'const':
                const_names.extend(names)

print(f"\nslice_by_index count: {len(slice_names)}")
print(f"const count: {len(const_names)}")

# Analyze slice patterns - collapse numbers to N
print("\n=== SLICE_BY_INDEX PATTERNS (top 20) ===")
slice_patterns = collections.Counter()
for name in slice_names:
    pat = re.sub(r'\d+', 'N', name)
    slice_patterns[pat] += 1
for p, c in slice_patterns.most_common(20):
    print(f"  {c:5d}  {p}")

# Show some sample names
print("\n=== SAMPLE SLICE NAMES (first 20) ===")
for n in slice_names[:20]:
    print(f"  {n}")

# Analyze const patterns
print("\n=== CONST PATTERNS (top 20) ===")
const_patterns = collections.Counter()
for name in const_names:
    pat = re.sub(r'\d+', 'N', name)
    const_patterns[pat] += 1
for p, c in const_patterns.most_common(20):
    print(f"  {c:5d}  {p}")

# Show some sample const names
print("\n=== SAMPLE CONST NAMES (first 20) ===")
for n in const_names[:20]:
    print(f"  {n}")

print("\nDone.", flush=True)
