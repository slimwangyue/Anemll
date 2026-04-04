#!/usr/bin/env python3
"""Analyze MIL ops in the prefill chunk to understand the 65K op bloat."""
import coremltools as ct
import collections

model_path = '/Users/yw68/Anemll/qwen3_5_4chunk_lut6_bs512_ctx2048/prefill_LUT6_chunk0_bs512.mlpackage'
model = ct.models.MLModel(model_path)
spec = model.get_spec()
prog = spec.mlProgram

# Gather all ops with their names and types
ops_by_type = collections.defaultdict(list)
for func in prog.functions.values():
    for block in func.block_specializations.values():
        for op in block.operations:
            out_names = [o.name for o in op.outputs]
            ops_by_type[op.type].append(out_names)

# Print summary
print("=== OP TYPE SUMMARY ===")
total = sum(len(v) for v in ops_by_type.values())
print(f"Total ops: {total}")
for op_type, ops in sorted(ops_by_type.items(), key=lambda x: -len(x[1]))[:15]:
    print(f"  {op_type}: {len(ops)}")
    # Show a few sample output names
    samples = ops[:5]
    for s in samples:
        print(f"    -> {s}")

# Analyze slice_by_index naming patterns
print("\n=== SLICE_BY_INDEX PATTERNS ===")
slice_names = [n for names in ops_by_type.get('slice_by_index', []) for n in names]
# Group by common substrings
patterns = collections.Counter()
for name in slice_names:
    # Try to find the pattern (remove digits at end)
    import re
    pattern = re.sub(r'_\d+$', '', name)
    pattern = re.sub(r'_\d+_', '_N_', pattern)
    patterns[pattern] += 1
for p, c in patterns.most_common(20):
    print(f"  {p}: {c}")

# Analyze const shapes
print("\n=== CONST SHAPES ===")
shape_counter = collections.Counter()
for func in prog.functions.values():
    for block in func.block_specializations.values():
        for op in block.operations:
            if op.type == 'const':
                for o in op.outputs:
                    t = o.type
                    # Try to get shape info from the type
                    if hasattr(t, 'tensorType'):
                        tt = t.tensorType
                        dims = []
                        for d in tt.dimensions:
                            if d.constant.size > 0:
                                dims.append(str(d.constant.size))
                            else:
                                dims.append('?')
                        shape_counter[str(dims)] += 1
                    elif hasattr(t, 'scalarType'):
                        shape_counter['scalar'] += 1
                    else:
                        shape_counter['unknown'] += 1

for s, c in shape_counter.most_common(20):
    print(f"  {s}: {c}")
