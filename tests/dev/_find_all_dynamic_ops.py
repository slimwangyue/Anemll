#!/usr/bin/env python3
"""Check all ops in the new prefill model that have any dynamic (input-dependent) arguments.
"""
import coremltools as ct
from collections import defaultdict, Counter

CHUNK_PATH = "/tmp/qwen35_ane_test/qwen35_prefill_chunk_01of04.mlpackage"

print(f"Loading {CHUNK_PATH}...")
model = ct.models.MLModel(CHUNK_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)
spec = model.get_spec()
prog = spec.mlProgram

model_inputs = set()
for inp in spec.description.input:
    model_inputs.add(inp.name)

# Build producer map
producer = {}
for fn in prog.functions.values():
    for blk in fn.block_specializations.values():
        for op in blk.operations:
            for out in op.outputs:
                producer[out.name] = (op.type, op)

def depends_on_input(name, visited=None, depth=0, max_depth=15):
    """Check if a value depends on any model input."""
    if visited is None:
        visited = set()
    if name in visited or depth > max_depth:
        return set()
    visited.add(name)
    result = set()
    if name in model_inputs:
        result.add(name)
        return result
    if name not in producer:
        return result
    op_type, op = producer[name]
    if op_type == "const":
        return result
    for k, v in op.inputs.items():
        for arg in v.arguments:
            if hasattr(arg, 'name'):
                result |= depends_on_input(arg.name, visited, depth+1, max_depth)
    return result

# Find ALL ops with dynamic dependencies (excluding const, identity, cast)
dynamic_ops = defaultdict(list)
op_count = Counter()
skip_types = {"const", "identity"}

for fn in prog.functions.values():
    for blk in fn.block_specializations.values():
        for op in blk.operations:
            if op.type in skip_types:
                continue
            op_count[op.type] += 1
            # Check each input argument for dynamic dependency
            has_dynamic = False
            dynamic_inputs = {}
            for k, v in op.inputs.items():
                for arg in v.arguments:
                    if hasattr(arg, 'name'):
                        deps = depends_on_input(arg.name)
                        if deps:
                            has_dynamic = True
                            dynamic_inputs[k] = deps
            if has_dynamic:
                output_name = op.outputs[0].name if op.outputs else "?"
                dynamic_ops[op.type].append((output_name, dynamic_inputs))

print(f"\nModel inputs: {model_inputs}")
print(f"\n{'='*60}")
print("Ops with dynamic (input-dependent) arguments")
print(f"{'='*60}")

for op_type in sorted(dynamic_ops.keys()):
    entries = dynamic_ops[op_type]
    total = op_count.get(op_type, 0)
    print(f"\n  {op_type}: {len(entries)}/{total} dynamic")
    for output, dinputs in entries[:5]:
        inputs_str = ", ".join(f"{k}→{list(v)}" for k, v in dinputs.items())
        print(f"    {output}: {inputs_str}")
    if len(entries) > 5:
        print(f"    ... ({len(entries)-5} more)")

# Summary
print(f"\n{'='*60}")
print("SUMMARY of dynamic op types")
print(f"{'='*60}")
for op_type in sorted(dynamic_ops.keys()):
    dyn = len(dynamic_ops[op_type])
    total = op_count.get(op_type, 0)
    which_inputs = set()
    for _, dinputs in dynamic_ops[op_type]:
        for deps in dinputs.values():
            which_inputs |= deps
    print(f"  {op_type:25s} {dyn:5d}/{total:5d} dynamic  deps={which_inputs}")
