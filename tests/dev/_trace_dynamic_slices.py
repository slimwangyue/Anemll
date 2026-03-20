#!/usr/bin/env python3
"""Trace slice_update/slice_by_index dependency chains to find dynamic bounds.
Also dump full op graph for state-related ops to find the root cause of ANE rejection.
"""
import coremltools as ct
from collections import defaultdict

CHUNK_PATH = "/tmp/qwen35_ane_test/qwen35_prefill_chunk_01of04.mlpackage"

print(f"Loading {CHUNK_PATH}...")
model = ct.models.MLModel(CHUNK_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)
spec = model.get_spec()
prog = spec.mlProgram

# Build a map from output name -> op that produced it
# Also build input dependency graph
producer = {}  # output_name -> (op_type, op)
consumers = defaultdict(list)  # output_name -> list of (op_type, consuming_op)
all_ops = []

for fn in prog.functions.values():
    for blk_name, blk in fn.block_specializations.items():
        for op in blk.operations:
            all_ops.append(op)
            for out in op.outputs:
                producer[out.name] = (op.type, op)
            for k, v in op.inputs.items():
                for arg in v.arguments:
                    if hasattr(arg, 'name'):
                        consumers[arg.name].append((op.type, op, k))

# Model inputs (these are dynamic)
model_inputs = set()
for inp in spec.description.input:
    model_inputs.add(inp.name)
    if hasattr(inp, 'shortDescription'):
        pass

print(f"\nModel inputs: {model_inputs}")

def trace_deps(name, depth=0, visited=None, max_depth=8):
    """Trace back through producers to find if a value depends on a model input."""
    if visited is None:
        visited = set()
    if name in visited or depth > max_depth:
        return []
    visited.add(name)
    
    if name in model_inputs:
        return [f"{'  '*depth}INPUT: {name}"]
    
    if name not in producer:
        return [f"{'  '*depth}CONST/UNKNOWN: {name}"]
    
    op_type, op = producer[name]
    result = [f"{'  '*depth}{name} <- {op_type}"]
    
    if op_type == "const":
        return [f"{'  '*depth}CONST: {name}"]
    
    for k, v in op.inputs.items():
        for arg in v.arguments:
            if hasattr(arg, 'name'):
                sub = trace_deps(arg.name, depth+1, visited, max_depth)
                result.extend(sub)
    
    return result

# Find all slice_update ops and trace their begin/end dependencies
print(f"\n{'='*60}")
print("Tracing slice_update begin/end dependencies")
print(f"{'='*60}")

su_idx = 0
for op in all_ops:
    if op.type == "slice_update":
        print(f"\n--- slice_update #{su_idx} ---")
        begin_name = None
        end_name = None
        update_name = None
        x_name = None
        for k, v in op.inputs.items():
            for arg in v.arguments:
                if hasattr(arg, 'name'):
                    if k == "begin":
                        begin_name = arg.name
                    elif k == "end":
                        end_name = arg.name
                    elif k == "update":
                        update_name = arg.name
                    elif k == "x":
                        x_name = arg.name
                    if k in ("begin", "end", "x", "update"):
                        print(f"  {k}: {arg.name}")
        
        # Trace begin dependency
        if begin_name:
            deps = trace_deps(begin_name)
            has_dynamic = any("INPUT:" in d for d in deps)
            print(f"  begin deps ({'DYNAMIC' if has_dynamic else 'STATIC'}):")
            for d in deps[:10]:
                print(f"    {d}")
        
        # Trace end dependency
        if end_name:
            deps = trace_deps(end_name)
            has_dynamic = any("INPUT:" in d for d in deps)
            print(f"  end deps ({'DYNAMIC' if has_dynamic else 'STATIC'}):")
            for d in deps[:10]:
                print(f"    {d}")
        
        su_idx += 1

# Also check if there are any ops from the model that use current_pos
print(f"\n{'='*60}")
print("Ops consuming 'current_pos' (direct and transitive)")
print(f"{'='*60}")

def find_consumers_transitive(name, depth=0, max_depth=4, visited=None):
    """Find ops that consume this value transitively."""
    if visited is None:
        visited = set()
    if name in visited or depth > max_depth:
        return []
    visited.add(name)
    
    result = []
    for op_type, op, input_key in consumers.get(name, []):
        out_names = [o.name for o in op.outputs]
        result.append(f"{'  '*depth}{op_type}({input_key}={name}) -> {out_names}")
        for o in op.outputs:
            result.extend(find_consumers_transitive(o.name, depth+1, max_depth, visited))
    return result

# current_pos might be renamed during conversion
for inp_name in model_inputs:
    if 'pos' in inp_name.lower() or 'current' in inp_name.lower():
        deps = find_consumers_transitive(inp_name, max_depth=6)
        print(f"\n'{inp_name}' is consumed by:")
        for d in deps[:30]:
            print(f"  {d}")
