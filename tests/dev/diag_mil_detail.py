#!/usr/bin/env python3
"""Deep-inspect the k_cache slice_update `begin` (concat_3) and its inputs.

Also check the `x` input path: read_state_0 vs cast_8.
"""
import coremltools as ct
from coremltools.proto import Model_pb2
import os

MODEL_DIR = "/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3"


def format_arg(binding):
    if binding.HasField('name'):
        return f"var={binding.name}"
    elif binding.HasField('value'):
        val = binding.value
        which = val.WhichOneof('value')
        if which == 'immediateValue':
            imm = val.immediateValue
            imm_which = imm.WhichOneof('value')
            if imm_which == 'tensor':
                t = imm.tensor
                if t.HasField('ints'):
                    return f"const_ints={list(t.ints.values)}"
                elif t.HasField('floats'):
                    return f"const_floats[{len(t.floats.values)}]"
                elif t.HasField('bytes'):
                    return f"const_bytes[{len(t.bytes.values)}]"
                else:
                    return f"tensor(?)"
            elif imm_which in ('i', 'f', 'b', 's'):
                return f"{imm_which}={getattr(imm, imm_which)}"
            else:
                return f"imm({imm_which})"
        elif which == 'blobFileValue':
            blob = val.blobFileValue
            return f"blob(offset={blob.offset})"
        else:
            return f"{which}=..."
    return "???"


def trace_var_backwards(block, target_var_name, depth=0, visited=None):
    """Find the operation that produces the given variable and print its inputs."""
    if visited is None:
        visited = set()
    if target_var_name in visited:
        return
    visited.add(target_var_name)

    prefix = "  " * depth
    for i, op in enumerate(block.operations):
        for out in op.outputs:
            if out.name == target_var_name:
                # Found the producing op
                tp = out.type
                shape_str = ""
                if tp.HasField('tensorType'):
                    dims = []
                    for d in tp.tensorType.dimensions:
                        if d.HasField('constant'):
                            dims.append(d.constant.size)
                        else:
                            dims.append('?')
                    shape_str = f" shape={dims}"
                print(f"{prefix}[{i:4d}] {op.type} -> {out.name}{shape_str}")
                for inp_name in op.inputs:
                    inp_arg = op.inputs[inp_name]
                    args = []
                    for binding in inp_arg.arguments:
                        args.append(format_arg(binding))
                    print(f"{prefix}       {inp_name}: {args}")
                    # Recurse into var references if depth allows
                    if depth < 4:
                        for binding in inp_arg.arguments:
                            if binding.HasField('name') and binding.name not in visited:
                                trace_var_backwards(block, binding.name, depth + 1, visited)
                return


def inspect_chunk_detail(chunk_idx):
    path = f"{MODEL_DIR}/prefill_LUT4_chunk{chunk_idx}.mlpackage"
    print(f"\n{'='*80}")
    print(f"Chunk {chunk_idx}: {path}")

    model = ct.models.MLModel(path)
    spec = model.get_spec()
    prog = spec.mlProgram
    fn = prog.functions["main"]

    block = None
    for bname in fn.block_specializations:
        block = fn.block_specializations[bname]
        break

    # Find k_cache and v_cache slice_update ops
    k_slice_idx = None
    v_slice_idx = None
    for i, op in enumerate(block.operations):
        if op.type == 'slice_update':
            for out in op.outputs:
                if 'k_cache_internal_tensor_assign' in out.name:
                    k_slice_idx = i
                elif 'v_cache_internal_tensor_assign' in out.name:
                    v_slice_idx = i

    print(f"\n  k_cache slice_update at op [{k_slice_idx}]")
    if k_slice_idx is not None:
        op = block.operations[k_slice_idx]
        # Print all inputs and trace them
        for inp_name in op.inputs:
            inp_arg = op.inputs[inp_name]
            for binding in inp_arg.arguments:
                arg_str = format_arg(binding)
                print(f"    {inp_name}: {arg_str}")
                if binding.HasField('name'):
                    print(f"    --- Tracing {binding.name} ---")
                    trace_var_backwards(block, binding.name, depth=1, visited=set())

    print(f"\n  v_cache slice_update at op [{v_slice_idx}]")
    if v_slice_idx is not None:
        op = block.operations[v_slice_idx]
        for inp_name in op.inputs:
            inp_arg = op.inputs[inp_name]
            for binding in inp_arg.arguments:
                arg_str = format_arg(binding)
                print(f"    {inp_name}: {arg_str}")
                if binding.HasField('name'):
                    print(f"    --- Tracing {binding.name} ---")
                    trace_var_backwards(block, binding.name, depth=1, visited=set())


if __name__ == "__main__":
    for idx in [1, 3]:
        inspect_chunk_detail(idx)
