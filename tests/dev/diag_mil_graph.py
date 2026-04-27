#!/usr/bin/env python3
"""Inspect MIL graph of prefill models to compare slice_update ops for k_cache/v_cache.

Compare a working chunk (chunk1) vs a broken chunk (chunk3) to find
why k_cache writes to position 0 in some chunks but not others.
"""
import coremltools as ct
import sys

MODEL_DIR = "/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3"

def inspect_prefill_mil(chunk_idx):
    """Load prefill mlpackage and inspect MIL ops related to k_cache/v_cache."""
    path = f"{MODEL_DIR}/prefill_LUT4_chunk{chunk_idx}.mlpackage"
    print(f"\n{'='*80}")
    print(f"Loading chunk {chunk_idx}: {path}")
    model = ct.models.MLModel(path)

    spec = model.get_spec()
    print(f"  Model type: {spec.WhichOneof('Type')}")

    # Check if it's a mlprogram
    if spec.WhichOneof('Type') == 'mlProgram':
        prog = spec.mlProgram
        print(f"  Functions: {list(prog.functions.keys())}" if hasattr(prog, 'functions') else "")

    # Try to load as MIL program
    try:
        mil_prog = ct.models.utils.load(path)
        print(f"  MIL program loaded")
    except Exception as e:
        print(f"  Could not load as MIL program: {e}")

    # Check state specs
    desc = spec.description
    print(f"\n  State tensors:")
    for st in desc.stateDescriptions:
        print(f"    {st.name}: shape={list(st.arrayType.shape)}")

    # Look at the protobuf for slice_update / scatter operations
    # The mlprogram spec has operations in blocks
    print(f"\n  Searching for slice_update/scatter ops on k_cache/v_cache...")

    # Use the spec's mlProgram to inspect ops
    if spec.WhichOneof('Type') == 'mlProgram':
        prog = spec.mlProgram
        for fn_name, fn in prog.functions.items():
            print(f"\n  Function: {fn_name}")
            block = fn.block_specializations.get("CoreML7", None) or fn.block_specializations.get("CoreML6", None)
            if block is None:
                # Try to get the default block
                for bname, bval in fn.block_specializations.items():
                    print(f"    Block specialization: {bname}")
                    block = bval
                    break

            if block is None:
                print("    No block found")
                continue

            # Count ops and find state-related ones
            op_count = len(block.operations)
            print(f"    Total operations: {op_count}")

            state_ops = []
            for i, op in enumerate(block.operations):
                op_type = op.type
                # Check for operations that reference k_cache or v_cache
                has_cache_ref = False
                cache_name = ""
                for inp_name, inp_arg in op.inputs.items():
                    if hasattr(inp_arg, 'name') and inp_arg.name and ('k_cache' in inp_arg.name or 'v_cache' in inp_arg.name):
                        has_cache_ref = True
                        cache_name = inp_arg.name
                    # Check for argument values
                    for binding in inp_arg.arguments:
                        if hasattr(binding, 'name') and binding.name and ('k_cache' in binding.name or 'v_cache' in binding.name):
                            has_cache_ref = True
                            cache_name = binding.name

                if has_cache_ref or 'slice' in op_type.lower() or 'scatter' in op_type.lower() or 'update' in op_type.lower():
                    state_ops.append((i, op_type, cache_name, op))

            print(f"    State/slice ops found: {len(state_ops)}")
            for idx, op_type, cache_name, op in state_ops:
                print(f"    [{idx}] {op_type} (cache_ref={cache_name})")
                # Print input bindings
                for inp_name, inp_arg in op.inputs.items():
                    bindings = []
                    for binding in inp_arg.arguments:
                        if hasattr(binding, 'name') and binding.name:
                            bindings.append(f"name={binding.name}")
                        elif hasattr(binding, 'value'):
                            val = binding.value
                            # Try to get immediate value
                            which = val.WhichOneof('value')
                            if which == 'immediateValue':
                                imm = val.immediateValue
                                imm_which = imm.WhichOneof('value')
                                if imm_which == 'tensor':
                                    t = imm.tensor
                                    if hasattr(t, 'ints') and t.ints.values:
                                        bindings.append(f"const_int={list(t.ints.values)}")
                                    elif hasattr(t, 'floats') and t.floats.values:
                                        bindings.append(f"const_float=[{len(t.floats.values)} vals]")
                                    elif hasattr(t, 'bytes') and t.bytes.values:
                                        bindings.append(f"const_bytes=[{len(t.bytes.values)} bytes]")
                                    else:
                                        bindings.append(f"tensor({t})")
                                elif imm_which == 'i':
                                    bindings.append(f"const_i={imm.i}")
                                elif imm_which == 'f':
                                    bindings.append(f"const_f={imm.f}")
                                elif imm_which == 'b':
                                    bindings.append(f"const_b={imm.b}")
                                elif imm_which == 's':
                                    bindings.append(f"const_s={imm.s}")
                                else:
                                    bindings.append(f"imm({imm_which})")
                            else:
                                bindings.append(f"val({which})")
                        else:
                            bindings.append("?")
                    print(f"        {inp_name}: {bindings}")
                # Print outputs
                for out in op.outputs:
                    print(f"        -> {out.name} shape={list(out.type.tensorType.dimensions) if out.type.HasField('tensorType') else '?'}")

    return model


def inspect_combined_mil(chunk_idx):
    """Load combined (dedup) mlpackage and inspect the prefill function."""
    path = f"{MODEL_DIR}/combined_LUT4_dedup/qwen3_5_FFN_PF_lut4_chunk_{chunk_idx+1:02d}of09.mlpackage"
    print(f"\n{'='*80}")
    print(f"Loading combined chunk {chunk_idx}: {path}")

    import os
    if not os.path.exists(path):
        print(f"  NOT FOUND")
        return None

    model = ct.models.MLModel(path)
    spec = model.get_spec()

    desc = spec.description
    print(f"  State tensors:")
    for st in desc.stateDescriptions:
        print(f"    {st.name}: shape={list(st.arrayType.shape)}")

    if spec.WhichOneof('Type') == 'mlProgram':
        prog = spec.mlProgram
        for fn_name, fn in prog.functions.items():
            if fn_name != 'prefill':
                continue
            print(f"\n  Function: {fn_name}")
            block = None
            for bname, bval in fn.block_specializations.items():
                print(f"    Block specialization: {bname}")
                block = bval
                break

            if block is None:
                print("    No block found")
                continue

            op_count = len(block.operations)
            print(f"    Total operations: {op_count}")

            # Find slice_update ops
            for i, op in enumerate(block.operations):
                op_type = op.type
                if 'slice_update' in op_type.lower() or 'scatter' in op_type.lower():
                    # Check if it's related to k_cache or v_cache
                    related = False
                    for inp_name, inp_arg in op.inputs.items():
                        for binding in inp_arg.arguments:
                            if hasattr(binding, 'name') and binding.name and ('cache' in binding.name.lower() or 'read_state' in binding.name.lower()):
                                related = True
                    if related or True:  # print all slice_update ops
                        print(f"\n    [{i}] {op_type}")
                        for inp_name, inp_arg in op.inputs.items():
                            bindings = []
                            for binding in inp_arg.arguments:
                                if hasattr(binding, 'name') and binding.name:
                                    bindings.append(f"name={binding.name}")
                                elif hasattr(binding, 'value'):
                                    val = binding.value
                                    which = val.WhichOneof('value')
                                    if which == 'immediateValue':
                                        imm = val.immediateValue
                                        imm_which = imm.WhichOneof('value')
                                        if imm_which == 'tensor':
                                            t = imm.tensor
                                            if hasattr(t, 'ints') and t.ints.values:
                                                bindings.append(f"const_int={list(t.ints.values)}")
                                            else:
                                                bindings.append(f"tensor(...)")
                                        elif imm_which == 'i':
                                            bindings.append(f"const_i={imm.i}")
                                        elif imm_which == 'f':
                                            bindings.append(f"const_f={imm.f}")
                                        elif imm_which == 'b':
                                            bindings.append(f"const_b={imm.b}")
                                        else:
                                            bindings.append(f"imm({imm_which})")
                                    else:
                                        bindings.append(f"val({which})")
                                else:
                                    bindings.append("?")
                            print(f"        {inp_name}: {bindings}")
                        for out in op.outputs:
                            print(f"        -> {out.name}")

    return model


if __name__ == "__main__":
    # Compare working chunks {1, 2, 5, 8} vs broken chunks {3, 4, 6, 7}
    print("=== Individual prefill models (pre-combine) ===")
    for idx in [1, 3]:  # 1=working, 3=broken
        inspect_prefill_mil(idx)

    print("\n\n=== Combined models (post-combine/dedup) ===")
    for idx in [1, 3]:  # 1=working, 3=broken
        inspect_combined_mil(idx)
