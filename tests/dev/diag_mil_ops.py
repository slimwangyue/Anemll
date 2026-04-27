#!/usr/bin/env python3
"""Inspect the MIL protobuf of prefill models to find slice_update ops on k_cache.

Dumps ops containing 'slice_update' or 'scatter' from the mlprogram spec.
"""
import coremltools as ct
from coremltools.proto import Model_pb2
import sys
import os

MODEL_DIR = "/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3"


def get_spec(path):
    spec = Model_pb2.Model()
    with open(os.path.join(path, "Data", "com.apple.CoreML", "model.mlmodel"), "rb") as f:
        # Try weights-first format
        pass
    # Just use coremltools
    model = ct.models.MLModel(path)
    return model.get_spec()


def dump_slice_update_ops(spec, function_name="main"):
    """Find and dump slice_update ops from the spec."""
    if spec.WhichOneof('Type') != 'mlProgram':
        print("  Not an mlProgram")
        return

    prog = spec.mlProgram
    fn = prog.functions.get(function_name, None)
    if fn is None:
        print(f"  Function '{function_name}' not found. Available: {list(prog.functions.keys())}")
        return

    # Get the block
    block = None
    for bname in fn.block_specializations:
        block = fn.block_specializations[bname]
        block_name = bname
        break

    if block is None:
        print("  No block found")
        return

    print(f"  Block: {block_name}, {len(block.operations)} operations")

    # Find all slice_update and coreml_update_state ops
    for i, op in enumerate(block.operations):
        op_type = op.type
        # We want: slice_update, coreml_update_state, or read_state
        if any(kw in op_type.lower() for kw in ['slice_update', 'scatter', 'update_state', 'read_state']):
            print(f"\n  [{i:4d}] {op_type}")
            for inp_name in op.inputs:
                inp_arg = op.inputs[inp_name]
                for binding in inp_arg.arguments:
                    if binding.HasField('name'):
                        print(f"         {inp_name}: var={binding.name}")
                    elif binding.HasField('value'):
                        val = binding.value
                        which = val.WhichOneof('value')
                        if which == 'immediateValue':
                            imm = val.immediateValue
                            imm_which = imm.WhichOneof('value')
                            if imm_which == 'tensor':
                                t = imm.tensor
                                # Check for int values
                                if t.HasField('ints'):
                                    vals = list(t.ints.values)
                                    print(f"         {inp_name}: const_ints={vals}")
                                elif t.HasField('bytes'):
                                    blen = len(t.bytes.values)
                                    print(f"         {inp_name}: const_bytes[{blen}]")
                                elif t.HasField('floats'):
                                    flen = len(t.floats.values)
                                    print(f"         {inp_name}: const_floats[{flen}]")
                                else:
                                    print(f"         {inp_name}: tensor(?)")
                            else:
                                getval = getattr(imm, imm_which, None)
                                print(f"         {inp_name}: {imm_which}={getval}")
                        elif which == 'blobFileValue':
                            blob = val.blobFileValue
                            print(f"         {inp_name}: blob(offset={blob.offset}, len={len(blob.fileName)})")
                        else:
                            print(f"         {inp_name}: {which}=...")
                    else:
                        print(f"         {inp_name}: ???")
            for out in op.outputs:
                tp = out.type
                if tp.HasField('tensorType'):
                    dims = list(tp.tensorType.dimensions)
                    print(f"         -> {out.name} shape={dims}")
                elif tp.HasField('stateType'):
                    wrapped = tp.stateType.wrappedType
                    if wrapped.HasField('tensorType'):
                        dims = list(wrapped.tensorType.dimensions)
                        print(f"         -> {out.name} [state] shape={dims}")
                    else:
                        print(f"         -> {out.name} [state]")
                else:
                    print(f"         -> {out.name} type={tp.WhichOneof('type')}")


def inspect_chunk(chunk_idx, function_name="main"):
    path = f"{MODEL_DIR}/prefill_LUT4_chunk{chunk_idx}.mlpackage"
    print(f"\n{'='*80}")
    print(f"Chunk {chunk_idx} (pre-combine): {path}")
    if not os.path.exists(path):
        print("  NOT FOUND")
        return
    model = ct.models.MLModel(path)
    spec = model.get_spec()
    dump_slice_update_ops(spec, function_name)


def inspect_combined_chunk(chunk_idx):
    path = f"{MODEL_DIR}/combined_LUT4_dedup/qwen3_5_FFN_PF_lut4_chunk_{chunk_idx+1:02d}of09.mlpackage"
    print(f"\n{'='*80}")
    print(f"Chunk {chunk_idx} (combined): {path}")
    if not os.path.exists(path):
        print("  NOT FOUND")
        return
    model = ct.models.MLModel(path)
    spec = model.get_spec()
    dump_slice_update_ops(spec, "prefill")


if __name__ == "__main__":
    # Compare: working chunks {1, 2, 5, 8} vs broken chunks {3, 4, 6, 7}
    print("=== Pre-combine prefill models ===")
    for idx in [1, 3]:  # 1=working, 3=broken
        inspect_chunk(idx)

    print("\n\n=== Combined models (prefill function) ===")
    for idx in [1, 3]:
        inspect_combined_chunk(idx)
