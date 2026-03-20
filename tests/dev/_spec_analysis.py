#!/usr/bin/env python3
"""Deep comparison between Qwen3.5 and a known ANE-working model.
Examine states, op types, and identify any blocker patterns.
"""
import coremltools as ct
from collections import Counter
import os

def full_spec_analysis(path, label):
    """Full analysis of a model's spec."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"Path: {path}")
    print(f"{'='*70}")

    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()

    # Check all fields in description
    desc = spec.description
    print(f"\nInputs ({len(desc.input)}):")
    for inp in desc.input:
        typ = inp.type
        if typ.HasField("multiArrayType"):
            print(f"  {inp.name}: multiArray shape={tuple(typ.multiArrayType.shape)} dtype={typ.multiArrayType.dataType}")
        elif typ.HasField("stateType"):
            shape = tuple(typ.stateType.multiArrayType.shape)
            print(f"  {inp.name}: STATE shape={shape} dtype={typ.stateType.multiArrayType.dataType}")
        else:
            print(f"  {inp.name}: {typ}")

    print(f"\nOutputs ({len(desc.output)}):")
    for out in desc.output:
        typ = out.type
        if typ.HasField("multiArrayType"):
            print(f"  {out.name}: multiArray shape={tuple(typ.multiArrayType.shape)} dtype={typ.multiArrayType.dataType}")
        else:
            print(f"  {out.name}: {typ}")

    # Check for state descriptions
    if hasattr(desc, 'state'):
        print(f"\nStates ({len(desc.state)}):")
        for st in desc.state:
            print(f"  {st}")
    else:
        print("\nNo 'state' field in description")

    # Check mlProgram functions
    prog = spec.mlProgram
    print(f"\nFunctions ({len(prog.functions)}):")
    for fn_name, fn in prog.functions.items():
        print(f"  {fn_name}:")
        print(f"    block_specializations: {list(fn.block_specializations.keys())}")
        if hasattr(fn, 'inputs'):
            for fn_inp in fn.inputs:
                print(f"    fn_input: {fn_inp.name} type={fn_inp.type}")

    # Op counts
    op_counts = Counter()
    for fn in prog.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                op_counts[op.type] += 1

    print(f"\nOp counts ({sum(op_counts.values())} total):")
    for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op_type:30s} {count:5d}")

    return op_counts

# Qwen3.5 decode chunk
q35_decode = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export/qwen35_FFN_chunk_01of04.mlpackage"
# Qwen3.5 prefill chunk
q35_prefill = "/tmp/qwen35_ane_test/qwen35_prefill_chunk_01of04.mlpackage"
# Qwen3.5 embeddings (should work on ANE - simpler model)
q35_embed = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export/qwen35_embeddings.mlpackage"

for path, label in [
    (q35_decode, "Qwen3.5 DECODE chunk 1"),
    (q35_embed, "Qwen3.5 EMBEDDINGS"),
]:
    if os.path.exists(path):
        full_spec_analysis(path, label)
    else:
        print(f"\nSKIP: {path}")
