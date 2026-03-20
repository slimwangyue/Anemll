#!/usr/bin/env python3
"""Compare MIL ops between working teacher (Qwen3) and failing Qwen3.5 prefill.
Also check for dynamic slice_by_index and slice_update ops.
"""
import coremltools as ct
from collections import Counter

QWEN35_PATH = "/tmp/qwen35_ane_test/qwen35_prefill_chunk_01of04.mlpackage"
# Teacher Qwen3 prefill - use existing export
TEACHER_PATH = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export/qwen35_prefill_lut4_chunk_01of04.mlpackage"

def analyze_model(path, label):
    """Analyze MIL ops in a model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {label}")
    print(f"Path: {path}")
    print(f"{'='*60}")

    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    prog = spec.mlProgram

    op_counts = Counter()
    state_info = []
    slice_update_details = []
    slice_by_index_details = []

    for inp in spec.description.input:
        if inp.type.HasField("stateType"):
            shape = tuple(inp.type.stateType.multiArrayType.shape)
            size_bytes = 2
            for d in shape:
                size_bytes *= d
            state_info.append((inp.name, shape, size_bytes))

    for fn in prog.functions.values():
        for blk_name, blk in fn.block_specializations.items():
            for op in blk.operations:
                op_type = op.type
                op_counts[op_type] += 1

                # Check slice_update for dynamic indices
                if op_type == "slice_update":
                    inputs = {}
                    for k, v in op.inputs.items():
                        if v.arguments:
                            arg = v.arguments[0]
                            inputs[k] = arg.name
                    slice_update_details.append(inputs)

                # Sample some slice_by_index to check for dynamic patterns
                if op_type == "slice_by_index" and len(slice_by_index_details) < 20:
                    inputs = {}
                    for k, v in op.inputs.items():
                        if v.arguments:
                            arg = v.arguments[0]
                            inputs[k] = arg.name
                    slice_by_index_details.append(inputs)

    # Print op counts
    print(f"\nOp counts ({sum(op_counts.values())} total):")
    for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op_type:30s} {count:5d}")

    # Print state info
    print(f"\nStates ({len(state_info)}):")
    total_state_mb = 0
    for name, shape, size_bytes in state_info:
        mb = size_bytes / 1024 / 1024
        total_state_mb += mb
        print(f"  {name}: shape={shape} ~{mb:.1f}MB")
    print(f"  Total state: ~{total_state_mb:.1f}MB")

    # Print slice_update details
    if slice_update_details:
        print(f"\nslice_update details ({len(slice_update_details)} total):")
        for i, inputs in enumerate(slice_update_details[:10]):
            print(f"  [{i}] {inputs}")

    return op_counts, state_info

import os

counts_35 = None
counts_teacher = None

if os.path.exists(QWEN35_PATH):
    counts_35, states_35 = analyze_model(QWEN35_PATH, "Qwen3.5 prefill (ANE-fixed)")
else:
    print(f"SKIP: {QWEN35_PATH} not found")

if os.path.exists(TEACHER_PATH):
    counts_teacher, states_teacher = analyze_model(TEACHER_PATH, "Teacher (Qwen3.5 lut4 prefill)")
else:
    print(f"SKIP teacher: {TEACHER_PATH} not found")
    # Try alternative paths
    alt_paths = [
        "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export/qwen35_FFN_chunk_01of04.mlpackage",
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            counts_teacher, states_teacher = analyze_model(alt, f"Alt: {os.path.basename(alt)}")
            break

# Compare
if counts_35 and counts_teacher:
    print(f"\n{'='*60}")
    print("COMPARISON: Qwen3.5 vs Teacher")
    print(f"{'='*60}")
    all_ops = set(counts_35.keys()) | set(counts_teacher.keys())
    for op_type in sorted(all_ops):
        c35 = counts_35.get(op_type, 0)
        ct_ = counts_teacher.get(op_type, 0)
        if c35 != ct_:
            print(f"  {op_type:30s}  Q3.5={c35:5d}  Teacher={ct_:5d}  diff={c35-ct_:+d}")
