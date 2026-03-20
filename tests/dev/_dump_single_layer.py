#!/usr/bin/env python3
"""Dump ALL ops from the single-layer model to find the ANE blocker."""
import coremltools as ct
import numpy as np

path = "/tmp/qwen35_ane_test/single_layer_0.mlpackage"
model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
spec = model.get_spec()
prog = spec.mlProgram

print("Full op list:")
for fn in prog.functions.values():
    for blk_name, blk in fn.block_specializations.items():
        print(f"\nBlock: {blk_name}")
        for i, op in enumerate(blk.operations):
            if op.type == "const":
                continue
            # Get input names
            inputs_str = []
            for k, v in op.inputs.items():
                for arg in v.arguments:
                    if hasattr(arg, 'name'):
                        inputs_str.append(f"{k}={arg.name}")
            # Get output names
            out_names = [o.name for o in op.outputs]
            print(f"  [{i:3d}] {op.type:20s} -> {out_names}  ({', '.join(inputs_str[:5])})")
