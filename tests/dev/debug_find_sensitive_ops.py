#!/usr/bin/env python3
"""Find norm and recurrence ops in FP16 MIL for selective precision targeting."""
import coremltools as ct

path = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3/ffn_LUT4_chunk0.mlpackage"
spec = ct.utils.load_spec(path)
prog = spec.mlProgram

keywords = ["norm", "rms", "recur", "delta", "conv_state", "gate"]
norm_types = {"layer_norm", "reduce_mean", "reduce_sum", "rsqrt"}

for fname, func in prog.functions.items():
    for bname, block in func.block_specializations.items():
        ops = list(block.operations)
        print(f"Function: {fname}, Block: {bname}, Total ops: {len(ops)}")
        print(f"\n--- Norm/Recurrence related ops ---")
        for i, op in enumerate(ops):
            out_name = op.outputs[0].name if op.outputs else ""
            name_lower = out_name.lower()
            show = False
            if any(kw in name_lower for kw in keywords):
                show = True
            if op.type in norm_types:
                show = True
            if show:
                print(f"  {i:>4} {op.type:<25} {out_name}")

        # Also print all layer_norm ops
        print(f"\n--- All layer_norm ops ---")
        for i, op in enumerate(ops):
            if op.type == "layer_norm":
                out_name = op.outputs[0].name if op.outputs else ""
                print(f"  {i:>4} layer_norm               {out_name}")

        # Print all conv ops (Conv2d for linear projections)
        print(f"\n--- First 30 conv ops ---")
        count = 0
        for i, op in enumerate(ops):
            if op.type == "conv":
                out_name = op.outputs[0].name if op.outputs else ""
                print(f"  {i:>4} conv                     {out_name}")
                count += 1
                if count >= 30:
                    break

        # Print ops around reduce_sum (used in recurrence)
        print(f"\n--- reduce_sum ops (recurrence matmul) ---")
        for i, op in enumerate(ops):
            if op.type == "reduce_sum":
                out_name = op.outputs[0].name if op.outputs else ""
                print(f"  {i:>4} reduce_sum               {out_name}")
