#!/usr/bin/env python3
"""Dump MIL op names and types from the smallest FFN chunk to understand precision-sensitive ops."""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import torch
import numpy as np
import coremltools as ct

from config import BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, PER_CHANNEL, FFN_PER_CHANNEL, CHUNK_RANGES
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen__Qwen3.5-4B")
    parser.add_argument("--chunk", type=int, default=0, help="Chunk to analyze")
    args = parser.parse_args()

    print(f"Loading model...")
    cfg = Qwen35Config.from_json(os.path.join(args.model, "config.json"))
    model = Qwen35ForCausalLM(cfg)
    model.load_pretrained_weights(args.model)
    model.eval()

    ci = args.chunk
    sl, el = CHUNK_RANGES[ci]
    print(f"\nChunk {ci}: layers [{sl}-{el-1}]")

    # Export without compute_precision conversion (keep original dtypes)
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=FFN_PER_CHANNEL,
                           compute_precision="float32")  # float32 = no conversion = keep originals
    ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                             override_start_layer=sl, override_end_layer=el)

    spec = ml.get_spec()
    prog = ml._mil_program
    
    print(f"\n{'='*80}")
    print(f"MIL ops in chunk {ci} (layers {sl}-{el-1})")
    print(f"{'='*80}")
    
    # Get all functions
    for fn_name, fn in prog.functions.items():
        print(f"\nFunction: {fn_name}")
        print(f"  Ops: {len(fn.operations)}")
        
        # Group by op type
        op_types = {}
        for op in fn.operations:
            op_type = op.op_type
            if op_type not in op_types:
                op_types[op_type] = []
            op_types[op_type].append(op)
        
        print(f"\n  Op type summary:")
        for optype, ops in sorted(op_types.items()):
            print(f"    {optype}: {len(ops)}")
        
        # Find precision-sensitive ops
        print(f"\n  Precision-sensitive ops (softmax, exp, matmul, reduce):")
        for op in fn.operations:
            if any(kw in op.op_type.lower() for kw in ['softmax', 'exp', 'reduce', 'matmul', 'linear']):
                # Get the dtype of the output
                out_dtype = "?"
                for out in op.outputs:
                    out_dtype = str(out.dtype) if hasattr(out, 'dtype') else "?"
                    break
                # Check if any input has layer attribution
                name_str = op.name if op.name else "unnamed"
                print(f"    {op.op_type:20s} | {out_dtype:8s} | {name_str[:80]}")
        
        # Find cast ops (indicate precision transitions)
        print(f"\n  CAST operations (precision transitions):")
        cast_count = 0
        for op in fn.operations:
            if op.op_type == 'cast':
                name_str = op.name if op.name else "unnamed"
                in_dtype = "?"
                out_dtype = "?"
                for inp_name, inp_val in op.inputs.items():
                    if inp_name == 'dtype':
                        out_dtype = str(inp_val.val) if hasattr(inp_val, 'val') else str(inp_val)
                    elif inp_name == 'x':
                        in_dtype = str(inp_val.dtype) if hasattr(inp_val, 'dtype') else "?"
                cast_count += 1
                if cast_count <= 30:
                    print(f"    {in_dtype:8s} -> {out_dtype:8s} | {name_str[:80]}")
        if cast_count > 30:
            print(f"    ... and {cast_count - 30} more cast ops")
        print(f"  Total cast ops: {cast_count}")


if __name__ == "__main__":
    main()
