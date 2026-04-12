#!/usr/bin/env python3
"""Inspect MIL ops from an already-exported .mlpackage to understand precision-sensitive op types."""
import sys, os, argparse
import coremltools as ct


def inspect_model(path, func_name=None):
    print(f"Loading {path}...")
    
    # Load as MLModel without framework loading (just parse spec)
    import coremltools.proto.Model_pb2 as proto
    from coremltools.models.utils import load_spec
    spec = load_spec(path)
    
    if not spec.HasField("mlProgram"):
        print("Not an ML Program model")
        return
    
    program = spec.mlProgram
    
    for fn in program.functions:
        fn_name = fn.name if hasattr(fn, 'name') else str(fn)
        print(f"\nFunction: {fn_name}")
        
        # Get block
        block = None
        for bs_key in fn.block_specializations:
            block = fn.block_specializations[bs_key]
            print(f"  Block specialization: {bs_key}")
            break
        
        if block is None:
            print("  No block found")
            continue
        
        ops = list(block.operations)
        print(f"  Total ops: {len(ops)}")
        
        # Count op types
        type_count = {}
        for op in ops:
            t = op.type
            type_count[t] = type_count.get(t, 0) + 1
        
        print("\n  Op type counts:")
        for t, c in sorted(type_count.items(), key=lambda x: -x[1]):
            print(f"    {t:30s} {c:5d}")
        
        # Find precision-sensitive ops
        sensitive_types = {'softmax', 'exp', 'reduce_sum', 'reduce_mean', 'matmul', 'linear', 'cast', 'rsqrt', 'real_div'}
        print(f"\n  Precision-sensitive ops:")
        for op in ops:
            if op.type in sensitive_types:
                out_info = ""
                for o in op.outputs:
                    if o.type.HasField("tensorType"):
                        out_info = f"dtype={o.type.tensorType.dataType}"
                    break
                print(f"    {op.type:15s} | {out_info:20s} | {op.name[:100]}")
        
        # Count cast directions
        print("\n  CAST ops summary:")
        cast_ops = [op for op in ops if op.type == 'cast']
        print(f"    Total cast ops: {len(cast_ops)}")
        for op in cast_ops[:20]:
            dtype_val = ""
            for attr in op.attributes:
                if attr.name == 'dtype':
                    dtype_val = str(attr.value)[:40]
            print(f"    {op.name[:60]:60s} → {dtype_val}")
        if len(cast_ops) > 20:
            print(f"    ... and {len(cast_ops)-20} more")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .mlpackage")
    parser.add_argument("--function", default=None)
    args = parser.parse_args()
    inspect_model(args.model, args.function)


if __name__ == "__main__":
    main()
