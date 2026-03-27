#!/usr/bin/env python3
"""Analyze MIL program ops to find FP16 vs FP32 differences and identify 
the sensitive ops (linear attention recurrence, RMSNorm) for selective precision.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import coremltools as ct

FP16_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"
FP32_DIR = "/tmp/qwen35_fp32_chunks"


def find_model(base, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = Path(base) / f"{name}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Missing {name} in {base}")


def analyze_mil_program(path, label):
    """Load MIL program and analyze operations."""
    print(f"\n  [{label}] Loading MIL: {Path(path).name}")
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    
    # Get the MIL program
    prog = model._mil_program
    
    op_types = Counter()
    dtype_by_op = defaultdict(Counter)
    cast_ops = []
    fp32_ops = []
    
    for func_name, func in prog.functions.items():
        for op in func.operations:
            op_types[op.op_type] += 1
            
            # Check output dtypes
            for out in op.outputs:
                dt = str(out.dtype) if hasattr(out, 'dtype') else '?'
                dtype_by_op[op.op_type][dt] += 1
                
                if 'float32' in dt.lower() or 'fp32' in dt.lower():
                    fp32_ops.append(f"{op.op_type}:{op.name}")
            
            # Track cast ops
            if op.op_type == 'cast':
                src_dt = str(op.inputs.get('x', {}).dtype) if hasattr(op.inputs.get('x', None), 'dtype') else '?'
                dst_dt = str(op.dtype.val) if hasattr(op, 'dtype') and hasattr(op.dtype, 'val') else '?'
                cast_ops.append(f"cast({src_dt}→{dst_dt})")
    
    print(f"  Total ops: {sum(op_types.values())}")
    print(f"\n  Op type distribution:")
    for op_type, count in sorted(op_types.items(), key=lambda x: -x[1])[:30]:
        dtypes = dict(dtype_by_op[op_type])
        print(f"    {op_type:<30} {count:>5}  dtypes={dtypes}")
    
    if cast_ops:
        cast_counts = Counter(cast_ops)
        print(f"\n  Cast operations:")
        for c, n in cast_counts.most_common():
            print(f"    {c}: {n}")
    
    if fp32_ops:
        print(f"\n  FP32 output ops ({len(fp32_ops)}):")
        for op_name in fp32_ops[:20]:
            print(f"    {op_name}")
        if len(fp32_ops) > 20:
            print(f"    ... and {len(fp32_ops)-20} more")
    
    del model
    return op_types, cast_ops


def compare_programs(fp16_path, fp32_path):
    """Compare FP16 vs FP32 MIL programs."""
    print("\n" + "=" * 70)
    print("  Comparing FP16 vs FP32 MIL programs")
    print("=" * 70)
    
    fp16_ops, fp16_casts = analyze_mil_program(fp16_path, "FP16")
    fp32_ops, fp32_casts = analyze_mil_program(fp32_path, "FP32")
    
    # Difference
    print("\n  Op count differences (FP32 - FP16):")
    all_ops = set(fp16_ops.keys()) | set(fp32_ops.keys())
    for op_type in sorted(all_ops):
        n16 = fp16_ops.get(op_type, 0)
        n32 = fp32_ops.get(op_type, 0)
        if n16 != n32:
            print(f"    {op_type:<30} FP16={n16:>5}  FP32={n32:>5}  diff={n32-n16:>+5}")
    
    print(f"\n  Cast ops FP16: {len(fp16_casts)}, FP32: {len(fp32_casts)}")
    print(f"  FP16 casts: {Counter(fp16_casts).most_common()}")
    print(f"  FP32 casts: {Counter(fp32_casts).most_common()}")


def find_sensitive_ops(path, label):
    """Identify the sensitive ops for recurrence and RMSNorm."""
    print(f"\n{'='*70}")
    print(f"  Identifying sensitive ops in [{label}]")
    print(f"{'='*70}")
    
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    prog = model._mil_program
    
    # Find ops related to: 
    # - RMSNorm/LayerNorm (reduce_mean, sub, layer_norm, mul with weight)
    # - Linear recurrence (matmul/conv in the recurrence path, elementwise ops)
    # - Cast ops (fp16→fp32 / fp32→fp16 boundaries)
    
    recurrence_keywords = ['recur', 'delta', 'conv_state', 'gate', 'norm', 'rms']
    
    for func_name, func in prog.functions.items():
        # Group by op name patterns
        patterns = defaultdict(list)
        for op in func.operations:
            name = op.name if hasattr(op, 'name') else ''
            for kw in recurrence_keywords:
                if kw in name.lower():
                    patterns[kw].append(f"{op.op_type}:{name}")
                    break
            
            # Also identify layer_norm and reduce_mean ops
            if op.op_type in ('layer_norm', 'instance_norm', 'reduce_mean', 'reduce_sum'):
                patterns['norm_ops'].append(f"{op.op_type}:{name}")
        
        for pattern, ops in sorted(patterns.items()):
            print(f"\n  [{pattern}] ({len(ops)} ops):")
            for o in ops[:10]:
                print(f"    {o}")
            if len(ops) > 10:
                print(f"    ... and {len(ops)-10} more")
    
    # Count total ops and estimate what fraction are recurrence/norm
    total = 0
    cast_count = 0
    for func_name, func in prog.functions.items():
        for op in func.operations:
            total += 1
            if op.op_type == 'cast':
                cast_count += 1
    
    print(f"\n  Summary: {total} total ops, {cast_count} cast ops")
    del model


def main():
    fp16_path = find_model(FP16_DIR, "ffn_LUT4_chunk0")
    fp32_path = find_model(FP32_DIR, "ffn_LUT4_chunk0")
    
    compare_programs(fp16_path, fp32_path)
    find_sensitive_ops(fp16_path, "FP16-chunk0")


if __name__ == "__main__":
    main()
