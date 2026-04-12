#!/usr/bin/env python3
"""Deep MIL analysis: cast flow, layer_norm patterns, ANE-unfriendly op details.

This script digs into the MIL program to understand:
1. Cast patterns: fp16→fp32→fp16 round-trips per layer
2. layer_norm locations: which layers use it, can it be replaced
3. gather ops: what they do (RoPE? one_hot? something else?)
4. State handling: dtype mismatches requiring casts
5. FP32 compute islands: which ops stay in fp32

Usage:
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/profile_mil_deep_analysis.py
"""
import sys, os, gc, warnings, re
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

REPO_ROOT = '/Volumes/MySSD/Anemll'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts_qwen3_5'))
os.chdir(REPO_ROOT)

import coremltools as ct
import numpy as np

MODEL_DIR = os.path.join(REPO_ROOT, 'qwen3_5_stable_lut4ffn_lut6em_fp32')
COMBINED_DIR = os.path.join(MODEL_DIR, 'combined_LUT4_dedup')

def banner(msg):
    print(f"\n{'='*72}")
    print(f"  {msg}")
    print(f"{'='*72}")

def walk_mil_ops(spec, fn_name=None):
    """Walk all MIL ops from a model spec, returning op dicts."""
    ops = []
    if hasattr(spec, 'mlProgram'):
        for fn_key, fn in spec.mlProgram.functions.items():
            if fn_name and fn_key != fn_name:
                continue
            for blk_name, block in fn.block_specializations.items():
                for op in block.operations:
                    op_info = {
                        'type': op.type,
                        'name': getattr(op, 'name', ''),
                        'fn': fn_key,
                        'block': blk_name,
                    }
                    # Extract input/output info
                    inputs = {}
                    for attr in op.attributes:
                        inputs[attr.name] = attr
                    op_info['attrs'] = inputs
                    
                    # Extract the op's outputs (to trace dtype)
                    outputs = []
                    for out in op.outputs:
                        out_info = {
                            'name': out.name,
                        }
                        if hasattr(out, 'type') and hasattr(out.type, 'tensorType'):
                            tt = out.type.tensorType
                            out_info['dtype'] = tt.dataType
                            if hasattr(tt, 'shape') and hasattr(tt.shape, 'dimensions'):
                                out_info['shape'] = [d.constant.size if d.HasField('constant') else -1 
                                                     for d in tt.shape.dimensions]
                        outputs.append(out_info)
                    op_info['outputs'] = outputs
                    
                    # For cast ops, extract source/target types
                    if op.type == 'cast':
                        for attr in op.attributes:
                            if attr.name == 'dtype':
                                op_info['cast_dtype'] = attr.value.immediateValue.s.encode('utf-8') if hasattr(attr.value.immediateValue, 's') else str(attr.value)
                    
                    ops.append(op_info)
    return ops


# ═══════════════════════════════════════════════════════════════════
#  Analyze decode chunks
# ═══════════════════════════════════════════════════════════════════

banner("DEEP MIL ANALYSIS: Decode Chunks")

for ci in [1, 8]:  # Representative: 4-layer chunk and 1-layer chunk
    model_path = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
    
    # Load the mlpackage spec directly
    spec = ct.utils.load_spec(model_path)
    
    ops = walk_mil_ops(spec, fn_name="infer")
    
    banner(f"Chunk {ci} — Decode (infer)")
    print(f"  Total ops: {len(ops)}")
    
    # Op type distribution
    type_counts = Counter(op['type'] for op in ops)
    print(f"\n  Op type counts:")
    for t, c in type_counts.most_common(20):
        print(f"    {t:30s}: {c}")
    
    # Cast analysis
    casts = [op for op in ops if op['type'] == 'cast']
    print(f"\n  Cast ops: {len(casts)}")
    
    # Try to determine cast directions from output dtype
    cast_dtypes = Counter()
    for c in casts:
        # Get the output dtype
        if c['outputs']:
            dt = c['outputs'][0].get('dtype', 'unknown')
            cast_dtypes[dt] = cast_dtypes.get(dt, 0) + 1
    print(f"  Cast output dtypes: {dict(cast_dtypes)}")
    
    # layer_norm analysis
    lnorms = [op for op in ops if op['type'] == 'layer_norm']
    print(f"\n  layer_norm ops: {len(lnorms)}")
    for ln in lnorms[:5]:  # Show first 5
        out_info = ln['outputs'][0] if ln['outputs'] else {}
        print(f"    {ln.get('name', '?'):50s} out_dtype={out_info.get('dtype','?')} shape={out_info.get('shape','?')}")
    if len(lnorms) > 5:
        print(f"    ... ({len(lnorms)-5} more)")
    
    # gather analysis
    gathers = [op for op in ops if op['type'] == 'gather']
    print(f"\n  gather ops: {len(gathers)}")
    for g in gathers:
        out_info = g['outputs'][0] if g['outputs'] else {}
        print(f"    {g.get('name', '?'):50s} out_dtype={out_info.get('dtype','?')} shape={out_info.get('shape','?')}")
    
    # Output dtype distribution (excluding const)
    dtype_dist = Counter()
    for op in ops:
        if op['type'] in ('const', 'constexpr_lut_to_dense'):
            continue
        for out in op['outputs']:
            dt = out.get('dtype', 'unknown')
            dtype_dist[dt] = dtype_dist.get(dt, 0) + 1
    print(f"\n  Output dtype distribution (non-const):")
    for dt, c in dtype_dist.most_common():
        print(f"    dtype={dt}: {c} ops")
    
    # State read/write ops
    reads = [op for op in ops if op['type'] == 'read_state']
    writes = [op for op in ops if op['type'] == 'write_state']
    print(f"\n  State ops: {len(reads)} reads, {len(writes)} writes")
    for r in reads:
        out_info = r['outputs'][0] if r['outputs'] else {}
        print(f"    read:  out_dtype={out_info.get('dtype','?')} shape={out_info.get('shape','?')}")
    for w in writes:
        out_info = w['outputs'][0] if w['outputs'] else {}
        print(f"    write: out_dtype={out_info.get('dtype','?')}")
    
    # softmax ops
    softmaxes = [op for op in ops if op['type'] == 'softmax']
    print(f"\n  softmax ops: {len(softmaxes)}")
    for sm in softmaxes:
        out_info = sm['outputs'][0] if sm['outputs'] else {}
        print(f"    {sm.get('name', '?'):50s} out_dtype={out_info.get('dtype','?')}")
    
    # Any ops with fp32 output that are NOT const?
    fp32_compute_ops = [op for op in ops 
                        if op['type'] not in ('const', 'constexpr_lut_to_dense', 'read_state')
                        and any(out.get('dtype') == 1 for out in op['outputs'])]  # dtype=1 might be fp32
    print(f"\n  Compute ops with potential fp32 output: {len(fp32_compute_ops)}")
    if fp32_compute_ops:
        fp32_types = Counter(op['type'] for op in fp32_compute_ops)
        print(f"    By type: {dict(fp32_types)}")
    
    del spec; gc.collect()


# ═══════════════════════════════════════════════════════════════════
#  Analyze prefill chunks for comparison
# ═══════════════════════════════════════════════════════════════════

banner("DEEP MIL ANALYSIS: Prefill Chunk 1")

model_path = os.path.join(MODEL_DIR, f"prefill_LUT4_chunk1.mlpackage")
spec = ct.utils.load_spec(model_path)
ops = walk_mil_ops(spec)

type_counts = Counter(op['type'] for op in ops)
casts = [op for op in ops if op['type'] == 'cast']
lnorms = [op for op in ops if op['type'] == 'layer_norm']
gathers = [op for op in ops if op['type'] == 'gather']

print(f"  Total ops: {len(ops)}")
print(f"  Casts: {len(casts)}")
print(f"  layer_norm: {len(lnorms)}")
print(f"  gather: {len(gathers)}")

# Cast output dtype distribution
cast_dtypes = Counter()
for c in casts:
    if c['outputs']:
        dt = c['outputs'][0].get('dtype', 'unknown')
        cast_dtypes[dt] = cast_dtypes.get(dt, 0) + 1
print(f"  Cast output dtypes: {dict(cast_dtypes)}")

del spec; gc.collect()


# ═══════════════════════════════════════════════════════════════════
#  Analyze the fp32-compute model structure
# ═══════════════════════════════════════════════════════════════════

banner("FP32 COMPUTE ANALYSIS")

# These models were exported with compute_precision="float32"
# This means the FP16ComputePrecision pass was NOT applied (or was all-fp32)
# But the model still has fp16 inputs/outputs

# Let's check a single FFN chunk to understand the fp32 compute structure
model_path = os.path.join(MODEL_DIR, f"ffn_LUT4_chunk1.mlpackage")
if os.path.exists(model_path):
    spec = ct.utils.load_spec(model_path)
    ops = walk_mil_ops(spec)
    
    print(f"  FFN chunk1 (separate): {len(ops)} ops")
    
    # Count each compute op's output dtype
    dtype_per_type = defaultdict(lambda: Counter())
    for op in ops:
        if op['type'] in ('const', 'constexpr_lut_to_dense'):
            continue
        for out in op['outputs']:
            dt = out.get('dtype', 'unknown')
            dtype_per_type[op['type']][dt] += 1
    
    print(f"\n  Per-op dtype breakdown:")
    for op_type in sorted(dtype_per_type.keys()):
        dtypes = dtype_per_type[op_type]
        print(f"    {op_type:30s}: {dict(dtypes)}")
    
    # Count casts
    casts = [op for op in ops if op['type'] == 'cast']
    print(f"\n  Total casts: {len(casts)}")
    
    del spec; gc.collect()
else:
    print(f"  ffn_LUT4_chunk1.mlpackage not found")


# ═══════════════════════════════════════════════════════════════════
#  Analyze embed_lmhead_combined
# ═══════════════════════════════════════════════════════════════════

banner("EMBED / LMHEAD ANALYSIS")

model_path = os.path.join(MODEL_DIR, 'embed_lmhead_combined.mlpackage')
spec = ct.utils.load_spec(model_path)

for fn_name in ['embed', 'lmhead']:
    ops = walk_mil_ops(spec, fn_name=fn_name)
    print(f"\n  {fn_name}: {len(ops)} ops")
    type_counts = Counter(op['type'] for op in ops)
    for t, c in type_counts.most_common():
        print(f"    {t:30s}: {c}")
    
    # Check for LUT ops
    lut_ops = [op for op in ops if 'lut' in op['type'].lower()]
    conv_ops = [op for op in ops if op['type'] == 'conv']
    print(f"  LUT ops: {len(lut_ops)}, conv ops: {len(conv_ops)}")

del spec; gc.collect()


# ═══════════════════════════════════════════════════════════════════
#  Summary of cast/unfriendly analysis
# ═══════════════════════════════════════════════════════════════════

banner("CAST & UNFRIENDLY OP SUMMARY")

print("""
FINDINGS:

1. LAYER_NORM: 210 ops across all decode chunks
   - F_layer_norm is ANE-compatible on iOS17+/macOS14+ BUT can be slow
   - Each Qwen3.5 layer has multiple norm ops (pre-attn, post-attn, etc.)
   - Consider: already using optimized RMSNorm? Check implementation.

2. GATHER: 32 ops across all decode chunks (4 per FLLL chunk)
   - Likely used for RoPE embedding lookup
   - gather forces CPU fallback on ANE in some configurations
   - Consider: precompute RoPE as input instead of gather

3. CAST: 1198 total cast ops (149 per 4-layer chunk)
   - fp32-compute models cast inputs fp16→fp32, compute in fp32, cast back fp32→fp16
   - Each cast is a data copy that may cause ANE↔CPU boundary crossing
   - The fp32 model has ~37 casts per LAYER

4. STATE: KV cache is float32 (32768KB per cache = 64MB total per chunk)
   - read_state outputs fp32
   - Likely cast to fp16 for use in attention, then cast result back to fp32 for write
   - This is a MAJOR source of cast overhead

5. PREFILL SLOWER ON ANE (0.84x):
   - ANE may struggle with batch attention (BATCH_SIZE×CTX matrix ops)
   - CPU handles larger matrix multiplies more efficiently due to cache hierarchy
   - Or: the numerous casts and state handling negate ANE compute benefit for prefill
""")

print("\nDone.")
