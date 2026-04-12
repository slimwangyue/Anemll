#!/usr/bin/env python3
"""Profile ANE vs CPU utilization for Qwen3.5-4B CoreML models.

Measures:
  1) Per-chunk decode latency (ANE vs CPU_ONLY) → ANE speedup ratio
  2) Per-chunk prefill latency (ANE vs CPU_ONLY)
  3) Embed/LMHead latency
  4) MIL op analysis: cast counts, ANE-unfriendly ops per chunk
  5) State read/write overhead
  6) Overall decode tok/s and prefill tok/s

Usage:
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/profile_ane_cpu_utilization.py
"""
import sys, os, time, gc, json, warnings
warnings.filterwarnings('ignore')

REPO_ROOT = '/Volumes/MySSD/Anemll'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts_qwen3_5'))
os.chdir(REPO_ROOT)

import numpy as np
import coremltools as ct

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

MODEL_DIR = os.path.join(REPO_ROOT, 'qwen3_5_stable_lut4ffn_lut6em_fp32')
COMBINED_DIR = os.path.join(MODEL_DIR, 'combined_LUT4_dedup')
TOKENIZER_DIR = MODEL_DIR

NUM_WARMUP = 3
NUM_MEASURE = 20       # decode iterations for timing
NUM_PREFILL_MEASURE = 5 # prefill iterations

# ── Helpers ──────────────────────────────────────────────────────

def banner(msg):
    print(f"\n{'='*72}")
    print(f"  {msg}")
    print(f"{'='*72}")

def fmt_ms(ms):
    if ms < 1:
        return f"{ms*1000:.1f}μs"
    return f"{ms:.2f}ms"

def measure_latency(fn, warmup=NUM_WARMUP, repeats=NUM_MEASURE):
    """Run fn() warmup+repeats times, return (mean_ms, std_ms, all_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), times


# ═══════════════════════════════════════════════════════════════════
#  PHASE 1: Per-chunk decode latency — ANE vs CPU_ONLY
# ═══════════════════════════════════════════════════════════════════

banner("PHASE 1: Per-Chunk Decode Latency (ANE vs CPU_ONLY)")

# Load embed + lmhead (always CPU_ONLY)
combined_el = os.path.join(MODEL_DIR, 'embed_lmhead_combined.mlpackage')
print("Loading embed (CPU_ONLY)...")
embed = ct.models.MLModel(combined_el, compute_units=ct.ComputeUnit.CPU_ONLY,
                          function_name="embed")
print("Loading lmhead (CPU_ONLY)...")
lmhead = ct.models.MLModel(combined_el, compute_units=ct.ComputeUnit.CPU_ONLY,
                           function_name="lmhead")

# Prepare dummy inputs
tok_input = np.array([[1]], dtype=np.int32)
hidden_single = np.zeros((1, 1, 2560), dtype=np.float16)

# Measure embed latency
print("\nMeasuring embed latency...")
def run_embed():
    return embed.predict({"input_ids": tok_input})
embed_mean, embed_std, _ = measure_latency(run_embed)
print(f"  Embed: {fmt_ms(embed_mean)} ± {fmt_ms(embed_std)}")

# Measure lmhead latency
print("Measuring lmhead latency...")
def run_lmhead():
    return lmhead.predict({"hidden_states": hidden_single})
lmhead_mean, lmhead_std, _ = measure_latency(run_lmhead)
print(f"  LMHead: {fmt_ms(lmhead_mean)} ± {fmt_ms(lmhead_std)}")

# Get input shapes per chunk
def get_chunk_input_shapes(model_path, fn_name="infer"):
    """Extract input shapes from a multi-function mlpackage."""
    spec = ct.utils.load_spec(model_path)
    shapes = {}
    for fn in spec.description.functions:
        if fn.name == fn_name:
            for inp in fn.input:
                try:
                    shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                except:
                    pass
            break
    return shapes

# Per-chunk timing for both compute units
chunk_results = {}

for cu_name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)]:
    banner(f"Decode timing: {cu_name}")
    for ci in range(NUM_CHUNKS):
        model_path = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
        inp_shapes = get_chunk_input_shapes(model_path)
        
        print(f"  Loading chunk {ci} ({cu_name})...", end=" ", flush=True)
        t0 = time.time()
        m = ct.models.MLModel(model_path, compute_units=cu, function_name="infer")
        print(f"{time.time()-t0:.1f}s")
        
        state = m.make_state()
        
        # Build inputs from shapes
        has_linear = 'linear_conv_state' in inp_shapes
        
        # Position in middle of context so KV cache has some entries
        pos = 50
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos+1] = 0
        
        inp = {
            "hidden_states": hidden_single.copy(),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
        }
        if has_linear:
            inp["linear_conv_state"] = np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16)
            inp["linear_recurrent_state"] = np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16)
        
        def run_chunk(model=m, inp=inp, state=state):
            return model.predict(inp, state=state)
        
        mean_ms, std_ms, all_ms = measure_latency(run_chunk)
        
        key = (ci, cu_name)
        chunk_results[key] = {
            'mean_ms': mean_ms, 'std_ms': std_ms,
            'layers': f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}",
            'n_layers': CHUNK_RANGES[ci][1] - CHUNK_RANGES[ci][0],
            'has_linear': has_linear,
        }
        
        print(f"  Chunk {ci} [{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}]: "
              f"{fmt_ms(mean_ms)} ± {fmt_ms(std_ms)} "
              f"({CHUNK_RANGES[ci][1]-CHUNK_RANGES[ci][0]} layers, "
              f"{'linear+full' if has_linear else 'full-only'})")
        
        del m, state; gc.collect()

# ═══════════════════════════════════════════════════════════════════
#  PHASE 1b: Decode speedup table
# ═══════════════════════════════════════════════════════════════════

banner("DECODE SPEEDUP TABLE: ANE vs CPU_ONLY")
print(f"  {'Chunk':>6s}  {'Layers':>8s}  {'CPU ms':>10s}  {'ANE ms':>10s}  {'Speedup':>8s}  {'ANE%':>6s}  Notes")
print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*20}")

total_cpu = 0
total_ane = 0
for ci in range(NUM_CHUNKS):
    cpu = chunk_results[(ci, "CPU_ONLY")]
    ane = chunk_results[(ci, "CPU_AND_NE")]
    speedup = cpu['mean_ms'] / ane['mean_ms'] if ane['mean_ms'] > 0 else 0
    # ANE utilization heuristic: if speedup > 1, ANE is being used
    # Higher speedup = more ANE utilization
    ane_pct = max(0, (1 - ane['mean_ms'] / cpu['mean_ms'])) * 100 if cpu['mean_ms'] > 0 else 0
    total_cpu += cpu['mean_ms']
    total_ane += ane['mean_ms']
    notes = ""
    if speedup < 1.5:
        notes = "⚠ LOW ANE BENEFIT"
    elif speedup < 2.0:
        notes = "~ moderate"
    else:
        notes = "✓ good ANE use"
    print(f"  {ci:>6d}  {cpu['layers']:>8s}  {cpu['mean_ms']:>8.2f}ms  {ane['mean_ms']:>8.2f}ms  {speedup:>7.2f}x  {ane_pct:>5.1f}%  {notes}")

total_speedup = total_cpu / total_ane if total_ane > 0 else 0
total_ane_pct = max(0, (1 - total_ane / total_cpu)) * 100 if total_cpu > 0 else 0
print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}")
print(f"  {'TOTAL':>6s}  {'':>8s}  {total_cpu:>8.2f}ms  {total_ane:>8.2f}ms  {total_speedup:>7.2f}x  {total_ane_pct:>5.1f}%")

# End-to-end decode latency
e2e_cpu = total_cpu + embed_mean + lmhead_mean
e2e_ane = total_ane + embed_mean + lmhead_mean
print(f"\n  End-to-end decode (chunks + embed + lmhead):")
print(f"    CPU_ONLY: {fmt_ms(e2e_cpu)} → {1000/e2e_cpu:.1f} tok/s")
print(f"    ANE:      {fmt_ms(e2e_ane)} → {1000/e2e_ane:.1f} tok/s")
print(f"    Embed overhead: {embed_mean/e2e_ane*100:.1f}% of total")
print(f"    LMHead overhead: {lmhead_mean/e2e_ane*100:.1f}% of total")


# ═══════════════════════════════════════════════════════════════════
#  PHASE 2: Prefill latency per chunk
# ═══════════════════════════════════════════════════════════════════

banner("PHASE 2: Per-Chunk Prefill Latency")

# Load embed_prefill
embed_prefill_path = os.path.join(MODEL_DIR, 'embed_prefill.mlpackage')
print("Loading embed_prefill (CPU_ONLY)...")
embed_prefill = ct.models.MLModel(embed_prefill_path, compute_units=ct.ComputeUnit.CPU_ONLY)

prefill_input = np.zeros((1, BATCH_SIZE), dtype=np.int32)
prefill_input[0, :10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print("Measuring embed_prefill latency...")
def run_embed_prefill():
    return embed_prefill.predict({"input_ids": prefill_input})
ep_mean, ep_std, _ = measure_latency(run_embed_prefill, repeats=NUM_PREFILL_MEASURE)
print(f"  embed_prefill (batch={BATCH_SIZE}): {fmt_ms(ep_mean)} ± {fmt_ms(ep_std)}")

hidden_prefill = np.zeros((1, BATCH_SIZE, 2560), dtype=np.float16)
mask_prefill = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
for i in range(BATCH_SIZE):
    mask_prefill[0, 0, i, :i+1] = 0
pos_ids_prefill = np.arange(BATCH_SIZE, dtype=np.int32)

prefill_results = {}

for cu_name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)]:
    banner(f"Prefill timing: {cu_name}")
    for ci in range(NUM_CHUNKS):
        model_path = os.path.join(MODEL_DIR, f"prefill_LUT4_chunk{ci}.mlpackage")
        if not os.path.exists(model_path):
            print(f"  Prefill chunk {ci}: NOT FOUND, skipping")
            continue
        
        inp_shapes = {}
        spec = ct.utils.load_spec(model_path)
        for inp_desc in spec.description.input:
            try:
                inp_shapes[inp_desc.name] = tuple(inp_desc.type.multiArrayType.shape)
            except:
                pass
        
        has_linear = 'linear_conv_state' in inp_shapes
        
        print(f"  Loading prefill chunk {ci} ({cu_name})...", end=" ", flush=True)
        t0 = time.time()
        m = ct.models.MLModel(model_path, compute_units=cu)
        state = m.make_state()
        print(f"{time.time()-t0:.1f}s")
        
        inp = {
            "hidden_states": hidden_prefill.copy(),
            "position_ids": pos_ids_prefill,
            "causal_mask": mask_prefill,
            "current_pos": np.array([0], dtype=np.int32),
        }
        if has_linear:
            inp["linear_conv_state"] = np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16)
            inp["linear_recurrent_state"] = np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16)
        if 'valid_len' in inp_shapes:
            inp["valid_len"] = np.array([BATCH_SIZE], dtype=np.int32)
        
        def run_prefill(model=m, inp=inp, state=state):
            return model.predict(inp, state=state)
        
        mean_ms, std_ms, _ = measure_latency(run_prefill, warmup=2, repeats=NUM_PREFILL_MEASURE)
        
        prefill_results[(ci, cu_name)] = {
            'mean_ms': mean_ms, 'std_ms': std_ms,
            'layers': f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}",
        }
        
        print(f"  Prefill chunk {ci} [{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}]: "
              f"{fmt_ms(mean_ms)} ± {fmt_ms(std_ms)}")
        
        del m, state; gc.collect()

# Prefill speedup table
if prefill_results:
    banner("PREFILL SPEEDUP TABLE: ANE vs CPU_ONLY")
    print(f"  {'Chunk':>6s}  {'Layers':>8s}  {'CPU ms':>10s}  {'ANE ms':>10s}  {'Speedup':>8s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")
    ptotal_cpu = 0
    ptotal_ane = 0
    for ci in range(NUM_CHUNKS):
        k_cpu = (ci, "CPU_ONLY")
        k_ane = (ci, "CPU_AND_NE")
        if k_cpu not in prefill_results or k_ane not in prefill_results:
            continue
        cpu = prefill_results[k_cpu]
        ane = prefill_results[k_ane]
        speedup = cpu['mean_ms'] / ane['mean_ms'] if ane['mean_ms'] > 0 else 0
        ptotal_cpu += cpu['mean_ms']
        ptotal_ane += ane['mean_ms']
        print(f"  {ci:>6d}  {cpu['layers']:>8s}  {cpu['mean_ms']:>8.1f}ms  {ane['mean_ms']:>8.1f}ms  {speedup:>7.2f}x")
    if ptotal_ane > 0:
        print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")
        speedup = ptotal_cpu / ptotal_ane
        print(f"  {'TOTAL':>6s}  {'':>8s}  {ptotal_cpu:>8.1f}ms  {ptotal_ane:>8.1f}ms  {speedup:>7.2f}x")
        print(f"\n  Prefill throughput (batch={BATCH_SIZE}):")
        print(f"    CPU_ONLY: {BATCH_SIZE/(ptotal_cpu+ep_mean)*1000:.0f} tok/s")
        print(f"    ANE:      {BATCH_SIZE/(ptotal_ane+ep_mean)*1000:.0f} tok/s")


# ═══════════════════════════════════════════════════════════════════
#  PHASE 3: MIL Op Analysis — Cast counts & ANE-unfriendly ops
# ═══════════════════════════════════════════════════════════════════

banner("PHASE 3: MIL Op Analysis")

# Known ANE-unfriendly op types
ANE_UNFRIENDLY = {
    'gather', 'scatter', 'scatter_nd', 'topk', 'argsort', 'sort',
    'non_zero', 'unique', 'where',  # conditional ops
    'cumsum',  # sequential dependency
    'while_loop', 'cond',  # control flow
    'layer_norm',  # sometimes falls back
}

def analyze_mil_ops(model_path, fn_name=None):
    """Analyze MIL program ops for a CoreML model."""
    try:
        if fn_name:
            m = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY,
                                  function_name=fn_name)
        else:
            m = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = m.get_spec()
        
        # Count ops from the spec's mlprogram
        op_counts = {}
        cast_count = 0
        cast_details = []
        unfriendly_ops = []
        total_ops = 0
        
        # Walk the MIL program ops
        if hasattr(spec, 'mlProgram'):
            for fn in spec.mlProgram.functions.values():
                for blk_name, block in fn.block_specializations.items():
                    for op in block.operations:
                        total_ops += 1
                        op_type = op.type
                        op_counts[op_type] = op_counts.get(op_type, 0) + 1
                        if op_type == 'cast':
                            cast_count += 1
                            # Try to extract dtype info
                            cast_details.append(op.type)
                        if op_type in ANE_UNFRIENDLY:
                            unfriendly_ops.append(op_type)
        
        del m; gc.collect()
        return {
            'total_ops': total_ops,
            'op_counts': op_counts,
            'cast_count': cast_count,
            'unfriendly_ops': unfriendly_ops,
            'unfriendly_count': len(unfriendly_ops),
        }
    except Exception as e:
        return {'error': str(e)}

print("Analyzing decode chunks (combined_LUT4_dedup)...")
decode_analysis = {}
for ci in range(NUM_CHUNKS):
    model_path = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
    result = analyze_mil_ops(model_path, fn_name="infer")
    decode_analysis[ci] = result
    if 'error' not in result:
        print(f"  Chunk {ci}: {result['total_ops']} ops, "
              f"{result['cast_count']} casts, "
              f"{result['unfriendly_count']} ANE-unfriendly")
        if result['unfriendly_ops']:
            from collections import Counter
            uf = Counter(result['unfriendly_ops'])
            print(f"    Unfriendly: {dict(uf)}")
    else:
        print(f"  Chunk {ci}: ERROR — {result['error']}")

print("\nAnalyzing prefill chunks...")
prefill_analysis = {}
for ci in range(NUM_CHUNKS):
    model_path = os.path.join(MODEL_DIR, f"prefill_LUT4_chunk{ci}.mlpackage")
    if not os.path.exists(model_path):
        continue
    result = analyze_mil_ops(model_path)
    prefill_analysis[ci] = result
    if 'error' not in result:
        print(f"  Prefill chunk {ci}: {result['total_ops']} ops, "
              f"{result['cast_count']} casts, "
              f"{result['unfriendly_count']} ANE-unfriendly")
    else:
        print(f"  Prefill chunk {ci}: ERROR — {result['error']}")

# Analyze embed and lmhead
print("\nAnalyzing embed/lmhead...")
for name, path, fn_name in [
    ("embed", combined_el, "embed"),
    ("lmhead", combined_el, "lmhead"),
    ("embed_prefill", embed_prefill_path, None),
]:
    result = analyze_mil_ops(path, fn_name=fn_name)
    if 'error' not in result:
        print(f"  {name}: {result['total_ops']} ops, "
              f"{result['cast_count']} casts")
    else:
        print(f"  {name}: ERROR — {result['error']}")


# Op-type distribution summary
banner("OP TYPE DISTRIBUTION (Decode chunks)")
all_op_counts = {}
for ci in range(NUM_CHUNKS):
    if 'error' in decode_analysis.get(ci, {}):
        continue
    for op, cnt in decode_analysis[ci].get('op_counts', {}).items():
        all_op_counts[op] = all_op_counts.get(op, 0) + cnt

print(f"  {'Op Type':30s}  {'Count':>8s}  {'%':>6s}")
print(f"  {'─'*30}  {'─'*8}  {'─'*6}")
total_all = sum(all_op_counts.values())
for op, cnt in sorted(all_op_counts.items(), key=lambda x: -x[1]):
    pct = cnt / total_all * 100 if total_all > 0 else 0
    flag = " ⚠" if op in ANE_UNFRIENDLY else ""
    if op == 'cast':
        flag = " ◄ CAST"
    print(f"  {op:30s}  {cnt:>8d}  {pct:>5.1f}%{flag}")


# ═══════════════════════════════════════════════════════════════════
#  PHASE 4: State Handling Overhead
# ═══════════════════════════════════════════════════════════════════

banner("PHASE 4: State Read/Write Overhead")

# Load one chunk to measure state access time
model_path = os.path.join(COMBINED_DIR, "chunk1.mlpackage")
m = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE, function_name="infer")
state = m.make_state()

# First, do a predict to populate state
inp_shapes = get_chunk_input_shapes(model_path)
has_linear = 'linear_conv_state' in inp_shapes
pos = 10
mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
mask[:,:,:,:pos+1] = 0
inp = {
    "hidden_states": hidden_single.copy(),
    "position_ids": np.array([pos], dtype=np.int32),
    "causal_mask": mask,
    "current_pos": np.array([pos], dtype=np.int32),
}
if has_linear:
    inp["linear_conv_state"] = np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16)
    inp["linear_recurrent_state"] = np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16)
m.predict(inp, state=state)

# Get list of state names
# We'll look at the spec for state info
spec = m.get_spec()
state_names = []
for fn in spec.description.functions:
    if fn.name == "infer":
        for st in fn.state:
            state_names.append(st.name)
        break

print(f"  State variables: {len(state_names)}")
for sn in state_names:
    try:
        val = state.read_state(name=sn)
        print(f"    {sn}: shape={val.shape}, dtype={val.dtype}, size={val.nbytes/1024:.0f}KB")
    except Exception as e:
        print(f"    {sn}: read error — {e}")

# Measure read_state latency
if state_names:
    def read_all_states():
        for sn in state_names:
            state.read_state(name=sn)
    
    r_mean, r_std, _ = measure_latency(read_all_states, warmup=3, repeats=20)
    print(f"\n  Read all states: {fmt_ms(r_mean)} ± {fmt_ms(r_std)}")
    
    # Measure write_state latency
    state_data = {sn: state.read_state(name=sn) for sn in state_names}
    def write_all_states():
        for sn in state_names:
            state.write_state(name=sn, value=state_data[sn])
    
    w_mean, w_std, _ = measure_latency(write_all_states, warmup=3, repeats=20)
    print(f"  Write all states: {fmt_ms(w_mean)} ± {fmt_ms(w_std)}")
    
    total_state_bytes = sum(state_data[sn].nbytes for sn in state_names)
    print(f"  Total state size: {total_state_bytes/1024/1024:.1f} MB")
    
    # Read+write overhead as % of decode step
    ane_decode = chunk_results.get((1, "CPU_AND_NE"), {}).get('mean_ms', 0)
    if ane_decode > 0:
        rw_pct = (r_mean + w_mean) / ane_decode * 100
        print(f"  R/W overhead vs decode step: {rw_pct:.1f}%")

del m, state; gc.collect()


# ═══════════════════════════════════════════════════════════════════
#  PHASE 5: Compute-unit isolation experiment
# ═══════════════════════════════════════════════════════════════════

banner("PHASE 5: Compute Unit Experiments (ALL vs CPU_AND_NE vs CPU_AND_GPU)")

# Test with ct.ComputeUnit.ALL (allows GPU as fallback too) vs CPU_AND_NE-only
for cu_name, cu in [("ALL", ct.ComputeUnit.ALL), ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU)]:
    print(f"\n  Testing {cu_name} for representative chunks 1, 4, 8...")
    for ci in [1, 4, 8]:
        model_path = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
        inp_shapes = get_chunk_input_shapes(model_path)
        
        try:
            m = ct.models.MLModel(model_path, compute_units=cu, function_name="infer")
            state = m.make_state()
            
            has_linear = 'linear_conv_state' in inp_shapes
            pos = 50
            mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
            mask[:,:,:,:pos+1] = 0
            inp = {
                "hidden_states": hidden_single.copy(),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if has_linear:
                inp["linear_conv_state"] = np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16)
                inp["linear_recurrent_state"] = np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16)
            
            def run_cu(model=m, inp=inp, state=state):
                return model.predict(inp, state=state)
            
            mean_ms, std_ms, _ = measure_latency(run_cu, warmup=3, repeats=15)
            
            # Compare with existing results
            ane_ms = chunk_results.get((ci, "CPU_AND_NE"), {}).get('mean_ms', 0)
            delta = ""
            if ane_ms > 0:
                diff = (mean_ms - ane_ms) / ane_ms * 100
                delta = f" ({diff:+.1f}% vs CPU_AND_NE)"
            
            print(f"    Chunk {ci} [{cu_name}]: {fmt_ms(mean_ms)}{delta}")
            
            del m, state; gc.collect()
        except Exception as e:
            print(f"    Chunk {ci} [{cu_name}]: FAILED — {e}")


# ═══════════════════════════════════════════════════════════════════
#  PHASE 6: End-to-End Pipeline Timing
# ═══════════════════════════════════════════════════════════════════

banner("PHASE 6: End-to-End Decode Pipeline (10 tokens)")

# Load all chunks with ANE
print("Loading all chunks (CPU_AND_NE)...")
ffns = []
states = []
lin_convs = []
lin_recs = []
inp_maps = []

for ci in range(NUM_CHUNKS):
    model_path = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
    m = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE, function_name="infer")
    ffns.append(m)
    states.append(m.make_state())
    
    inp_shapes = get_chunk_input_shapes(model_path)
    inp_maps.append(inp_shapes)
    
    has_linear = 'linear_conv_state' in inp_shapes
    if has_linear:
        lin_convs.append(np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16))
        lin_recs.append(np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16))
    else:
        lin_convs.append(None)
        lin_recs.append(None)

print("Running 10-token decode pipeline with detailed timing...")

# Detailed per-component timing
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

prompt_text = "The capital of France is"
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

# Prefill prompt token-by-token (simple mode)
for i, tid in enumerate(prompt_ids):
    tok = np.array([[tid]], dtype=np.int32)
    h = list(embed.predict({"input_ids": tok}).values())[0]
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:,:,:,:i+1] = 0
    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": h.astype(np.float16),
            "position_ids": np.array([i], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([i], dtype=np.int32),
        }
        if lin_convs[ci] is not None:
            inp["linear_conv_state"] = lin_convs[ci]
            inp["linear_recurrent_state"] = lin_recs[ci]
        out = ffns[ci].predict(inp, state=states[ci])
        h = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_convs[ci] = out['linear_conv_state_out']
            lin_recs[ci] = out['linear_recurrent_state_out']
    lm_out = lmhead.predict({"hidden_states": h.astype(np.float16)})
    last_tok = int(np.argmax(lm_out["logits"].flatten()))

pos = len(prompt_ids)
print(f"  Prefilled {len(prompt_ids)} tokens, next: {tokenizer.decode([last_tok])!r}")

# Timed decode loop
gen_tokens = [last_tok]
timing_breakdown = {'embed': [], 'chunks': [[] for _ in range(NUM_CHUNKS)], 
                    'lmhead': [], 'total': [], 'data_prep': []}

for step in range(10):
    t_total_start = time.perf_counter()
    
    # Embed
    tok = np.array([[gen_tokens[-1]]], dtype=np.int32)
    t0 = time.perf_counter()
    h = list(embed.predict({"input_ids": tok}).values())[0]
    timing_breakdown['embed'].append((time.perf_counter() - t0) * 1000)
    
    # Data prep
    t0 = time.perf_counter()
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:,:,:,:pos+1] = 0
    t_data = (time.perf_counter() - t0) * 1000
    timing_breakdown['data_prep'].append(t_data)
    
    # Chunks
    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": h.astype(np.float16),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
        }
        if lin_convs[ci] is not None:
            inp["linear_conv_state"] = lin_convs[ci]
            inp["linear_recurrent_state"] = lin_recs[ci]
        
        t0 = time.perf_counter()
        out = ffns[ci].predict(inp, state=states[ci])
        timing_breakdown['chunks'][ci].append((time.perf_counter() - t0) * 1000)
        
        h = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_convs[ci] = out['linear_conv_state_out']
            lin_recs[ci] = out['linear_recurrent_state_out']
    
    # LMHead
    t0 = time.perf_counter()
    lm_out = lmhead.predict({"hidden_states": h.astype(np.float16)})
    timing_breakdown['lmhead'].append((time.perf_counter() - t0) * 1000)
    
    next_tok = int(np.argmax(lm_out["logits"].flatten()))
    gen_tokens.append(next_tok)
    
    timing_breakdown['total'].append((time.perf_counter() - t_total_start) * 1000)
    pos += 1

# Print breakdown
generated = tokenizer.decode(gen_tokens)
print(f"\n  Generated: {generated!r}")

print(f"\n  {'Component':>20s}  {'Mean ms':>10s}  {'Std ms':>8s}  {'% Total':>8s}")
print(f"  {'─'*20}  {'─'*10}  {'─'*8}  {'─'*8}")

total_mean = np.mean(timing_breakdown['total'])
for name, vals in [
    ("embed", timing_breakdown['embed']),
    ("data_prep", timing_breakdown['data_prep']),
] + [(f"chunk_{ci}", timing_breakdown['chunks'][ci]) for ci in range(NUM_CHUNKS)] + [
    ("lmhead", timing_breakdown['lmhead']),
]:
    arr = np.array(vals)
    pct = arr.mean() / total_mean * 100
    print(f"  {name:>20s}  {arr.mean():>8.2f}ms  {arr.std():>6.2f}ms  {pct:>6.1f}%")

print(f"  {'─'*20}  {'─'*10}  {'─'*8}  {'─'*8}")
print(f"  {'TOTAL':>20s}  {total_mean:>8.2f}ms  {'':>8s}  {'100.0%':>8s}")
print(f"  → {1000/total_mean:.1f} tok/s")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 7: Summary & Recommendations
# ═══════════════════════════════════════════════════════════════════

banner("SUMMARY & RECOMMENDATIONS")

print("\n1. DECODE BREAKDOWN:")
sum_chunk_ms = sum(np.mean(timing_breakdown['chunks'][ci]) for ci in range(NUM_CHUNKS))
print(f"   Total decode step: {total_mean:.2f}ms ({1000/total_mean:.1f} tok/s)")
print(f"   Chunk inference:   {sum_chunk_ms:.2f}ms ({sum_chunk_ms/total_mean*100:.1f}%)")
print(f"   Embed:             {np.mean(timing_breakdown['embed']):.2f}ms ({np.mean(timing_breakdown['embed'])/total_mean*100:.1f}%)")
print(f"   LMHead:            {np.mean(timing_breakdown['lmhead']):.2f}ms ({np.mean(timing_breakdown['lmhead'])/total_mean*100:.1f}%)")
print(f"   Data prep:         {np.mean(timing_breakdown['data_prep']):.2f}ms ({np.mean(timing_breakdown['data_prep'])/total_mean*100:.1f}%)")

print(f"\n2. ANE SPEEDUP (decode):")
print(f"   Overall: {total_speedup:.2f}x over CPU_ONLY")
# Find worst chunks
worst_chunks = sorted(range(NUM_CHUNKS), 
    key=lambda ci: chunk_results.get((ci, "CPU_ONLY"), {}).get('mean_ms', 0) / 
                   max(chunk_results.get((ci, "CPU_AND_NE"), {}).get('mean_ms', 1), 0.01))
print(f"   Lowest ANE benefit: chunk {worst_chunks[0]} ({CHUNK_RANGES[worst_chunks[0]][0]}-{CHUNK_RANGES[worst_chunks[0]][1]-1})")
print(f"   Highest ANE benefit: chunk {worst_chunks[-1]} ({CHUNK_RANGES[worst_chunks[-1]][0]}-{CHUNK_RANGES[worst_chunks[-1]][1]-1})")

print(f"\n3. POTENTIAL BOTTLENECKS:")
# Embed/lmhead on CPU
embed_pct = np.mean(timing_breakdown['embed']) / total_mean * 100
lmhead_pct = np.mean(timing_breakdown['lmhead']) / total_mean * 100
if embed_pct > 10:
    print(f"   ⚠ Embed uses {embed_pct:.1f}% of token time (CPU_ONLY)")
if lmhead_pct > 10:
    print(f"   ⚠ LMHead uses {lmhead_pct:.1f}% of token time (CPU_ONLY)")

# Cast overhead
total_casts = sum(decode_analysis.get(ci, {}).get('cast_count', 0) for ci in range(NUM_CHUNKS))
if total_casts > 0:
    print(f"   ⚠ {total_casts} cast ops across all decode chunks (CPU↔ANE data conversion)")

# ANE-unfriendly ops
total_uf = sum(decode_analysis.get(ci, {}).get('unfriendly_count', 0) for ci in range(NUM_CHUNKS))
if total_uf > 0:
    print(f"   ⚠ {total_uf} ANE-unfriendly ops (may cause CPU fallback)")

print(f"\n4. EMBED/LMHEAD ON ANE EXPERIMENT NEEDED:")
print(f"   Current: embed={fmt_ms(embed_mean)}, lmhead={fmt_ms(lmhead_mean)} (both CPU_ONLY)")
print(f"   Recommendation: Test embed/lmhead on CPU_AND_NE to see if they benefit from ANE")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 8: Embed/LMHead ANE experiment
# ═══════════════════════════════════════════════════════════════════

banner("PHASE 8: Embed & LMHead ANE Experiment")

# embed single on ANE
print("Testing embed on CPU_AND_NE...")
try:
    embed_ane = ct.models.MLModel(combined_el, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                  function_name="embed")
    def run_embed_ane():
        return embed_ane.predict({"input_ids": tok_input})
    ea_mean, ea_std, _ = measure_latency(run_embed_ane)
    speedup = embed_mean / ea_mean if ea_mean > 0 else 0
    print(f"  embed CPU: {fmt_ms(embed_mean)}, ANE: {fmt_ms(ea_mean)} → {speedup:.2f}x")
    del embed_ane; gc.collect()
except Exception as e:
    print(f"  embed ANE: FAILED — {e}")

# lmhead on ANE
print("Testing lmhead on CPU_AND_NE...")
try:
    lmhead_ane = ct.models.MLModel(combined_el, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                   function_name="lmhead")
    def run_lmhead_ane():
        return lmhead_ane.predict({"hidden_states": hidden_single})
    la_mean, la_std, _ = measure_latency(run_lmhead_ane)
    speedup = lmhead_mean / la_mean if la_mean > 0 else 0
    print(f"  lmhead CPU: {fmt_ms(lmhead_mean)}, ANE: {fmt_ms(la_mean)} → {speedup:.2f}x")
    del lmhead_ane; gc.collect()
except Exception as e:
    print(f"  lmhead ANE: FAILED — {e}")

# embed_prefill on ANE
print("Testing embed_prefill on CPU_AND_NE...")
try:
    ep_ane = ct.models.MLModel(embed_prefill_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    def run_ep_ane():
        return ep_ane.predict({"input_ids": prefill_input})
    epa_mean, epa_std, _ = measure_latency(run_ep_ane, repeats=NUM_PREFILL_MEASURE)
    speedup = ep_mean / epa_mean if epa_mean > 0 else 0
    print(f"  embed_prefill CPU: {fmt_ms(ep_mean)}, ANE: {fmt_ms(epa_mean)} → {speedup:.2f}x")
    del ep_ane; gc.collect()
except Exception as e:
    print(f"  embed_prefill ANE: FAILED — {e}")

# lm_head_nosplit (separate file) on ANE
lmhead_nosplit = os.path.join(MODEL_DIR, 'lm_head_nosplit.mlpackage')
if os.path.exists(lmhead_nosplit):
    print("Testing lm_head_nosplit on CPU_AND_NE...")
    try:
        ln_ane = ct.models.MLModel(lmhead_nosplit, compute_units=ct.ComputeUnit.CPU_AND_NE)
        def run_ln_ane():
            return ln_ane.predict({"hidden_states": hidden_single})
        lna_mean, lna_std, _ = measure_latency(run_ln_ane)
        print(f"  lm_head_nosplit CPU_AND_NE: {fmt_ms(lna_mean)}")
        
        ln_cpu = ct.models.MLModel(lmhead_nosplit, compute_units=ct.ComputeUnit.CPU_ONLY)
        def run_ln_cpu():
            return ln_cpu.predict({"hidden_states": hidden_single})
        lnc_mean, lnc_std, _ = measure_latency(run_ln_cpu)
        speedup = lnc_mean / lna_mean if lna_mean > 0 else 0
        print(f"  lm_head_nosplit CPU: {fmt_ms(lnc_mean)}, ANE: {fmt_ms(lna_mean)} → {speedup:.2f}x")
        del ln_ane, ln_cpu; gc.collect()
    except Exception as e:
        print(f"  lm_head_nosplit: FAILED — {e}")


banner("PROFILING COMPLETE")
print("Done.")
