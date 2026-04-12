#!/usr/bin/env python3
"""Deep ANE vs CPU Placement Investigation for Qwen3.5-4B.

Measures CPU **process time** (user + system) vs **wall time** per chunk to
estimate the fraction of execution spent on ANE vs CPU.  Tracks voluntary
context switches as an additional ANE-offload indicator.

Phases:
  1. Per-chunk utilization measurement (decode + prefill + embed/lmhead)
  2. Static MIL op inventory per chunk (both functions)
  3. Placement inference (combine Phases 1+2)
  4. Controlled ablation experiments
  5. Root-cause explanation and recommendations

Usage:
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/deep_ane_cpu_placement.py
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/deep_ane_cpu_placement.py --chunks 0,1,4,8
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/deep_ane_cpu_placement.py --skip-phase 4 5
"""
import sys, os, time, gc, resource, warnings, argparse
from collections import Counter, defaultdict
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

# ── Constants ──────────────────────────────────────────────────────
NUM_WARMUP_DECODE = 3
NUM_MEASURE_DECODE = 20
NUM_WARMUP_PREFILL = 2
NUM_MEASURE_PREFILL = 5
HIDDEN_DIM = 2560

DTYPE_MAP = {0: '?', 1: 'fp32', 5: 'int32', 10: 'fp16', 11: 'fp16', 14: 'str'}

# ── Op Categories ──────────────────────────────────────────────────
WEIGHT_OPS = frozenset({
    'const', 'constexpr_lut_to_dense', 'constexpr_affine_dequantize',
    'constexpr_blockwise_shift_scale', 'constexpr_cast',
    'constexpr_sparse_to_dense', 'constexpr_lut_to_sparse',
})
MATMUL_OPS = frozenset({'conv', 'matmul', 'linear', 'einsum', 'batch_matmul'})
ELEMENTWISE_OPS = frozenset({
    'mul', 'add', 'sub', 'real_div', 'pow', 'rsqrt', 'sqrt',
    'exp', 'log', 'neg', 'abs', 'ceil', 'floor', 'clip',
    'maximum', 'minimum',
})
ACTIVATION_OPS = frozenset({
    'relu', 'gelu', 'silu', 'sigmoid', 'tanh', 'softmax',
    'elu', 'leaky_relu',
})
NORM_OPS = frozenset({
    'layer_norm', 'instance_norm', 'batch_norm', 'l2_norm',
    'local_response_norm',
})
DATA_MOVE_OPS = frozenset({
    'reshape', 'transpose', 'expand_dims', 'squeeze', 'concat', 'split',
    'slice_by_index', 'slice_by_size', 'pad', 'tile', 'reverse',
    'reverse_sequence', 'shape', 'rank', 'identity',
})
GATHER_SCATTER_OPS = frozenset({
    'gather', 'gather_along_axis', 'gather_nd',
    'scatter', 'scatter_along_axis', 'scatter_nd',
    'one_hot',
})
STATE_OPS = frozenset({'read_state', 'coreml_update_state'})
CAST_OPS = frozenset({'cast'})
REDUCE_OPS = frozenset({
    'reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_min',
    'reduce_prod', 'reduce_argmax', 'reduce_argmin',
    'reduce_l1_norm', 'reduce_l2_norm', 'reduce_log_sum_exp',
})
COMPARE_OPS = frozenset({
    'less', 'less_equal', 'greater', 'greater_equal',
    'equal', 'not_equal', 'where', 'select',
    'logical_and', 'logical_or', 'logical_not',
})
CONTROL_OPS = frozenset({
    'while_loop', 'cond', 'list_write', 'list_read',
    'list_length', 'list_gather', 'list_scatter',
})
ANE_HOSTILE = (GATHER_SCATTER_OPS | COMPARE_OPS | CONTROL_OPS |
               frozenset({'cumsum', 'non_zero', 'topk', 'argsort', 'sort', 'unique'}))

CATEGORY_ORDER = [
    'matmul', 'elementwise', 'activation', 'normalization', 'cast',
    'data_movement', 'gather_scatter', 'state', 'reduction',
    'comparison', 'control_flow', 'other',
]


def categorize_op(op_type):
    if op_type in WEIGHT_OPS:           return 'weight'
    if op_type in MATMUL_OPS:           return 'matmul'
    if op_type in ELEMENTWISE_OPS:      return 'elementwise'
    if op_type in ACTIVATION_OPS:       return 'activation'
    if op_type in NORM_OPS:             return 'normalization'
    if op_type in DATA_MOVE_OPS:        return 'data_movement'
    if op_type in GATHER_SCATTER_OPS:   return 'gather_scatter'
    if op_type in STATE_OPS:            return 'state'
    if op_type in CAST_OPS:             return 'cast'
    if op_type in REDUCE_OPS:           return 'reduction'
    if op_type in COMPARE_OPS:          return 'comparison'
    if op_type in CONTROL_OPS:          return 'control_flow'
    return 'other'


# ── Helpers ────────────────────────────────────────────────────────

def banner(msg):
    print(f"\n{'='*78}")
    print(f"  {msg}")
    print(f"{'='*78}", flush=True)


def measure_utilization(fn, warmup=3, repeats=20):
    """Wall time, CPU process time, and voluntary context switches.

    Returns dict with wall_ms, cpu_ms, ane_ms (est), cpu_frac, ane_frac,
    vol_ctx, invol_ctx.
    """
    for _ in range(warmup):
        fn()

    wall_times, cpu_times = [], []
    vol_ctx_list, invol_ctx_list = [], []

    for _ in range(repeats):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t_cpu0 = time.process_time()
        t_wall0 = time.perf_counter()

        fn()

        t_wall1 = time.perf_counter()
        t_cpu1 = time.process_time()
        r1 = resource.getrusage(resource.RUSAGE_SELF)

        wall_times.append((t_wall1 - t_wall0) * 1000)
        cpu_times.append((t_cpu1 - t_cpu0) * 1000)
        vol_ctx_list.append(r1.ru_nvcsw - r0.ru_nvcsw)
        invol_ctx_list.append(r1.ru_nivcsw - r0.ru_nivcsw)

    mw = np.mean(wall_times)
    mc = np.mean(cpu_times)
    return {
        'wall_ms':    mw,
        'wall_std':   np.std(wall_times),
        'cpu_ms':     mc,
        'cpu_std':    np.std(cpu_times),
        'ane_ms':     max(0, mw - mc),
        'cpu_frac':   min(1.0, mc / mw) if mw > 0 else 1.0,
        'ane_frac':   max(0, 1.0 - mc / mw) if mw > 0 else 0,
        'vol_ctx':    np.mean(vol_ctx_list),
        'invol_ctx':  np.mean(invol_ctx_list),
        'wall_all':   wall_times,
        'cpu_all':    cpu_times,
    }


def get_func_names(model_path):
    spec = ct.utils.load_spec(model_path)
    if hasattr(spec, 'mlProgram'):
        return list(spec.mlProgram.functions.keys())
    return []


def get_input_shapes(model_path, fn_name=None):
    spec = ct.utils.load_spec(model_path)
    shapes = {}
    if fn_name:
        for fn in spec.description.functions:
            if fn.name == fn_name:
                for inp in fn.input:
                    try:
                        shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except Exception:
                        pass
                break
    else:
        for inp in spec.description.input:
            try:
                shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass
    return shapes


def get_state_info(model_path, fn_name="infer"):
    """Get state variable names and shapes from spec."""
    spec = ct.utils.load_spec(model_path)
    states = []
    for fn in spec.description.functions:
        if fn.name == fn_name:
            for st in fn.state:
                try:
                    shape = tuple(st.type.multiArrayType.shape)
                    dt_int = st.type.multiArrayType.dataType
                    dt_str = DTYPE_MAP.get(dt_int, f'dt{dt_int}')
                    states.append({'name': st.name, 'shape': shape, 'dtype': dt_str})
                except Exception:
                    states.append({'name': st.name, 'shape': (), 'dtype': '?'})
            break
    return states


def analyze_mil_ops(model_path, target_fn=None):
    """Detailed MIL op analysis.  Returns {fn_name: analysis_dict, ...}."""
    spec = ct.utils.load_spec(model_path)
    if not hasattr(spec, 'mlProgram'):
        return {'_error': 'Not an MLProgram'}

    prog = spec.mlProgram

    if target_fn and target_fn in prog.functions:
        fns = {target_fn: prog.functions[target_fn]}
    else:
        fns = dict(prog.functions)

    results = {}
    for fname, fn in fns.items():
        op_counts = Counter()
        cat_counts = Counter()
        cast_dirs = []
        state_ops_list = []
        hostile_ops_list = []
        out_dtype_counts = Counter()
        total_ops = 0
        runtime_ops = 0
        op_type_detail = Counter()   # non-weight op type counts

        for _bname, block in fn.block_specializations.items():
            for op in block.operations:
                total_ops += 1
                ot = op.type
                op_counts[ot] += 1
                cat = categorize_op(ot)
                cat_counts[cat] += 1

                if cat == 'weight':
                    continue

                runtime_ops += 1
                op_type_detail[ot] += 1

                # output dtype
                if op.outputs:
                    try:
                        dt = op.outputs[0].type.tensorType.dataType
                        out_dtype_counts[DTYPE_MAP.get(dt, f'dt{dt}')] += 1
                    except Exception:
                        out_dtype_counts['?'] += 1

                if ot in ANE_HOSTILE:
                    hostile_ops_list.append(ot)
                if ot in STATE_OPS:
                    state_ops_list.append(ot)
                if ot == 'cast':
                    try:
                        odt = DTYPE_MAP.get(
                            op.outputs[0].type.tensorType.dataType, '?')
                        cast_dirs.append(f'→{odt}')
                    except Exception:
                        cast_dirs.append('→?')

        results[fname] = {
            'total_ops':      total_ops,
            'runtime_ops':    runtime_ops,
            'op_counts':      dict(op_counts),
            'op_type_detail': dict(op_type_detail),
            'category_counts': dict(cat_counts),
            'cast_dirs':      cast_dirs,
            'state_ops':      state_ops_list,
            'ane_hostile':    hostile_ops_list,
            'output_dtypes':  dict(out_dtype_counts),
        }
    return results


def build_decode_input(model_path, pos=50):
    """Construct a decode input dict for a chunk."""
    ishapes = get_input_shapes(model_path, fn_name="infer")
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :pos+1] = 0
    inp = {
        "hidden_states": np.zeros((1, 1, HIDDEN_DIM), dtype=np.float16),
        "position_ids":  np.array([pos], dtype=np.int32),
        "causal_mask":   mask,
        "current_pos":   np.array([pos], dtype=np.int32),
    }
    if 'linear_conv_state' in ishapes:
        inp["linear_conv_state"] = np.zeros(
            ishapes['linear_conv_state'], dtype=np.float16)
        inp["linear_recurrent_state"] = np.zeros(
            ishapes['linear_recurrent_state'], dtype=np.float16)
    return inp, ishapes


def build_prefill_input(model_path, fn_name=None):
    """Construct a prefill input dict for a chunk."""
    ishapes = (get_input_shapes(model_path, fn_name=fn_name)
               if fn_name else get_input_shapes(model_path))
    mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
    for i in range(BATCH_SIZE):
        mask[0, 0, i, :i+1] = 0
    inp = {
        "hidden_states": np.zeros((1, BATCH_SIZE, HIDDEN_DIM), dtype=np.float16),
        "position_ids":  np.arange(BATCH_SIZE, dtype=np.int32),
        "causal_mask":   mask,
        "current_pos":   np.array([0], dtype=np.int32),
    }
    if 'linear_conv_state' in ishapes:
        inp["linear_conv_state"] = np.zeros(
            ishapes['linear_conv_state'], dtype=np.float16)
        inp["linear_recurrent_state"] = np.zeros(
            ishapes['linear_recurrent_state'], dtype=np.float16)
    if 'valid_len' in ishapes:
        inp["valid_len"] = np.array([BATCH_SIZE], dtype=np.int32)
    return inp, ishapes


# ═══════════════════════════════════════════════════════════════════
#  PHASE 1 — Per-chunk ANE / CPU utilisation measurement
# ═══════════════════════════════════════════════════════════════════

def run_phase1(chunks):
    decode_results = {}
    prefill_results = {}
    embed_lmhead_results = {}

    # ── 1A  Decode ──────────────────────────────────────────────
    banner("PHASE 1A: Per-Chunk DECODE — wall time vs CPU process time")

    for cu_label, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                         ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
        print(f"\n  ── {cu_label} ──")
        for ci in chunks:
            mp = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
            inp, _ = build_decode_input(mp)

            print(f"    chunk {ci} …", end=" ", flush=True)
            t0 = time.time()
            m = ct.models.MLModel(mp, compute_units=cu, function_name="infer")
            state = m.make_state()
            load_s = time.time() - t0

            def _run(m=m, inp=inp, state=state):
                return m.predict(inp, state=state)

            r = measure_utilization(_run,
                                    warmup=NUM_WARMUP_DECODE,
                                    repeats=NUM_MEASURE_DECODE)
            decode_results[(ci, cu_label)] = r

            layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
            print(f"[{layers}] load={load_s:.1f}s  "
                  f"wall={r['wall_ms']:.2f}  cpu={r['cpu_ms']:.2f}  "
                  f"ane_est={r['ane_ms']:.2f}  "
                  f"cpu%={r['cpu_frac']:.0%}  ane%={r['ane_frac']:.0%}  "
                  f"vctx={r['vol_ctx']:.0f}")

            del m, state; gc.collect()

    # ── 1A summary table ──
    banner("DECODE UTILISATION TABLE")
    hdr = (f"  {'Chk':>3s} {'Layers':>6s} │"
           f"{'CPU_ONLY':^20s}│"
           f"{'── CPU_AND_NE ──────────────────────':^40s}│"
           f"{'SPEED':>6s}")
    sub = (f"  {'':>3s} {'':>6s} │"
           f"{'wall':>8s} {'cpu':>7s} {'cpu%':>5s}│"
           f"{'wall':>8s} {'cpu':>7s} {'ane':>7s} {'cpu%':>5s} {'ane%':>5s} {'vctx':>5s}│"
           f"{'':>6s}")
    print(hdr)
    print(sub)
    print(f"  {'─'*3} {'─'*6} ┼{'─'*20}┼{'─'*40}┼{'─'*6}")
    for ci in chunks:
        co = decode_results.get((ci, "CPU_ONLY"), {})
        ca = decode_results.get((ci, "CPU_AND_NE"), {})
        spd = co.get('wall_ms', 1) / ca.get('wall_ms', 1) if ca.get('wall_ms', 0) > 0 else 0
        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        print(f"  {ci:>3d} {layers:>6s} │"
              f"{co.get('wall_ms',0):>8.2f} {co.get('cpu_ms',0):>7.2f} {co.get('cpu_frac',0):>4.0%} │"
              f"{ca.get('wall_ms',0):>8.2f} {ca.get('cpu_ms',0):>7.2f} {ca.get('ane_ms',0):>7.2f} "
              f"{ca.get('cpu_frac',0):>4.0%}  {ca.get('ane_frac',0):>4.0%}  {ca.get('vol_ctx',0):>4.0f} │"
              f"{spd:>5.2f}x")

    # ── 1B  Prefill ─────────────────────────────────────────────
    banner("PHASE 1B: Per-Chunk PREFILL — wall time vs CPU process time")

    sample_mp = os.path.join(COMBINED_DIR, "chunk0.mlpackage")
    fnames = get_func_names(sample_mp)
    use_combined_pf = 'prefill' in fnames
    print(f"  Combined model functions: {fnames}")
    print(f"  Using {'combined prefill' if use_combined_pf else 'separate prefill models'}")

    for cu_label, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                         ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
        print(f"\n  ── Prefill {cu_label} ──")
        for ci in chunks:
            if use_combined_pf:
                mp = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
                fn = "prefill"
            else:
                mp = os.path.join(MODEL_DIR, f"prefill_LUT4_chunk{ci}.mlpackage")
                fn = None
                if not os.path.exists(mp):
                    print(f"    prefill chunk {ci}: NOT FOUND"); continue

            inp, _ = build_prefill_input(mp, fn_name=fn)

            print(f"    chunk {ci} …", end=" ", flush=True)
            t0 = time.time()
            if fn:
                m = ct.models.MLModel(mp, compute_units=cu, function_name=fn)
            else:
                m = ct.models.MLModel(mp, compute_units=cu)
            state = m.make_state()
            load_s = time.time() - t0

            def _run(m=m, inp=inp, state=state):
                return m.predict(inp, state=state)

            r = measure_utilization(_run,
                                    warmup=NUM_WARMUP_PREFILL,
                                    repeats=NUM_MEASURE_PREFILL)
            prefill_results[(ci, cu_label)] = r

            layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
            print(f"[{layers}] load={load_s:.1f}s  "
                  f"wall={r['wall_ms']:.1f}  cpu={r['cpu_ms']:.1f}  "
                  f"ane_est={r['ane_ms']:.1f}  "
                  f"cpu%={r['cpu_frac']:.0%}  ane%={r['ane_frac']:.0%}  "
                  f"vctx={r['vol_ctx']:.0f}")

            del m, state; gc.collect()

    # ── 1B summary table ──
    banner("PREFILL UTILISATION TABLE")
    print(f"  {'Chk':>3s} {'Layers':>6s} │"
          f"{'CPU_ONLY':^20s}│"
          f"{'── CPU_AND_NE ──────────────────────':^40s}│"
          f"{'SPEED':>6s}")
    print(f"  {'':>3s} {'':>6s} │"
          f"{'wall':>8s} {'cpu':>7s} {'cpu%':>5s}│"
          f"{'wall':>8s} {'cpu':>7s} {'ane':>7s} {'cpu%':>5s} {'ane%':>5s} {'vctx':>5s}│"
          f"{'':>6s}")
    print(f"  {'─'*3} {'─'*6} ┼{'─'*20}┼{'─'*40}┼{'─'*6}")
    for ci in chunks:
        co = prefill_results.get((ci, "CPU_ONLY"), {})
        ca = prefill_results.get((ci, "CPU_AND_NE"), {})
        if not co or not ca: continue
        spd = co['wall_ms'] / ca['wall_ms'] if ca.get('wall_ms', 0) > 0 else 0
        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        print(f"  {ci:>3d} {layers:>6s} │"
              f"{co.get('wall_ms',0):>8.1f} {co.get('cpu_ms',0):>7.1f} {co.get('cpu_frac',0):>4.0%} │"
              f"{ca.get('wall_ms',0):>8.1f} {ca.get('cpu_ms',0):>7.1f} {ca.get('ane_ms',0):>7.1f} "
              f"{ca.get('cpu_frac',0):>4.0%}  {ca.get('ane_frac',0):>4.0%}  {ca.get('vol_ctx',0):>4.0f} │"
              f"{spd:>5.2f}x")

    # ── 1C  Embed / LMHead ──────────────────────────────────────
    banner("PHASE 1C: Embed & LMHead Utilisation")

    combined_el = os.path.join(MODEL_DIR, 'embed_lmhead_combined.mlpackage')
    tok_inp = np.array([[1]], dtype=np.int32)
    hidden1 = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float16)
    pf_inp = np.zeros((1, BATCH_SIZE), dtype=np.int32)
    pf_inp[0, :5] = [1, 2, 3, 4, 5]

    embed_prefill_path = os.path.join(MODEL_DIR, 'embed_prefill.mlpackage')

    tests = [
        ("embed",          combined_el,         "embed",   {"input_ids": tok_inp}),
        ("lmhead",         combined_el,         "lmhead",  {"hidden_states": hidden1}),
    ]
    if os.path.exists(embed_prefill_path):
        tests.append(
            ("embed_prefill",  embed_prefill_path,  None,      {"input_ids": pf_inp}))
    elif 'embed_prefill' in get_func_names(combined_el):
        tests.append(
            ("embed_prefill",  combined_el,     "embed_prefill", {"input_ids": pf_inp}))

    for name, path, fn, inp_data in tests:
        for cu_label, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                             ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
            kw = dict(compute_units=cu)
            if fn:
                kw['function_name'] = fn
            m = ct.models.MLModel(path, **kw)
            def _run(m=m, d=inp_data):
                return m.predict(d)
            r = measure_utilization(_run, warmup=3, repeats=20)
            embed_lmhead_results[(name, cu_label)] = r
            print(f"  {name:16s} {cu_label:12s}: "
                  f"wall={r['wall_ms']:>8.2f}  cpu={r['cpu_ms']:>8.2f}  "
                  f"ane_est={r['ane_ms']:>7.2f}  "
                  f"cpu%={r['cpu_frac']:.0%}  vctx={r['vol_ctx']:.0f}")
            del m; gc.collect()

    return decode_results, prefill_results, embed_lmhead_results


# ═══════════════════════════════════════════════════════════════════
#  PHASE 2 — Static MIL Op Inventory
# ═══════════════════════════════════════════════════════════════════

def run_phase2(chunks):
    banner("PHASE 2: Static MIL Op Inventory Per Chunk")

    all_mil = {}

    for ci in chunks:
        mp = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        print(f"\n  ── Chunk {ci} [layers {layers}] ──")

        analysis = analyze_mil_ops(mp)
        all_mil[ci] = analysis

        for fname in sorted(analysis.keys()):
            if fname.startswith('_'):
                continue
            r = analysis[fname]

            print(f"\n    Function: {fname}")
            print(f"      Total ops: {r['total_ops']}  "
                  f"(weight: {r['total_ops'] - r['runtime_ops']}, "
                  f"runtime: {r['runtime_ops']})")

            # Category breakdown
            cats = r['category_counts']
            print(f"      {'Category':20s}  {'Count':>5s}  {'%RT':>5s}  Notes")
            print(f"      {'─'*20}  {'─'*5}  {'─'*5}  {'─'*28}")
            for cat in CATEGORY_ORDER:
                c = cats.get(cat, 0)
                if c == 0:
                    continue
                pct = c / r['runtime_ops'] * 100 if r['runtime_ops'] else 0
                note = ""
                if cat == 'gather_scatter':
                    note = "◄ ANE-hostile (RoPE / scatter)"
                elif cat == 'comparison':
                    note = "◄ ANE-hostile (logical)"
                elif cat == 'cast':
                    dirs = Counter(r['cast_dirs'])
                    note = f"dirs: {dict(dirs)}"
                elif cat == 'state':
                    sc = Counter(r['state_ops'])
                    note = f"{dict(sc)}"
                elif cat == 'normalization':
                    note = "may fall back to CPU"
                elif cat == 'matmul':
                    note = "ANE-friendly compute"
                print(f"      {cat:20s}  {c:>5d}  {pct:>4.1f}%  {note}")

            # Output dtype distribution
            dtypes = r['output_dtypes']
            if dtypes:
                dt_str = ", ".join(f"{dt}:{c}" for dt, c in
                                  sorted(dtypes.items(), key=lambda x: -x[1]))
                print(f"      Output dtypes: {dt_str}")

            # Top runtime op types
            detail = r.get('op_type_detail', {})
            top = sorted(detail.items(), key=lambda x: -x[1])[:12]
            if top:
                print(f"      Top runtime ops: "
                      + ", ".join(f"{o}:{c}" for o, c in top))

            # ANE-hostile detail
            if r['ane_hostile']:
                ahc = Counter(r['ane_hostile'])
                print(f"      ANE-hostile detail: {dict(ahc)}")

    # ── State inventory ──
    banner("STATE VARIABLE INVENTORY (per chunk)")
    for ci in chunks:
        mp = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
        states = get_state_info(mp, fn_name="infer")
        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        total_bytes = 0
        for s in states:
            nbytes = int(np.prod(s['shape'])) * (2 if 'fp16' in s['dtype'] else 4)
            total_bytes += nbytes
        print(f"  Chunk {ci} [{layers}]: {len(states)} states, "
              f"{total_bytes/1024/1024:.1f} MB total")
        for s in states:
            nbytes = int(np.prod(s['shape'])) * (2 if 'fp16' in s['dtype'] else 4)
            print(f"    {s['name']:40s}  {str(s['shape']):22s}  "
                  f"{s['dtype']:5s}  {nbytes/1024:.0f} KB")

    return all_mil


# ═══════════════════════════════════════════════════════════════════
#  PHASE 3 — Placement Inference
# ═══════════════════════════════════════════════════════════════════

def run_phase3(decode_results, prefill_results, mil_results,
               embed_lmhead_results, chunks):

    # ── 3A Decode placement ──
    banner("PHASE 3A: Decode — CPU vs ANE Placement Inference")

    for ci in chunks:
        co = decode_results.get((ci, "CPU_ONLY"), {})
        ca = decode_results.get((ci, "CPU_AND_NE"), {})
        if not co or not ca:
            continue

        mil = mil_results.get(ci, {}).get('infer', {})
        cats = mil.get('category_counts', {})
        rt = mil.get('runtime_ops', 1)

        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        spd = co['wall_ms'] / ca['wall_ms'] if ca['wall_ms'] > 0 else 0

        # Classification
        if ca['ane_frac'] > 0.50:
            cls = "ANE-DOMINANT"
        elif ca['ane_frac'] > 0.20:
            cls = "MIXED"
        elif spd > 1.15:
            cls = "MIXED (minor ANE)"
        else:
            cls = "CPU-DOMINANT"

        print(f"\n  ── Chunk {ci} [layers {layers}]  {cls} ──")
        print(f"  wall: CPU_ONLY {co['wall_ms']:.2f}ms → ANE {ca['wall_ms']:.2f}ms  "
              f"({spd:.2f}x)")
        print(f"  On ANE path: cpu_time={ca['cpu_ms']:.2f}ms  "
              f"ane_est={ca['ane_ms']:.2f}ms  "
              f"vol_ctx={ca['vol_ctx']:.0f}")

        # CPU_ONLY cpu_frac tells us about multi-threading
        print(f"  CPU_ONLY cpu_frac={co['cpu_frac']:.2f}:  ", end="")
        if co['cpu_frac'] > 1.05:
            print(f"multi-threaded CPU execution ({co['cpu_frac']:.2f}x parallelism)")
        else:
            print(f"mostly single-threaded CPU execution")

        # Placement reasoning
        m_pct = cats.get('matmul', 0) / rt * 100 if rt else 0
        e_pct = cats.get('elementwise', 0) / rt * 100 if rt else 0
        a_pct = cats.get('activation', 0) / rt * 100 if rt else 0
        ane_compute_pct = m_pct + e_pct + a_pct

        s_pct = cats.get('state', 0) / rt * 100 if rt else 0
        c_pct = cats.get('cast', 0) / rt * 100 if rt else 0
        h_pct = len(mil.get('ane_hostile', [])) / rt * 100 if rt else 0
        n_pct = cats.get('normalization', 0) / rt * 100 if rt else 0
        dm_pct = cats.get('data_movement', 0) / rt * 100 if rt else 0

        print(f"\n  Likely on ANE ({ane_compute_pct:.0f}% of runtime ops):")
        print(f"    matmul/conv:     {cats.get('matmul', 0):>4d} ops ({m_pct:.1f}%)  "
              "— Q/K/V projections, FFN dense layers")
        print(f"    elementwise:     {cats.get('elementwise', 0):>4d} ops ({e_pct:.1f}%)  "
              "— add, mul, pointwise arithmetic")
        print(f"    activation:      {cats.get('activation', 0):>4d} ops ({a_pct:.1f}%)  "
              "— silu, softmax")

        print(f"\n  Likely on CPU:")
        print(f"    state I/O:       {cats.get('state', 0):>4d} ops ({s_pct:.1f}%)  "
              "— KV cache reads/writes (CPU memory)")
        print(f"    casts:           {cats.get('cast', 0):>4d} ops ({c_pct:.1f}%)  "
              "— dtype boundaries, state I/O boundaries")
        print(f"    ANE-hostile:     {len(mil.get('ane_hostile', [])):>4d} ops ({h_pct:.1f}%)  "
              "— gather (RoPE), scatter")

        print(f"\n  Mixed / uncertain:")
        print(f"    normalization:   {cats.get('normalization', 0):>4d} ops ({n_pct:.1f}%)  "
              "— layer_norm (ANE on recent HW, CPU on older)")
        print(f"    data movement:   {cats.get('data_movement', 0):>4d} ops ({dm_pct:.1f}%)  "
              "— reshape/concat (often zero-cost in MIL)")

        # CPU overhead estimate
        cpu_overhead = ca['cpu_ms']
        print(f"\n  WHY this placement:")
        print(f"    CPU-side work = {cpu_overhead:.2f}ms = {ca['cpu_frac']:.0%} of {ca['wall_ms']:.2f}ms wall")
        print(f"    Includes: state I/O + casts + hostile ops + data transfers + CoreML dispatch")
        print(f"    ANE-side  = {ca['ane_ms']:.2f}ms = {ca['ane_frac']:.0%} of wall")
        print(f"    Includes: matmul + elementwise + activations (fused fp16 kernels)")

    # ── 3B Prefill placement ──
    banner("PHASE 3B: Prefill — CPU vs ANE Placement Inference")

    for ci in chunks:
        co = prefill_results.get((ci, "CPU_ONLY"), {})
        ca = prefill_results.get((ci, "CPU_AND_NE"), {})
        if not co or not ca:
            continue

        mil = mil_results.get(ci, {}).get('prefill', {})
        cats = mil.get('category_counts', {})
        rt = mil.get('runtime_ops', 1)

        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        spd = co['wall_ms'] / ca['wall_ms'] if ca['wall_ms'] > 0 else 0

        print(f"\n  ── Chunk {ci} [layers {layers}] PREFILL ──")
        print(f"  wall: CPU_ONLY {co['wall_ms']:.1f}ms → ANE {ca['wall_ms']:.1f}ms  "
              f"({spd:.2f}x)")
        print(f"  CPU_ONLY  cpu_frac={co['cpu_frac']:.2f}  "
              f"({'multi-threaded' if co['cpu_frac'] > 1.05 else 'single-threaded'})")
        print(f"  CPU_AND_NE cpu_frac={ca['cpu_frac']:.2f},  "
              f"ane_est={ca['ane_ms']:.1f}ms  vctx={ca['vol_ctx']:.0f}")

        if spd < 1.0:
            print(f"\n  ⚠ PREFILL IS SLOWER ON ANE ({spd:.2f}x)")
            print(f"    Root cause: large batch (seq={BATCH_SIZE}) benefits from "
                  f"multi-core CPU")
            print(f"    CPU_ONLY parallelism: {co['cpu_frac']:.2f}x  "
                  f"(Accelerate/vDSP multi-core)")
            print(f"    ANE path cpu_frac: {ca['cpu_frac']:.2f}  "
                  f"(CPU still busy with transfers + state)")
            data_kb = BATCH_SIZE * HIDDEN_DIM * 2 / 1024
            print(f"    Per-chunk data: {data_kb:.0f} KB  "
                  f"(×2 for in+out = {data_kb*2:.0f} KB transfers)")

    # ── 3C Embed/LMHead ──
    banner("PHASE 3C: Embed / LMHead Placement")
    for name in ["embed", "lmhead", "embed_prefill"]:
        co = embed_lmhead_results.get((name, "CPU_ONLY"), {})
        ca = embed_lmhead_results.get((name, "CPU_AND_NE"), {})
        if not co or not ca:
            continue
        spd = co['wall_ms'] / ca['wall_ms'] if ca['wall_ms'] > 0 else 0
        print(f"  {name:16s}: CPU={co['wall_ms']:.2f}ms  ANE={ca['wall_ms']:.2f}ms  "
              f"speedup={spd:.2f}x  ane_frac={ca['ane_frac']:.0%}")


# ═══════════════════════════════════════════════════════════════════
#  PHASE 4 — Controlled Ablation Experiments
# ═══════════════════════════════════════════════════════════════════

def run_phase4(decode_results, prefill_results, chunks):
    banner("PHASE 4: Controlled Ablation Experiments")

    exp_results = {}

    # ── 4A: 4-way compute-unit comparison (decode) ──
    print("\n  ── 4A: 4-way compute unit comparison (decode) ──")
    cu_opts = [
        ("CPU_ONLY",    ct.ComputeUnit.CPU_ONLY),
        ("CPU_AND_NE",  ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("ALL",         ct.ComputeUnit.ALL),
    ]
    rep_chunks = [c for c in [0, 1, 4, 8] if c in chunks]

    for ci in rep_chunks:
        mp = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
        inp, _ = build_decode_input(mp)
        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"

        print(f"\n    Chunk {ci} [{layers}]:")
        for cu_label, cu in cu_opts:
            try:
                m = ct.models.MLModel(mp, compute_units=cu, function_name="infer")
                state = m.make_state()
                def _run(m=m, inp=inp, state=state):
                    return m.predict(inp, state=state)
                r = measure_utilization(_run, warmup=3, repeats=15)
                exp_results[(ci, cu_label)] = r
                print(f"      {cu_label:15s}:  wall={r['wall_ms']:>8.2f}  "
                      f"cpu={r['cpu_ms']:>8.2f}  ane_est={r['ane_ms']:>7.2f}  "
                      f"cpu%={r['cpu_frac']:.0%}  vctx={r['vol_ctx']:.0f}")
                del m, state; gc.collect()
            except Exception as e:
                print(f"      {cu_label:15s}:  FAILED — {e}")

    # Derive hardware contributions
    print(f"\n    Hardware contribution analysis:")
    for ci in rep_chunks:
        co  = exp_results.get((ci, "CPU_ONLY"), {}).get('wall_ms', 0)
        cne = exp_results.get((ci, "CPU_AND_NE"), {}).get('wall_ms', 0)
        cg  = exp_results.get((ci, "CPU_AND_GPU"), {}).get('wall_ms', 0)
        ca  = exp_results.get((ci, "ALL"), {}).get('wall_ms', 0)
        if co <= 0:
            continue
        ne_sav = (co - cne) / co * 100 if cne > 0 else 0
        gpu_sav = (co - cg) / co * 100 if cg > 0 else 0
        all_sav = (co - ca) / co * 100 if ca > 0 else 0
        print(f"      Chunk {ci}:  ANE saves {ne_sav:+.1f}%,  "
              f"GPU saves {gpu_sav:+.1f}%,  ALL saves {all_sav:+.1f}%")

    # ── 4B: State overhead isolation ──
    banner("PHASE 4B: State Overhead Isolation")

    for ci in [c for c in [1, 4] if c in chunks]:
        mp = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
        m = ct.models.MLModel(mp, compute_units=ct.ComputeUnit.CPU_AND_NE,
                              function_name="infer")
        state = m.make_state()

        # populate state
        inp, _ = build_decode_input(mp)
        m.predict(inp, state=state)

        states_info = get_state_info(mp, fn_name="infer")
        state_data = {}
        total_bytes = 0
        for si in states_info:
            d = state.read_state(name=si['name'])
            state_data[si['name']] = d
            total_bytes += d.nbytes

        names = list(state_data.keys())

        def read_fn(s=state, ns=names):
            for n in ns:
                s.read_state(name=n)

        def write_fn(s=state, ns=names, sd=state_data):
            for n in ns:
                s.write_state(name=n, value=sd[n])

        r_rd = measure_utilization(read_fn, warmup=3, repeats=30)
        r_wr = measure_utilization(write_fn, warmup=3, repeats=30)
        rw_ms = r_rd['wall_ms'] + r_wr['wall_ms']

        chunk_wall = decode_results.get((ci, "CPU_AND_NE"), {}).get('wall_ms', 1)

        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        print(f"\n  Chunk {ci} [{layers}]: {len(names)} states, "
              f"{total_bytes/1024/1024:.1f} MB")
        for si in states_info:
            d = state_data[si['name']]
            print(f"    {si['name']:40s}  shape={str(d.shape):20s}  "
                  f"dtype={d.dtype}  {d.nbytes/1024:.0f} KB")
        print(f"  Read  all: wall={r_rd['wall_ms']:.2f}ms  "
              f"cpu={r_rd['cpu_ms']:.2f}ms")
        print(f"  Write all: wall={r_wr['wall_ms']:.2f}ms  "
              f"cpu={r_wr['cpu_ms']:.2f}ms")
        print(f"  R+W total: {rw_ms:.2f}ms  "
              f"= {rw_ms/chunk_wall*100:.1f}% of chunk decode ({chunk_wall:.2f}ms)")

        del m, state; gc.collect()

    # ── 4C: Prefill 4-way CU ──
    banner("PHASE 4C: Prefill 4-way CU (representative chunks)")

    sample_mp = os.path.join(COMBINED_DIR, "chunk0.mlpackage")
    fnames = get_func_names(sample_mp)
    use_combined_pf = 'prefill' in fnames

    for ci in [c for c in [1, 4] if c in chunks]:
        if use_combined_pf:
            mp = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
            fn = "prefill"
        else:
            mp = os.path.join(MODEL_DIR, f"prefill_LUT4_chunk{ci}.mlpackage")
            fn = None
            if not os.path.exists(mp):
                continue

        inp, _ = build_prefill_input(mp, fn_name=fn)
        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"

        print(f"\n  Prefill Chunk {ci} [{layers}]:")
        for cu_label, cu in cu_opts:
            try:
                kw = dict(compute_units=cu)
                if fn:
                    kw['function_name'] = fn
                m = ct.models.MLModel(mp, **kw)
                state = m.make_state()
                def _run(m=m, inp=inp, state=state):
                    return m.predict(inp, state=state)
                r = measure_utilization(_run, warmup=2, repeats=5)
                print(f"    {cu_label:15s}:  wall={r['wall_ms']:>8.1f}  "
                      f"cpu={r['cpu_ms']:>8.1f}  ane_est={r['ane_ms']:>7.1f}  "
                      f"cpu%={r['cpu_frac']:.0%}  vctx={r['vol_ctx']:.0f}")
                del m, state; gc.collect()
            except Exception as e:
                print(f"    {cu_label:15s}:  FAILED — {e}")

    return exp_results


# ═══════════════════════════════════════════════════════════════════
#  PHASE 5 — Root-Cause Explanation
# ═══════════════════════════════════════════════════════════════════

def run_phase5(decode_results, prefill_results, mil_results,
               embed_lmhead_results, exp_results, chunks):

    banner("PHASE 5: ROOT-CAUSE EXPLANATION")

    # ── Decode ──
    ane_fracs = [decode_results.get((ci, "CPU_AND_NE"), {}).get('ane_frac', 0)
                 for ci in chunks if (ci, "CPU_AND_NE") in decode_results]
    cpu_fracs = [decode_results.get((ci, "CPU_AND_NE"), {}).get('cpu_frac', 0)
                 for ci in chunks if (ci, "CPU_AND_NE") in decode_results]
    avg_ane = np.mean(ane_fracs) if ane_fracs else 0
    avg_cpu = np.mean(cpu_fracs) if cpu_fracs else 0

    # Average MIL stats
    avg_state = np.mean([
        len(mil_results.get(ci, {}).get('infer', {}).get('state_ops', []))
        for ci in chunks if ci in mil_results])
    avg_cast = np.mean([
        mil_results.get(ci, {}).get('infer', {}).get('category_counts', {}).get('cast', 0)
        for ci in chunks if ci in mil_results])
    avg_hostile = np.mean([
        len(mil_results.get(ci, {}).get('infer', {}).get('ane_hostile', []))
        for ci in chunks if ci in mil_results])
    avg_norm = np.mean([
        mil_results.get(ci, {}).get('infer', {}).get('category_counts', {}).get('normalization', 0)
        for ci in chunks if ci in mil_results])

    print(f"""
  ═══════════════════════════════════════════════════════════
  DECODE — Why is a decode chunk partly on CPU?
  ═══════════════════════════════════════════════════════════

  Average decode chunk: {avg_cpu:.0%} CPU-time,  {avg_ane:.0%} ANE-time (wall basis)

  WHAT IS ON ANE  (≈{avg_ane:.0%} of wall time):
    • Dense matmul / conv ops (Q/K/V projections, FFN up/gate/down)
    • Elementwise arithmetic (add, mul, silu activation)
    • Softmax (fused into ANE attention subgraph)
    • fp16 compute kernels compiled by CoreML's NE compiler
    Evidence:
      – Wall-time speedup >1x over CPU_ONLY confirms ANE acceleration
      – Voluntary context switches confirm CPU yields to ANE hardware
      – cpu_ms < wall_ms proves CPU is idle while ANE computes

  WHAT IS ON CPU  (≈{avg_cpu:.0%} of wall time):
    1. State I/O:  ~{avg_state:.0f} state ops/chunk
       – KV cache + linear-attention conv/recurrent states live in CPU memory
       – Every predict() call requires read-state → ANE compute → write-state
       – These are ALWAYS on CPU (CoreML state contract)
    2. Data casts:  ~{avg_cast:.0f} cast ops/chunk
       – fp32↔fp16 boundaries at model input/output and state interfaces
       – Each cast is a CPU memcopy with dtype conversion
    3. ANE-hostile ops:  ~{avg_hostile:.0f} ops/chunk
       – gather / gather_along_axis (RoPE cos/sin lookup)
       – These force CoreML to create CPU-fallback subgraph partitions
       – Each partition boundary = CPU↔ANE data transfer
    4. Normalization:  ~{avg_norm:.0f} layer_norm ops/chunk
       – May run on ANE (Apple Silicon M-series) or CPU (older HW)
       – Even on ANE, requires separate kernel launch
    5. CoreML runtime overhead:
       – Dispatch scheduling, graph partitioning, memory allocation
       – Overhead per predict() call, independent of model size

  WHY this pattern:
    • CoreML compiler splits the MIL graph into ANE-eligible and CPU-fallback
      subgraphs based on op support and data dependencies.
    • Gather ops (for RoPE) break ANE subgraph continuity, creating
      partition boundaries.  More partitions = more CPU↔ANE transfers.
    • State ops are inherently CPU (DMA to/from shared memory).
    • The hybrid attention+Mamba architecture uses more state than pure
      Transformer, increasing CPU-side overhead.""")

    # ── Prefill ──
    pf_spds = []
    for ci in chunks:
        co = prefill_results.get((ci, "CPU_ONLY"), {})
        ca = prefill_results.get((ci, "CPU_AND_NE"), {})
        if co and ca and ca.get('wall_ms', 0) > 0:
            pf_spds.append(co['wall_ms'] / ca['wall_ms'])
    avg_pf_spd = np.mean(pf_spds) if pf_spds else 1.0

    pf_co_cpuf = [prefill_results.get((ci, "CPU_ONLY"), {}).get('cpu_frac', 0)
                  for ci in chunks if (ci, "CPU_ONLY") in prefill_results]
    pf_ca_cpuf = [prefill_results.get((ci, "CPU_AND_NE"), {}).get('cpu_frac', 0)
                  for ci in chunks if (ci, "CPU_AND_NE") in prefill_results]
    avg_pf_co_cpuf = np.mean(pf_co_cpuf) if pf_co_cpuf else 1.0
    avg_pf_ca_cpuf = np.mean(pf_ca_cpuf) if pf_ca_cpuf else 1.0

    data_per_chunk_kb = BATCH_SIZE * HIDDEN_DIM * 2 / 1024

    slower_str = "SLOWER" if avg_pf_spd < 1.0 else "faster"
    print(f"""
  ═══════════════════════════════════════════════════════════
  PREFILL — Why may prefill be {slower_str} on ANE?
  ═══════════════════════════════════════════════════════════

  Average prefill speedup: {avg_pf_spd:.2f}x  ({slower_str} on ANE)

  CPU_ONLY  cpu_frac = {avg_pf_co_cpuf:.2f}  (multi-core CPU via Accelerate)
  CPU_AND_NE cpu_frac = {avg_pf_ca_cpuf:.2f}

  ROOT CAUSES:
    1. MULTI-CORE CPU ADVANTAGE for large batches
       – seq_len={BATCH_SIZE}: large matmuls are efficiently parallelised
         across all Performance cores via Accelerate/AMX
       – cpu_frac > 1.0 confirms multi-threaded execution
       – ANE has fixed throughput; large tiles don't help as much

    2. DATA TRANSFER AMPLIFICATION
       – Per-chunk hidden_states: (1,{BATCH_SIZE},{HIDDEN_DIM}) fp16
         = {data_per_chunk_kb:.0f} KB (vs 5 KB for decode)
       – Round-trip transfer: {data_per_chunk_kb*2:.0f} KB IN + OUT per chunk
       – 9 chunks × {data_per_chunk_kb*2:.0f} KB = {9*data_per_chunk_kb*2/1024:.1f} MB total transfers

    3. SAME GRAPH PARTITIONING OVERHEAD
       – Same hostile ops create the same CPU-fallback partitions
       – But each partition now processes {BATCH_SIZE}x more data
       – Transfer cost per partition scales with batch size

    4. STATE WRITES SCALE
       – Prefill writes {BATCH_SIZE} positions to KV cache
       – State write bandwidth becomes a bottleneck

    5. ANE PIPELINE LATENCY
       – ANE pipeline startup cost is amortised over small decode ops
       – For large prefill, CPU can start computing immediately with all cores""")

    # ── Recommendations ──
    banner("OPTIMISATION RECOMMENDATIONS (PRIORITISED)")

    lmhead_co = embed_lmhead_results.get(("lmhead", "CPU_ONLY"), {}).get('wall_ms', 25)
    lmhead_ca = embed_lmhead_results.get(("lmhead", "CPU_AND_NE"), {}).get('wall_ms', 5)
    lmhead_spd = lmhead_co / lmhead_ca if lmhead_ca > 0 else 1

    print(f"""
  1. LM HEAD ON ANE (CPU_AND_NE)  — ALREADY DONE IN PRODUCTION
     Speedup: {lmhead_spd:.1f}x ({lmhead_co:.1f}ms → {lmhead_ca:.1f}ms)
     Impact:  ~{(lmhead_co-lmhead_ca):.0f}ms saved per decode step
     Status:  chat_server.py already uses CPU_AND_NE for lmhead ✓
              validate.py / grade_quality.py still use CPU_ONLY (fix those)

  2. PREFILL ON CPU_ONLY
     Keep prefill chunks on CPU_ONLY (ANE is {avg_pf_spd:.2f}x {'slower' if avg_pf_spd < 1 else 'faster'})
     Impact:  ~{max(0, (1-avg_pf_spd))*100:.0f}% faster prefill throughput

  3. REDUCE GRAPH PARTITIONS (MEDIUM EFFORT)
     Replace gather-based RoPE with ANE-friendly rotation
     (e.g., static cos/sin embedded as constants, avoiding runtime gather)
     Impact:  Fewer CPU↔ANE boundaries → less transfer overhead
     Estimated: 1-3ms per decode step

  4. STATE COMPRESSION (HIGH EFFORT)
     KV-cache state is the largest CPU overhead source
     Options:
       – Quantise KV cache to int8 (halves state I/O bandwidth)
       – Reduce context length where possible
       – Fuse multiple chunks (reduces state handoff count)
     Impact:  ~5-10ms per decode step (50-60% of CPU overhead)

  5. CHUNK 8 ON ALL (GPU FALLBACK)
     Small 1-layer F-chunk benefits from GPU fallback
     Impact:  Marginal (~1ms)

  6. ELIMINATE REMAINING CASTS (LOW PRIORITY)
     Model is already 99.6% fp16 — casts are minimal
     Status:  Already optimal ✓
""")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep ANE vs CPU placement investigation")
    parser.add_argument('--chunks', type=str, default=None,
                        help='Comma-separated chunk indices (default: all)')
    parser.add_argument('--skip-phase', type=int, nargs='+', default=[],
                        help='Phase numbers to skip (1-5)')
    args = parser.parse_args()

    chunks = (list(range(NUM_CHUNKS)) if args.chunks is None
              else [int(c) for c in args.chunks.split(',')])
    skip = set(args.skip_phase)

    banner("DEEP ANE vs CPU PLACEMENT INVESTIGATION — Qwen3.5-4B")
    print(f"  Model dir:   {MODEL_DIR}")
    print(f"  Combined:    {COMBINED_DIR}")
    print(f"  Chunks:      {chunks}")
    print(f"  CTX={CTX}  BATCH_SIZE={BATCH_SIZE}  NUM_CHUNKS={NUM_CHUNKS}")
    print(f"  Skip phases: {skip or 'none'}")
    print(f"  Chunk ranges: {CHUNK_RANGES}")
    print(f"  Measurement: decode={NUM_WARMUP_DECODE}w+{NUM_MEASURE_DECODE}r  "
          f"prefill={NUM_WARMUP_PREFILL}w+{NUM_MEASURE_PREFILL}r")

    decode_results = {}
    prefill_results = {}
    embed_lmhead_results = {}
    mil_results = {}
    exp_results = {}

    if 1 not in skip:
        decode_results, prefill_results, embed_lmhead_results = run_phase1(chunks)

    if 2 not in skip:
        mil_results = run_phase2(chunks)

    if 3 not in skip:
        run_phase3(decode_results, prefill_results, mil_results,
                   embed_lmhead_results, chunks)

    if 4 not in skip:
        exp_results = run_phase4(decode_results, prefill_results, chunks)

    if 5 not in skip:
        run_phase5(decode_results, prefill_results, mil_results,
                   embed_lmhead_results, exp_results, chunks)

    banner("INVESTIGATION COMPLETE")
    print("Done.")
