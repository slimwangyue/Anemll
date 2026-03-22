#!/usr/bin/env python3
"""Comprehensive ANE profiling for Qwen3.5-4B — Milestone 2.0 (batch prefill + valid_len).

Profiles ALL model components (embed, lm_head, 4 decode FFN, 4 prefill FFN)
with focus on ANE performance, dynamic KV cache behavior, and throughput.

Sections:
  1. MIL Operation Analysis — ANE vs CPU op classification per component
  2. Model Loading & ANE Compilation — load time per component
  3. CPU+GPU vs CPU+ANE Speedup — per-component latency comparison
  4. Decode Profiling — per-step timing at varying positions (early/mid/late)
  5. Prefill vs Decode Throughput — batch prefill vs single-token decode
  6. Multi-Turn Conversation — full pipeline with per-turn breakdown
  7. KV Cache Position Sweep — timing at pos 0,10,50,100,200,500,900
  8. Summary Dashboard

Usage:
    python scripts_qwen3_5/profile.py
    python scripts_qwen3_5/profile.py --tokens 40 --skip-cpu-compare
    python scripts_qwen3_5/profile.py --export-dir /path/to/models --quick
"""
import sys, os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)  # must be first for config.py

import gc, time, argparse
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
from config import DEFAULT_OUTPUT, DEFAULT_HF_MODEL, CTX, NUM_CHUNKS, BATCH_SIZE

MODEL_PATH = DEFAULT_HF_MODEL
EXPORT_DIR = DEFAULT_OUTPUT

CONVERSATION_TURNS = [
    "What is a stack in computer science?",
    "How does it compare to a queue?",
    "Give me a Python example of each.",
]

# ── ANE op classification ────────────────────────────────────────────

ANE_OPS = {
    'conv', 'conv_transpose', 'linear', 'matmul', 'einsum',
    'add', 'mul', 'sub', 'real_div', 'floor_div', 'mod',
    'pow', 'sqrt', 'rsqrt', 'abs', 'neg', 'exp', 'log',
    'ceil', 'floor', 'round', 'clip', 'sign',
    'maximum', 'minimum',
    'relu', 'gelu', 'sigmoid', 'tanh', 'silu', 'leaky_relu',
    'elu', 'prelu', 'softplus',
    'layer_norm', 'instance_norm', 'batch_norm', 'l2_norm',
    'softmax', 'log_softmax',
    'reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_min', 'reduce_prod',
    'cumsum',
    'concat', 'split', 'stack', 'reshape', 'transpose',
    'expand_dims', 'squeeze', 'reverse',
    'slice_by_index', 'slice_by_size', 'pad',
    'tile', 'repeat',
    'gather', 'gather_along_axis', 'gather_nd',
    'select', 'where',
    'greater', 'greater_equal', 'less', 'less_equal',
    'equal', 'not_equal',
    'avg_pool', 'max_pool', 'l2_pool',
}

CPU_OPS = {
    'scatter', 'scatter_nd', 'scatter_along_axis',
    'slice_update',
    'topk', 'argsort', 'argmax', 'argmin',
    'one_hot', 'cast',
    'fill', 'fill_like',
    'while_loop', 'cond',
    'non_maximum_suppression',
    'shape', 'rank', 'range_1d', 'identity',
}

STATE_OPS = {'read_state', 'write_state'}

CONST_OPS = {
    'const',
    'constexpr_lut_to_dense',
    'constexpr_affine_dequantize',
    'constexpr_blockwise_shift_scale',
    'constexpr_sparse_to_dense',
    'constexpr_cast',
}


# ── Helpers ──────────────────────────────────────────────────────────

def _count_ops_in_block(block, op_counts):
    for op in block.operations:
        op_counts[op.type] = op_counts.get(op.type, 0) + 1
        for nested_block in op.blocks:
            _count_ops_in_block(nested_block, op_counts)


def analyze_model(path, label):
    spec = ct.utils.load_spec(path)
    op_counts = {}
    if spec.HasField('mlProgram'):
        prog = spec.mlProgram
        for fn_name in prog.functions:
            fn = prog.functions[fn_name]
            for block_name in fn.block_specializations:
                block = fn.block_specializations[block_name]
                _count_ops_in_block(block, op_counts)

    ane_count = sum(c for t, c in op_counts.items() if t in ANE_OPS)
    cpu_count = sum(c for t, c in op_counts.items() if t in CPU_OPS)
    state_count = sum(c for t, c in op_counts.items() if t in STATE_OPS)
    const_count = sum(c for t, c in op_counts.items() if t in CONST_OPS)
    unknown = {t: c for t, c in op_counts.items()
               if t not in ANE_OPS and t not in CPU_OPS
               and t not in STATE_OPS and t not in CONST_OPS}
    runtime_total = ane_count + cpu_count + state_count + sum(unknown.values())

    return {
        'label': label, 'op_counts': op_counts,
        'ane': ane_count, 'cpu': cpu_count,
        'state': state_count, 'const': const_count,
        'unknown': unknown, 'runtime_total': runtime_total,
    }


def timing_stats(times_sec):
    a = np.array(times_sec) * 1000
    if len(a) == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p95': 0, 'total': 0, 'n': 0}
    return {
        'mean': float(np.mean(a)), 'std': float(np.std(a)),
        'min': float(np.min(a)), 'max': float(np.max(a)),
        'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
        'total': float(np.sum(a)), 'n': len(a),
    }


def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def _ensure_ids(tpl_out):
    import torch
    if hasattr(tpl_out, 'input_ids'):
        ids = tpl_out.input_ids
    elif isinstance(tpl_out, torch.Tensor):
        ids = tpl_out
    else:
        ids = torch.tensor(tpl_out)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.to(torch.int32)


def _build_stop_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)
    return stop_ids


# ── Model loading helpers (same as chat_server.py) ──────────────────

def _load_model(path, compute_unit, function_name=None):
    """Load a CoreML model from .mlpackage or .mlmodelc."""
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, compute_unit)
    kwargs = {"compute_units": compute_unit}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def _find_model(base_dir, name):
    """Find model path, preferring .mlmodelc over .mlpackage."""
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


# ── Profiling Engine ─────────────────────────────────────────────────

class ProfileEngine:
    def __init__(self, out_dir, ffn_label, compute_unit, name="engine"):
        self.name = name
        self.out_dir = out_dir
        self.cu = compute_unit
        self.load_times = {}
        self.ane_failures = []

        # Detect combined dedup directory (same as chat_server.py)
        self.combined_dir = os.path.join(out_dir, "combined_LUT4_dedup")
        self.use_combined = os.path.isdir(self.combined_dir)

        # --- Embeddings ---
        t0 = time.time()
        self.embed = _load_model(_find_model(out_dir, "embeddings"), compute_unit)
        self.load_times['embed'] = time.time() - t0

        # --- LM Head (prefer logits, same as chat_server.py) ---
        t0 = time.time()
        try:
            lmhead_path = _find_model(out_dir, "lm_head_logits")
            self.lmhead = _load_model(lmhead_path, compute_unit)
            self.lmhead_mode = "logits"
            print(f"  Loaded logits lm_head (penalties enabled)")
        except FileNotFoundError:
            self.lmhead = _load_model(
                _find_model(out_dir, "lm_head"), compute_unit)
            spec = self.lmhead.get_spec()
            out_names = [o.name for o in spec.description.output]
            self.lmhead_mode = "logits" if ("logits" in out_names
                                            or "output_logits" in out_names
                                            ) else "argmax"
            print(f"  Loaded lm_head (mode={self.lmhead_mode})")
        if self.lmhead_mode == "logits":
            spec = self.lmhead.get_spec()
            out_names = [o.name for o in spec.description.output]
            self.logits_key = ("output_logits" if "output_logits" in out_names
                              else "logits")
        self.load_times['lmhead'] = time.time() - t0

        # --- FFN chunks (infer + prefill, same as chat_server.py) ---
        self.ffns = []
        self.prefills = []
        self.has_prefill = False

        for ci in range(NUM_CHUNKS):
            # --- infer instance ---
            if self.use_combined:
                path = _find_model(self.combined_dir, f"chunk{ci}")
                if path.endswith(".mlmodelc"):
                    self.use_combined = False
            if self.use_combined:
                print(f"  chunk {ci} infer  (combined)...", end="", flush=True)
                t0 = time.time()
                m_infer = _load_model(path, compute_unit, function_name="infer")
                print(f" {time.time()-t0:.0f}s")
            else:
                path = _find_model(out_dir, f"ffn_LUT4_chunk{ci}")
                print(f"  chunk {ci} infer  (separate)...", end="", flush=True)
                t0 = time.time()
                m_infer = _load_model(path, compute_unit)
                print(f" {time.time()-t0:.0f}s")
            self.ffns.append(m_infer)
            self.load_times[f'ffn_chunk{ci}'] = time.time() - t0

            # --- prefill instance ---
            m_prefill = None
            if self.use_combined:
                print(f"  chunk {ci} prefill (combined)...", end="", flush=True)
                t0 = time.time()
                m_prefill = _load_model(path, compute_unit, function_name="prefill")
                print(f" {time.time()-t0:.0f}s")
            else:
                try:
                    pf_path = _find_model(out_dir, f"prefill_LUT4_chunk{ci}")
                    print(f"  chunk {ci} prefill (separate)...", end="", flush=True)
                    t0 = time.time()
                    m_prefill = _load_model(pf_path, compute_unit)
                    print(f" {time.time()-t0:.0f}s")
                except FileNotFoundError:
                    print(f"  chunk {ci} prefill — not found, batch disabled")
            if m_prefill is not None:
                self.load_times[f'prefill_chunk{ci}'] = time.time() - t0
            self.prefills.append(m_prefill)

        self.has_prefill = all(p is not None for p in self.prefills)

        spec = self.ffns[0].get_spec()
        # Detect input shapes (same as chat_server.py _detect_shapes)
        self.inp_map = {}
        try:
            spec = self.ffns[0].get_spec()
            fn_inputs = None
            if self.use_combined and hasattr(spec.description, 'functions'):
                for fn in spec.description.functions:
                    if fn.name == "infer":
                        fn_inputs = fn.input
                        break
            if fn_inputs is None:
                fn_inputs = spec.description.input
            for inp in fn_inputs:
                try:
                    self.inp_map[inp.name] = tuple(
                        inp.type.multiArrayType.shape)
                except Exception:
                    pass
        except Exception:
            print("  Using default input shapes")
            self.inp_map = {
                'linear_conv_state': (8, 1024, 32),
                'linear_recurrent_state': (8, 32, 128, 128),
            }
        self.reset_all()

    def reset_all(self):
        self.states = [m.make_state() for m in self.ffns]
        self.prefill_states = [m.make_state() for m in self.prefills
                               if m is not None] if self.prefills else []
        self.lin_convs = [
            np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
            for _ in range(NUM_CHUNKS)]
        self.lin_recs = [
            np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
            for _ in range(NUM_CHUNKS)]
        # Pre-allocate reusable buffers (same as chat_server.py)
        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)
        self.clear_timing()

    def clear_timing(self):
        self.t_embed = []
        self.t_ffn = [[] for _ in range(NUM_CHUNKS)]
        self.t_lmhead = []
        self.t_step = []

    def _step(self, tok_id, pos):
        t_step = time.perf_counter()

        tok = self._tok_buf
        tok[0, 0] = tok_id
        t0 = time.perf_counter()
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        self.t_embed.append(time.perf_counter() - t0)

        mask = self._mask_buf
        mask[:, :, :, :] = -65504.0
        mask[:, :, :, :pos + 1] = 0

        pos_arr = self._pos_buf
        pos_arr[0] = pos
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            t0 = time.perf_counter()
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            self.t_ffn[ci].append(time.perf_counter() - t0)
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        t0 = time.perf_counter()
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        self.t_lmhead.append(time.perf_counter() - t0)

        self.t_step.append(time.perf_counter() - t_step)

        if self.lmhead_mode == "logits":
            return int(np.argmax(lm_out[self.logits_key].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    def prefill_and_decode(self, token_ids, start_pos, max_gen, stop_ids):
        for i, tid in enumerate(token_ids):
            pos = start_pos + i
            if pos >= CTX:
                break
            last_next = self._step(tid, pos)
        prefill_end_pos = start_pos + len(token_ids)
        tokens = [last_next]
        for gi in range(max_gen - 1):
            pos = prefill_end_pos + gi
            if pos >= CTX - 1:
                break
            next_id = self._step(tokens[-1], pos)
            tokens.append(next_id)
            if next_id in stop_ids:
                break
        return tokens, prefill_end_pos + len(tokens)

    def get_decode_report(self, skip_first_n=0):
        """Get timing report, optionally skipping first N steps (prefill)."""
        s = skip_first_n
        report = {
            'embed': timing_stats(self.t_embed[s:]),
            'lmhead': timing_stats(self.t_lmhead[s:]),
            'step': timing_stats(self.t_step[s:]),
            'ffn_chunks': [timing_stats(self.t_ffn[ci][s:]) for ci in range(NUM_CHUNKS)],
        }
        n_steps = len(self.t_step[s:])
        ffn_per_step = []
        for si in range(n_steps):
            idx = s + si
            ffn_sum = sum(self.t_ffn[ci][idx] for ci in range(NUM_CHUNKS)
                         if idx < len(self.t_ffn[ci]))
            ffn_per_step.append(ffn_sum)
        report['ffn_per_step'] = timing_stats(ffn_per_step)
        report['n_steps'] = n_steps
        return report

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        self.ffns = []
        for m in self.prefills:
            del m
        self.prefills = []
        gc.collect()


def print_section(title, num):
    print(f"\n{'='*85}")
    print(f"  SECTION {num}: {title}")
    print(f"{'='*85}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive ANE profiling for Qwen3.5-4B")
    parser.add_argument("--tokens", type=int, default=40, help="Max tokens per turn")
    parser.add_argument("--export-dir", type=str, default=EXPORT_DIR)
    parser.add_argument("--skip-cpu-compare", action="store_true",
                        help="Skip CPU+GPU vs CPU+ANE comparison (saves time)")
    parser.add_argument("--compare-steps", type=int, default=15,
                        help="Decode steps for CPU vs ANE comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Skip sections 3,5,7 for faster profiling")
    args = parser.parse_args()

    max_gen = args.tokens
    out_dir = args.export_dir
    ffn_label = "LUT4"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)

    print("=" * 85)
    print("  ANE PROFILING — Qwen3.5-4B Milestone 2.0 (Batch Prefill)")
    print(f"  Dynamic KV cache: current_pos[0] → aten::select → ANE-safe")
    print(f"  Config: LUT4 embed + LUT4 FFN × {NUM_CHUNKS} chunks + fp16 LM Head")
    print(f"  CTX={CTX}, BATCH={BATCH_SIZE}, Tokens/turn={max_gen}")
    print(f"  Models: {out_dir}")
    print("=" * 85)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: MIL Operation Analysis
    # ══════════════════════════════════════════════════════════════════
    print_section("MIL Operation Analysis", 1)

    # Discover model files the same way as chat_server.py
    combined_dir = os.path.join(out_dir, "combined_LUT4_dedup")
    use_combined = os.path.isdir(combined_dir)

    model_specs = []
    # Embeddings
    try:
        model_specs.append((_find_model(out_dir, "embeddings"), "Embeddings (LUT4)"))
    except FileNotFoundError:
        pass
    # LM Head (prefer logits, same as chat_server.py)
    try:
        model_specs.append((_find_model(out_dir, "lm_head_logits"), "LM Head logits"))
    except FileNotFoundError:
        try:
            model_specs.append((_find_model(out_dir, "lm_head"), "LM Head (fp16)"))
        except FileNotFoundError:
            pass
    # FFN chunks (combined or separate)
    if use_combined:
        for ci in range(NUM_CHUNKS):
            try:
                model_specs.append((_find_model(combined_dir, f"chunk{ci}"),
                                    f"FFN combined chunk{ci}"))
            except FileNotFoundError:
                pass
    else:
        for ci in range(NUM_CHUNKS):
            try:
                model_specs.append((_find_model(out_dir, f"ffn_{ffn_label}_chunk{ci}"),
                                    f"FFN decode chunk{ci}"))
            except FileNotFoundError:
                pass
        for ci in range(NUM_CHUNKS):
            try:
                model_specs.append((_find_model(out_dir, f"prefill_{ffn_label}_chunk{ci}"),
                                    f"FFN prefill chunk{ci}"))
            except FileNotFoundError:
                pass

    print(f"  Model loading mode: {'COMBINED' if use_combined else 'SEPARATE'}")

    analyses = []
    for path, label in model_specs:
        if os.path.exists(path):
            a = analyze_model(path, label)
            analyses.append(a)

    print(f"\n  {'Component':<25} {'Runtime':>8} {'ANE':>6} {'CPU':>6} {'State':>6} {'Const':>6} {'ANE%':>7}")
    print(f"  {'-'*68}")
    total_ane = total_cpu = total_state = total_runtime = 0
    for a in analyses:
        ane_pct = 100 * a['ane'] / a['runtime_total'] if a['runtime_total'] > 0 else 0
        print(f"  {a['label']:<25} {a['runtime_total']:>8} {a['ane']:>6} {a['cpu']:>6} "
              f"{a['state']:>6} {a['const']:>6} {ane_pct:>6.1f}%")
        total_ane += a['ane']
        total_cpu += a['cpu']
        total_state += a['state']
        total_runtime += a['runtime_total']
    print(f"  {'-'*68}")
    total_ane_pct = 100 * total_ane / total_runtime if total_runtime > 0 else 0
    print(f"  {'TOTAL':<25} {total_runtime:>8} {total_ane:>6} {total_cpu:>6} {total_state:>6} "
          f"{'':>6} {total_ane_pct:>6.1f}%")

    # CPU-bound ops detail for one FFN chunk
    ffn_analyses = [a for a in analyses if 'chunk0' in a['label']]
    if ffn_analyses:
        ffn_a = ffn_analyses[0]
        cpu_ops = {t: c for t, c in ffn_a['op_counts'].items() if t in CPU_OPS}
        if cpu_ops:
            print(f"\n  CPU-fallback ops in FFN decode chunk0:")
            for op_type, count in sorted(cpu_ops.items(), key=lambda x: -x[1]):
                print(f"    {op_type:<25} {count:>5}")
            print(f"    {'TOTAL':<25} {sum(cpu_ops.values()):>5}  "
                  f"(of {ffn_a['runtime_total']} runtime ops)")

    # State ops (KV cache read/write)
    state_ops = {t: c for t, c in ffn_a['op_counts'].items() if t in STATE_OPS}
    if state_ops:
        print(f"\n  State I/O ops (KV cache) in FFN decode chunk0:")
        for op_type, count in sorted(state_ops.items(), key=lambda x: -x[1]):
            print(f"    {op_type:<25} {count:>5}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: Model Loading & ANE Compilation
    # ══════════════════════════════════════════════════════════════════
    print_section("Model Loading & ANE Compilation", 2)

    print(f"\n  Loading all models with CPU_AND_NE...")
    t_total_load = time.time()
    engine = ProfileEngine(out_dir, ffn_label, ct.ComputeUnit.CPU_AND_NE,
                           name="ANE")
    t_total_load = time.time() - t_total_load

    print(f"\n  {'Component':<22} {'Load Time (s)':>14}")
    print(f"  {'-'*38}")
    for comp, lt in engine.load_times.items():
        print(f"  {comp:<22} {lt:>14.1f}")
    print(f"  {'-'*38}")
    print(f"  {'TOTAL':<22} {t_total_load:>14.1f}")

    # Model sizes
    print(f"\n  {'Component':<22} {'Size (MB)':>10}")
    print(f"  {'-'*34}")
    total_sz = 0
    for path, label in model_specs:
        if os.path.exists(path):
            sz = dir_size_mb(path)
            total_sz += sz
            print(f"  {label:<22} {sz:>10.1f}")
    print(f"  {'-'*34}")
    print(f"  {'TOTAL':<22} {total_sz:>10.1f}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: CPU+GPU vs CPU+ANE Speedup
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_cpu_compare:
        print_section("CPU+GPU vs CPU+ANE Speedup", 3)

        bench_prompt = "What is a stack?"
        bench_ids = _ensure_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": bench_prompt}],
            return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True))
        bench_tokens = bench_ids[0].tolist()
        n_compare = args.compare_steps

        # Warmup ANE engine (already loaded)
        for i in range(min(3, len(bench_tokens))):
            engine._step(bench_tokens[i], i)
        engine.reset_all()

        # Benchmark ANE
        print(f"\n  Running {len(bench_tokens)} prefill + {n_compare} decode (CPU+ANE)...")
        engine.prefill_and_decode(bench_tokens, 0, n_compare, stop_ids)
        ane_report = engine.get_decode_report(skip_first_n=len(bench_tokens))
        ane_full_report = engine.get_decode_report()
        engine.reset_all()
        engine.clear_timing()

        # Load CPU+GPU engine
        print(f"  Loading models with CPU_AND_GPU (no ANE)...")
        t0 = time.time()
        engine_gpu = ProfileEngine(out_dir, ffn_label, ct.ComputeUnit.CPU_AND_GPU,
                                   name="GPU")
        print(f"  Loaded in {time.time()-t0:.0f}s")

        # Warmup
        for i in range(min(3, len(bench_tokens))):
            engine_gpu._step(bench_tokens[i], i)
        engine_gpu.reset_all()

        # Benchmark GPU
        print(f"  Running {len(bench_tokens)} prefill + {n_compare} decode (CPU+GPU)...")
        engine_gpu.prefill_and_decode(bench_tokens, 0, n_compare, stop_ids)
        gpu_report = engine_gpu.get_decode_report(skip_first_n=len(bench_tokens))
        engine_gpu.cleanup()

        print(f"\n  Decode-only comparison ({n_compare} tokens):")
        print(f"  {'Component':<22} {'CPU+GPU':>12} {'CPU+ANE':>12} {'Speedup':>10} {'ANE?':>6}")
        print(f"  {'-'*64}")
        comparisons = [
            ("Embed", gpu_report['embed'], ane_report['embed']),
            ("FFN (4 chunks)", gpu_report['ffn_per_step'], ane_report['ffn_per_step']),
        ]
        for ci in range(NUM_CHUNKS):
            comparisons.append((f"  FFN chunk{ci}",
                gpu_report['ffn_chunks'][ci], ane_report['ffn_chunks'][ci]))
        comparisons.append(("LM Head", gpu_report['lmhead'], ane_report['lmhead']))
        comparisons.append(("Full step", gpu_report['step'], ane_report['step']))

        for label, gs, ans in comparisons:
            gpu_ms = gs['mean']
            ane_ms = ans['mean']
            speedup = gpu_ms / ane_ms if ane_ms > 0 else float('inf')
            is_ane = "YES" if speedup > 1.2 else ("maybe" if speedup > 1.05 else "no")
            print(f"  {label:<22} {gpu_ms:>10.2f}ms {ane_ms:>10.2f}ms {speedup:>9.2f}x {is_ane:>6}")

        gpu_step = gpu_report['step']['mean']
        ane_step = ane_report['step']['mean']
        total_speedup = gpu_step / ane_step if ane_step > 0 else 0
        print(f"\n  ANE acceleration: {total_speedup:.2f}x overall")
        print(f"  Throughput: CPU+GPU={1000/gpu_step:.1f} tok/s, CPU+ANE={1000/ane_step:.1f} tok/s")
    else:
        gpu_step = ane_step = total_speedup = 0

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4: Decode Timing at Varying KV Cache Positions
    # ══════════════════════════════════════════════════════════════════
    if not args.quick:
        print_section("KV Cache Position Sweep (Decode Latency)", 4)

        test_positions = [0, 5, 10, 50, 100, 200]
        test_positions = [p for p in test_positions if p < CTX - 5]

        print(f"\n  Feeding tokens to build KV cache, measuring step latency at key positions...")
        engine.reset_all()
        engine.clear_timing()

        pos_timings = {}
        # Feed tokens up to max position, timing each step
        max_pos = max(test_positions) + 3
        for pos in range(max_pos + 1):
            engine._step(1 + (pos % 100), pos)

        # Extract timings at target positions
        print(f"\n  {'Position':>10} {'Step (ms)':>10} {'FFN total':>10} {'Embed':>10} {'LM Head':>10}")
        print(f"  {'-'*52}")
        for target_pos in test_positions:
            if target_pos < len(engine.t_step):
                step_ms = engine.t_step[target_pos] * 1000
                ffn_ms = sum(engine.t_ffn[ci][target_pos] * 1000 for ci in range(NUM_CHUNKS))
                embed_ms = engine.t_embed[target_pos] * 1000
                lmhead_ms = engine.t_lmhead[target_pos] * 1000
                print(f"  {target_pos:>10} {step_ms:>10.2f} {ffn_ms:>10.2f} "
                      f"{embed_ms:>10.2f} {lmhead_ms:>10.2f}")
                pos_timings[target_pos] = step_ms

        # Check for position-dependent slowdown
        if len(pos_timings) >= 2:
            first = list(pos_timings.values())[0]
            last = list(pos_timings.values())[-1]
            ratio = last / first if first > 0 else 0
            print(f"\n  Position-dependent slowdown: {ratio:.2f}x (pos {list(pos_timings.keys())[-1]} vs pos {list(pos_timings.keys())[0]})")
            if ratio < 1.1:
                print(f"  Result: No significant slowdown — KV cache position does NOT affect latency")
            elif ratio < 1.3:
                print(f"  Result: Slight slowdown at high positions")
            else:
                print(f"  Result: Significant position-dependent slowdown — investigate")

        engine.reset_all()
        engine.clear_timing()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 5: Multi-Turn Conversation Profiling
    # ══════════════════════════════════════════════════════════════════
    print_section("Multi-Turn Conversation (CPU+ANE)", 5)

    # Warmup
    engine._step(1, 0)
    engine.reset_all()
    engine.clear_timing()

    conversation = []
    turn_reports = []

    for ti, user_msg in enumerate(CONVERSATION_TURNS):
        print(f"\n  Turn {ti+1}: \"{user_msg}\"")
        conversation.append({"role": "user", "content": user_msg})
        input_ids = _ensure_ids(tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True))
        prompt_len = input_ids.shape[1]
        if prompt_len + max_gen > CTX:
            input_ids = input_ids[:, -(CTX - max_gen):]
            prompt_len = input_ids.shape[1]

        engine.reset_all()
        engine.clear_timing()
        token_list = input_ids[0].tolist()

        t_turn = time.time()
        gen_tokens, end_pos = engine.prefill_and_decode(
            token_list, 0, max_gen, stop_ids)
        turn_time = time.time() - t_turn

        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": "<think>\n" + raw_text})

        # Separate prefill vs decode timing
        decode_report = engine.get_decode_report(skip_first_n=prompt_len)
        full_report = engine.get_decode_report()
        decode_steps = decode_report['n_steps']

        turn_reports.append({
            'turn': ti + 1, 'prompt_len': prompt_len,
            'decode_steps': decode_steps, 'turn_time': turn_time,
            'text': raw_text, 'gen_tokens': gen_tokens,
            'decode_report': decode_report, 'full_report': full_report,
        })

        d = decode_report
        step_ms = d['step']['mean']
        tok_per_sec = 1000 / step_ms if step_ms > 0 else 0

        print(f"    Prompt: {prompt_len} tok, Decode: {decode_steps} tok")
        print(f"    Turn time: {turn_time:.1f}s, Decode only: {d['step']['total']:.0f}ms")
        print(f"    Decode throughput: {tok_per_sec:.1f} tok/s ({step_ms:.1f} ms/tok)")
        print(f"    Text: {raw_text[:120]}")

        print(f"\n    {'Component':<22} {'Mean(ms)':>9} {'Std':>7} {'P50':>7} {'P95':>7} {'% step':>8}")
        print(f"    {'-'*62}")
        components = [
            ("Embed", d['embed']),
            ("FFN (4 chunks)", d['ffn_per_step']),
        ]
        for ci in range(NUM_CHUNKS):
            components.append((f"  chunk{ci}", d['ffn_chunks'][ci]))
        components.append(("LM Head", d['lmhead']))
        components.append(("Full step", d['step']))
        for clabel, cs in components:
            pct = 100 * cs['mean'] / d['step']['mean'] if d['step']['mean'] > 0 else 0
            if clabel == "Full step":
                print(f"    {'-'*62}")
            print(f"    {clabel:<22} {cs['mean']:>9.2f} {cs['std']:>7.2f} {cs['p50']:>7.2f} "
                  f"{cs['p95']:>7.2f} {pct:>7.1f}%")

    engine.cleanup()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 6: Summary Dashboard
    # ══════════════════════════════════════════════════════════════════
    print_section("Summary Dashboard", 6)

    # Model info
    print(f"\n  Architecture: Qwen3.5-4B (24 linear-attn + 8 full-attn layers)")
    print(f"  KV Indexing: tensor-value slice — current_pos[0] (Milestone 2.0)")
    print(f"  Quantization: LUT4 embed + LUT4 FFN + fp16 LM Head")
    print(f"  Context: {CTX} tokens, Batch: {BATCH_SIZE}, Chunks: {NUM_CHUNKS}")
    print(f"  Total model size: {total_sz:.0f} MB")

    # ANE utilization
    print(f"\n  ANE Operation Utilization:")
    print(f"    Total runtime ops:  {total_runtime:>8}")
    print(f"    ANE-eligible:       {total_ane:>8}  ({total_ane_pct:.1f}%)")
    print(f"    CPU-fallback:       {total_cpu:>8}  ({100*total_cpu/total_runtime:.1f}%)")
    print(f"    State I/O:          {total_state:>8}  ({100*total_state/total_runtime:.1f}%)")

    # Aggregate decode performance
    if turn_reports:
        all_decode_mean = np.mean([tr['decode_report']['step']['mean'] for tr in turn_reports])
        all_decode_p50 = np.mean([tr['decode_report']['step']['p50'] for tr in turn_reports])
        all_decode_p95 = np.mean([tr['decode_report']['step']['p95'] for tr in turn_reports])
        all_decode_ffn = np.mean([tr['decode_report']['ffn_per_step']['mean'] for tr in turn_reports])
        all_decode_embed = np.mean([tr['decode_report']['embed']['mean'] for tr in turn_reports])
        all_decode_lmhead = np.mean([tr['decode_report']['lmhead']['mean'] for tr in turn_reports])
        total_gen_tokens = sum(tr['decode_steps'] for tr in turn_reports)
        total_gen_time = sum(tr['decode_report']['step']['total'] for tr in turn_reports)
        avg_tok_per_sec = 1000 * total_gen_tokens / total_gen_time if total_gen_time > 0 else 0

        print(f"\n  Decode Performance (avg across {len(turn_reports)} turns):")
        print(f"    Mean latency:     {all_decode_mean:.1f} ms/tok")
        print(f"    P50 latency:      {all_decode_p50:.1f} ms/tok")
        print(f"    P95 latency:      {all_decode_p95:.1f} ms/tok")
        print(f"    Throughput:       {avg_tok_per_sec:.1f} tok/s")
        print(f"    Total generated:  {total_gen_tokens} tokens in {total_gen_time:.0f} ms")

        print(f"\n  Time Breakdown (per decode token, avg):")
        overhead = all_decode_mean - all_decode_ffn - all_decode_lmhead - all_decode_embed
        for lbl, val in [("FFN (4 chunks)", all_decode_ffn),
                         ("LM Head", all_decode_lmhead),
                         ("Embed", all_decode_embed),
                         ("Overhead", overhead)]:
            pct = 100 * val / all_decode_mean if all_decode_mean > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {lbl:<18} {val:>6.1f} ms  {pct:>5.1f}%  {bar}")

    if not args.skip_cpu_compare and gpu_step > 0:
        print(f"\n  ANE Acceleration:")
        print(f"    CPU+GPU:          {gpu_step:.1f} ms/tok ({1000/gpu_step:.1f} tok/s)")
        print(f"    CPU+ANE:          {ane_step:.1f} ms/tok ({1000/ane_step:.1f} tok/s)")
        print(f"    Speedup:          {total_speedup:.2f}x")

    # ANE failures
    if engine.ane_failures:
        print(f"\n  [WARNING] ANE compilation failures (fell back to CPU+GPU):")
        for f in engine.ane_failures:
            print(f"    - {f}")
        print(f"  (This may be transient ANE resource contention — try rerunning)")
    else:
        print(f"\n  ANE Status: All models compiled on ANE successfully")

    # Generated text
    if turn_reports:
        print(f"\n  Generated Text:")
        for tr in turn_reports:
            text_preview = tr['text'][:150].replace('\n', ' ')
            print(f"    Turn {tr['turn']}: {text_preview}")

    print(f"\n{'='*85}")
    print(f"  Profiling complete.")
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
