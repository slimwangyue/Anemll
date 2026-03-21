#!/usr/bin/env python3
"""Profile ANE utilization during multi-turn Qwen3.5-4B conversation.

Configuration: B+E (LUT4 embed + fp16 lm_head + LUT4 FFN chunks)

Measures:
  1. MIL operation breakdown per component (ANE vs CPU classification)
  2. CPU-only vs CPU+ANE timing comparison per component
  3. Multi-turn conversation with per-component per-token timing
  4. Overall throughput and ANE utilization summary

Usage:
    python tests/dev/_test_ane_profile.py --tokens 40
    python tests/dev/_test_ane_profile.py --tokens 40 --skip-cpu-compare
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, time, argparse
import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 1024
NUM_CHUNKS = 4
EXPORT_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1"

CONVERSATION_TURNS = [
    "What is a stack in computer science?",
    "How does it compare to a queue?",
    "Give me a Python example of each.",
]

# ── ANE op classification ────────────────────────────────────────────

# Ops that run on Apple Neural Engine
ANE_OPS = {
    # Core compute
    'conv', 'conv_transpose', 'linear', 'matmul', 'einsum',
    # Element-wise
    'add', 'mul', 'sub', 'real_div', 'floor_div', 'mod',
    'pow', 'sqrt', 'rsqrt', 'abs', 'neg', 'exp', 'log',
    'ceil', 'floor', 'round', 'clip', 'sign',
    'maximum', 'minimum',
    # Activations
    'relu', 'gelu', 'sigmoid', 'tanh', 'silu', 'leaky_relu',
    'elu', 'prelu', 'softplus',
    # Normalization
    'layer_norm', 'instance_norm', 'batch_norm', 'l2_norm',
    'softmax', 'log_softmax',
    # Reduction
    'reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_min', 'reduce_prod',
    'cumsum',
    # Data movement (ANE-accelerated reshape/slice)
    'concat', 'split', 'stack', 'reshape', 'transpose',
    'expand_dims', 'squeeze', 'reverse',
    'slice_by_index', 'slice_by_size', 'pad',
    'tile', 'repeat',
    # Gather (ANE handles embedding lookups)
    'gather', 'gather_along_axis', 'gather_nd',
    # Comparison / selection (may ANE or CPU depending on context)
    'select', 'where',
    'greater', 'greater_equal', 'less', 'less_equal',
    'equal', 'not_equal',
    # Pooling
    'avg_pool', 'max_pool', 'l2_pool',
}

# Ops that typically fall back to CPU
CPU_OPS = {
    'scatter', 'scatter_nd', 'scatter_along_axis',
    'slice_update',  # scatter-like state update
    'topk', 'argsort', 'argmax', 'argmin',
    'one_hot',
    'cast',
    'fill', 'fill_like',
    'while_loop', 'cond',
    'non_maximum_suppression',
    'shape', 'rank',
    'range_1d',
    'identity',
}

# State / infrastructure ops (not compute)
STATE_OPS = {'read_state', 'write_state'}

# Compile-time constants (not runtime)
CONST_OPS = {
    'const',
    'constexpr_lut_to_dense',
    'constexpr_affine_dequantize',
    'constexpr_blockwise_shift_scale',
    'constexpr_sparse_to_dense',
    'constexpr_cast',
}


# ── MIL analysis ─────────────────────────────────────────────────────

def _count_ops_in_block(block, op_counts):
    for op in block.operations:
        op_counts[op.type] = op_counts.get(op.type, 0) + 1
        for nested_block in op.blocks:
            _count_ops_in_block(nested_block, op_counts)


def analyze_model(path, label):
    """Load model spec and count MIL ops (no model compilation)."""
    spec = ct.utils.load_spec(path)
    op_counts = {}
    if spec.HasField('mlProgram'):
        prog = spec.mlProgram
        for fn_name in prog.functions:
            fn = prog.functions[fn_name]
            for block_name in fn.block_specializations:
                block = fn.block_specializations[block_name]
                _count_ops_in_block(block, op_counts)

    # Classify
    ane_count = sum(c for t, c in op_counts.items() if t in ANE_OPS)
    cpu_count = sum(c for t, c in op_counts.items() if t in CPU_OPS)
    state_count = sum(c for t, c in op_counts.items() if t in STATE_OPS)
    const_count = sum(c for t, c in op_counts.items() if t in CONST_OPS)
    unknown = {t: c for t, c in op_counts.items()
               if t not in ANE_OPS and t not in CPU_OPS
               and t not in STATE_OPS and t not in CONST_OPS}
    runtime_total = ane_count + cpu_count + state_count + sum(unknown.values())

    return {
        'label': label,
        'op_counts': op_counts,
        'ane': ane_count,
        'cpu': cpu_count,
        'state': state_count,
        'const': const_count,
        'unknown': unknown,
        'runtime_total': runtime_total,
    }


# ── Timing helpers ───────────────────────────────────────────────────

def timing_stats(times_sec):
    a = np.array(times_sec) * 1000  # to ms
    if len(a) == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p95': 0, 'total': 0}
    return {
        'mean': float(np.mean(a)),
        'std': float(np.std(a)),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'p50': float(np.percentile(a, 50)),
        'p95': float(np.percentile(a, 95)),
        'total': float(np.sum(a)),
    }


# ── Profiling engine ─────────────────────────────────────────────────

class ProfileEngine:
    def __init__(self, embed_path, lmhead_path, ffn_dir, ffn_label,
                 compute_unit, name="engine"):
        self.name = name
        self.embed = ct.models.MLModel(embed_path, compute_units=compute_unit)
        self.lmhead = ct.models.MLModel(lmhead_path, compute_units=compute_unit)
        self.ffns = []
        for ci in range(NUM_CHUNKS):
            m = ct.models.MLModel(
                os.path.join(ffn_dir, f"ffn_{ffn_label}_chunk{ci}.mlpackage"),
                compute_units=compute_unit)
            self.ffns.append(m)

        spec = self.ffns[0].get_spec()
        self.inp_map = {}
        for inp in spec.description.input:
            try:
                self.inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.has_linear = 'linear_conv_state' in self.inp_map
        self.reset_all()

    def reset_all(self):
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
                              for _ in range(NUM_CHUNKS)]
            self.lin_recs = [np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
                             for _ in range(NUM_CHUNKS)]
        else:
            self.lin_convs = [None] * NUM_CHUNKS
            self.lin_recs = [None] * NUM_CHUNKS
        self.t_embed = []
        self.t_ffn = [[] for _ in range(NUM_CHUNKS)]
        self.t_lmhead = []
        self.t_step = []

    def _step(self, tok_id, pos):
        t_step = time.perf_counter()

        # Embed
        tok = np.array([[tok_id]], dtype=np.int32)
        t0 = time.perf_counter()
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        self.t_embed.append(time.perf_counter() - t0)

        # FFN chunks
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if self.lin_convs[ci] is not None:
                inp["linear_conv_state"] = self.lin_convs[ci]
                inp["linear_recurrent_state"] = self.lin_recs[ci]
            t0 = time.perf_counter()
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            self.t_ffn[ci].append(time.perf_counter() - t0)
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        # LM Head
        t0 = time.perf_counter()
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        self.t_lmhead.append(time.perf_counter() - t0)

        self.t_step.append(time.perf_counter() - t_step)

        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
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
        end_pos = prefill_end_pos + len(tokens)
        return tokens, end_pos

    def get_timing_report(self):
        """Get aggregated timing for all recorded steps."""
        report = {
            'embed': timing_stats(self.t_embed),
            'lmhead': timing_stats(self.t_lmhead),
            'step': timing_stats(self.t_step),
            'ffn_chunks': [timing_stats(self.t_ffn[ci]) for ci in range(NUM_CHUNKS)],
        }
        # Aggregate all FFN chunks
        all_ffn = []
        for ci in range(NUM_CHUNKS):
            all_ffn.extend(self.t_ffn[ci])
        report['ffn_all'] = timing_stats(all_ffn)

        # Per-step FFN total (sum of 4 chunks per step)
        n_steps = len(self.t_embed)
        ffn_per_step = []
        for si in range(n_steps):
            ffn_sum = sum(self.t_ffn[ci][si] for ci in range(NUM_CHUNKS)
                         if si < len(self.t_ffn[ci]))
            ffn_per_step.append(ffn_sum)
        report['ffn_per_step'] = timing_stats(ffn_per_step)
        report['n_steps'] = n_steps
        return report

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        self.ffns = []
        gc.collect()


# ── Conversation helpers ─────────────────────────────────────────────

def _ensure_ids(tpl_out):
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


def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--export-dir", type=str, default=EXPORT_DIR)
    parser.add_argument("--skip-cpu-compare", action="store_true",
                        help="Skip CPU-only vs ANE timing comparison")
    parser.add_argument("--compare-steps", type=int, default=15,
                        help="Number of decode steps for CPU vs ANE comparison")
    args = parser.parse_args()

    max_gen = args.tokens
    out_dir = args.export_dir
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)

    # Paths (B+E config: LUT4 embed, fp16 lm_head, LUT4 FFN)
    embed_path = os.path.join(out_dir, "embeddings.mlpackage")
    lmhead_path = os.path.join(out_dir, "lm_head.mlpackage")
    ffn_label = "LUT4"

    print("=" * 85)
    print("  ANE PROFILING — Qwen3.5-4B (B+E: LUT4 embed + fp16 lmhead + LUT4 FFN)")
    print(f"  Tokens/turn: {max_gen}, Turns: {len(CONVERSATION_TURNS)}, CTX={CTX}")
    print("=" * 85)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: MIL Operation Analysis
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print("  SECTION 1: MIL Operation Analysis")
    print(f"{'='*85}")

    model_paths = [
        (embed_path, "Embeddings (LUT4)"),
        (lmhead_path, "LM Head (fp16)"),
    ]
    for ci in range(NUM_CHUNKS):
        model_paths.append((
            os.path.join(out_dir, f"ffn_{ffn_label}_chunk{ci}.mlpackage"),
            f"FFN chunk{ci} (LUT4)"
        ))

    analyses = []
    for path, label in model_paths:
        a = analyze_model(path, label)
        analyses.append(a)

    # Summary table
    print(f"\n  {'Component':<25} {'Runtime':>8} {'ANE':>6} {'CPU':>6} {'State':>6} {'ANE%':>7}")
    print(f"  {'-'*60}")
    total_ane = total_cpu = total_state = total_runtime = 0
    for a in analyses:
        ane_pct = 100 * a['ane'] / a['runtime_total'] if a['runtime_total'] > 0 else 0
        print(f"  {a['label']:<25} {a['runtime_total']:>8} {a['ane']:>6} {a['cpu']:>6} {a['state']:>6} {ane_pct:>6.1f}%")
        total_ane += a['ane']
        total_cpu += a['cpu']
        total_state += a['state']
        total_runtime += a['runtime_total']
    print(f"  {'-'*60}")
    total_ane_pct = 100 * total_ane / total_runtime if total_runtime > 0 else 0
    print(f"  {'TOTAL':<25} {total_runtime:>8} {total_ane:>6} {total_cpu:>6} {total_state:>6} {total_ane_pct:>6.1f}%")

    # Detailed op breakdown for FFN (most complex)
    ffn_a = analyses[2]  # chunk0 representative
    print(f"\n  FFN chunk0 detailed ops (runtime only, excl. const/constexpr):")
    runtime_ops = {t: c for t, c in ffn_a['op_counts'].items() if t not in CONST_OPS}
    for op_type, count in sorted(runtime_ops.items(), key=lambda x: -x[1]):
        device = "ANE" if op_type in ANE_OPS else ("CPU" if op_type in CPU_OPS else ("STATE" if op_type in STATE_OPS else "???"))
        print(f"    {op_type:<25} {count:>5}  [{device}]")

    if ffn_a['unknown']:
        print(f"\n  Unknown ops (not classified): {ffn_a['unknown']}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: CPU-only vs CPU+ANE Timing Comparison
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_cpu_compare:
        print(f"\n{'='*85}")
        print("  SECTION 2: CPU+GPU vs CPU+ANE Timing Comparison")
        print(f"  ({args.compare_steps} decode steps each)")
        print(f"  Note: CPU_ONLY fails for stateful FFN LUT4 models, using CPU_AND_GPU instead")
        print(f"{'='*85}")

        # Prepare a short prompt for benchmarking
        bench_prompt = "What is a stack?"
        bench_ids = _ensure_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": bench_prompt}],
            return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True))
        bench_tokens = bench_ids[0].tolist()

        # --- CPU_AND_NE first (primary target, always loads cleanly) ---
        print("\n  Loading models with CPU_AND_NE...")
        t0 = time.time()
        engine_ane = ProfileEngine(embed_path, lmhead_path, out_dir, ffn_label,
                                   ct.ComputeUnit.CPU_AND_NE, name="CPU_AND_NE")
        print(f"  Loaded in {time.time()-t0:.1f}s")

        # Warmup
        for i in range(min(3, len(bench_tokens))):
            engine_ane._step(bench_tokens[i], i)
        engine_ane.reset_all()

        # Benchmark
        print(f"  Running {len(bench_tokens)} prefill + {args.compare_steps} decode steps (CPU+ANE)...")
        engine_ane.prefill_and_decode(bench_tokens, 0, args.compare_steps, stop_ids)
        ane_report = engine_ane.get_timing_report()
        engine_ane.cleanup()

        # Clean temp compiled models between loads to avoid conflicts
        import glob
        for p in glob.glob("/private/var/folders/ss/*/T/*.mlmodelc"):
            try:
                import shutil
                shutil.rmtree(p)
            except Exception:
                pass

        # --- CPU_AND_GPU (no ANE) ---
        print("\n  Loading models with CPU_AND_GPU (no ANE)...")
        t0 = time.time()
        engine_nogpu = ProfileEngine(embed_path, lmhead_path, out_dir, ffn_label,
                                     ct.ComputeUnit.CPU_AND_GPU, name="CPU_AND_GPU")
        print(f"  Loaded in {time.time()-t0:.1f}s")

        # Warmup (3 steps)
        for i in range(min(3, len(bench_tokens))):
            engine_nogpu._step(bench_tokens[i], i)
        engine_nogpu.reset_all()

        # Benchmark
        print(f"  Running {len(bench_tokens)} prefill + {args.compare_steps} decode steps (CPU+GPU)...")
        engine_nogpu.prefill_and_decode(bench_tokens, 0, args.compare_steps, stop_ids)
        cpu_report = engine_nogpu.get_timing_report()
        engine_nogpu.cleanup()

        # Compare
        print(f"\n  {'Component':<22} {'CPU+GPU':>12} {'CPU+ANE':>12} {'Speedup':>10} {'ANE?':>6}")
        print(f"  {'-'*64}")
        comparisons = [
            ("Embed", cpu_report['embed'], ane_report['embed']),
            ("FFN (4 chunks/step)", cpu_report['ffn_per_step'], ane_report['ffn_per_step']),
        ]
        for ci in range(NUM_CHUNKS):
            comparisons.append(
                (f"  FFN chunk{ci}",
                 cpu_report['ffn_chunks'][ci], ane_report['ffn_chunks'][ci]))
        comparisons.append(("LM Head", cpu_report['lmhead'], ane_report['lmhead']))
        comparisons.append(("Full step", cpu_report['step'], ane_report['step']))

        for label, cpu_s, ane_s in comparisons:
            cpu_ms = cpu_s['mean']
            ane_ms = ane_s['mean']
            speedup = cpu_ms / ane_ms if ane_ms > 0 else float('inf')
            is_ane = "YES" if speedup > 1.2 else ("maybe" if speedup > 1.05 else "no")
            print(f"  {label:<22} {cpu_ms:>10.2f}ms {ane_ms:>10.2f}ms {speedup:>9.2f}x {is_ane:>6}")

        cpu_total_step = cpu_report['step']['mean']
        ane_total_step = ane_report['step']['mean']
        total_speedup = cpu_total_step / ane_total_step if ane_total_step > 0 else 0
        ane_time_saved = cpu_total_step - ane_total_step
        ane_pct_of_step = 100 * ane_time_saved / cpu_total_step if cpu_total_step > 0 else 0
        print(f"\n  ANE acceleration: {total_speedup:.2f}x overall speedup vs CPU+GPU")
        print(f"  ANE saves {ane_time_saved:.1f}ms/step ({ane_pct_of_step:.1f}% of CPU+GPU time)")
        print(f"  Throughput: CPU+GPU={1000/cpu_total_step:.1f} tok/s, CPU+ANE={1000/ane_total_step:.1f} tok/s")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: Multi-Turn Conversation Profiling (CPU+ANE)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print("  SECTION 3: Multi-Turn Conversation Profiling (CPU+ANE)")
    print(f"{'='*85}")

    engine = ProfileEngine(embed_path, lmhead_path, out_dir, ffn_label,
                           ct.ComputeUnit.CPU_AND_NE, name="B+E_ANE")

    # Warmup
    engine._step(1, 0)
    engine.reset_all()

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
        token_list = input_ids[0].tolist()

        t_turn = time.time()
        gen_tokens, end_pos = engine.prefill_and_decode(
            token_list, 0, max_gen, stop_ids)
        turn_time = time.time() - t_turn

        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": "<think>\n" + raw_text})

        report = engine.get_timing_report()
        total_steps = report['n_steps']
        prefill_steps = prompt_len
        decode_steps = total_steps - prefill_steps

        # Separate prefill vs decode timing
        decode_step_times = engine.t_step[prefill_steps:]
        decode_embed_times = engine.t_embed[prefill_steps:]
        decode_lmhead_times = engine.t_lmhead[prefill_steps:]
        decode_ffn_times = [engine.t_ffn[ci][prefill_steps:] for ci in range(NUM_CHUNKS)]

        d_step = timing_stats(decode_step_times)
        d_embed = timing_stats(decode_embed_times)
        d_lmhead = timing_stats(decode_lmhead_times)
        d_ffn_chunks = [timing_stats(decode_ffn_times[ci]) for ci in range(NUM_CHUNKS)]
        d_ffn_per_step = []
        for si in range(len(decode_step_times)):
            ffn_sum = sum(decode_ffn_times[ci][si] for ci in range(NUM_CHUNKS)
                         if si < len(decode_ffn_times[ci]))
            d_ffn_per_step.append(ffn_sum)
        d_ffn_total = timing_stats(d_ffn_per_step)

        turn_reports.append({
            'turn': ti + 1,
            'prompt_len': prompt_len,
            'decode_steps': decode_steps,
            'total_steps': total_steps,
            'turn_time': turn_time,
            'text': raw_text,
            'gen_tokens': gen_tokens,
            'decode_step': d_step,
            'decode_embed': d_embed,
            'decode_lmhead': d_lmhead,
            'decode_ffn_total': d_ffn_total,
            'decode_ffn_chunks': d_ffn_chunks,
        })

        step_ms = d_step['mean']
        tok_per_sec = 1000 / step_ms if step_ms > 0 else 0

        print(f"    Prompt: {prompt_len} tok, Decode: {decode_steps} tok, Total: {total_steps} steps")
        print(f"    Turn time: {turn_time:.1f}s, Decode only: {d_step['total']:.0f}ms")
        print(f"    Decode throughput: {tok_per_sec:.1f} tok/s ({step_ms:.1f} ms/tok)")
        print(f"    Text: {raw_text[:120]}")

        print(f"\n    {'Component':<22} {'Mean(ms)':>9} {'Std':>7} {'Min':>7} {'Max':>7} {'% step':>8}")
        print(f"    {'-'*62}")
        # Compute percentages
        components = [
            ("Embed", d_embed),
            ("FFN (4 chunks)", d_ffn_total),
        ]
        for ci in range(NUM_CHUNKS):
            components.append((f"  chunk{ci}", d_ffn_chunks[ci]))
        components.append(("LM Head", d_lmhead))
        components.append(("Full step", d_step))

        for clabel, cs in components:
            pct = 100 * cs['mean'] / d_step['mean'] if d_step['mean'] > 0 else 0
            if clabel == "Full step":
                print(f"    {'-'*62}")
            print(f"    {clabel:<22} {cs['mean']:>9.2f} {cs['std']:>7.2f} {cs['min']:>7.2f} {cs['max']:>7.2f} {pct:>7.1f}%")

        # Reset timing for next turn
        engine.t_embed.clear()
        engine.t_lmhead.clear()
        engine.t_step.clear()
        for ci in range(NUM_CHUNKS):
            engine.t_ffn[ci].clear()

    engine.cleanup()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4: Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print("  SECTION 4: SUMMARY")
    print(f"{'='*85}")

    # Model sizes
    embed_sz = dir_size_mb(embed_path)
    lmhead_sz = dir_size_mb(lmhead_path)
    ffn_total_sz = sum(dir_size_mb(os.path.join(out_dir, f"ffn_{ffn_label}_chunk{ci}.mlpackage"))
                       for ci in range(NUM_CHUNKS))
    total_sz = embed_sz + lmhead_sz + ffn_total_sz
    total_be = embed_sz + ffn_total_sz  # B+E: no separate lm_head

    print(f"\n  Model Size (B+E config):")
    print(f"    Embed (LUT4):      {embed_sz:>8.1f} MB")
    print(f"    FFN (LUT4 × 4):    {ffn_total_sz:>8.1f} MB")
    print(f"    LM Head (fp16):    {lmhead_sz:>8.1f} MB  (dedup → 0 MB)")
    print(f"    Current total:     {total_sz:>8.1f} MB")
    print(f"    With dedup (B+E):  {total_be:>8.1f} MB")

    # ANE operation stats
    print(f"\n  ANE Operation Utilization:")
    print(f"    Total runtime ops: {total_runtime}")
    print(f"    ANE-eligible:      {total_ane} ({total_ane_pct:.1f}%)")
    print(f"    CPU-fallback:      {total_cpu} ({100*total_cpu/total_runtime:.1f}%)")
    print(f"    State I/O:         {total_state} ({100*total_state/total_runtime:.1f}%)")

    # Aggregate decode performance across turns
    all_decode_mean = np.mean([tr['decode_step']['mean'] for tr in turn_reports])
    all_decode_embed = np.mean([tr['decode_embed']['mean'] for tr in turn_reports])
    all_decode_ffn = np.mean([tr['decode_ffn_total']['mean'] for tr in turn_reports])
    all_decode_lmhead = np.mean([tr['decode_lmhead']['mean'] for tr in turn_reports])
    total_gen_tokens = sum(tr['decode_steps'] for tr in turn_reports)
    total_gen_time = sum(tr['decode_step']['total'] for tr in turn_reports)
    avg_tok_per_sec = 1000 * total_gen_tokens / total_gen_time if total_gen_time > 0 else 0

    print(f"\n  Decode Performance (avg across {len(turn_reports)} turns):")
    print(f"    Step latency:     {all_decode_mean:.1f} ms/tok")
    print(f"    Throughput:       {avg_tok_per_sec:.1f} tok/s")
    print(f"    Total generated:  {total_gen_tokens} tokens in {total_gen_time:.0f} ms")

    print(f"\n  Time Breakdown (per decode token, avg):")
    print(f"    FFN (4 chunks):   {all_decode_ffn:.1f} ms ({100*all_decode_ffn/all_decode_mean:.1f}%)")
    print(f"    LM Head:          {all_decode_lmhead:.1f} ms ({100*all_decode_lmhead/all_decode_mean:.1f}%)")
    print(f"    Embed:            {all_decode_embed:.1f} ms ({100*all_decode_embed/all_decode_mean:.1f}%)")
    overhead = all_decode_mean - all_decode_ffn - all_decode_lmhead - all_decode_embed
    print(f"    Overhead/other:   {overhead:.1f} ms ({100*overhead/all_decode_mean:.1f}%)")

    if not args.skip_cpu_compare:
        print(f"\n  ANE Acceleration (from Section 2):")
        print(f"    CPU+GPU step:     {cpu_total_step:.1f} ms/tok ({1000/cpu_total_step:.1f} tok/s)")
        print(f"    CPU+ANE step:     {ane_total_step:.1f} ms/tok ({1000/ane_total_step:.1f} tok/s)")
        print(f"    Speedup:          {total_speedup:.2f}x")
        print(f"    ANE contribution: {ane_time_saved:.1f} ms saved/step ({ane_pct_of_step:.1f}% of CPU+GPU time)")

    # Generated text
    print(f"\n  Generated Text:")
    for tr in turn_reports:
        text_preview = tr['text'][:150].replace('\n', ' ')
        print(f"    Turn {tr['turn']}: {text_preview}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
