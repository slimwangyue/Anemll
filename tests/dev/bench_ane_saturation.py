#!/usr/bin/env python3
"""ANE saturation measurement for Qwen3.5-4B decode chunks.

Answers: Is a single chunk already saturating the ANE?

Method:
  1. Solo execution: run each chunk alone, measure wall-clock latency (T_solo)
  2. Concurrent execution: run same chunk from 2 threads with independent states (T_concurrent)
  3. Saturation ratio: T_concurrent / T_solo
     - Ratio ≈ 1.0 → spare ANE capacity (not saturated)
     - Ratio ≈ 2.0 → fully saturated by one call (serialized)
     - Ratio in between → partial overlap
  4. os_signpost instrumentation for Instruments/xctrace capture
  5. Inter-call gap analysis: measure dispatch overhead vs compute time

Usage:
    python tests/dev/bench_ane_saturation.py \
        --model-dir /Users/yw68/Anemll/qwen3_5_flll_9chunk \
        --iterations 50

    # Capture with Instruments (optional):
    xctrace record --template 'Time Profiler' --launch -- \
        python tests/dev/bench_ane_saturation.py --model-dir ... --iterations 20
"""
import sys, os, time, argparse, threading, ctypes, ctypes.util
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts_qwen3_5")
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _REPO_ROOT)

import coremltools as ct
from transformers import AutoTokenizer
from config import CTX, NUM_CHUNKS, BATCH_SIZE, FFN_LABEL, DEFAULT_HF_MODEL, CHUNK_RANGES

CU = ct.ComputeUnit.CPU_AND_NE


# ── os_signpost instrumentation ──────────────────────────────────

class Signposter:
    """Lightweight wrapper around Apple os_signpost API for Instruments."""

    def __init__(self, subsystem="com.anemll.bench", category="decode"):
        try:
            self._lib = ctypes.CDLL(ctypes.util.find_library('System'))
            self._log = self._lib.os_log_create(
                subsystem.encode(), category.encode())
            self._enabled = True
        except Exception:
            self._enabled = False

    def begin(self, name):
        """Emit signpost interval begin."""
        if not self._enabled:
            return time.perf_counter_ns()
        # Use perf_counter_ns as signpost ID for matching begin/end
        ts = time.perf_counter_ns()
        return ts

    def end(self, name, begin_ts):
        """Emit signpost interval end."""
        return time.perf_counter_ns()


# ── Model loading ────────────────────────────────────────────────

def _find_model(base_dir, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def _load_model(path, cu, function_name=None):
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, cu)
    kw = {"compute_units": cu}
    if function_name:
        kw["function_name"] = function_name
    return ct.models.MLModel(path, **kw)


class Models:
    """Load all models once."""

    def __init__(self, model_dir):
        self.combined_dir = os.path.join(model_dir, f"combined_{FFN_LABEL}_dedup")
        self.use_combined = os.path.isdir(self.combined_dir)

        print(f"  Loading from {model_dir} ({'COMBINED' if self.use_combined else 'SEPARATE'})")

        t0 = time.time()
        self.embed = _load_model(_find_model(model_dir, "embeddings"), CU)
        print(f"  embeddings: {time.time()-t0:.1f}s")

        t0 = time.time()
        try:
            self.lmhead = _load_model(_find_model(model_dir, "lm_head_logits"), CU)
        except FileNotFoundError:
            self.lmhead = _load_model(_find_model(model_dir, "lm_head"), CU)
        spec = self.lmhead.get_spec()
        out_names = [o.name for o in spec.description.output]
        self.logits_keys = sorted([n for n in out_names if n.startswith("logits")],
                                  key=lambda x: int(x.replace("logits", "") or "0"))
        print(f"  lm_head: {time.time()-t0:.1f}s")

        self.ffns = []
        for ci in range(NUM_CHUNKS):
            t0 = time.time()
            if self.use_combined:
                path = _find_model(self.combined_dir, f"chunk{ci}")
                m = _load_model(path, CU, function_name="infer")
            else:
                m = _load_model(
                    _find_model(model_dir, f"ffn_{FFN_LABEL}_chunk{ci}"), CU)
            self.ffns.append(m)
            print(f"  chunk{ci}: {time.time()-t0:.1f}s")

        # Detect state shapes
        self.conv_shapes = []
        self.rec_shapes = []
        for ci in range(NUM_CHUNKS):
            spec = self.ffns[ci].get_spec()
            cs, rs = (6, 1024, 32), (6, 32, 128, 128)
            inputs = spec.description.input
            if self.use_combined and hasattr(spec.description, 'functions'):
                for fn in spec.description.functions:
                    if fn.name == "infer":
                        inputs = fn.input
                        break
            for inp in inputs:
                if inp.name == 'linear_conv_state':
                    cs = tuple(inp.type.multiArrayType.shape)
                if inp.name == 'linear_recurrent_state':
                    rs = tuple(inp.type.multiArrayType.shape)
            self.conv_shapes.append(cs)
            self.rec_shapes.append(rs)


class ChunkState:
    """Per-instance mutable state for one chunk."""

    def __init__(self, models, chunk_idx):
        ci = chunk_idx
        self.ci = ci
        self.state = models.ffns[ci].make_state()
        self.lin_conv = np.zeros(models.conv_shapes[ci], dtype=np.float16)
        self.lin_rec = np.zeros(models.rec_shapes[ci], dtype=np.float16)

    def reset(self, models):
        self.state = models.ffns[self.ci].make_state()
        self.lin_conv[:] = 0
        self.lin_rec[:] = 0


# ── Measurement functions ────────────────────────────────────────

def make_dummy_hidden():
    """Create realistic hidden state tensor."""
    return np.random.randn(1, 1, 2560).astype(np.float16) * 0.01


def run_chunk_once(models, ci, chunk_state, hidden, pos=10):
    """Run one chunk predict. Returns (output_hidden, wall_ns)."""
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :pos + 1] = 0
    pos_arr = np.array([pos], dtype=np.int32)

    inp = {
        "hidden_states": hidden.astype(np.float16),
        "position_ids": pos_arr,
        "causal_mask": mask,
        "current_pos": pos_arr,
        "linear_conv_state": chunk_state.lin_conv,
        "linear_recurrent_state": chunk_state.lin_rec,
    }

    t0 = time.perf_counter_ns()
    out = models.ffns[ci].predict(inp, state=chunk_state.state)
    t1 = time.perf_counter_ns()

    hidden_out = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        chunk_state.lin_conv = out['linear_conv_state_out']
        chunk_state.lin_rec = out['linear_recurrent_state_out']

    return hidden_out, t1 - t0


def run_embed_once(models, tok_id=1):
    """Run embed predict. Returns (hidden, wall_ns)."""
    tok = np.array([[tok_id]], dtype=np.int32)
    t0 = time.perf_counter_ns()
    hidden = list(models.embed.predict({"input_ids": tok}).values())[0]
    t1 = time.perf_counter_ns()
    return hidden, t1 - t0


def run_lmhead_once(models, hidden):
    """Run lm_head predict. Returns (next_id, wall_ns)."""
    t0 = time.perf_counter_ns()
    lm_out = models.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    t1 = time.perf_counter_ns()
    parts = [lm_out[k].flatten().astype(np.float32) for k in models.logits_keys]
    logits = np.concatenate(parts)
    return int(np.argmax(logits)), t1 - t0


# ══════════════════════════════════════════════════════════════════
#  TEST 1: Solo chunk latency (single thread, no contention)
# ══════════════════════════════════════════════════════════════════

def measure_solo(models, iterations, warmup):
    """Measure each component solo."""
    results = {}

    # Embed
    print(f"\n  Measuring embed (solo, {iterations} iters)...")
    for _ in range(warmup):
        run_embed_once(models)
    times = []
    for _ in range(iterations):
        _, ns = run_embed_once(models)
        times.append(ns)
    results['embed'] = np.array(times)

    # LM Head
    print(f"  Measuring lm_head (solo, {iterations} iters)...")
    hidden = make_dummy_hidden()
    for _ in range(warmup):
        run_lmhead_once(models, hidden)
    times = []
    for _ in range(iterations):
        _, ns = run_lmhead_once(models, hidden)
        times.append(ns)
    results['lmhead'] = np.array(times)

    # Each FFN chunk
    for ci in range(NUM_CHUNKS):
        layers = f"L{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        print(f"  Measuring chunk{ci} ({layers}, solo, {iterations} iters)...")
        cs = ChunkState(models, ci)
        hidden = make_dummy_hidden()
        for _ in range(warmup):
            cs.reset(models)
            run_chunk_once(models, ci, cs, hidden)
        times = []
        for _ in range(iterations):
            cs.reset(models)
            h_out, ns = run_chunk_once(models, ci, cs, hidden)
            times.append(ns)
        results[f'chunk{ci}'] = np.array(times)

    return results


# ══════════════════════════════════════════════════════════════════
#  TEST 2: Concurrent chunk execution (2 threads, same chunk)
# ══════════════════════════════════════════════════════════════════

def measure_concurrent_chunk(models, ci, iterations, warmup):
    """Run the same chunk from 2 threads concurrently."""
    barrier = threading.Barrier(2)

    def worker(thread_id, times_out):
        cs = ChunkState(models, ci)
        hidden = make_dummy_hidden()

        # Warmup
        for _ in range(warmup):
            cs.reset(models)
            run_chunk_once(models, ci, cs, hidden)

        for i in range(iterations):
            cs.reset(models)
            barrier.wait()  # Sync both threads
            _, ns = run_chunk_once(models, ci, cs, hidden)
            times_out.append(ns)

    t0_times = []
    t1_times = []
    threads = [
        threading.Thread(target=worker, args=(0, t0_times)),
        threading.Thread(target=worker, args=(1, t1_times)),
    ]

    wall_start = time.perf_counter_ns()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_total = time.perf_counter_ns() - wall_start

    return np.array(t0_times), np.array(t1_times), wall_total


# ══════════════════════════════════════════════════════════════════
#  TEST 3: Concurrent different chunks (2 threads, chunk i vs chunk j)
# ══════════════════════════════════════════════════════════════════

def measure_concurrent_different(models, ci, cj, iterations, warmup):
    """Run chunk i and chunk j concurrently from 2 threads."""
    barrier = threading.Barrier(2)

    def worker(chunk_idx, times_out):
        cs = ChunkState(models, chunk_idx)
        hidden = make_dummy_hidden()
        for _ in range(warmup):
            cs.reset(models)
            run_chunk_once(models, chunk_idx, cs, hidden)
        for i in range(iterations):
            cs.reset(models)
            barrier.wait()
            _, ns = run_chunk_once(models, chunk_idx, cs, hidden)
            times_out.append(ns)

    ti_times = []
    tj_times = []
    threads = [
        threading.Thread(target=worker, args=(ci, ti_times)),
        threading.Thread(target=worker, args=(cj, tj_times)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return np.array(ti_times), np.array(tj_times)


# ══════════════════════════════════════════════════════════════════
#  TEST 4: Full decode pipeline with nanosecond timestamps
# ══════════════════════════════════════════════════════════════════

def measure_pipeline_timing(models, iterations, warmup):
    """Measure the full decode pipeline with per-call timestamps.
    Reports dispatch gaps between consecutive predict() calls."""

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL, trust_remote_code=True)
    msgs = [{"role": "user", "content": "Hello"}]
    prompt_ids = list(tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        enable_thinking=True, return_dict=False))

    # We need KV cache primed — do a few prefill tokens first
    all_states = [ChunkState(models, ci) for ci in range(NUM_CHUNKS)]

    # Prime with first few tokens sequentially
    for tok_idx, tid in enumerate(prompt_ids[:5]):
        tok = np.array([[tid]], dtype=np.int32)
        hidden = list(models.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :tok_idx + 1] = 0
        pos_arr = np.array([tok_idx], dtype=np.int32)
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": all_states[ci].lin_conv,
                "linear_recurrent_state": all_states[ci].lin_rec,
            }
            out = models.ffns[ci].predict(inp, state=all_states[ci].state)
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                all_states[ci].lin_conv = out['linear_conv_state_out']
                all_states[ci].lin_rec = out['linear_recurrent_state_out']

    # Now measure decode steps at positions 5..5+iterations
    pipeline_data = []
    cur_tok = 1  # dummy token ID

    for step in range(-warmup, iterations):
        pos = 5 + max(step, 0)
        tok = np.array([[cur_tok]], dtype=np.int32)

        timestamps = []  # list of (component, start_ns, end_ns)

        # Embed
        t0 = time.perf_counter_ns()
        hidden = list(models.embed.predict({"input_ids": tok}).values())[0]
        t1 = time.perf_counter_ns()
        timestamps.append(('embed', t0, t1))

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        pos_arr = np.array([pos], dtype=np.int32)

        # FFN chunks
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": all_states[ci].lin_conv,
                "linear_recurrent_state": all_states[ci].lin_rec,
            }
            t0 = time.perf_counter_ns()
            out = models.ffns[ci].predict(inp, state=all_states[ci].state)
            t1 = time.perf_counter_ns()
            timestamps.append((f'chunk{ci}', t0, t1))

            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                all_states[ci].lin_conv = out['linear_conv_state_out']
                all_states[ci].lin_rec = out['linear_recurrent_state_out']

        # LM Head
        t0 = time.perf_counter_ns()
        lm_out = models.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        t1 = time.perf_counter_ns()
        timestamps.append(('lmhead', t0, t1))

        parts = [lm_out[k].flatten().astype(np.float32) for k in models.logits_keys]
        logits = np.concatenate(parts)
        cur_tok = int(np.argmax(logits))

        if step >= 0:
            pipeline_data.append(timestamps)

    return pipeline_data


# ══════════════════════════════════════════════════════════════════
#  TEST 5: 2-stream concurrent full pipeline with timestamps
# ══════════════════════════════════════════════════════════════════

def measure_concurrent_pipeline(models, iterations, warmup):
    """Run 2 full decode pipelines concurrently, collecting timestamps."""
    barrier = threading.Barrier(2)

    def pipeline_worker(stream_id, result_out):
        all_states = [ChunkState(models, ci) for ci in range(NUM_CHUNKS)]

        # Prime KV
        for tok_idx in range(3):
            tok = np.array([[1]], dtype=np.int32)
            hidden = list(models.embed.predict({"input_ids": tok}).values())[0]
            mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
            mask[:, :, :, :tok_idx + 1] = 0
            pos_arr = np.array([tok_idx], dtype=np.int32)
            for ci in range(NUM_CHUNKS):
                inp = {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": pos_arr,
                    "causal_mask": mask,
                    "current_pos": pos_arr,
                    "linear_conv_state": all_states[ci].lin_conv,
                    "linear_recurrent_state": all_states[ci].lin_rec,
                }
                out = models.ffns[ci].predict(inp, state=all_states[ci].state)
                hidden = out["output_hidden_states"]
                if 'linear_conv_state_out' in out:
                    all_states[ci].lin_conv = out['linear_conv_state_out']
                    all_states[ci].lin_rec = out['linear_recurrent_state_out']

        cur_tok = 1
        data = []
        barrier.wait()  # Sync start

        for step in range(-warmup, iterations):
            pos = 3 + max(step, 0)
            tok = np.array([[cur_tok]], dtype=np.int32)
            timestamps = []

            t0 = time.perf_counter_ns()
            hidden = list(models.embed.predict({"input_ids": tok}).values())[0]
            t1 = time.perf_counter_ns()
            timestamps.append((f's{stream_id}_embed', t0, t1))

            mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
            mask[:, :, :, :pos + 1] = 0
            pos_arr = np.array([pos], dtype=np.int32)

            for ci in range(NUM_CHUNKS):
                inp = {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": pos_arr,
                    "causal_mask": mask,
                    "current_pos": pos_arr,
                    "linear_conv_state": all_states[ci].lin_conv,
                    "linear_recurrent_state": all_states[ci].lin_rec,
                }
                t0 = time.perf_counter_ns()
                out = models.ffns[ci].predict(inp, state=all_states[ci].state)
                t1 = time.perf_counter_ns()
                timestamps.append((f's{stream_id}_chunk{ci}', t0, t1))

                hidden = out["output_hidden_states"]
                if 'linear_conv_state_out' in out:
                    all_states[ci].lin_conv = out['linear_conv_state_out']
                    all_states[ci].lin_rec = out['linear_recurrent_state_out']

            t0 = time.perf_counter_ns()
            lm_out = models.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
            t1 = time.perf_counter_ns()
            timestamps.append((f's{stream_id}_lmhead', t0, t1))

            parts = [lm_out[k].flatten().astype(np.float32) for k in models.logits_keys]
            cur_tok = int(np.argmax(np.concatenate(parts)))

            if step >= 0:
                data.append(timestamps)

        result_out[stream_id] = data

    results = {}
    threads = [
        threading.Thread(target=pipeline_worker, args=(0, results)),
        threading.Thread(target=pipeline_worker, args=(1, results)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ── Reporting ────────────────────────────────────────────────────

def ns_to_ms(ns):
    return ns / 1_000_000


def report_solo(results):
    print(f"\n  {'Component':<12} {'Layers':<10} {'Mean(ms)':<10} {'Std':<8} "
          f"{'P50':<8} {'P95':<8} {'Min':<8} {'Max':<8}")
    print(f"  {'-'*72}")
    for key in ['embed'] + [f'chunk{ci}' for ci in range(NUM_CHUNKS)] + ['lmhead']:
        arr = results[key] / 1e6  # ns → ms
        if key.startswith('chunk'):
            ci = int(key.replace('chunk', ''))
            layers = f"L{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        elif key == 'embed':
            layers = "embed"
        else:
            layers = "lm_head"
        print(f"  {key:<12} {layers:<10} {np.mean(arr):<10.3f} {np.std(arr):<8.3f} "
              f"{np.percentile(arr, 50):<8.3f} {np.percentile(arr, 95):<8.3f} "
              f"{np.min(arr):<8.3f} {np.max(arr):<8.3f}")


def report_saturation(models, solo_results, iterations, warmup):
    """Run concurrent tests and report saturation ratios."""
    print(f"\n  ANE SATURATION TEST: Solo vs 2-Thread Concurrent (same chunk)")
    print(f"  {'Component':<12} {'Solo(ms)':<10} {'Conc(ms)':<10} "
          f"{'Ratio':<8} {'Saturated?':<12} {'Interpretation'}")
    print(f"  {'-'*78}")

    for ci in range(NUM_CHUNKS):
        solo_ms = np.mean(solo_results[f'chunk{ci}']) / 1e6
        layers = f"L{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"

        t0_times, t1_times, wall = measure_concurrent_chunk(
            models, ci, iterations, warmup)

        conc_ms_t0 = np.mean(t0_times) / 1e6
        conc_ms_t1 = np.mean(t1_times) / 1e6
        conc_ms = (conc_ms_t0 + conc_ms_t1) / 2

        ratio = conc_ms / solo_ms if solo_ms > 0 else 0

        if ratio > 1.8:
            status = "SATURATED"
            interp = "1 call fills ANE"
        elif ratio > 1.3:
            status = "PARTIAL"
            interp = "some spare capacity"
        else:
            status = "NOT SAT"
            interp = "significant spare capacity"

        print(f"  chunk{ci:<5} {solo_ms:<10.3f} {conc_ms:<10.3f} "
              f"{ratio:<8.2f} {status:<12} {interp} [{layers}]")

    # Also test embed and lmhead
    for label, component in [('embed', 'embed'), ('lmhead', 'lmhead')]:
        solo_ms = np.mean(solo_results[component]) / 1e6

        # These are simpler — just run them concurrently
        barrier = threading.Barrier(2)

        def worker(times_out):
            for _ in range(warmup):
                if label == 'embed':
                    run_embed_once(models)
                else:
                    run_lmhead_once(models, make_dummy_hidden())
            for _ in range(iterations):
                barrier.wait()
                if label == 'embed':
                    _, ns = run_embed_once(models)
                else:
                    _, ns = run_lmhead_once(models, make_dummy_hidden())
                times_out.append(ns)

        t0_times, t1_times = [], []
        threads = [
            threading.Thread(target=worker, args=(t0_times,)),
            threading.Thread(target=worker, args=(t1_times,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        conc_ms = (np.mean(t0_times) + np.mean(t1_times)) / 2 / 1e6
        ratio = conc_ms / solo_ms if solo_ms > 0 else 0

        if ratio > 1.8:
            status = "SATURATED"
            interp = "1 call fills ANE"
        elif ratio > 1.3:
            status = "PARTIAL"
            interp = "some spare capacity"
        else:
            status = "NOT SAT"
            interp = "significant spare capacity"

        print(f"  {label:<12} {solo_ms:<10.3f} {conc_ms:<10.3f} "
              f"{ratio:<8.2f} {status:<12} {interp}")


def report_cross_chunk_concurrent(models, solo_results, iterations, warmup):
    """Test if two DIFFERENT chunks can overlap on ANE."""
    print(f"\n  CROSS-CHUNK CONCURRENCY: Can different chunks overlap?")
    print(f"  {'Pair':<16} {'Solo sum(ms)':<14} {'Conc max(ms)':<14} "
          f"{'Overlap ratio':<14} {'Interpretation'}")
    print(f"  {'-'*70}")

    pairs = [(1, 5), (2, 6), (0, 8)]  # different sizes
    for ci, cj in pairs:
        solo_i = np.mean(solo_results[f'chunk{ci}']) / 1e6
        solo_j = np.mean(solo_results[f'chunk{cj}']) / 1e6
        solo_sum = solo_i + solo_j
        solo_max = max(solo_i, solo_j)

        ti_times, tj_times = measure_concurrent_different(
            models, ci, cj, iterations, warmup)

        conc_i = np.mean(ti_times) / 1e6
        conc_j = np.mean(tj_times) / 1e6
        conc_max = max(conc_i, conc_j)

        # If fully parallel: conc_max ≈ solo_max (1.0x)
        # If fully serial: conc_max ≈ solo_sum (= solo_i + solo_j)
        if solo_max > 0:
            overlap_ratio = (solo_sum - conc_max) / (solo_sum - solo_max) if solo_sum > solo_max else 0
        else:
            overlap_ratio = 0

        if conc_max > solo_sum * 0.85:
            interp = "SERIALIZED"
        elif conc_max < solo_max * 1.15:
            interp = "PARALLEL"
        else:
            interp = "PARTIAL overlap"

        print(f"  c{ci}+c{cj}{'':>9} {solo_sum:<14.3f} {conc_max:<14.3f} "
              f"{overlap_ratio:<14.2f} {interp}")


def report_pipeline_gaps(pipeline_data):
    """Analyze inter-call gaps in the sequential decode pipeline."""
    print(f"\n  PIPELINE TIMING (sequential decode, per-token):")

    all_durations = {f'chunk{ci}': [] for ci in range(NUM_CHUNKS)}
    all_durations['embed'] = []
    all_durations['lmhead'] = []
    all_gaps = []  # gap between end of call N and start of call N+1

    for step_data in pipeline_data:
        for i, (name, start, end) in enumerate(step_data):
            dur_us = (end - start) / 1000
            base_name = name
            if base_name in all_durations:
                all_durations[base_name].append(dur_us)

            # Measure gap to next call
            if i + 1 < len(step_data):
                next_start = step_data[i + 1][1]
                gap_us = (next_start - end) / 1000
                all_gaps.append((name, step_data[i + 1][0], gap_us))

    # Summarize gaps
    gap_by_pair = {}
    for src, dst, gap_us in all_gaps:
        key = f"{src}→{dst}"
        gap_by_pair.setdefault(key, []).append(gap_us)

    print(f"\n  Inter-call dispatch gaps (µs):")
    print(f"  {'Transition':<25} {'Mean(µs)':<10} {'P50':<8} {'P95':<8} {'Max':<8}")
    print(f"  {'-'*63}")
    total_gap_us = 0
    for key in sorted(gap_by_pair.keys()):
        arr = np.array(gap_by_pair[key])
        total_gap_us += np.mean(arr)
        print(f"  {key:<25} {np.mean(arr):<10.1f} {np.percentile(arr, 50):<8.1f} "
              f"{np.percentile(arr, 95):<8.1f} {np.max(arr):<8.1f}")

    # Total predict time vs total gap
    total_predict_us = sum(np.mean(v) for v in all_durations.values())
    print(f"\n  Summary:")
    print(f"    Total predict() time:    {total_predict_us:.1f} µs ({total_predict_us/1000:.2f} ms)")
    print(f"    Total dispatch gaps:     {total_gap_us:.1f} µs ({total_gap_us/1000:.3f} ms)")
    print(f"    Gap fraction:            {100*total_gap_us/total_predict_us:.3f}%")
    print(f"    Effective utilization:   {100*(1 - total_gap_us/(total_predict_us + total_gap_us)):.2f}%")


def report_concurrent_pipeline_overlap(data):
    """Analyze whether 2-stream concurrent pipeline calls actually overlap."""
    print(f"\n  2-STREAM CONCURRENT PIPELINE OVERLAP ANALYSIS:")

    # Collect all intervals
    s0_intervals = []
    s1_intervals = []
    for step_data in data[0]:
        for name, start, end in step_data:
            s0_intervals.append((start, end, name))
    for step_data in data[1]:
        for name, start, end in step_data:
            s1_intervals.append((start, end, name))

    # Check pairwise overlap
    overlap_ns = 0
    overlap_count = 0
    total_s0_ns = sum(end - start for start, end, _ in s0_intervals)
    total_s1_ns = sum(end - start for start, end, _ in s1_intervals)

    # Merge and sort all intervals by start time
    all_ints = [(s, e, 0, n) for s, e, n in s0_intervals] + \
               [(s, e, 1, n) for s, e, n in s1_intervals]
    all_ints.sort()

    for i in range(len(all_ints)):
        for j in range(i + 1, min(i + 30, len(all_ints))):
            si_s, si_e, si_id, si_n = all_ints[i]
            sj_s, sj_e, sj_id, sj_n = all_ints[j]
            if si_id == sj_id:
                continue
            if sj_s >= si_e:
                break
            ov = min(si_e, sj_e) - max(si_s, sj_s)
            if ov > 1000:  # > 1µs
                overlap_ns += ov
                overlap_count += 1

    total_compute_ns = total_s0_ns + total_s1_ns
    print(f"    Stream 0 total predict: {total_s0_ns/1e6:.1f} ms")
    print(f"    Stream 1 total predict: {total_s1_ns/1e6:.1f} ms")
    print(f"    Overlapping intervals:  {overlap_count}")
    print(f"    Total overlap:          {overlap_ns/1e6:.1f} ms")
    if total_compute_ns > 0:
        print(f"    Overlap fraction:       {100*overlap_ns/total_compute_ns:.1f}%")

    if overlap_count > 0 and overlap_ns > total_compute_ns * 0.05:
        print(f"    → Significant host-level overlap detected")
        print(f"    → But wall-clock slowdown shows ANE serializes internally")
    else:
        print(f"    → Minimal overlap — ANE serializes predict() calls")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ANE saturation measurement for decode chunks")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--iterations", type=int, default=50,
                        help="Iterations per measurement")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-concurrent-pipeline", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("  ANE SATURATION BENCHMARK — Qwen3.5-4B Decode (9-chunk)")
    print(f"  Device: Mac Studio M4 Pro (10P+4E, 36GB)")
    print(f"  CTX={CTX}, BATCH={BATCH_SIZE}, Chunks={NUM_CHUNKS}")
    print(f"  Iterations: {args.iterations}, Warmup: {args.warmup}")
    print(f"  Chunk partition: {CHUNK_RANGES}")
    print("=" * 78)

    # Load models
    print(f"\n  LOADING MODELS")
    import resource
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    models = Models(args.model_dir)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"  Memory: {rss0:.0f} → {rss1:.0f} MB (+{rss1-rss0:.0f} MB)")

    # ── Test 1: Solo latency ─────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  TEST 1: SOLO COMPONENT LATENCY")
    print(f"{'='*78}")
    solo = measure_solo(models, args.iterations, args.warmup)
    report_solo(solo)

    # ── Test 2: Same-chunk concurrent (saturation test) ──────────
    print(f"\n{'='*78}")
    print(f"  TEST 2: ANE SATURATION (2-thread concurrent, same chunk)")
    print(f"{'='*78}")
    report_saturation(models, solo, args.iterations, args.warmup)

    # ── Test 3: Cross-chunk concurrent ───────────────────────────
    print(f"\n{'='*78}")
    print(f"  TEST 3: CROSS-CHUNK CONCURRENCY")
    print(f"{'='*78}")
    report_cross_chunk_concurrent(models, solo, args.iterations, args.warmup)

    # ── Test 4: Pipeline gap analysis ────────────────────────────
    if not args.skip_pipeline:
        print(f"\n{'='*78}")
        print(f"  TEST 4: PIPELINE DISPATCH GAP ANALYSIS")
        print(f"{'='*78}")
        pipeline_data = measure_pipeline_timing(models, args.iterations, args.warmup)
        report_pipeline_gaps(pipeline_data)

    # ── Test 5: 2-stream concurrent pipeline overlap ─────────────
    if not args.skip_concurrent_pipeline:
        print(f"\n{'='*78}")
        print(f"  TEST 5: 2-STREAM CONCURRENT PIPELINE OVERLAP")
        print(f"{'='*78}")
        conc_data = measure_concurrent_pipeline(models, min(args.iterations, 20), args.warmup)
        report_concurrent_pipeline_overlap(conc_data)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  SUMMARY & CONCLUSION")
    print(f"{'='*78}")

    chunk_solos = [np.mean(solo[f'chunk{ci}']) / 1e6 for ci in range(NUM_CHUNKS)]
    total_ffn = sum(chunk_solos)
    embed_ms = np.mean(solo['embed']) / 1e6
    lmhead_ms = np.mean(solo['lmhead']) / 1e6
    total_step = total_ffn + embed_ms + lmhead_ms

    print(f"\n  Decode token budget:")
    print(f"    Embed:     {embed_ms:>8.2f} ms  ({100*embed_ms/total_step:>5.1f}%)")
    for ci in range(NUM_CHUNKS):
        print(f"    Chunk {ci}:   {chunk_solos[ci]:>8.2f} ms  ({100*chunk_solos[ci]/total_step:>5.1f}%)")
    print(f"    LM Head:   {lmhead_ms:>8.2f} ms  ({100*lmhead_ms/total_step:>5.1f}%)")
    print(f"    Total:     {total_step:>8.2f} ms  → {1000/total_step:.1f} tok/s")

    print(f"\n{'='*78}")
    print(f"  Benchmark complete.")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
