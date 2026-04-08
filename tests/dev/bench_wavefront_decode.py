#!/usr/bin/env python3
"""Wavefront decode scheduling benchmark for Qwen3.5-4B (9-chunk).

Answers: Does wavefront scheduling improve decode performance?

Dependency analysis:
  - hidden_states flows sequentially: chunk0→chunk1→...→chunk8
  - chunk i+1 CANNOT start until chunk i finishes (hard data dependency)
  - linear_conv_state / linear_recurrent_state are per-chunk (independent)
  - KV cache (CoreML state) is per-chunk (independent)
  - current_pos is read-only (same value for all chunks within a token)

  ⇒ Single-stream: ZERO opportunity for chunk-level overlap
  ⇒ Multi-stream: chunk i of stream A is independent of chunk j of stream B
     BUT does ANE actually execute them concurrently?

Benchmarks:
  A) Baseline: sequential decode (current path)
  B) Threaded multi-stream: concurrent sequences from separate threads
  C) Inter-chunk gap measurement: how much host overhead exists between chunks

Usage:
    python tests/dev/bench_wavefront_decode.py \
        --model-dir /Users/yw68/Anemll/qwen3_5_flll_9chunk \
        --decode-tokens 128
"""
import sys, os, time, argparse, threading, gc
import numpy as np
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts_qwen3_5")
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _REPO_ROOT)

import coremltools as ct
from transformers import AutoTokenizer
from config import CTX, NUM_CHUNKS, BATCH_SIZE, FFN_LABEL, DEFAULT_HF_MODEL, CHUNK_RANGES

CU = ct.ComputeUnit.CPU_AND_NE

# ── Model loading (reused from bench_prefill_decode.py) ──────────

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


# ── Engine: shared model loading, per-stream state ───────────────

class SharedModels:
    """Models loaded once, shared across streams (read-only)."""

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
        print(f"  lm_head: {time.time()-t0:.1f}s ({len(self.logits_keys)} split logits)")

        self.ffns = []
        self.prefills = []
        for ci in range(NUM_CHUNKS):
            t0 = time.time()
            if self.use_combined:
                path = _find_model(self.combined_dir, f"chunk{ci}")
                m_infer = _load_model(path, CU, function_name="infer")
                m_prefill = _load_model(path, CU, function_name="prefill")
            else:
                m_infer = _load_model(
                    _find_model(model_dir, f"ffn_{FFN_LABEL}_chunk{ci}"), CU)
                try:
                    m_prefill = _load_model(
                        _find_model(model_dir, f"prefill_{FFN_LABEL}_chunk{ci}"), CU)
                except FileNotFoundError:
                    m_prefill = None
            self.ffns.append(m_infer)
            self.prefills.append(m_prefill)
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

    def get_next_id(self, lm_out):
        parts = [lm_out[k].flatten().astype(np.float32) for k in self.logits_keys]
        logits = np.concatenate(parts)
        return int(np.argmax(logits)), logits


class StreamState:
    """Per-stream mutable state."""

    def __init__(self, models):
        self.m = models
        self.states = [m.make_state() for m in models.ffns]
        self.lin_convs = [np.zeros(models.conv_shapes[ci], dtype=np.float16)
                          for ci in range(NUM_CHUNKS)]
        self.lin_recs = [np.zeros(models.rec_shapes[ci], dtype=np.float16)
                         for ci in range(NUM_CHUNKS)]

    def reset(self):
        self.states = [m.make_state() for m in self.m.ffns]
        for ci in range(NUM_CHUNKS):
            self.lin_convs[ci][:] = 0
            self.lin_recs[ci][:] = 0


# ── Decode implementations ───────────────────────────────────────

def sequential_decode_step(models, stream, tok_id, pos):
    """Standard sequential decode: embed → chunk0 → chunk1 → ... → chunk8 → lmhead.
    Returns (next_id, logits, detailed_timings)."""
    timings = {}

    tok = np.array([[tok_id]], dtype=np.int32)
    t0 = time.perf_counter()
    hidden = list(models.embed.predict({"input_ids": tok}).values())[0]
    timings['embed'] = time.perf_counter() - t0

    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :pos + 1] = 0
    pos_arr = np.array([pos], dtype=np.int32)

    timings['chunks'] = []
    timings['gaps'] = []  # host overhead between chunks
    prev_end = time.perf_counter()

    for ci in range(NUM_CHUNKS):
        t_gap_start = time.perf_counter()
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_arr,
            "causal_mask": mask,
            "current_pos": pos_arr,
            "linear_conv_state": stream.lin_convs[ci],
            "linear_recurrent_state": stream.lin_recs[ci],
        }
        timings['gaps'].append(time.perf_counter() - prev_end)

        t0 = time.perf_counter()
        out = models.ffns[ci].predict(inp, state=stream.states[ci])
        t1 = time.perf_counter()
        timings['chunks'].append(t1 - t0)
        prev_end = t1

        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            stream.lin_convs[ci] = out['linear_conv_state_out']
            stream.lin_recs[ci] = out['linear_recurrent_state_out']

    t0 = time.perf_counter()
    lm_out = models.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    timings['lmhead'] = time.perf_counter() - t0

    next_id, logits = models.get_next_id(lm_out)
    timings['total'] = timings['embed'] + sum(timings['chunks']) + timings['lmhead']
    return next_id, logits, timings


def batch_prefill(models, stream, token_ids):
    """Batch prefill. Returns next_token_id."""
    valid_len = len(token_ids)
    stream.reset()

    input_ids = np.zeros((1, BATCH_SIZE), dtype=np.int32)
    input_ids[0, :valid_len] = token_ids
    hidden = list(models.embed.predict({"input_ids": input_ids}).values())[0]

    mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
    for i in range(valid_len):
        mask[0, 0, i, :i + 1] = 0
    pos_ids = np.zeros((BATCH_SIZE,), dtype=np.int32)
    pos_ids[:valid_len] = np.arange(valid_len, dtype=np.int32)
    cur_pos = np.array([0], dtype=np.int32)
    valid_len_arr = np.array([valid_len], dtype=np.int32)

    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": cur_pos,
            "linear_conv_state": stream.lin_convs[ci],
            "linear_recurrent_state": stream.lin_recs[ci],
            "valid_len": valid_len_arr,
        }
        out = models.prefills[ci].predict(inp, state=stream.states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            stream.lin_convs[ci] = out['linear_conv_state_out']
            stream.lin_recs[ci] = out['linear_recurrent_state_out']

    if hidden.ndim >= 3 and hidden.shape[1] > 1:
        hidden = hidden[:, valid_len - 1:valid_len, :]
    lm_out = models.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    next_id, _ = models.get_next_id(lm_out)
    return next_id


# ══════════════════════════════════════════════════════════════════
#  BENCHMARK A: Sequential single-stream decode (baseline)
# ══════════════════════════════════════════════════════════════════

def bench_sequential(models, prompt_ids, decode_tokens, warmup, trials, stop_ids):
    """Baseline sequential decode benchmark."""
    results = []

    for trial in range(-warmup, trials):
        stream = StreamState(models)
        first_tok = batch_prefill(models, stream, prompt_ids)
        start_pos = len(prompt_ids)

        gen_tokens = []
        all_timings = []
        cur_tok = first_tok

        t_total_start = time.perf_counter()
        for di in range(decode_tokens):
            pos = start_pos + di
            if pos >= CTX - 1:
                break
            cur_tok, logits, timings = sequential_decode_step(models, stream, cur_tok, pos)
            gen_tokens.append(cur_tok)
            all_timings.append(timings)
            if cur_tok in stop_ids:
                break
        t_total = time.perf_counter() - t_total_start

        n = len(gen_tokens)
        label = f"warmup {trial + warmup + 1}" if trial < 0 else f"trial {trial + 1}"
        tok_s = n / t_total if t_total > 0 else 0
        ms_tok = t_total * 1000 / n if n > 0 else 0
        print(f"    {label}: {n} tok in {t_total*1000:.0f} ms ({tok_s:.1f} tok/s, {ms_tok:.1f} ms/tok)")

        if trial >= 0:
            results.append({
                'gen_tokens': gen_tokens,
                'timings': all_timings,
                'total_time': t_total,
                'n_tokens': n,
            })

    return results


# ══════════════════════════════════════════════════════════════════
#  BENCHMARK B: Multi-stream concurrent decode
# ══════════════════════════════════════════════════════════════════

def bench_multistream(models, prompt_ids, decode_tokens, n_streams, warmup, trials, stop_ids):
    """Multi-stream decode: N streams run from separate threads.
    Tests whether ANE can overlap chunk execution across streams."""

    def stream_worker(stream_id, models, prompt_ids, decode_tokens, stop_ids,
                      result_dict, timing_dict, barrier):
        """Worker for one decode stream."""
        stream = StreamState(models)
        first_tok = batch_prefill(models, stream, prompt_ids)
        start_pos = len(prompt_ids)

        # Synchronize all streams to start decode together
        barrier.wait()

        gen_tokens = []
        timestamps = []  # (stream_id, event, time)
        cur_tok = first_tok

        t_start = time.perf_counter()
        for di in range(decode_tokens):
            pos = start_pos + di
            if pos >= CTX - 1:
                break

            tok = np.array([[cur_tok]], dtype=np.int32)
            t0 = time.perf_counter()
            hidden = list(models.embed.predict({"input_ids": tok}).values())[0]
            t_embed_end = time.perf_counter()
            timestamps.append((stream_id, 'embed', t0, t_embed_end))

            mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
            mask[:, :, :, :pos + 1] = 0
            pos_arr = np.array([pos], dtype=np.int32)

            for ci in range(NUM_CHUNKS):
                inp = {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": pos_arr,
                    "causal_mask": mask,
                    "current_pos": pos_arr,
                    "linear_conv_state": stream.lin_convs[ci],
                    "linear_recurrent_state": stream.lin_recs[ci],
                }
                t0 = time.perf_counter()
                out = models.ffns[ci].predict(inp, state=stream.states[ci])
                t1 = time.perf_counter()
                timestamps.append((stream_id, f'chunk{ci}', t0, t1))

                hidden = out["output_hidden_states"]
                if 'linear_conv_state_out' in out:
                    stream.lin_convs[ci] = out['linear_conv_state_out']
                    stream.lin_recs[ci] = out['linear_recurrent_state_out']

            t0 = time.perf_counter()
            lm_out = models.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
            t1 = time.perf_counter()
            timestamps.append((stream_id, 'lmhead', t0, t1))

            cur_tok, _ = models.get_next_id(lm_out)
            gen_tokens.append(cur_tok)
            if cur_tok in stop_ids:
                break

        t_total = time.perf_counter() - t_start
        result_dict[stream_id] = gen_tokens
        timing_dict[stream_id] = {
            'total_time': t_total,
            'n_tokens': len(gen_tokens),
            'timestamps': timestamps,
        }

    results = []
    for trial in range(-warmup, trials):
        result_dict = {}
        timing_dict = {}
        barrier = threading.Barrier(n_streams)

        threads = []
        for si in range(n_streams):
            t = threading.Thread(target=stream_worker,
                                 args=(si, models, prompt_ids, decode_tokens,
                                       stop_ids, result_dict, timing_dict, barrier))
            threads.append(t)

        t_wall_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t_wall = time.perf_counter() - t_wall_start

        total_tokens = sum(timing_dict[si]['n_tokens'] for si in range(n_streams))
        label = f"warmup {trial + warmup + 1}" if trial < 0 else f"trial {trial + 1}"
        per_stream_toks = [timing_dict[si]['n_tokens'] / timing_dict[si]['total_time']
                           for si in range(n_streams)]
        agg_toks = total_tokens / t_wall

        print(f"    {label}: {n_streams}×{decode_tokens} = {total_tokens} tok, "
              f"wall={t_wall*1000:.0f} ms, agg={agg_toks:.1f} tok/s, "
              f"per-stream=[{', '.join(f'{x:.1f}' for x in per_stream_toks)}] tok/s")

        if trial >= 0:
            results.append({
                'result_dict': result_dict,
                'timing_dict': timing_dict,
                'wall_time': t_wall,
                'total_tokens': total_tokens,
            })

    return results


# ══════════════════════════════════════════════════════════════════
#  BENCHMARK C: Inter-chunk gap & overlap analysis
# ══════════════════════════════════════════════════════════════════

def analyze_gaps(timings_list):
    """Analyze host-side overhead between chunk dispatches."""
    all_gaps = [[] for _ in range(NUM_CHUNKS)]
    all_chunks = [[] for _ in range(NUM_CHUNKS)]

    for timings in timings_list:
        for ci in range(NUM_CHUNKS):
            all_chunks[ci].append(timings['chunks'][ci] * 1000)
            if ci < len(timings['gaps']):
                all_gaps[ci].append(timings['gaps'][ci] * 1e6)  # microseconds

    print(f"\n  Per-chunk decode latency and inter-chunk gap:")
    print(f"  {'Chunk':>6} {'Layers':>8} {'Latency(ms)':>12} {'±std':>8} "
          f"{'Gap(µs)':>10} {'±std':>8}")
    print(f"  {'-'*56}")
    total_chunk_ms = 0
    total_gap_us = 0
    for ci in range(NUM_CHUNKS):
        layers = f"{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        cm = np.mean(all_chunks[ci])
        cs = np.std(all_chunks[ci])
        total_chunk_ms += cm
        if all_gaps[ci]:
            gm = np.mean(all_gaps[ci])
            gs = np.std(all_gaps[ci])
            total_gap_us += gm
            print(f"  {ci:>6} {layers:>8} {cm:>12.2f} {cs:>8.2f} {gm:>10.1f} {gs:>8.1f}")
        else:
            print(f"  {ci:>6} {layers:>8} {cm:>12.2f} {cs:>8.2f} {'N/A':>10}")
    print(f"  {'-'*56}")
    print(f"  {'Total':>6} {'':>8} {total_chunk_ms:>12.2f} {'':>8} {total_gap_us:>10.1f}")
    gap_pct = total_gap_us / (total_chunk_ms * 1000) * 100 if total_chunk_ms > 0 else 0
    print(f"  Gap overhead: {total_gap_us:.1f} µs = {gap_pct:.3f}% of chunk time")


def analyze_multistream_overlap(timing_dict, n_streams):
    """Check if chunks from different streams actually overlap in time."""
    if n_streams < 2:
        return

    # Collect all chunk intervals across streams
    all_intervals = []
    for si in range(n_streams):
        for entry in timing_dict[si]['timestamps']:
            sid, label, t_start, t_end = entry
            all_intervals.append((t_start, t_end, sid, label))

    all_intervals.sort(key=lambda x: x[0])

    # Check pairwise overlap between different streams
    overlap_count = 0
    overlap_total_ms = 0
    no_overlap_count = 0

    for i in range(len(all_intervals)):
        for j in range(i + 1, min(i + 20, len(all_intervals))):  # check nearby
            si_start, si_end, si_id, si_label = all_intervals[i]
            sj_start, sj_end, sj_id, sj_label = all_intervals[j]
            if si_id == sj_id:
                continue  # same stream
            if sj_start >= si_end:
                break  # no more overlaps possible

            # Overlap detected
            overlap_ms = (min(si_end, sj_end) - max(si_start, sj_start)) * 1000
            if overlap_ms > 0.01:  # > 10µs
                overlap_count += 1
                overlap_total_ms += overlap_ms

    print(f"\n  Cross-stream overlap analysis ({n_streams} streams):")
    print(f"    Overlapping intervals: {overlap_count}")
    print(f"    Total overlap: {overlap_total_ms:.1f} ms")
    if overlap_count > 0:
        print(f"    Average overlap: {overlap_total_ms/overlap_count:.2f} ms per pair")
        print(f"    → ANE DOES overlap execution across streams")
    else:
        print(f"    → ANE serializes execution — NO real overlap detected")


# ══════════════════════════════════════════════════════════════════
#  Correctness validation
# ══════════════════════════════════════════════════════════════════

def validate_correctness(baseline_results, multistream_results, n_streams):
    """Verify multi-stream produces same tokens as baseline (greedy decode)."""
    if not baseline_results or not multistream_results:
        return True

    baseline_tokens = baseline_results[0]['gen_tokens']  # reference from trial 1

    all_match = True
    for trial_idx, ms_result in enumerate(multistream_results):
        for si in range(n_streams):
            ms_tokens = ms_result['result_dict'][si]
            n = min(len(baseline_tokens), len(ms_tokens))
            match = baseline_tokens[:n] == ms_tokens[:n]
            if not match:
                # Find first divergence
                for k in range(n):
                    if baseline_tokens[k] != ms_tokens[k]:
                        print(f"  DIVERGENCE at trial {trial_idx+1} stream {si} "
                              f"token {k}: baseline={baseline_tokens[k]} "
                              f"multistream={ms_tokens[k]}")
                        all_match = False
                        break
            elif len(baseline_tokens) != len(ms_tokens):
                print(f"  LENGTH MISMATCH at trial {trial_idx+1} stream {si}: "
                      f"baseline={len(baseline_tokens)} multistream={len(ms_tokens)}")

    if all_match:
        print(f"  CORRECTNESS: All {n_streams} streams match baseline exactly")
    return all_match


# ══════════════════════════════════════════════════════════════════
#  Memory measurement
# ══════════════════════════════════════════════════════════════════

def get_rss_mb():
    """Get current process RSS in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wavefront decode scheduling benchmark for Qwen3.5-4B")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--streams", type=str, default="1,2,4",
                        help="Comma-separated stream counts for multi-stream test")
    parser.add_argument("--prompt", default="Explain what a neural network is in simple terms.")
    parser.add_argument("--skip-multistream", action="store_true",
                        help="Skip multi-stream benchmarks")
    args = parser.parse_args()

    stream_counts = [int(x) for x in args.streams.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)

    msgs = [{"role": "user", "content": args.prompt}]
    prompt_ids = list(tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        enable_thinking=True, return_dict=False))
    prompt_len = len(prompt_ids)

    print("=" * 78)
    print("  WAVEFRONT DECODE SCHEDULING BENCHMARK — Qwen3.5-4B (9-chunk)")
    print(f"  CTX={CTX}, BATCH={BATCH_SIZE}, Chunks={NUM_CHUNKS}")
    print(f"  Prompt: {prompt_len} tokens, Decode: {args.decode_tokens} tokens")
    print(f"  Warmup: {args.warmup}, Trials: {args.trials}")
    print(f"  Stream counts: {stream_counts}")
    print(f"  Chunk partition: {CHUNK_RANGES}")
    print("=" * 78)

    # ── Dependency analysis ──────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  SECTION 1: DECODE DEPENDENCY ANALYSIS")
    print(f"{'='*78}")
    print(f"""
  Decode pipeline for one token:
    embed(tok) → hidden
    chunk0(hidden) → hidden'    [layers 0-2, LLL]
    chunk1(hidden') → hidden''  [layers 3-6, FLLL]
    chunk2(hidden'') → ...      [layers 7-10, FLLL]
    ...
    chunk8(hidden) → hidden_out [layer 31, F]
    lmhead(hidden_out) → logits → argmax → next_tok

  HARD DATA DEPENDENCY: hidden flows chunk0 → chunk1 → ... → chunk8
  ⇒ Single-stream wavefront: CANNOT overlap chunks for same token
  ⇒ Multi-stream wavefront: chunk_i(stream_A) vs chunk_j(stream_B) independent
     BUT only if ANE hardware actually runs them concurrently

  Per-chunk state (independent, no cross-chunk dependency):
    - linear_conv_state: [{CHUNK_RANGES[0]}: {models_loaded_msg(0)}, ...]
    - linear_recurrent_state: same per-chunk
    - KV cache (CoreML state): per-chunk
""")

    # ── Load models ──────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  LOADING MODELS")
    print(f"{'='*78}")
    rss_before = get_rss_mb()
    models = SharedModels(args.model_dir)
    rss_after = get_rss_mb()
    print(f"  Memory: {rss_before:.0f} → {rss_after:.0f} MB (+{rss_after-rss_before:.0f} MB)")

    # ══════════════════════════════════════════════════════════════
    #  SECTION 2: BASELINE SEQUENTIAL DECODE
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*78}")
    print(f"  SECTION 2: BASELINE SEQUENTIAL DECODE (1 stream)")
    print(f"{'='*78}")

    baseline = bench_sequential(models, prompt_ids, args.decode_tokens,
                                args.warmup, args.trials, stop_ids)

    # Aggregate baseline results
    all_step_ms = []
    all_timings = []
    for r in baseline:
        for t in r['timings']:
            all_step_ms.append(t['total'] * 1000)
            all_timings.append(t)

    base_mean = np.mean(all_step_ms)
    base_p50 = np.percentile(all_step_ms, 50)
    base_p95 = np.percentile(all_step_ms, 95)
    base_toks = 1000 / base_mean

    print(f"\n  Baseline decode: {base_mean:.1f} ms/tok (P50={base_p50:.1f}, "
          f"P95={base_p95:.1f}), {base_toks:.1f} tok/s")

    # ── Inter-chunk gap analysis ─────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  SECTION 3: INTER-CHUNK GAP ANALYSIS")
    print(f"{'='*78}")
    analyze_gaps(all_timings)

    # Breakdown
    embed_ms = np.mean([t['embed'] for t in all_timings]) * 1000
    lmhead_ms = np.mean([t['lmhead'] for t in all_timings]) * 1000
    chunk_ms = [np.mean([t['chunks'][ci] for t in all_timings]) * 1000
                for ci in range(NUM_CHUNKS)]

    print(f"\n  Time breakdown per decode token:")
    print(f"    Embed:     {embed_ms:>8.2f} ms  ({100*embed_ms/base_mean:>5.1f}%)")
    for ci in range(NUM_CHUNKS):
        layers = f"L{CHUNK_RANGES[ci][0]}-{CHUNK_RANGES[ci][1]-1}"
        print(f"    Chunk {ci}:   {chunk_ms[ci]:>8.2f} ms  ({100*chunk_ms[ci]/base_mean:>5.1f}%)  [{layers}]")
    print(f"    LM Head:   {lmhead_ms:>8.2f} ms  ({100*lmhead_ms/base_mean:>5.1f}%)")
    overhead = base_mean - embed_ms - sum(chunk_ms) - lmhead_ms
    print(f"    Overhead:  {overhead:>8.2f} ms  ({100*overhead/base_mean:>5.1f}%)")

    if args.skip_multistream:
        print(f"\n  Skipping multi-stream benchmarks (--skip-multistream)")
        print_recommendation(base_mean, base_toks, None, [])
        return

    # ══════════════════════════════════════════════════════════════
    #  SECTION 4: MULTI-STREAM CONCURRENT DECODE
    # ══════════════════════════════════════════════════════════════
    ms_summaries = []
    for n_streams in stream_counts:
        if n_streams == 1:
            continue  # already benchmarked

        print(f"\n{'='*78}")
        print(f"  SECTION 4: MULTI-STREAM DECODE ({n_streams} concurrent streams)")
        print(f"{'='*78}")

        rss_before_ms = get_rss_mb()
        ms_results = bench_multistream(models, prompt_ids, args.decode_tokens,
                                        n_streams, args.warmup, args.trials, stop_ids)
        rss_after_ms = get_rss_mb()

        # Aggregate
        wall_times = [r['wall_time'] for r in ms_results]
        total_toks = [r['total_tokens'] for r in ms_results]
        agg_toks = [tt / wt for tt, wt in zip(total_toks, wall_times)]

        per_stream_times = []
        for r in ms_results:
            for si in range(n_streams):
                per_stream_times.append(
                    r['timing_dict'][si]['n_tokens'] / r['timing_dict'][si]['total_time'])

        wall_mean = np.mean(wall_times)
        agg_mean = np.mean(agg_toks)
        per_stream_mean = np.mean(per_stream_times)

        print(f"\n  {n_streams}-stream results:")
        print(f"    Wall time:        {wall_mean*1000:.0f} ms")
        print(f"    Aggregate:        {agg_mean:.1f} tok/s (all streams combined)")
        print(f"    Per-stream:       {per_stream_mean:.1f} tok/s")
        print(f"    vs baseline:      agg={agg_mean/base_toks:.2f}x, "
              f"per-stream={per_stream_mean/base_toks:.2f}x")
        print(f"    Memory:           {rss_before_ms:.0f} → {rss_after_ms:.0f} MB")

        ms_summaries.append({
            'n_streams': n_streams,
            'agg_toks': agg_mean,
            'per_stream_toks': per_stream_mean,
            'wall_mean': wall_mean,
        })

        # Overlap analysis (use last trial)
        if ms_results:
            analyze_multistream_overlap(ms_results[-1]['timing_dict'], n_streams)

        # Correctness check
        print(f"\n  Correctness check:")
        validate_correctness(baseline, ms_results, n_streams)

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY & RECOMMENDATION
    # ══════════════════════════════════════════════════════════════
    print_recommendation(base_mean, base_toks, ms_summaries, stream_counts)


def models_loaded_msg(ci):
    start, end = CHUNK_RANGES[ci]
    n = end - start
    return f"{n} layers"


def print_recommendation(base_mean, base_toks, ms_summaries, stream_counts):
    print(f"\n{'='*78}")
    print(f"  SUMMARY & RECOMMENDATION")
    print(f"{'='*78}")

    print(f"\n  Baseline single-stream decode: {base_mean:.1f} ms/tok ({base_toks:.1f} tok/s)")

    if ms_summaries:
        print(f"\n  {'Streams':>8} {'Agg tok/s':>12} {'Per-stream':>12} {'Agg speedup':>12} {'Per-stream':>12}")
        print(f"  {'-'*60}")
        print(f"  {'1':>8} {base_toks:>12.1f} {base_toks:>12.1f} {'1.00x':>12} {'1.00x':>12}")
        for ms in ms_summaries:
            agg_ratio = ms['agg_toks'] / base_toks
            ps_ratio = ms['per_stream_toks'] / base_toks
            print(f"  {ms['n_streams']:>8} {ms['agg_toks']:>12.1f} {ms['per_stream_toks']:>12.1f} "
                  f"{agg_ratio:>11.2f}x {ps_ratio:>11.2f}x")

    print(f"""
  DEPENDENCY STRUCTURE:
    Token decode: embed → chunk0 → chunk1 → ... → chunk8 → lmhead
    Hidden states flow SEQUENTIALLY through all 9 chunks.
    ⇒ Single-stream wavefront: IMPOSSIBLE — hard data dependency.
    ⇒ Per-token latency = Σ(all chunk latencies) — no pipeline possible.
""")

    if ms_summaries:
        best_agg = max(ms['agg_toks'] for ms in ms_summaries)
        best_ms = [ms for ms in ms_summaries if ms['agg_toks'] == best_agg][0]
        agg_speedup = best_agg / base_toks

        if agg_speedup > 1.15:
            print(f"  Multi-stream ({best_ms['n_streams']} streams) shows {agg_speedup:.2f}x "
                  f"aggregate throughput improvement.")
            print(f"  → ANE can overlap chunk execution across independent streams.")
            print(f"  RECOMMENDATION: (B) Use wavefront scheduling for multi-stream decode")
        elif agg_speedup > 1.02:
            print(f"  Multi-stream shows marginal improvement ({agg_speedup:.2f}x).")
            print(f"  → ANE mostly serializes execution across threads.")
            print(f"  RECOMMENDATION: (A) Keep sequential decode, or (D) reduce chunk count")
        else:
            print(f"  Multi-stream shows NO improvement ({agg_speedup:.2f}x).")
            print(f"  → ANE completely serializes predict() calls.")
            print(f"  RECOMMENDATION: (A) Keep sequential decode")
    else:
        print(f"  RECOMMENDATION: (A) Keep sequential decode — single-stream wavefront")
        print(f"  cannot help due to hard data dependency chain")

    print(f"""
  NEXT STEPS:
    1. (D) Reduce chunk count (9→4 or 9→2) to reduce host dispatch overhead
    2. (C) Focus on PREFILL wavefront (batch prefill is already parallelizable per-chunk)
    3. Compile to .mlmodelc (avoids JIT compilation overhead per predict())
""")

    print(f"{'='*78}")
    print(f"  Benchmark complete.")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
