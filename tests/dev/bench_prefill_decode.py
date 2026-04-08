#!/usr/bin/env python3
"""Benchmark prefill and decode speed for Qwen3.5-4B on ANE.

Measures:
  1. Batch prefill latency (BATCH_SIZE tokens via prefill function)
  2. Single-token decode throughput (via infer function)
  3. End-to-end generation (prefill + decode combined)

Usage:
    python tests/dev/bench_prefill_decode.py --model-dir /path/to/models
    python tests/dev/bench_prefill_decode.py \
        --model-dir /Users/yw68/Anemll/qwen3_5_flll_9chunk \
        --model-dir2 /Users/yw68/Anemll/qwen3_5_flll_9chunk_lut4 \
        --decode-tokens 60 --warmup 3 --trials 5
"""
import sys, os, time, argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts_qwen3_5")
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _REPO_ROOT)

import coremltools as ct
from transformers import AutoTokenizer
from config import CTX, NUM_CHUNKS, BATCH_SIZE, FFN_LABEL, DEFAULT_HF_MODEL

CU = ct.ComputeUnit.CPU_AND_NE

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


class Engine:
    """Lightweight inference engine for benchmarking."""

    def __init__(self, model_dir, label=""):
        self.label = label
        self.model_dir = model_dir

        # Detect combined directory (may be symlinked)
        self.combined_dir = os.path.join(model_dir, f"combined_{FFN_LABEL}_dedup")
        self.use_combined = os.path.isdir(self.combined_dir)

        print(f"\n  Loading {label} from {model_dir}")
        print(f"  Mode: {'COMBINED' if self.use_combined else 'SEPARATE'}")

        # Embeddings
        t0 = time.time()
        self.embed = _load_model(_find_model(model_dir, "embeddings"), CU)
        print(f"  embeddings: {time.time()-t0:.1f}s")

        # LM Head
        t0 = time.time()
        try:
            self.lmhead = _load_model(_find_model(model_dir, "lm_head_logits"), CU)
        except FileNotFoundError:
            self.lmhead = _load_model(_find_model(model_dir, "lm_head"), CU)
        spec = self.lmhead.get_spec()
        out_names = [o.name for o in spec.description.output]
        # Detect split logits (logits1..logitsN) vs single logits vs argmax
        self.logits_keys = sorted([n for n in out_names if n.startswith("logits")],
                                  key=lambda x: int(x.replace("logits", "") or "0"))
        if self.logits_keys:
            self.lmhead_mode = "split_logits"
        elif "output_logits" in out_names:
            self.lmhead_mode = "logits"
            self.logits_key = "output_logits"
        elif "logits" in out_names:
            self.lmhead_mode = "logits"
            self.logits_key = "logits"
        else:
            self.lmhead_mode = "argmax"
        print(f"  lm_head: {time.time()-t0:.1f}s (mode={self.lmhead_mode}, outputs={len(out_names)})")

        # FFN chunks (infer + prefill)
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
            print(f"  chunk{ci}: {time.time()-t0:.1f}s", end="")
            if m_prefill is None:
                print(" (no prefill!)", end="")
            print()

        self.has_prefill = all(p is not None for p in self.prefills)

        # Detect state shapes from infer function
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

        self._reset()

    def _get_next_id(self, lm_out):
        """Extract next token ID from lm_head output."""
        if self.lmhead_mode == "split_logits":
            parts = [lm_out[k].flatten().astype(np.float32) for k in self.logits_keys]
            logits = np.concatenate(parts)
            return int(np.argmax(logits))
        elif self.lmhead_mode == "logits":
            return int(np.argmax(lm_out[self.logits_key].flatten()))
        else:
            return int(lm_out["argmax_idx"].flatten()[0])

    def _reset(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [np.zeros(self.conv_shapes[ci], dtype=np.float16)
                          for ci in range(NUM_CHUNKS)]
        self.lin_recs = [np.zeros(self.rec_shapes[ci], dtype=np.float16)
                         for ci in range(NUM_CHUNKS)]

    # ── Batch prefill ────────────────────────────────────────────

    def batch_prefill(self, token_ids):
        """Run batch prefill for all prompt tokens. Returns (next_token_id, timings_dict)."""
        valid_len = len(token_ids)
        assert valid_len <= BATCH_SIZE, f"Prompt {valid_len} > BATCH_SIZE {BATCH_SIZE}"
        self._reset()

        # Embed
        input_ids = np.zeros((1, BATCH_SIZE), dtype=np.int32)
        input_ids[0, :valid_len] = token_ids

        t_embed = time.perf_counter()
        hidden = list(self.embed.predict({"input_ids": input_ids}).values())[0]
        t_embed = time.perf_counter() - t_embed

        # Causal mask [1, 1, BATCH_SIZE, CTX]
        mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
        for i in range(valid_len):
            mask[0, 0, i, :i + 1] = 0

        pos_ids = np.zeros((BATCH_SIZE,), dtype=np.int32)
        pos_ids[:valid_len] = np.arange(valid_len, dtype=np.int32)
        cur_pos = np.array([0], dtype=np.int32)
        valid_len_arr = np.array([valid_len], dtype=np.int32)

        # FFN chunks
        t_chunks = []
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": cur_pos,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
                "valid_len": valid_len_arr,
            }
            t0 = time.perf_counter()
            out = self.prefills[ci].predict(inp, state=self.states[ci])
            t_chunks.append(time.perf_counter() - t0)
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        # LM Head on last valid token
        if hidden.ndim >= 3 and hidden.shape[1] > 1:
            hidden = hidden[:, valid_len - 1:valid_len, :]

        t_lmhead = time.perf_counter()
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        t_lmhead = time.perf_counter() - t_lmhead

        next_id = self._get_next_id(lm_out)

        return next_id, {
            'embed': t_embed,
            'chunks': t_chunks,
            'lmhead': t_lmhead,
            'total': t_embed + sum(t_chunks) + t_lmhead,
        }

    # ── Single-token decode ──────────────────────────────────────

    def decode_step(self, tok_id, pos):
        """Run single-token decode. Returns (next_token_id, timings_dict)."""
        tok = np.array([[tok_id]], dtype=np.int32)

        t_embed = time.perf_counter()
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        t_embed = time.perf_counter() - t_embed

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        pos_arr = np.array([pos], dtype=np.int32)

        t_chunks = []
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
            t_chunks.append(time.perf_counter() - t0)
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        t_lmhead = time.perf_counter()
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        t_lmhead = time.perf_counter() - t_lmhead

        next_id = self._get_next_id(lm_out)

        return next_id, {
            'embed': t_embed,
            'chunks': t_chunks,
            'lmhead': t_lmhead,
            'total': t_embed + sum(t_chunks) + t_lmhead,
        }


# ── Benchmark runner ─────────────────────────────────────────────

def bench_one(engine, prompt_ids, decode_tokens, warmup, trials, stop_ids):
    """Benchmark a single engine. Returns results dict."""
    valid_len = len(prompt_ids)

    # ── Batch prefill benchmark ──────────────────────────────────
    print(f"\n  --- Batch Prefill ({valid_len} tokens, BATCH={BATCH_SIZE}) ---")
    if not engine.has_prefill:
        print(f"  SKIPPED: no prefill models available")
        prefill_results = None
    else:
        # Warmup
        for w in range(warmup):
            nid, t = engine.batch_prefill(prompt_ids)
            print(f"    warmup {w+1}: {t['total']*1000:.1f} ms")

        pf_timings = []
        for trial in range(trials):
            nid, t = engine.batch_prefill(prompt_ids)
            pf_timings.append(t)
            chunk_str = " ".join(f"{c*1000:.0f}" for c in t['chunks'])
            print(f"    trial {trial+1}: {t['total']*1000:.1f} ms  "
                  f"[emb={t['embed']*1000:.1f} ffn=[{chunk_str}] lm={t['lmhead']*1000:.1f}]")

        pf_totals = [t['total'] for t in pf_timings]
        prefill_results = {
            'mean_ms': np.mean(pf_totals) * 1000,
            'std_ms': np.std(pf_totals) * 1000,
            'min_ms': np.min(pf_totals) * 1000,
            'tok_per_sec': valid_len / np.mean(pf_totals),
            'embed_ms': np.mean([t['embed'] for t in pf_timings]) * 1000,
            'ffn_ms': np.mean([sum(t['chunks']) for t in pf_timings]) * 1000,
            'lmhead_ms': np.mean([t['lmhead'] for t in pf_timings]) * 1000,
            'chunk_ms': [np.mean([t['chunks'][ci] for t in pf_timings]) * 1000
                         for ci in range(NUM_CHUNKS)],
        }

    # ── Decode benchmark ─────────────────────────────────────────
    print(f"\n  --- Decode ({decode_tokens} tokens) ---")

    # Set up state via prefill first
    if engine.has_prefill:
        first_tok, _ = engine.batch_prefill(prompt_ids)
    else:
        # Sequential prefill fallback
        engine._reset()
        for i, tid in enumerate(prompt_ids):
            first_tok, _ = engine.decode_step(tid, i)

    start_pos = valid_len

    # Warmup decode
    engine._reset()
    if engine.has_prefill:
        first_tok, _ = engine.batch_prefill(prompt_ids)
    else:
        for i, tid in enumerate(prompt_ids):
            first_tok, _ = engine.decode_step(tid, i)

    for w in range(min(warmup, 3)):
        _, t = engine.decode_step(first_tok, start_pos + w)
        print(f"    warmup {w+1}: {t['total']*1000:.1f} ms/tok")

    # Full decode benchmark: prefill then generate decode_tokens
    decode_timings = []
    for trial in range(trials):
        engine._reset()
        if engine.has_prefill:
            cur_tok, _ = engine.batch_prefill(prompt_ids)
        else:
            for i, tid in enumerate(prompt_ids):
                cur_tok, _ = engine.decode_step(tid, i)

        trial_times = []
        gen_tokens = []
        for di in range(decode_tokens):
            pos = start_pos + di
            if pos >= CTX - 1:
                break
            cur_tok, t = engine.decode_step(cur_tok, pos)
            trial_times.append(t)
            gen_tokens.append(cur_tok)
            if cur_tok in stop_ids:
                break

        decode_timings.append(trial_times)
        total_ms = sum(t['total'] for t in trial_times) * 1000
        n = len(trial_times)
        avg = total_ms / n if n > 0 else 0
        print(f"    trial {trial+1}: {n} tok in {total_ms:.0f} ms "
              f"({1000/avg:.1f} tok/s, {avg:.1f} ms/tok)")

    # Aggregate decode results
    all_step_ms = []
    all_embed_ms = []
    all_ffn_ms = []
    all_lmhead_ms = []
    all_chunk_ms = [[] for _ in range(NUM_CHUNKS)]

    for trial_times in decode_timings:
        for t in trial_times:
            all_step_ms.append(t['total'] * 1000)
            all_embed_ms.append(t['embed'] * 1000)
            all_ffn_ms.append(sum(t['chunks']) * 1000)
            all_lmhead_ms.append(t['lmhead'] * 1000)
            for ci in range(NUM_CHUNKS):
                all_chunk_ms[ci].append(t['chunks'][ci] * 1000)

    decode_results = {
        'mean_ms': np.mean(all_step_ms),
        'std_ms': np.std(all_step_ms),
        'p50_ms': np.percentile(all_step_ms, 50),
        'p95_ms': np.percentile(all_step_ms, 95),
        'min_ms': np.min(all_step_ms),
        'tok_per_sec': 1000 / np.mean(all_step_ms),
        'embed_ms': np.mean(all_embed_ms),
        'ffn_ms': np.mean(all_ffn_ms),
        'lmhead_ms': np.mean(all_lmhead_ms),
        'chunk_ms': [np.mean(all_chunk_ms[ci]) for ci in range(NUM_CHUNKS)],
        'n_steps': len(all_step_ms),
        'n_trials': trials,
    }

    return prefill_results, decode_results


def print_results(label, prefill, decode, prompt_len):
    """Pretty-print benchmark results for one engine."""
    print(f"\n{'='*75}")
    print(f"  {label}")
    print(f"{'='*75}")

    if prefill is not None:
        print(f"\n  PREFILL ({prompt_len} tokens, batch={BATCH_SIZE}):")
        print(f"    Total:          {prefill['mean_ms']:>8.1f} ± {prefill['std_ms']:.1f} ms")
        print(f"    Throughput:     {prefill['tok_per_sec']:>8.0f} tok/s")
        print(f"    Breakdown:")
        print(f"      Embed:        {prefill['embed_ms']:>8.1f} ms")
        print(f"      FFN total:    {prefill['ffn_ms']:>8.1f} ms")
        for ci, cms in enumerate(prefill['chunk_ms']):
            print(f"        chunk{ci}:    {cms:>8.1f} ms")
        print(f"      LM Head:      {prefill['lmhead_ms']:>8.1f} ms")

    print(f"\n  DECODE ({decode['n_steps']//decode['n_trials']} tok × {decode['n_trials']} trials):")
    print(f"    Mean:           {decode['mean_ms']:>8.1f} ms/tok")
    print(f"    P50:            {decode['p50_ms']:>8.1f} ms/tok")
    print(f"    P95:            {decode['p95_ms']:>8.1f} ms/tok")
    print(f"    Throughput:     {decode['tok_per_sec']:>8.1f} tok/s")
    print(f"    Breakdown (per token):")
    pct_e = 100 * decode['embed_ms'] / decode['mean_ms']
    pct_f = 100 * decode['ffn_ms'] / decode['mean_ms']
    pct_l = 100 * decode['lmhead_ms'] / decode['mean_ms']
    pct_o = 100 - pct_e - pct_f - pct_l
    print(f"      Embed:        {decode['embed_ms']:>8.2f} ms  ({pct_e:>5.1f}%)")
    print(f"      FFN total:    {decode['ffn_ms']:>8.2f} ms  ({pct_f:>5.1f}%)")
    for ci, cms in enumerate(decode['chunk_ms']):
        print(f"        chunk{ci}:    {cms:>8.2f} ms")
    print(f"      LM Head:      {decode['lmhead_ms']:>8.2f} ms  ({pct_l:>5.1f}%)")
    print(f"      Overhead:     {decode['mean_ms'] - decode['embed_ms'] - decode['ffn_ms'] - decode['lmhead_ms']:>8.2f} ms  ({pct_o:>5.1f}%)")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prefill & decode speed for Qwen3.5-4B on ANE")
    parser.add_argument("--model-dir", required=True,
                        help="Primary model directory (e.g. LUT6)")
    parser.add_argument("--model-dir2",
                        help="Optional second model directory (e.g. LUT4) for comparison")
    parser.add_argument("--label", default="LUT6", help="Label for primary model")
    parser.add_argument("--label2", default="LUT4", help="Label for second model")
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL,
                        help="HuggingFace model for tokenizer")
    parser.add_argument("--prompt", default="Explain what a neural network is in simple terms.",
                        help="Prompt text for benchmarking")
    parser.add_argument("--decode-tokens", type=int, default=60,
                        help="Number of decode tokens to generate")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

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

    print("=" * 75)
    print("  Qwen3.5-4B Prefill & Decode Benchmark (ANE)")
    print(f"  CTX={CTX}, BATCH={BATCH_SIZE}, Chunks={NUM_CHUNKS}")
    print(f"  Prompt: {prompt_len} tokens, Decode: {args.decode_tokens} tokens")
    print(f"  Warmup: {args.warmup}, Trials: {args.trials}")
    print(f"  Stop IDs: {stop_ids}")
    print("=" * 75)

    results = []

    # ── Primary model ────────────────────────────────────────────
    engine1 = Engine(args.model_dir, label=args.label)
    pf1, dc1 = bench_one(engine1, prompt_ids, args.decode_tokens,
                          args.warmup, args.trials, stop_ids)
    results.append((args.label, pf1, dc1))
    del engine1

    # ── Optional second model ────────────────────────────────────
    if args.model_dir2:
        import gc; gc.collect()
        engine2 = Engine(args.model_dir2, label=args.label2)
        pf2, dc2 = bench_one(engine2, prompt_ids, args.decode_tokens,
                              args.warmup, args.trials, stop_ids)
        results.append((args.label2, pf2, dc2))
        del engine2

    # ── Print results ────────────────────────────────────────────
    for label, pf, dc in results:
        print_results(label, pf, dc, prompt_len)

    # ── Comparison table ─────────────────────────────────────────
    if len(results) == 2:
        l1, pf1, dc1 = results[0]
        l2, pf2, dc2 = results[1]

        print(f"\n{'='*75}")
        print(f"  COMPARISON: {l1} vs {l2}")
        print(f"{'='*75}")

        print(f"\n  {'Metric':<30} {l1:>15} {l2:>15} {'Ratio':>10}")
        print(f"  {'-'*72}")

        if pf1 and pf2:
            print(f"  {'Prefill (ms)':<30} {pf1['mean_ms']:>14.1f} {pf2['mean_ms']:>14.1f} "
                  f"{pf2['mean_ms']/pf1['mean_ms']:>9.2f}x")
            print(f"  {'Prefill (tok/s)':<30} {pf1['tok_per_sec']:>14.0f} {pf2['tok_per_sec']:>14.0f} "
                  f"{pf2['tok_per_sec']/pf1['tok_per_sec']:>9.2f}x")

        print(f"  {'Decode mean (ms/tok)':<30} {dc1['mean_ms']:>14.1f} {dc2['mean_ms']:>14.1f} "
              f"{dc2['mean_ms']/dc1['mean_ms']:>9.2f}x")
        print(f"  {'Decode P50 (ms/tok)':<30} {dc1['p50_ms']:>14.1f} {dc2['p50_ms']:>14.1f} "
              f"{dc2['p50_ms']/dc1['p50_ms']:>9.2f}x")
        print(f"  {'Decode (tok/s)':<30} {dc1['tok_per_sec']:>14.1f} {dc2['tok_per_sec']:>14.1f} "
              f"{dc2['tok_per_sec']/dc1['tok_per_sec']:>9.2f}x")
        print(f"  {'Decode FFN (ms/tok)':<30} {dc1['ffn_ms']:>14.2f} {dc2['ffn_ms']:>14.2f} "
              f"{dc2['ffn_ms']/dc1['ffn_ms']:>9.2f}x")
        print(f"  {'Decode Embed (ms/tok)':<30} {dc1['embed_ms']:>14.2f} {dc2['embed_ms']:>14.2f} "
              f"{dc2['embed_ms']/dc1['embed_ms']:>9.2f}x")
        print(f"  {'Decode LM Head (ms/tok)':<30} {dc1['lmhead_ms']:>14.2f} {dc2['lmhead_ms']:>14.2f} "
              f"{dc2['lmhead_ms']/dc1['lmhead_ms']:>9.2f}x")

    print(f"\n{'='*75}")
    print(f"  Benchmark complete.")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
