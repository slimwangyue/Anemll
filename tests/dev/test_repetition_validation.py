#!/usr/bin/env python3
"""Milestone 2.1 Repetition Validation Test.

Tests the LUT6 gs=4 stable models with 20+ diverse prompts specifically
designed to probe repetition, coherence, and quality issues.

Prompt categories:
  - Long-form generation (story, essay, explanation) — most repetition-prone
  - Multi-step reasoning (math, logic, planning)
  - Creative writing (poetry, dialogue, humor)
  - Structured output (lists, code, recipes)
  - Multilingual (Chinese, English)

Metrics:
  - Repetition ratio: fraction of 4-gram repeats in output
  - Unique token ratio: unique tokens / total tokens
  - Output length: how many tokens before EOS (early stop = possible issue)
  - Text quality: manual-review flag for gibberish/degeneration

Usage:
    python tests/dev/test_repetition_validation.py
    python tests/dev/test_repetition_validation.py --model-dir /path/to/models
    python tests/dev/test_repetition_validation.py --max-new-tokens 128
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import coremltools as ct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

NEG_INF = np.float16(-65504.0)

# ─── Prompts designed to trigger repetition and quality issues ───────

PROMPTS = [
    # === Long-form generation (high repetition risk) ===
    {
        "prompt": "Write a detailed essay about the history of artificial intelligence, "
                  "from its origins in the 1950s to the present day.",
        "label": "en_essay_ai",
        "category": "long_form",
        "max_tokens": 300,
    },
    {
        "prompt": "Tell me a story about a dragon who befriends a young girl in a "
                  "small village. Include dialogue and a surprising twist ending.",
        "label": "en_story_dragon",
        "category": "long_form",
        "max_tokens": 300,
    },
    {
        "prompt": "Describe step by step how photosynthesis works, from light absorption "
                  "to glucose production. Be thorough and scientific.",
        "label": "en_explain_photosynthesis",
        "category": "long_form",
        "max_tokens": 300,
    },
    {
        "prompt": "写一篇关于中国古代四大发明的详细文章，介绍每项发明的历史背景、"
                  "发明过程和对世界的影响。",
        "label": "zh_essay_inventions",
        "category": "long_form",
        "max_tokens": 300,
    },
    {
        "prompt": "讲一个关于一只流浪猫和一位老人之间友情的温暖故事。",
        "label": "zh_story_cat",
        "category": "long_form",
        "max_tokens": 300,
    },

    # === Multi-step reasoning ===
    {
        "prompt": "A farmer has 3 fields. Field A produces 120 kg of wheat per hectare, "
                  "Field B produces 95 kg, and Field C produces 150 kg. If he has 5 hectares "
                  "of A, 8 hectares of B, and 3 hectares of C, what is his total wheat "
                  "production? Show your work step by step.",
        "label": "en_math_multistep",
        "category": "reasoning",
        "max_tokens": 150,
    },
    {
        "prompt": "If all roses are flowers, and some flowers fade quickly, can we conclude "
                  "that some roses fade quickly? Explain your reasoning carefully.",
        "label": "en_logic_syllogism",
        "category": "reasoning",
        "max_tokens": 150,
    },
    {
        "prompt": "Plan a 3-day trip to Tokyo for a first-time visitor. Include specific "
                  "neighborhoods, restaurants, and activities for each day.",
        "label": "en_planning_tokyo",
        "category": "reasoning",
        "max_tokens": 200,
    },
    {
        "prompt": "一个水池有两个进水管和一个出水管。进水管A每小时进水3吨，"
                  "进水管B每小时进水2吨，出水管每小时排水1吨。水池容量是40吨，"
                  "问多少小时能装满？请详细解答。",
        "label": "zh_math_pool",
        "category": "reasoning",
        "max_tokens": 150,
    },

    # === Creative writing (diversity-demanding) ===
    {
        "prompt": "Write a haiku about each of the four seasons. Label each one.",
        "label": "en_creative_haiku",
        "category": "creative",
        "max_tokens": 120,
    },
    {
        "prompt": "Write a short dialogue between a pessimistic robot and an optimistic "
                  "toaster about the meaning of existence.",
        "label": "en_creative_dialogue",
        "category": "creative",
        "max_tokens": 200,
    },
    {
        "prompt": "Invent a new word and write its dictionary entry: pronunciation, "
                  "part of speech, definition, and example sentences.",
        "label": "en_creative_newword",
        "category": "creative",
        "max_tokens": 150,
    },
    {
        "prompt": "用五言绝句的格式写四首诗，分别描写春、夏、秋、冬。",
        "label": "zh_creative_seasons",
        "category": "creative",
        "max_tokens": 150,
    },

    # === Structured output (lists, code, tables) ===
    {
        "prompt": "List the top 10 tallest mountains in the world with their heights "
                  "in meters and the countries they are located in.",
        "label": "en_list_mountains",
        "category": "structured",
        "max_tokens": 150,
    },
    {
        "prompt": "Write a Python class for a simple linked list with methods: append, "
                  "prepend, delete, find, and __str__.",
        "label": "en_code_linkedlist",
        "category": "structured",
        "max_tokens": 200,
    },
    {
        "prompt": "Compare TCP and UDP protocols. Present your answer as a table with "
                  "at least 6 comparison criteria.",
        "label": "en_table_protocols",
        "category": "structured",
        "max_tokens": 150,
    },
    {
        "prompt": "列出中国八大菜系，每个菜系写出代表菜、特点和发源地。",
        "label": "zh_list_cuisines",
        "category": "structured",
        "max_tokens": 200,
    },

    # === Edge cases (short answers, factual, adversarial) ===
    {
        "prompt": "What are the first 20 prime numbers?",
        "label": "en_factual_primes",
        "category": "factual",
        "max_tokens": 80,
    },
    {
        "prompt": "Translate the following to French: 'The quick brown fox jumps over "
                  "the lazy dog.'",
        "label": "en_translate_french",
        "category": "factual",
        "max_tokens": 60,
    },
    {
        "prompt": "Repeat the word 'hello' exactly 5 times, separated by commas.",
        "label": "en_adversarial_repeat",
        "category": "adversarial",
        "max_tokens": 60,
    },
    {
        "prompt": "Continue this pattern: 1, 1, 2, 3, 5, 8, 13, ...",
        "label": "en_pattern_fibonacci",
        "category": "adversarial",
        "max_tokens": 80,
    },
    {
        "prompt": "Write a paragraph where every sentence starts with the next letter "
                  "of the alphabet, starting from A.",
        "label": "en_adversarial_alphabet",
        "category": "creative",
        "max_tokens": 200,
    },
    {
        "prompt": "请用一句话总结《三国演义》的主题。",
        "label": "zh_factual_summary",
        "category": "factual",
        "max_tokens": 80,
    },

    # === Stress tests (designed to trigger repetition) ===
    {
        "prompt": "Explain the concept of recursion in computer science. Use multiple "
                  "examples. Then explain it again from a different angle. Then give "
                  "a third explanation for a 5-year-old.",
        "label": "en_stress_recursion",
        "category": "stress",
        "max_tokens": 400,
    },
    {
        "prompt": "Write a very long and detailed recipe for chocolate chip cookies. "
                  "Include every single step, measurement, timing, and tip you can think of.",
        "label": "en_stress_recipe",
        "category": "stress",
        "max_tokens": 400,
    },
    {
        "prompt": "Count from 1 to 50 and for each number write whether it is odd or even.",
        "label": "en_stress_counting",
        "category": "stress",
        "max_tokens": 400,
    },
    {
        "prompt": "Write a conversation between 5 different characters at a dinner party. "
                  "Each character should have a distinct personality and speaking style. "
                  "Continue the conversation for at least 20 exchanges.",
        "label": "en_stress_dialogue",
        "category": "stress",
        "max_tokens": 500,
    },
    {
        "prompt": "详细描述一天中从早到晚的所有活动安排，包括每个时间段做什么、"
                  "吃什么、去哪里。要非常详细。",
        "label": "zh_stress_schedule",
        "category": "stress",
        "max_tokens": 400,
    },
]


# ─── Repetition metrics ─────────────────────────────────────────────

def compute_ngram_repeat_ratio(token_ids: list[int], n: int = 4) -> float:
    """Fraction of n-grams that are repeated."""
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams) if ngrams else 0.0


def compute_unique_token_ratio(token_ids: list[int]) -> float:
    """Unique tokens / total tokens."""
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / len(token_ids)


def detect_degeneration(token_ids: list[int], window: int = 16) -> bool:
    """Detect if output degenerates into a short repeating loop."""
    if len(token_ids) < window * 2:
        return False
    # Check if the last `window` tokens repeat the previous window
    tail = token_ids[-window:]
    prev = token_ids[-2*window:-window]
    if tail == prev:
        return True
    # Check for very short loops (2-4 tokens repeating)
    for loop_len in range(2, 6):
        if len(token_ids) >= loop_len * 4:
            pattern = token_ids[-loop_len:]
            repeated = all(
                token_ids[-(i+1)] == pattern[-(i % loop_len) - 1]
                for i in range(loop_len * 3)
            )
            if repeated:
                return True
    return False


# ─── CoreML Runtime (reused from debug_lut6_comparison.py) ──────────

def _find(base, name):
    for ext in (".mlpackage", ".mlmodelc"):
        p = Path(base) / f"{name}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Missing: {name} in {base}")


class CoreMLRuntime:
    def __init__(self, model_dir, tokenizer, ctx, num_chunks):
        cu = ct.ComputeUnit.CPU_AND_NE
        self.ctx = ctx
        self.num_chunks = num_chunks
        self.tokenizer = tokenizer

        print(f"  Loading models from {model_dir}...", end="", flush=True)
        t0 = time.time()
        self.embed = ct.models.MLModel(_find(model_dir, "embeddings"), compute_units=cu)
        self.lmhead = ct.models.MLModel(_find(model_dir, "lm_head"), compute_units=cu)
        self.ffns = []
        for ci in range(num_chunks):
            self.ffns.append(ct.models.MLModel(
                _find(model_dir, f"ffn_LUT4_chunk{ci}"), compute_units=cu))
        print(f" {time.time()-t0:.0f}s")

        # Detect logits output format
        spec = self.lmhead.get_spec()
        out_names = [o.name for o in spec.description.output]
        split = sorted([n for n in out_names if n.startswith("logits") and n[6:].isdigit()])
        if split:
            self.logits_keys, self.logits_key = split, None
        elif "output_logits" in out_names:
            self.logits_keys, self.logits_key = None, "output_logits"
        elif "logits" in out_names:
            self.logits_keys, self.logits_key = None, "logits"
        else:
            self.logits_keys, self.logits_key = None, None

        # Detect state shapes
        spec0 = self.ffns[0].get_spec()
        imap = {}
        for inp in spec0.description.input:
            try:
                imap[inp.name] = tuple(int(x) for x in inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.conv_shape = imap.get("linear_conv_state", (8, 1024, 32))
        self.rec_shape = imap.get("linear_recurrent_state", (8, 32, 128, 128))

        # Stop tokens
        self.stop_ids = set()
        if tokenizer.eos_token_id is not None:
            self.stop_ids.add(int(tokenizer.eos_token_id))
        for s in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
            tid = tokenizer.convert_tokens_to_ids(s)
            if tid is not None and tid != tokenizer.unk_token_id:
                self.stop_ids.add(int(tid))
        self.reset()

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [np.zeros(self.conv_shape, dtype=np.float16) for _ in range(self.num_chunks)]
        self.lin_recs = [np.zeros(self.rec_shape, dtype=np.float16) for _ in range(self.num_chunks)]
        self._tok = np.zeros((1, 1), dtype=np.int32)
        self._mask = np.full((1, 1, 1, self.ctx), NEG_INF, dtype=np.float16)
        self._pos = np.zeros((1,), dtype=np.int32)

    def _extract_logits(self, lm_out):
        if self.logits_keys:
            return np.concatenate([lm_out[k].reshape(-1).astype(np.float32) for k in self.logits_keys])
        if self.logits_key:
            return lm_out[self.logits_key].reshape(-1).astype(np.float32)
        return None

    def _step(self, tid, pos):
        self._tok[0, 0] = np.int32(tid)
        hidden = list(self.embed.predict({"input_ids": self._tok}).values())[0]
        self._mask[:] = NEG_INF
        self._mask[:, :, :, :pos + 1] = 0
        self._pos[0] = np.int32(pos)
        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": self._pos, "causal_mask": self._mask,
                "current_pos": self._pos,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if "linear_conv_state_out" in out:
                self.lin_convs[ci] = out["linear_conv_state_out"]
                self.lin_recs[ci] = out["linear_recurrent_state_out"]
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        logits = self._extract_logits(lm_out)
        nxt = int(np.argmax(logits)) if logits is not None else int(lm_out["argmax_idx"].reshape(-1)[0])
        return nxt, logits

    @staticmethod
    def _sample(logits, generated_ids, temperature=0.7, top_p=0.9,
                repetition_penalty=1.1, frequency_penalty=0.2):
        """Apply penalties + temperature/top-p sampling."""
        if logits is None:
            return None  # argmax-only lm_head, no sampling possible

        # Repetition + frequency penalty
        token_counts = {}
        for tid in generated_ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1
        for tid, count in token_counts.items():
            if repetition_penalty != 1.0:
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty
            if frequency_penalty != 0.0:
                logits[tid] -= frequency_penalty * count

        # Temperature + top-p
        if temperature <= 0 or top_p <= 0:
            return int(np.argmax(logits))

        logits_f = logits.astype(np.float64)
        logits_f /= temperature
        logits_f -= np.max(logits_f)
        probs = np.exp(logits_f)
        probs /= probs.sum()

        if top_p < 1.0:
            sorted_idx = np.argsort(-probs)
            sorted_probs = probs[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumsum, top_p) + 1
            mask = np.zeros_like(probs, dtype=bool)
            mask[sorted_idx[:cutoff]] = True
            probs[~mask] = 0.0
            probs /= probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    def generate(self, prompt_ids, max_new_tokens,
                 temperature=0.7, top_p=0.9,
                 repetition_penalty=1.1, frequency_penalty=0.2):
        self.reset()
        t0 = time.time()
        for pos, tid in enumerate(prompt_ids):
            nxt, logits = self._step(tid, pos)
        t_prompt = time.time()
        # Apply sampling to first decode token
        if logits is not None:
            sampled = self._sample(logits, [], temperature, top_p,
                                   repetition_penalty, frequency_penalty)
            if sampled is not None:
                nxt = sampled
        gen = [nxt]
        p = len(prompt_ids)
        for _ in range(max_new_tokens - 1):
            if gen[-1] in self.stop_ids or p >= self.ctx - 1:
                break
            nxt, logits = self._step(gen[-1], p)
            p += 1
            if logits is not None:
                sampled = self._sample(logits, gen, temperature, top_p,
                                       repetition_penalty, frequency_penalty)
                if sampled is not None:
                    nxt = sampled
            gen.append(nxt)
        t_done = time.time()
        prompt_ms = (t_prompt - t0) * 1000
        decode_ms = (t_done - t_prompt) * 1000
        return [int(x) for x in gen], prompt_ms, decode_ms

    def close(self):
        del self.embed, self.lmhead, self.ffns, self.states
        gc.collect()


def make_prompt_ids(tokenizer, prompt):
    msgs = [{"role": "user", "content": prompt}]
    out = tokenizer.apply_chat_template(
        msgs, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False)
    ids = out.input_ids[0].tolist() if hasattr(out, "input_ids") else out[0].tolist()
    return [int(x) for x in ids]


# ─── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    ap.add_argument("--tokenizer", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Override per-prompt max_tokens")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--frequency-penalty", type=float, default=0.2)
    ap.add_argument("--out", default="tests/dev/repetition_validation_report.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    rt = CoreMLRuntime(args.model_dir, tokenizer, args.ctx, args.num_chunks)

    results = {}
    categories = {}

    # Counters
    total_prompts = 0
    total_degenerate = 0
    total_high_repeat = 0

    print(f"\n{'='*90}")
    print(f"  REPETITION VALIDATION — Milestone 2.1 (LUT6 gs=4)")
    print(f"  Sampling: temp={args.temperature}, top_p={args.top_p}, "
          f"rep_penalty={args.repetition_penalty}, freq_penalty={args.frequency_penalty}")
    print(f"{'='*90}")

    for p in PROMPTS:
        label = p["label"]
        category = p["category"]
        max_tok = args.max_new_tokens or p.get("max_tokens", 128)
        prompt_ids = make_prompt_ids(tokenizer, p["prompt"])

        gen, prompt_ms, decode_ms = rt.generate(
            prompt_ids, max_tok,
            temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            frequency_penalty=args.frequency_penalty)
        text = tokenizer.decode(gen, skip_special_tokens=True)

        # Metrics
        rep_4gram = compute_ngram_repeat_ratio(gen, n=4)
        rep_3gram = compute_ngram_repeat_ratio(gen, n=3)
        unique_ratio = compute_unique_token_ratio(gen)
        degenerate = detect_degeneration(gen)
        tok_per_s = len(gen) / (decode_ms / 1000) if decode_ms > 0 else 0

        # Flags
        is_high_repeat = rep_4gram > 0.3
        is_short = len(gen) < 10 and max_tok > 20
        is_empty = len(gen) == 0 or not text.strip()

        total_prompts += 1
        if degenerate:
            total_degenerate += 1
        if is_high_repeat:
            total_high_repeat += 1

        # Status symbol
        if degenerate:
            status = "🔴 DEGEN"
        elif is_high_repeat:
            status = "🟡 REPEAT"
        elif is_short or is_empty:
            status = "🟡 SHORT"
        else:
            status = "🟢 OK"

        result = {
            "text": text[:500],
            "gen_tokens": len(gen),
            "max_tokens": max_tok,
            "prompt_tokens": len(prompt_ids),
            "rep_4gram": round(rep_4gram, 4),
            "rep_3gram": round(rep_3gram, 4),
            "unique_ratio": round(unique_ratio, 4),
            "degenerate": degenerate,
            "tok_per_s": round(tok_per_s, 1),
            "decode_ms": round(decode_ms, 0),
            "status": status,
        }
        results[label] = result
        categories.setdefault(category, []).append(label)

        # Print
        print(f"\n  [{label}] {status}  ({len(gen)} tok, {tok_per_s:.1f} tok/s, "
              f"4gram_rep={rep_4gram:.2%}, unique={unique_ratio:.2%})")
        print(f"    Prompt: {p['prompt'][:70]}...")
        text_preview = text[:120].replace('\n', ' ')
        print(f"    Output: {text_preview}...")
        if degenerate:
            print(f"    ⚠️  DEGENERATION DETECTED — output enters repeating loop")
        if is_high_repeat:
            print(f"    ⚠️  HIGH REPETITION — 4-gram repeat ratio {rep_4gram:.2%}")

    # ── Summary ──
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")

    print(f"\n  Total prompts:     {total_prompts}")
    print(f"  Degenerate:        {total_degenerate} ({total_degenerate/total_prompts*100:.0f}%)")
    print(f"  High repetition:   {total_high_repeat} ({total_high_repeat/total_prompts*100:.0f}%)")
    print(f"  Clean:             {total_prompts - total_degenerate - total_high_repeat} "
          f"({(total_prompts - total_degenerate - total_high_repeat)/total_prompts*100:.0f}%)")

    # Per-category summary
    print(f"\n  {'Category':<15} {'Prompts':>8} {'Degen':>6} {'Hi-Rep':>7} {'Avg 4gram':>10} {'Avg Unique':>11}")
    print(f"  {'-'*60}")
    for cat, labels in sorted(categories.items()):
        n = len(labels)
        n_deg = sum(1 for l in labels if results[l]["degenerate"])
        n_rep = sum(1 for l in labels if results[l]["rep_4gram"] > 0.3)
        avg_rep = sum(results[l]["rep_4gram"] for l in labels) / n
        avg_unq = sum(results[l]["unique_ratio"] for l in labels) / n
        print(f"  {cat:<15} {n:>8} {n_deg:>6} {n_rep:>7} {avg_rep:>9.2%} {avg_unq:>10.2%}")

    # Detailed table
    print(f"\n  {'Label':<30} {'Status':<12} {'Tok':>5} {'4gram%':>8} {'Uniq%':>7} {'tok/s':>7}")
    print(f"  {'-'*72}")
    for p in PROMPTS:
        r = results[p["label"]]
        print(f"  {p['label']:<30} {r['status']:<12} {r['gen_tokens']:>5} "
              f"{r['rep_4gram']:>7.2%} {r['unique_ratio']:>6.2%} {r['tok_per_s']:>6.1f}")

    # Save report
    report = {
        "model_dir": args.model_dir,
        "milestone": "2.1",
        "quantization": "LUT6 gs=4",
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "frequency_penalty": args.frequency_penalty,
        },
        "total_prompts": total_prompts,
        "total_degenerate": total_degenerate,
        "total_high_repeat": total_high_repeat,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  Report saved: {out}")

    # Verdict
    if total_degenerate == 0 and total_high_repeat == 0:
        print(f"\n  ✅ PASS — No repetition or degeneration issues detected")
    elif total_degenerate > 0:
        print(f"\n  ❌ FAIL — {total_degenerate} prompt(s) show degeneration")
    else:
        print(f"\n  ⚠️  WARNING — {total_high_repeat} prompt(s) show high repetition")

    rt.close()


if __name__ == "__main__":
    main()
