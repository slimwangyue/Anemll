#!/usr/bin/env python3
"""Multi-prompt FP32/FP16/Hybrid ANE parity test.

Phase 1: Run diverse prompts through HF → FP16(ANE) → FP32(ANE) and compare outputs.
Phase 2: Test hybrid precision combos (mix FP16/FP32 per chunk) to find sweet spot.

Memory strategy: load one engine at a time, free before loading next.
  HF model (~8GB) → free → FP16 CoreML (~2GB) → free → FP32 CoreML → free → hybrids

Usage:
    python tests/dev/debug_fp32_multiprompt_hybrid.py \
        --hf-model /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
        --fp16-dir /Users/yw68/Anemll_remote_run/qwen35_milestone1_3 \
        --fp32-dir /tmp/qwen35_fp32_chunks \
        --tokenizer /Users/yw68/Anemll/qwen3_5_stable_models
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.modules["profile"] = importlib.import_module("profile")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ───────────────────────────── constants ─────────────────────────────
NEG_INF = np.float16(-65504.0)

PROMPTS = [
    # Chinese
    {"prompt": "教我做红烧肉", "label": "zh_recipe"},
    {"prompt": "解释量子纠缠", "label": "zh_physics"},
    {"prompt": "写一首关于春天的诗", "label": "zh_poem"},
    # English
    {"prompt": "Explain how a neural network learns", "label": "en_nn"},
    {"prompt": "Write a Python function to sort a list", "label": "en_code"},
    {"prompt": "What is the capital of France?", "label": "en_factoid"},
    # Reasoning
    {"prompt": "If a train travels at 60 km/h for 2.5 hours, how far does it go?", "label": "en_math"},
    # Longer / creative
    {"prompt": "Tell me a short story about a robot who learns to cook", "label": "en_story"},
]

# ───────────────────────────── helpers ───────────────────────────────

def cosine(a, b):
    x = a.astype(np.float64).reshape(-1)
    y = b.astype(np.float64).reshape(-1)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def make_prompt_ids(tokenizer, prompt, enable_thinking=False):
    msgs = [{"role": "user", "content": prompt}]
    out = tokenizer.apply_chat_template(
        msgs, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=enable_thinking)
    ids = out.input_ids[0].tolist() if hasattr(out, "input_ids") else out[0].tolist()
    return [int(x) for x in ids]


# ───────────────────────────── HF runtime ────────────────────────────

class HFRuntime:
    """Runs HuggingFace model on CPU for reference outputs."""

    def __init__(self, model_path, tokenizer):
        import torch
        from transformers import AutoModelForCausalLM
        self.tokenizer = tokenizer
        print("  [HF] Loading model (CPU, float16)...")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16)
        self.model.eval()
        print(f"  [HF] Loaded in {time.time()-t0:.1f}s")

    def generate(self, prompt_ids, max_new_tokens):
        import torch
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0)
        gen_ids = out[0, len(prompt_ids):].tolist()
        return [int(x) for x in gen_ids]

    def close(self):
        del self.model
        gc.collect()
        try:
            import torch; torch.mps.empty_cache()
        except Exception:
            pass


# ───────────────────────────── CoreML runtime ────────────────────────

class CoreMLRuntime:
    """Runs CoreML FFN chunks on ANE. Supports hybrid (mixed FP16/FP32 per chunk)."""

    def __init__(self, chunk_dirs, tokenizer, ctx, num_chunks, label="CoreML"):
        """
        chunk_dirs: list of directories, one per chunk.
                    Each dir should contain ffn_LUT4_chunk{i}.mlpackage, plus
                    embeddings.mlpackage and lm_head.mlpackage (or symlinks).
        """
        import coremltools as ct
        self.ct = ct
        cu = ct.ComputeUnit.CPU_AND_NE
        self.ctx = ctx
        self.num_chunks = num_chunks
        self.label = label
        self.tokenizer = tokenizer

        # Load embed + lm_head from first dir
        base = chunk_dirs[0]
        print(f"    [{label}] Loading embeddings...")
        self.embed = self._load(_find(base, "embeddings"), cu)
        print(f"    [{label}] Loading lm_head...")
        self.lmhead = self._load(_find(base, "lm_head"), cu)

        # Load FFN chunks from specified directories
        self.ffns = []
        for ci in range(num_chunks):
            d = chunk_dirs[ci] if ci < len(chunk_dirs) else chunk_dirs[-1]
            print(f"    [{label}] Loading ffn chunk {ci} from {Path(d).name}...")
            self.ffns.append(self._load(_find(d, f"ffn_LUT4_chunk{ci}"), cu))

        # Detect lm_head output format
        spec = self.lmhead.get_spec()
        out_names = [o.name for o in spec.description.output]
        split = sorted([n for n in out_names if n.startswith("logits") and n[6:].isdigit()])
        if split:
            self.logits_keys = split
            self.logits_key = None
            self.lm_mode = "logits"
        elif "output_logits" in out_names or "logits" in out_names:
            self.logits_keys = None
            self.logits_key = "output_logits" if "output_logits" in out_names else "logits"
            self.lm_mode = "logits"
        else:
            self.logits_keys = None
            self.logits_key = None
            self.lm_mode = "argmax"

        # Detect state shapes from chunk0
        spec0 = self.ffns[0].get_spec()
        imap = {}
        for inp in spec0.description.input:
            try:
                imap[inp.name] = tuple(int(x) for x in inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.conv_shape = imap.get("linear_conv_state", (8, 1024, 32))
        self.rec_shape = imap.get("linear_recurrent_state", (8, 32, 128, 128))

        # Stop token ids
        self.stop_ids = set()
        if tokenizer.eos_token_id is not None:
            self.stop_ids.add(int(tokenizer.eos_token_id))
        for s in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
            tid = tokenizer.convert_tokens_to_ids(s)
            if tid is not None and tid != tokenizer.unk_token_id:
                self.stop_ids.add(int(tid))

        self.reset()

    @staticmethod
    def _load(path, cu):
        import coremltools as ct
        if path.endswith(".mlmodelc"):
            return ct.models.CompiledMLModel(path, cu)
        return ct.models.MLModel(path, compute_units=cu)

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [np.zeros(self.conv_shape, dtype=np.float16) for _ in range(self.num_chunks)]
        self.lin_recs = [np.zeros(self.rec_shape, dtype=np.float16) for _ in range(self.num_chunks)]
        self._tok = np.zeros((1, 1), dtype=np.int32)
        self._mask = np.full((1, 1, 1, self.ctx), NEG_INF, dtype=np.float16)
        self._pos = np.zeros((1,), dtype=np.int32)

    def _extract_logits(self, lm_out):
        if self.lm_mode != "logits":
            return None
        if self.logits_keys:
            return np.concatenate([lm_out[k].reshape(-1).astype(np.float32) for k in self.logits_keys])
        return lm_out[self.logits_key].reshape(-1).astype(np.float32)

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

    def generate(self, prompt_ids, max_new_tokens):
        self.reset()
        # Process prompt
        for pos, tid in enumerate(prompt_ids):
            nxt, logits = self._step(tid, pos)
        # Decode
        gen = [nxt]
        pos = len(prompt_ids)
        for _ in range(max_new_tokens - 1):
            if gen[-1] in self.stop_ids or pos >= self.ctx - 1:
                break
            nxt, _ = self._step(gen[-1], pos)
            pos += 1
            gen.append(nxt)
        return [int(x) for x in gen]

    def timed_generate(self, prompt_ids, max_new_tokens):
        """Generate with timing. Returns (tokens, prompt_ms, decode_ms)."""
        self.reset()
        t0 = time.time()
        for pos, tid in enumerate(prompt_ids):
            nxt, _ = self._step(tid, pos)
        t_prompt = time.time()
        gen = [nxt]
        p = len(prompt_ids)
        for _ in range(max_new_tokens - 1):
            if gen[-1] in self.stop_ids or p >= self.ctx - 1:
                break
            nxt, _ = self._step(gen[-1], p)
            p += 1
            gen.append(nxt)
        t_decode = time.time()
        prompt_ms = (t_prompt - t0) * 1000
        decode_ms = (t_decode - t_prompt) * 1000
        return [int(x) for x in gen], prompt_ms, decode_ms

    def close(self):
        del self.embed, self.lmhead, self.ffns, self.states
        del self.lin_convs, self.lin_recs
        gc.collect()


def _find(base_dir, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = str(Path(base_dir) / f"{name}{ext}")
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"Missing: {name} in {base_dir}")


def token_parity(a, b):
    n = min(len(a), len(b))
    m = sum(1 for i in range(n) if a[i] == b[i])
    fd = next((i for i in range(n) if a[i] != b[i]), None)
    return {"len": n, "match": m, "ratio": m / n if n else 0, "first_div": fd}


def judge_quality(hf_text, test_text, prompt):
    """Heuristic quality score 0-3:
       0 = garbage/refusal, 1 = off-topic, 2 = reasonable, 3 = matches HF.
    """
    if not test_text.strip():
        return 0
    # Check for refusal patterns
    refusal = any(p in test_text.lower() for p in
                  ["i can't", "i cannot", "i'm unable", "不能", "无法"])
    if refusal and "不能" not in hf_text.lower() and "can't" not in hf_text.lower():
        return 0
    # Check semantic overlap (shared substrings)
    hf_words = set(hf_text.lower().split())
    test_words = set(test_text.lower().split())
    overlap = len(hf_words & test_words) / max(len(hf_words), 1)
    if overlap > 0.3:
        return 3
    if overlap > 0.1:
        return 2
    # Check if answer is at least topical
    return 1


# ─────────────────────── PHASE 1: Multi-prompt ───────────────────────

def phase1_hf(args, tokenizer, prompts):
    """Generate HF reference outputs for all prompts. Returns dict keyed by label."""
    results = {}
    hf = HFRuntime(args.hf_model, tokenizer)
    for pi, p in enumerate(prompts):
        prompt_ids = make_prompt_ids(tokenizer, p["prompt"], enable_thinking=False)
        print(f"  [{pi+1}/{len(prompts)}] HF generating: {p['label']} ({len(prompt_ids)} toks)...")
        t0 = time.time()
        gen = hf.generate(prompt_ids, args.max_new_tokens)
        t1 = time.time()
        text = tokenizer.decode(gen, skip_special_tokens=False)
        results[p["label"]] = {
            "prompt": p["prompt"], "prompt_ids": prompt_ids, "gen": gen,
            "text": text, "time_s": t1 - t0,
        }
        print(f"    → {len(gen)} tokens, {t1-t0:.1f}s: {text[:80]}...")
    hf.close()
    return results


def phase1_coreml(args, tokenizer, prompts, hf_results, model_dir, label, chunk_dirs=None):
    """Generate CoreML outputs. chunk_dirs: list per chunk (for hybrid)."""
    if chunk_dirs is None:
        chunk_dirs = [model_dir] * args.num_chunks
    rt = CoreMLRuntime(chunk_dirs, tokenizer, args.ctx, args.num_chunks, label=label)
    results = {}
    for pi, p in enumerate(prompts):
        hf_r = hf_results[p["label"]]
        prompt_ids = hf_r["prompt_ids"]
        print(f"  [{pi+1}/{len(prompts)}] {label} generating: {p['label']}...")
        gen, prompt_ms, decode_ms = rt.timed_generate(prompt_ids, args.max_new_tokens)
        text = tokenizer.decode(gen, skip_special_tokens=False)
        parity = token_parity(hf_r["gen"], gen)
        hf_text = hf_r["text"]
        quality = judge_quality(hf_text, text, p["prompt"])
        first_match = (gen[0] == hf_r["gen"][0]) if gen and hf_r["gen"] else False
        results[p["label"]] = {
            "gen": gen, "text": text,
            "prompt_ms": prompt_ms, "decode_ms": decode_ms,
            "parity": parity, "quality": quality, "first_match": first_match,
        }
        sym = "✓" if first_match else "✗"
        print(f"    → first={gen[0]} {sym}  quality={quality}  "
              f"parity={parity['match']}/{parity['len']}  "
              f"decode={decode_ms:.0f}ms: {text[:80]}...")
    rt.close()
    return results


# ─────────────────── PHASE 2: Hybrid precision ───────────────────────

# Hybrid configs: which chunks get FP32. True=FP32, False=FP16
HYBRID_CONFIGS = [
    {"label": "all_fp16",   "mask": [False, False, False, False]},
    {"label": "all_fp32",   "mask": [True,  True,  True,  True]},
    {"label": "c3_fp32",    "mask": [False, False, False, True]},   # only last chunk
    {"label": "c23_fp32",   "mask": [False, False, True,  True]},   # last two
    {"label": "c123_fp32",  "mask": [False, True,  True,  True]},   # last three
    {"label": "c0_fp32",    "mask": [True,  False, False, False]},   # only first chunk
    {"label": "c03_fp32",   "mask": [True,  False, False, True]},    # first + last
]


def phase2_hybrid(args, tokenizer, prompts, hf_results):
    """Test hybrid precision combos: mix FP16/FP32 chunks."""
    results = {}

    for cfg in HYBRID_CONFIGS:
        # Build chunk_dirs list
        chunk_dirs = []
        for ci, use_fp32 in enumerate(cfg["mask"]):
            chunk_dirs.append(args.fp32_dir if use_fp32 else args.fp16_dir)

        label = cfg["label"]
        n32 = sum(cfg["mask"])
        print(f"\n  === Hybrid: {label} ({n32}/4 FP32 chunks) ===")

        rt = CoreMLRuntime(chunk_dirs, tokenizer, args.ctx, args.num_chunks, label=label)
        cfg_results = {}

        total_prompt_ms = 0
        total_decode_ms = 0
        total_quality = 0
        total_first_match = 0

        for pi, p in enumerate(prompts):
            hf_r = hf_results[p["label"]]
            prompt_ids = hf_r["prompt_ids"]
            gen, prompt_ms, decode_ms = rt.timed_generate(prompt_ids, args.max_new_tokens)
            text = tokenizer.decode(gen, skip_special_tokens=False)
            parity = token_parity(hf_r["gen"], gen)
            quality = judge_quality(hf_r["text"], text, p["prompt"])
            first_match = (gen[0] == hf_r["gen"][0]) if gen and hf_r["gen"] else False

            cfg_results[p["label"]] = {
                "gen": gen, "text": text,
                "prompt_ms": prompt_ms, "decode_ms": decode_ms,
                "parity": parity, "quality": quality, "first_match": first_match,
            }

            total_prompt_ms += prompt_ms
            total_decode_ms += decode_ms
            total_quality += quality
            total_first_match += int(first_match)

            sym = "✓" if first_match else "✗"
            print(f"    {p['label']}: first={gen[0]} {sym}  q={quality}  "
                  f"match={parity['match']}/{parity['len']}  "
                  f"dec={decode_ms:.0f}ms")

        rt.close()

        n = len(prompts)
        avg_quality = total_quality / n
        avg_decode_ms = total_decode_ms / n
        first_match_rate = total_first_match / n

        results[label] = {
            "mask": cfg["mask"], "num_fp32": n32,
            "per_prompt": cfg_results,
            "avg_quality": avg_quality,
            "avg_decode_ms": avg_decode_ms,
            "first_match_rate": first_match_rate,
            "total_prompt_ms": total_prompt_ms,
            "total_decode_ms": total_decode_ms,
        }

        print(f"  ── {label}: avg_quality={avg_quality:.2f}/3  "
              f"first_match={first_match_rate*100:.0f}%  "
              f"avg_decode={avg_decode_ms:.0f}ms")

    return results


def print_summary(hf_results, fp16_results, fp32_results, hybrid_results, tokenizer):
    """Print a comprehensive comparison table."""
    prompts_labels = list(hf_results.keys())

    print("\n" + "=" * 100)
    print("  PHASE 1: Multi-Prompt FP16 vs FP32 vs HF")
    print("=" * 100)

    print(f"\n  {'Prompt':<15} {'HF 1st':>8} {'FP16 1st':>9} {'FP32 1st':>9} "
          f"{'FP16 Q':>7} {'FP32 Q':>7} {'FP16 dec':>9} {'FP32 dec':>9}")
    print(f"  {'-'*88}")

    fp16_q_sum, fp32_q_sum = 0, 0
    fp16_fm_sum, fp32_fm_sum = 0, 0
    fp16_dec_sum, fp32_dec_sum = 0, 0

    for lab in prompts_labels:
        hf_r = hf_results[lab]
        fp16_r = fp16_results[lab]
        fp32_r = fp32_results[lab]
        hf1 = hf_r["gen"][0] if hf_r["gen"] else -1
        fp16_1 = fp16_r["gen"][0] if fp16_r["gen"] else -1
        fp32_1 = fp32_r["gen"][0] if fp32_r["gen"] else -1

        fp16_sym = "✓" if fp16_r["first_match"] else "✗"
        fp32_sym = "✓" if fp32_r["first_match"] else "✗"

        print(f"  {lab:<15} {hf1:>8} {fp16_1:>7}{fp16_sym:>2} {fp32_1:>7}{fp32_sym:>2} "
              f"{fp16_r['quality']:>5}/3  {fp32_r['quality']:>5}/3  "
              f"{fp16_r['decode_ms']:>7.0f}ms {fp32_r['decode_ms']:>7.0f}ms")

        fp16_q_sum += fp16_r["quality"]
        fp32_q_sum += fp32_r["quality"]
        fp16_fm_sum += int(fp16_r["first_match"])
        fp32_fm_sum += int(fp32_r["first_match"])
        fp16_dec_sum += fp16_r["decode_ms"]
        fp32_dec_sum += fp32_r["decode_ms"]

    n = len(prompts_labels)
    print(f"  {'-'*88}")
    print(f"  {'AVERAGE':<15} {'':>8} {'':>9} {'':>9} "
          f"{fp16_q_sum/n:>5.1f}/3  {fp32_q_sum/n:>5.1f}/3  "
          f"{fp16_dec_sum/n:>7.0f}ms {fp32_dec_sum/n:>7.0f}ms")
    print(f"  First token match:  FP16={fp16_fm_sum}/{n} ({fp16_fm_sum/n*100:.0f}%)  "
          f"FP32={fp32_fm_sum}/{n} ({fp32_fm_sum/n*100:.0f}%)")

    # Full text comparison
    print(f"\n  {'─'*90}")
    print(f"  Full output comparison:")
    print(f"  {'─'*90}")
    for lab in prompts_labels:
        hf_t = hf_results[lab]["text"][:120]
        fp16_t = fp16_results[lab]["text"][:120]
        fp32_t = fp32_results[lab]["text"][:120]
        print(f"\n  [{lab}] prompt: {hf_results[lab]['prompt']}")
        print(f"    HF:   {hf_t}")
        print(f"    FP16: {fp16_t}")
        print(f"    FP32: {fp32_t}")

    if not hybrid_results:
        return

    print("\n" + "=" * 100)
    print("  PHASE 2: Hybrid Precision Sweet-Spot Analysis")
    print("=" * 100)

    print(f"\n  {'Config':<14} {'FP32 chunks':>12} {'Avg Q':>7} {'1st match':>10} "
          f"{'Avg dec ms':>11} {'Speedup':>8}")
    print(f"  {'-'*70}")

    all_fp32_dec = hybrid_results.get("all_fp32", {}).get("avg_decode_ms", 1)
    all_fp16_dec = hybrid_results.get("all_fp16", {}).get("avg_decode_ms", 1)

    for cfg in HYBRID_CONFIGS:
        lab = cfg["label"]
        if lab not in hybrid_results:
            continue
        hr = hybrid_results[lab]
        n32 = hr["num_fp32"]
        speedup_vs_full32 = all_fp32_dec / hr["avg_decode_ms"] if hr["avg_decode_ms"] > 0 else 0
        print(f"  {lab:<14} {n32:>4}/4       {hr['avg_quality']:>5.2f}/3 "
              f"{hr['first_match_rate']*100:>8.0f}%  "
              f"{hr['avg_decode_ms']:>9.0f}ms {speedup_vs_full32:>7.2f}x")

    # Best sweet spot
    candidates = []
    for cfg in HYBRID_CONFIGS:
        lab = cfg["label"]
        if lab in hybrid_results:
            hr = hybrid_results[lab]
            candidates.append((lab, hr["avg_quality"], hr["first_match_rate"],
                               hr["avg_decode_ms"], hr["num_fp32"]))

    # Sort by quality desc, then decode_ms asc
    candidates.sort(key=lambda x: (-x[1], -x[2], x[3]))
    best = candidates[0]
    print(f"\n  SWEET SPOT: {best[0]} — quality={best[1]:.2f}/3, "
          f"first_match={best[2]*100:.0f}%, decode={best[3]:.0f}ms, "
          f"FP32 chunks={best[4]}/4")

    # Find best tradeoff (highest quality per ms)
    for c in candidates:
        if c[1] >= 2.0 and c[3] < all_fp32_dec * 0.9:
            print(f"  BEST TRADEOFF: {c[0]} — quality={c[1]:.2f}/3, "
                  f"first_match={c[2]*100:.0f}%, "
                  f"decode={c[3]:.0f}ms ({c[3]/all_fp16_dec:.2f}x fp16 latency)")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B")
    ap.add_argument("--fp16-dir", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1_3")
    ap.add_argument("--fp32-dir", default="/tmp/qwen35_fp32_chunks")
    ap.add_argument("--tokenizer", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--skip-hf", action="store_true", help="Skip HF and load cached results")
    ap.add_argument("--skip-phase2", action="store_true", help="Skip hybrid phase")
    ap.add_argument("--out", default="tests/dev/fp32_multiprompt_hybrid_report.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    prompts = PROMPTS
    cache_path = Path("tests/dev/fp32_multiprompt_hf_cache.json")

    # ── Phase 1a: HF reference ──
    if args.skip_hf and cache_path.exists():
        print("Loading cached HF results...")
        with open(cache_path) as f:
            hf_results = json.load(f)
    else:
        print("\n" + "=" * 70)
        print("  PHASE 1a: HF Reference Generation")
        print("=" * 70)
        hf_results = phase1_hf(args, tokenizer, prompts)
        # Cache HF results for reuse
        cache_path.write_text(json.dumps(hf_results, ensure_ascii=False, indent=2))
        print(f"  Cached HF results → {cache_path}")

    # ── Phase 1b: FP16 ANE ──
    print("\n" + "=" * 70)
    print("  PHASE 1b: FP16 (ANE) Generation")
    print("=" * 70)
    fp16_results = phase1_coreml(args, tokenizer, prompts, hf_results, args.fp16_dir, "FP16")

    # ── Phase 1c: FP32 ANE ──
    print("\n" + "=" * 70)
    print("  PHASE 1c: FP32 (ANE) Generation")
    print("=" * 70)
    fp32_results = phase1_coreml(args, tokenizer, prompts, hf_results, args.fp32_dir, "FP32")

    # ── Phase 2: Hybrid ──
    hybrid_results = {}
    if not args.skip_phase2:
        print("\n" + "=" * 70)
        print("  PHASE 2: Hybrid Precision Experiments")
        print("=" * 70)
        hybrid_results = phase2_hybrid(args, tokenizer, prompts, hf_results)

    # ── Summary ──
    print_summary(hf_results, fp16_results, fp32_results, hybrid_results, tokenizer)

    # ── Save report ──
    report = {
        "prompts": [p["prompt"] for p in prompts],
        "ctx": args.ctx, "num_chunks": args.num_chunks,
        "max_new_tokens": args.max_new_tokens,
        "hf": {k: {"prompt": v["prompt"], "gen": v["gen"], "text": v["text"]}
               for k, v in hf_results.items()},
        "fp16": {k: {"gen": v["gen"], "text": v["text"], "quality": v["quality"],
                      "first_match": v["first_match"], "decode_ms": v["decode_ms"],
                      "parity": v["parity"]}
                 for k, v in fp16_results.items()},
        "fp32": {k: {"gen": v["gen"], "text": v["text"], "quality": v["quality"],
                      "first_match": v["first_match"], "decode_ms": v["decode_ms"],
                      "parity": v["parity"]}
                 for k, v in fp32_results.items()},
        "hybrid": {},
    }
    for hlab, hdata in hybrid_results.items():
        report["hybrid"][hlab] = {
            "mask": hdata["mask"], "num_fp32": hdata["num_fp32"],
            "avg_quality": hdata["avg_quality"],
            "first_match_rate": hdata["first_match_rate"],
            "avg_decode_ms": hdata["avg_decode_ms"],
            "per_prompt": {
                k: {"gen": v["gen"][:16], "text": v["text"][:200],
                    "quality": v["quality"], "first_match": v["first_match"],
                    "decode_ms": v["decode_ms"]}
                for k, v in hdata["per_prompt"].items()
            },
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  Full report saved: {out}")


if __name__ == "__main__":
    main()
