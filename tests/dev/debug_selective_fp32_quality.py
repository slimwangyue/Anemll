#!/usr/bin/env python3
"""Quick quality test: Selective-FP32 (norm+recurrence only) vs FP16 vs Full-FP32 vs HF.

Uses cached HF results and runs selective-FP32 models on ANE across 8 diverse prompts.
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

import coremltools as ct
import numpy as np

sys.modules["profile"] = importlib.import_module("profile")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import AutoTokenizer

NEG_INF = np.float16(-65504.0)

PROMPTS = [
    {"prompt": "教我做红烧肉", "label": "zh_recipe"},
    {"prompt": "解释量子纠缠", "label": "zh_physics"},
    {"prompt": "写一首关于春天的诗", "label": "zh_poem"},
    {"prompt": "Explain how a neural network learns", "label": "en_nn"},
    {"prompt": "Write a Python function to sort a list", "label": "en_code"},
    {"prompt": "What is the capital of France?", "label": "en_factoid"},
    {"prompt": "If a train travels at 60 km/h for 2.5 hours, how far does it go?", "label": "en_math"},
    {"prompt": "Tell me a short story about a robot who learns to cook", "label": "en_story"},
]


def make_prompt_ids(tokenizer, prompt, enable_thinking=False):
    msgs = [{"role": "user", "content": prompt}]
    out = tokenizer.apply_chat_template(
        msgs, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=enable_thinking)
    ids = out.input_ids[0].tolist() if hasattr(out, "input_ids") else out[0].tolist()
    return [int(x) for x in ids]


def _find(base, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = Path(base) / f"{name}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Missing: {name} in {base}")


class CoreMLRuntime:
    def __init__(self, model_dir, tokenizer, ctx, num_chunks, label="CoreML"):
        cu = ct.ComputeUnit.CPU_AND_NE
        self.ctx, self.num_chunks, self.label = ctx, num_chunks, label
        self.tokenizer = tokenizer

        print(f"    [{label}] Loading models...")
        self.embed = ct.models.MLModel(_find(model_dir, "embeddings"), compute_units=cu)
        self.lmhead = ct.models.MLModel(_find(model_dir, "lm_head"), compute_units=cu)
        self.ffns = []
        for ci in range(num_chunks):
            self.ffns.append(ct.models.MLModel(
                _find(model_dir, f"ffn_LUT4_chunk{ci}"), compute_units=cu))

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

        spec0 = self.ffns[0].get_spec()
        imap = {}
        for inp in spec0.description.input:
            try:
                imap[inp.name] = tuple(int(x) for x in inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.conv_shape = imap.get("linear_conv_state", (8, 1024, 32))
        self.rec_shape = imap.get("linear_recurrent_state", (8, 32, 128, 128))

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
        return nxt

    def generate(self, prompt_ids, max_new_tokens):
        self.reset()
        for pos, tid in enumerate(prompt_ids):
            nxt = self._step(tid, pos)
        gen = [nxt]
        p = len(prompt_ids)
        t0 = time.time()
        for _ in range(max_new_tokens - 1):
            if gen[-1] in self.stop_ids or p >= self.ctx - 1:
                break
            nxt = self._step(gen[-1], p)
            p += 1
            gen.append(nxt)
        dec_ms = (time.time() - t0) * 1000
        return [int(x) for x in gen], dec_ms

    def close(self):
        del self.embed, self.lmhead, self.ffns, self.states
        gc.collect()


def judge_quality(hf_text, test_text, prompt):
    if not test_text.strip():
        return 0
    refusal = any(p in test_text.lower() for p in ["i can't", "i cannot", "无法", "不能"])
    if refusal and "不能" not in hf_text.lower() and "can't" not in hf_text.lower():
        return 0
    hf_words = set(hf_text.lower().split())
    test_words = set(test_text.lower().split())
    overlap = len(hf_words & test_words) / max(len(hf_words), 1)
    if overlap > 0.3:
        return 3
    if overlap > 0.1:
        return 2
    return 1


def run_all_prompts(rt, tokenizer, prompts, hf_results, max_new_tokens, label):
    results = {}
    for pi, p in enumerate(prompts):
        hf_r = hf_results[p["label"]]
        prompt_ids = make_prompt_ids(tokenizer, p["prompt"], enable_thinking=False)
        gen, dec_ms = rt.generate(prompt_ids, max_new_tokens)
        text = tokenizer.decode(gen, skip_special_tokens=False)
        first_match = gen[0] == hf_r["gen"][0] if gen and hf_r["gen"] else False
        quality = judge_quality(hf_r["text"], text, p["prompt"])
        results[p["label"]] = {
            "gen": gen, "text": text, "decode_ms": dec_ms,
            "first_match": first_match, "quality": quality,
        }
        sym = "✓" if first_match else "✗"
        print(f"    {p['label']:<15} first={gen[0]:>7} {sym}  q={quality}  dec={dec_ms:.0f}ms  {text[:60]}...")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16-dir", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1_3")
    ap.add_argument("--fp32-dir", default="/tmp/qwen35_fp32_chunks")
    ap.add_argument("--sel-dir", default="/tmp/qwen35_selective_chunks")
    ap.add_argument("--tokenizer", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    ap.add_argument("--hf-cache", default="tests/dev/fp32_multiprompt_hf_cache.json")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="tests/dev/selective_fp32_quality_report.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    # Load cached HF results
    print("Loading cached HF results...")
    with open(args.hf_cache) as f:
        hf_results = json.load(f)

    configs = [
        ("FP16", args.fp16_dir),
        ("Selective-FP32", args.sel_dir),
        ("Full-FP32", args.fp32_dir),
    ]

    all_results = {}
    for label, model_dir in configs:
        print(f"\n{'='*70}")
        print(f"  {label} — {model_dir}")
        print(f"{'='*70}")
        rt = CoreMLRuntime(model_dir, tokenizer, args.ctx, args.num_chunks, label=label)
        results = run_all_prompts(rt, tokenizer, PROMPTS, hf_results, args.max_new_tokens, label)
        all_results[label] = results
        rt.close()

    # ── Summary table ──
    print(f"\n{'='*100}")
    print(f"  SUMMARY: FP16 vs Selective-FP32 vs Full-FP32")
    print(f"{'='*100}")

    print(f"\n  {'Prompt':<15}", end="")
    for label, _ in configs:
        print(f" {label+' 1st':>14} {'Q':>3} {'dec ms':>7}", end="")
    print()
    print(f"  {'-'*90}")

    totals = {label: {"q": 0, "fm": 0, "dec": 0} for label, _ in configs}

    for p in PROMPTS:
        lab = p["label"]
        hf_first = hf_results[lab]["gen"][0]
        print(f"  {lab:<15}", end="")
        for label, _ in configs:
            r = all_results[label][lab]
            first = r["gen"][0] if r["gen"] else -1
            sym = "✓" if r["first_match"] else "✗"
            print(f" {first:>12}{sym:>2} {r['quality']:>3} {r['decode_ms']:>7.0f}", end="")
            totals[label]["q"] += r["quality"]
            totals[label]["fm"] += int(r["first_match"])
            totals[label]["dec"] += r["decode_ms"]
        print(f"  (HF={hf_first})")

    n = len(PROMPTS)
    print(f"  {'-'*90}")
    print(f"  {'AVERAGE':<15}", end="")
    for label, _ in configs:
        t = totals[label]
        print(f" {'':>14} {t['q']/n:>3.1f} {t['dec']/n:>7.0f}", end="")
    print()
    print(f"  {'1st match':<15}", end="")
    for label, _ in configs:
        t = totals[label]
        print(f" {t['fm']}/{n} ({t['fm']/n*100:.0f}%){'':>10}", end="")
    print()

    # Latency comparison  
    fp16_dec = totals["FP16"]["dec"] / n
    sel_dec = totals["Selective-FP32"]["dec"] / n
    fp32_dec = totals["Full-FP32"]["dec"] / n
    print(f"\n  Latency overhead vs FP16:")
    print(f"    Full-FP32:      {fp32_dec:.0f}ms ({fp32_dec/fp16_dec:.2f}x)")
    print(f"    Selective-FP32: {sel_dec:.0f}ms ({sel_dec/fp16_dec:.2f}x) ← {(1-sel_dec/fp32_dec)*100:.0f}% faster than full-FP32")

    # Full text comparison
    print(f"\n  {'─'*90}")
    print(f"  Selected output comparison:")
    for p in PROMPTS[:4]:
        lab = p["label"]
        print(f"\n  [{lab}] {p['prompt']}")
        print(f"    HF:        {hf_results[lab]['text'][:100]}")
        for label, _ in configs:
            r = all_results[label][lab]
            print(f"    {label:<15} {r['text'][:100]}")

    # Save report
    report = {
        "prompts": [p["prompt"] for p in PROMPTS],
        "configs": {label: {
            "per_prompt": {lab: {"gen": r["gen"][:16], "text": r["text"][:300],
                                  "quality": r["quality"], "first_match": r["first_match"],
                                  "decode_ms": r["decode_ms"]}
                          for lab, r in results.items()},
            "avg_quality": totals[label]["q"] / n,
            "first_match_rate": totals[label]["fm"] / n,
            "avg_decode_ms": totals[label]["dec"] / n,
        } for label, results in [(l, all_results[l]) for l, _ in configs]},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  Report saved: {out}")


if __name__ == "__main__":
    main()
