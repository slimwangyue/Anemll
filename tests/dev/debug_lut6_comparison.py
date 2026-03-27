#!/usr/bin/env python3
"""LUT6 vs LUT4 comprehensive comparison.

Export FFN chunks with LUT6 at various per_channel group sizes and compare
quality + latency + size against LUT4-FP16, LUT4-FP32 baselines.

Configs to test:
  - LUT4 gs=8  FP16 (existing baseline)
  - LUT4 gs=8  FP32 (existing from /tmp/qwen35_fp32_chunks)
  - LUT6 gs=8  FP16 (higher-fidelity weights, same FP16 compute)
  - LUT6 gs=4  FP16 (finer-grained LUT6)
  - LUT6 gs=16 FP16 (coarser-grained LUT6)
  - LUT6 gs=1  FP16 (per-tensor palettization)

Memory strategy: export one config at a time (load HF → export → free).

Usage:
    python tests/dev/debug_lut6_comparison.py --phase export  # export all
    python tests/dev/debug_lut6_comparison.py --phase test    # test all
    python tests/dev/debug_lut6_comparison.py                 # both
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
import torch

sys.modules["profile"] = importlib.import_module("profile")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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

# ─── Configs ─────────────────────────────────────────────────────────
CONFIGS = [
    {
        "label": "LUT4_gs8_fp16",
        "dir": "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3",
        "export": False,  # already exists
    },
    {
        "label": "LUT6_gs16_fp16",
        "dir": "/tmp/qwen35_lut6_gs16",
        "lut_bits": 6, "per_channel": 16,
        "export": True,
    },
    {
        "label": "LUT6_gs8_fp16",
        "dir": "/tmp/qwen35_lut6_gs8",
        "lut_bits": 6, "per_channel": 8,
        "export": True,
    },
    {
        "label": "LUT6_gs4_fp16",
        "dir": "/tmp/qwen35_lut6_gs4",
        "lut_bits": 6, "per_channel": 4,
        "export": True,
    },
    {
        "label": "LUT6_gs1_fp16",
        "dir": "/tmp/qwen35_lut6_gs1",
        "lut_bits": 6, "per_channel": 1,
        "export": True,
    },
]


# ─── Export ──────────────────────────────────────────────────────────

def export_all_chunks(cfg, hf_model_path, ctx, num_chunks):
    """Export all 4 FFN chunks for a given config."""
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config

    out_dir = cfg["dir"]
    os.makedirs(out_dir, exist_ok=True)
    lut_bits = cfg["lut_bits"]
    per_channel = cfg["per_channel"]

    print(f"\n  Loading HF model...")
    t0 = time.time()
    config = Qwen35Config.from_json(os.path.join(hf_model_path, "config.json"))
    config.context_length = ctx
    config.state_length = max(config.state_length, ctx)
    model = Qwen35ForCausalLM(config)
    ok = model.load_pretrained_weights(hf_model_path)
    if not ok:
        raise RuntimeError(f"Failed to load weights from {hf_model_path}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")

    converter = Qwen35Converter(
        model=model,
        context_length=ctx,
        num_chunks=num_chunks,
        lut_bits=lut_bits,
        per_channel=per_channel,
    )

    for ci in range(num_chunks):
        out_path = os.path.join(out_dir, f"ffn_LUT4_chunk{ci}.mlpackage")
        if Path(out_path).exists():
            print(f"  Chunk {ci} already exists, skipping")
            continue
        print(f"\n  Exporting chunk {ci} (LUT{lut_bits} gs={per_channel})...")
        t0 = time.time()
        mlmodel = converter.convert_part_2(model, chunk_idx=ci, total_chunks=num_chunks)
        mlmodel.save(out_path)
        print(f"  Saved: {out_path} ({time.time()-t0:.1f}s)")
        sz_mb = sum(f.stat().st_size for f in Path(out_path).rglob("*") if f.is_file()) / 1e6
        print(f"  Size: {sz_mb:.1f}MB")
        del mlmodel
        gc.collect()

    # Symlink embeddings + lm_head
    base = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"
    for name in ["embeddings.mlpackage", "lm_head.mlpackage"]:
        link = os.path.join(out_dir, name)
        if not os.path.exists(link):
            os.symlink(os.path.join(base, name), link)
            print(f"  Symlinked {name}")

    del model, converter
    gc.collect()


# ─── Test Runtime ────────────────────────────────────────────────────

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

        print(f"    [{label}] Loading...", end="", flush=True)
        t0 = time.time()
        self.embed = ct.models.MLModel(_find(model_dir, "embeddings"), compute_units=cu)
        self.lmhead = ct.models.MLModel(_find(model_dir, "lm_head"), compute_units=cu)
        self.ffns = []
        for ci in range(num_chunks):
            self.ffns.append(ct.models.MLModel(
                _find(model_dir, f"ffn_LUT4_chunk{ci}"), compute_units=cu))
        print(f" {time.time()-t0:.0f}s")

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
        t0 = time.time()
        for pos, tid in enumerate(prompt_ids):
            nxt = self._step(tid, pos)
        t_prompt = time.time()
        gen = [nxt]
        p = len(prompt_ids)
        for _ in range(max_new_tokens - 1):
            if gen[-1] in self.stop_ids or p >= self.ctx - 1:
                break
            nxt = self._step(gen[-1], p)
            p += 1
            gen.append(nxt)
        t_done = time.time()
        prompt_ms = (t_prompt - t0) * 1000
        decode_ms = (t_done - t_prompt) * 1000
        return [int(x) for x in gen], prompt_ms, decode_ms

    def benchmark_chunk(self, chunk_idx=0, n_steps=12):
        """Benchmark a single chunk's predict latency."""
        spec = self.ffns[chunk_idx].get_spec()
        inputs = {}
        for inp in spec.description.input:
            try:
                shape = tuple(int(x) for x in inp.type.multiArrayType.shape)
                dt = inp.type.multiArrayType.dataType
                if dt == 131104:
                    inputs[inp.name] = np.zeros(shape, dtype=np.int32)
                else:
                    inputs[inp.name] = np.zeros(shape, dtype=np.float16)
            except Exception:
                pass
        state = self.ffns[chunk_idx].make_state()
        # Warmup
        for _ in range(3):
            self.ffns[chunk_idx].predict(inputs, state=state)
        times = []
        for _ in range(n_steps):
            t0 = time.time()
            self.ffns[chunk_idx].predict(inputs, state=state)
            times.append(time.time() - t0)
        return np.mean(times) * 1000, np.std(times) * 1000

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


def get_model_size_mb(model_dir, num_chunks):
    """Total size of FFN chunks in MB."""
    total = 0
    for ci in range(num_chunks):
        chunk_dir = Path(model_dir) / f"ffn_LUT4_chunk{ci}.mlpackage"
        if chunk_dir.exists():
            total += sum(f.stat().st_size for f in chunk_dir.rglob("*") if f.is_file())
    return total / 1e6


# ─── Test phase ──────────────────────────────────────────────────────

def test_config(cfg, tokenizer, hf_results, ctx, num_chunks, max_new_tokens):
    """Test one config: quality across 8 prompts + chunk latency + size."""
    model_dir = cfg["dir"]
    label = cfg["label"]

    if not Path(model_dir).exists():
        print(f"  [{label}] Directory not found, skipping")
        return None

    rt = CoreMLRuntime(model_dir, tokenizer, ctx, num_chunks, label=label)

    # Benchmark single chunk
    chunk_ms, chunk_std = rt.benchmark_chunk(0)

    # Model size
    size_mb = get_model_size_mb(model_dir, num_chunks)

    results = {"per_prompt": {}, "chunk_ms": chunk_ms, "chunk_std": chunk_std,
               "size_mb": size_mb}
    total_q, total_fm, total_dec = 0, 0, 0

    for pi, p in enumerate(PROMPTS):
        hf_r = hf_results[p["label"]]
        prompt_ids = make_prompt_ids(tokenizer, p["prompt"])
        gen, prompt_ms, decode_ms = rt.generate(prompt_ids, max_new_tokens)
        text = tokenizer.decode(gen, skip_special_tokens=False)
        first_match = gen[0] == hf_r["gen"][0] if gen and hf_r["gen"] else False
        quality = judge_quality(hf_r["text"], text, p["prompt"])

        results["per_prompt"][p["label"]] = {
            "gen": gen[:16], "text": text[:300],
            "first_match": first_match, "quality": quality,
            "decode_ms": decode_ms, "first_tok": gen[0] if gen else -1,
        }

        total_q += quality
        total_fm += int(first_match)
        total_dec += decode_ms

        sym = "✓" if first_match else "✗"
        print(f"    {p['label']:<15} first={gen[0]:>7} {sym}  q={quality}  "
              f"dec={decode_ms:.0f}ms  {text[:60]}...")

    n = len(PROMPTS)
    results["avg_quality"] = total_q / n
    results["first_match_rate"] = total_fm / n
    results["avg_decode_ms"] = total_dec / n

    print(f"  ── {label}: qual={total_q/n:.2f}/3  1st_match={total_fm}/{n}  "
          f"dec={total_dec/n:.0f}ms  chunk={chunk_ms:.1f}ms  size={size_mb:.0f}MB")

    rt.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B")
    ap.add_argument("--tokenizer", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    ap.add_argument("--hf-cache", default="tests/dev/fp32_multiprompt_hf_cache.json")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--phase", default="both", choices=["export", "test", "both"])
    ap.add_argument("--out", default="tests/dev/lut6_comparison_report.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    # ── Export phase ──
    if args.phase in ("export", "both"):
        for cfg in CONFIGS:
            if not cfg.get("export", False):
                continue
            label = cfg["label"]
            out_dir = cfg["dir"]
            # Check if already done
            all_exist = all(
                Path(out_dir, f"ffn_LUT4_chunk{ci}.mlpackage").exists()
                for ci in range(args.num_chunks)
            )
            if all_exist:
                print(f"\n  [{label}] All chunks exist, skipping export")
                continue

            print(f"\n{'='*70}")
            print(f"  EXPORT: {label} → {out_dir}")
            print(f"{'='*70}")
            export_all_chunks(cfg, args.hf_model, args.ctx, args.num_chunks)

    # ── Test phase ──
    if args.phase in ("test", "both"):
        print("\nLoading cached HF results...")
        with open(args.hf_cache) as f:
            hf_results = json.load(f)

        all_results = {}
        for cfg in CONFIGS:
            label = cfg["label"]
            print(f"\n{'='*70}")
            print(f"  TEST: {label}")
            print(f"{'='*70}")
            r = test_config(cfg, tokenizer, hf_results, args.ctx, args.num_chunks,
                            args.max_new_tokens)
            if r:
                all_results[label] = r

        # ── Summary ──
        print(f"\n{'='*100}")
        print(f"  COMPREHENSIVE COMPARISON")
        print(f"{'='*100}")

        print(f"\n  {'Config':<20} {'1st match':>10} {'Avg Q':>7} {'Chunk ms':>10} "
              f"{'Decode ms':>10} {'Size MB':>8} {'Overhead':>9}")
        print(f"  {'-'*80}")

        fp16_chunk = all_results.get("LUT4_gs8_fp16", {}).get("chunk_ms", 1)
        fp16_decode = all_results.get("LUT4_gs8_fp16", {}).get("avg_decode_ms", 1)

        for cfg in CONFIGS:
            label = cfg["label"]
            if label not in all_results:
                continue
            r = all_results[label]
            overhead = r["chunk_ms"] / fp16_chunk if fp16_chunk > 0 else 0
            print(f"  {label:<20} {r['first_match_rate']*100:>8.0f}%  "
                  f"{r['avg_quality']:>5.2f}/3 {r['chunk_ms']:>8.1f}ms "
                  f"{r['avg_decode_ms']:>9.0f}ms {r['size_mb']:>7.0f}MB "
                  f"{overhead:>8.2f}x")

        # Per-prompt breakdown for top configs
        print(f"\n  Per-prompt first-token match:")
        print(f"  {'Prompt':<15}", end="")
        for cfg in CONFIGS:
            if cfg["label"] in all_results:
                print(f" {cfg['label'][:12]:>13}", end="")
        print(f" {'HF ref':>8}")
        print(f"  {'-'*90}")

        for p in PROMPTS:
            lab = p["label"]
            hf_first = hf_results[lab]["gen"][0]
            print(f"  {lab:<15}", end="")
            for cfg in CONFIGS:
                if cfg["label"] not in all_results:
                    continue
                r = all_results[cfg["label"]]
                pp = r["per_prompt"].get(lab, {})
                ft = pp.get("first_tok", -1)
                fm = pp.get("first_match", False)
                sym = "✓" if fm else "✗"
                print(f" {ft:>11}{sym:>2}", end="")
            print(f" {hf_first:>8}")

        # Text comparison for key prompts
        print(f"\n  Selected outputs:")
        for lab in ["zh_recipe", "zh_physics", "en_code", "en_story"]:
            hf_text = hf_results[lab]["text"][:80]
            print(f"\n  [{lab}] HF: {hf_text}")
            for cfg in CONFIGS:
                if cfg["label"] not in all_results:
                    continue
                pp = all_results[cfg["label"]]["per_prompt"].get(lab, {})
                print(f"    {cfg['label']:<20} {pp.get('text', '')[:80]}")

        # Save report
        report = {
            "configs": {label: r for label, r in all_results.items()},
            "prompts": [p["prompt"] for p in PROMPTS],
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n  Report saved: {out}")


if __name__ == "__main__":
    main()
