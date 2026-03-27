#!/usr/bin/env python3
"""Run CoreML FP32 models on real ANE and compare against cached HF/FP16 results.

Reads HF reference data from the previous report, then runs only CoreML-FP32
inference on ANE. No HF model loading needed (saves 8GB RAM).

Usage:
    python tests/dev/debug_fp32_ane_only_parity.py \
        --fp32-dir /tmp/qwen35_fp32_chunks \
        --existing-dir /Users/yw68/Anemll_remote_run/qwen35_milestone1_3 \
        --tokenizer /Users/yw68/Anemll/qwen3_5_stable_models \
        --cached-report tests/dev/fp32_stage_parity_comparison_report.json
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np

sys.modules["profile"] = importlib.import_module("profile")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import AutoTokenizer

NEG_INF = np.float16(-65504.0)


def cosine(a, b):
    x, y = a.astype(np.float64).reshape(-1), b.astype(np.float64).reshape(-1)
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    return float(np.dot(x, y) / (nx * ny + 1e-12))


def stage_metrics(a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {"cosine": cosine(a, b), "max_abs": float(d.max()), "mean_abs": float(d.mean())}


def make_prompt_ids(tokenizer, prompt, enable_thinking):
    msgs = [{"role": "user", "content": prompt}]
    out = tokenizer.apply_chat_template(msgs, return_tensors="pt",
                                         add_generation_prompt=True, enable_thinking=enable_thinking)
    ids = out.input_ids[0].tolist() if hasattr(out, "input_ids") else out[0].tolist()
    return [int(x) for x in ids]


def token_compare(a, b):
    n = min(len(a), len(b))
    m = sum(1 for i in range(n) if a[i] == b[i])
    first_div = next((i for i in range(n) if a[i] != b[i]), None)
    return {"common_len": n, "match_count": m, "match_ratio": (m / n) if n else 0, "first_divergence_index": first_div}


def _find_model(base_dir, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = str(Path(base_dir) / f"{name}{ext}")
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"Missing: {name} in {base_dir}")


def _load_model(path, cu, function_name=None):
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, cu)
    kw = {"compute_units": cu}
    if function_name:
        kw["function_name"] = function_name
    return ct.models.MLModel(path, **kw)


class CoreMLRuntime:
    def __init__(self, model_dir, tokenizer_path, ctx, num_chunks):
        cu = ct.ComputeUnit.CPU_AND_NE
        self.ctx = ctx
        self.num_chunks = num_chunks
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

        print(f"    Loading embeddings...")
        self.embed = _load_model(_find_model(model_dir, "embeddings"), cu)
        print(f"    Loading lm_head...")
        self.lmhead = _load_model(_find_model(model_dir, "lm_head"), cu)

        self.ffns = []
        for ci in range(num_chunks):
            print(f"    Loading ffn chunk {ci}...")
            path = _find_model(model_dir, f"ffn_LUT4_chunk{ci}")
            self.ffns.append(_load_model(path, cu))

        spec = self.lmhead.get_spec()
        out_names = [o.name for o in spec.description.output]
        if "logits" in out_names or "output_logits" in out_names:
            self.lm_mode = "logits"
            split = sorted([n for n in out_names if n.startswith("logits") and n[6:].isdigit()])
            self.logits_keys = split if split else None
            self.logits_key = None if split else ("output_logits" if "output_logits" in out_names else "logits")
        else:
            self.lm_mode = "argmax"; self.logits_key = None; self.logits_keys = None

        spec0 = self.ffns[0].get_spec()
        inmap = {}
        for inp in spec0.description.input:
            try:
                inmap[inp.name] = tuple(int(x) for x in inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.conv_shape = inmap.get("linear_conv_state", (8, 1024, 32))
        self.rec_shape = inmap.get("linear_recurrent_state", (8, 32, 128, 128))

        self.stop_ids = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_ids.add(int(self.tokenizer.eos_token_id))
        for s in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(s)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_ids.add(int(tid))
        self.reset()

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [np.zeros(self.conv_shape, dtype=np.float16) for _ in range(self.num_chunks)]
        self.lin_recs = [np.zeros(self.rec_shape, dtype=np.float16) for _ in range(self.num_chunks)]
        self.tok_buf = np.zeros((1, 1), dtype=np.int32)
        self.mask_buf = np.full((1, 1, 1, self.ctx), NEG_INF, dtype=np.float16)
        self.pos_buf = np.zeros((1,), dtype=np.int32)

    def extract_logits(self, lm_out):
        if self.lm_mode != "logits":
            return None
        if self.logits_keys:
            return np.concatenate([lm_out[k].reshape(-1).astype(np.float32) for k in self.logits_keys])
        return lm_out[self.logits_key].reshape(-1).astype(np.float32)

    def step(self, token_id, pos, capture_stages=False):
        self.tok_buf[0, 0] = np.int32(token_id)
        embed_out = self.embed.predict({"input_ids": self.tok_buf})
        hidden = list(embed_out.values())[0]
        self.mask_buf[:, :, :, :] = NEG_INF
        self.mask_buf[:, :, :, :pos + 1] = 0
        self.pos_buf[0] = np.int32(pos)
        chunk_hiddens = []
        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": self.pos_buf, "causal_mask": self.mask_buf,
                "current_pos": self.pos_buf,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if "linear_conv_state_out" in out:
                self.lin_convs[ci] = out["linear_conv_state_out"]
                self.lin_recs[ci] = out["linear_recurrent_state_out"]
            if capture_stages:
                chunk_hiddens.append(hidden.copy())
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        logits = self.extract_logits(lm_out)
        next_id = int(np.argmax(logits)) if logits is not None else int(lm_out["argmax_idx"].reshape(-1)[0])
        if capture_stages:
            return next_id, logits, embed_out["hidden_states"].copy(), chunk_hiddens
        return next_id, logits, None, None

    def run_prompt(self, prompt_ids):
        for pos, tid in enumerate(prompt_ids):
            cap = (pos == len(prompt_ids) - 1)
            nxt, logits, emb, chunks = self.step(tid, pos, capture_stages=cap)
        return nxt, emb, chunks

    def decode(self, start_id, start_pos, max_tokens):
        gen = [int(start_id)]
        pos = int(start_pos)
        for _ in range(max_tokens - 1):
            if gen[-1] in self.stop_ids or pos >= self.ctx - 1:
                break
            nxt, _, _, _ = self.step(gen[-1], pos)
            pos += 1
            gen.append(int(nxt))
        return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-dir", default="/tmp/qwen35_fp32_chunks")
    ap.add_argument("--existing-dir", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1_3")
    ap.add_argument("--tokenizer", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    ap.add_argument("--cached-report", default="tests/dev/fp32_stage_parity_comparison_report.json")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="tests/dev/fp32_ane_parity_report.json")
    args = ap.parse_args()

    # Load cached HF + FP16 results
    print("Loading cached report...")
    with open(args.cached_report) as f:
        cached = json.load(f)
    prompt = cached["prompt"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    report = {"prompt": prompt, "ctx": args.ctx, "num_chunks": args.num_chunks, "runs": []}

    for run_idx, cached_run in enumerate(cached["runs"]):
        enable_thinking = cached_run["enable_thinking"]
        label = "think_on" if enable_thinking else "think_off"
        print(f"\n{'='*70}")
        print(f"  {label.upper()} — prompt: {prompt}")
        print(f"{'='*70}")

        prompt_ids = make_prompt_ids(tokenizer, prompt, enable_thinking)
        prompt_len = len(prompt_ids)

        # Cached data
        hf_first = cached_run["first_token"]["hf"]
        fp16_first = cached_run["first_token"]["coreml"]
        hf_gen = cached_run["hf_gen"]
        fp16_gen = cached_run["coreml_gen"]
        fp16_stage_cos = {}
        for key in ["embed", "chunk0", "chunk1", "chunk2", "chunk3"]:
            if key in cached_run["stage_metrics"]:
                fp16_stage_cos[key] = cached_run["stage_metrics"][key]["coreml_vs_hf"]["cosine"]

        # Run CoreML FP32 on ANE
        print("  [CoreML-FP32] Loading on ANE...")
        rt32 = CoreMLRuntime(args.fp32_dir, args.tokenizer, args.ctx, args.num_chunks)
        print("  [CoreML-FP32] Running prompt...")
        cm32_first, cm32_embed, cm32_chunks = rt32.run_prompt(prompt_ids)
        print("  [CoreML-FP32] Decoding...")
        cm32_gen = rt32.decode(cm32_first, prompt_len, args.max_new_tokens)
        del rt32
        gc.collect()

        fp32_parity = token_compare(hf_gen, cm32_gen)
        fp16_parity = token_compare(hf_gen, fp16_gen)

        # ── Print results ──
        print(f"\n  First token: HF={hf_first}  FP16(ANE)={fp16_first}  FP32(ANE)={cm32_first}")
        print(f"  FP16 first match: {fp16_first == hf_first}  |  FP32 first match: {cm32_first == hf_first}")

        print(f"\n  {'Stage':<12} {'FP16(ANE)↔HF cos':<22} {'FP32(ANE)↔HF cos':<22} {'Gap closed'}")
        print(f"  {'-'*65}")
        # We can't compute FP32 stage cosine vs HF directly (we don't have HF hidden states),
        # but we CAN compare FP32 output vs FP16 output to see improvement
        # For now report decoded token match
        print(f"\n  (Stage-level cosine for FP32 requires HF hidden states;")
        print(f"   cached FP16 cosines shown for reference)")
        for key, c16 in fp16_stage_cos.items():
            print(f"  {key:<12} cos={c16:<20.10f} (FP32 hidden comparison not cached)")

        print(f"\n  Decode parity vs HF ({args.max_new_tokens} tokens):")
        print(f"    FP16(ANE): {fp16_parity['match_count']}/{fp16_parity['common_len']} "
              f"({fp16_parity['match_ratio']*100:.1f}%) first_div={fp16_parity['first_divergence_index']}")
        print(f"    FP32(ANE): {fp32_parity['match_count']}/{fp32_parity['common_len']} "
              f"({fp32_parity['match_ratio']*100:.1f}%) first_div={fp32_parity['first_divergence_index']}")

        cm32_text = tokenizer.decode(cm32_gen, skip_special_tokens=False)[:400]
        hf_text = cached_run.get("hf_text", "")[:150]
        fp16_text = cached_run.get("coreml_text", "")[:150]
        print(f"\n  HF:        {hf_text}")
        print(f"  FP16(ANE): {fp16_text}")
        print(f"  FP32(ANE): {cm32_text[:150]}")

        report["runs"].append({
            "enable_thinking": enable_thinking, "prompt_len": prompt_len,
            "first_token": {"hf": hf_first, "fp16": fp16_first, "fp32": int(cm32_first),
                           "fp16_match": fp16_first == hf_first, "fp32_match": int(cm32_first) == hf_first},
            "decode_parity_fp16": fp16_parity, "decode_parity_fp32": fp32_parity,
            "fp16_stage_cosines": fp16_stage_cos,
            "hf_gen": hf_gen, "fp16_gen": fp16_gen, "fp32_gen": cm32_gen[:48],
            "hf_text": hf_text, "fp16_text": fp16_text, "fp32_text": cm32_text,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved: {out}")


if __name__ == "__main__":
    main()
