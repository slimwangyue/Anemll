#!/usr/bin/env python3
"""Export FFN decode chunks with compute_precision=FLOAT32, then run
real ANE stage-parity comparison against HF fp16.

Phase 1: Export 4 FFN decode chunks with FP32 precision (one at a time to fit 16GB)
Phase 2: Symlink embeddings + lm_head from existing models
Phase 3: Run 3-way parity: CoreML-FP16(ANE) vs CoreML-FP32(ANE) vs HF-fp16

Usage:
    python tests/dev/debug_fp32_ane_export_and_parity.py \
        --hf-model /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
        --existing-dir /Users/yw68/Anemll_remote_run/qwen35_milestone1_3 \
        --tokenizer /Users/yw68/Anemll/qwen3_5_stable_models \
        --fp32-dir /tmp/qwen35_fp32_chunks \
        --ctx 1024 --num-chunks 4 --max-new-tokens 48
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
from typing import Any

import coremltools as ct
import numpy as np
import torch

sys.modules["profile"] = importlib.import_module("profile")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from anemll.models.qwen3_5_model import (
    MODEL_DTYPE,
    TEST_DEVICE,
    Qwen35Config,
    Qwen35ForCausalLM,
    ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from transformers import AutoModelForCausalLM, AutoTokenizer

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


# ════════════════════════════════════════════════════════════════════
# Phase 1: Export FFN chunks with FP32 precision
# ════════════════════════════════════════════════════════════════════

def export_fp32_chunk(hf_model_path: str, ctx: int, num_chunks: int, chunk_idx: int,
                       output_path: str, lut_bits: int = 4, per_channel: int = 8):
    """Export a single FFN decode chunk with compute_precision=FLOAT32."""
    print(f"\n  [export] Loading model for chunk {chunk_idx}...")
    cfg = Qwen35Config.from_json(os.path.join(hf_model_path, "config.json"))
    cfg.context_length = ctx
    cfg.state_length = ctx
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(hf_model_path), "weight load failed"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    conv = Qwen35Converter(model, context_length=ctx, batch_size=256,
                           num_chunks=num_chunks, lut_bits=lut_bits, per_channel=per_channel)

    # Call convert_part_2 but intercept the compute_precision
    # We'll monkey-patch ct.convert to use FLOAT32
    original_convert = ct.convert

    def patched_convert(*args, **kwargs):
        kwargs['compute_precision'] = ct.precision.FLOAT32
        print(f"    [patched] Using compute_precision=FLOAT32")
        return original_convert(*args, **kwargs)

    ct.convert = patched_convert
    try:
        print(f"  [export] Converting chunk {chunk_idx} (FP32 + LUT{lut_bits})...")
        t0 = time.time()
        ml = conv.convert_part_2(model, chunk_idx=chunk_idx, total_chunks=num_chunks)
        ml.save(output_path)
        print(f"  [export] Saved chunk {chunk_idx} ({time.time()-t0:.1f}s): {output_path}")
    finally:
        ct.convert = original_convert

    del ml, conv, model
    gc.collect()


# ════════════════════════════════════════════════════════════════════
# CoreML runtime (same as stage parity script)
# ════════════════════════════════════════════════════════════════════

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

        self.embed = _load_model(_find_model(model_dir, "embeddings"), cu)
        self.lmhead = _load_model(_find_model(model_dir, "lm_head"), cu)

        self.ffns = []
        for ci in range(num_chunks):
            path = _find_model(model_dir, f"ffn_LUT4_chunk{ci}")
            self.ffns.append(_load_model(path, cu))

        # lm_head mode
        spec = self.lmhead.get_spec()
        out_names = [o.name for o in spec.description.output]
        if "logits" in out_names or "output_logits" in out_names:
            self.lm_mode = "logits"
            split = sorted([n for n in out_names if n.startswith("logits") and n[6:].isdigit()])
            self.logits_keys = split if split else None
            self.logits_key = None if split else ("output_logits" if "output_logits" in out_names else "logits")
        else:
            self.lm_mode = "argmax"
            self.logits_key = None
            self.logits_keys = None

        # State shapes from first chunk
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


# ════════════════════════════════════════════════════════════════════
# HF reference
# ════════════════════════════════════════════════════════════════════

def hf_run_prompt(model, prompt_ids):
    past = None
    with torch.no_grad():
        for i, tid in enumerate(prompt_ids):
            x = torch.tensor([[tid]], dtype=torch.long, device=model.device)
            out = model(input_ids=x, past_key_values=past, use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            if i == len(prompt_ids) - 1:
                return int(torch.argmax(out.logits[0, -1, :]).item()), out.hidden_states
    return None, None


def hf_decode(model, start_id, max_tokens, stop_ids):
    gen = [int(start_id)]
    past = None
    with torch.no_grad():
        x = torch.tensor([[start_id]], dtype=torch.long, device=model.device)
        out = model(input_ids=x, use_cache=True)
        past = out.past_key_values
        for _ in range(max_tokens - 1):
            if gen[-1] in stop_ids:
                break
            x = torch.tensor([[gen[-1]]], dtype=torch.long, device=model.device)
            out = model(input_ids=x, past_key_values=past, use_cache=True)
            past = out.past_key_values
            gen.append(int(torch.argmax(out.logits[0, -1, :]).item()))
    return gen


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--existing-dir", required=True, help="Dir with current FP16 models")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--fp32-dir", default="/tmp/qwen35_fp32_chunks")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--prompt", default="教我做红烧肉")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--skip-export", action="store_true", help="Skip re-export if FP32 chunks exist")
    ap.add_argument("--out", default="tests/dev/fp32_ane_parity_report.json")
    args = ap.parse_args()

    os.makedirs(args.fp32_dir, exist_ok=True)

    # ── Phase 1: Export FP32 FFN chunks ──
    print("=" * 70)
    print("  Phase 1: Export FFN chunks with compute_precision=FLOAT32")
    print("=" * 70)

    for ci in range(args.num_chunks):
        out_path = os.path.join(args.fp32_dir, f"ffn_LUT4_chunk{ci}.mlpackage")
        if args.skip_export and os.path.exists(out_path):
            print(f"  [skip] chunk {ci} already exists")
            continue
        export_fp32_chunk(args.hf_model, args.ctx, args.num_chunks, ci, out_path)

    # ── Phase 2: Symlink embeddings + lm_head ──
    print(f"\n{'='*70}")
    print("  Phase 2: Link embeddings + lm_head from existing models")
    print(f"{'='*70}")

    for name in ["embeddings", "lm_head"]:
        src = _find_model(args.existing_dir, name)
        dst = os.path.join(args.fp32_dir, os.path.basename(src))
        if os.path.exists(dst):
            if os.path.islink(dst):
                os.unlink(dst)
            else:
                print(f"  [keep] {dst}")
                continue
        os.symlink(src, dst)
        print(f"  Linked: {os.path.basename(src)} -> {src}")

    # ── Phase 3: Parity comparison on ANE ──
    print(f"\n{'='*70}")
    print("  Phase 3: Real ANE Parity — FP16 vs FP32 vs HF fp16")
    print(f"{'='*70}")

    hf_tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)
    cfg_path = os.path.join(args.hf_model, "config.json")
    with open(cfg_path) as f:
        raw_cfg = json.load(f)
    hf_num_layers = raw_cfg.get("num_hidden_layers", raw_cfg.get("text_config", {}).get("num_hidden_layers", 36))
    base, rem = divmod(hf_num_layers, args.num_chunks)
    chunk_cum = []
    csum = 0
    for ci in range(args.num_chunks):
        csum += base + (1 if ci < rem else 0)
        chunk_cum.append(csum)

    report = {"prompt": args.prompt, "ctx": args.ctx, "num_chunks": args.num_chunks,
              "fp32_dir": args.fp32_dir, "existing_dir": args.existing_dir, "runs": []}

    for enable_thinking in (False, True):
        label = "think_on" if enable_thinking else "think_off"
        print(f"\n{'='*70}")
        print(f"  {label.upper()} — prompt: {args.prompt}")
        print(f"{'='*70}")

        prompt_ids = make_prompt_ids(hf_tok, args.prompt, enable_thinking)
        prompt_len = len(prompt_ids)
        print(f"  Prompt length: {prompt_len} tokens")

        # ── HF fp16 ──
        print("  [HF] Loading...")
        hf = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.float16, low_cpu_mem_usage=True)
        hf = hf.to("cpu").eval()

        print("  [HF] Running prompt + decode...")
        hf_first, hf_hidden = hf_run_prompt(hf, prompt_ids)

        stop_ids = set()
        if hf_tok.eos_token_id is not None:
            stop_ids.add(int(hf_tok.eos_token_id))
        for s in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
            tid = hf_tok.convert_tokens_to_ids(s)
            if tid is not None and tid != hf_tok.unk_token_id:
                stop_ids.add(int(tid))
        hf_gen = hf_decode(hf, hf_first, args.max_new_tokens, stop_ids)

        hf_embed_np = hf_hidden[0][0, -1:, :].detach().float().cpu().numpy().reshape(1, 1, -1)
        hf_chunk_nps = []
        for ci in range(args.num_chunks):
            idx = min(chunk_cum[ci], hf_num_layers)
            hf_chunk_nps.append(hf_hidden[idx][0, -1:, :].detach().float().cpu().numpy().reshape(1, 1, -1))
        del hf, hf_hidden
        gc.collect()
        import time as _t; _t.sleep(3)  # let OS reclaim pages

        # ── CoreML FP32 (newly exported, ANE) — test this FIRST (main goal) ──
        print("  [CoreML-FP32] Loading...")
        rt32 = CoreMLRuntime(args.fp32_dir, args.tokenizer, args.ctx, args.num_chunks)
        print("  [CoreML-FP32] Running prompt + decode...")
        cm32_first, cm32_embed, cm32_chunks = rt32.run_prompt(prompt_ids)
        cm32_gen = rt32.decode(cm32_first, prompt_len, args.max_new_tokens)
        del rt32
        gc.collect()
        _t.sleep(2)

        # ── CoreML FP16 (existing, ANE) ──
        print("  [CoreML-FP16] Loading...")
        rt16 = CoreMLRuntime(args.existing_dir, args.tokenizer, args.ctx, args.num_chunks)
        print("  [CoreML-FP16] Running prompt + decode...")
        cm16_first, cm16_embed, cm16_chunks = rt16.run_prompt(prompt_ids)
        cm16_gen = rt16.decode(cm16_first, prompt_len, args.max_new_tokens)
        del rt16
        gc.collect()

        # ── Stage metrics ──
        stages = {}
        stages["embed"] = {
            "fp16_vs_hf": stage_metrics(cm16_embed.reshape(1, 1, -1), hf_embed_np),
            "fp32_vs_hf": stage_metrics(cm32_embed.reshape(1, 1, -1), hf_embed_np),
        }
        for ci in range(args.num_chunks):
            stages[f"chunk{ci}"] = {
                "hf_layer": chunk_cum[ci],
                "fp16_vs_hf": stage_metrics(cm16_chunks[ci].reshape(1, 1, -1), hf_chunk_nps[ci]) if cm16_chunks else None,
                "fp32_vs_hf": stage_metrics(cm32_chunks[ci].reshape(1, 1, -1), hf_chunk_nps[ci]) if cm32_chunks else None,
            }

        first_token = {
            "hf": int(hf_first), "fp16": int(cm16_first), "fp32": int(cm32_first),
            "fp16_match": int(cm16_first) == int(hf_first),
            "fp32_match": int(cm32_first) == int(hf_first),
        }
        fp16_parity = token_compare(hf_gen, cm16_gen)
        fp32_parity = token_compare(hf_gen, cm32_gen)

        # ── Print summary ──
        print(f"\n  First token: HF={hf_first}  FP16={cm16_first}  FP32={cm32_first}")
        print(f"  {'Stage':<12} {'CoreML-FP16(ANE)↔HF':<24} {'CoreML-FP32(ANE)↔HF':<24} {'Gap closed'}")
        print(f"  {'-'*75}")
        for sn in ["embed"] + [f"chunk{ci}" for ci in range(args.num_chunks)]:
            c16 = stages[sn]["fp16_vs_hf"]["cosine"] if stages[sn].get("fp16_vs_hf") else 0
            c32 = stages[sn]["fp32_vs_hf"]["cosine"] if stages[sn].get("fp32_vs_hf") else 0
            gap = ((c32 - c16) / (1.0 - c16) * 100) if c16 < 1.0 else 0
            print(f"  {sn:<12} cos={c16:<20.10f} cos={c32:<20.10f} {gap:+.1f}%")

        print(f"\n  Decode parity vs HF ({args.max_new_tokens} tokens):")
        print(f"    FP16(ANE): {fp16_parity['match_count']}/{fp16_parity['common_len']} "
              f"({fp16_parity['match_ratio']*100:.1f}%) first_div={fp16_parity['first_divergence_index']}")
        print(f"    FP32(ANE): {fp32_parity['match_count']}/{fp32_parity['common_len']} "
              f"({fp32_parity['match_ratio']*100:.1f}%) first_div={fp32_parity['first_divergence_index']}")

        hf_text = hf_tok.decode(hf_gen, skip_special_tokens=False)[:400]
        cm16_text = hf_tok.decode(cm16_gen, skip_special_tokens=False)[:400]
        cm32_text = hf_tok.decode(cm32_gen, skip_special_tokens=False)[:400]
        print(f"\n  HF:        {hf_text[:150]}")
        print(f"  FP16(ANE): {cm16_text[:150]}")
        print(f"  FP32(ANE): {cm32_text[:150]}")

        report["runs"].append({
            "enable_thinking": enable_thinking, "prompt_len": prompt_len,
            "first_token": first_token,
            "decode_parity_fp16": fp16_parity, "decode_parity_fp32": fp32_parity,
            "stage_metrics": stages,
            "hf_gen": hf_gen[:48], "fp16_gen": cm16_gen[:48], "fp32_gen": cm32_gen[:48],
            "hf_text": hf_text, "fp16_text": cm16_text, "fp32_text": cm32_text,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved: {out}")


if __name__ == "__main__":
    main()
