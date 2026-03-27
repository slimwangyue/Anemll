#!/usr/bin/env python3
"""Stage parity comparison: CoreML-FP16 (ANE) vs ANEMLL-PyTorch (FP32-proxy) vs HF-fp16.

Runs three inference paths on the same prompt:
  1. CoreML FP16 (existing ANE models) — current baseline
  2. ANEMLL PyTorch fp16 (same weights as CoreML, CPU) — upper bound for FP32 CoreML
     (because our mitigation experiment proved FP32-CoreML matches PyTorch to cos=0.99995)
  3. HF fp16 (original HuggingFace model) — reference

This avoids hours of re-exporting: the PyTorch path IS the Strategy-D ceiling.

Usage:
    python tests/dev/debug_fp32_stage_parity_comparison.py \
        --coreml-dir /Users/yw68/Anemll_remote_run/qwen35_milestone1_3 \
        --tokenizer /Users/yw68/Anemll/qwen3_5_stable_models \
        --hf-model /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
        --ctx 1024 --num-chunks 4 --max-new-tokens 48
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Guard against profile module shadowing when transformers imports torch._dynamo.
sys.modules["profile"] = importlib.import_module("profile")

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from anemll.models.qwen3_5_model import (
    MODEL_DTYPE,
    Qwen35Config,
    Qwen35ForCausalLM,
    ane_conv_state_shape,
)

NEG_INF = np.float16(-65504.0)


# ════════════════════════════════════════════════════════════════════
# Common utilities
# ════════════════════════════════════════════════════════════════════

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64).reshape(-1)
    y = b.astype(np.float64).reshape(-1)
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def stage_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    aa, bb = a.astype(np.float32), b.astype(np.float32)
    d = np.abs(aa - bb)
    return {"cosine": cosine(a, b), "max_abs": float(d.max()), "mean_abs": float(d.mean())}


def make_prompt_ids(tokenizer, prompt: str, enable_thinking: bool) -> list[int]:
    msgs = [{"role": "user", "content": prompt}]
    out = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    ids = out.input_ids[0].tolist() if hasattr(out, "input_ids") else out[0].tolist()
    return [int(x) for x in ids]


def token_compare(a: list[int], b: list[int]) -> dict:
    n = min(len(a), len(b))
    m = sum(1 for i in range(n) if a[i] == b[i])
    first_div = next((i for i in range(n) if a[i] != b[i]), None)
    return {"a_len": len(a), "b_len": len(b), "common_len": n, "match_count": m,
            "match_ratio": (m / n) if n else 0.0, "first_divergence_index": first_div}


# ════════════════════════════════════════════════════════════════════
# Path 1: CoreML FP16 (existing ANE models)
# ════════════════════════════════════════════════════════════════════

def _find_model(base_dir: str, name: str) -> str:
    for ext in (".mlmodelc", ".mlpackage"):
        p = str(Path(base_dir) / f"{name}{ext}")
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"Missing model: {name} in {base_dir}")


def _load_model(path: str, cu, function_name=None):
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, cu)
    kwargs = {"compute_units": cu}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


@dataclass
class CoreMLRuntime:
    embed: Any; lmhead: Any; ffns: list; tokenizer: Any
    ctx: int; num_chunks: int; stop_ids: set
    lmhead_mode: str; logits_key: str | None; logits_keys: list | None
    states: list; lin_convs: list; lin_recs: list
    tok_buf: np.ndarray; mask_buf: np.ndarray; pos_buf: np.ndarray


def build_coreml_runtime(model_dir, tokenizer_path, ctx, num_chunks):
    cu = ct.ComputeUnit.CPU_AND_NE
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    embed = _load_model(_find_model(model_dir, "embeddings"), cu)
    lmhead = _load_model(_find_model(model_dir, "lm_head"), cu)

    ffns = []
    combined = Path(model_dir) / "combined_LUT4_dedup"
    use_combined = combined.is_dir()
    for ci in range(num_chunks):
        if use_combined:
            path = _find_model(str(combined), f"chunk{ci}")
            ffns.append(_load_model(path, cu, function_name="infer"))
        else:
            path = _find_model(model_dir, f"ffn_LUT4_chunk{ci}")
            ffns.append(_load_model(path, cu))

    spec = lmhead.get_spec()
    out_names = [o.name for o in spec.description.output]
    if "logits" in out_names or "output_logits" in out_names:
        lm_mode = "logits"
        split = sorted([n for n in out_names if n.startswith("logits") and n[6:].isdigit()])
        logits_keys = split if split else None
        logits_key = None if split else ("output_logits" if "output_logits" in out_names else "logits")
    else:
        lm_mode = "argmax"; logits_key = None; logits_keys = None

    spec0 = ffns[0].get_spec()
    inmap = {}
    for inp in spec0.description.input:
        try:
            inmap[inp.name] = tuple(int(x) for x in inp.type.multiArrayType.shape)
        except Exception:
            pass
    conv_shape = inmap.get("linear_conv_state", (8, 1024, 32))
    rec_shape = inmap.get("linear_recurrent_state", (8, 32, 128, 128))

    states = [m.make_state() for m in ffns]
    lin_convs = [np.zeros(conv_shape, dtype=np.float16) for _ in range(num_chunks)]
    lin_recs = [np.zeros(rec_shape, dtype=np.float16) for _ in range(num_chunks)]

    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for s in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(int(tid))

    return CoreMLRuntime(
        embed=embed, lmhead=lmhead, ffns=ffns, tokenizer=tokenizer,
        ctx=ctx, num_chunks=num_chunks, stop_ids=stop_ids,
        lmhead_mode=lm_mode, logits_key=logits_key, logits_keys=logits_keys,
        states=states, lin_convs=lin_convs, lin_recs=lin_recs,
        tok_buf=np.zeros((1, 1), dtype=np.int32),
        mask_buf=np.full((1, 1, 1, ctx), NEG_INF, dtype=np.float16),
        pos_buf=np.zeros((1,), dtype=np.int32),
    )


def coreml_extract_logits(rt, lm_out):
    if rt.lmhead_mode != "logits":
        return None
    if rt.logits_keys:
        return np.concatenate([lm_out[k].reshape(-1).astype(np.float32) for k in rt.logits_keys])
    return lm_out[rt.logits_key].reshape(-1).astype(np.float32)


def coreml_step(rt, token_id, pos, capture_stages=False):
    rt.tok_buf[0, 0] = np.int32(token_id)
    embed_out = rt.embed.predict({"input_ids": rt.tok_buf})
    hidden = list(embed_out.values())[0]
    rt.mask_buf[:, :, :, :] = NEG_INF
    rt.mask_buf[:, :, :, :pos + 1] = 0
    rt.pos_buf[0] = np.int32(pos)
    chunk_hiddens = []
    for ci in range(rt.num_chunks):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": rt.pos_buf, "causal_mask": rt.mask_buf,
            "current_pos": rt.pos_buf,
            "linear_conv_state": rt.lin_convs[ci],
            "linear_recurrent_state": rt.lin_recs[ci],
        }
        out = rt.ffns[ci].predict(inp, state=rt.states[ci])
        hidden = out["output_hidden_states"]
        if "linear_conv_state_out" in out:
            rt.lin_convs[ci] = out["linear_conv_state_out"]
            rt.lin_recs[ci] = out["linear_recurrent_state_out"]
        if capture_stages:
            chunk_hiddens.append(hidden.copy())
    lm_out = rt.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    logits = coreml_extract_logits(rt, lm_out)
    next_id = int(np.argmax(logits)) if logits is not None else int(lm_out["argmax_idx"].reshape(-1)[0])
    if capture_stages:
        return next_id, logits, embed_out["hidden_states"].copy(), chunk_hiddens
    return next_id, logits, None, None


def coreml_run_prompt(rt, prompt_ids):
    for pos, tid in enumerate(prompt_ids):
        cap = (pos == len(prompt_ids) - 1)
        nxt, logits, emb, chunks = coreml_step(rt, tid, pos, capture_stages=cap)
    return nxt, emb, chunks


def coreml_decode(rt, start_id, start_pos, max_tokens):
    gen = [int(start_id)]
    pos = int(start_pos)
    for _ in range(max_tokens - 1):
        if gen[-1] in rt.stop_ids or pos >= rt.ctx - 1:
            break
        nxt, _, _, _ = coreml_step(rt, gen[-1], pos)
        pos += 1
        gen.append(int(nxt))
    return gen


# ════════════════════════════════════════════════════════════════════
# Path 2: ANEMLL PyTorch (same weights, CPU fp16 — FP32-CoreML proxy)
# ════════════════════════════════════════════════════════════════════

class AnemllPyTorchRunner:
    """Run the ANEMLL model chunk-by-chunk in pure PyTorch, capturing per-chunk hidden states."""

    def __init__(self, hf_model_path: str, ctx: int, num_chunks: int):
        self.ctx = ctx
        self.num_chunks = num_chunks
        self.cfg = Qwen35Config.from_json(os.path.join(hf_model_path, "config.json"))
        self.cfg.context_length = ctx
        self.cfg.state_length = ctx
        self.model = Qwen35ForCausalLM(self.cfg)
        assert self.model.load_pretrained_weights(hf_model_path), "weight load failed"
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Compute chunk boundaries (same logic as converter)
        total_layers = self.cfg.num_hidden_layers
        base, rem = divmod(total_layers, num_chunks)
        self.chunk_boundaries = []
        for ci in range(num_chunks):
            start = ci * base + min(ci, rem)
            end = start + base + (1 if ci < rem else 0)
            self.chunk_boundaries.append((start, end))

        # KV cache and linear states
        self.k_cache = torch.zeros(
            total_layers, self.cfg.num_key_value_heads, ctx, self.cfg.head_dim, dtype=MODEL_DTYPE)
        self.v_cache = torch.zeros(
            total_layers, self.cfg.num_key_value_heads, ctx, self.cfg.head_dim, dtype=MODEL_DTYPE)

        if self.cfg.has_linear_attention():
            conv_dim = (
                self.cfg.text_config.linear_num_key_heads * self.cfg.text_config.linear_key_head_dim * 2
                + self.cfg.text_config.linear_num_value_heads * self.cfg.text_config.linear_value_head_dim
            )
            conv_kernel = max(1, int(self.cfg.text_config.linear_conv_kernel_dim))
            # Use ANE-safe shape (same as converter export path)
            ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
            self.lin_conv = torch.zeros(total_layers, ane_dim1, ane_dim2, dtype=MODEL_DTYPE)
            v_heads = self.cfg.text_config.linear_num_value_heads
            k_dim = self.cfg.text_config.linear_key_head_dim
            v_dim = self.cfg.text_config.linear_value_head_dim
            self.lin_rec = torch.zeros(total_layers, v_heads, k_dim, v_dim, dtype=MODEL_DTYPE)
        else:
            self.lin_conv = None
            self.lin_rec = None

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        if self.lin_conv is not None:
            self.lin_conv.zero_()
            self.lin_rec.zero_()

    def step(self, token_id: int, pos: int, capture_stages: bool = False):
        """Single-token forward through all chunks, returning per-chunk hidden states."""
        with torch.no_grad():
            ids = torch.tensor([[token_id]], dtype=torch.int32)
            embed = self.model.model.embed_tokens(ids).to(MODEL_DTYPE)
            hidden = embed

            position_ids = torch.tensor([pos], dtype=torch.long)
            causal_mask = torch.full((1, 1, 1, self.ctx), -65504.0, dtype=MODEL_DTYPE)
            causal_mask[:, :, :, :pos + 1] = 0
            current_pos = torch.tensor([pos], dtype=torch.int32)

            chunk_hiddens = []
            for ci, (start, end) in enumerate(self.chunk_boundaries):
                hidden = self.model.model.process_layers_regular_single_token_export_local_state(
                    hidden_states=hidden,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    kv_cache_0=None,
                    k_cache=self.k_cache[start:end],
                    v_cache=self.v_cache[start:end],
                    linear_conv_state=self.lin_conv[start:end] if self.lin_conv is not None else None,
                    linear_recurrent_state=self.lin_rec[start:end] if self.lin_rec is not None else None,
                    start_layer=start,
                    end_layer=end,
                    apply_final_norm=(ci == self.num_chunks - 1),
                )
                if capture_stages:
                    chunk_hiddens.append(hidden.detach().cpu().numpy().copy())

            # lm_head (split into 16 Conv2d parts)
            h = hidden.permute(0, 2, 1).unsqueeze(2)
            parts = [getattr(self.model, f"lm_head16_{i+1}")(h).squeeze(2).permute(0, 2, 1)
                     for i in range(self.model.lm_head_split)]
            logits = torch.cat(parts, dim=-1)
            next_id = int(torch.argmax(logits[0, -1, :]).item())

            embed_np = embed.detach().cpu().numpy()
            return next_id, logits[0, -1, :].detach().cpu().numpy(), embed_np, chunk_hiddens if capture_stages else None

    def run_prompt(self, prompt_ids: list[int]):
        for pos, tid in enumerate(prompt_ids):
            cap = (pos == len(prompt_ids) - 1)
            nxt, logits, emb, chunks = self.step(tid, pos, capture_stages=cap)
        return nxt, emb, chunks

    def decode(self, start_id: int, start_pos: int, max_tokens: int, stop_ids: set):
        gen = [int(start_id)]
        pos = int(start_pos)
        for _ in range(max_tokens - 1):
            if gen[-1] in stop_ids or pos >= self.ctx - 1:
                break
            nxt, _, _, _ = self.step(gen[-1], pos)
            pos += 1
            gen.append(int(nxt))
        return gen


# ════════════════════════════════════════════════════════════════════
# Path 3: HuggingFace fp16
# ════════════════════════════════════════════════════════════════════

def hf_run_prompt(model, prompt_ids):
    past = None
    final_hidden = None
    with torch.no_grad():
        for i, tid in enumerate(prompt_ids):
            x = torch.tensor([[tid]], dtype=torch.long, device=model.device)
            out = model(input_ids=x, past_key_values=past, use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            if i == len(prompt_ids) - 1:
                final_hidden = out.hidden_states
                next_id = int(torch.argmax(out.logits[0, -1, :]).item())
    return next_id, final_hidden


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
            nxt = int(torch.argmax(out.logits[0, -1, :]).item())
            gen.append(nxt)
    return gen


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coreml-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--prompt", default="教我做红烧肉")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--hf-device", choices=["auto", "cpu", "mps"], default="cpu")
    ap.add_argument("--out", default="tests/dev/fp32_stage_parity_comparison_report.json")
    args = ap.parse_args()

    hf_tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)

    # Chunk boundary layer indices (needed for HF hidden_states mapping)
    # We'll read num_layers from config.json directly
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

    report = {"prompt": args.prompt, "ctx": args.ctx, "num_chunks": args.num_chunks, "runs": []}

    for enable_thinking in (False, True):
        label = "think_on" if enable_thinking else "think_off"
        print(f"\n{'='*70}")
        print(f"  {label.upper()} — prompt: {args.prompt}")
        print(f"{'='*70}")

        prompt_ids = make_prompt_ids(hf_tok, args.prompt, enable_thinking)
        prompt_len = len(prompt_ids)
        print(f"  Prompt length: {prompt_len} tokens")

        # ── Phase 1: HF fp16 ──
        print("  [Phase 1] Loading HF fp16...")
        hf = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.float16, low_cpu_mem_usage=True)
        hf = hf.to(args.hf_device).eval()

        print("  Running HF fp16...")
        hf_first, hf_hidden = hf_run_prompt(hf, prompt_ids)
        print("  Decoding (HF)...")
        stop_ids = set()
        if hf_tok.eos_token_id is not None:
            stop_ids.add(int(hf_tok.eos_token_id))
        for s in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
            tid = hf_tok.convert_tokens_to_ids(s)
            if tid is not None and tid != hf_tok.unk_token_id:
                stop_ids.add(int(tid))
        hf_gen = hf_decode(hf, hf_first, args.max_new_tokens, stop_ids)

        # Extract HF hidden states for stage comparison (keep as numpy)
        hf_embed_np = hf_hidden[0][0, -1:, :].detach().float().cpu().numpy().reshape(1, 1, -1)
        hf_chunk_nps = []
        for ci in range(args.num_chunks):
            idx = min(chunk_cum[ci], hf_num_layers)
            hf_chunk_nps.append(hf_hidden[idx][0, -1:, :].detach().float().cpu().numpy().reshape(1, 1, -1))

        del hf, hf_hidden
        gc.collect()

        # ── Phase 2: ANEMLL PyTorch (Strategy-D proxy) ──
        print("  [Phase 2] Loading ANEMLL PyTorch (FP32-proxy)...")
        pt_runner = AnemllPyTorchRunner(args.hf_model, args.ctx, args.num_chunks)

        print("  Running ANEMLL PyTorch...")
        pt_first, pt_embed, pt_chunks = pt_runner.run_prompt(prompt_ids)
        print("  Decoding (ANEMLL PyTorch)...")
        pt_gen = pt_runner.decode(pt_first, prompt_len, args.max_new_tokens, stop_ids)

        del pt_runner
        gc.collect()

        # ── Phase 3: CoreML FP16 (ANE) ──
        print("  [Phase 3] Loading CoreML FP16 (ANE)...")
        rt = build_coreml_runtime(args.coreml_dir, args.tokenizer, args.ctx, args.num_chunks)

        print("  Running CoreML FP16...")
        cm_first, cm_embed, cm_chunks = coreml_run_prompt(rt, prompt_ids)
        print("  Decoding (CoreML)...")
        cm_gen = coreml_decode(rt, cm_first, prompt_len, args.max_new_tokens)

        del rt
        gc.collect()

        # ── Stage metrics ──
        cm_embed_3d = cm_embed.reshape(1, 1, -1)
        pt_embed_3d = pt_embed.reshape(1, 1, -1)

        stages = {}
        stages["embed"] = {
            "coreml_vs_hf": stage_metrics(cm_embed_3d, hf_embed_np),
            "pytorch_vs_hf": stage_metrics(pt_embed_3d, hf_embed_np),
        }

        for ci in range(args.num_chunks):
            hf_chunk = hf_chunk_nps[ci]
            cm_chunk = cm_chunks[ci].reshape(1, 1, -1) if cm_chunks else None
            pt_chunk = pt_chunks[ci].reshape(1, 1, -1) if pt_chunks else None
            stages[f"chunk{ci}"] = {
                "hf_layer_index": chunk_cum[ci],
                "coreml_vs_hf": stage_metrics(cm_chunk, hf_chunk) if cm_chunk is not None else None,
                "pytorch_vs_hf": stage_metrics(pt_chunk, hf_chunk) if pt_chunk is not None else None,
            }

        # ── First token ──
        first_token = {
            "hf": int(hf_first), "coreml": int(cm_first), "pytorch": int(pt_first),
            "cm_match": int(cm_first) == int(hf_first),
            "pt_match": int(pt_first) == int(hf_first),
        }

        cm_parity = token_compare(hf_gen, cm_gen)
        pt_parity = token_compare(hf_gen, pt_gen)

        # ── Print summary ──
        print(f"\n  First token: HF={hf_first}  CoreML={cm_first}  PyTorch={pt_first}")
        print(f"  {'Stage':<12} {'CoreML-FP16↔HF':<22} {'PyTorch(D-proxy)↔HF':<22} {'Improvement'}")
        print(f"  {'-'*70}")
        for stage_name in ["embed"] + [f"chunk{ci}" for ci in range(args.num_chunks)]:
            cm_cos = stages[stage_name]["coreml_vs_hf"]["cosine"] if stages[stage_name]["coreml_vs_hf"] else 0
            pt_cos = stages[stage_name]["pytorch_vs_hf"]["cosine"] if stages[stage_name]["pytorch_vs_hf"] else 0
            gap_closed = ((pt_cos - cm_cos) / (1.0 - cm_cos) * 100) if cm_cos < 1.0 else 0
            print(f"  {stage_name:<12} cos={cm_cos:<18.10f} cos={pt_cos:<18.10f} {gap_closed:+.1f}% gap closed")

        print(f"\n  Decode parity vs HF ({args.max_new_tokens} tokens):")
        print(f"    CoreML-FP16: {cm_parity['match_count']}/{cm_parity['common_len']} "
              f"({cm_parity['match_ratio']*100:.1f}%) first_div={cm_parity['first_divergence_index']}")
        print(f"    PyTorch(D):  {pt_parity['match_count']}/{pt_parity['common_len']} "
              f"({pt_parity['match_ratio']*100:.1f}%) first_div={pt_parity['first_divergence_index']}")

        hf_text = hf_tok.decode(hf_gen, skip_special_tokens=False)[:400]
        cm_text = hf_tok.decode(cm_gen, skip_special_tokens=False)[:400]
        pt_text = hf_tok.decode(pt_gen, skip_special_tokens=False)[:400]
        print(f"\n  HF:      {hf_text[:120]}")
        print(f"  CoreML:  {cm_text[:120]}")
        print(f"  PyTorch: {pt_text[:120]}")

        run = {
            "enable_thinking": enable_thinking,
            "prompt_len": prompt_len,
            "first_token": first_token,
            "decode_parity_coreml_vs_hf": cm_parity,
            "decode_parity_pytorch_vs_hf": pt_parity,
            "stage_metrics": stages,
            "hf_gen": hf_gen[:48], "coreml_gen": cm_gen[:48], "pytorch_gen": pt_gen[:48],
            "hf_text": hf_text, "coreml_text": cm_text, "pytorch_text": pt_text,
        }
        report["runs"].append(run)

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved: {out}")


if __name__ == "__main__":
    main()
