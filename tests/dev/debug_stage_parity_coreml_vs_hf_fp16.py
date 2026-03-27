#!/usr/bin/env python3
"""Stage parity validation: stable 4-chunk ANE CoreML vs HF fp16.

Outputs:
- Embedding hidden parity (final prompt token step)
- Per-chunk boundary hidden parity (mapped by inferred layers/chunk)
- First next-token parity
- Short greedy decode parity (first divergence index)
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Guard against profile module shadowing when transformers imports torch._dynamo.
sys.modules["profile"] = importlib.import_module("profile")

NEG_INF = np.float16(-65504.0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float32).reshape(-1)
    y = b.astype(np.float32).reshape(-1)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    d = np.abs(aa - bb)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "cosine": cosine(aa, bb),
    }


def _find_model(base_dir: str, name: str) -> str:
    for ext in (".mlmodelc", ".mlpackage"):
        p = str(Path(base_dir) / f"{name}{ext}")
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"Missing model: {name} in {base_dir}")


def _load_model(path: str, compute_unit, function_name: str | None = None):
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, compute_unit)
    kwargs: dict[str, Any] = {"compute_units": compute_unit}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def _chunk_layer_count(model) -> int:
    spec = model.get_spec()
    # Prefer spec input shape from selected function; fallback to generic input.
    inputs = spec.description.input
    for inp in inputs:
        if inp.name == "linear_conv_state":
            shp = tuple(inp.type.multiArrayType.shape)
            if len(shp) >= 1:
                return int(shp[0])
    # conservative fallback
    return 0


@dataclass
class CoreMLRuntime:
    embed: Any
    lmhead: Any
    ffns: list[Any]
    tokenizer: Any
    ctx: int
    num_chunks: int
    stop_ids: set[int]
    lmhead_mode: str
    logits_key: str | None
    logits_keys: list[str] | None

    states: list[Any]
    lin_convs: list[np.ndarray]
    lin_recs: list[np.ndarray]

    tok_buf: np.ndarray
    mask_buf: np.ndarray
    pos_buf: np.ndarray


def build_coreml_runtime(model_dir: str, tokenizer_path: str, ctx: int, num_chunks: int) -> CoreMLRuntime:
    cu = ct.ComputeUnit.CPU_AND_NE
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    embed = _load_model(_find_model(model_dir, "embeddings"), cu)
    lmhead = _load_model(_find_model(model_dir, "lm_head"), cu)

    ffns: list[Any] = []
    combined = Path(model_dir) / "combined_LUT4_dedup"
    use_combined = combined.is_dir()
    for ci in range(num_chunks):
        if use_combined:
            path = _find_model(str(combined), f"chunk{ci}")
            ffns.append(_load_model(path, cu, function_name="infer"))
        else:
            path = _find_model(model_dir, f"ffn_LUT4_chunk{ci}")
            ffns.append(_load_model(path, cu))

    # lm_head mode detect
    spec = lmhead.get_spec()
    out_names = [o.name for o in spec.description.output]
    if "logits" in out_names or "output_logits" in out_names:
        lm_mode = "logits"
        split = sorted([n for n in out_names if n.startswith("logits") and n[6:].isdigit()])
        if split:
            logits_keys = split
            logits_key = None
        else:
            logits_keys = None
            logits_key = "output_logits" if "output_logits" in out_names else "logits"
    else:
        lm_mode = "argmax"
        logits_key = None
        logits_keys = None

    # infer state shapes from first chunk
    spec0 = ffns[0].get_spec()
    inmap: dict[str, tuple[int, ...]] = {}
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

    tok_buf = np.zeros((1, 1), dtype=np.int32)
    mask_buf = np.full((1, 1, 1, ctx), NEG_INF, dtype=np.float16)
    pos_buf = np.zeros((1,), dtype=np.int32)

    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for s in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(int(tid))

    return CoreMLRuntime(
        embed=embed,
        lmhead=lmhead,
        ffns=ffns,
        tokenizer=tokenizer,
        ctx=ctx,
        num_chunks=num_chunks,
        stop_ids=stop_ids,
        lmhead_mode=lm_mode,
        logits_key=logits_key,
        logits_keys=logits_keys,
        states=states,
        lin_convs=lin_convs,
        lin_recs=lin_recs,
        tok_buf=tok_buf,
        mask_buf=mask_buf,
        pos_buf=pos_buf,
    )


def make_prompt_ids(tokenizer, prompt: str, enable_thinking: bool) -> list[int]:
    msgs = [{"role": "user", "content": prompt}]
    out = tokenizer.apply_chat_template(
        msgs,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if hasattr(out, "input_ids"):
        ids = out.input_ids[0].tolist()
    else:
        ids = out[0].tolist()
    return [int(x) for x in ids]


def coreml_extract_logits(rt: CoreMLRuntime, lm_out: dict[str, np.ndarray]) -> np.ndarray | None:
    if rt.lmhead_mode != "logits":
        return None
    if rt.logits_keys:
        parts = [lm_out[k].reshape(-1).astype(np.float32) for k in rt.logits_keys]
        return np.concatenate(parts)
    assert rt.logits_key is not None
    return lm_out[rt.logits_key].reshape(-1).astype(np.float32)


def coreml_step(rt: CoreMLRuntime, token_id: int, pos: int, capture_stages: bool = False):
    rt.tok_buf[0, 0] = np.int32(token_id)
    embed_out = rt.embed.predict({"input_ids": rt.tok_buf})
    hidden = list(embed_out.values())[0]

    rt.mask_buf[:, :, :, :] = NEG_INF
    rt.mask_buf[:, :, :, : pos + 1] = 0
    rt.pos_buf[0] = np.int32(pos)

    chunk_hiddens: list[np.ndarray] = []

    for ci in range(rt.num_chunks):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": rt.pos_buf,
            "causal_mask": rt.mask_buf,
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
    if logits is not None:
        next_id = int(np.argmax(logits))
    else:
        next_id = int(lm_out["argmax_idx"].reshape(-1)[0])

    if capture_stages:
        return next_id, logits, embed_out["hidden_states"].copy(), chunk_hiddens
    return next_id, logits, None, None


def run_coreml_prompt(rt: CoreMLRuntime, prompt_ids: list[int], capture_final_stages: bool = True):
    final_embed = None
    final_chunks = None
    next_id = None
    for pos, tid in enumerate(prompt_ids):
        cap = capture_final_stages and (pos == len(prompt_ids) - 1)
        next_id, logits, emb, chunks = coreml_step(rt, tid, pos, capture_stages=cap)
        if cap:
            final_embed = emb
            final_chunks = chunks
    return next_id, final_embed, final_chunks


def run_hf_prompt(model, tokenizer, prompt_ids: list[int]):
    device = model.device
    past = None
    final_hidden_states = None
    next_id = None

    with torch.no_grad():
        for i, tid in enumerate(prompt_ids):
            x = torch.tensor([[tid]], dtype=torch.long, device=device)
            out = model(
                input_ids=x,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = out.past_key_values
            next_id = int(torch.argmax(out.logits[0, -1, :]).item())
            if i == len(prompt_ids) - 1:
                final_hidden_states = out.hidden_states

    assert final_hidden_states is not None
    return next_id, final_hidden_states


def decode_hf(model, start_id: int, max_new_tokens: int, stop_ids: set[int]) -> list[int]:
    device = model.device
    gen = [int(start_id)]
    past = None

    # Bootstrap with first token to create cache.
    with torch.no_grad():
        x = torch.tensor([[start_id]], dtype=torch.long, device=device)
        out = model(input_ids=x, use_cache=True)
        past = out.past_key_values

        for _ in range(max_new_tokens - 1):
            if gen[-1] in stop_ids:
                break
            x = torch.tensor([[gen[-1]]], dtype=torch.long, device=device)
            out = model(input_ids=x, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(torch.argmax(out.logits[0, -1, :]).item())
            gen.append(nxt)

    return gen


def decode_coreml(rt: CoreMLRuntime, start_id: int, start_pos: int, max_new_tokens: int) -> tuple[list[int], str | None]:
    gen = [int(start_id)]
    pos = int(start_pos)
    for _ in range(max_new_tokens - 1):
        if gen[-1] in rt.stop_ids:
            break
        if pos >= rt.ctx - 1:
            break
        try:
            nxt, _logits, _e, _c = coreml_step(rt, gen[-1], pos, capture_stages=False)
        except Exception as e:
            return gen, str(e)
        pos += 1
        gen.append(int(nxt))
    return gen, None


def compare_token_ids(a: list[int], b: list[int]) -> dict[str, Any]:
    n = min(len(a), len(b))
    m = sum(1 for i in range(n) if a[i] == b[i])
    first_div = None
    for i in range(n):
        if a[i] != b[i]:
            first_div = i
            break
    return {
        "a_len": len(a),
        "b_len": len(b),
        "common_len": n,
        "match_count": m,
        "match_ratio": (m / n) if n else 0.0,
        "first_divergence_index": first_div,
    }


def choose_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "mps":
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coreml-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--prompt", default="教我做红烧肉")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--hf-device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--out", default="tests/dev/stage_parity_coreml_vs_hf_fp16_report.json")
    args = ap.parse_args()

    print("[load] HF fp16 model...")
    hf_tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)
    hf = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    dev = choose_device(args.hf_device)
    hf = hf.to(dev).eval()
    print(f"[load] HF device={dev}")

    print("[load] CoreML runtime...")
    rt = build_coreml_runtime(args.coreml_dir, args.tokenizer, args.ctx, args.num_chunks)

    # Infer per-chunk layer counts from model inputs.
    chunk_layer_counts = [_chunk_layer_count(m) for m in rt.ffns]

    hf_num_layers = int(getattr(hf.config, "num_hidden_layers", -1))
    if sum(chunk_layer_counts) <= 0 and hf_num_layers > 0:
        # Fallback: even split across chunks when CoreML spec does not expose shapes.
        base = hf_num_layers // args.num_chunks
        rem = hf_num_layers % args.num_chunks
        chunk_layer_counts = [base + (1 if i < rem else 0) for i in range(args.num_chunks)]

    report: dict[str, Any] = {
        "prompt": args.prompt,
        "ctx": args.ctx,
        "num_chunks": args.num_chunks,
        "chunk_layer_counts": chunk_layer_counts,
        "hf_num_layers": hf_num_layers,
        "runs": [],
    }

    for enable_thinking in (False, True):
        print(f"\n=== enable_thinking={enable_thinking} ===")

        # Reset runtime state for each run.
        rt.states = [m.make_state() for m in rt.ffns]
        rt.lin_convs = [np.zeros_like(rt.lin_convs[0]) for _ in range(rt.num_chunks)]
        rt.lin_recs = [np.zeros_like(rt.lin_recs[0]) for _ in range(rt.num_chunks)]

        prompt_ids = make_prompt_ids(rt.tokenizer, args.prompt, enable_thinking)

        coreml_first, cm_embed_last, cm_chunk_last = run_coreml_prompt(rt, prompt_ids, capture_final_stages=True)
        hf_first, hf_hidden_states = run_hf_prompt(hf, hf_tok, prompt_ids)

        # Stage metrics at final prompt token step.
        stage_metrics: dict[str, Any] = {}

        hf_embed_last = hf_hidden_states[0][0, -1:, :].detach().float().cpu().numpy().reshape(1, 1, -1)
        stage_metrics["embedding_last_step"] = metrics(cm_embed_last, hf_embed_last)

        # Map chunk boundaries by cumulative layer count.
        hf_layers = hf_num_layers
        csum = 0
        for ci, cc in enumerate(chunk_layer_counts):
            csum += int(cc)
            # hidden_states index: 0 is embeddings, layer outputs are 1..num_layers
            idx = min(max(csum, 1), max(hf_layers, 1))
            hf_chunk = hf_hidden_states[idx][0, -1:, :].detach().float().cpu().numpy().reshape(1, 1, -1)
            cm_chunk = cm_chunk_last[ci]
            stage_metrics[f"chunk{ci}_boundary"] = metrics(cm_chunk, hf_chunk)
            stage_metrics[f"chunk{ci}_hf_layer_index"] = idx

        # First token parity.
        first_parity = {
            "coreml_first": int(coreml_first),
            "hf_first": int(hf_first),
            "match": int(coreml_first) == int(hf_first),
        }

        # Short decode parity from first token onward.
        cm_gen, cm_decode_error = decode_coreml(
            rt, int(coreml_first), start_pos=len(prompt_ids), max_new_tokens=args.max_new_tokens)
        hf_gen = decode_hf(hf, int(hf_first), max_new_tokens=args.max_new_tokens, stop_ids=rt.stop_ids)
        decode_parity = compare_token_ids(hf_gen, cm_gen)

        run_item = {
            "enable_thinking": enable_thinking,
            "prompt_len": len(prompt_ids),
            "first_token": first_parity,
            "decode_parity": decode_parity,
            "coreml_decode_error": cm_decode_error,
            "stage_metrics": stage_metrics,
            "hf_first_32": hf_gen[:32],
            "coreml_first_32": cm_gen[:32],
            "hf_text_preview": hf_tok.decode(hf_gen, skip_special_tokens=False)[:800],
            "coreml_text_preview": hf_tok.decode(cm_gen, skip_special_tokens=False)[:800],
        }
        report["runs"].append(run_item)

        print(json.dumps({
            "prompt_len": run_item["prompt_len"],
            "first_token": run_item["first_token"],
            "decode_parity": run_item["decode_parity"],
            "embedding_cos": run_item["stage_metrics"]["embedding_last_step"]["cosine"],
        }, ensure_ascii=False, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved report to: {out}")


if __name__ == "__main__":
    main()
