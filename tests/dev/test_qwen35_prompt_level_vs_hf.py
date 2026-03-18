#!/usr/bin/env python3
"""Prompt-level validation for Qwen3.5 ANEMLL vs HF (normal + think mode).

Runs short/medium/long prompts with fixed-size context behavior and writes a
comparison report to markdown.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM


@dataclass
class CaseResult:
    mode: str
    prompt_name: str
    input_tokens_full: int
    input_tokens_used: int
    hf_text: str
    ours_text: str
    similarity: float


def _build_prompts() -> Dict[str, str]:
    filler = (
        "Project note filler: latency target, memory budget, throughput estimate, and deployment checklist. "
    )
    # Force > fixed size by repetition while keeping the real question in the tail.
    long_prompt = (
        (filler * 180)
        + "Final question: what number comes after 41? Answer with one number only."
    )
    return {
        "short": "What is 2 + 2? Return only the number.",
        "medium": (
            "Explain the difference between stack and heap memory in simple terms, "
            "then give one practical debugging tip."
        ),
        "long": long_prompt,
    }


def _build_input_ids(tokenizer, prompt: str, think_mode: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    try:
        out = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=think_mode,
        )
        if hasattr(out, "input_ids"):
            ids = out.input_ids
            mask = out.attention_mask if hasattr(out, "attention_mask") and out.attention_mask is not None else torch.ones_like(ids)
            return ids, mask
        if isinstance(out, torch.Tensor):
            ids = out
            return ids, torch.ones_like(ids)
        ids = torch.tensor(out, dtype=torch.long).unsqueeze(0)
        return ids, torch.ones_like(ids)
    except Exception:
        # Fallback when template kwargs are unsupported.
        prefix = "Think step by step before final answer.\n" if think_mode else ""
        tok = tokenizer(prefix + prompt, return_tensors="pt", add_special_tokens=True)
        return tok.input_ids, tok.attention_mask


def _truncate_for_fixed_window(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, state_length: int, max_new_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep = max(1, state_length - max_new_tokens)
    return input_ids[:, -keep:], attention_mask[:, -keep:]


def _hf_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> str:
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][input_ids.shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _ours_generate(model: Qwen35ForCausalLM, tokenizer, input_ids: torch.Tensor, max_new_tokens: int) -> str:
    input_ids = input_ids.to(torch.long)
    prompt_len = input_ids.shape[1]
    pos = torch.arange(prompt_len, dtype=torch.long)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            position_ids=pos,
            causal_mask=None,
            current_pos=0,
            IN_PREFILL=True,
        )
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        gen: List[int] = []
        for step in range(max_new_tokens):
            token = int(next_id.item())
            gen.append(token)
            if tokenizer.eos_token_id is not None and token == int(tokenizer.eos_token_id):
                break
            cache_pos = prompt_len + step
            decode_logits = model(
                input_ids=next_id.to(torch.long),
                position_ids=torch.tensor([cache_pos], dtype=torch.long),
                causal_mask=None,
                current_pos=cache_pos,
                IN_PREFILL=False,
            )
            next_id = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()


def _write_report(path: str, model_path: str, max_new_tokens: int, state_length: int, rows: List[CaseResult]) -> None:
    lines: List[str] = []
    lines.append("# Qwen3.5 Prompt-Level Validation (ANEMLL vs HF)")
    lines.append("")
    lines.append(f"- model_path: `{model_path}`")
    lines.append(f"- fixed_state_length: `{state_length}`")
    lines.append(f"- max_new_tokens: `{max_new_tokens}`")
    lines.append("- input policy: both HF and ANEMLL use the same fixed-window-truncated prompt")
    lines.append("")
    lines.append("| mode | prompt | full_tokens | used_tokens | similarity |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.mode} | {r.prompt_name} | {r.input_tokens_full} | {r.input_tokens_used} | {r.similarity:.4f} |"
        )
    lines.append("")
    for r in rows:
        lines.append(f"## {r.mode} / {r.prompt_name}")
        lines.append("")
        lines.append(f"- full_tokens: `{r.input_tokens_full}`")
        lines.append(f"- used_tokens: `{r.input_tokens_used}`")
        lines.append(f"- similarity: `{r.similarity:.4f}`")
        lines.append("- HF answer:")
        lines.append("```text")
        lines.append(r.hf_text or "<empty>")
        lines.append("```")
        lines.append("- ANEMLL answer:")
        lines.append("```text")
        lines.append(r.ours_text or "<empty>")
        lines.append("```")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--report-path",
        default="tests/dev/qwen35_prompt_validation_report.md",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    prompts = _build_prompts()
    modes = [("normal", False), ("think", True)]

    # HF pass first, then free memory before loading ANEMLL model.
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
    ).eval()
    hf_outputs: Dict[str, str] = {}
    token_meta: Dict[str, tuple[int, int]] = {}
    for mode_name, think_mode in modes:
        for prompt_name, prompt in prompts.items():
            key = f"{mode_name}:{prompt_name}"
            ids, attn_mask = _build_input_ids(tokenizer, prompt, think_mode=think_mode)
            full_len = int(ids.shape[1])
            ids_used, mask_used = _truncate_for_fixed_window(
                ids, attn_mask, cfg.state_length, args.max_new_tokens
            )
            used_len = int(ids_used.shape[1])
            hf_outputs[key] = _hf_generate(
                hf_model, tokenizer, ids_used, mask_used, args.max_new_tokens
            )
            token_meta[key] = (full_len, used_len)

    del hf_model
    gc.collect()

    anemll_model = Qwen35ForCausalLM(cfg).half().eval()
    if not anemll_model.load_pretrained_weights(args.model_path):
        raise RuntimeError("ANEMLL loader failed.")

    rows: List[CaseResult] = []
    for mode_name, think_mode in modes:
        for prompt_name, prompt in prompts.items():
            key = f"{mode_name}:{prompt_name}"
            ids, attn_mask = _build_input_ids(tokenizer, prompt, think_mode=think_mode)
            ids_used, _ = _truncate_for_fixed_window(ids, attn_mask, cfg.state_length, args.max_new_tokens)
            ours_text = _ours_generate(anemll_model, tokenizer, ids_used, args.max_new_tokens)
            hf_text = hf_outputs[key]
            rows.append(
                CaseResult(
                    mode=mode_name,
                    prompt_name=prompt_name,
                    input_tokens_full=token_meta[key][0],
                    input_tokens_used=token_meta[key][1],
                    hf_text=hf_text,
                    ours_text=ours_text,
                    similarity=_similarity(hf_text, ours_text),
                )
            )

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    _write_report(args.report_path, args.model_path, args.max_new_tokens, cfg.state_length, rows)
    print(f"wrote report: {args.report_path}")
    for r in rows:
        print(
            f"{r.mode:6s} {r.prompt_name:6s} full={r.input_tokens_full:4d} "
            f"used={r.input_tokens_used:4d} sim={r.similarity:.4f}"
        )


if __name__ == "__main__":
    main()
