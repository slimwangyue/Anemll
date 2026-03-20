#!/usr/bin/env python3
"""Prompt-level parity for exported Qwen3.5 chunked CoreML models vs HF.

This uses the exported chunked FFN infer models for both prompt ingestion and
decode. It is intentionally single-token for prompt ingestion so it can run
before/without multifunction FFN+Prefill chunk assembly.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CaseResult:
    prompt_name: str
    full_tokens: int
    used_tokens: int
    hf_text: str
    coreml_text: str
    similarity: float


def _build_prompts() -> Dict[str, str]:
    return {
        "short": "What is 2 + 2? Return only the number.",
        "medium": (
            "Explain the difference between stack and heap memory in simple terms, "
            "then give one practical debugging tip."
        ),
    }


def _build_input_ids(tokenizer, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    try:
        out = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(out, "input_ids"):
            ids = out.input_ids
            mask = out.attention_mask if hasattr(out, "attention_mask") and out.attention_mask is not None else torch.ones_like(ids)
            return ids, mask
        if isinstance(out, torch.Tensor):
            ids = out
            return ids, torch.ones_like(ids)
    except Exception:
        pass

    tok = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return tok.input_ids, tok.attention_mask


def _truncate_for_fixed_window(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, context_length: int, max_new_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep = max(1, context_length - max_new_tokens)
    return input_ids[:, -keep:], attention_mask[:, -keep:]


def _make_causal_mask(length: int, start: int = 0) -> np.ndarray:
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return mask


def _parse_logits(output: Dict[str, np.ndarray]) -> torch.Tensor:
    if "output_logits" in output:
        return torch.from_numpy(output["output_logits"])
    if "logits" in output:
        return torch.from_numpy(output["logits"])

    logit_keys = [
        key for key in output.keys()
        if key.startswith("logits") and key[6:].isdigit()
    ]
    if logit_keys:
        logit_keys = sorted(logit_keys, key=lambda k: int(k[6:]))
        return torch.cat([torch.from_numpy(output[k]) for k in logit_keys], dim=-1)

    if "argmax_idx" in output and "argmax_val" in output:
        argmax_idx = output["argmax_idx"].reshape(-1)
        argmax_val = output["argmax_val"].reshape(-1)
        best_chunk = int(np.argmax(argmax_val))
        local_idx = int(argmax_idx[best_chunk])
        # Fallback chunk size inference for the current lm head export.
        chunk_size = 15520
        vocab_idx = best_chunk * chunk_size + max(0, local_idx)
        fake = torch.full((1, 1, (best_chunk + 1) * chunk_size), -1e9, dtype=torch.float32)
        fake[0, 0, vocab_idx] = 0.0
        return fake

    raise KeyError(f"Unrecognized lm_head outputs: {list(output.keys())}")


def _resolve_model_path(path: Path) -> Path:
    if path.exists():
        compiled = path.with_suffix(".mlmodelc")
        if path.suffix == ".mlpackage" and compiled.exists():
            return compiled
        return path
    if path.suffix == ".mlpackage":
        compiled = path.with_suffix(".mlmodelc")
        if compiled.exists():
            return compiled
    raise FileNotFoundError(path)


def _load_model(path: Path, compute_unit):
    resolved = _resolve_model_path(path)
    if resolved.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(resolved), compute_unit)
    return ct.models.MLModel(str(resolved), compute_units=compute_unit)


def _find_chunk_paths(export_dir: Path, prefix: str, count: int) -> List[Path]:
    out = []
    for idx in range(1, count + 1):
        base = export_dir / f"{prefix}_FFN_chunk_{idx:02d}of{count:02d}.mlpackage"
        if base.exists() or base.with_suffix(".mlmodelc").exists():
            out.append(base)
            continue
        out.append(export_dir / f"{prefix}_FFN_lut4_chunk_{idx:02d}of{count:02d}.mlpackage")
    return out


def _hf_generate(model, tokenizer, input_ids, attention_mask, max_new_tokens: int) -> str:
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _run_coreml_chunked(
    embed_model,
    ffn_models,
    lm_head_model,
    tokenizer,
    input_ids: torch.Tensor,
    context_length: int,
    max_new_tokens: int,
) -> str:
    state = ffn_models[0].make_state()
    causal_mask = _make_causal_mask(context_length, 0)

    def run_one_token(token_id: int, pos: int) -> np.ndarray:
        hidden = embed_model.predict({"input_ids": np.array([[token_id]], dtype=np.int32)})["hidden_states"]
        position_ids = np.array([pos], dtype=np.int32)
        single_mask = causal_mask[:, :, pos:pos + 1, :]
        inputs = {
            "hidden_states": hidden,
            "position_ids": position_ids,
            "causal_mask": single_mask,
            "current_pos": position_ids,
        }
        if len(ffn_models) == 4:
            out = ffn_models[0].predict(inputs, state)
            inputs["hidden_states"] = out["output_hidden_states"]
            out = ffn_models[1].predict(inputs, state)
            inputs["hidden_states"] = out["output_hidden_states"]
            out = ffn_models[2].predict(inputs, state)
            inputs["hidden_states"] = out["output_hidden_states"]
            out = ffn_models[3].predict(inputs, state)
            inputs["hidden_states"] = out["output_hidden_states"]
        else:
            for model in ffn_models:
                out = model.predict(inputs, state)
                inputs["hidden_states"] = out["output_hidden_states"]
        return inputs["hidden_states"]

    prompt_len = int(input_ids.shape[1])
    last_hidden = None
    for pos in range(prompt_len):
        last_hidden = run_one_token(int(input_ids[0, pos].item()), pos)

    assert last_hidden is not None
    generated: List[int] = []
    next_logits = _parse_logits(lm_head_model.predict({"hidden_states": last_hidden}))
    next_id = int(torch.argmax(next_logits[0, -1, :]).item())

    for step in range(max_new_tokens):
        generated.append(next_id)
        if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
            break
        pos = prompt_len + step
        last_hidden = run_one_token(next_id, pos)
        next_logits = _parse_logits(lm_head_model.predict({"hidden_states": last_hidden}))
        next_id = int(torch.argmax(next_logits[0, -1, :]).item())

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _write_report(path: Path, model_path: str, export_dir: str, rows: List[CaseResult], compute_unit: str) -> None:
    lines = [
        "# Qwen3.5 Exported Chunk Prompt Parity",
        "",
        f"- model_path: `{model_path}`",
        f"- export_dir: `{export_dir}`",
        f"- compute_unit: `{compute_unit}`",
        "- runtime path: exported `FFN` chunk `infer` only for prompt ingestion and decode",
        "",
        "| prompt | full_tokens | used_tokens | similarity |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row.prompt_name} | {row.full_tokens} | {row.used_tokens} | {row.similarity:.4f} |")
    lines.append("")
    for row in rows:
        lines.extend([
            f"## {row.prompt_name}",
            "",
            f"- full_tokens: `{row.full_tokens}`",
            f"- used_tokens: `{row.used_tokens}`",
            f"- similarity: `{row.similarity:.4f}`",
            "- HF answer:",
            "```text",
            row.hf_text or "<empty>",
            "```",
            "- CoreML answer:",
            "```text",
            row.coreml_text or "<empty>",
            "```",
            "",
        ])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--prefix", default="qwen35")
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--compute-unit", choices=["cpu_only", "cpu_and_gpu", "all"], default="cpu_only")
    parser.add_argument("--report-path", default="tests/dev/qwen35_exported_chunk_prompt_parity_report.md")
    args = parser.parse_args()

    compute_unit_map = {
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
    }
    compute_unit = compute_unit_map[args.compute_unit]

    export_dir = Path(args.export_dir)
    embed_model = _load_model(export_dir / f"{args.prefix}_embeddings.mlpackage", compute_unit)
    lm_head_model = _load_model(export_dir / f"{args.prefix}_lm_head_lut6.mlpackage", compute_unit)
    ffn_models = [_load_model(path, compute_unit) for path in _find_chunk_paths(export_dir, args.prefix, args.num_chunks)]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16).eval()

    rows: List[CaseResult] = []
    for prompt_name, prompt in _build_prompts().items():
        ids, attn = _build_input_ids(tokenizer, prompt)
        full_len = int(ids.shape[1])
        ids_used, attn_used = _truncate_for_fixed_window(ids, attn, args.context_length, args.max_new_tokens)
        used_len = int(ids_used.shape[1])
        hf_text = _hf_generate(hf_model, tokenizer, ids_used, attn_used, args.max_new_tokens)
        coreml_text = _run_coreml_chunked(
            embed_model,
            ffn_models,
            lm_head_model,
            tokenizer,
            ids_used.to(torch.long),
            args.context_length,
            args.max_new_tokens,
        )
        rows.append(
            CaseResult(
                prompt_name=prompt_name,
                full_tokens=full_len,
                used_tokens=used_len,
                hf_text=hf_text,
                coreml_text=coreml_text,
                similarity=SequenceMatcher(a=hf_text, b=coreml_text).ratio(),
            )
        )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path, args.model_path, args.export_dir, rows, args.compute_unit)
    print(f"wrote report: {report_path}")
    for row in rows:
        print(
            f"{row.prompt_name:6s} full={row.full_tokens:4d} "
            f"used={row.used_tokens:4d} sim={row.similarity:.4f}"
        )


if __name__ == "__main__":
    main()
