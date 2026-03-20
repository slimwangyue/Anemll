#!/usr/bin/env python3
"""Model-level parity for Qwen3.5 ANEMLL vs HF focused on linear-attention behavior.

This validates the whole model rather than a single layer by comparing:
1. Prefill logits over the full fixed-window prompt.
2. Decode-step logits for greedy generation.
3. Generated text similarity.

Two ANEMLL linear modes are supported:
- `monolithic`: regular model path.
- `split4_ref`: a PyTorch reference wrapper that preserves the 4-stage linear path
  (`Proj | Conv | Layout | (Core+Norm)`) while keeping the same math.
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

from anemll.models.qwen3_5_model import MODEL_DTYPE, Qwen35Config, Qwen35ForCausalLM


def _metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a32 = a.float().reshape(-1)
    b32 = b.float().reshape(-1)
    diff = (a32 - b32).abs()
    cos = torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0)).item()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a32 - b32) ** 2)).item()),
        "cosine": float(cos),
    }


def _kl_div(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.float()
    b32 = b.float()
    p = torch.softmax(a32, dim=-1)
    q = torch.softmax(b32, dim=-1)
    return float((p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(dim=-1).mean().item())


def _top1_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = torch.argmax(a, dim=-1)
    bb = torch.argmax(b, dim=-1)
    return float((aa == bb).float().mean().item())


def _format_metrics(name: str, m: Dict[str, float], extra: Dict[str, float] | None = None) -> str:
    parts = [
        f"{name:20s}",
        f"max_abs={m['max_abs']:.6e}",
        f"mean_abs={m['mean_abs']:.6e}",
        f"rmse={m['rmse']:.6e}",
        f"cosine={m['cosine']:.8f}",
    ]
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v:.6e}" if abs(v) < 1000 else f"{k}={v:.6f}")
    return " ".join(parts)


@dataclass
class CaseResult:
    mode: str
    prompt_name: str
    input_tokens_full: int
    input_tokens_used: int
    prefill_metrics: Dict[str, float]
    prefill_kl: float
    prefill_top1: float
    decode_mean_metrics: Dict[str, float]
    decode_mean_kl: float
    decode_top1: float
    hf_text: str
    ours_text: str
    similarity: float


def _build_prompts() -> Dict[str, str]:
    filler = (
        "Project note filler: latency target, memory budget, throughput estimate, and deployment checklist. "
    )
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
        prefix = "Think step by step before final answer.\n" if think_mode else ""
        tok = tokenizer(prefix + prompt, return_tensors="pt", add_special_tokens=True)
        return tok.input_ids, tok.attention_mask


def _truncate_for_fixed_window(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, state_length: int, max_new_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep = max(1, state_length - max_new_tokens)
    return input_ids[:, -keep:], attention_mask[:, -keep:]


class Split4LinearAttentionReference(torch.nn.Module):
    """PyTorch reference preserving the 4-stage linear path."""

    def __init__(self, attn: torch.nn.Module):
        super().__init__()
        self.attn = attn

    def _run(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        has_previous_state: bool,
        expected_batch_size: int | None = None,
        expected_seq_len: int | None = None,
        force_recurrent: bool = False,
        force_fp16_math: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = int(expected_batch_size) if expected_batch_size is not None else hidden_states.shape[0]
        seq_len = int(expected_seq_len) if expected_seq_len is not None else hidden_states.shape[1]
        hidden_cf = self.attn._to_channels_first_4d(hidden_states)

        mixed_qkv_pre = self.attn._conv2d_proj_cf(self.attn.in_proj_qkv, hidden_cf)
        z_cf = self.attn._conv2d_proj_cf(self.attn.in_proj_z, hidden_cf)
        b_cf = self.attn._conv2d_proj_cf(self.attn.in_proj_b, hidden_cf)
        a_cf = self.attn._conv2d_proj_cf(self.attn.in_proj_a, hidden_cf)

        z = self.attn._from_channels_first_4d(z_cf).reshape(bsz, seq_len, self.attn.num_v_heads, self.attn.head_v_dim)
        b = self.attn._from_channels_first_4d(b_cf)
        a = self.attn._from_channels_first_4d(a_cf)

        if conv_state is None:
            conv_state = torch.zeros(
                (bsz, self.attn.conv_dim, self.attn.linear_conv_kernel_dim),
                dtype=MODEL_DTYPE,
                device=hidden_states.device,
            )
        conv_out_cf, next_conv_state = self.attn._causal_conv_update_cf(
            mixed_qkv_pre, conv_state, expected_seq_len=expected_seq_len
        )

        query_cf, key_cf, value_cf = torch.split(
            conv_out_cf, [self.attn.key_dim, self.attn.key_dim, self.attn.value_dim], dim=1
        )
        query = self.attn._from_channels_first_4d(query_cf).reshape(bsz, seq_len, self.attn.num_k_heads, self.attn.head_k_dim)
        key = self.attn._from_channels_first_4d(key_cf).reshape(bsz, seq_len, self.attn.num_k_heads, self.attn.head_k_dim)
        value = self.attn._from_channels_first_4d(value_cf).reshape(bsz, seq_len, self.attn.num_v_heads, self.attn.head_v_dim)

        beta = b.sigmoid()
        if force_fp16_math:
            g = -self.attn.A_log.to(MODEL_DTYPE).exp() * torch.nn.functional.softplus(a.to(MODEL_DTYPE) + self.attn.dt_bias)
        else:
            g = -self.attn.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.attn.dt_bias)
        if self.attn.num_v_heads // self.attn.num_k_heads > 1:
            rep = self.attn.num_v_heads // self.attn.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        if recurrent_state is None:
            recurrent_state = torch.zeros(
                (bsz, self.attn.num_v_heads, self.attn.head_k_dim, self.attn.head_v_dim),
                dtype=torch.float32,
                device=hidden_states.device,
            )

        if force_recurrent or (has_previous_state and seq_len == 1):
            core, next_recurrent_state = self.attn._recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                recurrent_state=recurrent_state,
                output_final_state=True,
                expected_batch_size=expected_batch_size,
                expected_num_heads=self.attn.num_v_heads,
                expected_seq_len=expected_seq_len,
                expected_k_dim=self.attn.head_k_dim,
                expected_v_dim=self.attn.head_v_dim,
                math_dtype=MODEL_DTYPE if force_fp16_math else torch.float32,
            )
        else:
            core, next_recurrent_state = self.attn._chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if has_previous_state else None,
                output_final_state=True,
                expected_batch_size=expected_batch_size,
                expected_num_heads=self.attn.num_v_heads,
                expected_seq_len=expected_seq_len,
                expected_k_dim=self.attn.head_k_dim,
                expected_v_dim=self.attn.head_v_dim,
                math_dtype=MODEL_DTYPE if force_fp16_math else torch.float32,
            )

        core = self.attn.norm(core.reshape(-1, self.attn.head_v_dim), z.reshape(-1, self.attn.head_v_dim)).reshape(
            bsz, seq_len, self.attn.value_dim
        )
        out = self.attn._from_channels_first_4d(
            self.attn._conv2d_proj_cf(self.attn.out_proj, self.attn._to_channels_first_4d(core))
        )
        return out, next_conv_state, next_recurrent_state

    def forward(self, hidden_states: torch.Tensor, causal_mask, position_ids):
        out, _, _ = self._run(hidden_states, None, None, False)
        return out

    def forward_prefill(self, hidden_states, conv_state, recurrent_state, has_previous_state=False, causal_mask=None):
        return self._run(hidden_states, conv_state, recurrent_state, has_previous_state)

    def forward_regular(self, hidden_states, conv_state, recurrent_state, has_previous_state=True, causal_mask=None):
        return self._run(hidden_states, conv_state, recurrent_state, has_previous_state)


def _apply_linear_mode(model: Qwen35ForCausalLM, linear_mode: str) -> None:
    if linear_mode == "monolithic":
        return
    if linear_mode != "split4_ref":
        raise ValueError(f"Unsupported linear_mode: {linear_mode}")
    for layer in model.model.layers:
        if getattr(layer, "layer_type", None) == "linear_attention":
            layer.self_attn = Split4LinearAttentionReference(layer.self_attn)


def _hf_prefill_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits.detach().cpu()


def _ours_prefill_logits(model: Qwen35ForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    pos = torch.arange(input_ids.shape[1], dtype=torch.long)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids.to(torch.long),
            position_ids=pos,
            causal_mask=None,
            current_pos=torch.tensor([0], dtype=torch.long),
            IN_PREFILL=True,
        )
    return logits.detach().cpu()


def _hf_generate_with_logits(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[str, List[torch.Tensor]]:
    logits_steps: List[torch.Tensor] = []
    generated = input_ids.clone()
    mask = attention_mask.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=generated, attention_mask=mask, use_cache=False)
            step_logits = out.logits[:, -1, :].detach().cpu()
            logits_steps.append(step_logits)
            next_id = torch.argmax(step_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id.to(generated.device)], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_id, device=mask.device)], dim=1)
            if tokenizer.eos_token_id is not None and int(next_id.item()) == int(tokenizer.eos_token_id):
                break
    gen_ids = generated[0][input_ids.shape[1] :].detach().cpu().tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip(), logits_steps


def _ours_generate_with_logits(
    model: Qwen35ForCausalLM,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[str, List[torch.Tensor]]:
    input_ids = input_ids.to(torch.long)
    prompt_len = input_ids.shape[1]
    pos = torch.arange(prompt_len, dtype=torch.long)
    logits_steps: List[torch.Tensor] = []
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            position_ids=pos,
            causal_mask=None,
            current_pos=0,
            IN_PREFILL=True,
        )
        step_logits = logits[:, -1, :].detach().cpu()
        logits_steps.append(step_logits)
        next_id = torch.argmax(step_logits, dim=-1, keepdim=True)
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
                current_pos=torch.tensor([cache_pos], dtype=torch.long),
                IN_PREFILL=False,
            )
            step_logits = decode_logits[:, -1, :].detach().cpu()
            logits_steps.append(step_logits)
            next_id = torch.argmax(step_logits, dim=-1, keepdim=True)
    return tokenizer.decode(gen, skip_special_tokens=True).strip(), logits_steps


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()


def _mean_metric_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}


def _write_report(path: str, model_path: str, max_new_tokens: int, state_length: int, linear_mode: str, rows: List[CaseResult]) -> None:
    lines: List[str] = []
    lines.append("# Qwen3.5 Linear Model-Level Validation (ANEMLL vs HF)")
    lines.append("")
    lines.append(f"- model_path: `{model_path}`")
    lines.append(f"- fixed_state_length: `{state_length}`")
    lines.append(f"- max_new_tokens: `{max_new_tokens}`")
    lines.append(f"- linear_mode: `{linear_mode}`")
    lines.append("- input policy: both HF and ANEMLL use the same fixed-window-truncated prompt")
    lines.append("")
    lines.append("| mode | prompt | full_tokens | used_tokens | prefill_cos | decode_cos | top1 | text_sim |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.mode} | {r.prompt_name} | {r.input_tokens_full} | {r.input_tokens_used} | "
            f"{r.prefill_metrics['cosine']:.4f} | {r.decode_mean_metrics['cosine']:.4f} | {r.decode_top1:.4f} | {r.similarity:.4f} |"
        )
    lines.append("")
    for r in rows:
        lines.append(f"## {r.mode} / {r.prompt_name}")
        lines.append("")
        lines.append(f"- full_tokens: `{r.input_tokens_full}`")
        lines.append(f"- used_tokens: `{r.input_tokens_used}`")
        lines.append(f"- prefill: `{_format_metrics('prefill', r.prefill_metrics, {'kl': r.prefill_kl, 'top1': r.prefill_top1})}`")
        lines.append(f"- decode_mean: `{_format_metrics('decode_mean', r.decode_mean_metrics, {'kl': r.decode_mean_kl, 'top1': r.decode_top1})}`")
        lines.append(f"- text_similarity: `{r.similarity:.4f}`")
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
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--linear-mode", choices=["monolithic", "split4_ref"], default="monolithic")
    parser.add_argument("--prompt-names", type=str, default="short,medium,long")
    parser.add_argument("--report-path", default="tests/dev/qwen35_linear_model_level_validation_report.md")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    prompts = _build_prompts()
    prompt_names = [x.strip() for x in args.prompt_names.split(",") if x.strip()]
    modes = [("normal", False), ("think", True)]

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
    ).eval()

    hf_prefill: Dict[str, torch.Tensor] = {}
    hf_decode_text: Dict[str, str] = {}
    hf_decode_logits: Dict[str, List[torch.Tensor]] = {}
    token_meta: Dict[str, Tuple[int, int]] = {}
    truncated_ids: Dict[str, torch.Tensor] = {}
    truncated_mask: Dict[str, torch.Tensor] = {}
    for mode_name, think_mode in modes:
        for prompt_name in prompt_names:
            prompt = prompts[prompt_name]
            key = f"{mode_name}:{prompt_name}"
            ids, attn_mask = _build_input_ids(tokenizer, prompt, think_mode=think_mode)
            full_len = int(ids.shape[1])
            ids_used, mask_used = _truncate_for_fixed_window(ids, attn_mask, cfg.state_length, args.max_new_tokens)
            used_len = int(ids_used.shape[1])
            hf_prefill[key] = _hf_prefill_logits(hf_model, ids_used, mask_used)
            text, step_logits = _hf_generate_with_logits(hf_model, tokenizer, ids_used, mask_used, args.max_new_tokens)
            hf_decode_text[key] = text
            hf_decode_logits[key] = step_logits
            token_meta[key] = (full_len, used_len)
            truncated_ids[key] = ids_used
            truncated_mask[key] = mask_used

    del hf_model
    gc.collect()

    anemll_model = Qwen35ForCausalLM(cfg).half().eval()
    if not anemll_model.load_pretrained_weights(args.model_path):
        raise RuntimeError("ANEMLL loader failed.")
    _apply_linear_mode(anemll_model, args.linear_mode)

    rows: List[CaseResult] = []
    for mode_name, think_mode in modes:
        for prompt_name in prompt_names:
            key = f"{mode_name}:{prompt_name}"
            ids_used = truncated_ids[key]
            ours_prefill = _ours_prefill_logits(anemll_model, ids_used)
            ours_text, ours_decode_logits = _ours_generate_with_logits(anemll_model, tokenizer, ids_used, args.max_new_tokens)
            prefill_metrics = _metrics(ours_prefill, hf_prefill[key])
            prefill_kl = _kl_div(ours_prefill, hf_prefill[key])
            prefill_top1 = _top1_agreement(ours_prefill, hf_prefill[key])

            paired_steps = min(len(ours_decode_logits), len(hf_decode_logits[key]))
            decode_metrics_rows = [_metrics(ours_decode_logits[i], hf_decode_logits[key][i]) for i in range(paired_steps)]
            decode_kl_rows = [_kl_div(ours_decode_logits[i], hf_decode_logits[key][i]) for i in range(paired_steps)]
            decode_top1_rows = [_top1_agreement(ours_decode_logits[i], hf_decode_logits[key][i]) for i in range(paired_steps)]
            decode_mean_metrics = _mean_metric_dict(decode_metrics_rows) if decode_metrics_rows else {
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "rmse": 0.0,
                "cosine": 1.0,
            }
            decode_mean_kl = float(sum(decode_kl_rows) / len(decode_kl_rows)) if decode_kl_rows else 0.0
            decode_top1 = float(sum(decode_top1_rows) / len(decode_top1_rows)) if decode_top1_rows else 1.0

            rows.append(
                CaseResult(
                    mode=mode_name,
                    prompt_name=prompt_name,
                    input_tokens_full=token_meta[key][0],
                    input_tokens_used=token_meta[key][1],
                    prefill_metrics=prefill_metrics,
                    prefill_kl=prefill_kl,
                    prefill_top1=prefill_top1,
                    decode_mean_metrics=decode_mean_metrics,
                    decode_mean_kl=decode_mean_kl,
                    decode_top1=decode_top1,
                    hf_text=hf_decode_text[key],
                    ours_text=ours_text,
                    similarity=_similarity(hf_decode_text[key], ours_text),
                )
            )

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    _write_report(args.report_path, args.model_path, args.max_new_tokens, cfg.state_length, args.linear_mode, rows)
    print(f"wrote report: {args.report_path}")
    for r in rows:
        print(
            f"{r.mode:6s} {r.prompt_name:6s} full={r.input_tokens_full:4d} used={r.input_tokens_used:4d} "
            f"prefill_cos={r.prefill_metrics['cosine']:.6f} decode_cos={r.decode_mean_metrics['cosine']:.6f} "
            f"decode_top1={r.decode_top1:.4f} sim={r.similarity:.4f}"
        )


if __name__ == "__main__":
    main()
