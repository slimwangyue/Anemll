#!/usr/bin/env python3
"""Minimal prefill chunk-1 probe for exported Qwen3.5 chunk runtime."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer

from anemll.models.qwen3_5_model import MODEL_DTYPE, Qwen35Config, Qwen35ForCausalLM


def _resolve_model_path(path: Path, prefer_compiled: bool = True) -> Path:
    if path.exists():
        compiled = path.with_suffix(".mlmodelc")
        if prefer_compiled and path.suffix == ".mlpackage" and compiled.exists():
            return compiled
        return path
    if prefer_compiled and path.suffix == ".mlpackage":
        compiled = path.with_suffix(".mlmodelc")
        if compiled.exists():
            return compiled
    raise FileNotFoundError(path)


def _load_model(path: Path, compute_unit, *, prefer_compiled: bool = True):
    resolved = _resolve_model_path(path, prefer_compiled=prefer_compiled)
    if resolved.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(resolved), compute_unit)
    return ct.models.MLModel(str(resolved), compute_units=compute_unit)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = a_flat.norm() * b_flat.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / denom)


def _make_causal_mask(length: int, context_length: int) -> torch.Tensor:
    mask = torch.full((1, 1, length, context_length), float("-inf"), dtype=torch.float16)
    row_idx = torch.arange(length).reshape(length, 1)
    col_idx = torch.arange(context_length).reshape(1, context_length)
    mask[:, :, col_idx <= row_idx] = 0
    return mask


def _build_fixed_prompt_ids(tokenizer, prompt: str, target_len: int) -> torch.Tensor:
    text = prompt
    while True:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
        if ids.shape[1] >= target_len:
            return ids[:, :target_len]
        text = text + " " + prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--prefix", default="qwen35")
    parser.add_argument(
        "--prompt",
        default="Explain stack and heap memory in one paragraph and give one debugging tip.",
    )
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args()

    compute_unit = ct.ComputeUnit.CPU_ONLY
    export_dir = Path(args.export_dir)
    seq_len = args.context_length

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    input_ids = _build_fixed_prompt_ids(tokenizer, args.prompt, seq_len)
    print(f"prompt_tokens={input_ids.shape[1]}")

    cfg = Qwen35Config.from_json(str(Path(args.model_path) / "config.json"))
    torch_model = Qwen35ForCausalLM(cfg).half().eval()
    print("loading_repo_weights...")
    ok = torch_model.load_pretrained_weights(args.model_path)
    if not ok:
        raise RuntimeError("Failed to load repo weights")
    print("loading_repo_weights_done")

    with torch.no_grad():
        torch_hidden = torch_model.model.embed_tokens(input_ids.to(torch.int32)).to(torch.float16)

    position_ids = torch.arange(seq_len, dtype=torch.int32)
    causal_mask = _make_causal_mask(seq_len, args.context_length)
    current_pos = torch.tensor(0, dtype=torch.int32)

    local_layers = 8
    kv = torch.zeros((2 * local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE)
    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    conv = torch.zeros((local_layers, conv_dim, conv_kernel), dtype=MODEL_DTYPE)
    rec = torch.zeros(
        (
            local_layers,
            cfg.text_config.linear_num_value_heads,
            cfg.text_config.linear_key_head_dim,
            cfg.text_config.linear_value_head_dim,
        ),
        dtype=MODEL_DTYPE,
    )

    with torch.no_grad():
        torch_chunk = torch_model.model.process_layers_prefill_export_local_state(
            hidden_states=torch_hidden.clone(),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            kv_cache_0=kv,
            linear_conv_state=conv,
            linear_recurrent_state=rec,
            start_layer=0,
            end_layer=8,
            apply_final_norm=False,
            expected_batch_size=1,
            expected_seq_len=seq_len,
        )
    print("torch_chunk1_done")

    # CoreML package loading is much more stable once the full PyTorch model and
    # temporary state tensors are dropped. We only need the chunk output tensor
    # for the parity check below.
    del torch_model, cfg, kv, conv, rec, torch_hidden
    gc.collect()

    embed_model = _load_model(export_dir / f"{args.prefix}_embeddings.mlpackage", compute_unit)
    chunk_model = _load_model(
        export_dir / f"{args.prefix}_prefill_chunk_01of04.mlpackage",
        compute_unit,
        prefer_compiled=False,
    )
    print("coreml_models_loaded")
    state = chunk_model.make_state()
    print("coreml_state_created")
    hidden_np = embed_model.predict({"input_ids": input_ids.numpy().astype(np.int32)})["hidden_states"]
    print(f"embed_hidden_shape={hidden_np.shape}")
    out = chunk_model.predict(
        {
            "hidden_states": hidden_np,
            "position_ids": position_ids.numpy(),
            "causal_mask": causal_mask.numpy(),
            "current_pos": np.array([0], dtype=np.int32),
        },
        state,
    )
    coreml_chunk = torch.from_numpy(out["output_hidden_states"])
    diff = (coreml_chunk.float() - torch_chunk.float()).abs()
    print(
        f"chunk1_prefill max_abs={float(diff.max()):.6f} "
        f"mean_abs={float(diff.mean()):.6f} "
        f"cos={_cosine(coreml_chunk, torch_chunk):.6f}"
    )


if __name__ == "__main__":
    main()
