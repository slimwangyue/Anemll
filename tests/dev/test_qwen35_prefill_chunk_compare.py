#!/usr/bin/env python3
"""Compare exported Qwen3.5 prefill chunks against repo PyTorch on a 256-token prompt."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer

from anemll.models.qwen3_5_model import MODEL_DTYPE, Qwen35Config, Qwen35ForCausalLM


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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = a_flat.norm() * b_flat.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / denom)


def _chunk_ranges(num_layers: int, num_chunks: int) -> List[Tuple[int, int]]:
    layers_per_chunk = num_layers // num_chunks
    out = []
    start = 0
    for idx in range(num_chunks):
        end = start + layers_per_chunk
        if idx == num_chunks - 1:
            end = num_layers
        out.append((start, end))
        start = end
    return out


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


def _parse_logits(output: Dict[str, np.ndarray]) -> torch.Tensor:
    if "output_logits" in output:
        return torch.from_numpy(output["output_logits"])
    if "logits" in output:
        return torch.from_numpy(output["logits"])
    raise KeyError(f"Unrecognized lm_head outputs: {list(output.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--prefix", default="qwen35")
    parser.add_argument(
        "--prompt",
        default="Explain stack and heap memory in one paragraph and give one debugging tip.",
    )
    parser.add_argument("--num-chunks", type=int, default=4)
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
        raise RuntimeError("Failed to load Qwen3.5 weights into repo model")
    print("loading_repo_weights_done")

    with torch.no_grad():
        torch_hidden = torch_model.model.embed_tokens(input_ids.to(torch.int32)).to(torch.float16)

    position_ids = torch.arange(seq_len, dtype=torch.int32)
    causal_mask = _make_causal_mask(seq_len, args.context_length)
    current_pos = torch.tensor(0, dtype=torch.int32)

    chunk_ranges = _chunk_ranges(cfg.num_hidden_layers, args.num_chunks)
    local_states: List[Dict[str, torch.Tensor]] = []
    for start, end in chunk_ranges:
        local_layers = end - start
        kv = torch.zeros(
            (2 * local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=MODEL_DTYPE,
        )
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
        local_states.append({"kv": kv, "conv": conv, "rec": rec})

    torch_chunk_hidden: List[torch.Tensor] = []
    with torch.no_grad():
        hidden = torch_hidden.clone()
        for (start, end), state in zip(chunk_ranges, local_states):
            print(f"torch_chunk_start {start}:{end}")
            hidden = torch_model.model.process_layers_prefill_export_local_state(
                hidden_states=hidden,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                kv_cache_0=state["kv"],
                linear_conv_state=state["conv"],
                linear_recurrent_state=state["rec"],
                start_layer=start,
                end_layer=end,
                apply_final_norm=False,
                expected_batch_size=1,
                expected_seq_len=seq_len,
            )
            torch_chunk_hidden.append(hidden.clone())
            print(f"torch_chunk_done {start}:{end}")

    with torch.no_grad():
        torch_final = torch_model.model.norm(torch_chunk_hidden[-1].clone())
        torch_logits = torch_model.lm_head(torch_final[:, -1:, :].transpose(1, 2)).transpose(1, 2)
        torch_top = int(torch.argmax(torch_logits[0, -1, :]).item())

    embed_model = _load_model(export_dir / f"{args.prefix}_embeddings.mlpackage", compute_unit)
    print("coreml_embed_loaded")
    chunk_models = [
        _load_model(export_dir / f"{args.prefix}_prefill_chunk_{idx:02d}of{args.num_chunks:02d}.mlpackage", compute_unit)
        for idx in range(1, args.num_chunks + 1)
    ]
    print("coreml_prefill_chunks_loaded")
    coreml_states = [model.make_state() for model in chunk_models]
    print("coreml_states_created")
    lm_head_model = _load_model(export_dir / f"{args.prefix}_lm_head_lut6.mlpackage", compute_unit)
    print("coreml_lmhead_loaded")

    hidden_np = embed_model.predict({"input_ids": input_ids.numpy().astype(np.int32)})["hidden_states"]
    inputs = {
        "hidden_states": hidden_np,
        "position_ids": position_ids.numpy(),
        "causal_mask": causal_mask.numpy(),
        "current_pos": np.array([0], dtype=np.int32),
    }
    coreml_chunk_hidden: List[torch.Tensor] = []
    for chunk_idx, (chunk_model, state) in enumerate(zip(chunk_models, coreml_states), start=1):
        print(f"coreml_chunk_start {chunk_idx}")
        out = chunk_model.predict(inputs, state)
        inputs["hidden_states"] = out["output_hidden_states"]
        coreml_chunk_hidden.append(torch.from_numpy(inputs["hidden_states"]))
        print(f"coreml_chunk_done {chunk_idx}")

    for idx, ((start, end), torch_out, coreml_out) in enumerate(
        zip(chunk_ranges, torch_chunk_hidden, coreml_chunk_hidden), start=1
    ):
        diff = (coreml_out.float() - torch_out.float()).abs()
        print(
            f"chunk {idx} layers {start}:{end} "
            f"max_abs={float(diff.max()):.6f} "
            f"mean_abs={float(diff.mean()):.6f} "
            f"cos={_cosine(coreml_out, torch_out):.6f}"
        )

    coreml_logits = _parse_logits(
        lm_head_model.predict({"hidden_states": coreml_chunk_hidden[-1][:, -1:, :].numpy().astype(np.float16)})
    )
    coreml_top = int(torch.argmax(coreml_logits[0, -1, :]).item())
    print(f"torch_last_top={torch_top} token={tokenizer.decode([torch_top])!r}")
    print(f"coreml_last_top={coreml_top} token={tokenizer.decode([coreml_top])!r}")


if __name__ == "__main__":
    main()
