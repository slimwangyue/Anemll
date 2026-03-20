#!/usr/bin/env python3
"""Phase 1: Generate PyTorch reference outputs for prefill parity validation.

Run this ONCE to save reference outputs. Then run phase2 to compare CoreML.
This script uses ~8 GB RAM (model weights). Do NOT run simultaneously with CoreML.

Usage:
    python -u prefill_parity_phase1.py
    python -u prefill_parity_phase1.py --seq-len 128 --num-chunks 2
"""
import argparse
import os
import sys

import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
from anemll.models.qwen3_5_model import MODEL_DTYPE, Qwen35Config, Qwen35ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B")
    parser.add_argument("--output-dir", default="/tmp/qwen35_prefill_parity")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument(
        "--prompt",
        default="Explain stack and heap memory in one paragraph and give one debugging tip.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seq_len = args.seq_len
    num_chunks = args.num_chunks

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    text = args.prompt
    while True:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
        if ids.shape[1] >= seq_len:
            ids = ids[:, :seq_len]
            break
        text = text + " " + args.prompt
    print(f"prompt_tokens={ids.shape[1]}")
    np.save(f"{args.output_dir}/input_ids.npy", ids.numpy())

    # Load model
    cfg = Qwen35Config.from_json(str(Path(args.model_path) / "config.json"))
    model = Qwen35ForCausalLM(cfg).half().eval()
    print("Loading weights...")
    ok = model.load_pretrained_weights(args.model_path)
    assert ok, "Failed to load weights"
    print("Weights loaded")

    # Embeddings
    with torch.no_grad():
        torch_hidden = model.model.embed_tokens(ids.to(torch.int32)).to(torch.float16)
    np.save(f"{args.output_dir}/embed_out.npy", torch_hidden.numpy())
    print(f"embed shape={torch_hidden.shape}")

    # Create shared inputs
    position_ids = torch.arange(seq_len, dtype=torch.int32)
    causal_mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float16)
    row = torch.arange(seq_len).reshape(seq_len, 1)
    col = torch.arange(seq_len).reshape(1, seq_len)
    causal_mask[:, :, col <= row] = 0
    current_pos = torch.tensor(0, dtype=torch.int32)

    # Compute chunk ranges
    total_layers = cfg.num_hidden_layers
    chunk_ranges = []
    start = 0
    for i in range(num_chunks):
        end = start + (total_layers // num_chunks) if i < num_chunks - 1 else total_layers
        chunk_ranges.append((start, end))
        start = end

    # Run each chunk
    hidden = torch_hidden.clone()
    with torch.no_grad():
        for ci, (s, e) in enumerate(chunk_ranges):
            local_layers = e - s
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
            hidden = model.model.process_layers_prefill_export_local_state(
                hidden_states=hidden,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                kv_cache_0=kv,
                linear_conv_state=conv,
                linear_recurrent_state=rec,
                start_layer=s,
                end_layer=e,
                apply_final_norm=False,
                expected_batch_size=1,
                expected_seq_len=seq_len,
            )
            np.save(f"{args.output_dir}/torch_chunk{ci+1}.npy", hidden.numpy())
            print(f"chunk {ci+1} ({s}:{e}) shape={hidden.shape} max={hidden.abs().max():.4f}")

    # LM head on last token (for top-1 comparison)
    with torch.no_grad():
        torch_final = model.model.norm(hidden.clone())
        torch_logits = (
            model.lm_head(torch_final[:, -1:, :].permute(0, 2, 1).unsqueeze(2))
            .squeeze(2)
            .permute(0, 2, 1)
        )
        torch_top = int(torch.argmax(torch_logits[0, -1, :]).item())
        print(f"torch_top={torch_top} token={tokenizer.decode([torch_top])!r}")
        np.save(f"{args.output_dir}/torch_top.npy", np.array([torch_top]))

    print(f"\nPhase 1 complete. Files saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
