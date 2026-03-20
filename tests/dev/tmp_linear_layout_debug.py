#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B")
    parser.add_argument("--layer-idx", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    model = Qwen35ForCausalLM(cfg).half().eval()
    if not model.load_pretrained_weights(args.model_path):
        raise RuntimeError("failed to load weights")
    attn = model.model.layers[args.layer_idx].self_attn

    torch.manual_seed(7)
    hidden_states = torch.randn(1, args.seq_len, cfg.hidden_size, dtype=torch.float16)
    conv_state = torch.zeros((1, attn.conv_dim, attn.linear_conv_kernel_dim), dtype=torch.float16)

    print("Linear Token-Mixer Layout Flow")
    attn.print_token_mixer_layout(hidden_states, conv_state=conv_state, expected_seq_len=args.seq_len)


if __name__ == "__main__":
    main()
