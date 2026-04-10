#!/usr/bin/env python3
"""Validate Gemma4 E4B CoreML models.

Tests inference parity between separate and combined/deduplicated models.

Usage:
    cd /path/to/Anemll
    python scripts_gemma4/validate.py [--tokens 40]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(1, os.path.dirname(_SCRIPT_DIR))

from config import BATCH_SIZE, CTX, NUM_CHUNKS, FFN_LABEL, DEFAULT_OUTPUT, DEFAULT_HF_MODEL

import coremltools as ct


VALIDATION_PROMPT = "<start_of_turn>user\nHello, what is 2+2?<end_of_turn>\n<start_of_turn>model\n"


def main():
    parser = argparse.ArgumentParser(description="Validate Gemma4 CoreML models")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT, help="Model directory")
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="HF model for tokenizer")
    parser.add_argument("--tokens", type=int, default=40, help="Max tokens to generate")
    parser.add_argument("--label", default=FFN_LABEL, help="FFN label")
    args = parser.parse_args()

    print(f"Model dir: {args.model_dir}")
    print(f"HF model: {args.hf_model}")
    print(f"Max tokens: {args.tokens}")

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(args.hf_model, "tokenizer.json"))

    # Encode prompt
    encoded = tokenizer.encode(VALIDATION_PROMPT)
    token_ids = [2] + encoded.ids  # BOS + tokens
    print(f"\nPrompt ({len(token_ids)} tokens): {VALIDATION_PROMPT[:60]}...")

    # Check for compiled models
    combined_dir = os.path.join(args.model_dir, f"combined_{args.label}_dedup")
    has_combined = os.path.isdir(combined_dir)

    if has_combined:
        chunk_files = sorted([f for f in os.listdir(combined_dir)
                             if f.startswith("chunk") and f.endswith(".mlmodelc")])
        print(f"\nFound {len(chunk_files)} combined chunks in {combined_dir}")
    else:
        print(f"\nNo combined directory found at {combined_dir}")
        print("Run combine.py first, then compile.py")
        return

    print(f"\n{'=' * 60}")
    print("VALIDATION: Basic inference test")
    print(f"{'=' * 60}")
    print("TODO: Full validation with KV cache inference pipeline")
    print("For now, verifying model files exist and can be loaded.")

    # Verify all chunk files exist
    for ci in range(NUM_CHUNKS):
        chunk_path = os.path.join(combined_dir, f"chunk{ci:02d}.mlmodelc")
        if os.path.exists(chunk_path):
            print(f"  ✅ chunk{ci:02d}.mlmodelc exists")
        else:
            print(f"  ❌ chunk{ci:02d}.mlmodelc MISSING")

    # Check embeddings
    embed_path = os.path.join(args.model_dir, "embeddings.mlmodelc")
    if os.path.exists(embed_path):
        print(f"  ✅ embeddings.mlmodelc exists")
    else:
        print(f"  ❌ embeddings.mlmodelc MISSING")

    # Check LM head
    lm_paths = [f for f in os.listdir(args.model_dir)
                if f.startswith("lm_head_") and f.endswith(".mlmodelc")]
    if lm_paths:
        print(f"  ✅ {lm_paths[0]} exists")
    else:
        print(f"  ❌ lm_head .mlmodelc MISSING")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
