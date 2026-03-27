#!/usr/bin/env python3
"""Compare GPT-2 vs Qwen2 pre-tokenization regex patterns."""

import regex

# Qwen2 pattern (from transformers Qwen2Tokenizer source)
qwen_pat = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# GPT-2 pattern (used by the Swift QwenTokenizer currently)
gpt2_pat = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

print("Qwen2 pattern:", qwen_pat)
print("GPT-2 pattern:", gpt2_pat)

# Test 1: Simple user message
test = "What is the capital of France?"
gpt2_matches = [m.group() for m in regex.finditer(gpt2_pat, test)]
qwen_matches = [m.group() for m in regex.finditer(qwen_pat, test)]
print("\n=== Test: user message ===")
print(f"Input: {test!r}")
print(f"GPT-2 splits: {gpt2_matches}")
print(f"Qwen2 splits: {qwen_matches}")

# Test 2: Full ChatML prompt (without special tokens, since those are split separately)
parts = [
    "system",
    "You are a helpful assistant.",
    "user",
    "What is the capital of France?",
    "assistant",
]
for part in parts:
    gpt2_m = [m.group() for m in regex.finditer(gpt2_pat, part)]
    qwen_m = [m.group() for m in regex.finditer(qwen_pat, part)]
    if gpt2_m != qwen_m:
        print(f"\n*** MISMATCH for {part!r}:")
        print(f"  GPT-2: {gpt2_m}")
        print(f"  Qwen2: {qwen_m}")

# Test 3: Encode with HF tokenizer for reference
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/Users/yw68/Anemll/qwen3_5_stable_models/")

prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
ids = tok.encode(prompt)
print(f"\n=== HF reference tokenization ===")
print(f"Token IDs ({len(ids)}): {ids}")
for tid in ids:
    print(f"  {tid} -> {tok.decode([tid])!r}")
