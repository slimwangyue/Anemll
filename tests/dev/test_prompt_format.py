#!/usr/bin/env python3
"""Compare Swift tokenization vs HF tokenization to find prompt format bugs."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("qwen3_5_stable_models/")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I'm Alice"},
]
hf_text = tok.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
hf_ids = tok.encode(hf_text)

print("=== HF FORMAT (correct) ===")
print(repr(hf_text))
print(f"IDs ({len(hf_ids)}):", hf_ids)
for i, t in enumerate(hf_ids):
    print(f"  [{i:2d}] {t:6d} {tok.decode([t])!r}")

# Swift prompt format (with extra newlines from multi-line strings)
swift_prompt = (
    "<|im_start|>system\n"
    "You are a helpful assistant.\n"
    "<|im_end|>\n"
    "\n"
    "<|im_start|>user\n"
    "I'm Alice\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
swift_ids = tok.encode(swift_prompt)

print("\n=== SWIFT FORMAT (current) ===")
print(repr(swift_prompt))
print(f"IDs ({len(swift_ids)}):", swift_ids)
for i, t in enumerate(swift_ids):
    print(f"  [{i:2d}] {t:6d} {tok.decode([t])!r}")

# Correct format (no extra newlines)
correct_prompt = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "I'm Alice<|im_end|>\n"
    "<|im_start|>assistant\n"
)
correct_ids = tok.encode(correct_prompt)

print("\n=== CORRECTED FORMAT ===")
print(repr(correct_prompt))
print(f"IDs ({len(correct_ids)}):", correct_ids)
for i, t in enumerate(correct_ids):
    print(f"  [{i:2d}] {t:6d} {tok.decode([t])!r}")
print(f"Matches HF: {correct_ids == hf_ids[:len(correct_ids)]}")

# User actual Swift IDs
print("\n=== USER ACTUAL SWIFT IDs ===")
print("First 10:", [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 198])
for t in [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 198]:
    print(f"  {t:6d} {tok.decode([t])!r}")
print("Last 10:", [40, 515, 76, 28445, 198, 248046, 198, 248045, 74455, 198])
for t in [40, 515, 76, 28445, 198, 248046, 198, 248045, 74455, 198]:
    print(f"  {t:6d} {tok.decode([t])!r}")
