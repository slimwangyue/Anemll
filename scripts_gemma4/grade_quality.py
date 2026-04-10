#!/usr/bin/env python3
"""Grade quality of Gemma4 E4B CoreML vs PyTorch FP16/FP32.

Compares generation output across backends for quality assessment.

Usage:
    cd /path/to/Anemll
    python scripts_gemma4/grade_quality.py --hf-model ~/local_llm/models/google__gemma-4-E4B-it
"""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(1, os.path.dirname(_SCRIPT_DIR))

from config import DEFAULT_HF_MODEL, DEFAULT_OUTPUT, CTX, BATCH_SIZE


GRADING_PROMPTS = [
    "Explain the concept of recursion in simple terms.",
    "Write a haiku about the ocean.",
    "What is the capital of France and what is it known for?",
    "def fibonacci(n):\n    # Complete this function\n",
    "Translate to Spanish: 'The weather is beautiful today.'",
]


def main():
    parser = argparse.ArgumentParser(description="Grade Gemma4 model quality")
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="HF model path")
    parser.add_argument("--coreml-dir", default=DEFAULT_OUTPUT, help="CoreML model directory")
    parser.add_argument("--tokens", type=int, default=100, help="Max tokens per prompt")
    args = parser.parse_args()

    print(f"HF Model: {args.hf_model}")
    print(f"CoreML Dir: {args.coreml_dir}")
    print(f"Max Tokens: {args.tokens}")

    # Run PyTorch FP32 reference
    os.environ['ANEMLL_ALLOW_MISSING_WEIGHTS'] = '1'
    import torch
    from config import Gemma4ForCausalLM, Gemma4Config
    import anemll.models.gemma4_model as g4
    g4.MODEL_DTYPE = torch.float32

    cfg = Gemma4Config.from_json(os.path.join(args.hf_model, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX

    model = Gemma4ForCausalLM(cfg)
    model.load_pretrained_weights(args.hf_model)
    model.float()
    model.eval()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(args.hf_model, "tokenizer.json"))

    NEGINF = -1e9

    for pi, prompt in enumerate(GRADING_PROMPTS):
        print(f"\n{'='*60}")
        print(f"PROMPT {pi+1}: {prompt[:60]}")
        print(f"{'='*60}")

        encoded = tokenizer.encode(prompt)
        input_tokens = [2] + encoded.ids
        generated = list(input_tokens)

        with torch.no_grad():
            # Feed prompt tokens
            for idx, tok_id in enumerate(generated):
                tok = torch.tensor([[tok_id]], dtype=torch.long)
                pos = torch.tensor([idx], dtype=torch.long)
                cp = torch.tensor([idx], dtype=torch.long)
                um = torch.zeros((1, 1, CTX, 1), dtype=torch.float32)
                um[0, 0, idx, 0] = 1.0
                cm = torch.full((1, 1, 1, CTX), NEGINF, dtype=torch.float32)
                cm[0, 0, 0, :idx + 1] = 0.0
                output = model(tok, um, pos, cm, cp)
                if isinstance(output, tuple):
                    logits = torch.cat(list(output), dim=-1)
                else:
                    logits = output

            # Generate
            for g in range(args.tokens):
                next_token = logits.argmax(dim=-1).item()
                if next_token == 1:  # EOS
                    break
                generated.append(next_token)
                idx = len(generated) - 1
                if idx >= CTX - 1:
                    break
                tok = torch.tensor([[next_token]], dtype=torch.long)
                pos = torch.tensor([idx], dtype=torch.long)
                cp = torch.tensor([idx], dtype=torch.long)
                um = torch.zeros((1, 1, CTX, 1), dtype=torch.float32)
                um[0, 0, idx, 0] = 1.0
                cm = torch.full((1, 1, 1, CTX), NEGINF, dtype=torch.float32)
                cm[0, 0, 0, :idx + 1] = 0.0
                output = model(tok, um, pos, cm, cp)
                if isinstance(output, tuple):
                    logits = torch.cat(list(output), dim=-1)
                else:
                    logits = output

        text = tokenizer.decode(generated)
        print(f"\nPyTorch FP32 ({len(generated) - len(input_tokens)} tokens):")
        print(text)

    print(f"\n{'='*60}")
    print("Quality grading complete.")


if __name__ == "__main__":
    main()
