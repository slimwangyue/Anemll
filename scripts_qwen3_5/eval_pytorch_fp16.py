#!/usr/bin/env python3
"""Evaluate Qwen3.5-4B using the ANEMLL PyTorch model in FP16 (no quantization).

Runs the same 6 prompts as eval_blockrecur.py using identical sampling parameters
(temperature=0.7, top_p=0.8, presence_penalty=1.5) to serve as a ground-truth
FP16 baseline for comparison against LUT6-quantized ANE inference.
"""

import sys, os, time, json
import numpy as np

# Add ANEMLL to path
sys.path.insert(0, "/Users/yw68/Anemll")

import torch
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"
MAX_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.8
PRESENCE_PENALTY = 1.5

PROMPTS = [
    "Explain the key differences between TCP and UDP protocols in computer networking. "
    "Include at least three specific differences and when you would use each one.",

    "A farmer has a rectangular field that is 120 meters long and 80 meters wide. "
    "He wants to build a fence around the entire field and also divide it into four "
    "equal sections with internal fences parallel to the shorter side. How many meters "
    "of fencing material does he need in total?",

    "请用三段话描述一个未来城市的生活场景。第一段描述交通，第二段描述住宅，"
    "第三段描述人们的日常工作和娱乐方式。每段至少写三句话。",

    "List the top 5 largest countries in the world by land area. For each country, "
    "provide its approximate area in square kilometers, its capital city, and the "
    "continent it is primarily located on. Format your answer as a numbered list.",

    "Write a Python function called 'merge_sorted_lists' that takes two sorted "
    "lists of integers and returns a single sorted list containing all elements "
    "from both lists. Use the merge step of merge sort, not the built-in sort. "
    "Include a brief docstring explaining the time complexity.",

    "If a train leaves Station A at 9:00 AM traveling east at 80 km/h, and another "
    "train leaves Station B (which is 500 km east of A) at 10:00 AM traveling west "
    "at 120 km/h, at what time will the two trains meet? Show your work step by step "
    "and express the answer in hours and minutes.",
]


def sample_token(logits_np, generated_ids, presence_penalty, temperature, top_p):
    """Sample with penalties + temperature + top-p, matching chat_server logic."""
    token_counts = {}
    for tid in generated_ids:
        token_counts[tid] = token_counts.get(tid, 0) + 1

    for tid, count in token_counts.items():
        if presence_penalty != 0.0:
            logits_np[tid] -= presence_penalty

    if temperature <= 0 or top_p <= 0:
        return int(np.argmax(logits_np))

    logits_f = logits_np.astype(np.float64)
    logits_f /= temperature
    logits_f -= np.max(logits_f)
    probs = np.exp(logits_f)
    probs /= probs.sum()

    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        mask = np.zeros_like(probs, dtype=bool)
        mask[sorted_idx[:cutoff]] = True
        probs[~mask] = 0.0
        probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))


def main():
    print("=" * 70)
    print("  PyTorch FP16 Baseline Evaluation")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Sampling: temp={TEMPERATURE}, top_p={TOP_P}, presence_penalty={PRESENCE_PENALTY}")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    # Collect stop IDs
    stop_ids = {eos_id} if eos_id is not None else set()
    for name in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    # Load model
    print("Loading ANEMLL PyTorch model...", flush=True)
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

    config = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    model = Qwen35ForCausalLM(config)
    ok = model.load_pretrained_weights(MODEL_PATH)
    if not ok:
        print("ERROR: Failed to load weights")
        sys.exit(1)
    model.eval()
    # Ensure entire model is in float16
    model = model.half()
    print(f"Model loaded. Layers: {len(model.model.layers)}")

    device = "cpu"  # FP16 on CPU; MPS has issues with some ops
    print(f"Device: {device}")

    results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\n  [{i+1}/{len(PROMPTS)}] Q: {prompt[:80]}...")

        # Tokenize with chat template
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
        if hasattr(input_ids, 'input_ids'):
            input_ids = input_ids.input_ids[0].tolist()
        else:
            input_ids = list(input_ids)

        prompt_len = len(input_ids)

        # Reset model state
        model.model.kv_cache_0.zero_()
        if hasattr(model.model, 'linear_conv_state'):
            model.model.linear_conv_state.zero_()
            model.model.linear_recurrent_state.zero_()

        generated_ids = []
        t_start = time.time()
        t_first_token = None

        with torch.no_grad():
            # Prefill: process prompt tokens one at a time (sequential)
            for pos in range(prompt_len):
                tok = torch.tensor([[input_ids[pos]]], dtype=torch.long, device=device)
                pos_ids = torch.tensor([[pos]], dtype=torch.long, device=device)
                current_pos = torch.tensor([pos], dtype=torch.long, device=device)
                logits = model(tok, pos_ids, causal_mask=None, current_pos=current_pos)

            # logits from last prompt token
            logits_np = logits[0, 0].float().cpu().numpy()
            first_id = sample_token(logits_np, generated_ids, PRESENCE_PENALTY, TEMPERATURE, TOP_P)
            generated_ids.append(first_id)
            t_first_token = time.time()

            # Decode loop
            for step in range(MAX_TOKENS - 1):
                if generated_ids[-1] in stop_ids:
                    break
                pos = prompt_len + step
                tok = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
                pos_ids = torch.tensor([[pos]], dtype=torch.long, device=device)
                current_pos = torch.tensor([pos], dtype=torch.long, device=device)
                logits = model(tok, pos_ids, causal_mask=None, current_pos=current_pos)
                logits_np = logits[0, 0].float().cpu().numpy()
                next_id = sample_token(logits_np, generated_ids, PRESENCE_PENALTY, TEMPERATURE, TOP_P)
                generated_ids.append(next_id)

        t_end = time.time()

        # Determine stop reason
        stop_reason = "length"
        if generated_ids and generated_ids[-1] in stop_ids:
            stop_reason = "eos"

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Strip thinking tags
        if "<think>" in text:
            end_tag = text.find("</think>")
            if end_tag >= 0:
                text = text[end_tag + len("</think>"):].strip()

        ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else 0
        decode_tokens = len(generated_ids)
        decode_elapsed = t_end - t_first_token if t_first_token else 0
        decode_tps = decode_tokens / decode_elapsed if decode_elapsed > 0 else 0

        print(f"  A: {text[:120]}...")
        print(f"  Tokens: {decode_tokens}, TTFT: {ttft_ms:.0f}ms, Decode: {decode_tps:.2f} tok/s, Stop: {stop_reason}")

        results.append({
            "prompt": prompt,
            "text": text,
            "decode_tokens": decode_tokens,
            "ttft_ms": ttft_ms,
            "decode_tps": decode_tps,
            "stop_reason": stop_reason,
            "prompt_len": prompt_len,
        })

    # Summary
    print("\n" + "=" * 70)
    print("  PYTORCH FP16 RESULTS SUMMARY")
    print("=" * 70)
    for i, r in enumerate(results):
        print(f"\n  Prompt {i+1}: {r['prompt'][:60]}...")
        print(f"    Stop: {r['stop_reason']} | Tokens: {r['decode_tokens']} | TTFT: {r['ttft_ms']:.0f}ms | Decode: {r['decode_tps']:.1f} tok/s")
        print(f"    {r['text'][:200]}")

    avg_ttft = np.mean([r['ttft_ms'] for r in results])
    avg_tps = np.mean([r['decode_tps'] for r in results])
    eos_count = sum(1 for r in results if r['stop_reason'] == 'eos')
    length_count = sum(1 for r in results if r['stop_reason'] == 'length')

    print(f"\n  Avg TTFT: {avg_ttft:.0f}ms")
    print(f"  Avg decode: {avg_tps:.2f} tok/s")
    print(f"  Stop reasons: eos={eos_count}, length={length_count}")

    # Save results
    out_path = "/tmp/eval_pytorch_fp16_results.json"
    with open(out_path, "w") as f:
        json.dump({"pytorch_fp16": results}, f, indent=2, ensure_ascii=False)
    print(f"\n  Raw results saved to {out_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
