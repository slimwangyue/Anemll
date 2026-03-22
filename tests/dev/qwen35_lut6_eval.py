#!/usr/bin/env python3
"""Evaluate LUT6 LM head for Qwen3.5-4B vs fp16 baseline.

Exports a LUT6 LM head, compares:
  1. Top-1 agreement rate
  2. Size reduction
  3. Text generation quality (20 tokens)

Usage:
    python tests/dev/qwen35_lut6_eval.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import gc, time
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1"
CTX = 1024
BATCH_SIZE = 256
NUM_CHUNKS = 4
CU = ct.ComputeUnit.CPU_AND_NE
LUT6_PATH = os.path.join(MODEL_DIR, "lm_head_LUT6.mlpackage")


def export_lut6_lmhead():
    """Export LM head with LUT6 quantization."""
    if os.path.exists(LUT6_PATH):
        print(f"LUT6 LM head already exists: {LUT6_PATH}")
        return

    print("Exporting LUT6 LM head...")
    import torch
    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(MODEL_PATH), "Weight loading failed"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=6, per_channel=8)
    ml = conv.convert_part_3(model, argmax_in_model=False)
    ml.save(LUT6_PATH)
    print(f"  Exported LUT6 LM head in {time.time()-t0:.1f}s")

    del ml, conv, model
    gc.collect()


def get_model_size_mb(path):
    """Get total file size of a model (handles directories)."""
    total = 0
    if os.path.isdir(path):
        for dp, _, fns in os.walk(path):
            for f in fns:
                total += os.path.getsize(os.path.join(dp, f))
    else:
        total = os.path.getsize(path)
    return total / (1024 * 1024)


def compare_logits(fp16_model, lut6_model, hidden_states_list):
    """Compare logits between fp16 and LUT6 lm heads."""
    top1_match = 0
    top5_match = 0
    total = len(hidden_states_list)

    for hidden in hidden_states_list:
        inp = {"hidden_states": hidden.astype(np.float16)}
        fp16_out = fp16_model.predict(inp)
        lut6_out = lut6_model.predict(inp)

        if "logits" in fp16_out:
            fp16_logits = fp16_out["logits"].flatten()
            lut6_logits = lut6_out["logits"].flatten()

            fp16_top1 = np.argmax(fp16_logits)
            lut6_top1 = np.argmax(lut6_logits)

            if fp16_top1 == lut6_top1:
                top1_match += 1

            fp16_top5 = set(np.argsort(fp16_logits)[-5:])
            lut6_top5 = set(np.argsort(lut6_logits)[-5:])
            if fp16_top1 in lut6_top5:
                top5_match += 1
        else:
            # argmax mode
            fp16_idx = int(fp16_out["argmax_idx"].flatten()[0])
            lut6_idx = int(lut6_out["argmax_idx"].flatten()[0])
            if fp16_idx == lut6_idx:
                top1_match += 1
                top5_match += 1  # can't measure top5 in argmax mode

    return top1_match / total, top5_match / total


def generate_tokens(embed, lmhead, ffns, states, lin_convs, lin_recs,
                    prompt_tokens, max_gen, tokenizer, stop_ids):
    """Generate tokens and collect hidden states for evaluation."""
    hidden_states_list = []
    pos = 0

    for tok_id in prompt_tokens:
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
            }
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
        pos += 1

    # Collect hidden states during generation
    lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if "logits" in lm_out:
        last_next = int(np.argmax(lm_out["logits"].flatten()))
    else:
        last_next = int(lm_out["argmax_idx"].flatten()[0])
    hidden_states_list.append(hidden.copy())
    gen_tokens = [last_next]

    for _ in range(max_gen - 1):
        if pos >= CTX - 1:
            break
        tok = np.array([[gen_tokens[-1]]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
            }
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
        pos += 1
        hidden_states_list.append(hidden.copy())

        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            next_id = int(np.argmax(lm_out["logits"].flatten()))
        else:
            next_id = int(lm_out["argmax_idx"].flatten()[0])
        gen_tokens.append(next_id)

        if next_id in stop_ids:
            break

    return gen_tokens, hidden_states_list


def main():
    print("=" * 60)
    print("Qwen3.5-4B LUT6 LM Head Evaluation")
    print("=" * 60)

    # Step 1: Export LUT6 LM head if not exists
    export_lut6_lmhead()

    # Step 2: Size comparison
    fp16_size = get_model_size_mb(os.path.join(MODEL_DIR, "lm_head.mlpackage"))
    lut6_size = get_model_size_mb(LUT6_PATH)
    reduction = (1 - lut6_size / fp16_size) * 100
    print(f"\n── Size Comparison ──")
    print(f"  fp16 LM head: {fp16_size:.0f} MB")
    print(f"  LUT6 LM head: {lut6_size:.0f} MB")
    print(f"  Reduction: {reduction:.1f}%")

    # Step 3: Load models
    print(f"\n── Loading Models ──")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)

    embed = ct.models.MLModel(os.path.join(MODEL_DIR, "embeddings.mlpackage"), compute_units=CU)
    fp16_lmhead = ct.models.MLModel(os.path.join(MODEL_DIR, "lm_head.mlpackage"), compute_units=CU)
    lut6_lmhead = ct.models.MLModel(LUT6_PATH, compute_units=CU)
    ffns = []
    for ci in range(NUM_CHUNKS):
        ffns.append(ct.models.MLModel(
            os.path.join(MODEL_DIR, f"ffn_LUT4_chunk{ci}.mlpackage"), compute_units=CU))

    # Get input shapes
    spec = ffns[0].get_spec()
    inp_map = {}
    for inp in spec.description.input:
        try:
            inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
        except:
            pass

    # Step 4: Generate with fp16 and collect hidden states
    print(f"\n── Generation Test ──")
    prompts = [
        "What is the capital of France?",
        "Write a Python function to compute fibonacci numbers.",
        "Explain quantum computing in simple terms.",
    ]

    results = []
    for pi, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=True)
        if hasattr(input_ids, 'input_ids'):
            input_ids = input_ids.input_ids
        prompt_tokens = input_ids[0].tolist()

        print(f"\n  Prompt {pi+1}: '{prompt[:50]}...' ({len(prompt_tokens)} tokens)")

        # Generate with fp16
        states = [m.make_state() for m in ffns]
        lin_convs = [np.zeros(inp_map['linear_conv_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
        lin_recs = [np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]

        fp16_tokens, hidden_list = generate_tokens(
            embed, fp16_lmhead, ffns, states, lin_convs, lin_recs,
            prompt_tokens, 30, tokenizer, stop_ids)
        fp16_text = tokenizer.decode(fp16_tokens, skip_special_tokens=True)

        # Compare logits (fp16 vs LUT6) on collected hidden states
        top1_rate, top5_rate = compare_logits(fp16_lmhead, lut6_lmhead, hidden_list)
        results.append((top1_rate, top5_rate))

        # Generate with LUT6 (same model state, just different lm_head)
        states2 = [m.make_state() for m in ffns]
        lin_convs2 = [np.zeros(inp_map['linear_conv_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
        lin_recs2 = [np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]

        lut6_tokens, _ = generate_tokens(
            embed, lut6_lmhead, ffns, states2, lin_convs2, lin_recs2,
            prompt_tokens, 30, tokenizer, stop_ids)
        lut6_text = tokenizer.decode(lut6_tokens, skip_special_tokens=True)

        token_match = sum(1 for a, b in zip(fp16_tokens, lut6_tokens) if a == b) / min(len(fp16_tokens), len(lut6_tokens))

        print(f"    fp16: {fp16_text[:80]}...")
        print(f"    LUT6: {lut6_text[:80]}...")
        print(f"    Top-1 agreement: {top1_rate*100:.1f}%")
        print(f"    Top-5 agreement: {top5_rate*100:.1f}%")
        print(f"    Token match: {token_match*100:.1f}% ({sum(1 for a, b in zip(fp16_tokens, lut6_tokens) if a == b)}/{min(len(fp16_tokens), len(lut6_tokens))})")

    # Summary
    avg_top1 = sum(r[0] for r in results) / len(results) * 100
    avg_top5 = sum(r[1] for r in results) / len(results) * 100

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  fp16 size: {fp16_size:.0f} MB")
    print(f"  LUT6 size: {lut6_size:.0f} MB ({reduction:.1f}% reduction)")
    print(f"  Avg Top-1 agreement: {avg_top1:.1f}%")
    print(f"  Avg Top-5 agreement: {avg_top5:.1f}%")

    if avg_top1 >= 95:
        print(f"\n  RECOMMENDATION: LUT6 LM head is viable (>{avg_top1:.0f}% top-1 match)")
    elif avg_top1 >= 85:
        print(f"\n  RECOMMENDATION: LUT6 has moderate accuracy loss, consider testing further")
    else:
        print(f"\n  RECOMMENDATION: LUT6 has significant accuracy loss, keep fp16")

    return 0


if __name__ == "__main__":
    sys.exit(main())
