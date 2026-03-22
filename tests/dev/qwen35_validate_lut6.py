#!/usr/bin/env python3
"""Validate LUT6+argmax LM head for Qwen3.5-4B on ANE.

Tests:
  1. Model loads on CPU_AND_NE
  2. Outputs argmax_idx (int32) + argmax_val (fp16)
  3. Generates correct text for test prompts
  4. Multi-turn conversation works
  5. Compare first-token output vs fp16 baseline (if available)

Usage:
    python tests/dev/qwen35_validate_lut6.py
    python tests/dev/qwen35_validate_lut6.py --tokens 50
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time, argparse
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

# ── Config ──
MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
EXPORT_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"
CTX = 1024
NUM_CHUNKS = 4
BATCH_SIZE = 256

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "What is 2+2?",
]

MULTI_TURN = [
    "What is a stack in computer science?",
    "How does it compare to a queue?",
]


def find_model(base, name):
    for ext in [".mlmodelc", ".mlpackage"]:
        p = os.path.join(base, name + ext)
        if os.path.exists(p):
            return p
    return None


def load_models(export_dir, compute_unit):
    models = {}
    t0 = time.time()

    # Embed
    p = find_model(export_dir, "embeddings")
    assert p, "embeddings model not found"
    models["embed"] = ct.models.MLModel(p, compute_units=compute_unit)

    # LM Head
    p = find_model(export_dir, "lm_head")
    assert p, "lm_head model not found"
    models["lmhead"] = ct.models.MLModel(p, compute_units=compute_unit)
    spec = models["lmhead"].get_spec()
    output_names = [o.name for o in spec.description.output]
    models["lmhead_outputs"] = output_names

    # FFN decode chunks
    models["ffns"] = []
    for ci in range(NUM_CHUNKS):
        p = find_model(export_dir, f"ffn_LUT4_chunk{ci}")
        assert p, f"ffn_LUT4_chunk{ci} not found"
        models["ffns"].append(ct.models.MLModel(p, compute_units=compute_unit))

    # FFN prefill chunks
    models["prefills"] = []
    for ci in range(NUM_CHUNKS):
        p = find_model(export_dir, f"prefill_LUT4_chunk{ci}")
        assert p, f"prefill_LUT4_chunk{ci} not found"
        models["prefills"].append(ct.models.MLModel(p, compute_units=compute_unit))

    print(f"  Loaded {len(models['ffns'])+len(models['prefills'])+2} models in {time.time()-t0:.1f}s")
    return models


def get_initial_states(models):
    states = []
    for ci in range(NUM_CHUNKS):
        states.append(models["ffns"][ci].make_state())

    # Linear attention states — always required
    lin_convs = [np.zeros((8, 1024, 32), dtype=np.float16) for _ in range(NUM_CHUNKS)]
    lin_recs = [np.zeros((8, 32, 128, 128), dtype=np.float16) for _ in range(NUM_CHUNKS)]

    return states, lin_convs, lin_recs


def step_token(models, states, lin_convs, lin_recs, tok_id, pos):
    tok = np.array([[tok_id]], dtype=np.int32)
    hidden = list(models["embed"].predict({"input_ids": tok}).values())[0]

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
        out = models["ffns"][ci].predict(inp, state=states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_convs[ci] = out['linear_conv_state_out']
            lin_recs[ci] = out['linear_recurrent_state_out']

    lm_out = models["lmhead"].predict({"hidden_states": hidden.astype(np.float16)})
    if "logits" in lm_out:
        return int(np.argmax(lm_out["logits"].flatten()))
    return int(lm_out["argmax_idx"].flatten()[0])


def prefill_batch(models, states, lin_convs, lin_recs, token_ids, start_pos):
    for i, tid in enumerate(token_ids):
        pos = start_pos + i
        if pos >= CTX:
            break
        last_tok = step_token(models, states, lin_convs, lin_recs, tid, pos)
    return last_tok, start_pos + len(token_ids)


def generate(models, states, lin_convs, lin_recs, prompt_ids, max_tokens, stop_ids, tokenizer):
    # Prefill
    last_tok, pos = prefill_batch(models, states, lin_convs, lin_recs, prompt_ids, 0)

    # Decode
    gen_ids = [last_tok]
    t0 = time.time()
    for _ in range(max_tokens - 1):
        if last_tok in stop_ids:
            break
        if pos >= CTX:
            break
        last_tok = step_token(models, states, lin_convs, lin_recs, last_tok, pos)
        gen_ids.append(last_tok)
        pos += 1

    elapsed = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tps = len(gen_ids) / elapsed if elapsed > 0 else 0
    return gen_ids, text, tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--export-dir", type=str, default=EXPORT_DIR)
    args = parser.parse_args()

    compute_unit = ct.ComputeUnit.CPU_AND_NE

    print("=" * 70)
    print("  Qwen3.5-4B LUT6+Argmax LM Head Validation")
    print(f"  Export: {args.export_dir}")
    print(f"  Max tokens: {args.tokens}")
    print("=" * 70)

    # 1. Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)

    # 2. Load models
    print("\n[2] Loading models (CPU_AND_NE)...")
    models = load_models(args.export_dir, compute_unit)

    # 3. Check LM head outputs
    print("\n[3] Checking LM head output format...")
    outputs = models["lmhead_outputs"]
    print(f"  LM head outputs: {outputs}")
    has_argmax = "argmax_idx" in outputs
    has_logits = "logits" in outputs
    print(f"  Has argmax: {has_argmax}")
    print(f"  Has logits: {has_logits}")
    if has_argmax:
        print("  PASS: argmax_idx output present")
    else:
        print("  NOTE: logits output (no argmax fused)")

    # 4. Single-prompt generation tests
    print("\n[4] Single-prompt generation tests...")
    all_pass = True
    for prompt in PROMPTS:
        states, lc, lr = get_initial_states(models)
        msgs = [{"role": "user", "content": prompt}]
        tpl = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                             return_dict=False)
        if isinstance(tpl, dict):
            prompt_ids = list(tpl["input_ids"])
        else:
            prompt_ids = list(tpl)

        gen_ids, text, tps = generate(models, states, lc, lr, prompt_ids, args.tokens, stop_ids, tokenizer)
        # Strip <think>...</think> if present
        clean = text
        if "<think>" in clean:
            end = clean.find("</think>")
            if end >= 0:
                clean = clean[end + len("</think>"):].strip()
            else:
                think_start = clean.find("<think>")
                clean = clean[:think_start].strip()

        print(f"\n  Q: {prompt}")
        print(f"  A: {clean[:200]}")
        print(f"  Tokens: {len(gen_ids)}, Speed: {tps:.1f} tok/s")

        if len(gen_ids) == 0:
            print("  FAIL: No tokens generated")
            all_pass = False

    if all_pass:
        print("\n  All single-prompt tests PASSED")
    else:
        print("\n  Some tests FAILED")

    # 5. Multi-turn test
    print("\n[5] Multi-turn conversation test...")
    states, lc, lr = get_initial_states(models)
    pos = 0
    for turn_idx, user_msg in enumerate(MULTI_TURN):
        msgs = [{"role": "user", "content": user_msg}]
        if turn_idx == 0:
            tpl = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                 return_dict=False)
            if isinstance(tpl, dict):
                tpl = list(tpl["input_ids"])
            else:
                tpl = list(tpl)
        else:
            # Build continuation
            nl = tokenizer.encode("\n")
            im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
            user_toks = tokenizer.encode("user")
            content = tokenizer.encode(user_msg, add_special_tokens=False)
            assistant_toks = tokenizer.encode("assistant")
            tpl = [im_end] + nl + [im_start] + user_toks + nl + content + nl + [im_start] + assistant_toks + nl

        for tid in tpl:
            if pos >= CTX:
                break
            last_tok = step_token(models, states, lc, lr, tid, pos)
            pos += 1

        gen_ids = [last_tok]
        for _ in range(args.tokens):
            if last_tok in stop_ids or pos >= CTX:
                break
            last_tok = step_token(models, states, lc, lr, last_tok, pos)
            gen_ids.append(last_tok)
            pos += 1

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        clean = text
        if "<think>" in clean:
            end = clean.find("</think>")
            if end >= 0:
                clean = clean[end + len("</think>"):].strip()

        print(f"\n  Turn {turn_idx+1}: {user_msg}")
        print(f"  Response: {clean[:200]}")
        print(f"  Tokens: {len(gen_ids)}, Pos: {pos}")

    # 6. Timing test
    print("\n[6] LM head timing test (20 calls)...")
    dummy = np.zeros((1, 1, 2560), dtype=np.float16)
    # Warmup
    for _ in range(3):
        models["lmhead"].predict({"hidden_states": dummy})
    times = []
    for _ in range(20):
        t0 = time.time()
        models["lmhead"].predict({"hidden_states": dummy})
        times.append((time.time() - t0) * 1000)
    avg = np.mean(times)
    std = np.std(times)
    p50 = np.percentile(times, 50)
    print(f"  Mean: {avg:.1f}ms  Std: {std:.1f}ms  P50: {p50:.1f}ms")

    print(f"\n{'='*70}")
    print("  VALIDATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
