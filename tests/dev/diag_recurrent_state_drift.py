#!/usr/bin/env python3
"""Diagnose recurrent state drift in CoreML Qwen3.5 linear attention layers.

Runs the CoreML model token-by-token and records:
  - Recurrent state statistics (L2 norm, max abs) per step
  - Top-5 logit probabilities per step
  - Token entropy per step

This reveals whether the FP16 recurrent state degrades over time, causing
the model to lock into repetitive predictions.

Usage:
  cd /Users/yw68/Anemll
  source .venv/bin/activate
  python tests/dev/diag_recurrent_state_drift.py
"""
import sys, os, time
import numpy as np

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))

from config import (
    DEFAULT_OUTPUT, DEFAULT_HF_MODEL, CTX, NUM_CHUNKS,
    CHUNK_RANGES, LUT_BITS,
)
import coremltools as ct
from transformers import AutoTokenizer

MODEL_DIR = os.path.join(REPO_ROOT, "qwen3_5_flll_9chunk")
COMBINED_DIR = os.path.join(MODEL_DIR, f"combined_LUT{LUT_BITS}_dedup")
HF_PATH = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
COMPUTE_UNIT = ct.ComputeUnit.CPU_AND_NE

MAX_TOKENS = 120  # Generate enough to see degeneration
PROMPT = "What is a stack in computer science? Explain in detail."


def find_model(base_dir, name):
    for ext in [".mlpackage", ".mlmodelc"]:
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def load_model(path, function_name=None):
    kwargs = {}
    if function_name:
        kwargs["function_name"] = function_name
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, COMPUTE_UNIT)
    return ct.models.MLModel(path, compute_units=COMPUTE_UNIT, **kwargs)


def softmax(x):
    x = x - x.max()
    e = np.exp(x.astype(np.float32))
    return e / e.sum()


def entropy(probs):
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log2(probs))


def main():
    print("=" * 70)
    print("Qwen3.5-4B Recurrent State Drift Diagnostic")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, trust_remote_code=True)

    # Encode prompt with chat template
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Prompt: {PROMPT}")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print()

    # Load models
    print("Loading embed...")
    embed_path = find_model(MODEL_DIR, "embeddings")
    embed = load_model(embed_path)

    print("Loading lm_head...")
    lm_head_path = find_model(MODEL_DIR, "lm_head_logits")
    lm_head = load_model(lm_head_path)

    print(f"Loading {NUM_CHUNKS} FFN chunks (infer)...")
    ffns = []
    inp_maps = []
    for ci in range(NUM_CHUNKS):
        path = find_model(COMBINED_DIR, f"chunk{ci}")
        print(f"  chunk {ci}...", end="", flush=True)
        t0 = time.time()
        m = load_model(path, function_name="infer")
        print(f" {time.time()-t0:.1f}s")
        ffns.append(m)
        spec = m.get_spec()
        imap = {}
        fn_spec = None
        for fn in spec.description.functions:
            if fn.name == "infer":
                fn_spec = fn
                break
        if fn_spec:
            for inp in fn_spec.input:
                try:
                    imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                except:
                    pass
        else:
            for inp in spec.description.input:
                try:
                    imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                except:
                    pass
        inp_maps.append(imap)

    has_linear = 'linear_conv_state' in inp_maps[0]
    print(f"Has linear attention: {has_linear}")

    # Initialize states
    states = [m.make_state() for m in ffns]
    if has_linear:
        lin_convs = [np.zeros(inp_maps[ci]['linear_conv_state'], dtype=np.float16)
                     for ci in range(NUM_CHUNKS)]
        lin_recs = [np.zeros(inp_maps[ci]['linear_recurrent_state'], dtype=np.float16)
                    for ci in range(NUM_CHUNKS)]
    else:
        lin_convs = [None] * NUM_CHUNKS
        lin_recs = [None] * NUM_CHUNKS

    # Helper: extract argmax from lm_head output
    def argmax_from_lm(lm_out):
        keys = sorted(lm_out.keys())
        if len(keys) == 1:
            logits = lm_out[keys[0]].flatten()
        else:
            # Multi-split lm_head: sort by numeric suffix
            numeric_keys = sorted(keys, key=lambda k: int(''.join(filter(str.isdigit, k)) or 0))
            logits = np.concatenate([lm_out[k].flatten() for k in numeric_keys])
        return logits

    # Step function that also returns recurrent state stats
    def step(tok_id, pos):
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0

        rec_stats = []
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if lin_convs[ci] is not None:
                inp["linear_conv_state"] = lin_convs[ci]
                inp["linear_recurrent_state"] = lin_recs[ci]
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
                # Record recurrent state statistics
                rec = lin_recs[ci].astype(np.float32)
                rec_stats.append({
                    'chunk': ci,
                    'l2_norm': np.sqrt(np.sum(rec ** 2)),
                    'max_abs': np.max(np.abs(rec)),
                    'mean': np.mean(rec),
                    'std': np.std(rec),
                    'has_nan': bool(np.any(np.isnan(rec))),
                    'has_inf': bool(np.any(np.isinf(rec))),
                })

        lm_out = lm_head.predict({"hidden_states": hidden.astype(np.float16)})
        logits = argmax_from_lm(lm_out)
        return logits, rec_stats

    # Run prefill
    print(f"\n{'='*70}")
    print("Running prefill + decode...")
    print(f"{'='*70}\n")

    # Prefill phase (skip lm_head for all except last token)
    for i, tid in enumerate(prompt_tokens[:-1]):
        tok = np.array([[tid]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :i + 1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([i], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([i], dtype=np.int32),
            }
            if lin_convs[ci] is not None:
                inp["linear_conv_state"] = lin_convs[ci]
                inp["linear_recurrent_state"] = lin_recs[ci]
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']

    print(f"Prefill done ({len(prompt_tokens)-1} tokens)")

    # Last prefill token + first decode token
    pos = len(prompt_tokens) - 1
    logits, rec_stats = step(prompt_tokens[-1], pos)
    probs = softmax(logits)
    next_tok = int(np.argmax(probs))
    pos += 1

    stop_ids = {248044, 248046}
    generated = [next_tok]

    # Header
    print(f"\n{'Step':>4} {'Token':>8} {'Text':<20} {'Top1Prob':>9} {'Entropy':>8} ", end="")
    if rec_stats:
        print(f"{'RecNorm(c0)':>12} {'RecMax(c0)':>11} {'RecNorm(cN)':>12} {'RecMax(cN)':>11}", end="")
    print()
    print("-" * 120)

    # Print first token
    txt = tokenizer.decode([next_tok])
    top1_prob = float(probs[next_tok])
    ent = entropy(probs)
    line = f"{0:4d} {next_tok:8d} {repr(txt):<20} {top1_prob:9.4f} {ent:8.2f} "
    if rec_stats:
        line += f"{rec_stats[0]['l2_norm']:12.4f} {rec_stats[0]['max_abs']:11.6f} "
        line += f"{rec_stats[-1]['l2_norm']:12.4f} {rec_stats[-1]['max_abs']:11.6f}"
    print(line)

    # Decode loop
    for gi in range(1, MAX_TOKENS):
        if next_tok in stop_ids or pos >= CTX - 1:
            break

        logits, rec_stats = step(next_tok, pos)
        probs = softmax(logits)
        next_tok = int(np.argmax(probs))
        generated.append(next_tok)

        txt = tokenizer.decode([next_tok])
        top1_prob = float(probs[next_tok])
        ent = entropy(probs)

        line = f"{gi:4d} {next_tok:8d} {repr(txt):<20} {top1_prob:9.4f} {ent:8.2f} "
        if rec_stats:
            line += f"{rec_stats[0]['l2_norm']:12.4f} {rec_stats[0]['max_abs']:11.6f} "
            line += f"{rec_stats[-1]['l2_norm']:12.4f} {rec_stats[-1]['max_abs']:11.6f}"
        print(line)

        pos += 1

    # Summary
    full_text = tokenizer.decode(generated)
    print(f"\n{'='*70}")
    print(f"GENERATED TEXT ({len(generated)} tokens):")
    print(f"{'='*70}")
    print(full_text)
    print(f"\n{'='*70}")

    # Final recurrent state analysis
    if has_linear:
        print("\nFinal Recurrent State Summary:")
        for ci in range(NUM_CHUNKS):
            rec = lin_recs[ci].astype(np.float32)
            print(f"  Chunk {ci}: L2={np.sqrt(np.sum(rec**2)):.4f}, "
                  f"max={np.max(np.abs(rec)):.6f}, "
                  f"mean={np.mean(rec):.6f}, std={np.std(rec):.6f}, "
                  f"NaN={np.any(np.isnan(rec))}, Inf={np.any(np.isinf(rec))}")


if __name__ == "__main__":
    main()
