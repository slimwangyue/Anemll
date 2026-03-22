#!/usr/bin/env python3
"""Parity test: dynamic KV (milestone1_2) vs baseline (milestone1).

Uses subprocess isolation to avoid memory pressure / segfaults.

Usage:
    python tests/dev/test_dynamic_kv_parity.py           # runs both phases
    python tests/dev/test_dynamic_kv_parity.py baseline   # phase 1 only
    python tests/dev/test_dynamic_kv_parity.py dynamic    # phase 2 only (needs baseline first)
"""
import sys, os, time, json, gc, subprocess

BASELINE_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1"
DYNAMIC_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_2"
HF_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 1024
NUM_CHUNKS = 4
MAX_GEN = 30
RESULT_FILE = "/tmp/kv_parity_results.json"
PROMPT = "What is 2+2?"


def get_prompt_tokens():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, use_fast=False)
    messages = [{"role": "user", "content": PROMPT}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=True)
    if hasattr(input_ids, 'input_ids'):
        input_ids = input_ids.input_ids
    return input_ids[0].tolist(), tokenizer


def run_baseline():
    import numpy as np
    import coremltools as ct

    print("=" * 60)
    print("PHASE 1: BASELINE (milestone1, CPU_AND_NE)")
    print("=" * 60)

    prompt_tokens, tokenizer = get_prompt_tokens()
    print(f"Prompt: {len(prompt_tokens)} tokens")

    cu = ct.ComputeUnit.CPU_AND_NE
    embed = ct.models.MLModel(os.path.join(BASELINE_DIR, "embeddings.mlpackage"), compute_units=cu)
    lmhead = ct.models.MLModel(os.path.join(BASELINE_DIR, "lm_head.mlpackage"), compute_units=cu)
    ffns = []
    for i in range(NUM_CHUNKS):
        print(f"  Loading baseline ffn chunk {i}...")
        ffns.append(ct.models.MLModel(os.path.join(BASELINE_DIR, f"ffn_LUT4_chunk{i}.mlpackage"), compute_units=cu))

    spec = ffns[0].get_spec()
    inp_map = {}
    for inp in spec.description.input:
        try:
            inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
        except Exception:
            pass

    states = [m.make_state() for m in ffns]
    lin_convs = [np.zeros(inp_map['linear_conv_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
    lin_recs = [np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]

    def step(tok_id, pos):
        nonlocal lin_convs, lin_recs
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
        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    t0 = time.time()
    pos = 0
    for tok_id in prompt_tokens:
        last_next = step(tok_id, pos)
        pos += 1
    t_pf = time.time() - t0
    print(f"  Prefill: {len(prompt_tokens)} tok in {t_pf:.1f}s ({len(prompt_tokens)/t_pf:.0f} tok/s)")

    tokens = [last_next]
    t_dec = time.time()
    for _ in range(MAX_GEN - 1):
        if pos >= CTX - 1:
            break
        nxt = step(tokens[-1], pos)
        pos += 1
        tokens.append(nxt)
    t_decode = time.time() - t_dec
    print(f"  Decode: {len(tokens)} tok in {t_decode:.1f}s ({len(tokens)/t_decode:.1f} tok/s)")

    text = tokenizer.decode(tokens, skip_special_tokens=False)
    print(f"  Output: {text[:200]}")

    with open(RESULT_FILE, "w") as f:
        json.dump({"tokens": tokens, "prompt_tokens": prompt_tokens,
                    "t_pf": t_pf, "t_dec": t_decode}, f)
    print(f"  Saved baseline tokens to {RESULT_FILE}")
    return 0


def run_dynamic():
    import numpy as np
    import coremltools as ct

    print("=" * 60)
    print("PHASE 2: DYNAMIC KV (milestone1_2, CPU_AND_GPU)")
    print("=" * 60)

    if not os.path.exists(RESULT_FILE):
        print("ERROR: No baseline results found. Run 'baseline' phase first.")
        return 1

    with open(RESULT_FILE) as f:
        baseline = json.load(f)
    baseline_tokens = baseline["tokens"]
    prompt_tokens = baseline["prompt_tokens"]

    _, tokenizer = get_prompt_tokens()
    print(f"Prompt: {len(prompt_tokens)} tokens, baseline generated: {len(baseline_tokens)} tokens")

    cu = ct.ComputeUnit.CPU_AND_GPU
    embed = ct.models.MLModel(os.path.join(DYNAMIC_DIR, "embeddings.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
    lmhead = ct.models.MLModel(os.path.join(DYNAMIC_DIR, "lm_head.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
    ffns = []
    for i in range(NUM_CHUNKS):
        print(f"  Loading dynamic ffn chunk {i} (CPU_AND_GPU)...")
        ffns.append(ct.models.MLModel(os.path.join(DYNAMIC_DIR, f"ffn_LUT4_chunk{i}.mlpackage"), compute_units=cu))

    spec = ffns[0].get_spec()
    inp_map = {}
    for inp in spec.description.input:
        try:
            inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
        except Exception:
            pass

    states = [m.make_state() for m in ffns]
    lin_convs = [np.zeros(inp_map['linear_conv_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
    lin_recs = [np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]

    def step(tok_id, pos):
        nonlocal lin_convs, lin_recs
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        kv_write_end = np.zeros((pos + 1,), dtype=np.int32)
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
                "kv_write_end": kv_write_end,
            }
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    t0 = time.time()
    pos = 0
    for tok_id in prompt_tokens:
        last_next = step(tok_id, pos)
        pos += 1
    t_pf = time.time() - t0
    print(f"  Prefill: {len(prompt_tokens)} tok in {t_pf:.1f}s ({len(prompt_tokens)/t_pf:.0f} tok/s)")

    tokens = [last_next]
    t_dec = time.time()
    for _ in range(MAX_GEN - 1):
        if pos >= CTX - 1:
            break
        nxt = step(tokens[-1], pos)
        pos += 1
        tokens.append(nxt)
    t_decode = time.time() - t_dec
    print(f"  Decode: {len(tokens)} tok in {t_decode:.1f}s ({len(tokens)/t_decode:.1f} tok/s)")

    text = tokenizer.decode(tokens, skip_special_tokens=False)
    print(f"  Output: {text[:200]}")

    # Compare
    print(f"\n{'='*60}")
    print("PARITY CHECK")
    print(f"{'='*60}")
    baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=False)
    print(f"  Baseline: {baseline_text[:200]}")
    print(f"  Dynamic:  {text[:200]}")

    match = tokens == baseline_tokens
    if match:
        print(f"\n  PASS: 100% token match ({len(tokens)} tokens)")
    else:
        for i in range(min(len(tokens), len(baseline_tokens))):
            if tokens[i] != baseline_tokens[i]:
                print(f"\n  MISMATCH at token {i}")
                print(f"    baseline={baseline_tokens[i]} ({tokenizer.decode([baseline_tokens[i]])})")
                print(f"    dynamic ={tokens[i]} ({tokenizer.decode([tokens[i]])})")
                break
        n_match = sum(1 for a, b in zip(tokens, baseline_tokens) if a == b)
        print(f"    {n_match}/{min(len(tokens), len(baseline_tokens))} matched")

    print(f"\n  Speed comparison:")
    print(f"    Baseline decode: {baseline['t_dec']:.1f}s ({len(baseline_tokens)/baseline['t_dec']:.1f} tok/s)")
    print(f"    Dynamic  decode: {t_decode:.1f}s ({len(tokens)/t_decode:.1f} tok/s)")

    return 0 if match else 1


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "baseline":
        return run_baseline()
    elif mode == "dynamic":
        return run_dynamic()
    elif mode == "both":
        print("Running two-phase parity test (separate processes)...\n")
        script = os.path.abspath(__file__)
        r1 = subprocess.run([sys.executable, script, "baseline"], timeout=300)
        if r1.returncode != 0:
            print("Baseline phase failed!")
            return 1
        print("\n--- Waiting 5s for memory cleanup ---\n")
        time.sleep(5)
        r2 = subprocess.run([sys.executable, script, "dynamic"], timeout=600)
        return r2.returncode
    else:
        print(f"Unknown mode: {mode}. Use baseline/dynamic/both.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
