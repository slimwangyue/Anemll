#!/usr/bin/env python3
"""Hybrid decode test: chunks 0,1 on ANE, chunks 2,3 on CPU_AND_GPU.
Compare speed with all-GPU and all-ANE (baseline) approaches.
"""
import os, sys, time, json
import numpy as np
import coremltools as ct

MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_2"
HF_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 1024
NUM_CHUNKS = 4
MAX_GEN = 30
RESULT_FILE = "/tmp/kv_parity_results.json"

# Hybrid: ANE for chunks 0,1; GPU for chunks 2,3
COMPUTE_UNITS = [
    ct.ComputeUnit.CPU_AND_NE,   # chunk 0
    ct.ComputeUnit.CPU_AND_NE,   # chunk 1
    ct.ComputeUnit.CPU_AND_GPU,  # chunk 2
    ct.ComputeUnit.CPU_AND_GPU,  # chunk 3
]

def main():
    from transformers import AutoTokenizer

    # Load baseline results for comparison
    with open(RESULT_FILE) as f:
        baseline = json.load(f)
    baseline_tokens = baseline["tokens"]
    prompt_tokens = baseline["prompt_tokens"]

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, use_fast=False)
    print(f"Prompt: {len(prompt_tokens)} tokens")

    # Load models with hybrid compute units
    embed = ct.models.MLModel(
        os.path.join(MODEL_DIR, "embeddings.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_NE)
    lmhead = ct.models.MLModel(
        os.path.join(MODEL_DIR, "lm_head.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_NE)

    ffns = []
    for i in range(NUM_CHUNKS):
        cu = COMPUTE_UNITS[i]
        cu_name = "ANE" if cu == ct.ComputeUnit.CPU_AND_NE else "GPU"
        print(f"  Loading chunk {i} ({cu_name})...")
        ffns.append(ct.models.MLModel(
            os.path.join(MODEL_DIR, f"ffn_LUT4_chunk{i}.mlpackage"),
            compute_units=cu))

    spec = ffns[0].get_spec()
    inp_map = {}
    for inp in spec.description.input:
        try:
            inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
        except Exception:
            pass

    states = [m.make_state() for m in ffns]
    lin_convs = [np.zeros(inp_map["linear_conv_state"], dtype=np.float16) for _ in range(NUM_CHUNKS)]
    lin_recs = [np.zeros(inp_map["linear_recurrent_state"], dtype=np.float16) for _ in range(NUM_CHUNKS)]

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
            if "linear_conv_state_out" in out:
                lin_convs[ci] = out["linear_conv_state_out"]
                lin_recs[ci] = out["linear_recurrent_state_out"]
        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    # Prefill
    t0 = time.time()
    pos = 0
    for tok_id in prompt_tokens:
        last_next = step(tok_id, pos)
        pos += 1
    t_pf = time.time() - t0
    print(f"  Prefill: {len(prompt_tokens)} tok in {t_pf:.1f}s ({len(prompt_tokens)/t_pf:.0f} tok/s)")

    # Decode
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

    # Parity check
    match = tokens == baseline_tokens
    n_match = sum(1 for a, b in zip(tokens, baseline_tokens) if a == b)
    print(f"\n  Parity: {n_match}/{min(len(tokens), len(baseline_tokens))} tokens match")
    if match:
        print("  PASS: 100% match")

    print(f"\n  Speed comparison:")
    print(f"    Baseline (ANE):    {baseline['t_dec']:.1f}s ({len(baseline_tokens)/baseline['t_dec']:.1f} tok/s)")
    print(f"    Hybrid (ANE+GPU): {t_decode:.1f}s ({len(tokens)/t_decode:.1f} tok/s)")


if __name__ == "__main__":
    main()
