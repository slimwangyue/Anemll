#!/usr/bin/env python3
"""Parity test: batch prefill vs token-by-token for Qwen3.5-4B.

Run in two phases to avoid OOM/segfault from loading too many models:
  python tests/dev/qwen35_prefill_parity_test.py seq   # baseline
  python tests/dev/qwen35_prefill_parity_test.py pf    # batch prefill + compare

Or run both sequentially (models freed between phases):
  python tests/dev/qwen35_prefill_parity_test.py both
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time, gc, json
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1"
COMBINED_DIR = os.path.join(MODEL_DIR, "combined_LUT4_dedup")
HF_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
BATCH_SIZE = 256
CTX = 1024
NUM_CHUNKS = 4
MAX_GEN = 20
CU = ct.ComputeUnit.CPU_AND_NE
RESULT_FILE = "/tmp/prefill_parity_results.json"


def build_long_prompt(tokenizer):
    msg = "Explain the following concepts in great detail: " + ", ".join([f"concept_{i}" for i in range(200)])
    messages = [{"role": "user", "content": msg}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=True)
    if hasattr(input_ids, 'input_ids'):
        input_ids = input_ids.input_ids
    tokens = input_ids[0].tolist()
    # Truncate to fit within CTX - MAX_GEN
    max_prompt = CTX - MAX_GEN - 10
    if len(tokens) > max_prompt:
        tokens = tokens[:max_prompt]
    return tokens


def run_sequential(prompt_tokens, max_gen):
    print("\n[seq] Loading separate FFN models...")
    embed = ct.models.MLModel(os.path.join(MODEL_DIR, "embeddings.mlpackage"), compute_units=CU)
    lmhead = ct.models.MLModel(os.path.join(MODEL_DIR, "lm_head.mlpackage"), compute_units=CU)
    ffns = []
    for ci in range(NUM_CHUNKS):
        ffns.append(ct.models.MLModel(
            os.path.join(MODEL_DIR, f"ffn_LUT4_chunk{ci}.mlpackage"), compute_units=CU))

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

    print(f"[seq] Processing {len(prompt_tokens)} tokens one-by-one...")
    t0 = time.time()
    pos = 0
    for tok_id in prompt_tokens:
        last_next = step(tok_id, pos)
        pos += 1
    t_pf = time.time() - t0
    print(f"[seq] Prefill: {len(prompt_tokens)} tokens in {t_pf*1000:.0f}ms ({len(prompt_tokens)/t_pf:.0f} tok/s)")

    tokens = [last_next]
    t_dec = time.time()
    for _ in range(max_gen - 1):
        if pos >= CTX - 1:
            break
        nxt = step(tokens[-1], pos)
        pos += 1
        tokens.append(nxt)
    t_decode = time.time() - t_dec
    print(f"[seq] Decode: {len(tokens)} tokens in {t_decode*1000:.0f}ms")

    del embed, lmhead, ffns, states
    gc.collect()
    return tokens, t_pf, t_decode


def run_batch_prefill(prompt_tokens, max_gen):
    print("\n[pf] Loading combined dedup models...")
    embed = ct.models.MLModel(os.path.join(MODEL_DIR, "embeddings.mlpackage"), compute_units=CU)
    lmhead = ct.models.MLModel(os.path.join(MODEL_DIR, "lm_head.mlpackage"), compute_units=CU)

    decode_ffns = []
    prefill_ffns = []
    for ci in range(NUM_CHUNKS):
        p = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
        print(f"  chunk{ci} (infer)...")
        decode_ffns.append(ct.models.MLModel(p, compute_units=CU, function_name="infer"))
    for ci in range(NUM_CHUNKS):
        p = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
        print(f"  chunk{ci} (prefill)...")
        prefill_ffns.append(ct.models.MLModel(p, compute_units=CU, function_name="prefill"))

    states = [m.make_state() for m in decode_ffns]

    spec = decode_ffns[0].get_spec()
    inp_map = {}
    for fn in spec.description.functions:
        if fn.name == "infer":
            for inp in fn.input:
                try:
                    inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
                except Exception:
                    pass
            break

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
            out = decode_ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    n = len(prompt_tokens)
    pos = 0

    # Batch prefill first BATCH_SIZE tokens
    print(f"[pf] Batch prefill: {BATCH_SIZE} tokens...")
    t0 = time.time()
    batch = prompt_tokens[:BATCH_SIZE]
    input_ids = np.array([batch], dtype=np.int32)
    hidden = list(embed.predict({"input_ids": input_ids}).values())[0]

    mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
    for i in range(BATCH_SIZE):
        mask[0, 0, i, :i + 1] = 0

    pos_ids = np.arange(BATCH_SIZE, dtype=np.int32)
    current_pos = np.array([0], dtype=np.int32)

    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": current_pos,
            "linear_conv_state": lin_convs[ci],
            "linear_recurrent_state": lin_recs[ci],
        }
        out = prefill_ffns[ci].predict(inp, state=states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_convs[ci] = out['linear_conv_state_out']
            lin_recs[ci] = out['linear_recurrent_state_out']

    lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if "logits" in lm_out:
        last_next = int(np.argmax(lm_out["logits"].flatten()))
    else:
        last_next = int(lm_out["argmax_idx"].flatten()[0])
    pos = BATCH_SIZE
    t_batch = time.time() - t0
    print(f"[pf] Batch: {BATCH_SIZE} tokens in {t_batch*1000:.0f}ms ({BATCH_SIZE/t_batch:.0f} tok/s)")

    # Sequential remaining
    t_seq = time.time()
    for tok_id in prompt_tokens[BATCH_SIZE:]:
        last_next = step(tok_id, pos)
        pos += 1
    t_seq_elapsed = time.time() - t_seq
    n_seq = n - BATCH_SIZE
    t_pf_total = time.time() - t0
    print(f"[pf] Sequential: {n_seq} tokens in {t_seq_elapsed*1000:.0f}ms")
    print(f"[pf] Total prefill: {n} tokens in {t_pf_total*1000:.0f}ms")

    # Generate
    tokens = [last_next]
    t_dec = time.time()
    for _ in range(max_gen - 1):
        if pos >= CTX - 1:
            break
        nxt = step(tokens[-1], pos)
        pos += 1
        tokens.append(nxt)
    t_decode = time.time() - t_dec
    print(f"[pf] Decode: {len(tokens)} tokens in {t_decode*1000:.0f}ms")

    del embed, lmhead, decode_ffns, prefill_ffns, states
    gc.collect()
    return tokens, t_pf_total, t_decode


def main():
    print("=" * 70)
    print("Qwen3.5-4B Batch Prefill Parity Test")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, use_fast=False)
    prompt_tokens = build_long_prompt(tokenizer)
    print(f"Prompt: {len(prompt_tokens)} tokens (>= {BATCH_SIZE} required)")
    assert len(prompt_tokens) >= BATCH_SIZE, f"Need >= {BATCH_SIZE} tokens"

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("seq", "both"):
        seq_tokens, t_pf_seq, t_dec_seq = run_sequential(prompt_tokens, MAX_GEN)
        seq_text = tokenizer.decode(seq_tokens, skip_special_tokens=False)
        print(f"\n[seq] Generated: {seq_text[:120]}...")
        with open(RESULT_FILE, "w") as f:
            json.dump({"seq_tokens": seq_tokens, "t_pf": t_pf_seq, "t_dec": t_dec_seq}, f)
        gc.collect()
        import time as _t; _t.sleep(2)  # let CoreML release resources

    if mode in ("pf", "both"):
        pf_tokens, t_pf_pf, t_dec_pf = run_batch_prefill(prompt_tokens, MAX_GEN)
        pf_text = tokenizer.decode(pf_tokens, skip_special_tokens=False)
        print(f"\n[pf] Generated: {pf_text[:120]}...")

        if mode == "both":
            seq_tokens_compare = seq_tokens
        elif os.path.exists(RESULT_FILE):
            with open(RESULT_FILE) as f:
                data = json.load(f)
            seq_tokens_compare = data["seq_tokens"]
        else:
            print("No sequential results found. Run with 'seq' first.")
            return 1

        print(f"\n{'='*50}")
        print("PARITY CHECK")
        print(f"{'='*50}")
        match = pf_tokens == seq_tokens_compare
        print(f"  Sequential: {seq_tokens_compare[:10]}")
        print(f"  Prefill:    {pf_tokens[:10]}")
        if match:
            print(f"\n  PASS: 100% token match ({len(pf_tokens)} tokens)")
        else:
            for i in range(min(len(pf_tokens), len(seq_tokens_compare))):
                if pf_tokens[i] != seq_tokens_compare[i]:
                    print(f"\n  FAIL: First mismatch at token {i}")
                    print(f"    seq={seq_tokens_compare[i]} ({tokenizer.decode([seq_tokens_compare[i]])})")
                    print(f"    pf ={pf_tokens[i]} ({tokenizer.decode([pf_tokens[i]])})")
                    break
            n_match = sum(1 for a, b in zip(pf_tokens, seq_tokens_compare) if a == b)
            print(f"    Matched {n_match}/{min(len(pf_tokens), len(seq_tokens_compare))}")
        return 0 if match else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
