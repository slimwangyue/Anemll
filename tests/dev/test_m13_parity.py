#!/usr/bin/env python3
"""Parity check: milestone1 (static) vs milestone1_3 (tensor-value slice).

Runs the same prompt through both model sets and compares token-by-token.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

M1_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1"
M13_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"
TOKENIZER_DIR = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 1024
NUM_CHUNKS = 4
LABEL = "LUT4"
MAX_TOKENS = 30


class Engine:
    def __init__(self, model_dir, cu):
        self.embed = ct.models.MLModel(
            os.path.join(model_dir, "embeddings.mlpackage"), compute_units=cu)
        self.lmhead = ct.models.MLModel(
            os.path.join(model_dir, "lm_head.mlpackage"), compute_units=cu)
        self.ffns = []
        for ci in range(NUM_CHUNKS):
            m = ct.models.MLModel(
                os.path.join(model_dir, f"ffn_{LABEL}_chunk{ci}.mlpackage"),
                compute_units=cu)
            self.ffns.append(m)

        spec = self.ffns[0].get_spec()
        self.inp_map = {}
        for inp in spec.description.input:
            try:
                self.inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.has_linear = 'linear_conv_state' in self.inp_map
        self.reset()

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
                              for _ in range(NUM_CHUNKS)]
            self.lin_recs = [np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
                             for _ in range(NUM_CHUNKS)]
        else:
            self.lin_convs = [None] * NUM_CHUNKS
            self.lin_recs = [None] * NUM_CHUNKS

    def step(self, tok_id, pos):
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if self.lin_convs[ci] is not None:
                inp["linear_conv_state"] = self.lin_convs[ci]
                inp["linear_recurrent_state"] = self.lin_recs[ci]
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])


def run_prompt(engine, input_ids, max_tokens, stop_ids):
    """Prefill one-by-one, then decode."""
    last = None
    for i, tid in enumerate(input_ids):
        last = engine.step(tid, i)

    generated = [last]
    pos = len(input_ids)
    for gi in range(max_tokens - 1):
        if pos >= CTX - 1:
            break
        nxt = engine.step(generated[-1], pos)
        generated.append(nxt)
        pos += 1
        if nxt in stop_ids:
            break
    return generated


def main():
    cu = ct.ComputeUnit.CPU_AND_NE
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=False)
    stop_ids = set()
    for sid in ["<|im_end|>", "<|endoftext|>"]:
        tids = tokenizer.encode(sid, add_special_tokens=False)
        if tids:
            stop_ids.add(tids[0])

    prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(prompt)

    print("=" * 70)
    print("  Parity: Milestone 1 (static) vs Milestone 1.3 (tensor-value slice)")
    print(f"  Prompt: {len(input_ids)} tokens | Max decode: {MAX_TOKENS}")
    print("=" * 70)

    # Load milestone 1
    print("\nLoading Milestone 1 models (static)...")
    t0 = time.time()
    e1 = Engine(M1_DIR, cu)
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Load milestone 1.3
    print("\nLoading Milestone 1.3 models (tensor-value slice)...")
    t0 = time.time()
    e13 = Engine(M13_DIR, cu)
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Run M1
    print("\nGenerating with M1 (static)...")
    t0 = time.time()
    m1_tokens = run_prompt(e1, input_ids, MAX_TOKENS, stop_ids)
    t1 = time.time() - t0
    m1_text = tokenizer.decode(m1_tokens, skip_special_tokens=True)
    print(f"  {len(m1_tokens)} tokens in {t1:.1f}s")
    print(f"  Output: {m1_text}")

    # Run M1.3
    print("\nGenerating with M1.3 (tensor-value slice)...")
    t0 = time.time()
    m13_tokens = run_prompt(e13, input_ids, MAX_TOKENS, stop_ids)
    t13 = time.time() - t0
    m13_text = tokenizer.decode(m13_tokens, skip_special_tokens=True)
    print(f"  {len(m13_tokens)} tokens in {t13:.1f}s")
    print(f"  Output: {m13_text}")

    # Compare
    max_len = max(len(m1_tokens), len(m13_tokens))
    matches = 0
    diverge_pos = -1
    for i in range(min(len(m1_tokens), len(m13_tokens))):
        if m1_tokens[i] == m13_tokens[i]:
            matches += 1
        elif diverge_pos < 0:
            diverge_pos = i
    total = min(len(m1_tokens), len(m13_tokens))
    pct = 100 * matches / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"  PARITY RESULT")
    print(f"{'='*70}")
    print(f"  Token match: {matches}/{total} ({pct:.1f}%)")
    if diverge_pos >= 0:
        print(f"  First divergence at position {diverge_pos}:")
        print(f"    M1:   token {m1_tokens[diverge_pos]} = '{tokenizer.decode([m1_tokens[diverge_pos]])}'")
        print(f"    M1.3: token {m13_tokens[diverge_pos]} = '{tokenizer.decode([m13_tokens[diverge_pos]])}'")
    else:
        print(f"  100% MATCH!")

    # Token-by-token comparison
    print(f"\n  Token-by-token:")
    for i in range(total):
        mark = "✓" if m1_tokens[i] == m13_tokens[i] else "✗"
        t1_w = tokenizer.decode([m1_tokens[i]])
        t13_w = tokenizer.decode([m13_tokens[i]])
        if m1_tokens[i] == m13_tokens[i]:
            print(f"    {i:3d}: {mark} {m1_tokens[i]:6d} '{t1_w}'")
        else:
            print(f"    {i:3d}: {mark} M1={m1_tokens[i]:6d} '{t1_w}' vs M1.3={m13_tokens[i]:6d} '{t13_w}'")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
