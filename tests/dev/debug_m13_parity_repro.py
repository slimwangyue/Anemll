#!/usr/bin/env python3
"""Reproduce token-level parity between two CoreML model directories.

Default compares stable vs milestone1_3 because historical milestone1 artifacts
may be unavailable locally.
"""

import argparse
import json
import os
import time

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

CTX = 1024
NUM_CHUNKS = 4
LABEL = "LUT4"


class Engine:
    def __init__(self, model_dir: str, compute_unit):
        self.model_dir = model_dir
        self.embed = ct.models.MLModel(
            os.path.join(model_dir, "embeddings.mlpackage"), compute_units=compute_unit
        )
        self.lmhead = ct.models.MLModel(
            os.path.join(model_dir, "lm_head.mlpackage"), compute_units=compute_unit
        )
        self.ffns = [
            ct.models.MLModel(
                os.path.join(model_dir, f"ffn_{LABEL}_chunk{i}.mlpackage"),
                compute_units=compute_unit,
            )
            for i in range(NUM_CHUNKS)
        ]

        spec = self.ffns[0].get_spec()
        self.inp_map = {}
        for inp in spec.description.input:
            try:
                self.inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.has_linear = "linear_conv_state" in self.inp_map
        self.reset()

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [
                np.zeros(self.inp_map["linear_conv_state"], dtype=np.float16)
                for _ in range(NUM_CHUNKS)
            ]
            self.lin_recs = [
                np.zeros(self.inp_map["linear_recurrent_state"], dtype=np.float16)
                for _ in range(NUM_CHUNKS)
            ]
        else:
            self.lin_convs = [None] * NUM_CHUNKS
            self.lin_recs = [None] * NUM_CHUNKS

    def step(self, tok_id: int, pos: int) -> int:
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, : pos + 1] = 0

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
            if "linear_conv_state_out" in out:
                self.lin_convs[ci] = out["linear_conv_state_out"]
                self.lin_recs[ci] = out["linear_recurrent_state_out"]

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])


def run_prompt(engine: Engine, input_ids, max_tokens: int, stop_ids):
    last = None
    for i, tok in enumerate(input_ids):
        last = engine.step(tok, i)

    out = [last]
    pos = len(input_ids)
    for _ in range(max_tokens - 1):
        if pos >= CTX - 1:
            break
        nxt = engine.step(out[-1], pos)
        out.append(nxt)
        pos += 1
        if nxt in stop_ids:
            break
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare token parity between two model dirs")
    parser.add_argument("--model-a", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    parser.add_argument("--model-b", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1_3")
    parser.add_argument("--tokenizer", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    parser.add_argument("--prompt", default="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--out", default="tests/dev/m13_parity_repro_report.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    stop_ids = set()
    for sid in ["<|im_end|>", "<|endoftext|>"]:
        tids = tokenizer.encode(sid, add_special_tokens=False)
        if tids:
            stop_ids.add(tids[0])

    input_ids = tokenizer.encode(args.prompt)
    cu = ct.ComputeUnit.CPU_AND_NE

    print(f"Loading A: {args.model_a}")
    t0 = time.time()
    ea = Engine(args.model_a, cu)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"Loading B: {args.model_b}")
    t0 = time.time()
    eb = Engine(args.model_b, cu)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Running generation A...")
    ta = run_prompt(ea, input_ids, args.max_tokens, stop_ids)
    print("Running generation B...")
    tb = run_prompt(eb, input_ids, args.max_tokens, stop_ids)

    total = min(len(ta), len(tb))
    matches = sum(1 for a, b in zip(ta, tb) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(ta, tb)) if a != b), None)
    ratio = matches / total if total else 0.0

    report = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "prompt_len": len(input_ids),
        "a_len": len(ta),
        "b_len": len(tb),
        "common_len": total,
        "matches": matches,
        "match_ratio": ratio,
        "first_divergence_index": first_div,
        "a_text": tokenizer.decode(ta, skip_special_tokens=True),
        "b_text": tokenizer.decode(tb, skip_special_tokens=True),
        "a_tokens": ta,
        "b_tokens": tb,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"match: {matches}/{total} ({ratio*100:.1f}%)")
    print(f"first_divergence_index: {first_div}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
