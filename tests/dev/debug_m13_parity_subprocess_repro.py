#!/usr/bin/env python3
"""Subprocess-isolated token parity repro between two model directories."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKER = r'''
import json, os, sys, time
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

model_dir = sys.argv[1]
out_file = sys.argv[2]
tokenizer_dir = sys.argv[3]
prompt = sys.argv[4]
max_tokens = int(sys.argv[5])

CTX = 1024
NUM_CHUNKS = 4
LABEL = "LUT4"

cu = ct.ComputeUnit.CPU_AND_NE
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

embed = ct.models.MLModel(os.path.join(model_dir, "embeddings.mlpackage"), compute_units=cu)
lmhead = ct.models.MLModel(os.path.join(model_dir, "lm_head.mlpackage"), compute_units=cu)
ffns = [
    ct.models.MLModel(os.path.join(model_dir, f"ffn_{LABEL}_chunk{i}.mlpackage"), compute_units=cu)
    for i in range(NUM_CHUNKS)
]

spec = ffns[0].get_spec()
inp_map = {}
for inp in spec.description.input:
    try:
        inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
    except Exception:
        pass
has_linear = "linear_conv_state" in inp_map
states = [m.make_state() for m in ffns]
if has_linear:
    lin_convs = [np.zeros(inp_map["linear_conv_state"], dtype=np.float16) for _ in range(NUM_CHUNKS)]
    lin_recs = [np.zeros(inp_map["linear_recurrent_state"], dtype=np.float16) for _ in range(NUM_CHUNKS)]
else:
    lin_convs = [None] * NUM_CHUNKS
    lin_recs = [None] * NUM_CHUNKS

stop_ids = set()
for sid in ["<|im_end|>", "<|endoftext|>"]:
    ids = tok.encode(sid, add_special_tokens=False)
    if ids:
        stop_ids.add(ids[0])

def step(tok_id, pos):
    t = np.array([[tok_id]], dtype=np.int32)
    hidden = list(embed.predict({"input_ids": t}).values())[0]
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :pos+1] = 0
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
        if "linear_conv_state_out" in out:
            lin_convs[ci] = out["linear_conv_state_out"]
            lin_recs[ci] = out["linear_recurrent_state_out"]
    lm = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if "logits" in lm:
        return int(np.argmax(lm["logits"].flatten()))
    return int(lm["argmax_idx"].flatten()[0])

ids = tok.encode(prompt)
last = None
for i, tid in enumerate(ids):
    last = step(tid, i)

gen = [last]
pos = len(ids)
for _ in range(max_tokens - 1):
    if pos >= CTX - 1:
        break
    nxt = step(gen[-1], pos)
    gen.append(nxt)
    pos += 1
    if nxt in stop_ids:
        break

with open(out_file, "w", encoding="utf-8") as f:
    json.dump({
        "tokens": gen,
        "text": tok.decode(gen, skip_special_tokens=True),
    }, f, ensure_ascii=False)
'''


def run_worker(python_bin: str, model_dir: str, out_file: str, tokenizer: str, prompt: str, max_tokens: int):
    worker_path = "/tmp/_m13_parity_worker.py"
    Path(worker_path).write_text(WORKER, encoding="utf-8")
    cmd = [python_bin, worker_path, model_dir, out_file, tokenizer, prompt, str(max_tokens)]
    return subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default="/Users/yw68/Anemll/.venv/bin/python")
    parser.add_argument("--model-a", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    parser.add_argument("--model-b", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1_3")
    parser.add_argument("--tokenizer", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    parser.add_argument("--prompt", default="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--out", default="tests/dev/m13_parity_subprocess_repro_report.json")
    args = parser.parse_args()

    a_json = "/tmp/m13_parity_a.json"
    b_json = "/tmp/m13_parity_b.json"

    ra = run_worker(args.python_bin, args.model_a, a_json, args.tokenizer, args.prompt, args.max_tokens)
    if ra.returncode != 0:
        print(f"A worker failed: {ra.returncode}")
        sys.exit(ra.returncode)

    rb = run_worker(args.python_bin, args.model_b, b_json, args.tokenizer, args.prompt, args.max_tokens)
    if rb.returncode != 0:
        print(f"B worker failed: {rb.returncode}")
        sys.exit(rb.returncode)

    a = json.loads(Path(a_json).read_text(encoding="utf-8"))
    b = json.loads(Path(b_json).read_text(encoding="utf-8"))

    ta = a["tokens"]
    tb = b["tokens"]
    total = min(len(ta), len(tb))
    matches = sum(1 for x, y in zip(ta, tb) if x == y)
    first_div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)

    report = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "common_len": total,
        "matches": matches,
        "match_ratio": (matches / total) if total else 0.0,
        "first_divergence_index": first_div,
        "a_text": a["text"],
        "b_text": b["text"],
        "a_tokens": ta,
        "b_tokens": tb,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"match: {matches}/{total} ({(100*matches/total) if total else 0:.1f}%)")
    print(f"first_divergence_index: {first_div}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
