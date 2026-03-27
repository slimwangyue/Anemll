#!/usr/bin/env python3
"""Prompt-sweep parity repro (subprocess-isolated) between two CoreML model dirs.

Runs multiple prompts through both model sets and reports token-level parity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


WORKER = r'''
import json, os, sys
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

model_dir = sys.argv[1]
out_file = sys.argv[2]
tokenizer_dir = sys.argv[3]
prompts_json = sys.argv[4]
max_tokens = int(sys.argv[5])

CTX = 1024
NUM_CHUNKS = 4
LABEL = "LUT4"

prompts = json.loads(prompts_json)
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
cu = ct.ComputeUnit.CPU_AND_NE

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

stop_ids = set()
for sid in ["<|im_end|>", "<|endoftext|>"]:
    ids = tok.encode(sid, add_special_tokens=False)
    if ids:
        stop_ids.add(ids[0])

def reset_states():
    states = [m.make_state() for m in ffns]
    if has_linear:
        lin_convs = [np.zeros(inp_map["linear_conv_state"], dtype=np.float16) for _ in range(NUM_CHUNKS)]
        lin_recs = [np.zeros(inp_map["linear_recurrent_state"], dtype=np.float16) for _ in range(NUM_CHUNKS)]
    else:
        lin_convs = [None] * NUM_CHUNKS
        lin_recs = [None] * NUM_CHUNKS
    return states, lin_convs, lin_recs

def step(states, lin_convs, lin_recs, tok_id, pos):
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

rows = []
for prompt in prompts:
    # Match prior parity prompt format
    wrapped = f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
    input_ids = tok.encode(wrapped)
    states, lin_convs, lin_recs = reset_states()

    last = None
    for i, tid in enumerate(input_ids):
        last = step(states, lin_convs, lin_recs, tid, i)

    gen = [last]
    pos = len(input_ids)
    for _ in range(max_tokens - 1):
        if pos >= CTX - 1:
            break
        nxt = step(states, lin_convs, lin_recs, gen[-1], pos)
        gen.append(nxt)
        pos += 1
        if nxt in stop_ids:
            break

    rows.append({
        "prompt": prompt,
        "prompt_len": len(input_ids),
        "tokens": gen,
        "text": tok.decode(gen, skip_special_tokens=True),
    })

with open(out_file, "w", encoding="utf-8") as f:
    json.dump({"model_dir": model_dir, "rows": rows}, f, ensure_ascii=False, indent=2)
'''


def run_worker(python_bin: str, model_dir: str, out_file: str, tokenizer: str, prompts: list[str], max_tokens: int) -> int:
    worker_path = Path("/tmp/m13_sweep_worker.py")
    worker_path.write_text(WORKER, encoding="utf-8")
    cmd = [
        python_bin,
        str(worker_path),
        model_dir,
        out_file,
        tokenizer,
        json.dumps(prompts, ensure_ascii=False),
        str(max_tokens),
    ]
    return subprocess.run(cmd).returncode


def compare(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    total_common = 0
    total_match = 0
    for ar, br in zip(a_rows, b_rows):
        at = ar["tokens"]
        bt = br["tokens"]
        common = min(len(at), len(bt))
        match = sum(1 for x, y in zip(at, bt) if x == y)
        ratio = (match / common) if common else 0.0
        first_div = next((i for i, (x, y) in enumerate(zip(at, bt)) if x != y), None)
        total_common += common
        total_match += match
        items.append({
            "prompt": ar["prompt"],
            "common_len": common,
            "match": match,
            "match_ratio": ratio,
            "first_divergence_index": first_div,
            "a_text": ar["text"],
            "b_text": br["text"],
        })
    overall = (total_match / total_common) if total_common else 0.0
    return {"overall_match_ratio": overall, "total_common": total_common, "total_match": total_match, "items": items}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-sweep parity between two model dirs")
    parser.add_argument("--python-bin", default="/Users/yw68/Anemll/.venv/bin/python")
    parser.add_argument("--model-a", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    parser.add_argument("--model-b", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1_3")
    parser.add_argument("--tokenizer", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--out", default="tests/dev/m13_parity_prompt_sweep_report.json")
    args = parser.parse_args()

    prompts = [
        "What is the capital of France?",
        "Explain stack vs queue in one paragraph.",
        "Write a tiny Python factorial function.",
        "教我做红烧肉",
        "总结一下机器学习和深度学习的区别。",
        "Give 3 bullet points about memory debugging.",
        "What comes after 41? Reply with one number.",
        "Describe recursion with one simple example.",
        "如何提高模型推理速度？",
        "Translate: hello world to Chinese.",
    ]

    a_json = "/tmp/m13_sweep_a.json"
    b_json = "/tmp/m13_sweep_b.json"

    rc = run_worker(args.python_bin, args.model_a, a_json, args.tokenizer, prompts, args.max_tokens)
    if rc != 0:
        raise SystemExit(f"model-a worker failed: {rc}")

    rc = run_worker(args.python_bin, args.model_b, b_json, args.tokenizer, prompts, args.max_tokens)
    if rc != 0:
        raise SystemExit(f"model-b worker failed: {rc}")

    a = json.loads(Path(a_json).read_text(encoding="utf-8"))
    b = json.loads(Path(b_json).read_text(encoding="utf-8"))
    cmp = compare(a["rows"], b["rows"])

    out = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "max_tokens": args.max_tokens,
        "num_prompts": len(prompts),
        **cmp,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"overall: {out['total_match']}/{out['total_common']} ({100*out['overall_match_ratio']:.2f}%)")
    for item in out["items"]:
        print(
            f"- {item['prompt'][:38]:38s}  "
            f"{item['match']:>3d}/{item['common_len']:<3d}  "
            f"{100*item['match_ratio']:.1f}%  "
            f"first_div={item['first_divergence_index']}"
        )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
