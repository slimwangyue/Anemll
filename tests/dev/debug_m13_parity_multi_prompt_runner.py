#!/usr/bin/env python3
"""Robust multi-prompt parity runner using one subprocess repro per prompt."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROMPTS = [
    "What is the capital of France?",
    "Explain stack vs queue in one paragraph.",
    "Write a tiny Python factorial function.",
    "教我做红烧肉",
    "总结一下机器学习和深度学习的区别。",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default="/Users/yw68/Anemll/.venv/bin/python")
    parser.add_argument("--model-a", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    parser.add_argument("--model-b", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1_3")
    parser.add_argument("--tokenizer", default="/Users/yw68/Anemll/qwen3_5_stable_models")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--out", default="tests/dev/m13_parity_multi_prompt_report.json")
    args = parser.parse_args()

    items = []
    total_common = 0
    total_match = 0

    for i, prompt in enumerate(PROMPTS, start=1):
        tmp_out = f"/tmp/m13_parity_prompt_{i}.json"
        cmd = [
            args.python_bin,
            "tests/dev/debug_m13_parity_subprocess_repro.py",
            "--model-a", args.model_a,
            "--model-b", args.model_b,
            "--tokenizer", args.tokenizer,
            "--prompt", f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n",
            "--max-tokens", str(args.max_tokens),
            "--out", tmp_out,
        ]

        rc = subprocess.run(cmd).returncode
        if rc != 0:
            items.append({
                "prompt": prompt,
                "status": "failed",
                "returncode": rc,
            })
            continue

        report = json.loads(Path(tmp_out).read_text(encoding="utf-8"))
        common = int(report.get("common_len", 0))
        match = int(report.get("matches", 0))
        total_common += common
        total_match += match
        items.append({
            "prompt": prompt,
            "status": "ok",
            "common_len": common,
            "matches": match,
            "match_ratio": report.get("match_ratio", 0.0),
            "first_divergence_index": report.get("first_divergence_index"),
            "a_text": report.get("a_text", ""),
            "b_text": report.get("b_text", ""),
        })

    overall = (total_match / total_common) if total_common else 0.0
    out = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "num_prompts": len(PROMPTS),
        "successful_prompts": sum(1 for x in items if x.get("status") == "ok"),
        "total_common": total_common,
        "total_match": total_match,
        "overall_match_ratio": overall,
        "items": items,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"overall: {total_match}/{total_common} ({100*overall:.2f}%)")
    for item in items:
        if item["status"] != "ok":
            print(f"- FAIL: {item['prompt'][:40]} rc={item['returncode']}")
            continue
        print(
            f"- {item['prompt'][:40]:40s} {item['matches']:>3d}/{item['common_len']:<3d} "
            f"{100*item['match_ratio']:.1f}% first_div={item['first_divergence_index']}"
        )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
