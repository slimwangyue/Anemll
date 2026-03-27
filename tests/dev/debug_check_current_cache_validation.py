#!/usr/bin/env python3
"""Cache validation for current exported Qwen3.5 model set.

Checks:
1) Static MIL check: slice_update begin/end depends on model inputs (dynamic index path).
2) Runtime decode smoke: run sequential decode steps on ANE across increasing positions.

Usage:
  /Users/yw68/Anemll/.venv/bin/python tests/dev/debug_check_current_cache_validation.py \
      --model-dir /Users/yw68/Anemll_remote_run/qwen35_milestone1_3 \
      --tokenizer /Users/yw68/Anemll/qwen3_5_stable_models
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer


@dataclass
class SliceUpdateCheck:
    model_name: str
    total_slice_updates: int
    dynamic_slice_updates: int
    has_dynamic_write: bool


def _is_dynamic_name(
    name: str | None,
    model_inputs: set[str],
    producer: Dict[str, Tuple[str, object]],
    visited: set[str] | None = None,
    depth: int = 0,
) -> bool:
    if not name:
        return False
    if visited is None:
        visited = set()
    if name in visited or depth > 20:
        return False
    visited.add(name)

    if name in model_inputs:
        return True
    if name not in producer:
        return False

    op_type, op = producer[name]
    if op_type == "const":
        return False

    for val in op.inputs.values():
        for arg in val.arguments:
            arg_name = getattr(arg, "name", None)
            if arg_name and _is_dynamic_name(arg_name, model_inputs, producer, visited, depth + 1):
                return True
    return False


def check_dynamic_slice_updates(model_path: str, model_name: str) -> SliceUpdateCheck:
    model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()

    model_inputs = set(inp.name for inp in spec.description.input)

    prog = spec.mlProgram
    producer: Dict[str, Tuple[str, object]] = {}
    for fn in prog.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                for out in op.outputs:
                    producer[out.name] = (op.type, op)

    total = 0
    dynamic = 0
    for fn in prog.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                if op.type != "slice_update":
                    continue
                total += 1
                begin_name = None
                end_name = None
                for key, val in op.inputs.items():
                    for arg in val.arguments:
                        arg_name = getattr(arg, "name", None)
                        if not arg_name:
                            continue
                        if key == "begin":
                            begin_name = arg_name
                        elif key == "end":
                            end_name = arg_name

                begin_dyn = _is_dynamic_name(begin_name, model_inputs, producer)
                end_dyn = _is_dynamic_name(end_name, model_inputs, producer)
                if begin_dyn or end_dyn:
                    dynamic += 1

    del model
    return SliceUpdateCheck(
        model_name=model_name,
        total_slice_updates=total,
        dynamic_slice_updates=dynamic,
        has_dynamic_write=(dynamic > 0),
    )


def _extract_shape_map(spec) -> Dict[str, Tuple[int, ...]]:
    shape_map: Dict[str, Tuple[int, ...]] = {}
    for inp in spec.description.input:
        try:
            shape_map[inp.name] = tuple(inp.type.multiArrayType.shape)
        except Exception:
            continue
    return shape_map


def run_decode_smoke(model_dir: str, tokenizer_dir: str, steps: int, ctx: int) -> Dict[str, object]:
    cu = ct.ComputeUnit.CPU_AND_NE

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
    prompt = "<|im_start|>user\nSay hello in one sentence.<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) < steps:
        # Ensure enough positions are exercised.
        prompt_ids = (prompt_ids * ((steps // max(1, len(prompt_ids))) + 1))[:steps]
    else:
        prompt_ids = prompt_ids[:steps]

    embed = ct.models.MLModel(os.path.join(model_dir, "embeddings.mlpackage"), compute_units=cu)
    lmhead = ct.models.MLModel(os.path.join(model_dir, "lm_head.mlpackage"), compute_units=cu)
    ffns = [
        ct.models.MLModel(os.path.join(model_dir, f"ffn_LUT4_chunk{i}.mlpackage"), compute_units=cu)
        for i in range(4)
    ]

    spec0 = ffns[0].get_spec()
    shape_map = _extract_shape_map(spec0)
    has_linear = "linear_conv_state" in shape_map and "linear_recurrent_state" in shape_map

    states = [m.make_state() for m in ffns]
    if has_linear:
        lin_convs = [np.zeros(shape_map["linear_conv_state"], dtype=np.float16) for _ in range(4)]
        lin_recs = [np.zeros(shape_map["linear_recurrent_state"], dtype=np.float16) for _ in range(4)]
    else:
        lin_convs = [None] * 4
        lin_recs = [None] * 4

    generated: List[int] = []
    t0 = time.time()
    for pos, tok_id in enumerate(prompt_ids):
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]

        mask = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float16)
        mask[:, :, :, : pos + 1] = 0

        for ci in range(4):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if has_linear:
                inp["linear_conv_state"] = lin_convs[ci]
                inp["linear_recurrent_state"] = lin_recs[ci]

            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if "linear_conv_state_out" in out:
                lin_convs[ci] = out["linear_conv_state_out"]
                lin_recs[ci] = out["linear_recurrent_state_out"]

        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            nxt = int(np.argmax(lm_out["logits"].flatten()))
        else:
            nxt = int(lm_out["argmax_idx"].flatten()[0])
        generated.append(nxt)

    elapsed = time.time() - t0

    return {
        "steps": steps,
        "elapsed_sec": elapsed,
        "tok_per_sec": steps / max(elapsed, 1e-9),
        "generated_preview": generated[:8],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Current model cache validation")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--out", default="tests/dev/current_model_cache_validation_report.json")
    args = parser.parse_args()

    checks: List[SliceUpdateCheck] = []
    for kind in ("ffn", "prefill"):
        for i in range(4):
            name = f"{kind}_LUT4_chunk{i}.mlpackage"
            path = os.path.join(args.model_dir, name)
            checks.append(check_dynamic_slice_updates(path, name))

    static_pass = all(c.has_dynamic_write for c in checks)

    runtime = run_decode_smoke(args.model_dir, args.tokenizer, args.steps, args.ctx)

    report = {
        "model_dir": args.model_dir,
        "ctx": args.ctx,
        "steps": args.steps,
        "static_dynamic_slice_check": [c.__dict__ for c in checks],
        "static_pass": static_pass,
        "runtime_decode_smoke": runtime,
        "overall_pass": static_pass,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {args.out}")

    return 0 if static_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
