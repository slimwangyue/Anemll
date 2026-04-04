#!/usr/bin/env python3
"""Verify exact-prefill multifunction chunk packages on Apple Silicon ANE."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import FFN_LABEL
from prefill_buckets import (
    EXACT_PREFILL_CTX,
    PREFILL_BUCKETS_DESC,
    available_prefill_buckets,
    bucket_function_name,
)


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            full = Path(root) / name
            if not full.is_symlink():
                total += full.stat().st_size
    return total


def function_inputs(spec, function_name: str):
    if hasattr(spec.description, "functions"):
        for fn in spec.description.functions:
            if fn.name == function_name:
                return fn.input
    return spec.description.input


def shape_map(inputs) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for inp in inputs:
        try:
            out[inp.name] = tuple(inp.type.multiArrayType.shape)
        except Exception:
            continue
    return out


def compile_package(mlpackage_path: Path, output_dir: Path, commands: list[str]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    commands.append(f"xcrun coremlcompiler compile {mlpackage_path} {output_dir}")
    started = time.time()
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(mlpackage_path), str(output_dir)],
        capture_output=True,
        text=True,
    )
    compiled_path = output_dir / mlpackage_path.with_suffix(".mlmodelc").name
    return {
        "succeeded": result.returncode == 0 and compiled_path.exists(),
        "returncode": result.returncode,
        "elapsedMs": (time.time() - started) * 1000,
        "compiledPath": str(compiled_path),
        "compiledSizeBytes": dir_size_bytes(compiled_path) if compiled_path.exists() else 0,
        "stderr": result.stderr[-1000:],
        "stdout": result.stdout[-1000:],
    }


def make_inputs_for_function(function_name: str, shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    if function_name == "infer":
        seq_len = 1
    elif function_name.startswith("prefill_bs"):
        seq_len = int(function_name.replace("prefill_bs", ""))
    else:
        seq_len = 1

    for name, shape in shapes.items():
        if name == "input_ids":
            inputs[name] = np.zeros(shape, dtype=np.int32)
        elif name == "position_ids":
            if len(shape) == 1:
                inputs[name] = np.arange(seq_len, dtype=np.int32)
            else:
                inputs[name] = np.arange(seq_len, dtype=np.int32).reshape(shape)
        elif name == "current_pos":
            inputs[name] = np.array([0], dtype=np.int32)
        elif name == "causal_mask":
            mask = np.full(shape, -65504.0, dtype=np.float16)
            if len(shape) == 4:
                for row in range(seq_len):
                    mask[0, 0, row, : row + 1] = 0
            inputs[name] = mask
        elif name in {"linear_conv_state", "linear_recurrent_state"}:
            inputs[name] = np.zeros(shape, dtype=np.float16)
        elif name == "hidden_states":
            inputs[name] = np.zeros(shape, dtype=np.float16)
        else:
            inputs[name] = np.zeros(shape, dtype=np.float16)
    return inputs


def _child_probe_entrypoint() -> int:
    probe_path = Path(os.environ["ANEMLL_VERIFY_PATH"])
    function_name = os.environ.get("ANEMLL_VERIFY_FUNCTION") or None
    try:
        result = verify_model_load_and_predict_impl(probe_path, function_name=function_name)
        print(json.dumps(result))
        return 0
    except Exception as exc:
        print(json.dumps({
            "loadSucceeded": False,
            "predictSucceeded": False,
            "error": str(exc),
        }))
        return 1


def verify_model_load_and_predict_impl(path: Path, function_name: str | None = None) -> dict[str, object]:
    started = time.time()
    kwargs = {"compute_units": ct.ComputeUnit.CPU_AND_NE}
    if function_name is not None:
        kwargs["function_name"] = function_name
    model = ct.models.MLModel(str(path), **kwargs)
    load_ms = (time.time() - started) * 1000
    spec = model.get_spec()
    inputs = make_inputs_for_function(function_name or "main", shape_map(function_inputs(spec, function_name or "main")))
    state = model.make_state() if hasattr(model, "make_state") else None
    pred_started = time.time()
    outputs = model.predict(inputs, state=state) if state is not None else model.predict(inputs)
    predict_ms = (time.time() - pred_started) * 1000
    output_shapes = {}
    for name, value in outputs.items():
        shape = getattr(value, "shape", None)
        output_shapes[name] = list(shape) if shape is not None else None
    return {
        "loadSucceeded": True,
        "loadMs": load_ms,
        "predictSucceeded": True,
        "predictMs": predict_ms,
        "outputShapes": output_shapes,
    }


def verify_model_load_and_predict(
    path: Path,
    function_name: str | None = None,
    timeout_seconds: int = 300,
) -> dict[str, object]:
    env = os.environ.copy()
    env["ANEMLL_VERIFY_PATH"] = str(path)
    env["ANEMLL_VERIFY_FUNCTION"] = function_name or ""
    proc = subprocess.run(
        [sys.executable, __file__, "--child-probe"],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_seconds,
    )
    payload = (proc.stdout or "").strip().splitlines()
    if not payload:
        return {
            "loadSucceeded": False,
            "predictSucceeded": False,
            "error": f"empty child output (rc={proc.returncode}) stderr={proc.stderr[-500:]}",
        }
    try:
        result = json.loads(payload[-1])
    except json.JSONDecodeError:
        result = {
            "loadSucceeded": False,
            "predictSucceeded": False,
            "error": f"invalid child output (rc={proc.returncode}) stdout={payload[-1][:500]} stderr={proc.stderr[-500:]}",
        }
    result["childReturncode"] = proc.returncode
    return result


def verify_chunk_package(path: Path, compile_dir: Path, commands: list[str]) -> dict[str, object]:
    spec = ct.utils.load_spec(str(path))
    function_names = sorted(fn.name for fn in spec.description.functions)
    expected = ["infer"] + [bucket_function_name(bucket) for bucket in sorted(PREFILL_BUCKETS_DESC)]
    result: dict[str, object] = {
        "path": str(path),
        "functionNames": function_names,
        "functionsMatch": function_names == sorted(expected),
        "availableBuckets": available_prefill_buckets(function_names),
        "packageSizeBytes": dir_size_bytes(path),
    }
    result["compile"] = compile_package(path, compile_dir, commands)

    ane_checks = {}
    for function_name in expected:
        try:
            ane_checks[function_name] = verify_model_load_and_predict(path, function_name=function_name)
        except Exception as exc:
            ane_checks[function_name] = {
                "loadSucceeded": False,
                "predictSucceeded": False,
                "error": str(exc),
            }
    result["aneChecks"] = ane_checks
    return result


def verify_single_package(path: Path, compile_dir: Path, commands: list[str]) -> dict[str, object]:
    spec = ct.utils.load_spec(str(path))
    result: dict[str, object] = {
        "path": str(path),
        "inputNames": [inp.name for inp in spec.description.input],
        "outputNames": [out.name for out in spec.description.output],
        "packageSizeBytes": dir_size_bytes(path),
    }
    result["compile"] = compile_package(path, compile_dir, commands)
    try:
        result["aneCheck"] = verify_model_load_and_predict(path, function_name=None)
    except Exception as exc:
        result["aneCheck"] = {
            "loadSucceeded": False,
            "predictSucceeded": False,
            "error": str(exc),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify exact-prefill chunk packages")
    parser.add_argument("--model-dir")
    parser.add_argument("--report")
    parser.add_argument("--child-probe", action="store_true")
    args = parser.parse_args()

    if args.child_probe:
        return _child_probe_entrypoint()

    if not args.model_dir or not args.report:
        parser.error("the following arguments are required: --model-dir, --report")

    model_dir = Path(args.model_dir)
    report_path = Path(args.report)
    compile_dir = model_dir / "compiled_verification"
    if compile_dir.exists():
        shutil.rmtree(compile_dir)
    compile_dir.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    combined_dir = model_dir / f"combined_{FFN_LABEL}_dedup"
    expected_chunks = [combined_dir / f"chunk{idx}.mlpackage" for idx in range(4)]
    embeddings_path = model_dir / "embeddings.mlpackage"
    lm_head_path = model_dir / "lm_head.mlpackage"

    report: dict[str, object] = {
        "modelDir": str(model_dir),
        "contextLength": EXACT_PREFILL_CTX,
        "prefillBuckets": list(PREFILL_BUCKETS_DESC),
        "commands": commands,
        "artifacts": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report["artifacts"]["embeddings"] = verify_single_package(embeddings_path, compile_dir, commands)
    report["artifacts"]["lm_head"] = verify_single_package(lm_head_path, compile_dir, commands)

    chunk_reports = {}
    for chunk_path in expected_chunks:
        chunk_reports[chunk_path.stem] = verify_chunk_package(chunk_path, compile_dir, commands)
    report["artifacts"]["chunks"] = chunk_reports

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
