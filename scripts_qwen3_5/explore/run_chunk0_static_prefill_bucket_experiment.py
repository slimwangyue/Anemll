#!/usr/bin/env python3
"""Chunk0 static prefill bucket experiment for Qwen3.5 on iOS ANE.

Exports chunk0 infer + exact prefill buckets for CTX=2048, combines them into
one deduped multifunction package, compiles that package, copies it into the
local_llm iOS bundle under a dedicated resource name, and benchmarks each
prefill bucket on device via xcodebuild/XCTest.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import coremltools as ct

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts_qwen3_5.explore.explore_config import HF_MODEL, OUTPUT_ROOT
from scripts_qwen3_5.config import FFN_PER_CHANNEL
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from anemll.utils.combine_models import _save_multifunction_dedup


BUCKETS = [32, 64, 128, 256, 512, 1024]
TARGETS = ["infer"] + [f"prefill_bs{bucket}" for bucket in BUCKETS]
CTX = 2048
CHUNK_INDEX = 0
NUM_CHUNKS = 4
LUT_BITS = 6
RESOURCE_NAME = "chunk0_prefill_bucket_experiment_ctx2048_lut6"
RESULT_MARKER = "[ANEMLL_BENCH_RESULT]"
PROMPT_LENGTHS = [32, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048]
BENCHMARK_CONFIG_FILENAME = "benchmark_probe_config.json"


def default_model_candidates() -> list[Path]:
    return [
        Path(HF_MODEL),
        Path("/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"),
        _REPO_ROOT / "models" / "Qwen__Qwen3.5-4B",
    ]


def resolve_model_path(requested: str) -> Path:
    requested_path = Path(requested).expanduser().resolve()
    if requested_path.exists():
        return requested_path

    for candidate in default_model_candidates():
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            print(
                f"[model] Requested path not found: {requested_path}. "
                f"Using fallback model path: {resolved}"
            )
            return resolved

    raise FileNotFoundError(
        "Could not locate the HuggingFace Qwen3.5-4B directory. "
        f"Tried: {requested_path} and "
        + ", ".join(str(path.expanduser().resolve()) for path in default_model_candidates())
    )


def dir_size_mb(path: Path) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if not file_path.is_symlink():
                total += file_path.stat().st_size
    return total / (1024 * 1024)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = True,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"[run] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            text=True,
            capture_output=capture_output,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        result = subprocess.CompletedProcess(
            cmd,
            returncode=124,
            stdout=stdout,
            stderr=stderr,
        )
    if capture_output and result.stdout:
        print(result.stdout)
    if capture_output and result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )
    return result


def summarize_text_tail(text: str, *, lines: int = 40) -> str:
    if not text:
        return ""
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def benchmark_device_preflight(
    *,
    local_llm_root: Path,
    device_id: str,
) -> dict[str, object]:
    xctrace_result = run_command(
        ["xcrun", "xctrace", "list", "devices"],
        check=False,
    )
    xctrace_output = "\n".join(
        part for part in [xctrace_result.stdout, xctrace_result.stderr] if part
    )

    section: str | None = None
    xctrace_state = "not_listed"
    for line in xctrace_output.splitlines():
        stripped = line.strip()
        if stripped == "== Devices ==":
            section = "devices"
            continue
        if stripped == "== Devices Offline ==":
            section = "offline"
            continue
        if stripped.startswith("== ") and stripped.endswith(" =="):
            section = None
            continue
        if device_id in stripped:
            if section == "devices":
                xctrace_state = "available"
            elif section == "offline":
                xctrace_state = "offline"
            else:
                xctrace_state = "listed"
            break

    showdestinations_result = run_command(
        [
            "xcodebuild",
            "-project",
            str(local_llm_root / "local_llm.xcodeproj"),
            "-scheme",
            "local_llmUnitTests",
            "-showdestinations",
        ],
        cwd=local_llm_root,
        check=False,
    )
    showdestinations_output = "\n".join(
        part for part in [showdestinations_result.stdout, showdestinations_result.stderr] if part
    )
    xcodebuild_destination_found = (
        device_id in showdestinations_output
        or f"id:{device_id}" in showdestinations_output
        or f"id={device_id}" in showdestinations_output
    )

    # xctrace can lag behind CoreDevice/Xcode. Trust xcodebuild's destination
    # resolution for XCTest eligibility, and keep xctrace only as diagnostic
    # context.
    available = xcodebuild_destination_found
    reason: str | None = None
    if not available:
        if xctrace_state == "offline":
            reason = (
                f"Device {device_id} is listed by xctrace under 'Devices Offline', "
                "so xcodebuild cannot target it yet."
            )
        elif xctrace_state == "not_listed":
            reason = (
                f"Device {device_id} is not visible to xctrace, so it is not currently "
                "connected/recognized by Xcode."
            )
        elif not xcodebuild_destination_found:
            reason = (
                f"Device {device_id} is not present in xcodebuild -showdestinations for "
                "the local_llmUnitTests scheme."
            )
        else:
            reason = f"Device {device_id} is not available for on-device XCTest."

    return {
        "deviceID": device_id,
        "available": available,
        "reason": reason,
        "xctraceState": xctrace_state,
        "xcodebuildDestinationFound": xcodebuild_destination_found,
        "xctraceOutputTail": summarize_text_tail(xctrace_output),
        "xcodebuildShowdestinationsTail": summarize_text_tail(showdestinations_output),
    }


def load_qwen_model(model_path: Path) -> Qwen35ForCausalLM:
    cfg = Qwen35Config.from_json(str(model_path / "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    ok = model.load_pretrained_weights(str(model_path))
    if not ok:
        raise RuntimeError(f"Failed to load Qwen3.5 weights from {model_path}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def export_target_path(output_dir: Path, target: str) -> Path:
    if target == "infer":
        return output_dir / "chunk0_infer.mlpackage"
    if target.startswith("prefill_bs"):
        return output_dir / f"chunk0_{target}.mlpackage"
    raise ValueError(f"Unsupported export target: {target}")


def export_single_target(
    model_path: Path,
    output_dir: Path,
    target: str,
    *,
    skip_existing: bool,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = export_target_path(output_dir, target)
    if skip_existing and target_path.exists():
        print(f"[skip] {target_path.name}")
        return f"skip export target {target}"

    model = load_qwen_model(model_path)

    try:
        if target == "infer":
            print("[export] chunk0 infer")
            conv = Qwen35Converter(
                model,
                context_length=CTX,
                batch_size=max(BUCKETS),
                num_chunks=NUM_CHUNKS,
                lut_bits=LUT_BITS,
                per_channel=FFN_PER_CHANNEL,
            )
            mlmodel = conv.convert_part_2(model, chunk_idx=CHUNK_INDEX, total_chunks=NUM_CHUNKS)
            if target_path.exists():
                shutil.rmtree(target_path)
            mlmodel.save(str(target_path))
            del mlmodel, conv
            gc.collect()
            return (
                "export infer: "
                f"Qwen35Converter(context_length={CTX}, batch_size={max(BUCKETS)}, "
                f"num_chunks={NUM_CHUNKS}, lut_bits={LUT_BITS}, per_channel={FFN_PER_CHANNEL})"
            )

        if target.startswith("prefill_bs"):
            bucket = int(target.replace("prefill_bs", ""))
            print(f"[export] chunk0 {target}")
            conv = Qwen35Converter(
                model,
                context_length=CTX,
                batch_size=bucket,
                num_chunks=NUM_CHUNKS,
                lut_bits=LUT_BITS,
                per_channel=FFN_PER_CHANNEL,
            )
            mlmodel = conv.convert_part_2_prefill_exact(
                model,
                chunk_idx=CHUNK_INDEX,
                total_chunks=NUM_CHUNKS,
                exact_seq_len=bucket,
            )
            if target_path.exists():
                shutil.rmtree(target_path)
            mlmodel.save(str(target_path))
            del mlmodel, conv
            gc.collect()
            return (
                "export prefill bucket: "
                f"Qwen35Converter(context_length={CTX}, batch_size={bucket}, "
                f"num_chunks={NUM_CHUNKS}, lut_bits={LUT_BITS}, per_channel={FFN_PER_CHANNEL})."
                "convert_part_2_prefill_exact(chunk_idx=0, total_chunks=4, exact_seq_len="
                f"{bucket})"
            )

        raise ValueError(f"Unsupported export target: {target}")
    finally:
        del model
        gc.collect()


def export_targets_via_subprocess(
    model_path: Path,
    output_dir: Path,
    *,
    skip_existing: bool,
    chunk_postprocess_workers: int,
) -> list[str]:
    commands: list[str] = []
    for target in TARGETS:
        target_path = export_target_path(output_dir, target)
        if skip_existing and target_path.exists():
            print(f"[skip] {target_path.name}")
            commands.append(f"skip export target {target}")
            continue

        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--model",
            str(model_path),
            "--output",
            str(output_dir),
            "--skip-combine",
            "--skip-compile",
            "--skip-copy",
            "--skip-benchmark",
            "--export-target",
            target,
            "--chunk-postprocess-workers",
            str(chunk_postprocess_workers),
        ]
        if skip_existing:
            cmd.append("--skip-existing")
        run_command(cmd, cwd=_REPO_ROOT, capture_output=False)
        if target == "infer":
            commands.append(
                "export infer subprocess: "
                f"workers={chunk_postprocess_workers}, output={target_path.name}"
            )
        else:
            commands.append(
                f"export {target} subprocess: "
                f"workers={chunk_postprocess_workers}, output={target_path.name}"
            )
    return commands


def combine_chunk0_buckets(output_dir: Path, *, skip_existing: bool) -> list[tuple[str, str, str]]:
    combined_path = output_dir / "chunk0.mlpackage"
    if combined_path.exists() and not skip_existing:
        shutil.rmtree(combined_path)

    sources = [(str(output_dir / "chunk0_infer.mlpackage"), "main", "infer")]
    for bucket in BUCKETS:
        sources.append(
            (str(output_dir / f"chunk0_prefill_bs{bucket}.mlpackage"), "main", f"prefill_bs{bucket}")
        )

    if skip_existing and combined_path.exists():
        print(f"[skip] {combined_path.name}")
    else:
        print("[combine] chunk0 multifunction dedup package")
        _save_multifunction_dedup(sources, str(combined_path), dedup_weights=True, verbose=False)
    return sources


def compile_combined_package(output_dir: Path, *, skip_existing: bool) -> Path:
    combined_path = output_dir / "chunk0.mlpackage"
    compiled_path = output_dir / "chunk0.mlmodelc"
    if skip_existing and compiled_path.exists():
        print(f"[skip] {compiled_path.name}")
        return compiled_path
    if compiled_path.exists():
        shutil.rmtree(compiled_path)
    run_command(["xcrun", "coremlcompiler", "compile", str(combined_path), str(output_dir)])
    if not compiled_path.exists():
        raise RuntimeError(f"Expected compiled model at {compiled_path}")
    return compiled_path


def function_names_from_spec(spec) -> list[str]:
    if getattr(spec.description, "functions", None):
        return [fn.name for fn in spec.description.functions]
    if spec.HasField("mlProgram"):
        return list(spec.mlProgram.functions.keys())
    return ["main"]


def spec_function_inputs(spec) -> dict[str, dict[str, list[int]]]:
    inputs_by_function: dict[str, dict[str, list[int]]] = {}
    for function in getattr(spec.description, "functions", []):
        function_inputs: dict[str, list[int]] = {}
        for desc in function.input:
            if desc.type.WhichOneof("Type") != "multiArrayType":
                continue
            function_inputs[desc.name] = [int(dim) for dim in desc.type.multiArrayType.shape]
        inputs_by_function[function.name] = function_inputs
    return inputs_by_function


def verify_combined_package(output_dir: Path) -> dict[str, object]:
    combined_path = output_dir / "chunk0.mlpackage"
    spec = ct.utils.load_spec(str(combined_path))
    expected_functions = ["infer"] + [f"prefill_bs{bucket}" for bucket in BUCKETS]
    actual_functions = sorted(function_names_from_spec(spec))
    verification: dict[str, object] = {
        "combined_path": str(combined_path),
        "expected_functions": expected_functions,
        "actual_functions": actual_functions,
        "functions_match": sorted(expected_functions) == actual_functions,
        "function_inputs": spec_function_inputs(spec),
    }

    bucket_checks: dict[str, dict[str, object]] = {}
    for bucket in BUCKETS:
        name = f"prefill_bs{bucket}"
        inputs = verification["function_inputs"][name]
        hidden_shape = inputs.get("hidden_states")
        mask_shape = inputs.get("causal_mask")
        bucket_checks[name] = {
            "has_valid_len": "valid_len" in inputs,
            "hidden_states_shape": hidden_shape,
            "causal_mask_shape": mask_shape,
            "hidden_states_seq_matches": bool(hidden_shape and len(hidden_shape) >= 2 and hidden_shape[1] == bucket),
            "causal_mask_ctx_matches": bool(mask_shape and len(mask_shape) >= 4 and mask_shape[3] == CTX),
        }
    verification["bucket_checks"] = bucket_checks
    return verification


def copy_package_into_local_llm(
    combined_path: Path,
    *,
    local_llm_root: Path,
    resource_name: str,
) -> Path:
    bundle_dir = local_llm_root / "local_llm" / "Resources" / "Models.bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    target = bundle_dir / f"{resource_name}.mlpackage"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(combined_path, target)
    return target


def write_benchmark_config(
    *,
    local_llm_root: Path,
    resource_name: str,
    function_name: str,
    warmups: int,
    trials: int,
) -> Path:
    config_path = (
        local_llm_root
        / "local_llm"
        / "Resources"
        / "Models.bundle"
        / BENCHMARK_CONFIG_FILENAME
    )
    payload = {
        "modelResource": resource_name,
        "modelExtension": "mlpackage",
        "functionName": function_name,
        "warmups": warmups,
        "trials": trials,
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def parse_result_from_xcode_output(output: str) -> dict[str, object] | None:
    ansi_stripped = re.sub(r"\x1B\[[0-9;]*[ -/]*[@-~]", "", output)
    for line in ansi_stripped.splitlines():
        marker_index = line.find(RESULT_MARKER)
        if marker_index >= 0:
            payload = line[marker_index + len(RESULT_MARKER):].strip()
            if payload:
                return json.loads(payload)
    match = re.search(re.escape(RESULT_MARKER) + r"(\{.*?\})(?:\r?\n|$)", ansi_stripped, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return None


def benchmark_bucket_on_device(
    *,
    local_llm_root: Path,
    device_id: str,
    resource_name: str,
    function_name: str,
    warmups: int,
    trials: int,
    result_bundle_path: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    env = os.environ.copy()
    env["ANEMLL_BENCH_MODEL_RESOURCE"] = resource_name
    env["ANEMLL_BENCH_MODEL_EXT"] = "mlpackage"
    env["ANEMLL_BENCH_MODEL_FUNCTION"] = function_name
    env["ANEMLL_BENCH_WARMUPS"] = str(warmups)
    env["ANEMLL_BENCH_TRIALS"] = str(trials)
    config_path = write_benchmark_config(
        local_llm_root=local_llm_root,
        resource_name=resource_name,
        function_name=function_name,
        warmups=warmups,
        trials=trials,
    )
    cmd = [
        "xcodebuild",
        "-project",
        str(local_llm_root / "local_llm.xcodeproj"),
        "-scheme",
        "local_llmUnitTests",
        "-configuration",
        "Release",
        "-destination",
        f"id={device_id}",
        "-resultBundlePath",
        str(result_bundle_path),
        "-only-testing:local_llmTests/AnemllDeviceProbeTests/testBenchmarkSelectedModelFunctionOnANE",
        "test",
        "ENABLE_TESTABILITY=YES",
    ]
    started_at = iso_now()
    result = run_command(
        cmd,
        cwd=local_llm_root,
        env=env,
        check=False,
        timeout_seconds=timeout_seconds,
    )
    combined_output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    log_path = result_bundle_path.with_suffix(".log")
    log_path.write_text(combined_output, encoding="utf-8")
    parsed = parse_result_from_xcode_output(combined_output)
    if parsed is None:
        parsed = {
            "modelResource": resource_name,
            "modelExtension": "mlpackage",
            "functionName": function_name,
            "computeUnits": "cpuAndNeuralEngine",
            "warmupCount": warmups,
            "trialCount": trials,
            "sequenceLength": None,
            "compileSucceeded": False,
            "compileDurationMs": None,
            "compileError": None,
            "loadSucceeded": False,
            "loadDurationMs": None,
            "loadError": None,
            "firstPredictionSucceeded": False,
            "firstPredictionDurationMs": None,
            "firstPredictionError": None,
            "memoryBeforeLoadMB": None,
            "memoryAfterLoadMB": None,
            "memoryAfterFirstPredictionMB": None,
            "stats": None,
            "timestamp": iso_now(),
            "infrastructureError": (
                f"xcodebuild timed out after {timeout_seconds}s"
                if result.returncode == 124
                else f"xcodebuild exited {result.returncode}"
            ),
        }
    parsed["bucketSize"] = int(function_name.replace("prefill_bs", ""))
    parsed["xcodebuildReturnCode"] = result.returncode
    parsed["startedAt"] = started_at
    parsed["finishedAt"] = iso_now()
    parsed["benchmarkConfigPath"] = str(config_path)
    parsed["xcodebuildLogPath"] = str(log_path)
    return parsed


def compose_cost(length: int, latencies: dict[int, float], buckets: list[int]) -> float | None:
    remaining = length
    total = 0.0
    for bucket in sorted(buckets, reverse=True):
        while remaining >= bucket:
            if bucket not in latencies:
                return None
            total += latencies[bucket]
            remaining -= bucket
    return total if remaining == 0 else None


def apply_recommendation_rules(results: list[dict[str, object]]) -> dict[str, object]:
    successful = {
        int(item["bucketSize"]): item
        for item in results
        if item.get("loadSucceeded") and item.get("firstPredictionSucceeded") and item.get("stats")
    }
    latencies = {
        bucket: float(successful[bucket]["stats"]["medianMs"])
        for bucket in successful
        if successful[bucket]["stats"]["medianMs"] is not None
    }

    kept: list[int] = []
    dropped: dict[int, str] = {}
    if 32 in latencies:
        kept.append(32)
    else:
        dropped[32] = "Mandatory crossover bucket did not load and run on ANE."

    for bucket in [64, 128, 256, 512, 1024]:
        if bucket not in latencies:
            dropped[bucket] = "Bucket failed ANE compile/load or first prediction."
            continue
        composition = compose_cost(bucket, latencies, kept)
        if composition is None:
            kept.append(bucket)
            continue
        direct = latencies[bucket]
        if direct <= composition * 0.85:
            kept.append(bucket)
        else:
            dropped[bucket] = (
                f"Direct median {direct:.2f} ms is not at least 15% faster than "
                f"composition cost {composition:.2f} ms."
            )

    changed = True
    while changed:
        changed = False
        for bucket in sorted([value for value in kept if value != 32], reverse=True):
            without = [value for value in kept if value != bucket]
            max_improvement = 0.0
            for prompt_len in PROMPT_LENGTHS:
                with_cost = compose_cost(prompt_len, latencies, kept)
                without_cost = compose_cost(prompt_len, latencies, without)
                if with_cost is None or without_cost is None:
                    continue
                improvement = (without_cost - with_cost) / without_cost if without_cost > 0 else 0.0
                max_improvement = max(max_improvement, improvement)
            if max_improvement <= 0.10:
                kept.remove(bucket)
                dropped[bucket] = (
                    f"Removing bucket changes no simulated prompt length by more than 10% "
                    f"(max improvement {max_improvement * 100:.1f}%)."
                )
                changed = True
                break

    simulations = []
    for prompt_len in PROMPT_LENGTHS:
        simulations.append(
            {
                "promptLength": prompt_len,
                "keptBucketsCostMs": compose_cost(prompt_len, latencies, kept),
            }
        )

    return {
        "keptBuckets": kept,
        "droppedBuckets": [{"bucket": bucket, "reason": reason} for bucket, reason in sorted(dropped.items())],
        "simulations": simulations,
    }


def write_results_markdown(
    output_path: Path,
    *,
    artifact_sizes: dict[str, float],
    verification: dict[str, object],
    results: list[dict[str, object]],
    recommendation: dict[str, object],
    commands: list[str],
    benchmark_preflight: dict[str, object] | None,
) -> None:
    lines: list[str] = []
    lines.append("# Chunk0 Static Prefill Bucket Experiment")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Context length: `{CTX}`")
    lines.append(f"- Buckets tested: `{', '.join(str(bucket) for bucket in BUCKETS)}`")
    lines.append("- Device target: `Yue` (`00008140-001A1CD02082201C`)")
    lines.append("- Build configuration: `Release`")
    lines.append("- Compute units: `.cpuAndNeuralEngine`")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    for command in commands:
        lines.append(f"- `{command}`")
    lines.append("")
    lines.append("## Benchmark Preflight")
    lines.append("")
    if benchmark_preflight is None:
        lines.append("- Not run")
    else:
        lines.append(f"- Available: `{benchmark_preflight['available']}`")
        lines.append(f"- xctrace state: `{benchmark_preflight['xctraceState']}`")
        lines.append(
            f"- xcodebuild destination found: `{benchmark_preflight['xcodebuildDestinationFound']}`"
        )
        if benchmark_preflight.get("reason"):
            lines.append(f"- Reason: `{benchmark_preflight['reason']}`")
    lines.append("")
    lines.append("## Artifact Sizes")
    lines.append("")
    lines.append(f"- Combined `.mlpackage`: `{artifact_sizes['combined_mlpackage_mb']:.1f} MB`")
    lines.append(f"- Combined `.mlmodelc`: `{artifact_sizes['combined_mlmodelc_mb']:.1f} MB`")
    lines.append("- Deployment package count stays the same, but the multifunction chunk now carries six exact prefill functions plus `infer`, which increases function-selection and compile/load surface area.")
    lines.append("")
    lines.append("## Static Verification")
    lines.append("")
    lines.append(f"- Functions match expected set: `{verification['functions_match']}`")
    lines.append(f"- Actual functions: `{', '.join(verification['actual_functions'])}`")
    for function_name, checks in verification["bucket_checks"].items():
        lines.append(
            f"- `{function_name}`: valid_len present=`{checks['has_valid_len']}`, "
            f"hidden_states=`{checks['hidden_states_shape']}`, causal_mask=`{checks['causal_mask_shape']}`"
        )
    lines.append("")
    lines.append("## Raw Results")
    lines.append("")
    lines.append("| Bucket | ANE Load | Compile ms | Load ms | Median ms | Mean ms | P95 ms | ms/token | Resident Δ MB | Error |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in sorted(results, key=lambda value: value["bucketSize"]):
        stats = item.get("stats") or {}
        memory_before = item.get("memoryBeforeLoadMB")
        memory_after_first = item.get("memoryAfterFirstPredictionMB")
        resident_delta = None
        if memory_before is not None and memory_after_first is not None:
            resident_delta = memory_after_first - memory_before
        error = (
            item.get("firstPredictionError")
            or item.get("loadError")
            or item.get("compileError")
            or item.get("infrastructureError")
            or ""
        )
        lines.append(
            "| "
            f"{item['bucketSize']} | "
            f"{'yes' if item.get('loadSucceeded') else 'no'} | "
            f"{format_optional(item.get('compileDurationMs'))} | "
            f"{format_optional(item.get('loadDurationMs'))} | "
            f"{format_optional(stats.get('medianMs'))} | "
            f"{format_optional(stats.get('meanMs'))} | "
            f"{format_optional(stats.get('p95Ms'))} | "
            f"{format_optional(stats.get('msPerToken'))} | "
            f"{format_optional(resident_delta)} | "
            f"{error.replace('|', '/')} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if not results and benchmark_preflight is not None and not benchmark_preflight["available"]:
        lines.append(
            "- Recommendation pending: on-device ANE benchmark data is unavailable because the "
            "target iPhone is not currently an available Xcode destination."
        )
    else:
        lines.append(f"- Keep: `{', '.join(str(value) for value in recommendation['keptBuckets']) or 'none'}`")
        for item in recommendation["droppedBuckets"]:
            lines.append(f"- Drop `{item['bucket']}`: {item['reason']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the chunk0 static prefill bucket experiment")
    parser.add_argument("--model", default=HF_MODEL, help="Path to the HuggingFace Qwen3.5 model")
    parser.add_argument(
        "--output",
        default=str(Path(OUTPUT_ROOT) / "chunk0_prefill_buckets_ctx2048_lut6"),
        help="Experiment output directory",
    )
    parser.add_argument(
        "--local-llm-root",
        default="/Users/yw68/local_llm/local_llm",
        help="Path to the local_llm Xcode project root",
    )
    parser.add_argument("--device-id", default="00008140-001A1CD02082201C")
    parser.add_argument("--resource-name", default=RESOURCE_NAME)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--bucket-timeout-seconds",
        type=int,
        default=600,
        help="Per-bucket xcodebuild timeout for device benchmark runs",
    )
    parser.add_argument(
        "--benchmark-target",
        choices=[f"prefill_bs{bucket}" for bucket in BUCKETS],
        help="Optional: benchmark only one bucket function instead of the full sweep",
    )
    parser.add_argument(
        "--export-target",
        choices=TARGETS,
        help="Internal helper mode: export exactly one function package and exit",
    )
    parser.add_argument(
        "--chunk-postprocess-workers",
        type=int,
        default=8,
        help="Worker count for LUT postprocess on chunk exports",
    )
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    output_dir = Path(args.output).resolve()
    local_llm_root = Path(args.local_llm_root).resolve()
    results_json_path = output_dir / "results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_bundles_dir = output_dir / "xcresults"
    result_bundles_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QWEN35_CHUNK_POSTPROCESS_WORKERS"] = str(args.chunk_postprocess_workers)

    if args.export_target:
        export_single_target(
            model_path,
            output_dir,
            args.export_target,
            skip_existing=args.skip_existing,
        )
        return 0

    commands: list[str] = []
    existing_results_by_bucket: dict[int, dict[str, object]] = {}
    if args.benchmark_target and results_json_path.exists():
        try:
            previous_payload = json.loads(results_json_path.read_text(encoding="utf-8"))
            for item in previous_payload.get("results", []):
                bucket_size = item.get("bucketSize")
                if isinstance(bucket_size, int):
                    existing_results_by_bucket[bucket_size] = item
        except Exception:
            existing_results_by_bucket = {}
    results: list[dict[str, object]] = []
    benchmark_preflight: dict[str, object] | None = None

    if not args.skip_export:
        commands.extend(
            export_targets_via_subprocess(
                model_path,
                output_dir,
                skip_existing=args.skip_existing,
                chunk_postprocess_workers=args.chunk_postprocess_workers,
            )
        )

    if not args.skip_combine:
        sources = combine_chunk0_buckets(output_dir, skip_existing=args.skip_existing)
        commands.append(
            "_save_multifunction_dedup("
            + ", ".join(f"{Path(path).name}:{target}" for path, _, target in sources)
            + ")"
        )

    compiled_path = output_dir / "chunk0.mlmodelc"
    if not args.skip_compile:
        compile_combined_package(output_dir, skip_existing=args.skip_existing)
        commands.append(f"xcrun coremlcompiler compile {output_dir / 'chunk0.mlpackage'} {output_dir}")

    verification = verify_combined_package(output_dir)

    if not args.skip_copy:
        copied_path = copy_package_into_local_llm(
            output_dir / "chunk0.mlpackage",
            local_llm_root=local_llm_root,
            resource_name=args.resource_name,
        )
        commands.append(f"copy {output_dir / 'chunk0.mlpackage'} -> {copied_path}")

    if not args.skip_benchmark:
        benchmark_preflight = benchmark_device_preflight(
            local_llm_root=local_llm_root,
            device_id=args.device_id,
        )
        commands.append(
            "benchmark preflight "
            f"(device={args.device_id}, available={benchmark_preflight['available']}, "
            f"xctrace_state={benchmark_preflight['xctraceState']})"
        )
        if benchmark_preflight["available"]:
            benchmark_targets = (
                [args.benchmark_target]
                if args.benchmark_target
                else [f"prefill_bs{bucket}" for bucket in BUCKETS]
            )
            for function_name in benchmark_targets:
                print(f"[bench] {function_name}")
                result_bundle_path = result_bundles_dir / f"{function_name}.xcresult"
                if result_bundle_path.exists():
                    shutil.rmtree(result_bundle_path)
                result = benchmark_bucket_on_device(
                    local_llm_root=local_llm_root,
                    device_id=args.device_id,
                    resource_name=args.resource_name,
                    function_name=function_name,
                    warmups=args.warmups,
                    trials=args.trials,
                    result_bundle_path=result_bundle_path,
                    timeout_seconds=args.bucket_timeout_seconds,
                )
                existing_results_by_bucket[result["bucketSize"]] = result
                commands.append(
                    "xcodebuild test "
                    f"(function={function_name}, warmups={args.warmups}, trials={args.trials}, "
                    f"device={args.device_id}, timeout={args.bucket_timeout_seconds}s)"
                )
        else:
            print(f"[bench] Skipping device benchmarks: {benchmark_preflight['reason']}")

    results = [
        existing_results_by_bucket[bucket]
        for bucket in sorted(existing_results_by_bucket)
    ]

    artifact_sizes = {
        "combined_mlpackage_mb": dir_size_mb(output_dir / "chunk0.mlpackage"),
        "combined_mlmodelc_mb": dir_size_mb(compiled_path) if compiled_path.exists() else 0.0,
    }
    if results:
        recommendation = apply_recommendation_rules(results)
        recommendation_status = "complete"
    elif benchmark_preflight is not None and not benchmark_preflight["available"]:
        recommendation = {
            "keptBuckets": [],
            "droppedBuckets": [],
            "simulations": [],
        }
        recommendation_status = "pending_device_benchmark"
    else:
        recommendation = {
            "keptBuckets": [],
            "droppedBuckets": [],
            "simulations": [],
        }
        recommendation_status = "no_results"

    payload = {
        "timestamp": iso_now(),
        "config": {
            "contextLength": CTX,
            "chunkIndex": CHUNK_INDEX,
            "numChunks": NUM_CHUNKS,
            "lutBits": LUT_BITS,
            "chunkPostprocessWorkers": args.chunk_postprocess_workers,
            "buckets": BUCKETS,
            "deviceID": args.device_id,
            "resourceName": args.resource_name,
            "warmups": args.warmups,
            "trials": args.trials,
        },
        "artifactSizesMB": artifact_sizes,
        "verification": verification,
        "results": results,
        "recommendation": recommendation,
        "recommendationStatus": recommendation_status,
        "commands": commands,
        "benchmarkPreflight": benchmark_preflight,
    }
    results_md_path = output_dir / "results.md"
    results_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_results_markdown(
        results_md_path,
        artifact_sizes=artifact_sizes,
        verification=verification,
        results=results,
        recommendation=recommendation,
        commands=commands,
        benchmark_preflight=benchmark_preflight,
    )
    print(f"[done] Wrote {results_json_path}")
    print(f"[done] Wrote {results_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
