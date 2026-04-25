#!/usr/bin/env python3
"""Chunk-1 combined experiment: E2+E3+E5 stacked together.

Tests the combination of all three individually-positive ideas:
  E2  ssm_alpha/ssm_beta → FP16
  E3  ssm_qkv/ssm_z/ssm_out → LUT6
  E5  ssm_qkv/ssm_z/ssm_out → group_size=2

Stacked variants:
  E0   baseline           — D2 policy: LUT4 gs=4, F-layer attn FP16
  E23  E2+E3              — SSM small FP16 + SSM proj LUT6
  E25  E2+E5              — SSM small FP16 + SSM proj gs=2
  E35  E3+E5              — SSM proj LUT6 gs=2
  E235 E2+E3+E5           — SSM small FP16 + SSM proj LUT6 gs=2

Usage:
    cd /Volumes/MySSD/Anemll
    python tests/dev/chunk1_combined_experiment.py --model models/Qwen__Qwen3.5-4B
"""
import argparse
import gc
import json
import os
import sys
import tempfile
import time
import warnings

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_QW_SCRIPTS = os.path.join(_REPO_ROOT, "scripts_qwen3_5")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _QW_SCRIPTS not in sys.path:
    sys.path.insert(0, _QW_SCRIPTS)

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES, FFN_PER_CHANNEL
from fp16_ablation import (
    discover_weight_ops,
    _match_op_to_family,
    _evaluate_chunk_quality,
    _load_model,
    _convert_chunk_fp16,
)

import coremltools as ct
import coremltools.optimize as cto

CHUNK_IDX = 1
N_EVAL = 16

D2_FP16_FAMILIES = ["attn_q", "attn_kv", "attn_o"]
SSM_FAMILIES_SMALL = ["ssm_alpha", "ssm_beta"]
SSM_FAMILIES_PROJ  = ["ssm_qkv", "ssm_z", "ssm_out"]


def _mlpackage_size_mb(mlmodel, label="model"):
    tmpdir = tempfile.mkdtemp(prefix=f"chunk1_{label}_")
    path = os.path.join(tmpdir, f"{label}.mlpackage")
    mlmodel.save(path)
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return total / (1024 * 1024)


def _measure_predict_time(mlmodel, n_iters=5):
    spec = mlmodel.get_spec()
    fn_inputs = None
    for fn in spec.description.functions:
        if fn.name == "infer":
            fn_inputs = fn.input
            break
    if fn_inputs is None:
        fn_inputs = spec.description.input

    input_shapes = {}
    for inp in fn_inputs:
        try:
            shape = tuple(inp.type.multiArrayType.shape)
            input_shapes[inp.name] = shape
        except Exception:
            pass

    np.random.seed(123)
    inputs = {}
    for name, shape in input_shapes.items():
        if name in ("position_ids", "current_pos"):
            inputs[name] = np.array([5], dtype=np.int32)
        elif name == "causal_mask":
            mask = np.full(shape, -65504.0, dtype=np.float16)
            mask[:, :, :, :6] = 0
            inputs[name] = mask
        else:
            inputs[name] = np.random.randn(*shape).astype(np.float16) * 0.1

    try:
        state = mlmodel.make_state()
        _ = mlmodel.predict(inputs, state=state)
    except Exception:
        try:
            _ = mlmodel.predict(inputs)
        except Exception:
            return -1.0

    times = []
    for _ in range(n_iters):
        try:
            state = mlmodel.make_state()
            t0 = time.perf_counter()
            _ = mlmodel.predict(inputs, state=state)
            times.append((time.perf_counter() - t0) * 1000)
        except Exception:
            t0 = time.perf_counter()
            _ = mlmodel.predict(inputs)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def _build_lut_config(mlmodel, fp16_families, lut_bits=4, per_channel=4,
                      lut6_families=None, gs2_families=None,
                      lut6_gs2_families=None):
    """Build OptimizationConfig with per-op overrides.

    lut6_gs2_families: families to use LUT6 + group_size=2 (E3+E5 stacked)
    """
    global_cfg = cto.coreml.OpPalettizerConfig(
        mode="kmeans", nbits=lut_bits,
        granularity="per_grouped_channel", group_size=per_channel,
        num_kmeans_workers=1,
    )
    config = cto.coreml.OptimizationConfig(global_config=global_cfg)

    fp16_set = set(fp16_families or [])
    lut6_set = set(lut6_families or [])
    gs2_set = set(gs2_families or [])
    lut6_gs2_set = set(lut6_gs2_families or [])

    ops = discover_weight_ops(mlmodel)
    for op_name, info in ops.items():
        family = info.get("family")
        if not family:
            continue

        if family in fp16_set:
            config.set_op_name(op_name, None)
        elif family in lut6_gs2_set:
            config.set_op_name(op_name, cto.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=6,
                granularity="per_grouped_channel",
                group_size=2,
                num_kmeans_workers=1,
            ))
        elif family in lut6_set:
            config.set_op_name(op_name, cto.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=6,
                granularity="per_grouped_channel",
                group_size=per_channel,
                num_kmeans_workers=1,
            ))
        elif family in gs2_set:
            config.set_op_name(op_name, cto.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=lut_bits,
                granularity="per_grouped_channel",
                group_size=2,
                num_kmeans_workers=1,
            ))

    return config


def run_experiment(name, fp16_ref, fp16_model_for_quant, desc, **quant_kwargs):
    print(f"\n{'─'*60}")
    print(f"  {name}: {desc}")
    print(f"{'─'*60}")

    t0 = time.time()
    config = _build_lut_config(fp16_model_for_quant, **quant_kwargs)

    print(f"  Palettizing...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quant_model = cto.coreml.palettize_weights(fp16_model_for_quant, config)
    t_palettize = time.time() - t0
    print(f"  Palettized in {t_palettize:.1f}s")

    print(f"  Evaluating quality ({N_EVAL} samples)...")
    t_q0 = time.time()
    quality = _evaluate_chunk_quality(fp16_ref, quant_model, CHUNK_IDX, N_EVAL)
    t_eval = time.time() - t_q0
    cos_sim = quality.get("avg_cosine_sim", 0)
    mse = quality.get("avg_mse", 0)
    max_err = quality.get("avg_max_error", 0)
    print(f"  cos_sim={cos_sim:.6f}  mse={mse:.8f}  max_err={max_err:.6f}  ({t_eval:.1f}s)")

    print(f"  Measuring size...")
    size_mb = _mlpackage_size_mb(quant_model, label=name)
    print(f"  size={size_mb:.1f} MB")

    print(f"  Measuring predict time...")
    predict_ms = _measure_predict_time(quant_model, n_iters=5)
    print(f"  predict_time={predict_ms:.1f} ms")

    result = {
        "name": name,
        "desc": desc,
        "cos_sim": cos_sim,
        "mse": mse,
        "max_err": max_err,
        "size_mb": size_mb,
        "predict_ms": predict_ms,
        "palettize_time_s": t_palettize,
    }

    del quant_model
    gc.collect()
    return result


def _print_summary(results, baseline_cos, baseline_size, baseline_ms):
    print(f"\n{'═'*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*80}")
    header = f"  {'Name':<20} {'cos_sim':>10} {'Δcos':>8} {'size_MB':>10} {'Δsize':>8} {'ms':>8} {'Δms':>8}"
    print(header)
    print(f"  {'─'*76}")
    for r in results:
        d_cos = r['cos_sim'] - baseline_cos if r['name'] != 'FP16_ref' else 0
        d_size = r['size_mb'] - baseline_size if r['name'] != 'FP16_ref' else 0
        d_ms = r['predict_ms'] - baseline_ms if r['name'] != 'FP16_ref' else 0
        print(f"  {r['name']:<20} {r['cos_sim']:>10.6f} {d_cos:>+8.4f} {r['size_mb']:>10.1f} {d_size:>+8.1f} {r['predict_ms']:>8.1f} {d_ms:>+8.1f}")

    # Rank (exclude FP16_ref and E0_baseline)
    candidates = [r for r in results if r['name'] not in ('FP16_ref', 'E0_baseline')]
    if candidates:
        candidates.sort(key=lambda r: r['cos_sim'], reverse=True)
        print(f"\n  RANKING (by cos_sim):")
        for i, r in enumerate(candidates, 1):
            d_cos = r['cos_sim'] - baseline_cos
            d_size = r['size_mb'] - baseline_size
            print(f"    #{i} {r['name']:<20} Δcos={d_cos:+.4f}  Δsize={d_size:+.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Chunk-1 combined E2+E3+E5 experiment")
    parser.add_argument("--model", default="models/Qwen__Qwen3.5-4B")
    args = parser.parse_args()

    print("=" * 70)
    print("  Chunk-1 Combined Experiment: E2+E3+E5 stacked")
    print(f"  Chunk {CHUNK_IDX}: layers {CHUNK_RANGES[CHUNK_IDX]}")
    print(f"  Eval samples: {N_EVAL}")
    print(f"  Model: {args.model}")
    print("=" * 70)

    print("\n  Loading model...")
    t0 = time.time()
    model = _load_model(args.model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\n  Converting chunk 1 to FP16 reference...")
    t0 = time.time()
    fp16_ref = _convert_chunk_fp16(model, CHUNK_IDX)
    print(f"  FP16 reference ready ({time.time()-t0:.1f}s)")

    print("\n  Measuring FP16 reference size & speed...")
    fp16_size = _mlpackage_size_mb(fp16_ref, "fp16_ref")
    fp16_predict = _measure_predict_time(fp16_ref, n_iters=5)
    print(f"  FP16 ref: size={fp16_size:.1f} MB, predict={fp16_predict:.1f} ms")

    results = []
    results.append({
        "name": "FP16_ref",
        "desc": "FP16 reference (no quantization)",
        "cos_sim": 1.0, "mse": 0.0, "max_err": 0.0,
        "size_mb": fp16_size, "predict_ms": fp16_predict,
        "palettize_time_s": 0,
    })

    # ── E0: BASELINE ──
    r = run_experiment(
        "E0_baseline", fp16_ref, fp16_ref,
        "Current D2: LUT4 gs=4, F-layer attn Q/K/V/O → FP16",
        fp16_families=D2_FP16_FAMILIES,
        lut_bits=4, per_channel=4,
    )
    results.append(r)
    baseline_cos = r["cos_sim"]
    baseline_size = r["size_mb"]
    baseline_ms = r["predict_ms"]

    # ── E2: ssm_alpha/beta FP16 (individual, for comparison) ──
    r = run_experiment(
        "E2_ssm_fp16", fp16_ref, fp16_ref,
        "D2 + ssm_alpha/ssm_beta → FP16",
        fp16_families=D2_FP16_FAMILIES + SSM_FAMILIES_SMALL,
        lut_bits=4, per_channel=4,
    )
    results.append(r)

    # ── E3: SSM proj LUT6 (individual, for comparison) ──
    r = run_experiment(
        "E3_mixed_lut6", fp16_ref, fp16_ref,
        "D2 + LUT6 for ssm_qkv/ssm_z/ssm_out",
        fp16_families=D2_FP16_FAMILIES,
        lut_bits=4, per_channel=4,
        lut6_families=SSM_FAMILIES_PROJ,
    )
    results.append(r)

    # ── E5: SSM proj gs=2 (individual, for comparison) ──
    r = run_experiment(
        "E5_gs2_ssm", fp16_ref, fp16_ref,
        "D2 + group_size=2 for ssm_qkv/ssm_z/ssm_out",
        fp16_families=D2_FP16_FAMILIES,
        lut_bits=4, per_channel=4,
        gs2_families=SSM_FAMILIES_PROJ,
    )
    results.append(r)

    # ── E23: E2+E3 — SSM small FP16 + SSM proj LUT6 ──
    r = run_experiment(
        "E23_fp16+lut6", fp16_ref, fp16_ref,
        "D2 + ssm_alpha/beta FP16 + ssm_proj LUT6",
        fp16_families=D2_FP16_FAMILIES + SSM_FAMILIES_SMALL,
        lut_bits=4, per_channel=4,
        lut6_families=SSM_FAMILIES_PROJ,
    )
    results.append(r)

    # ── E25: E2+E5 — SSM small FP16 + SSM proj gs=2 ──
    r = run_experiment(
        "E25_fp16+gs2", fp16_ref, fp16_ref,
        "D2 + ssm_alpha/beta FP16 + ssm_proj gs=2",
        fp16_families=D2_FP16_FAMILIES + SSM_FAMILIES_SMALL,
        lut_bits=4, per_channel=4,
        gs2_families=SSM_FAMILIES_PROJ,
    )
    results.append(r)

    # ── E35: E3+E5 — SSM proj LUT6 + gs=2 ──
    r = run_experiment(
        "E35_lut6+gs2", fp16_ref, fp16_ref,
        "D2 + ssm_proj LUT6 gs=2",
        fp16_families=D2_FP16_FAMILIES,
        lut_bits=4, per_channel=4,
        lut6_gs2_families=SSM_FAMILIES_PROJ,
    )
    results.append(r)

    # ── E235: E2+E3+E5 — SSM small FP16 + SSM proj LUT6 gs=2 ──
    r = run_experiment(
        "E235_all", fp16_ref, fp16_ref,
        "D2 + ssm_alpha/beta FP16 + ssm_proj LUT6 gs=2",
        fp16_families=D2_FP16_FAMILIES + SSM_FAMILIES_SMALL,
        lut_bits=4, per_channel=4,
        lut6_gs2_families=SSM_FAMILIES_PROJ,
    )
    results.append(r)

    # ── Summary ──
    _print_summary(results, baseline_cos, baseline_size, baseline_ms)

    out_dir = os.path.join(_REPO_ROOT, "tests", "dev")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chunk1_combined_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    del fp16_ref, model
    gc.collect()


if __name__ == "__main__":
    main()
