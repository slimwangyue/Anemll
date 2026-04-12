#!/usr/bin/env python3
"""Test: Export L-layer chunks with force_fp16_math=True and measure ANE%.

Hypothesis: L-layer recurrence in FP32 forces ~50% of ops to CPU.
With FP16 math, the recurrence should stay on ANE → higher utilization.

Approach:
  1. Monkey-patch L-layer forward methods to inject force_fp16_math=True
  2. Export chunk 0 (LLL) in two variants: default (fp32 math) vs fp16 math
  3. Measure ANE% for both
  4. Compare accuracy: run same input through both and check cosine similarity
  5. If chunk 0 works, test an FLLL chunk too

Uses existing V4 precision policy for F layers (kv_cache ops stay fp32).
"""
import argparse
import gc
import os
import resource
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

REPO_ROOT = "/Volumes/MySSD/Anemll"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
os.chdir(REPO_ROOT)

import coremltools as ct
from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

# F (full attention) layer indices for Qwen3.5-4B
F_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}

HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
OUTPUT_DIR = os.path.join(REPO_ROOT, "artifacts", "l_layer_fp16_math_ane")
HIDDEN = 2560
WARMUP = 10
RUNS = 30


def load_model():
    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


def patch_l_layers_fp16_math(model, start_layer, end_layer, enable=True):
    """Monkey-patch L-layer forward methods to inject force_fp16_math."""
    patched = []
    cfg = model.config
    layer_types = cfg.text_config.layer_types
    for li in range(start_layer, end_layer):
        if layer_types[li] == "linear_attention":
            layer = model.model.layers[li]
            attn = layer.self_attn
            # Patch forward_regular (decode path)
            orig_reg = attn.forward_regular
            def make_patched_reg(orig):
                def patched(*args, **kwargs):
                    kwargs["force_fp16_math"] = enable
                    return orig(*args, **kwargs)
                return patched
            attn.forward_regular = make_patched_reg(orig_reg)
            patched.append((attn, "forward_regular", orig_reg))
            # Patch forward_prefill_export (prefill path)
            orig_pre = attn.forward_prefill_export
            def make_patched_pre(orig):
                def patched(*args, **kwargs):
                    kwargs["force_fp16_math"] = enable
                    return orig(*args, **kwargs)
                return patched
            attn.forward_prefill_export = make_patched_pre(orig_pre)
            patched.append((attn, "forward_prefill_export", orig_pre))
    return patched


def restore_patches(patched):
    for obj, name, orig in patched:
        setattr(obj, name, orig)


def export_chunk(model, chunk_idx, label, skip_existing=False):
    """Export a single chunk decode model."""
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
    out_path = os.path.join(OUTPUT_DIR, f"chunk{chunk_idx}_{label}_decode.mlpackage")
    if skip_existing and os.path.exists(out_path):
        print(f"  [skip] {out_path}")
        return out_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
        compute_precision="float16",
    )
    ml = conv.convert_part_2(
        model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
        override_start_layer=CHUNK_RANGES[chunk_idx][0],
        override_end_layer=CHUNK_RANGES[chunk_idx][1],
    )
    ml.save(out_path)
    elapsed = time.time() - t0
    print(f"  Exported {label} decode chunk{chunk_idx} in {elapsed:.1f}s")
    del ml, conv
    gc.collect()
    return out_path


def export_chunk_prefill(model, chunk_idx, label, skip_existing=False):
    """Export a single chunk prefill model."""
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
    out_path = os.path.join(OUTPUT_DIR, f"chunk{chunk_idx}_{label}_prefill.mlpackage")
    if skip_existing and os.path.exists(out_path):
        print(f"  [skip] {out_path}")
        return out_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
        compute_precision="float16",
    )
    ml = conv.convert_part_2_prefill(
        model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
        override_start_layer=CHUNK_RANGES[chunk_idx][0],
        override_end_layer=CHUNK_RANGES[chunk_idx][1],
    )
    ml.save(out_path)
    elapsed = time.time() - t0
    print(f"  Exported {label} prefill chunk{chunk_idx} in {elapsed:.1f}s")
    del ml, conv
    gc.collect()
    return out_path


def make_decode_inputs(chunk_idx):
    """Create decode inputs for a given chunk."""
    nl = CHUNK_RANGES[chunk_idx][1] - CHUNK_RANGES[chunk_idx][0]
    start, end = CHUNK_RANGES[chunk_idx]
    has_f = any(l in F_LAYERS for l in range(start, end))
    num_f = sum(1 for l in range(start, end) if l in F_LAYERS)
    pred = {
        "hidden_states": np.random.randn(1, 1, HIDDEN).astype(np.float16),
        "position_ids": np.array([CTX // 2], dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, CTX), dtype=np.float16),
        "current_pos": np.array([CTX // 2], dtype=np.int32),
        "linear_conv_state": np.zeros((nl, 1024, 32), dtype=np.float16),
        "linear_recurrent_state": np.zeros((nl, 32, 128, 128), dtype=np.float16),
    }
    return pred


def measure_ane(model_path, pred, label):
    """Load model, measure ANE% via CPU time method."""
    try:
        ml = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = ml.make_state()
        # Warmup
        for _ in range(WARMUP):
            ml.predict(pred, state=state)
        times, cpus = [], []
        for _ in range(RUNS):
            r0 = resource.getrusage(resource.RUSAGE_SELF)
            t0 = time.perf_counter()
            ml.predict(pred, state=state)
            t1 = time.perf_counter()
            r1 = resource.getrusage(resource.RUSAGE_SELF)
            wall = t1 - t0
            cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
            times.append(wall)
            cpus.append(cpu)
        w = np.median(times) * 1000
        c = np.median(cpus) * 1000
        cp = c / w * 100
        ane = max(0, 100 - cp)
        print(f"  {label}: wall={w:.2f}ms  cpu={c:.2f}ms  CPU%={cp:.1f}%  ANE%={ane:.1f}%")
        return ml, state, {"wall_ms": w, "cpu_ms": c, "cpu_pct": cp, "ane_pct": ane}
    except Exception as e:
        print(f"  {label}: FAILED — {e}")
        return None, None, None


def compare_outputs(ml_a, state_a, ml_b, state_b, pred, label_a, label_b):
    """Compare two models on the same input."""
    out_a = ml_a.predict(pred, state=state_a)
    out_b = ml_b.predict(pred, state=state_b)
    for key in ["output_hidden_states"]:
        if key in out_a and key in out_b:
            a = np.asarray(out_a[key]).flatten().astype(np.float64)
            b = np.asarray(out_b[key]).flatten().astype(np.float64)
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            max_abs = float(np.max(np.abs(a - b)))
            print(f"  {label_a} vs {label_b} [{key}]: cos={cos:.8f}  max_abs_diff={max_abs:.6f}")
    # Also compare states
    for key in out_a:
        if "state" in key.lower() and key in out_b:
            a = np.asarray(out_a[key]).flatten().astype(np.float64)
            b = np.asarray(out_b[key]).flatten().astype(np.float64)
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            print(f"  {label_a} vs {label_b} [{key}]: cos={cos:.8f}")


def run_experiment(args):
    print("=" * 70)
    print("  L-Layer FP16 Math → ANE Utilization Experiment")
    print(f"  CTX={CTX}, BATCH_SIZE={BATCH_SIZE}, NUM_CHUNKS={NUM_CHUNKS}")
    print("=" * 70)

    # Load model once
    print("\nLoading model...")
    model, cfg = load_model()

    chunks_to_test = args.chunks
    results = {}

    for ci in chunks_to_test:
        start, end = CHUNK_RANGES[ci]
        layer_types = cfg.text_config.layer_types[start:end]
        pattern = "".join("F" if l in F_LAYERS else "L" for l in range(start, end))
        print(f"\n{'='*70}")
        print(f"  Chunk {ci}: layers {start}-{end-1}, pattern={pattern}")
        print(f"{'='*70}")

        # --- Variant A: Default (FP32 L-math, FP16 compute) ---
        print(f"\n[A] Exporting with default math (FP32 recurrence)...")
        path_a = export_chunk(model, ci, "default", skip_existing=args.skip_existing)

        # --- Variant B: FP16 L-math ---
        print(f"\n[B] Exporting with FP16 L-layer math...")
        patches = patch_l_layers_fp16_math(model, start, end, enable=True)
        path_b = export_chunk(model, ci, "fp16math", skip_existing=args.skip_existing)
        restore_patches(patches)

        # --- Measure ANE% ---
        print(f"\n  Measuring ANE utilization...")
        pred = make_decode_inputs(ci)
        ml_a, state_a, res_a = measure_ane(path_a, pred, f"chunk{ci}_default")
        ml_b, state_b, res_b = measure_ane(path_b, pred, f"chunk{ci}_fp16math")

        # --- Compare accuracy ---
        if ml_a is not None and ml_b is not None:
            print(f"\n  Accuracy comparison (same input):")
            # Use deterministic input
            det_pred = make_decode_inputs(ci)
            det_pred["hidden_states"] = np.ones((1, 1, HIDDEN), dtype=np.float16) * 0.01
            compare_outputs(ml_a, state_a, ml_b, state_b, det_pred, "default", "fp16math")

        results[ci] = {"default": res_a, "fp16math": res_b, "pattern": pattern}

        # Cleanup
        del ml_a, ml_b, state_a, state_b
        gc.collect()

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Chunk':>6} {'Pattern':>8} {'Default ANE%':>14} {'FP16math ANE%':>14} {'Default Wall':>13} {'FP16math Wall':>14} {'Improvement':>12}")
    for ci, r in sorted(results.items()):
        d = r["default"]
        f = r["fp16math"]
        if d and f:
            imp = f["ane_pct"] - d["ane_pct"]
            print(f"  {ci:>4}   {r['pattern']:>8}  {d['ane_pct']:>10.1f}%   {f['ane_pct']:>10.1f}%   {d['wall_ms']:>10.2f}ms  {f['wall_ms']:>10.2f}ms  {imp:>+9.1f}%")
        else:
            status_d = "FAIL" if not d else f"{d['ane_pct']:.1f}%"
            status_f = "FAIL" if not f else f"{f['ane_pct']:.1f}%"
            print(f"  {ci:>4}   {r['pattern']:>8}  {status_d:>14} {status_f:>14}")


def main():
    parser = argparse.ArgumentParser(description="L-layer FP16 math ANE experiment")
    parser.add_argument("--chunks", type=str, default="0",
                        help="Comma-separated chunk indices to test (default: 0)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if .mlpackage already exists")
    args = parser.parse_args()
    args.chunks = [int(x) for x in args.chunks.split(",")]
    run_experiment(args)


if __name__ == "__main__":
    main()
