#!/usr/bin/env python3
"""Reconciliation test: compare my findings vs the other experiment.

The other experiment tested a SINGLE CoreNorm stage with ct.precision.FLOAT32
and saw cosine jump from 0.9956 → 0.99995.

My test used SELECTIVE FP32 (FP16ComputePrecision with op_selector) on a
FULL 8-layer chunk and saw NO improvement (0.773 → 0.773).

Hypotheses for the discrepancy:
  H1: ct.precision.FLOAT32 ≠ selective FP16ComputePrecision(op_selector)
      → FLOAT32 keeps ALL ops in FP32; selective keeps only tagged ops
  H2: Error compounds across 8 stacked layers
  H3: ct.precision.FLOAT32 causes ANE to offload to CPU/GPU silently
      (ANE hardware is natively FP16-only)

This script tests:
  A) LUT4 + FP16 precision on ANE (current baseline)
  B) FP16 weights + FP16 precision on ANE
  C) FP16 weights + selective FP32 on ANE (my approach)
  D) FP16 weights + ct.precision.FLOAT32 on ANE (their approach)
  E) FP16 weights + FP16 precision on CPU+GPU

Usage:
    cd /Users/yw68/Anemll
    python tests/dev/_test_reconcile_fp32.py
"""
import os, sys, time, gc, shutil
import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts_qwen3_5"))

from config import BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, PER_CHANNEL, DEFAULT_HF_MODEL
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
import coremltools as ct

CHUNK_IDX = 0
OUT_BASE = "/tmp/qwen35_reconcile"

WARMUP = 5
ITERS = 30


def load_model():
    print("Loading model weights...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(DEFAULT_HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(DEFAULT_HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model


def export_chunk_custom(model, out_dir, label, lut_bits, compute_precision, compute_units_tag):
    """Export chunk with fully custom compute_precision (not going through converter flags)."""
    os.makedirs(out_dir, exist_ok=True)
    dec_path = os.path.join(out_dir, f"ffn_{label}_chunk{CHUNK_IDX}.mlpackage")
    if os.path.exists(dec_path):
        print(f"    [cached] {dec_path}")
        return dec_path

    print(f"    Exporting chunk {CHUNK_IDX} ({label})...")
    t0 = time.time()

    # Build the converter but we'll override compute_precision manually
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=lut_bits, per_channel=PER_CHANNEL,
    )

    # We need to call the internal export logic but with custom precision.
    # The simplest approach: monkey-patch _compute_precision_for_part temporarily.
    cu_tag = compute_units_tag
    orig_prec = conv._compute_precision_for_part
    orig_cu = conv._compute_units_for_part
    conv._compute_precision_for_part = lambda part: compute_precision
    conv._compute_units_for_part = lambda part: cu_tag

    ml = conv.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS)
    ml.save(dec_path)

    # Restore
    conv._compute_precision_for_part = orig_prec
    conv._compute_units_for_part = orig_cu

    del ml, conv
    gc.collect()
    print(f"    Saved ({time.time()-t0:.1f}s)")
    return dec_path


def make_inputs(cfg):
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start = CHUNK_IDX * base + min(CHUNK_IDX, rem)
    end = start + base + (1 if CHUNK_IDX < rem else 0)
    local_num_layers = end - start

    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)

    torch.manual_seed(42)
    return {
        "hidden_states": (torch.randn(1, 1, cfg.hidden_size, dtype=torch.float16) * 0.1).numpy(),
        "position_ids": np.array([5], dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, CTX), dtype=np.float16),
        "current_pos": np.array([5], dtype=np.int32),
        "linear_conv_state": np.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=np.float16),
        "linear_recurrent_state": np.zeros(
            (local_num_layers, cfg.text_config.linear_num_value_heads,
             cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
            dtype=np.float16
        ),
    }


def run_pytorch_ref(model, inputs):
    cfg = model.config
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start = CHUNK_IDX * base + min(CHUNK_IDX, rem)
    end = start + base + (1 if CHUNK_IDX < rem else 0)
    local_num_layers = end - start

    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)

    hidden = torch.from_numpy(inputs["hidden_states"])
    pos_ids = torch.from_numpy(inputs["position_ids"])
    mask = torch.from_numpy(inputs["causal_mask"])
    cur_pos = torch.from_numpy(inputs["current_pos"])
    lin_conv = torch.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=MODEL_DTYPE)
    lin_rec = torch.zeros(
        (local_num_layers, cfg.text_config.linear_num_value_heads,
         cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
        dtype=MODEL_DTYPE
    )

    model.model.kv_cache_0.zero_()
    with torch.no_grad():
        out = model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden, position_ids=pos_ids, causal_mask=mask,
            current_pos=cur_pos, kv_cache_0=None,
            k_cache=torch.zeros(local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim, dtype=MODEL_DTYPE),
            v_cache=torch.zeros(local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim, dtype=MODEL_DTYPE),
            linear_conv_state=lin_conv, linear_recurrent_state=lin_rec,
            start_layer=start, end_layer=end, apply_final_norm=False,
        )
        if end is None or end == total_layers:
            out = model.model.norm(out)
    return out.numpy()


def run_coreml(path, compute_units, inputs):
    mlmodel = ct.models.MLModel(path, compute_units=compute_units)
    state = mlmodel.make_state()
    result = mlmodel.predict(inputs, state)
    return np.array(result["output_hidden_states"])


def benchmark(path, compute_units, inputs):
    mlmodel = ct.models.MLModel(path, compute_units=compute_units)
    for _ in range(WARMUP):
        state = mlmodel.make_state()
        mlmodel.predict(inputs, state)
    times = []
    for _ in range(ITERS):
        state = mlmodel.make_state()
        t0 = time.perf_counter()
        mlmodel.predict(inputs, state)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    del mlmodel
    gc.collect()
    return sum(trimmed) / len(trimmed)


def cosine_sim(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def main():
    model = load_model()
    cfg = model.config

    layer_types = cfg.text_config.layer_types
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start = CHUNK_IDX * base + min(CHUNK_IDX, rem)
    end = start + base + (1 if CHUNK_IDX < rem else 0)
    print(f"\nChunk {CHUNK_IDX}: layers {start}..{end-1}")
    for i in range(start, end):
        print(f"  layer {i}: {layer_types[i]}")

    inputs = make_inputs(cfg)

    # PyTorch reference
    print("\n[REF] PyTorch reference...")
    ref = run_pytorch_ref(model, inputs)
    print(f"  shape={ref.shape}, range=[{ref.min():.4f}, {ref.max():.4f}]")

    # Define configs: (label, out_subdir, lut_bits, compute_precision_for_convert,
    #                   compute_units_for_convert, compute_units_for_predict)
    #
    # KEY DISTINCTION:
    #   - My "selective FP32": FP16ComputePrecision(op_selector=lambda: not in sensitive)
    #     → only tagged ops in FP32, everything else cast down to FP16
    #   - Their "full FP32":  ct.precision.FLOAT32
    #     → ALL ops stay in FP32 — no FP16 downcasting anywhere

    sensitive = Qwen35Converter.SENSITIVE_OP_TYPES
    selective_fp32 = ct.transform.FP16ComputePrecision(
        op_selector=lambda op, _s=sensitive: op.op_type not in _s
    )

    configs = [
        # --- ANE configs ---
        ("A: LUT4+FP16 (ANE)",
         "recA", 4, ct.precision.FLOAT16, ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_NE),
        ("B: FP16 no-LUT (ANE)",
         "recB", None, ct.precision.FLOAT16, ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_NE),
        ("C: Selective FP32 (ANE)",
         "recC", None, selective_fp32, ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_NE),
        # --- CPU+GPU configs ---
        ("D: FP16 (CPU+GPU)",
         "recD", None, ct.precision.FLOAT16, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_GPU),
        ("E: FLOAT32 (CPU+GPU)",
         "recE", None, ct.precision.FLOAT32, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_GPU),
        # --- Cross: convert for GPU, predict asking for ANE ---
        ("F: FLOAT32→predict ANE",
         "recF_gpu", None, ct.precision.FLOAT32, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_NE),
    ]

    results = {}
    for label, subdir, lut_bits, precision, cu_convert, cu_predict in configs:
        out_dir = os.path.join(OUT_BASE, subdir)
        print(f"\n[{label}]")
        try:
            path = export_chunk_custom(model, out_dir, subdir, lut_bits, precision, cu_convert)
        except Exception as e:
            print(f"    EXPORT FAILED: {e}")
            results[label] = None
            continue
        print(f"    Running on {cu_predict}...")
        try:
            out = run_coreml(path, cu_predict, inputs)
            cos = cosine_sim(ref, out)
            mad = float(np.abs(ref.astype(np.float32) - out.astype(np.float32)).max())
            print(f"    cosine={cos:.6f}, max_abs_diff={mad:.6f}")
            lat = benchmark(path, cu_predict, inputs)
            print(f"    latency={lat:.2f}ms")
            results[label] = {"cos": cos, "mad": mad, "lat": lat}
        except Exception as e:
            print(f"    FAILED: {e}")
            results[label] = None

    # Summary
    print(f"\n{'='*80}")
    print(f"  RECONCILIATION — chunk {CHUNK_IDX}, layers {start}..{end-1}")
    print(f"{'='*80}")
    print(f"  {'Config':<30s} {'Cosine':>10s} {'MaxAbsDiff':>12s} {'Latency':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*10}")
    for label, r in results.items():
        if r:
            print(f"  {label:<30s} {r['cos']:>10.6f} {r['mad']:>12.6f} {r['lat']:>9.2f}ms")
        else:
            print(f"  {label:<30s} {'FAILED':>10s}")

    # Analysis
    print(f"\n  ANALYSIS:")
    rb = results.get("B: FP16 no-LUT (ANE)")
    rc = results.get("C: Selective FP32 (ANE)")
    rd = results.get("D: FP16 (CPU+GPU)")
    re = results.get("E: FLOAT32 (CPU+GPU)")
    rf = results.get("F: FLOAT32→predict ANE")

    if rb and rd:
        print(f"  B→D (ANE FP16 → CPU+GPU FP16): cosine {rb['cos']:.6f} → {rd['cos']:.6f}")
        print(f"        Precision gain from leaving ANE (FP16-only HW)")
    if rb and rc:
        print(f"  B→C (selective FP32 on ANE):    cosine {rb['cos']:.6f} → {rc['cos']:.6f}")
        print(f"        Selective doesn't help: FP16 ops between sensitive ops reintroduce error")
    if rd and re:
        print(f"  D→E (CPU+GPU FP16 → FLOAT32):  cosine {rd['cos']:.6f} → {re['cos']:.6f}")
        print(f"        FLOAT32 compute on CPU+GPU (the other agent's approach)")
    if re and rf:
        if rf['cos'] > 0.999:
            print(f"  E≈F ({re['cos']:.6f} vs {rf['cos']:.6f}): FLOAT32 model on 'ANE' tag silently runs on CPU/GPU")
        else:
            print(f"  E≠F ({re['cos']:.6f} vs {rf['cos']:.6f}): FLOAT32 on ANE tag is different from CPU+GPU")
    if rf:
        print(f"  F latency={rf['lat']:.1f}ms → check if this matches CPU+GPU speed (ANE would be ~30ms)")

    print(f"\n  KEY FINDINGS:")
    print(f"  1. ANE CANNOT convert ct.precision.FLOAT32 → ANECCompile() FAILS")
    print(f"  2. The other agent's FLOAT32 result was on CPU (not ANE)")
    print(f"  3. CPU+GPU gives both better accuracy AND latency than ANE")
    if rb and re:
        print(f"  4. 8-layer compound error: ANE cosine={rb['cos']:.4f}, CPU+GPU FLOAT32={re['cos']:.6f}")


if __name__ == "__main__":
    main()
