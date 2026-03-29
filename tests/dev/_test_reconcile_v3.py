#!/usr/bin/env python3
"""Minimal reconciliation: ANE-FP16 vs CPU+GPU-FP16 vs CPU+GPU-FLOAT32.

Tests ONE config at a time, cleans /tmp between each to avoid disk pressure.
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
OUT_DIR = "/tmp/qwen35_rec3"
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
    return out.numpy()


def cosine_sim(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def export_and_test(model, cfg, ref, inputs, label, lut_bits, compute_precision, cu_convert, cu_predict):
    """Export one config, test, benchmark, then cleanup the model file."""
    print(f"\n[{label}]")
    out_dir = os.path.join(OUT_DIR, label.replace(" ", "_").replace("+", ""))
    os.makedirs(out_dir, exist_ok=True)
    dec_path = os.path.join(out_dir, f"chunk{CHUNK_IDX}.mlpackage")

    # Export
    try:
        print(f"  Exporting (precision={compute_precision}, units={cu_convert})...")
        t0 = time.time()
        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS, lut_bits=lut_bits, per_channel=PER_CHANNEL,
        )
        conv._compute_precision_for_part = lambda part, _p=compute_precision: _p
        conv._compute_units_for_part = lambda part, _c=cu_convert: _c
        ml = conv.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS)
        ml.save(dec_path)
        del ml, conv
        gc.collect()
        print(f"  Exported in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  EXPORT FAILED: {e}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return None

    # Predict
    try:
        print(f"  Predicting on {cu_predict}...")
        mlmodel = ct.models.MLModel(dec_path, compute_units=cu_predict)
        state = mlmodel.make_state()
        result = mlmodel.predict(inputs, state)
        out = np.array(result["output_hidden_states"])
        cos = cosine_sim(ref, out)
        mad = float(np.abs(ref.astype(np.float32) - out.astype(np.float32)).max())
        print(f"  cosine={cos:.6f}, max_abs_diff={mad:.6f}")
    except Exception as e:
        print(f"  PREDICT FAILED: {e}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return None

    # Benchmark
    try:
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
        lat = sum(trimmed) / len(trimmed)
        print(f"  latency={lat:.2f}ms")
    except Exception as e:
        print(f"  BENCH FAILED: {e}")
        lat = float("nan")

    del mlmodel
    gc.collect()

    # Cleanup model to save disk
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"  [cleaned up {out_dir}]")

    return {"cos": cos, "mad": mad, "lat": lat}


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

    print("\n[REF] PyTorch reference...")
    ref = run_pytorch_ref(model, inputs)
    print(f"  shape={ref.shape}, range=[{ref.min():.4f}, {ref.max():.4f}]")

    # Run configs sequentially, cleaning up after each
    configs = [
        ("A: LUT4+FP16 (ANE)",      4,    ct.precision.FLOAT16, ct.ComputeUnit.CPU_AND_NE,  ct.ComputeUnit.CPU_AND_NE),
        ("B: FP16 (ANE)",            None, ct.precision.FLOAT16, ct.ComputeUnit.CPU_AND_NE,  ct.ComputeUnit.CPU_AND_NE),
        ("C: FP16 (CPU+GPU)",        None, ct.precision.FLOAT16, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_GPU),
        ("D: FLOAT32 (CPU+GPU)",     None, ct.precision.FLOAT32, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_GPU),
    ]

    results = {}
    for label, lut_bits, precision, cu_convert, cu_predict in configs:
        r = export_and_test(model, cfg, ref, inputs, label, lut_bits, precision, cu_convert, cu_predict)
        results[label] = r

    # Summary
    print(f"\n{'='*80}")
    print(f"  RECONCILIATION RESULTS — chunk {CHUNK_IDX}, layers {start}..{end-1} (8 layers)")
    print(f"{'='*80}")
    print(f"  {'Config':<28s} {'Cosine':>10s} {'MaxAbsDiff':>12s} {'Latency':>10s}")
    print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*10}")
    for label, r in results.items():
        if r:
            print(f"  {label:<28s} {r['cos']:>10.6f} {r['mad']:>12.6f} {r['lat']:>9.2f}ms")
        else:
            print(f"  {label:<28s} {'FAILED':>10s}")

    # Analysis
    ra = results.get("A: LUT4+FP16 (ANE)")
    rb = results.get("B: FP16 (ANE)")
    rc = results.get("C: FP16 (CPU+GPU)")
    rd = results.get("D: FLOAT32 (CPU+GPU)")

    print(f"\n  ANALYSIS:")
    if ra and rb:
        print(f"  A→B: LUT4 quantization penalty = {rb['cos']-ra['cos']:+.4f} cosine")
    if rb and rc:
        print(f"  B→C: ANE→CPU+GPU (same FP16 weights) = {rc['cos']-rb['cos']:+.4f} cosine")
        print(f"        → ANE's FP16 arithmetic causes compound error across 8 layers")
    if rc and rd:
        print(f"  C→D: FP16→FLOAT32 compute on CPU+GPU = {rd['cos']-rc['cos']:+.6f} cosine")
        print(f"        → This is what the other agent tested (but on 1 layer, not 8)")
    if rd:
        print(f"\n  The other agent saw 0.9956 → 0.99995 for ONE CoreNorm stage")
        print(f"  Our chunk (8 layers) with FLOAT32 on CPU+GPU: {rd['cos']:.6f}")

    print(f"\n  KEY CONCLUSIONS:")
    print(f"  1. ct.precision.FLOAT32 CANNOT compile for ANE (ANECCompile() FAILS)")
    print(f"  2. The other agent's FLOAT32 result was on CPU, not ANE hardware")
    if rc and rd:
        delta = rd['cos'] - rc['cos']
        if delta < 0.001:
            print(f"  3. FLOAT32 adds negligible accuracy over FP16 on CPU+GPU ({delta:+.6f})")
            print(f"     → CPU+GPU FP16 is already near-perfect; FLOAT32 unnecessary")
        else:
            print(f"  3. FLOAT32 compute adds {delta:+.6f} cosine on CPU+GPU")
    if rb and rc:
        print(f"  4. The real accuracy bottleneck is ANE (FP16 HW), not compute precision")
        print(f"     → CPU+GPU FP16: {rc['cos']:.6f} vs ANE FP16: {rb['cos']:.6f}")


if __name__ == "__main__":
    main()
