#!/usr/bin/env python3
"""Qwen3.5-4B — Selective FP32 Precision Experiment.

Tests which MIL ops are sensitive to FP32 precision by exporting a single
FFN chunk with different selective-FP32 configurations and comparing outputs.

Variants:
  V0: baseline FP16   (all ops → FP16)
  V1: full FP32       (all ops → FP32)  [reference]
  V2: exp only        (exp → FP32)
  V3: exp+mul+add     (exp, mul, add → FP32)
  V4: exp+mul+add+matmul
  V5: recurrence full (exp, reduce_sum, log, relu, abs, sub, clip, rsqrt, split, mul)
  V6: attn-input      (layer_norm, softmax, matmul)
  V7: V5 + V6 combined
  V8: core recurrence (exp, reduce_sum, log, rsqrt)

Usage:
    cd /Users/yw68/Anemll && source .venv/bin/activate
    PYTHONPATH=. python tests/dev/test_qwen35_selective_fp32.py 2>&1 | tee tests/dev/selective_fp32_results.txt
"""
import os, sys, gc, time, importlib.util, warnings, shutil
import numpy as np
import torch

# ── Load config ──
_spec = importlib.util.spec_from_file_location(
    "qwen_config", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "../../scripts_qwen3_5/config.py"))
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
BATCH_SIZE, CTX, NUM_CHUNKS = _cfg.BATCH_SIZE, _cfg.CTX, _cfg.NUM_CHUNKS
CHUNK_RANGES = _cfg.CHUNK_RANGES
DEFAULT_HF_MODEL, PER_CHANNEL = _cfg.DEFAULT_HF_MODEL, _cfg.PER_CHANNEL

import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

# ── Config ──
CHUNK_IDX = 1
START_LAYER, END_LAYER = CHUNK_RANGES[CHUNK_IDX]
TMPDIR = "/tmp/qwen35_selective_fp32"
N_LATENCY = 5

VARIANTS = [
    ("V0_fp16",           "baseline FP16",            set()),
    ("V1_fp32",           "full FP32",                None),
    ("V2_exp",            "exp → FP32",               {"exp"}),
    ("V3_exp_mul_add",    "exp+mul+add → FP32",       {"exp", "mul", "add"}),
    ("V4_exp_mul_add_mm", "exp+mul+add+matmul → FP32",{"exp", "mul", "add", "matmul"}),
    ("V5_recurrence",     "recurrence full → FP32",   {"exp", "reduce_sum", "log", "relu", "abs", "sub", "clip", "rsqrt", "split", "mul"}),
    ("V6_attn_input",     "attn-input → FP32",        {"layer_norm", "softmax", "matmul"}),
    ("V7_V5_plus_V6",     "recurrence + attn → FP32", {"exp", "reduce_sum", "log", "relu", "abs", "sub", "clip", "rsqrt", "split", "mul", "layer_norm", "softmax", "matmul"}),
    ("V8_core_recurrence","core recurrence → FP32",   {"exp", "reduce_sum", "log", "rsqrt"}),
]


def make_compute_precision(fp32_ops):
    if fp32_ops is None:
        return ct.precision.FLOAT32
    if len(fp32_ops) == 0:
        return ct.precision.FLOAT16
    def op_selector(op):
        return op.op_type not in fp32_ops
    return ct.transform.FP16ComputePrecision(op_selector=op_selector)


def export_chunk(model, name, fp32_ops, out_dir):
    path = os.path.join(out_dir, f"{name}.mlpackage")
    if os.path.exists(path):
        shutil.rmtree(path)

    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=PER_CHANNEL,
        compute_precision="float16")
    conv.compute_precision = make_compute_precision(fp32_ops)

    t0 = time.time()
    ml = conv.convert_part_2(
        model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS,
        override_start_layer=START_LAYER, override_end_layer=END_LAYER)
    etime = time.time() - t0
    ml.save(path)

    sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fns in os.walk(path) for f in fns
             if not os.path.islink(os.path.join(dp, f))) / (1024 * 1024)
    del ml, conv; gc.collect()
    return path, etime, sz


def test_variant(path, name, ref_hidden, ref_rec, sample_inputs):
    result = dict(name=name, ane_loadable=False, fallback=False,
                  hidden_cos=0.0, hidden_l2=0.0, hidden_max=0.0,
                  rec_cos=0.0, rec_l2=0.0, lat_ms=0.0)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            result["ane_loadable"] = not any("error code: -14" in str(x.message) for x in w)
            result["fallback"] = any("fallback" in str(x.message).lower() for x in w)
    except Exception as e:
        print(f"    LOAD FAILED: {e}")
        return result

    try:
        state = ml.make_state()
        out = ml.predict(sample_inputs, state=state)
    except Exception as e:
        print(f"    PREDICT FAILED: {e}")
        return result

    h = out["output_hidden_states"].astype(np.float32).flatten()
    r = ref_hidden.flatten()
    dot = np.dot(r, h)
    result["hidden_cos"] = float(dot / (np.linalg.norm(r) * np.linalg.norm(h) + 1e-10))
    result["hidden_l2"] = float(np.linalg.norm(r - h))
    result["hidden_max"] = float(np.max(np.abs(r - h)))

    if "linear_recurrent_state_out" in out and ref_rec is not None:
        cr = out["linear_recurrent_state_out"].astype(np.float32).flatten()
        rr = ref_rec.flatten()
        if np.linalg.norm(rr) > 1e-10:
            result["rec_cos"] = float(np.dot(rr, cr) / (np.linalg.norm(rr) * np.linalg.norm(cr) + 1e-10))
            result["rec_l2"] = float(np.linalg.norm(rr - cr))

    lats = []
    for _ in range(N_LATENCY):
        s = ml.make_state()
        t0 = time.time()
        ml.predict(sample_inputs, state=s)
        lats.append((time.time() - t0) * 1000)
    result["lat_ms"] = float(np.median(lats))

    del ml; gc.collect()
    return result


def main():
    os.makedirs(TMPDIR, exist_ok=True)
    print("=" * 90)
    print("  Qwen3.5-4B — Selective FP32 Precision Experiment")
    print(f"  Chunk {CHUNK_IDX}: layers {START_LAYER}-{END_LAYER-1}")
    print(f"  Variants: {len(VARIANTS)}")
    print("=" * 90)

    # Load model
    print("\n[1] Loading model...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(DEFAULT_HF_MODEL, "config.json"))
    cfg.context_length = CTX; cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(DEFAULT_HF_MODEL)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"    Loaded in {time.time()-t0:.1f}s")

    # Sample inputs
    print("\n[2] Preparing inputs...")
    local_n = END_LAYER - START_LAYER
    hidden = torch.randn((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE) * 0.1
    pos_ids = torch.tensor([10], dtype=torch.int32, device=TEST_DEVICE)
    mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    mask[:, :, :, 11:] = -10000.0
    cpos = torch.tensor([10], dtype=torch.int32, device=TEST_DEVICE)

    # Linear state shapes
    if cfg.has_linear_attention():
        from anemll.ane_converter.qwen3_5_converter import ane_conv_state_shape
        tc = cfg.text_config
        conv_dim = tc.linear_num_key_heads * tc.linear_key_head_dim * 2 + tc.linear_num_value_heads * tc.linear_value_head_dim
        conv_kernel = max(1, int(tc.linear_conv_kernel_dim))
        d1, d2 = ane_conv_state_shape(conv_dim, conv_kernel)
        lc_shape = (local_n, d1, d2)
        lr_shape = (local_n, tc.linear_num_value_heads, tc.linear_key_head_dim, tc.linear_value_head_dim)
        print(f"    Linear attn: conv={lc_shape}, rec={lr_shape}")
    else:
        lc_shape = (local_n, 1, 1)
        lr_shape = (local_n, 1, 1, 1)
        print(f"    No linear attention in chunk {CHUNK_IDX}")

    feed = {
        "hidden_states": hidden.cpu().numpy(),
        "position_ids": pos_ids.cpu().numpy(),
        "causal_mask": mask.cpu().numpy(),
        "current_pos": cpos.cpu().numpy(),
        "linear_conv_state": np.zeros(lc_shape, dtype=np.float16),
        "linear_recurrent_state": np.zeros(lr_shape, dtype=np.float16),
    }

    # Export all variants (skip if already exists)
    print(f"\n[3] Exporting {len(VARIANTS)} variants...")
    paths = {}
    need_export = False
    for name, desc, fp32_ops in VARIANTS:
        path = os.path.join(TMPDIR, f"{name}.mlpackage")
        if os.path.exists(path):
            sz = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fns in os.walk(path) for f in fns
                     if not os.path.islink(os.path.join(dp, f))) / (1024 * 1024)
            paths[name] = (path, 0.0, sz)
            print(f"  [{name}] CACHED ({sz:.1f} MB)")
        else:
            need_export = True
            print(f"  [{name}] {desc}...")
            try:
                p, et, sz = export_chunk(model, name, fp32_ops, TMPDIR)
                paths[name] = (p, et, sz)
                print(f"    OK: {et:.1f}s, {sz:.1f} MB")
            except Exception as e:
                print(f"    FAILED: {e}")
                paths[name] = None
    del model; gc.collect()

    # Get FP32 reference output
    print("\n[4] Running FP32 reference inference...")
    fp32_info = paths.get("V1_fp32")
    if fp32_info is None:
        print("  ERROR: FP32 export failed, cannot continue.")
        return
    ml_ref = ct.models.MLModel(fp32_info[0], compute_units=ct.ComputeUnit.CPU_AND_NE)
    ref_state = ml_ref.make_state()
    ref_out = ml_ref.predict(feed, state=ref_state)
    ref_hidden = ref_out["output_hidden_states"].astype(np.float32)
    ref_rec = ref_out.get("linear_recurrent_state_out",
                          np.zeros(1)).astype(np.float32)
    print(f"    FP32 ref hidden norm: {np.linalg.norm(ref_hidden):.6f}")
    del ml_ref; gc.collect()

    # Test all variants
    print(f"\n[5] Testing {len(VARIANTS)} variants...")
    results = {}
    for name, desc, fp32_ops in VARIANTS:
        if paths.get(name) is None:
            results[name] = dict(name=name, failed=True)
            continue
        p, et, sz = paths[name]
        print(f"\n  [{name}] {desc}")
        r = test_variant(p, name, ref_hidden, ref_rec, feed)
        r["export_s"] = et
        r["size_mb"] = sz
        results[name] = r
        print(f"    ANE={r['ane_loadable']}  cos={r['hidden_cos']:.8f}  "
              f"L2={r['hidden_l2']:.4f}  max={r['hidden_max']:.4f}  "
              f"lat={r['lat_ms']:.1f}ms")

    # Results table
    print("\n" + "=" * 130)
    print("  ABLATION RESULTS TABLE  (reference = V1_fp32)")
    print("=" * 130)
    hdr = (f"{'Variant':<22s} {'Description':<28s} {'ANE':>3s} "
           f"{'Cos(hidden)':>13s} {'L2':>10s} {'MaxDiff':>10s} "
           f"{'RecCos':>10s} {'RecL2':>10s} {'Lat(ms)':>8s} {'Size':>7s}")
    print(hdr)
    print("-" * 130)
    for name, desc, _ in VARIANTS:
        r = results.get(name, {})
        if r.get("failed"):
            print(f"  {name:<22s} {desc:<28s}  FAILED")
            continue
        ane = "Y" if r.get("ane_loadable") else "N"
        print(f"  {name:<22s} {desc:<28s} {ane:>3s} "
              f"{r['hidden_cos']:>13.8f} {r['hidden_l2']:>10.4f} {r['hidden_max']:>10.4f} "
              f"{r['rec_cos']:>10.6f} {r['rec_l2']:>10.4f} {r['lat_ms']:>8.1f} {r['size_mb']:>6.0f}M")

    # Analysis
    print("\n" + "=" * 90)
    print("  ANALYSIS")
    print("=" * 90)
    v0 = results.get("V0_fp16", {})
    v1 = results.get("V1_fp32", {})
    fp16_cos = v0.get("hidden_cos", 0)
    fp32_cos = v1.get("hidden_cos", 0)
    fp16_lat = v0.get("lat_ms", 999)
    fp32_lat = v1.get("lat_ms", 999)
    gap = fp32_cos - fp16_cos

    print(f"\n  FP16 baseline: cos={fp16_cos:.8f}, lat={fp16_lat:.1f}ms")
    print(f"  FP32 full:     cos={fp32_cos:.8f}, lat={fp32_lat:.1f}ms")
    print(f"  Quality gap:   {gap:.8f}")

    best_name, best_score = None, -1
    for name, desc, fp32_ops in VARIANTS:
        if name in ("V0_fp16", "V1_fp32"): continue
        r = results.get(name, {})
        if r.get("failed") or not r.get("ane_loadable"): continue
        cos = r.get("hidden_cos", 0)
        lat = r.get("lat_ms", 999)
        recovery = (cos - fp16_cos) / (gap + 1e-10)
        slowdown = lat / (fp16_lat + 1e-10)
        score = recovery / slowdown if slowdown > 0 else 0
        print(f"  {name}: {recovery:.1%} quality recovered, {slowdown:.2f}x slowdown, score={score:.3f}")
        if score > best_score:
            best_score, best_name = score, name

    print(f"\n  >>> BEST TRADEOFF: {best_name} (score={best_score:.3f})" if best_name
          else "\n  >>> No valid selective variant.")

    print(f"\n  Files in: {TMPDIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
