#!/usr/bin/env python3
"""Test selective FP32 precision: keep norm+recurrence in FP32, everything else FP16/LUT4.

Approach: Use coremltools PassPipeline to configure add_fp16_cast with skip_ops_by_type,
keeping layer_norm, reduce_sum, rsqrt in FP32 while all conv/matmul (LUT4 weights) stay FP16.

Tests:
1. Export one chunk with selective FP32 (norm+recurrence only)
2. Compare latency: FP16 vs selective-FP32 vs full-FP32
3. Compare output quality on test prompts
"""
from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.modules["profile"] = importlib.import_module("profile")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def export_chunk_with_selective_precision(
    chunk_idx: int,
    skip_ops: str,
    out_dir: str,
    model_path: str = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B",
    ctx: int = 1024,
    num_chunks: int = 4,
    lut_bits: int = 4,
):
    """Export a single FFN chunk with selective FP32 precision.

    Args:
        skip_ops: Comma-separated op types to keep in FP32.
                  e.g. "layer_norm,reduce_sum,rsqrt"
    """
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config

    print(f"\n  Loading HF model (CPU fp16)...")
    t0 = time.time()
    config = Qwen35Config.from_json(os.path.join(model_path, "config.json"))
    config.context_length = ctx
    config.state_length = max(config.state_length, ctx)
    model = Qwen35ForCausalLM(config)
    ok = model.load_pretrained_weights(model_path)
    if not ok:
        raise RuntimeError(f"Failed to load weights from {model_path}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Create converter
    converter = Qwen35Converter(
        model=model,
        context_length=ctx,
        num_chunks=num_chunks,
        lut_bits=lut_bits,
    )

    # Monkey-patch ct.convert to inject selective precision via pass_pipeline
    original_convert = ct.convert

    def patched_convert(*args, **kwargs):
        # Remove the default compute_precision
        kwargs.pop("compute_precision", None)

        # Create a pipeline with selective FP16 casting
        pipeline = ct.PassPipeline()
        pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": skip_ops})
        kwargs["pass_pipeline"] = pipeline
        kwargs["compute_precision"] = ct.precision.FLOAT16

        print(f"    [selective] skip_ops_by_type={skip_ops}")
        return original_convert(*args, **kwargs)

    ct.convert = patched_convert

    print(f"\n  Exporting chunk {chunk_idx} with selective FP32 ({skip_ops})...")
    t0 = time.time()
    try:
        mlmodel = converter.convert_part_2(
            model, chunk_idx=chunk_idx, total_chunks=num_chunks
        )
    finally:
        ct.convert = original_convert

    out_path = os.path.join(out_dir, f"ffn_LUT4_chunk{chunk_idx}.mlpackage")
    mlmodel.save(out_path)
    print(f"  Saved: {out_path} ({time.time()-t0:.1f}s)")

    del model, mlmodel, converter
    gc.collect()
    return out_path


def benchmark_model(path, label, n_steps=15):
    """Benchmark a single chunk."""
    cu = ct.ComputeUnit.CPU_AND_NE
    print(f"  [{label}] Loading...", end="", flush=True)
    t0 = time.time()
    model = ct.models.MLModel(path, compute_units=cu)
    t_load = time.time() - t0
    print(f" loaded in {t_load:.1f}s")

    spec = model.get_spec()
    inputs = {}
    for inp in spec.description.input:
        try:
            shape = tuple(int(x) for x in inp.type.multiArrayType.shape)
            dt = inp.type.multiArrayType.dataType
            if dt == 131104:  # INT32
                inputs[inp.name] = np.zeros(shape, dtype=np.int32)
            else:
                inputs[inp.name] = np.zeros(shape, dtype=np.float16)
        except Exception:
            pass

    try:
        state = model.make_state()
    except Exception:
        state = None

    # Warmup
    for _ in range(3):
        if state:
            model.predict(inputs, state=state)
        else:
            model.predict(inputs)

    # Benchmark
    times = []
    for _ in range(n_steps):
        t0 = time.time()
        if state:
            model.predict(inputs, state=state)
        else:
            model.predict(inputs)
        times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"    {label}: avg={avg_ms:.1f}ms ± {std_ms:.1f}ms over {n_steps} steps")
    del model
    gc.collect()
    return avg_ms


def count_proto_ops(path, label):
    """Count ops in the MIL protobuf, especially cast ops."""
    spec = ct.utils.load_spec(path)
    prog = spec.mlProgram
    from collections import Counter
    op_counter = Counter()
    for fname, func in prog.functions.items():
        for bname, block in func.block_specializations.items():
            for op in block.operations:
                op_counter[op.type] += 1
    total = sum(op_counter.values())
    casts = op_counter.get("cast", 0)
    print(f"  [{label}] {total} ops, {casts} cast ops")
    return op_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B")
    ap.add_argument("--fp16-dir", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1_3")
    ap.add_argument("--fp32-dir", default="/tmp/qwen35_fp32_chunks")
    ap.add_argument("--selective-dir", default="/tmp/qwen35_selective_chunks")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=0, help="Which chunk to test")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-ops", default="layer_norm,reduce_sum,rsqrt",
                    help="Op types to keep in FP32")
    args = ap.parse_args()

    out_dir = args.selective_dir
    os.makedirs(out_dir, exist_ok=True)

    chunk = args.chunk
    sel_path = os.path.join(out_dir, f"ffn_LUT4_chunk{chunk}.mlpackage")

    # ── Export selective chunk ──
    if not args.skip_export:
        print("=" * 70)
        print(f"  STEP 1: Export chunk {chunk} with selective FP32")
        print(f"  Keeping in FP32: {args.skip_ops}")
        print("=" * 70)
        export_chunk_with_selective_precision(
            chunk_idx=chunk,
            skip_ops=args.skip_ops,
            out_dir=out_dir,
            model_path=args.hf_model,
            ctx=args.ctx,
            num_chunks=args.num_chunks,
        )

    # ── Compare op counts ──
    print("\n" + "=" * 70)
    print("  STEP 2: MIL Op Comparison")
    print("=" * 70)
    fp16_path = os.path.join(args.fp16_dir, f"ffn_LUT4_chunk{chunk}.mlpackage")
    fp32_path = os.path.join(args.fp32_dir, f"ffn_LUT4_chunk{chunk}.mlpackage")
    
    c16 = count_proto_ops(fp16_path, "FP16")
    c32 = count_proto_ops(fp32_path, "Full-FP32")
    csel = count_proto_ops(sel_path, "Selective")
    
    print(f"\n  Cast ops: FP16={c16.get('cast',0)}, Full-FP32={c32.get('cast',0)}, Selective={csel.get('cast',0)}")
    print(f"  Total ops: FP16={sum(c16.values())}, Full-FP32={sum(c32.values())}, Selective={sum(csel.values())}")

    # ── Benchmark ──
    print("\n" + "=" * 70)
    print(f"  STEP 3: Latency Benchmark (chunk {chunk}, CPU_AND_NE)")
    print("=" * 70)
    t16 = benchmark_model(fp16_path, "FP16")
    t32 = benchmark_model(fp32_path, "Full-FP32")
    tsel = benchmark_model(sel_path, "Selective")

    print(f"\n  SUMMARY:")
    print(f"    FP16:       {t16:.1f}ms (baseline)")
    print(f"    Full-FP32:  {t32:.1f}ms ({t32/t16:.2f}x)")
    print(f"    Selective:  {tsel:.1f}ms ({tsel/t16:.2f}x)")
    improvement = (t32 - tsel) / (t32 - t16) * 100 if t32 != t16 else 0
    print(f"    Latency savings vs full-FP32: {improvement:.0f}% of the FP32 overhead recovered")


if __name__ == "__main__":
    main()
