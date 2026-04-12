#!/usr/bin/env python3
"""
Measure ANE utilization for all V4 9-chunk models (decode + prefill).

Uses CPU-time proxy: ANE% = 100 - (cpu_time / wall_time * 100).
"""
import gc
import os
import resource
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_qwen3_5"))
os.chdir(REPO)

import coremltools as ct

from config import BATCH_SIZE, CTX, CHUNK_RANGES

HIDDEN = 2560
NUM_KV_HEADS = 4
HEAD_DIM = 256
MODEL_DIR = os.path.join(REPO, "artifacts", "v4_all_chunks", "assembled")
WARMUP = 10
RUNS = 30

# Number of layers per chunk
CHUNK_NLAYERS = [el - sl for sl, el in CHUNK_RANGES]


def measure(path, name, pred, use_state=False):
    """Load and measure ANE utilization."""
    try:
        ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e:
        print(f"  {name:45s} LOAD FAIL: {e}")
        return None

    state = None
    if use_state:
        try:
            state = ml.make_state()
        except Exception as e:
            print(f"  {name:45s} STATE FAIL: {e}")
            return None

    # Warmup
    for _ in range(WARMUP):
        ml.predict(pred, state=state) if state else ml.predict(pred)

    times, cpus = [], []
    for _ in range(RUNS):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        ml.predict(pred, state=state) if state else ml.predict(pred)
        t1 = time.perf_counter()
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        times.append(t1 - t0)
        cpus.append((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))

    w = np.median(times) * 1000
    c = np.median(cpus) * 1000
    cp = c / w * 100 if w > 0 else 0
    ane = max(0, 100 - cp)
    print(f"  {name:45s} wall={w:7.2f}ms cpu={c:7.2f}ms CPU%={cp:5.1f}% ANE%={ane:5.1f}%")
    del ml
    gc.collect()
    return {"wall": w, "cpu": c, "cpu_pct": cp, "ane_pct": ane}


def make_ffn_decode_pred(chunk_idx):
    """Build prediction dict for a decode chunk (seq_len=1)."""
    nl = CHUNK_NLAYERS[chunk_idx]
    return {
        "hidden_states": np.random.randn(1, 1, HIDDEN).astype(np.float16),
        "position_ids": np.array([CTX // 2], dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, CTX), dtype=np.float16),
        "current_pos": np.array([CTX // 2], dtype=np.int32),
        "linear_conv_state": np.zeros((nl, 1024, 32), dtype=np.float16),
        "linear_recurrent_state": np.zeros((nl, 32, 128, 128), dtype=np.float16),
    }


def make_ffn_prefill_pred(chunk_idx):
    """Build prediction dict for a prefill chunk (seq_len=BATCH_SIZE)."""
    nl = CHUNK_NLAYERS[chunk_idx]
    return {
        "hidden_states": np.random.randn(1, BATCH_SIZE, HIDDEN).astype(np.float16),
        "position_ids": np.arange(BATCH_SIZE, dtype=np.int32).reshape(1, -1),
        "causal_mask": np.zeros((1, 1, BATCH_SIZE, CTX), dtype=np.float16),
        "current_pos": np.array([0], dtype=np.int32),
        "linear_conv_state": np.zeros((nl, 1024, 32), dtype=np.float16),
        "linear_recurrent_state": np.zeros((nl, 32, 128, 128), dtype=np.float16),
    }


def main():
    print("=" * 80)
    print("  V4 ALL-CHUNKS ANE UTILIZATION MEASUREMENT")
    print(f"  Model dir: {MODEL_DIR}")
    print(f"  CTX={CTX}, BATCH={BATCH_SIZE}")
    print("=" * 80)

    results = {}

    # ── Embed (single token) ──
    print("\n[EMBED]")
    embed_path = os.path.join(MODEL_DIR, "embed_single.mlpackage")
    if os.path.exists(embed_path):
        pred = {"input_ids": np.array([[1]], dtype=np.int32)}
        results["embed_single"] = measure(embed_path, "embed_single", pred)

    # ── Embed (prefill) ──
    embed_pf_path = os.path.join(MODEL_DIR, "embed_prefill.mlpackage")
    if os.path.exists(embed_pf_path):
        pred = {"input_ids": np.ones((1, BATCH_SIZE), dtype=np.int32)}
        results["embed_prefill"] = measure(embed_pf_path, "embed_prefill", pred)

    # ── FFN decode chunks (seq_len=1, with state) ──
    print("\n[FFN DECODE — seq_len=1]")
    for i in range(9):
        name = f"ffn_LUT4_chunk{i}"
        path = os.path.join(MODEL_DIR, f"{name}.mlpackage")
        if not os.path.exists(path):
            print(f"  {name:45s} MISSING")
            continue
        pred = make_ffn_decode_pred(i)
        results[name] = measure(path, name, pred, use_state=True)
        gc.collect()

    # ── FFN prefill chunks (seq_len=BATCH_SIZE, with state) ──
    print(f"\n[FFN PREFILL — seq_len={BATCH_SIZE}]")
    for i in range(9):
        name = f"prefill_LUT4_chunk{i}"
        path = os.path.join(MODEL_DIR, f"{name}.mlpackage")
        if not os.path.exists(path):
            print(f"  {name:45s} MISSING")
            continue
        pred = make_ffn_prefill_pred(i)
        results[name] = measure(path, name, pred, use_state=True)
        gc.collect()

    # ── LM Head ──
    print("\n[LM HEAD]")
    lm_path = os.path.join(MODEL_DIR, "lm_head_nosplit.mlpackage")
    if os.path.exists(lm_path):
        pred = {"hidden_states": np.random.randn(1, 1, HIDDEN).astype(np.float16)}
        results["lm_head_nosplit"] = measure(lm_path, "lm_head_nosplit", pred)

    # ── Combined embed+lmhead ──
    combined_path = os.path.join(MODEL_DIR, "embed_lmhead_combined.mlpackage")
    if os.path.exists(combined_path):
        pred = {"input_ids": np.array([[1]], dtype=np.int32)}
        results["embed_lmhead_combined"] = measure(combined_path, "embed_lmhead_combined", pred)

    # ── Summary ──
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    fmt = "  {:<45s} {:>7s} {:>9s} {:>9s}"
    print(fmt.format("Model", "ANE%", "Wall(ms)", "CPU(ms)"))
    print("  " + "-" * 73)

    decode_anes = []
    prefill_anes = []
    for name in sorted(results.keys()):
        r = results[name]
        if r:
            print(fmt.format(name, f"{r['ane_pct']:.1f}%", f"{r['wall']:.2f}", f"{r['cpu']:.2f}"))
            if name.startswith("ffn_"):
                decode_anes.append(r["ane_pct"])
            if name.startswith("prefill_"):
                prefill_anes.append(r["ane_pct"])
        else:
            print(fmt.format(name, "FAILED", "-", "-"))

    if decode_anes:
        avg = sum(decode_anes) / len(decode_anes)
        print(f"\n  Decode avg ANE:  {avg:.1f}% (across {len(decode_anes)} chunks)")
    if prefill_anes:
        avg = sum(prefill_anes) / len(prefill_anes)
        print(f"  Prefill avg ANE: {avg:.1f}% (across {len(prefill_anes)} chunks)")

    # Per-chunk decode breakdown with layer info
    if decode_anes:
        print(f"\n  Decode per-chunk:")
        patterns = ["LLL", "FLLL", "FLLL", "FLLL", "FLLL", "FLLL", "FLLL", "FLLL", "F"]
        for i in range(9):
            name = f"ffn_LUT4_chunk{i}"
            r = results.get(name)
            sl, el = CHUNK_RANGES[i]
            pat = patterns[i]
            if r:
                print(f"    chunk {i} [{pat:4s}] L{sl:2d}-{el-1:2d}: ANE={r['ane_pct']:5.1f}% wall={r['wall']:.2f}ms")

    print("=" * 80)


if __name__ == "__main__":
    main()
