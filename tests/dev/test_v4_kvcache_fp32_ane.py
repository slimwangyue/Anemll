#!/usr/bin/env python3
"""
Test: Does keeping KV cache ops in FP32 disable ANE?

Compares three precision policies on a single F-layer chunk (chunk 2, FLLL):
  A) Full FP16   — everything FP16 (baseline, known ~92% ANE)
  B) V4 policy   — FP16 except 6 kv_cache_state ops in FP32
  C) Full FP32   — everything FP32 (known 0% ANE, compiler fails)

Also tests a single-op variant:
  D) FP16, only slice_update ops FP32 (2 ops)
  E) FP16, only identity ops FP32 (2 ops)

Output: artifacts/v4_kvcache_ane_test/
"""
import gc
import os
import re
import resource
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
os.chdir(REPO_ROOT)

import torch

torch.set_grad_enabled(False)

import coremltools as ct
from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "v4_kvcache_ane_test")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

HIDDEN_SIZE = 2560
NUM_KV_HEADS = 4
HEAD_DIM = 256
LAYER_PATTERN = re.compile(r"layers[._](\d+)")
F_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
L_LAYERS = set(range(32)) - F_LAYERS

# Chunk 2: layers 7-10 (FLLL)
CHUNK_IDX = 2
CHUNK_START, CHUNK_END = CHUNK_RANGES[CHUNK_IDX]

# ═══════════════════════════════════════════════════════════════
#  KV CACHE OP DETECTION (from V4 policy)
# ═══════════════════════════════════════════════════════════════

def _is_kv_cache_op(op):
    name_lower = op.name.lower()
    if "cache" in name_lower:
        return True
    if op.op_type == "identity":
        return True
    if op.op_type == "squeeze":
        try:
            for out_var in op.outputs:
                for child in out_var.child_ops:
                    if "cache" in child.name.lower():
                        return True
        except (AttributeError, TypeError):
            pass
    if op.op_type == "slice_by_index":
        try:
            for out_var in op.outputs:
                for child in out_var.child_ops:
                    if child.op_type == "identity":
                        return True
        except (AttributeError, TypeError):
            pass
    return False


def _is_slice_update_cache(op):
    """Only slice_update ops with 'cache' in name."""
    return op.op_type == "slice_update" and "cache" in op.name.lower()


def _is_identity_op(op):
    """Only identity pass-through ops."""
    return op.op_type == "identity"


def make_selector(mode, fp32_layers=F_LAYERS, fp16_layers=L_LAYERS):
    """Create an op_selector for a given mode.

    Modes:
      'all_fp16'     → select everything for FP16 (returns True always)
      'v4_kvcache'   → FP16 except kv_cache_state ops in F layers
      'only_slice_update' → FP16 except slice_update cache ops in F layers
      'only_identity'     → FP16 except identity ops in F layers
    """
    _cache = {}
    stats = {"total": 0, "fp32": 0, "fp16": 0, "fp32_ops": []}

    def _get_layers(op, visited=None):
        op_id = id(op)
        if op_id in _cache:
            return _cache[op_id]
        if visited is None:
            visited = set()
        if op_id in visited:
            return set()
        visited.add(op_id)
        layers = set()
        m = LAYER_PATTERN.search(op.name)
        if m:
            layers.add(int(m.group(1)))
        for inp_val in op.inputs.values():
            if isinstance(inp_val, (list, tuple)):
                for v in inp_val:
                    if hasattr(v, "op") and v.op is not None:
                        layers |= _get_layers(v.op, visited)
            elif hasattr(inp_val, "op") and inp_val.op is not None:
                layers |= _get_layers(inp_val.op, visited)
        _cache[op_id] = layers
        return layers

    def selector(op):
        stats["total"] += 1

        if mode == "all_fp16":
            stats["fp16"] += 1
            return True

        layers = _get_layers(op)
        if not layers:
            stats["fp16"] += 1
            return True  # pre-layer ops → FP16

        home = max(layers)

        if home in fp16_layers:
            stats["fp16"] += 1
            return True  # L layer → FP16

        if home in fp32_layers:
            # F layer — check mode
            if mode == "v4_kvcache":
                is_fp32 = _is_kv_cache_op(op)
            elif mode == "only_slice_update":
                is_fp32 = _is_slice_update_cache(op)
            elif mode == "only_identity":
                is_fp32 = _is_identity_op(op)
            else:
                is_fp32 = False

            if is_fp32:
                stats["fp32"] += 1
                stats["fp32_ops"].append(f"{op.op_type}:{op.name}")
                return False  # keep FP32
            else:
                stats["fp16"] += 1
                return True
        else:
            stats["fp16"] += 1
            return True

    selector.stats = stats
    return selector


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model():
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ═══════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════

def export_variant(model, name, mode):
    """Export decode chunk 2 with given precision mode."""
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    path = os.path.join(ARTIFACT_DIR, f"{name}_decode.mlpackage")
    if os.path.exists(path):
        print(f"  [{name}] Already exists, skipping export")
        return path

    print(f"  [{name}] Exporting (mode={mode})...")
    t0 = time.time()

    if mode == "full_fp32":
        # baseline: ct.precision.FLOAT32
        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
            compute_precision="float32",
        )
        ml = conv.convert_part_2(
            model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS,
            override_start_layer=CHUNK_START, override_end_layer=CHUNK_END,
        )
    else:
        sel = make_selector(mode)
        custom = FP16ComputePrecision(op_selector=sel)
        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
            compute_precision="float32",
        )
        conv.compute_precision = custom
        ml = conv.convert_part_2(
            model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS,
            override_start_layer=CHUNK_START, override_end_layer=CHUNK_END,
        )

        print(f"    FP16 ops: {sel.stats['fp16']}, FP32 ops: {sel.stats['fp32']}")
        if sel.stats["fp32_ops"]:
            for op_info in sel.stats["fp32_ops"]:
                print(f"      FP32: {op_info}")

    ml.save(path)
    del ml, conv
    gc.collect()
    print(f"    Saved in {time.time() - t0:.1f}s")
    return path


# ═══════════════════════════════════════════════════════════════
#  ANE MEASUREMENT
# ═══════════════════════════════════════════════════════════════

def measure_ane(path, name, warmup=10, runs=30):
    """Load model and measure ANE utilization via CPU-time proxy."""
    try:
        ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e:
        print(f"  [{name}] FAILED to load: {e}")
        return None

    try:
        state = ml.make_state()
    except Exception as e:
        print(f"  [{name}] FAILED make_state: {e}")
        return None

    pred = {
        "hidden_states": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float16),
        "position_ids": np.array([CTX // 2], dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, CTX), dtype=np.float16),
        "current_pos": np.array([CTX // 2], dtype=np.int32),
    }

    # Warmup
    for _ in range(warmup):
        ml.predict(pred, state=state)

    # Measure
    times, cpus = [], []
    for _ in range(runs):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        ml.predict(pred, state=state)
        t1 = time.perf_counter()
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        times.append(t1 - t0)
        cpus.append((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))

    wall = np.median(times) * 1000
    cpu = np.median(cpus) * 1000
    cpu_pct = cpu / wall * 100 if wall > 0 else 0
    ane_pct = max(0, 100 - cpu_pct)

    print(f"  [{name:25s}] wall={wall:.2f}ms cpu={cpu:.2f}ms CPU%={cpu_pct:.1f}% ANE%={ane_pct:.1f}%")
    del ml, state
    gc.collect()
    return {"wall": wall, "cpu": cpu, "cpu_pct": cpu_pct, "ane_pct": ane_pct}


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  V4 KV-CACHE FP32 — ANE IMPACT TEST")
    print(f"  Chunk {CHUNK_IDX}: layers {CHUNK_START}-{CHUNK_END - 1} (FLLL)")
    print(f"  CTX={CTX}, BATCH={BATCH_SIZE}")
    print("=" * 70)

    # Variants to test
    variants = [
        ("A_full_fp16", "all_fp16"),
        ("B_v4_kvcache_fp32", "v4_kvcache"),
        ("C_only_slice_update_fp32", "only_slice_update"),
        ("D_only_identity_fp32", "only_identity"),
        ("E_full_fp32", "full_fp32"),
    ]

    # Export
    print("\n[PHASE 1] Export variants")
    model = load_model()
    paths = {}
    for name, mode in variants:
        paths[name] = export_variant(model, name, mode)
    del model
    gc.collect()

    # Measure
    print(f"\n[PHASE 2] Measure ANE utilization (decode, CTX={CTX})")
    results = {}
    for name, _ in variants:
        results[name] = measure_ane(paths[name], name)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — Does KV cache FP32 disable ANE?")
    print(f"{'=' * 70}")
    print(f"  {'Variant':<30s} {'ANE%':>8s} {'Wall(ms)':>10s} {'CPU(ms)':>10s}")
    print(f"  {'-' * 60}")
    for name, _ in variants:
        r = results.get(name)
        if r:
            print(f"  {name:<30s} {r['ane_pct']:>7.1f}% {r['wall']:>9.2f} {r['cpu']:>9.2f}")
        else:
            print(f"  {name:<30s}    FAILED")

    print(f"\n  Key question: Does B (V4, 6 FP32 ops) still load on ANE?")
    b = results.get("B_v4_kvcache_fp32")
    if b and b["ane_pct"] > 50:
        print(f"  >>> YES — V4 achieves {b['ane_pct']:.1f}% ANE (kv_cache FP32 does NOT kill ANE)")
    elif b:
        print(f"  >>> PARTIAL — V4 at {b['ane_pct']:.1f}% ANE (some degradation)")
    else:
        print(f"  >>> NO — V4 failed to load on ANE")

    print("=" * 70)


if __name__ == "__main__":
    main()
