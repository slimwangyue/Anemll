#!/usr/bin/env python3
"""
V4 ALL CHUNKS — CTX=16384 Export + LUT4 Dedup

Exports all 9 chunks at CTX=16384 (4096*4) with V4 precision policy
(kv_cache FP32, everything else FP16), then combines decode+prefill
into multifunction dedup models.

Chunk 2 is already exported — will be skipped with --skip-existing.

Output: artifacts/v4_all_chunks_ctx16384/
"""
import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
os.chdir(REPO_ROOT)

import numpy as np
import torch
torch.set_grad_enabled(False)

import coremltools as ct
from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision

from config import NUM_CHUNKS, CHUNK_RANGES

HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
LAYER_PATTERN = re.compile(r"layers[._](\d+)")

CTX = 16384
BATCH_SIZE = 4096

F_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
L_LAYERS = set(range(32)) - F_LAYERS

ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "v4_all_chunks_ctx16384")
# Chunk 2 was already exported here:
CHUNK2_EXISTING = os.path.join(REPO_ROOT, "artifacts", "v4_chunk2_ctx16384")


# ═══════════════════════════════════════════════════════════════════
#  V4 SELECTOR
# ═══════════════════════════════════════════════════════════════════

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


def make_v4_selector(fp16_layers, fp32_layers, audit_log=None):
    fp16_set = set(fp16_layers)
    fp32_set = set(fp32_layers)
    _cache = {}

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
        layers = _get_layers(op)
        if not layers:
            selected = True
        else:
            home = max(layers)
            if home in fp16_set:
                selected = True
            elif home in fp32_set:
                selected = not _is_kv_cache_op(op)
            else:
                selected = True
        if audit_log is not None:
            audit_log.append({
                "name": op.name,
                "op_type": op.op_type,
                "home_layer": max(layers) if layers else None,
                "selected": selected,
            })
        return selected

    return selector


# ═══════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════

def load_model():
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    print(f"Loading model (ctx={CTX}, batch={BATCH_SIZE})...")
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded: {cfg.num_hidden_layers} layers, hidden={cfg.hidden_size}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════

def export_chunk(model, chunk_idx, chunk_dir, skip_existing=False):
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    sl, el = CHUNK_RANGES[chunk_idx]
    fp32_layers = [li for li in range(sl, el) if li in F_LAYERS]
    fp16_layers = [li for li in range(sl, el) if li in L_LAYERS]

    os.makedirs(chunk_dir, exist_ok=True)
    results = {}

    for phase, convert_fn_name in [("decode", "convert_part_2"),
                                   ("prefill", "convert_part_2_prefill")]:
        pkg_path = os.path.join(chunk_dir, f"{phase}.mlpackage")
        audit_path = os.path.join(chunk_dir, f"selected_ops_{phase}.json")

        if skip_existing and os.path.exists(pkg_path):
            print(f"    [skip] {phase} (exists)")
            results[phase] = {"skipped": True}
            continue

        audit_log = []
        selector = make_v4_selector(fp16_layers, fp32_layers, audit_log=audit_log)

        t0 = time.time()
        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
            compute_precision="float32",
        )
        conv.compute_precision = FP16ComputePrecision(op_selector=selector)

        convert_fn = getattr(conv, convert_fn_name)
        ml = convert_fn(
            model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
            override_start_layer=sl, override_end_layer=el,
        )
        ml.save(pkg_path)
        elapsed = time.time() - t0
        del ml, conv
        gc.collect()

        with open(audit_path, "w") as f:
            json.dump(audit_log, f, indent=1)

        total = len(audit_log)
        selected = sum(1 for e in audit_log if e["selected"])
        fp32_kept = total - selected

        print(f"    {phase}: {selected}/{total} ops FP16, {fp32_kept} FP32 ({elapsed:.1f}s)")
        results[phase] = {
            "total_ops": total,
            "selected_fp16": selected,
            "fp32_kept": fp32_kept,
            "elapsed": elapsed,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
#  COMBINE (DEDUP)
# ═══════════════════════════════════════════════════════════════════

def combine_chunk(chunk_idx, chunk_dir, combined_dir, skip_existing=False):
    from anemll.utils.combine_models import _save_multifunction_dedup

    combined_path = os.path.join(combined_dir, f"chunk{chunk_idx}.mlpackage")

    if skip_existing and os.path.exists(combined_path) and not os.path.islink(combined_path):
        print(f"    [skip] chunk {chunk_idx} combine (exists)")
        return

    if os.path.lexists(combined_path):
        if os.path.islink(combined_path):
            os.unlink(combined_path)
        else:
            shutil.rmtree(combined_path)

    dec_path = os.path.join(chunk_dir, "decode.mlpackage")
    pf_path = os.path.join(chunk_dir, "prefill.mlpackage")

    sources = [
        (dec_path, "main", "infer"),
        (pf_path, "main", "prefill"),
    ]

    t0 = time.time()
    _save_multifunction_dedup(sources, combined_path, dedup_weights=True, verbose=False)
    elapsed = time.time() - t0

    # Size
    total_bytes = 0
    for root, dirs, files in os.walk(combined_path):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))

    print(f"    chunk {chunk_idx}: combined {total_bytes / (1024**2):.1f} MB ({elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Comma-separated chunk indices (default: all)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export, only combine")
    args = parser.parse_args()

    chunk_indices = list(range(NUM_CHUNKS))
    if args.chunks:
        chunk_indices = [int(x) for x in args.chunks.split(",")]

    print("=" * 70)
    print(f"  V4 ALL CHUNKS — CTX={CTX} Export + LUT4 Dedup")
    print(f"  Chunks: {chunk_indices}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  Output: {ARTIFACT_DIR}")
    print("=" * 70)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    combined_dir = os.path.join(ARTIFACT_DIR, "combined_LUT4_dedup")
    os.makedirs(combined_dir, exist_ok=True)

    # ── Phase 1: Link chunk 2 from existing export ──
    chunk2_dir = os.path.join(ARTIFACT_DIR, "chunk_2")
    if not os.path.exists(chunk2_dir) and os.path.exists(CHUNK2_EXISTING):
        print(f"\n  Linking chunk 2 from {CHUNK2_EXISTING}")
        os.symlink(CHUNK2_EXISTING, chunk2_dir)

    # Also link the combined chunk2 if it exists
    chunk2_combined_src = os.path.join(CHUNK2_EXISTING, "combined_LUT4_dedup", "chunk2.mlpackage")
    chunk2_combined_dst = os.path.join(combined_dir, "chunk2.mlpackage")
    if not os.path.exists(chunk2_combined_dst) and os.path.exists(chunk2_combined_src):
        os.symlink(chunk2_combined_src, chunk2_combined_dst)
        print(f"  Linked chunk 2 combined dedup")

    # ── Phase 2: Export ──
    if not args.skip_export:
        model = load_model()
        all_results = {}

        for ci in chunk_indices:
            sl, el = CHUNK_RANGES[ci]
            fl = [li for li in range(sl, el) if li in F_LAYERS]
            ll = [li for li in range(sl, el) if li in L_LAYERS]
            print(f"\n── Chunk {ci}: layers {sl}–{el-1} "
                  f"(F={fl}, L={ll}) ──")

            chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")

            # Skip chunk 2 if already linked
            if ci == 2 and os.path.islink(chunk_dir):
                print(f"    [skip] chunk 2 (linked from previous export)")
                continue

            results = export_chunk(model, ci, chunk_dir, skip_existing=args.skip_existing)
            all_results[ci] = results

        del model
        gc.collect()

        # Save export results
        with open(os.path.join(ARTIFACT_DIR, "export_results.json"), "w") as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # ── Phase 3: Combine (dedup) ──
    print(f"\n── Combining chunks (LUT4 dedup) ──")
    for ci in chunk_indices:
        chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")

        # For chunk 2, use the existing linked dir
        if ci == 2 and os.path.islink(chunk_dir):
            # Decode/prefill are in the linked CTX16384 dir directly
            actual_dir = os.path.realpath(chunk_dir)
            dec_exists = os.path.exists(os.path.join(actual_dir, "decode.mlpackage"))
            pf_exists = os.path.exists(os.path.join(actual_dir, "prefill.mlpackage"))
            if dec_exists and pf_exists:
                combine_chunk(ci, actual_dir, combined_dir, skip_existing=args.skip_existing)
            else:
                print(f"    [skip] chunk {ci} combine (missing decode/prefill)")
            continue

        dec_exists = os.path.exists(os.path.join(chunk_dir, "decode.mlpackage"))
        pf_exists = os.path.exists(os.path.join(chunk_dir, "prefill.mlpackage"))
        if dec_exists and pf_exists:
            combine_chunk(ci, chunk_dir, combined_dir, skip_existing=args.skip_existing)
        else:
            print(f"    [skip] chunk {ci} combine (missing decode/prefill)")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — CTX={CTX}")
    print(f"{'=' * 70}")
    for ci in range(NUM_CHUNKS):
        sl, el = CHUNK_RANGES[ci]
        fl = [li for li in range(sl, el) if li in F_LAYERS]
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if os.path.exists(combined_path):
            total_bytes = 0
            for root, dirs, files in os.walk(combined_path):
                for fn in files:
                    total_bytes += os.path.getsize(os.path.join(root, fn))
            print(f"  chunk {ci}: layers {sl:2d}–{el-1:2d}  F={fl!s:12s}  "
                  f"combined={total_bytes / (1024**2):.1f} MB  ✓")
        else:
            print(f"  chunk {ci}: layers {sl:2d}–{el-1:2d}  F={fl!s:12s}  MISSING")

    print(f"\n  Combined dir: {combined_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
