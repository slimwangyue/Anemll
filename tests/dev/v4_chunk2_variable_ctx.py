#!/usr/bin/env python3
"""
V4 CHUNK2 — Variable Context Length Export

Exports chunk 2 (layers 7-10, FLLL) at multiple context lengths
using the validated V4 precision policy (kv_cache FP32, rest FP16).

Context lengths: 4096, 8192, 16384
BATCH_SIZE scales proportionally: CTX/4

Output: artifacts/v4_chunk2_ctx{CTX}/
"""
import argparse
import gc
import json
import os
import re
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

F_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
L_LAYERS = set(range(32)) - F_LAYERS

CHUNK_IDX = 2
CHUNK_START, CHUNK_END = CHUNK_RANGES[CHUNK_IDX]  # layers 7-10

CTX_CONFIGS = [
    (4096,  1024),   # CTX=4096,  BATCH=1024
    (8192,  2048),   # CTX=8192,  BATCH=2048
    (16384, 4096),   # CTX=16384, BATCH=4096
]


# ═══════════════════════════════════════════════════════════════════
#  V4 SELECTOR (from all_chunks_v4_kvcache_fp32.py)
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
            home = None
            selected = True
        else:
            home = max(layers)
            if home in fp16_set:
                selected = True
            elif home in fp32_set:
                is_cache = _is_kv_cache_op(op)
                selected = not is_cache
            else:
                selected = True
        if audit_log is not None:
            audit_log.append({
                "name": op.name,
                "op_type": op.op_type,
                "transitive_layers": sorted(layers),
                "home_layer": home,
                "selected": selected,
            })
        return selected

    return selector


# ═══════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════

def load_model(ctx):
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    print(f"  Loading model (ctx={ctx})...")
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = ctx
    cfg.state_length = ctx
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def export_chunk2(model, ctx, batch_size, out_dir, skip_existing=False):
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    fp32_layers = [li for li in range(CHUNK_START, CHUNK_END) if li in F_LAYERS]
    fp16_layers = [li for li in range(CHUNK_START, CHUNK_END) if li in L_LAYERS]

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for phase, convert_fn_name in [("decode", "convert_part_2"),
                                   ("prefill", "convert_part_2_prefill")]:
        pkg_path = os.path.join(out_dir, f"{phase}.mlpackage")
        audit_path = os.path.join(out_dir, f"selected_ops_{phase}.json")

        if skip_existing and os.path.exists(pkg_path):
            print(f"    [skip] {phase} (exists)")
            results[phase] = {"path": pkg_path, "skipped": True}
            continue

        audit_log = []
        selector = make_v4_selector(fp16_layers, fp32_layers, audit_log=audit_log)

        t0 = time.time()
        conv = Qwen35Converter(
            model, context_length=ctx, batch_size=batch_size,
            num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
            compute_precision="float32",
        )
        conv.compute_precision = FP16ComputePrecision(op_selector=selector)

        convert_fn = getattr(conv, convert_fn_name)
        ml = convert_fn(
            model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS,
            override_start_layer=CHUNK_START, override_end_layer=CHUNK_END,
        )
        ml.save(pkg_path)
        elapsed = time.time() - t0
        del ml, conv
        gc.collect()

        with open(audit_path, "w") as f:
            json.dump(audit_log, f, indent=1)

        total = len(audit_log)
        selected = sum(1 for e in audit_log if e["selected"])
        fp32_kept = sum(1 for e in audit_log
                        if not e["selected"] and e["home_layer"] is not None
                        and e["home_layer"] in F_LAYERS)

        print(f"    {phase}: {selected}/{total} ops FP16, {fp32_kept} F-layer ops FP32 ({elapsed:.1f}s)")
        results[phase] = {
            "path": pkg_path,
            "total_ops": total,
            "selected_fp16": selected,
            "fp32_kept": fp32_kept,
            "elapsed": elapsed,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ctx", type=int, nargs="+", default=None,
                        help="Specific CTX values to export (default: all three)")
    args = parser.parse_args()

    configs = CTX_CONFIGS
    if args.ctx:
        configs = [(c, b) for c, b in CTX_CONFIGS if c in args.ctx]
        if not configs:
            print(f"ERROR: No matching CTX values. Available: {[c for c,_ in CTX_CONFIGS]}")
            sys.exit(1)

    print("=" * 70)
    print("  V4 CHUNK 2 — Variable Context Length Export")
    print(f"  Chunk 2: layers {CHUNK_START}–{CHUNK_END-1} (FLLL)")
    print(f"  F layers (kv_cache FP32): {sorted(li for li in range(CHUNK_START, CHUNK_END) if li in F_LAYERS)}")
    print(f"  L layers (full FP16):     {sorted(li for li in range(CHUNK_START, CHUNK_END) if li in L_LAYERS)}")
    print(f"  Configs: {configs}")
    print("=" * 70)

    all_results = {}

    for ctx, batch_size in configs:
        print(f"\n── CTX={ctx}, BATCH={batch_size} ──")
        out_dir = os.path.join(REPO_ROOT, "artifacts", f"v4_chunk2_ctx{ctx}")

        # KV cache memory estimate
        kv_bytes = 2 * 4 * 4 * ctx * 256 * 2  # (2*num_layers, kv_heads, state_len, head_dim) * fp16
        kv_mb = kv_bytes / (1024**2)
        print(f"  KV cache: ({8}, 4, {ctx}, 256) = {kv_mb:.0f} MB")

        model = load_model(ctx)
        results = export_chunk2(model, ctx, batch_size, out_dir, skip_existing=args.skip_existing)
        all_results[ctx] = results

        # Free model memory before next CTX
        del model
        gc.collect()

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for ctx, batch_size in configs:
        r = all_results.get(ctx, {})
        out_dir = os.path.join(REPO_ROOT, "artifacts", f"v4_chunk2_ctx{ctx}")
        kv_mb = 2 * 4 * 4 * ctx * 256 * 2 / (1024**2)
        print(f"\n  CTX={ctx} (BATCH={batch_size}, KV cache={kv_mb:.0f} MB)")
        print(f"    Output: {out_dir}")
        for phase in ["decode", "prefill"]:
            pr = r.get(phase, {})
            if pr.get("skipped"):
                print(f"    {phase}: skipped (exists)")
            elif "elapsed" in pr:
                print(f"    {phase}: {pr['selected_fp16']}/{pr['total_ops']} FP16, "
                      f"{pr['fp32_kept']} FP32 ({pr['elapsed']:.1f}s)")

    # Save combined results
    results_path = os.path.join(REPO_ROOT, "artifacts", "v4_chunk2_variable_ctx_results.json")
    serializable = {}
    for ctx, r in all_results.items():
        serializable[str(ctx)] = {}
        for phase, pr in r.items():
            serializable[str(ctx)][phase] = {k: v for k, v in pr.items() if k != "path"}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
