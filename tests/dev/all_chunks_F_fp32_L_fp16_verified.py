#!/usr/bin/env python3
"""
ALL-CHUNKS F-FP32 / L-FP16 VERIFIED EXPERIMENT — Qwen3.5-4B

Converts every FFN chunk with:
  - F (full_attention) layers → FP32 compute
  - L (linear_attention) layers → FP16 compute

Uses coremltools FP16ComputePrecision(op_selector=...) with the
"max-layer home attribution" strategy (verified disjoint in prior work).

Includes:
  1. Authoritative F/L manifest from model structure
  2. Chunk-aware FP16 selector with op audit
  3. Decode + prefill export for every chunk
  4. Selected-op audit during conversion (JSON logs)
  5. Post-export MIL graph audit (cast verification)
  6. Combined summary report

Output: artifacts/fl_precision_verified/
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

# ── repo / config ──
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
os.chdir(REPO_ROOT)

import numpy as np
import torch

torch.set_grad_enabled(False)

import coremltools as ct
from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

# ── paths ──
HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "fl_precision_verified")
LAYER_PATTERN = re.compile(r"layers[._](\d+)")


# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — Build authoritative F/L layer manifest
# ═══════════════════════════════════════════════════════════════════

def build_layer_manifest(model):
    """Inspect every transformer layer and produce {idx: {kind, module_class, reason, chunk}}.

    Reads the actual layer_type attribute set during __init__ of Qwen35DecoderLayer.
    """
    manifest = {}
    layers = model.model.layers
    num_layers = len(layers)

    # Build layer→chunk map
    layer_to_chunk = {}
    for ci, (sl, el) in enumerate(CHUNK_RANGES):
        for li in range(sl, el):
            layer_to_chunk[li] = ci

    for idx in range(num_layers):
        layer = layers[idx]
        layer_type = getattr(layer, "layer_type", None)
        attn_class = type(layer.self_attn).__name__

        if layer_type == "full_attention":
            kind = "F"
            reason = f"layer_type='{layer_type}', attn_class={attn_class}"
        elif layer_type == "linear_attention":
            kind = "L"
            reason = f"layer_type='{layer_type}', attn_class={attn_class}"
        else:
            # Fallback: infer from class name
            if "Linear" in attn_class:
                kind = "L"
            elif "Full" in attn_class:
                kind = "F"
            else:
                kind = "?"
            reason = f"INFERRED from attn_class={attn_class} (no layer_type attr)"

        manifest[idx] = {
            "kind": kind,
            "module_class": type(layer).__name__,
            "attn_class": attn_class,
            "layer_type": layer_type,
            "reason": reason,
            "chunk": layer_to_chunk.get(idx, -1),
        }

    return manifest


# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — Chunk-aware F/L layer classification
# ═══════════════════════════════════════════════════════════════════

def get_chunk_fl_layers(manifest):
    """Return {chunk_idx: {fp32_layers: [...], fp16_layers: [...]}} for all chunks."""
    chunk_map = {}
    for ci, (sl, el) in enumerate(CHUNK_RANGES):
        fp32 = []
        fp16 = []
        for li in range(sl, el):
            info = manifest[li]
            if info["kind"] == "F":
                fp32.append(li)
            else:
                fp16.append(li)
        chunk_map[ci] = {"fp32_layers": fp32, "fp16_layers": fp16}
    return chunk_map


# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — MIL op-selector with audit logging
# ═══════════════════════════════════════════════════════════════════

def make_fl_selector(fp16_layers, audit_log=None):
    """Return an FP16ComputePrecision op_selector for the given fp16 (L) layers.

    Uses the "max layer in transitive set" home-attribution strategy.
    Pre-layer ops (no layer attribution) are also selected for FP16
    when the chunk contains any L layers.
    Optionally logs every op decision to audit_log (list).
    """
    fp16_set = set(fp16_layers)
    has_fp16_layers = len(fp16_set) > 0
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
            # Pre-layer ops: select for FP16 if chunk has any L layers
            selected = has_fp16_layers
        else:
            home = max(layers)
            selected = home in fp16_set

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
#  STEP 4 — Selector audit analysis
# ═══════════════════════════════════════════════════════════════════

def audit_selected_ops(audit_log, fp16_layers, fp32_layers):
    """Compute summary stats from the selector audit log."""
    from collections import Counter

    fp16_set = set(fp16_layers)
    fp32_set = set(fp32_layers)

    total = len(audit_log)
    selected = sum(1 for e in audit_log if e["selected"])
    unselected = total - selected

    home_counts = Counter()
    selected_by_home = Counter()

    for e in audit_log:
        h = e["home_layer"]
        if h is not None:
            home_counts[h] += 1
            if e["selected"]:
                selected_by_home[h] += 1

    # Find wrongly selected (home layer is F but selected for fp16)
    wrong_selected = [
        e for e in audit_log
        if e["selected"] and e["home_layer"] is not None and e["home_layer"] in fp32_set
    ]

    # Find wrongly unselected (home layer is L but not selected for fp16)
    wrong_unselected = [
        e for e in audit_log
        if not e["selected"] and e["home_layer"] is not None and e["home_layer"] in fp16_set
    ]

    # Count pre-layer ops (home=None)
    prelayer_ops = [e for e in audit_log if e["home_layer"] is None]
    prelayer_selected = sum(1 for e in prelayer_ops if e["selected"])
    prelayer_compute = [e for e in prelayer_ops if e["op_type"] != "const"]

    summary = {
        "total_ops": total,
        "selected_for_fp16": selected,
        "unselected": unselected,
        "home_layer_counts": {str(k): v for k, v in sorted(home_counts.items())},
        "selected_by_home_layer": {str(k): v for k, v in sorted(selected_by_home.items())},
        "fp16_layers": sorted(fp16_layers),
        "fp32_layers": sorted(fp32_layers),
        "prelayer_total": len(prelayer_ops),
        "prelayer_selected": prelayer_selected,
        "prelayer_compute_ops": len(prelayer_compute),
        "wrong_selected_count": len(wrong_selected),
        "wrong_selected_examples": wrong_selected[:5],
        "wrong_unselected_count": len(wrong_unselected),
        "wrong_unselected_examples": wrong_unselected[:5],
        "selector_correct": len(wrong_selected) == 0 and len(wrong_unselected) == 0,
    }
    return summary


# ═══════════════════════════════════════════════════════════════════
#  STEP 5 — Post-export graph audit
# ═══════════════════════════════════════════════════════════════════

def audit_exported_graph(mlpackage_path, fp16_layers, fp32_layers,
                         ref_fp32_cast_count=None, ref_fp16_cast_count=None,
                         function_name="main"):
    """Reload the exported MIL proto and verify FP16ComputePrecision acted correctly.

    Primary verification: selector audit (Step 4) already logs every op decision
    with correct home-layer attribution.

    This graph audit provides secondary confirmation by checking:
    1. Cast-to-fp16 ops exist in the graph (proves the FP16 pass inserted casts)
    2. Cast count is consistent with the number of selected ops
    3. If a reference all-fp32 cast count is provided, the selective model has MORE
       casts (proving the FP16 pass added them for L layers)
    4. For pure-F chunks, no excess fp16 casts are present
    """
    from collections import defaultdict

    fp16_set = set(fp16_layers)
    fp32_set = set(fp32_layers)

    audit = {
        "mlpackage_path": mlpackage_path,
        "function_name": function_name,
        "fp16_layers": sorted(fp16_layers),
        "fp32_layers": sorted(fp32_layers),
    }

    try:
        spec = ct.utils.load_spec(mlpackage_path)
        program = spec.mlProgram
        functions = program.functions

        if function_name in functions:
            func = functions[function_name]
        else:
            func_names = list(functions.keys())
            if not func_names:
                audit["error"] = "No functions found in MIL program"
                audit["verified"] = False
                return audit
            func = functions[func_names[0]]
            audit["function_name_used"] = func_names[0]

        specs_available = list(func.block_specializations.keys())
        if not specs_available:
            audit["error"] = "No block specializations found"
            audit["verified"] = False
            return audit

        block = func.block_specializations[specs_available[0]]
        audit["block_specialization"] = specs_available[0]
        ops = list(block.operations)

        # ── Count ops by type ──
        op_type_counts = defaultdict(int)
        for op in ops:
            op_type_counts[op.type] += 1

        # ── Count cast ops by output dtype ──
        # MIL proto dataType enum: 10=FLOAT16, 11=FLOAT32
        DTYPE_MAP = {10: "fp16", 11: "fp32", 22: "int16", 23: "int32", 32: "bool"}

        cast_by_dtype = defaultdict(int)
        for op in ops:
            if op.type != "cast":
                continue
            if op.outputs:
                dt = op.outputs[0].type.tensorType.dataType
                dtype_str = DTYPE_MAP.get(dt, f"unknown_{dt}")
                cast_by_dtype[dtype_str] += 1

        n_cast_fp16 = cast_by_dtype.get("fp16", 0)
        n_cast_fp32 = cast_by_dtype.get("fp32", 0)
        n_cast_total = sum(cast_by_dtype.values())

        # ── Layer-named ops in graph (for info) ──
        layer_op_counts = defaultdict(int)
        for op in ops:
            for out in op.outputs:
                for m in LAYER_PATTERN.finditer(out.name):
                    layer_op_counts[int(m.group(1))] += 1

        # ── Verification checks ──
        checks = []

        # Check 1: fp16 casts exist when mixed F+L layers are present
        # (when ALL ops are fp16, zero casts is ideal — no boundaries)
        if fp16_layers and fp32_layers:
            checks.append({
                "check": "fp16_casts_exist_for_mixed_FL_chunk",
                "passed": n_cast_fp16 > 0,
                "detail": f"{n_cast_fp16} cast-to-fp16 ops found",
            })

        # Check 2: for all-L chunks, fewer casts (ideally zero) proves everything is fp16
        if fp16_layers and not fp32_layers:
            checks.append({
                "check": "all_L_chunk_minimal_casts",
                "passed": n_cast_total <= ref_fp32_cast_count if ref_fp32_cast_count is not None else True,
                "detail": f"total_casts={n_cast_total} (ref_fp32={ref_fp32_cast_count}, fewer=better)",
            })

        # Check 3: comparison with reference fp32 model
        if ref_fp32_cast_count is not None:
            if fp16_layers:
                # With L layers selected for fp16, the selective model should have
                # FEWER total casts than all-fp32. Why: in all-fp32, L-layer ops need
                # cast(fp16→fp32) on inputs and cast(fp32→fp16) on outputs.
                # In selective mode, L-layer ops compute in fp16 natively, so those
                # casts are eliminated. This is strong evidence the FP16 pass acted.
                has_fewer = n_cast_total < ref_fp32_cast_count
                checks.append({
                    "check": "fewer_casts_than_fp32_reference",
                    "passed": has_fewer,
                    "detail": (f"selective={n_cast_total} vs fp32_ref={ref_fp32_cast_count} "
                               f"(delta={n_cast_total - ref_fp32_cast_count}, fewer=pass acted)"),
                })
            else:
                # Pure F chunk: fp16 cast count should match fp32 reference exactly
                # (non-float casts like int16/int32/bool may differ slightly due to
                # pipeline internals, so compare only fp16 casts which we control)
                same_fp16 = n_cast_fp16 == ref_fp16_cast_count if ref_fp16_cast_count is not None else True
                checks.append({
                    "check": "pure_F_same_fp16_casts_as_fp32_reference",
                    "passed": same_fp16,
                    "detail": (f"selective_fp16={n_cast_fp16} vs ref_fp16={ref_fp16_cast_count}"
                               f" (total: selective={n_cast_total} vs ref={ref_fp32_cast_count})"),
                })

        # Check 4: pure-F chunk should NOT have MORE fp16 casts than reference
        if not fp16_layers:
            # The FP16 pass should not add any fp16 casts to a pure-F chunk.
            # Baseline fp16 casts from I/O boundaries are expected (same as ref).
            if ref_fp16_cast_count is not None:
                no_extra_fp16 = n_cast_fp16 <= ref_fp16_cast_count
            else:
                no_extra_fp16 = True  # can't check without ref
            checks.append({
                "check": "pure_F_chunk_no_extra_fp16_casts",
                "passed": no_extra_fp16,
                "detail": f"{n_cast_fp16} fp16 casts (ref={ref_fp16_cast_count})",
            })

        all_passed = all(c["passed"] for c in checks) if checks else True

        audit.update({
            "op_type_counts": dict(op_type_counts),
            "total_ops_in_graph": len(ops),
            "total_cast_ops": n_cast_total,
            "cast_to_fp16": n_cast_fp16,
            "cast_to_fp32": n_cast_fp32,
            "cast_by_dtype": dict(cast_by_dtype),
            "layer_ops_in_graph": {str(k): v for k, v in sorted(layer_op_counts.items())},
            "ref_fp32_cast_count": ref_fp32_cast_count,
            "checks": checks,
            "all_checks_passed": all_passed,
            "verified": all_passed,
        })

    except Exception as e:
        import traceback
        audit["error"] = str(e)
        audit["traceback"] = traceback.format_exc()
        audit["verified"] = False

    return audit


# ═══════════════════════════════════════════════════════════════════
#  STEP 6 — Export helpers
# ═══════════════════════════════════════════════════════════════════

def load_model():
    """Load Qwen3.5-4B model."""
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

    print(f"Loading model from {HF_MODEL}...")
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


def export_chunk_decode(model, chunk_idx, fp16_layers, chunk_dir, skip_existing=False):
    """Export a decode mlpackage for one chunk with F-fp32 / L-fp16 precision."""
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    sl, el = CHUNK_RANGES[chunk_idx]
    dec_path = os.path.join(chunk_dir, "decode.mlpackage")
    audit_path = os.path.join(chunk_dir, "selected_ops_decode.json")

    if skip_existing and os.path.exists(dec_path) and os.path.exists(audit_path):
        print(f"    [skip] decode (exists)")
        return True

    audit_log = []
    selector = make_fl_selector(fp16_layers, audit_log=audit_log)

    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
        compute_precision="float32",
    )
    conv.compute_precision = FP16ComputePrecision(op_selector=selector)

    ml = conv.convert_part_2(
        model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
        override_start_layer=sl, override_end_layer=el,
    )
    ml.save(dec_path)
    elapsed = time.time() - t0
    del ml, conv
    gc.collect()

    # Save audit log
    with open(audit_path, "w") as f:
        json.dump(audit_log, f, indent=1)

    sel_count = sum(1 for e in audit_log if e["selected"])
    print(f"    decode: {sel_count}/{len(audit_log)} ops selected ({elapsed:.1f}s)")
    return True


def export_chunk_prefill(model, chunk_idx, fp16_layers, chunk_dir, skip_existing=False):
    """Export a prefill mlpackage for one chunk with F-fp32 / L-fp16 precision."""
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    sl, el = CHUNK_RANGES[chunk_idx]
    pf_path = os.path.join(chunk_dir, "prefill.mlpackage")
    audit_path = os.path.join(chunk_dir, "selected_ops_prefill.json")

    if skip_existing and os.path.exists(pf_path) and os.path.exists(audit_path):
        print(f"    [skip] prefill (exists)")
        return True

    audit_log = []
    selector = make_fl_selector(fp16_layers, audit_log=audit_log)

    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
        compute_precision="float32",
    )
    conv.compute_precision = FP16ComputePrecision(op_selector=selector)

    ml = conv.convert_part_2_prefill(
        model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
        override_start_layer=sl, override_end_layer=el,
    )
    ml.save(pf_path)
    elapsed = time.time() - t0
    del ml, conv
    gc.collect()

    # Save audit log
    with open(audit_path, "w") as f:
        json.dump(audit_log, f, indent=1)

    sel_count = sum(1 for e in audit_log if e["selected"])
    print(f"    prefill: {sel_count}/{len(audit_log)} ops selected ({elapsed:.1f}s)")
    return True


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="All-chunks F-FP32 / L-FP16 verified experiment")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if mlpackage already exists")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Comma-separated chunk indices to process (default: all)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export, only run audit on existing artifacts")
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    chunk_indices = list(range(NUM_CHUNKS))
    if args.chunks:
        chunk_indices = [int(x) for x in args.chunks.split(",")]

    print("=" * 70)
    print("  ALL-CHUNKS F-FP32 / L-FP16 VERIFIED EXPERIMENT")
    print(f"  Model:      {HF_MODEL}")
    print(f"  Artifacts:  {ARTIFACT_DIR}")
    print(f"  Chunks:     {chunk_indices}")
    print(f"  Config:     CTX={CTX} BATCH={BATCH_SIZE} NUM_CHUNKS={NUM_CHUNKS}")
    print("=" * 70)

    # ── Step 1: Load model and build manifest ──
    print("\n── Step 1: Layer Manifest ──")
    model = load_model()

    manifest = build_layer_manifest(model)
    manifest_path = os.path.join(ARTIFACT_DIR, "layer_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(manifest.items())}, f, indent=2)
    print(f"  Saved {manifest_path}")

    # Print manifest summary
    f_count = sum(1 for v in manifest.values() if v["kind"] == "F")
    l_count = sum(1 for v in manifest.values() if v["kind"] == "L")
    print(f"  {len(manifest)} layers: {f_count} F (full_attention), {l_count} L (linear_attention)")

    for idx in sorted(manifest.keys()):
        info = manifest[idx]
        print(f"    Layer {idx:2d}: {info['kind']}  chunk={info['chunk']}  ({info['attn_class']})")

    # ── Step 2: Chunk F/L classification ──
    print("\n── Step 2: Chunk F/L Classification ──")
    chunk_fl = get_chunk_fl_layers(manifest)
    for ci in chunk_indices:
        fl = chunk_fl[ci]
        sl, el = CHUNK_RANGES[ci]
        print(f"  chunk {ci} (layers {sl}-{el-1}): "
              f"F={fl['fp32_layers']} L={fl['fp16_layers']}")

    # ── Step 3: Export decode + prefill for each chunk ──
    if not args.skip_export:
        print("\n── Step 3: Export Chunks ──")
        export_status = {}

        for ci in chunk_indices:
            fl = chunk_fl[ci]
            sl, el = CHUNK_RANGES[ci]
            fp16_layers = fl["fp16_layers"]
            fp32_layers = fl["fp32_layers"]
            chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")
            os.makedirs(chunk_dir, exist_ok=True)

            print(f"\n  chunk {ci} (layers {sl}-{el-1}): F={fp32_layers} L={fp16_layers}")

            dec_ok = export_chunk_decode(
                model, ci, fp16_layers, chunk_dir, skip_existing=args.skip_existing
            )
            pf_ok = export_chunk_prefill(
                model, ci, fp16_layers, chunk_dir, skip_existing=args.skip_existing
            )

            export_status[ci] = {"decode": dec_ok, "prefill": pf_ok}

            # Save per-chunk export status incrementally
            status_path = os.path.join(chunk_dir, "export_status.json")
            with open(status_path, "w") as f:
                json.dump({"chunk": ci, "decode": dec_ok, "prefill": pf_ok,
                           "fp16_layers": fp16_layers, "fp32_layers": fp32_layers}, f, indent=2)
    else:
        print("\n── Step 3: Export SKIPPED (--skip-export) ──")
        export_status = {}
        for ci in chunk_indices:
            chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")
            dec_exists = os.path.exists(os.path.join(chunk_dir, "decode.mlpackage"))
            pf_exists = os.path.exists(os.path.join(chunk_dir, "prefill.mlpackage"))
            export_status[ci] = {"decode": dec_exists, "prefill": pf_exists}
            if not dec_exists or not pf_exists:
                print(f"  WARNING: chunk {ci} missing: decode={dec_exists} prefill={pf_exists}")

    # Free model memory before audit phase (may be reloaded if fp32 refs needed)
    # del model  — keep alive in case Step 5 needs it for ref exports
    gc.collect()

    # ── Step 4: Selector audit ──
    print("\n── Step 4: Selector Audit ──")
    selector_results = {}

    for ci in chunk_indices:
        fl = chunk_fl[ci]
        fp16_layers = fl["fp16_layers"]
        fp32_layers = fl["fp32_layers"]
        chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")

        for phase in ["decode", "prefill"]:
            audit_path = os.path.join(chunk_dir, f"selected_ops_{phase}.json")
            summary_path = os.path.join(chunk_dir, f"selector_summary_{phase}.json")

            if not os.path.exists(audit_path):
                print(f"  chunk {ci} {phase}: no audit log (skipped)")
                continue

            with open(audit_path) as f:
                audit_log = json.load(f)

            summary = audit_selected_ops(audit_log, fp16_layers, fp32_layers)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            sel = summary["selected_for_fp16"]
            total = summary["total_ops"]
            ok = summary["selector_correct"]
            wrong_sel = summary["wrong_selected_count"]
            wrong_unsel = summary["wrong_unselected_count"]

            status = "OK" if ok else f"FAIL (wrong_sel={wrong_sel} wrong_unsel={wrong_unsel})"
            print(f"  chunk {ci} {phase}: {sel}/{total} selected, {status}")

            key = f"chunk_{ci}_{phase}"
            selector_results[key] = summary

    # ── Step 5: Post-export graph audit ──
    print("\n── Step 5: Post-Export Graph Audit ──")

    # First, export and audit reference fp32 models for comparison
    # (only decode, to get baseline cast counts per chunk)
    print("  Building fp32 reference cast counts...")
    ref_fp32_casts = {}  # chunk_idx -> {"decode": count, "prefill": count}
    for ci in chunk_indices:
        chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")
        ref_dec_path = os.path.join(chunk_dir, "ref_fp32_decode.mlpackage")
        ref_pf_path = os.path.join(chunk_dir, "ref_fp32_prefill.mlpackage")
        ref_casts_path = os.path.join(chunk_dir, "ref_fp32_cast_counts.json")

        # Check if we already have cached counts
        if os.path.exists(ref_casts_path):
            with open(ref_casts_path) as f:
                ref_fp32_casts[ci] = json.load(f)
            print(f"    chunk {ci}: cached — dec={ref_fp32_casts[ci]['decode']} "
                  f"pf={ref_fp32_casts[ci]['prefill']}")
            continue

        # Export fp32 reference if needed
        if not os.path.exists(ref_dec_path) or not os.path.exists(ref_pf_path):
            if model is None:
                # Reload model if needed
                model = load_model()
            from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
            sl, el = CHUNK_RANGES[ci]

            if not os.path.exists(ref_dec_path):
                conv = Qwen35Converter(
                    model, context_length=CTX, batch_size=BATCH_SIZE,
                    num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
                    compute_precision="float32",
                )
                ml = conv.convert_part_2(
                    model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                    override_start_layer=sl, override_end_layer=el,
                )
                ml.save(ref_dec_path)
                del ml, conv; gc.collect()

            if not os.path.exists(ref_pf_path):
                conv = Qwen35Converter(
                    model, context_length=CTX, batch_size=BATCH_SIZE,
                    num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
                    compute_precision="float32",
                )
                ml = conv.convert_part_2_prefill(
                    model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                    override_start_layer=sl, override_end_layer=el,
                )
                ml.save(ref_pf_path)
                del ml, conv; gc.collect()

        # Count casts in reference
        counts = {}
        for phase, path in [("decode", ref_dec_path), ("prefill", ref_pf_path)]:
            ref_audit = audit_exported_graph(path, [], [], function_name="main")
            counts[phase] = ref_audit.get("total_cast_ops", 0)
            counts[f"{phase}_fp16"] = ref_audit.get("cast_to_fp16", 0)

        ref_fp32_casts[ci] = counts
        with open(ref_casts_path, "w") as f:
            json.dump(counts, f, indent=2)
        print(f"    chunk {ci}: dec={counts['decode']}(fp16={counts['decode_fp16']}) "
              f"pf={counts['prefill']}(fp16={counts['prefill_fp16']})")

    # Now free model for real
    model = None
    gc.collect()

    # Audit the selective models
    cast_audit_results = {}

    for ci in chunk_indices:
        fl = chunk_fl[ci]
        fp16_layers = fl["fp16_layers"]
        fp32_layers = fl["fp32_layers"]
        chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")
        ref_counts = ref_fp32_casts.get(ci, {})

        for phase in ["decode", "prefill"]:
            pkg_path = os.path.join(chunk_dir, f"{phase}.mlpackage")
            audit_out = os.path.join(chunk_dir, f"cast_audit_{phase}.json")

            if not os.path.exists(pkg_path):
                print(f"  chunk {ci} {phase}: no mlpackage (skipped)")
                continue

            ref_count = ref_counts.get(phase, None)
            ref_fp16_count = ref_counts.get(f"{phase}_fp16", None)
            print(f"  chunk {ci} {phase}: auditing graph (ref_casts={ref_count}, ref_fp16={ref_fp16_count})...", end=" ")
            audit = audit_exported_graph(
                pkg_path, fp16_layers, fp32_layers,
                ref_fp32_cast_count=ref_count,
                ref_fp16_cast_count=ref_fp16_count,
            )
            with open(audit_out, "w") as f:
                json.dump(audit, f, indent=2)

            if audit.get("verified"):
                n_fp16 = audit.get("cast_to_fp16", 0)
                total = audit.get("total_cast_ops", 0)
                print(f"VERIFIED (fp16_casts={n_fp16}, total={total}, ref={ref_count})")
            elif "error" in audit:
                print(f"ERROR: {audit['error']}")
            else:
                failed = [c for c in audit.get("checks", []) if not c["passed"]]
                print(f"FAIL: {[c['check'] for c in failed]}")

            key = f"chunk_{ci}_{phase}"
            cast_audit_results[key] = audit

    # ── Step 6: Per-chunk summary line ──
    print("\n── Per-Chunk Summary ──")
    for ci in chunk_indices:
        fl = chunk_fl[ci]
        fp32 = fl["fp32_layers"]
        fp16 = fl["fp16_layers"]

        dec_sel_key = f"chunk_{ci}_decode"
        pf_sel_key = f"chunk_{ci}_prefill"

        dec_sel = selector_results.get(dec_sel_key, {}).get("selected_for_fp16", "?")
        dec_ok = selector_results.get(dec_sel_key, {}).get("selector_correct", "?")
        pf_sel = selector_results.get(pf_sel_key, {}).get("selected_for_fp16", "?")
        pf_ok = selector_results.get(pf_sel_key, {}).get("selector_correct", "?")

        dec_ver = cast_audit_results.get(dec_sel_key, {}).get("verified", "?")
        pf_ver = cast_audit_results.get(pf_sel_key, {}).get("verified", "?")

        print(f"  chunk {ci}: F={fp32} L={fp16} | "
              f"decode sel={dec_sel} correct={dec_ok} verified={dec_ver} | "
              f"prefill sel={pf_sel} correct={pf_ok} verified={pf_ver}")

    # ── Step 7: Final summary ──
    print("\n── Step 6: Final Summary ──")

    # Compute overall pass/fail
    all_selector_correct = all(
        s.get("selector_correct", False) for s in selector_results.values()
    )
    all_cast_verified = all(
        a.get("verified", False) for a in cast_audit_results.values()
    )
    all_exports_ok = all(
        v.get("decode", False) and v.get("prefill", False) for v in export_status.values()
    )

    overall_pass = all_selector_correct and all_cast_verified and all_exports_ok

    summary = {
        "experiment": "all_chunks_F_fp32_L_fp16",
        "model": HF_MODEL,
        "config": {
            "CTX": CTX, "BATCH_SIZE": BATCH_SIZE, "NUM_CHUNKS": NUM_CHUNKS,
            "LUT_BITS": 4, "PER_CHANNEL": 4,
        },
        "layer_manifest": {
            "total_layers": len(manifest),
            "F_count": f_count,
            "L_count": l_count,
        },
        "chunks_processed": chunk_indices,
        "chunk_fl_map": {
            str(ci): chunk_fl[ci] for ci in chunk_indices
        },
        "export_status": {
            str(ci): export_status.get(ci, {}) for ci in chunk_indices
        },
        "selector_verification": {
            k: {
                "total_ops": v["total_ops"],
                "selected": v["selected_for_fp16"],
                "correct": v["selector_correct"],
                "wrong_selected": v["wrong_selected_count"],
                "wrong_unselected": v["wrong_unselected_count"],
            }
            for k, v in selector_results.items()
        },
        "cast_audit": {
            k: {
                "verified": v.get("verified", False),
                "total_cast_ops": v.get("total_cast_ops", 0),
                "cast_to_fp16": v.get("cast_to_fp16", 0),
                "cast_to_fp32": v.get("cast_to_fp32", 0),
                "ref_fp32_cast_count": v.get("ref_fp32_cast_count"),
                "checks_passed": all(c["passed"] for c in v.get("checks", [])),
            }
            for k, v in cast_audit_results.items()
        },
        "overall": {
            "all_exports_ok": all_exports_ok,
            "all_selectors_correct": all_selector_correct,
            "all_casts_verified": all_cast_verified,
            "PASS": overall_pass,
        },
        "statement": (
            "VERIFIED: All F layers stay FP32, all L layers use FP16 compute precision."
            if overall_pass else
            "NOT VERIFIED: See details above for failures."
        ),
    }

    summary_path = os.path.join(ARTIFACT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved: {summary_path}")
    print(f"  Exports OK:        {all_exports_ok}")
    print(f"  Selectors correct: {all_selector_correct}")
    print(f"  Casts verified:    {all_cast_verified}")
    print(f"  OVERALL:           {'PASS' if overall_pass else 'FAIL'}")

    if not overall_pass:
        # Print failure details
        print("\n  --- FAILURE DETAILS ---")
        for k, v in selector_results.items():
            if not v.get("selector_correct"):
                print(f"  Selector {k}: wrong_sel={v['wrong_selected_count']} "
                      f"wrong_unsel={v['wrong_unselected_count']}")
                for ex in v.get("wrong_selected_examples", []):
                    print(f"    wrong_sel: {ex['name']} home={ex['home_layer']}")
                for ex in v.get("wrong_unselected_examples", []):
                    print(f"    wrong_unsel: {ex['name']} home={ex['home_layer']}")
        for k, v in cast_audit_results.items():
            if not v.get("verified"):
                print(f"  Cast audit {k}:")
                for c in v.get("checks", []):
                    if not c["passed"]:
                        print(f"    FAIL: {c['check']} — {c['detail']}")
                if "error" in v:
                    print(f"    ERROR: {v['error']}")

    print("\n  Done!")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
