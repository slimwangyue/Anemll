#!/usr/bin/env python3
"""
SHRINK F-LAYER FP32 ISLAND — Qwen3.5-4B chunk2 experiment

Investigates which FP32 op categories inside F (full_attention) layers
can be safely relaxed to FP16 without degrading generation quality.

V2 baseline: F layers ← FP32, L layers ← FP16  (373 precision-switch casts)
This experiment: progressively relax F-layer subcategories to FP16.

Scope: chunk2 only (layers 7-10, F=7, L=8,9,10)

CATEGORIES (safest → riskiest):
  1. output_boundary  — post-MLP norm + reshape at F→L boundary
  2. layer_norm       — QK norms + post-attention norm
  3. rope             — rotary position embedding math
  4. intermediate     — projections, attention, MLP, residuals
  5. kv_cache_state   — KV cache read/write ops

Variants (cumulative reduction):
  V0 (baseline)      — all F-layer ops FP32  (= V2)
  V1  minus output_boundary
  V2  minus output_boundary + layer_norm
  V3  minus output_boundary + layer_norm + rope
  V4  minus output_boundary + layer_norm + rope + intermediate
  V5  minus all (= full FP16)  [kv_cache also relaxed]

For each variant:
  1. Export chunk2 decode + prefill with modified selector
  2. Audit cast counts
  3. Assemble full 9-chunk pipeline (V2 chunks 0,1,3-8 + variant chunk2)
  4. Combine (dedup) the variant chunk
  5. Run 3-turn validation
  6. Report token match vs baseline

Output: artifacts/f_layer_fp32_shrink/
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
V2_ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "fl_precision_verified")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "f_layer_fp32_shrink")
FP32_MODEL_DIR = os.path.join(REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp32")
LAYER_PATTERN = re.compile(r"layers[._](\d+)")

# Chunk 2: layers 7-10.  F=7, L=8,9,10
CHUNK_IDX = 2
F_LAYER = 7
L_LAYERS = [8, 9, 10]
FP16_LAYERS = L_LAYERS  # L layers always FP16
FP32_LAYERS = [F_LAYER]  # F layer always FP32 (in V2 baseline)


# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — Classify F-layer ops into categories
# ═══════════════════════════════════════════════════════════════════

def classify_f_layer_ops(audit_log):
    """Classify all ops with home_layer == F_LAYER into semantic categories.

    Uses the V2 audit log from chunk2 decode. Returns:
      {category: [op_name, ...]}
    and also returns per-op classification: [{name, op_type, category}, ...]
    """
    f_ops = [e for e in audit_log if e["home_layer"] == F_LAYER]

    # Build ordered name list for positional analysis
    names = [e["name"] for e in f_ops]
    types = {e["name"]: e["op_type"] for e in f_ops}

    # Classification by name patterns + op type + position
    categories = {
        "weight_const": [],      # const weight tensors (no compute effect)
        "output_boundary": [],   # post-MLP norm + reshape for next layer
        "layer_norm": [],        # QK norms + post-attention norm
        "rope": [],              # rotary position embedding
        "kv_cache_state": [],    # KV cache read/write
        "intermediate": [],      # everything else (projections, attention, MLP, etc.)
    }

    per_op = []

    for e in f_ops:
        name = e["name"]
        op_type = e["op_type"]
        cat = _classify_single_op(name, op_type, names, types)
        categories[cat].append(name)
        per_op.append({"name": name, "op_type": op_type, "category": cat})

    return categories, per_op


def _classify_single_op(name, op_type, all_names, all_types):
    """Classify a single F-layer op into a category.

    Uses name patterns, op type, and position relative to known landmarks.
    """
    layer_prefix = f"model.model.layers.{F_LAYER}"

    # ── weight_const: const ops for model weights ──
    if op_type == "const" and layer_prefix in name:
        return "weight_const"

    # ── kv_cache_state: cache write (slice_update) and cache read (identity) ──
    if "cache" in name.lower():
        return "kv_cache_state"
    if op_type == "identity" and name.startswith("identity_"):
        return "kv_cache_state"
    # The squeeze + slice_by_index ops adjacent to cache ops
    # Op 49: squeeze 589 (before k_cache write)
    # Op 51: squeeze 613 (before v_cache write)
    # Op 53: slice_by_index 640 (k cache read)
    # Op 55: slice_by_index 647 (v cache read)
    if op_type == "squeeze" and name.isdigit():
        idx = all_names.index(name)
        # Check if next op is a cache operation
        if idx + 1 < len(all_names):
            next_name = all_names[idx + 1]
            if "cache" in next_name.lower():
                return "kv_cache_state"
    if op_type == "slice_by_index" and name.isdigit():
        idx = all_names.index(name)
        # Check if next op is identity (cache read pattern)
        if idx + 1 < len(all_names):
            next_name = all_names[idx + 1]
            if next_name.startswith("identity_"):
                return "kv_cache_state"

    # ── output_boundary: post-MLP norm + reshape for next layer ──
    # These are the LAST ops in the F layer: norm + transpose + expand_dims
    # Pattern: after the last residual add (hidden_states.19), there's a norm
    # block (mul, concat, layer_norm, slice) then reshape (mul, transpose, expand_dims)
    # Detect by suffix patterns in the tail of the op list
    if name in ("x_bsh.1", "x_bc1s.1"):
        return "output_boundary"
    # Find the last layer_norm — ops after the last add (residual) that form the
    # output norm are output_boundary
    if name.startswith("normed.") or name.startswith("x_bsh") or name.startswith("x_bc1s"):
        # These could be either internal norms or output boundary
        # Check position: output boundary is at the end
        idx = all_names.index(name)
        # Count remaining ops after this one (excluding this)
        remaining = len(all_names) - idx - 1
        if remaining <= 6:  # last 7 ops form the output boundary
            return "output_boundary"

    # Check if this is part of the tail output boundary block
    idx = all_names.index(name)
    total = len(all_names)
    if total - idx <= 7:  # last 7 ops
        return "output_boundary"

    # ── rope: rotary position embedding ──
    # Pattern: ops between query_states concat and cache writes
    # Names: q_rot, k_rot, q_pass, k_pass, x1, x2, query_states, key_states
    rope_names = {"q_rot", "k_rot", "q_pass", "k_pass",
                  "q_rot.1", "k_rot.1", "x1", "x2", "x1.1", "x2.1",
                  "query_states.1", "key_states.1"}
    if name in rope_names:
        return "rope"
    # RoPE mul/concat/add ops (numeric names in the rope region)
    if name.isdigit() and op_type in ("mul", "concat", "add", "slice_by_index"):
        idx = all_names.index(name)
        # Check if we're between q_rot.1/k_rot.1 area and query_states.1
        # RoPE region is roughly between QK norm outputs and cache writes
        for i, n in enumerate(all_names):
            if n in ("q_rot.1", "k_rot.1"):
                rope_start = i
                break
        else:
            rope_start = 999
        for i, n in enumerate(all_names):
            if n == "key_states.1":
                rope_end = i
                break
        else:
            rope_end = 0
        if rope_start <= idx <= rope_end:
            return "rope"

    # ── layer_norm: QK norms + post-attention norm ──
    # QK norm: layer_norm normed.5/normed.9 + associated mul/concat/slice
    # Post-attn norm: layer_norm normed.13 + associated mul/concat/slice
    if op_type == "layer_norm":
        # Check if this is NOT the output boundary norm (already handled above)
        return "layer_norm"
    if name.startswith("normed."):
        return "layer_norm"
    # The mul + concat that prepare input for layer_norm
    if name.startswith("input.") and op_type == "concat":
        return "layer_norm"
    # The mul ops that precede concat for norm input
    if op_type == "mul" and name.isdigit():
        idx = all_names.index(name)
        if idx + 1 < len(all_names):
            next_name = all_names[idx + 1]
            if next_name.startswith("input.") and all_types.get(next_name) == "concat":
                return "layer_norm"
    # The scaling mul after slice (q, k, x.1, x_bsh.1 patterns)
    if name in ("q", "k", "x.1"):
        return "layer_norm"

    # ── Everything else is intermediate compute ──
    return "intermediate"


# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — MIL selector with category-based relaxation
# ═══════════════════════════════════════════════════════════════════

def make_shrink_selector(fp16_layers, relax_categories, category_op_names,
                         audit_log=None):
    """Return an FP16ComputePrecision op_selector.

    Same max-layer home attribution as V2, but ops in `relax_categories`
    that belong to the F layer are also selected for FP16.

    Args:
        fp16_layers: L layer indices (always FP16)
        relax_categories: set of category names to relax to FP16
        category_op_names: {category: set(op_names)} from classification
        audit_log: optional list for logging decisions
    """
    fp16_set = set(fp16_layers)
    has_fp16_layers = len(fp16_set) > 0

    # Build lookup: op_name -> should_relax
    relax_names = set()
    for cat in relax_categories:
        if cat in category_op_names:
            relax_names |= set(category_op_names[cat])

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
            # Pre-layer ops: FP16 if chunk has any L layers
            selected = has_fp16_layers
        else:
            home = max(layers)
            if home in fp16_set:
                selected = True  # L layer → FP16
            elif home == F_LAYER and op.name in relax_names:
                selected = True  # Relaxed F-layer op → FP16
            else:
                selected = False  # Non-relaxed F-layer op → FP32

        if audit_log is not None:
            audit_log.append({
                "name": op.name,
                "op_type": op.op_type,
                "transitive_layers": sorted(layers),
                "home_layer": home,
                "selected": selected,
                "relaxed": selected and home == F_LAYER,
            })
        return selected

    return selector


# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — Export helpers (reuse V2 pattern)
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


def export_variant_chunk(model, variant_name, relax_categories, category_op_names,
                         variant_dir, skip_existing=False):
    """Export decode + prefill for chunk2 with the given category relaxation.

    Returns dict with paths and audit info.
    """
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    sl, el = CHUNK_RANGES[CHUNK_IDX]
    result = {"variant": variant_name, "relax_categories": list(relax_categories)}

    for phase, convert_fn_name in [("decode", "convert_part_2"),
                                   ("prefill", "convert_part_2_prefill")]:
        pkg_path = os.path.join(variant_dir, f"{phase}.mlpackage")
        audit_path = os.path.join(variant_dir, f"selected_ops_{phase}.json")

        if skip_existing and os.path.exists(pkg_path) and os.path.exists(audit_path):
            print(f"    [skip] {phase} (exists)")
            result[f"{phase}_path"] = pkg_path
            continue

        audit_log = []
        selector = make_shrink_selector(
            FP16_LAYERS, relax_categories, category_op_names, audit_log=audit_log
        )

        t0 = time.time()
        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
            compute_precision="float32",
        )
        conv.compute_precision = FP16ComputePrecision(op_selector=selector)

        convert_fn = getattr(conv, convert_fn_name)
        ml = convert_fn(
            model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS,
            override_start_layer=sl, override_end_layer=el,
        )
        ml.save(pkg_path)
        elapsed = time.time() - t0
        del ml, conv
        gc.collect()

        # Save audit log
        with open(audit_path, "w") as f:
            json.dump(audit_log, f, indent=1)

        # Summary
        sel = sum(1 for e in audit_log if e["selected"])
        relaxed = sum(1 for e in audit_log if e.get("relaxed"))
        total = len(audit_log)
        print(f"    {phase}: {sel}/{total} selected, {relaxed} relaxed ({elapsed:.1f}s)")

        result[f"{phase}_path"] = pkg_path
        result[f"{phase}_selected"] = sel
        result[f"{phase}_relaxed"] = relaxed
        result[f"{phase}_total"] = total

    return result


# ═══════════════════════════════════════════════════════════════════
#  STEP 4 — Cast audit (reuse V2 pattern)
# ═══════════════════════════════════════════════════════════════════

DTYPE_MAP = {10: "fp16", 11: "fp32", 22: "int16", 23: "int32", 32: "bool"}


def audit_casts(mlpackage_path, function_name="main"):
    """Count cast ops by output dtype in a MIL proto."""
    from collections import defaultdict

    spec = ct.utils.load_spec(mlpackage_path)
    program = spec.mlProgram
    func = program.functions[function_name] if function_name in program.functions \
        else program.functions[list(program.functions.keys())[0]]

    block = func.block_specializations[list(func.block_specializations.keys())[0]]
    ops = list(block.operations)

    cast_by_dtype = defaultdict(int)

    for op in ops:
        if op.type != "cast":
            continue
        if not op.outputs:
            continue
        out_dt = DTYPE_MAP.get(op.outputs[0].type.tensorType.dataType, "?")
        cast_by_dtype[out_dt] += 1

    # fp16↔fp32 precision switches = min(cast_to_fp16, cast_to_fp32)
    # (each switch needs one up-cast and one down-cast at the boundary)
    n_fp16 = cast_by_dtype.get("fp16", 0)
    n_fp32 = cast_by_dtype.get("fp32", 0)

    return {
        "total_cast_ops": sum(cast_by_dtype.values()),
        "cast_to_fp16": n_fp16,
        "cast_to_fp32": n_fp32,
        "cast_by_dtype": dict(cast_by_dtype),
        "total_ops": len(ops),
    }


def count_precision_switches_simple(mlpackage_path, function_name="main"):
    """Quick count of fp16↔fp32 cast ops (proxy for precision switches)."""
    spec = ct.utils.load_spec(mlpackage_path)
    func = spec.mlProgram.functions.get(function_name)
    if func is None:
        func = list(spec.mlProgram.functions.values())[0]
    block = list(func.block_specializations.values())[0]

    n_fp16 = 0
    n_fp32 = 0
    for op in block.operations:
        if op.type != "cast":
            continue
        if op.outputs:
            dt = op.outputs[0].type.tensorType.dataType
            if dt == 10:
                n_fp16 += 1
            elif dt == 11:
                n_fp32 += 1
    return n_fp16, n_fp32


# ═══════════════════════════════════════════════════════════════════
#  STEP 5 — Assemble full pipeline + combine + validate
# ═══════════════════════════════════════════════════════════════════

def assemble_variant_pipeline(variant_dir, assembled_dir):
    """Create a staging directory with variant chunk2 + V2 for all other chunks.

    Symlinks all V2 artifacts except chunk2 decode/prefill, which come from variant_dir.
    """
    os.makedirs(assembled_dir, exist_ok=True)

    # Symlink embed/lmhead from FP32 model dir
    for fname in ["embed_single.mlpackage", "embed_lmhead_combined.mlpackage",
                   "lm_head_nosplit.mlpackage"]:
        src = os.path.join(FP32_MODEL_DIR, fname)
        dst = os.path.join(assembled_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    # Symlink tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = os.path.join(FP32_MODEL_DIR, fname)
        dst = os.path.join(assembled_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            if os.path.islink(src):
                shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)

    # Symlink FFN chunks: V2 for all except chunk2
    for ci in range(NUM_CHUNKS):
        if ci == CHUNK_IDX:
            # Use variant
            dec_src = os.path.join(variant_dir, "decode.mlpackage")
            pf_src = os.path.join(variant_dir, "prefill.mlpackage")
        else:
            # Use V2
            dec_src = os.path.join(V2_ARTIFACT_DIR, f"chunk_{ci}", "decode.mlpackage")
            pf_src = os.path.join(V2_ARTIFACT_DIR, f"chunk_{ci}", "prefill.mlpackage")

        dec_dst = os.path.join(assembled_dir, f"ffn_LUT4_chunk{ci}.mlpackage")
        pf_dst = os.path.join(assembled_dir, f"prefill_LUT4_chunk{ci}.mlpackage")

        for s, d in [(dec_src, dec_dst), (pf_src, pf_dst)]:
            if os.path.lexists(d):
                os.unlink(d)
            os.symlink(os.path.abspath(s), d)


def combine_variant_chunk(assembled_dir, skip_existing=False):
    """Run combine on the assembled pipeline (only chunk2 needs re-combining)."""
    from anemll.utils.combine_models import _save_multifunction_dedup

    combined_dir = os.path.join(assembled_dir, "combined_LUT4_dedup")
    os.makedirs(combined_dir, exist_ok=True)

    for ci in range(NUM_CHUNKS):
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")

        if ci != CHUNK_IDX:
            # Symlink from V2 combined
            v2_combined = os.path.join(V2_ARTIFACT_DIR, "assembled",
                                       "combined_LUT4_dedup", f"chunk{ci}.mlpackage")
            if os.path.lexists(combined_path):
                if os.path.islink(combined_path):
                    os.unlink(combined_path)
                else:
                    shutil.rmtree(combined_path)
            os.symlink(os.path.abspath(v2_combined), combined_path)
            continue

        # Combine variant chunk2
        if skip_existing and os.path.exists(combined_path) and not os.path.islink(combined_path):
            print(f"    [skip] combined chunk {ci} (exists)")
            continue

        if os.path.lexists(combined_path):
            if os.path.islink(combined_path):
                os.unlink(combined_path)
            else:
                shutil.rmtree(combined_path)

        dec_path = os.path.join(assembled_dir, f"ffn_LUT4_chunk{ci}.mlpackage")
        pf_path = os.path.join(assembled_dir, f"prefill_LUT4_chunk{ci}.mlpackage")
        sources = [
            (dec_path, "main", "infer"),
            (pf_path, "main", "prefill"),
        ]

        t0 = time.time()
        _save_multifunction_dedup(sources, combined_path, dedup_weights=True, verbose=False)
        print(f"    combined chunk {ci} ({time.time()-t0:.1f}s)")


def validate_variant(assembled_dir, variant_name, max_tokens=40):
    """Run 3-turn validation on the variant pipeline. Returns token results."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(assembled_dir, use_fast=False)

    # Build stop IDs
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)

    tpl_tokens = {
        "im_start": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "think": tokenizer.convert_tokens_to_ids("<think>"),
        "nl": 198,
        "user": tokenizer.encode("user", add_special_tokens=False),
        "assistant": tokenizer.encode("assistant", add_special_tokens=False),
    }

    TURNS = [
        "What is a stack in computer science?",
        "How does it compare to a queue?",
        "Give me a Python example of each.",
    ]

    combined_dir = os.path.join(assembled_dir, "combined_LUT4_dedup")
    compute_unit = ct.ComputeUnit.CPU_AND_NE

    # Import DedupEngine from validate.py
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
    from validate import DedupEngine, run_fresh, _build_stop_ids, _get_template_tokens, _ensure_ids

    print(f"    Loading dedup engine for {variant_name}...")
    engine = DedupEngine(combined_dir, assembled_dir, compute_unit)

    print(f"    Running 3-turn validation...")
    results = run_fresh(engine, tokenizer, tpl_tokens, TURNS, max_tokens, stop_ids,
                        f"Shrink-{variant_name}")
    engine.cleanup()

    # Collect tokens per turn
    turn_tokens = []
    for r in results:
        turn_tokens.append({
            "turn": r["turn"],
            "prompt_len": r["prompt_len"],
            "tokens": r["tokens"],
            "text": r["text"],
            "end_pos": r["end_pos"],
        })

    return turn_tokens


def load_v2_baseline_tokens():
    """Load V2 baseline tokens from previous validation (or run fresh)."""
    cache_path = os.path.join(ARTIFACT_DIR, "v2_baseline_tokens.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return None


def save_v2_baseline_tokens(tokens):
    cache_path = os.path.join(ARTIFACT_DIR, "v2_baseline_tokens.json")
    with open(cache_path, "w") as f:
        json.dump(tokens, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
#  STEP 6 — Compare variant tokens to V2 baseline
# ═══════════════════════════════════════════════════════════════════

def compare_tokens(baseline_turns, variant_turns):
    """Compare token-by-token across all turns. Returns per-turn stats and overall."""
    results = []
    total_match = 0
    total_tokens = 0

    for bt, vt in zip(baseline_turns, variant_turns):
        b_toks = bt["tokens"]
        v_toks = vt["tokens"]
        matches = sum(1 for a, b in zip(b_toks, v_toks) if a == b)
        total = min(len(b_toks), len(v_toks))
        pct = 100 * matches / total if total > 0 else 0

        results.append({
            "turn": bt["turn"],
            "matches": matches,
            "total": total,
            "pct": pct,
            "baseline_len": len(b_toks),
            "variant_len": len(v_toks),
        })
        total_match += matches
        total_tokens += total

    overall_pct = 100 * total_match / total_tokens if total_tokens > 0 else 0
    return results, overall_pct


# ═══════════════════════════════════════════════════════════════════
#  STEP 7 — Summary report
# ═══════════════════════════════════════════════════════════════════

def generate_report(all_results, classification_summary, report_path):
    """Generate markdown report from all variant results."""
    lines = [
        "# F-Layer FP32 Shrink Experiment — Qwen3.5-4B Chunk 2",
        "",
        "## Classification Summary",
        "",
        "| Category | Op Count | Compute Ops | Description |",
        "|----------|----------|-------------|-------------|",
    ]

    descs = {
        "weight_const": "Const weight tensors (no compute effect)",
        "output_boundary": "Post-MLP norm + reshape at F→L boundary",
        "layer_norm": "QK norms + post-attention norm",
        "rope": "Rotary position embedding math",
        "kv_cache_state": "KV cache read/write ops",
        "intermediate": "Projections, attention, MLP, residuals",
    }

    for cat, info in sorted(classification_summary.items()):
        lines.append(f"| {cat} | {info['total']} | {info['compute']} | {descs.get(cat, '')} |")

    lines += [
        "",
        "## Variant Results",
        "",
        "| Variant | Relaxed Cats | FP32 Ops Remaining | Cast-to-FP16 | Cast-to-FP32 | "
        "Turn 1 Match | Turn 2 Match | Turn 3 Match | Overall |",
        "|---------|-------------|--------------------|--------------|--------------"
        "|-------------|-------------|-------------|---------|",
    ]

    for vr in all_results:
        cats = ", ".join(vr["relax_categories"]) if vr["relax_categories"] else "(none)"
        fp32_rem = vr.get("fp32_ops_remaining", "?")
        c16 = vr.get("decode_cast_fp16", "?")
        c32 = vr.get("decode_cast_fp32", "?")
        turns = vr.get("turn_results", [])
        t_strs = [f"{t['pct']:.0f}%" for t in turns] if turns else ["?", "?", "?"]
        while len(t_strs) < 3:
            t_strs.append("?")
        overall = f"{vr.get('overall_pct', 0):.0f}%"
        lines.append(f"| {vr['name']} | {cats} | {fp32_rem} | {c16} | {c32} | "
                     f"{t_strs[0]} | {t_strs[1]} | {t_strs[2]} | {overall} |")

    lines += [
        "",
        "## Conclusions",
        "",
        "Categories that can be safely relaxed to FP16 (100% token match):",
        "",
    ]

    safe_cats = []
    unsafe_cats = []
    for vr in all_results:
        if vr.get("overall_pct", 0) == 100 and vr["relax_categories"]:
            # The newly added category in this variant
            new_cat = vr["relax_categories"][-1] if vr["relax_categories"] else None
            if new_cat:
                safe_cats.append(new_cat)
        elif vr["relax_categories"]:
            new_cat = vr["relax_categories"][-1] if vr["relax_categories"] else None
            if new_cat:
                unsafe_cats.append(new_cat)

    if safe_cats:
        for cat in safe_cats:
            lines.append(f"- **{cat}** — SAFE to relax")
    else:
        lines.append("- (none found)")

    if unsafe_cats:
        lines.append("")
        lines.append("Categories that CANNOT be safely relaxed:")
        for cat in unsafe_cats:
            lines.append(f"- **{cat}** — UNSAFE (degrades generation)")

    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

# Variant definitions: (name, categories_to_relax)
VARIANTS = [
    ("V0_baseline", []),
    ("V1_minus_output_boundary", ["output_boundary"]),
    ("V2_minus_norm", ["output_boundary", "layer_norm"]),
    ("V3_minus_rope", ["output_boundary", "layer_norm", "rope"]),
    ("V4_minus_intermediate", ["output_boundary", "layer_norm", "rope", "intermediate"]),
    ("V5_full_fp16", ["output_boundary", "layer_norm", "rope", "intermediate", "kv_cache_state"]),
    # ── Diagnostic variants (non-cumulative) ──
    ("D1_only_intermediate", ["intermediate"]),  # Just the compute core
    ("D2_norm_rope_intermediate", ["layer_norm", "rope", "intermediate"]),  # Contiguous compute block
    ("D3_only_norm", ["layer_norm"]),  # Just norms → does it fail alone?
    ("D4_only_rope", ["rope"]),  # Just rope → does it fail alone?
    ("D5_only_kv_cache", ["kv_cache_state"]),  # Just cache → does it fail alone?
]


def main():
    parser = argparse.ArgumentParser(
        description="Shrink F-layer FP32 island experiment")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if mlpackage already exists")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated variant names to run (default: all)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip validation (only export + audit)")
    parser.add_argument("--tokens", type=int, default=40,
                        help="Max tokens per turn for validation")
    parser.add_argument("--only-classify", action="store_true",
                        help="Only run classification, don't export")
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("=" * 70)
    print("  SHRINK F-LAYER FP32 ISLAND — Qwen3.5-4B chunk2")
    print(f"  Artifacts:  {ARTIFACT_DIR}")
    print(f"  V2 source:  {V2_ARTIFACT_DIR}")
    print(f"  Chunk:      {CHUNK_IDX} (layers {CHUNK_RANGES[CHUNK_IDX][0]}-"
          f"{CHUNK_RANGES[CHUNK_IDX][1]-1})")
    print(f"  F layer:    {F_LAYER}")
    print(f"  L layers:   {L_LAYERS}")
    print("=" * 70)

    # ── Step 1: Load V2 audit log and classify F-layer ops ──
    print("\n── Step 1: Classify F-layer ops ──")
    v2_audit_path = os.path.join(V2_ARTIFACT_DIR, "chunk_2", "selected_ops_decode.json")
    if not os.path.exists(v2_audit_path):
        print(f"  ERROR: V2 audit log not found: {v2_audit_path}")
        print("  Run all_chunks_F_fp32_L_fp16_verified.py first.")
        sys.exit(1)

    with open(v2_audit_path) as f:
        v2_audit = json.load(f)

    categories, per_op = classify_f_layer_ops(v2_audit)

    # Summary
    classification_summary = {}
    print(f"\n  F-layer ops (home={F_LAYER}): {sum(len(v) for v in categories.values())} total")
    for cat in ["weight_const", "output_boundary", "layer_norm", "rope",
                "kv_cache_state", "intermediate"]:
        ops = categories[cat]
        compute = [n for n in ops if not any(
            e["op_type"] == "const" for e in v2_audit
            if e["name"] == n and e["home_layer"] == F_LAYER
        )]
        classification_summary[cat] = {"total": len(ops), "compute": len(compute)}
        print(f"    {cat:20s}: {len(ops):3d} total, {len(compute):3d} compute")
        for n in ops:
            ot = next((e["op_type"] for e in v2_audit if e["name"] == n and e["home_layer"] == F_LAYER), "?")
            print(f"      {ot:20s} {n}")

    # Save classification
    class_path = os.path.join(ARTIFACT_DIR, "op_classification.json")
    with open(class_path, "w") as f:
        json.dump({
            "categories": {k: v for k, v in categories.items()},
            "per_op": per_op,
            "summary": classification_summary,
        }, f, indent=2)
    print(f"\n  Saved {class_path}")

    if args.only_classify:
        print("\n  --only-classify: stopping here.")
        return

    # ── Step 2: Load model ──
    print("\n── Step 2: Load model ──")
    model = load_model()

    # ── Step 3: Select variants to run ──
    selected_variants = VARIANTS
    if args.variants:
        var_names = set(args.variants.split(","))
        selected_variants = [(n, c) for n, c in VARIANTS if n in var_names]

    # ── Step 4: Export + audit each variant ──
    print("\n── Step 3: Export variants ──")
    all_results = []

    for var_name, relax_cats in selected_variants:
        print(f"\n  === {var_name} (relax: {relax_cats or 'none'}) ===")
        variant_dir = os.path.join(ARTIFACT_DIR, var_name)
        os.makedirs(variant_dir, exist_ok=True)

        # Export
        export_result = export_variant_chunk(
            model, var_name, set(relax_cats), categories,
            variant_dir, skip_existing=args.skip_existing,
        )

        # Audit casts
        print(f"    Auditing casts...")
        dec_path = os.path.join(variant_dir, "decode.mlpackage")
        pf_path = os.path.join(variant_dir, "prefill.mlpackage")

        dec_audit = audit_casts(dec_path)
        pf_audit = audit_casts(pf_path)

        audit_result = {
            "decode": dec_audit,
            "prefill": pf_audit,
        }
        with open(os.path.join(variant_dir, "cast_audit.json"), "w") as f:
            json.dump(audit_result, f, indent=2)

        print(f"    decode:  total_casts={dec_audit['total_cast_ops']}, "
              f"fp16={dec_audit['cast_to_fp16']}, fp32={dec_audit['cast_to_fp32']}")
        print(f"    prefill: total_casts={pf_audit['total_cast_ops']}, "
              f"fp16={pf_audit['cast_to_fp16']}, fp32={pf_audit['cast_to_fp32']}")

        # Count remaining FP32 ops in F layer
        sel_audit_path = os.path.join(variant_dir, "selected_ops_decode.json")
        fp32_remaining = "?"
        if os.path.exists(sel_audit_path):
            with open(sel_audit_path) as f:
                sel_audit = json.load(f)
            fp32_remaining = sum(
                1 for e in sel_audit
                if e["home_layer"] == F_LAYER and not e["selected"]
            )

        vr = {
            "name": var_name,
            "relax_categories": relax_cats,
            "fp32_ops_remaining": fp32_remaining,
            "decode_cast_fp16": dec_audit["cast_to_fp16"],
            "decode_cast_fp32": dec_audit["cast_to_fp32"],
            "decode_total_casts": dec_audit["total_cast_ops"],
            "prefill_cast_fp16": pf_audit["cast_to_fp16"],
            "prefill_cast_fp32": pf_audit["cast_to_fp32"],
        }
        all_results.append(vr)

    # Free model before validation
    del model
    gc.collect()

    # ── Step 5: Assemble + combine + validate each variant ──
    if not args.skip_validate:
        print("\n── Step 4: Assemble + Validate ──")

        # First, get V2 baseline tokens
        baseline_tokens = load_v2_baseline_tokens()

        for vr in all_results:
            var_name = vr["name"]
            variant_dir = os.path.join(ARTIFACT_DIR, var_name)
            assembled_dir = os.path.join(variant_dir, "assembled")

            print(f"\n  === Validating {var_name} ===")

            # Assemble
            print(f"    Assembling pipeline...")
            assemble_variant_pipeline(variant_dir, assembled_dir)

            # Combine
            print(f"    Combining chunk2...")
            combine_variant_chunk(assembled_dir, skip_existing=args.skip_existing)

            # Validate
            turn_tokens = validate_variant(
                assembled_dir, var_name, max_tokens=args.tokens
            )

            # Save tokens
            with open(os.path.join(variant_dir, "gen_tokens.json"), "w") as f:
                json.dump(turn_tokens, f, indent=2)

            # If this is V0 baseline and we don't have cached baseline tokens
            if var_name == "V0_baseline" and baseline_tokens is None:
                baseline_tokens = turn_tokens
                save_v2_baseline_tokens(turn_tokens)
                print(f"    Saved as V2 baseline reference")
                vr["turn_results"] = [{"turn": t["turn"], "matches": len(t["tokens"]),
                                        "total": len(t["tokens"]), "pct": 100.0,
                                        "baseline_len": len(t["tokens"]),
                                        "variant_len": len(t["tokens"])}
                                       for t in turn_tokens]
                vr["overall_pct"] = 100.0
                continue

            # Compare to baseline
            if baseline_tokens:
                turn_results, overall_pct = compare_tokens(baseline_tokens, turn_tokens)
                vr["turn_results"] = turn_results
                vr["overall_pct"] = overall_pct

                for tr in turn_results:
                    status = "PASS" if tr["pct"] == 100 else "FAIL"
                    print(f"    Turn {tr['turn']}: {tr['matches']}/{tr['total']} "
                          f"({tr['pct']:.0f}%) [{status}]")
                status = "PASS" if overall_pct == 100 else "FAIL"
                print(f"    Overall: {overall_pct:.0f}% [{status}]")
            else:
                print(f"    WARNING: no baseline tokens, skipping comparison")

    # ── Step 6: Save results and generate report ──
    print("\n── Step 5: Generate report ──")

    results_path = os.path.join(ARTIFACT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    report_path = os.path.join(ARTIFACT_DIR, "report.md")
    report = generate_report(all_results, classification_summary, report_path)
    print(f"\n  Saved {report_path}")
    print(f"  Saved {results_path}")

    # Print summary table
    print("\n── SUMMARY ──")
    print(f"  {'Variant':<30} {'FP32 Ops':>8} {'Casts':>6} {'Match':>8}")
    print(f"  {'-'*56}")
    for vr in all_results:
        fp32 = vr.get("fp32_ops_remaining", "?")
        casts = vr.get("decode_total_casts", "?")
        match = f"{vr.get('overall_pct', 0):.0f}%" if "overall_pct" in vr else "N/A"
        print(f"  {vr['name']:<30} {str(fp32):>8} {str(casts):>6} {match:>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()
