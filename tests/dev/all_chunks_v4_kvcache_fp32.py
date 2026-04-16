#!/usr/bin/env python3
"""
ALL-CHUNKS V4 PRECISION POLICY — Qwen3.5-4B

Applies V4 precision policy to all 9 chunks:
  - L layers: FP16 (unchanged from V2)
  - F layers: FP16 except kv_cache_state ops (FP32)
  - Pre-layer ops: FP16 (always)

Then assembles the full pipeline, combines with dedup,
and validates end-to-end correctness.

Phases:
  1. Export all chunks with V4 precision policy
  2. Assemble staging directory + combine (dedup)
  3. Standard 3-turn validation
  4. Custom prompt comparison vs FP32 reference
  5. Cast analysis and final report

Output: artifacts/v4_all_chunks/
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
import coremltools.optimize as cto
from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES, FFN_PER_CHANNEL

# ── paths ──
HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
V2_ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "fl_precision_verified")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "v4_all_chunks")
FP32_MODEL_DIR = os.path.join(REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp32")
LAYER_PATTERN = re.compile(r"layers[._](\d+)")

# ── F/L layer manifest ──
# Qwen3.5-4B: 8 F (full_attention), 24 L (linear_attention)
# [FLLL] 9-chunk partition
F_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
L_LAYERS = set(range(32)) - F_LAYERS


# ═══════════════════════════════════════════════════════════════════
#  V4 SELECTOR — KV cache stays FP32, everything else FP16
# ═══════════════════════════════════════════════════════════════════

def _is_kv_cache_op(op):
    """Identify kv_cache_state ops by type and graph structure.

    KV cache ops in F (full_attention) layers:
      - slice_update with "cache" in name (cache writes)
      - identity ops (cache read pass-throughs)
      - squeeze feeding cache writes (pre-write reshape)
      - slice_by_index feeding identity (cache read extraction)
    """
    name_lower = op.name.lower()

    # Direct: ops with "cache" in name (slice_update for k/v cache writes)
    if "cache" in name_lower:
        return True

    # Identity ops are used for state reads in coremltools
    if op.op_type == "identity":
        return True

    # Squeeze feeding a cache write (check output consumers)
    if op.op_type == "squeeze":
        try:
            for out_var in op.outputs:
                for child in out_var.child_ops:
                    if "cache" in child.name.lower():
                        return True
        except (AttributeError, TypeError):
            pass

    # Slice_by_index feeding an identity (cache read extraction)
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
    """V4 op_selector: FP16 for everything except F-layer kv_cache_state.

    Uses the same max-layer home attribution as V2.
    """
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
            selected = True  # Pre-layer ops: always FP16 in V4
        else:
            home = max(layers)
            if home in fp16_set:
                selected = True  # L layer → FP16
            elif home in fp32_set:
                # F layer: FP16 unless kv_cache_state
                is_cache = _is_kv_cache_op(op)
                selected = not is_cache
            else:
                selected = True  # unexpected → FP16

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
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

def load_model():
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


# ═══════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════

def get_chunk_fl(chunk_idx):
    """Return (fp32_layers, fp16_layers) for a chunk."""
    sl, el = CHUNK_RANGES[chunk_idx]
    fp32 = [li for li in range(sl, el) if li in F_LAYERS]
    fp16 = [li for li in range(sl, el) if li in L_LAYERS]
    return fp32, fp16


def _apply_selective_lut4(mlmodel, fp16_families, lut_bits=4, per_channel=FFN_PER_CHANNEL):
    """Apply LUT4 palettization while keeping specified families in FP16.

    Uses discover_weight_ops from fp16_ablation to identify weight ops
    and skip palettization for ops matching fp16_families.
    """
    from fp16_ablation import _build_selective_lut_config
    opt_config = _build_selective_lut_config(
        mlmodel, fp16_families, lut_bits=lut_bits, per_channel=per_channel,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cto.coreml.palettize_weights(mlmodel, opt_config)


# D2_fp16_attn_all: keep F-layer attention Q/K/V/O in FP16, everything else LUT4
D2_FP16_ATTN_FAMILIES = ["attn_q", "attn_kv", "attn_o"]


def export_chunk(model, chunk_idx, chunk_dir, skip_existing=False, fp16_attn=False):
    """Export decode + prefill for one chunk with V4 precision policy.

    Args:
        fp16_attn: If True, apply D2_fp16_attn_all policy — keep F-layer
            attention weights (q/k/v/o_proj) in FP16 instead of LUT4.
            Only affects chunks containing F-layers.
    """
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    sl, el = CHUNK_RANGES[chunk_idx]
    fp32_layers, fp16_layers = get_chunk_fl(chunk_idx)

    # D2 selective palettization only matters for chunks with F-layers
    use_selective_lut = fp16_attn and bool(fp32_layers)
    if use_selective_lut:
        print(f"    [D2] Selective LUT4: keeping attn Q/K/V/O in FP16 for F-layers {fp32_layers}")

    results = {}

    for phase, convert_fn_name in [("decode", "convert_part_2"),
                                   ("prefill", "convert_part_2_prefill")]:
        pkg_path = os.path.join(chunk_dir, f"{phase}.mlpackage")
        audit_path = os.path.join(chunk_dir, f"selected_ops_{phase}.json")

        if skip_existing and os.path.exists(pkg_path):
            print(f"    [skip] {phase} (exists)")
            results[phase] = {"path": pkg_path, "skipped": True}
            continue

        audit_log = []
        selector = make_v4_selector(fp16_layers, fp32_layers, audit_log=audit_log)

        # When using selective LUT, convert without palettization (lut_bits=None)
        # and apply selective LUT4 post-conversion. Otherwise use normal lut_bits=4.
        effective_lut_bits = None if use_selective_lut else 4

        t0 = time.time()
        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS, lut_bits=effective_lut_bits, per_channel=4,
            compute_precision="float32",
        )
        conv.compute_precision = FP16ComputePrecision(op_selector=selector)

        convert_fn = getattr(conv, convert_fn_name)
        ml = convert_fn(
            model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
            override_start_layer=sl, override_end_layer=el,
        )

        # Apply selective palettization: LUT4 for all weights except F-layer attention
        if use_selective_lut:
            print(f"    [{phase}] Applying selective LUT4 (D2_fp16_attn_all)...")
            t_pal = time.time()
            ml = _apply_selective_lut4(ml, D2_FP16_ATTN_FAMILIES)
            print(f"    [{phase}] Selective palettization done ({time.time()-t_pal:.1f}s)")

        ml.save(pkg_path)
        elapsed = time.time() - t0
        del ml, conv
        gc.collect()

        # Save audit log
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
#  CAST AUDIT
# ═══════════════════════════════════════════════════════════════════

DTYPE_MAP = {10: "fp16", 11: "fp32", 22: "int16", 23: "int32", 32: "bool"}


def count_casts(mlpackage_path, function_name="main"):
    """Count cast ops by output dtype."""
    from collections import defaultdict

    spec = ct.utils.load_spec(mlpackage_path)
    funcs = spec.mlProgram.functions
    func = funcs[function_name] if function_name in funcs else list(funcs.values())[0]
    block = list(func.block_specializations.values())[0]

    casts = defaultdict(int)
    for op in block.operations:
        if op.type == "cast" and op.outputs:
            dt = DTYPE_MAP.get(op.outputs[0].type.tensorType.dataType, "?")
            casts[dt] += 1

    return {
        "total": sum(casts.values()),
        "fp16": casts.get("fp16", 0),
        "fp32": casts.get("fp32", 0),
    }


def count_combined_casts(combined_path, fn_name="infer"):
    """Count casts in a combined (multifunction) model."""
    try:
        return count_casts(combined_path, fn_name)
    except Exception:
        return {"total": 0, "fp16": 0, "fp32": 0}


# ═══════════════════════════════════════════════════════════════════
#  ASSEMBLE + COMBINE
# ═══════════════════════════════════════════════════════════════════

def assemble_pipeline(assembled_dir):
    """Create staging directory with V4 chunks + shared embed/lmhead."""
    os.makedirs(assembled_dir, exist_ok=True)

    # Symlink embed/lmhead from FP32 model dir
    for fname in ["embed_single.mlpackage", "embed_lmhead_combined.mlpackage",
                   "embed_prefill.mlpackage", "lm_head_nosplit.mlpackage"]:
        src = os.path.join(FP32_MODEL_DIR, fname)
        dst = os.path.join(assembled_dir, fname)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)

    # Copy tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = os.path.join(FP32_MODEL_DIR, fname)
        dst = os.path.join(assembled_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Symlink V4 chunks
    for ci in range(NUM_CHUNKS):
        chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")
        dec_src = os.path.join(chunk_dir, "decode.mlpackage")
        pf_src = os.path.join(chunk_dir, "prefill.mlpackage")

        dec_dst = os.path.join(assembled_dir, f"ffn_LUT4_chunk{ci}.mlpackage")
        pf_dst = os.path.join(assembled_dir, f"prefill_LUT4_chunk{ci}.mlpackage")

        for s, d in [(dec_src, pf_dst), (dec_src, dec_dst)]:
            pass
        for s, d in [(dec_src, dec_dst), (pf_src, pf_dst)]:
            if os.path.lexists(d):
                os.unlink(d)
            os.symlink(os.path.abspath(s), d)


def combine_all(assembled_dir, skip_existing=False):
    """Combine decode+prefill into multifunction dedup models."""
    from anemll.utils.combine_models import _save_multifunction_dedup

    combined_dir = os.path.join(assembled_dir, "combined_LUT4_dedup")
    os.makedirs(combined_dir, exist_ok=True)

    for ci in range(NUM_CHUNKS):
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")

        if skip_existing and os.path.exists(combined_path) and not os.path.islink(combined_path):
            print(f"    [skip] chunk {ci} (exists)")
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
        print(f"    chunk {ci} ({time.time() - t0:.1f}s)")

    return combined_dir


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION — standard 3-turn
# ═══════════════════════════════════════════════════════════════════

def validate_standard(assembled_dir, max_tokens=40):
    """Run standard 3-turn validation using validate.py's DedupEngine."""
    from transformers import AutoTokenizer
    from validate import (DedupEngine, run_fresh, run_incremental,
                          _build_stop_ids, _get_template_tokens, _ensure_ids)

    TURNS = [
        "What is a stack in computer science?",
        "How does it compare to a queue?",
        "Give me a Python example of each.",
    ]

    combined_dir = os.path.join(assembled_dir, "combined_LUT4_dedup")
    compute_unit = ct.ComputeUnit.CPU_AND_NE

    tokenizer = AutoTokenizer.from_pretrained(assembled_dir, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)
    tpl_tokens = _get_template_tokens(tokenizer)

    print("    Loading V4 dedup engine...")
    engine = DedupEngine(combined_dir, assembled_dir, compute_unit)

    print("    Running fresh mode...")
    fresh = run_fresh(engine, tokenizer, tpl_tokens, TURNS, max_tokens,
                      stop_ids, "V4-fresh")

    print("    Running incremental mode...")
    engine.reset_all()
    inc = run_incremental(engine, tokenizer, tpl_tokens, TURNS, max_tokens,
                          stop_ids, "V4-incremental")

    engine.cleanup()

    # Compare fresh vs incremental
    results = {"fresh": fresh, "incremental": inc, "turns": TURNS}
    all_pass = True
    for ti in range(len(TURNS)):
        f_toks = fresh[ti]["tokens"]
        i_toks = inc[ti]["tokens"]
        matches = sum(1 for a, b in zip(f_toks, i_toks) if a == b)
        total = min(len(f_toks), len(i_toks))
        pct = 100 * matches / total if total > 0 else 0
        status = "PASS" if pct == 100 else "FAIL"
        if pct < 100:
            all_pass = False
        print(f"      Turn {ti+1} fresh vs inc: {matches}/{total} ({pct:.0f}%) [{status}]")

    results["fresh_vs_inc_pass"] = all_pass
    return results


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION — custom prompts + FP32 comparison
# ═══════════════════════════════════════════════════════════════════

def _generate_single_turn(engine, tokenizer, prompt, max_tokens, stop_ids):
    """Generate tokens for a single prompt."""
    tpl_tokens = {
        "im_start": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "think": tokenizer.convert_tokens_to_ids("<think>"),
        "nl": 198,
        "user": tokenizer.encode("user", add_special_tokens=False),
        "assistant": tokenizer.encode("assistant", add_special_tokens=False),
    }

    from validate import _ensure_ids

    conversation = [{"role": "user", "content": prompt}]
    input_ids = _ensure_ids(tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=True))
    token_list = input_ids[0].tolist()

    engine.reset_all()
    gen_tokens, end_pos, pf_ms, dc_ms = engine.prefill_and_decode(
        token_list, 0, max_tokens, stop_ids)

    raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "prompt_len": len(token_list),
        "tokens": gen_tokens,
        "text": raw_text,
        "end_pos": end_pos,
        "prefill_ms": pf_ms,
        "decode_ms": dc_ms,
    }


def _check_repetition(text, min_len=20, max_repeats=3):
    """Check if any substring of length >= min_len repeats > max_repeats times."""
    for length in range(min_len, min(60, len(text) // 2)):
        for start in range(len(text) - length * 2):
            substr = text[start:start + length]
            count = text.count(substr)
            if count > max_repeats:
                return True, substr[:40]
    return False, None


def validate_custom(assembled_dir, max_tokens=120):
    """Run custom prompts on V4 and FP32, compare."""
    from transformers import AutoTokenizer
    from validate import DedupEngine, _build_stop_ids

    PROMPTS = [
        "What is a stack in computer science?",
        "教我做红烧鱼",
        "A farmer has 17 sheep. All but 9 run away. How many are left?",
    ]

    tokenizer = AutoTokenizer.from_pretrained(assembled_dir, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)
    compute_unit = ct.ComputeUnit.CPU_AND_NE

    combined_v4 = os.path.join(assembled_dir, "combined_LUT4_dedup")
    combined_fp32 = os.path.join(FP32_MODEL_DIR, "combined_LUT4_dedup")

    # ── Generate with V4 model ──
    print("    Loading V4 model...")
    v4_engine = DedupEngine(combined_v4, assembled_dir, compute_unit)
    v4_results = []
    for prompt in PROMPTS:
        print(f"      V4: {prompt[:50]}...")
        r = _generate_single_turn(v4_engine, tokenizer, prompt, max_tokens, stop_ids)
        v4_results.append(r)
    v4_engine.cleanup()
    gc.collect()

    # ── Generate with FP32 reference ──
    print("    Loading FP32 reference model...")
    fp32_engine = DedupEngine(combined_fp32, FP32_MODEL_DIR, compute_unit)
    fp32_results = []
    for prompt in PROMPTS:
        print(f"      FP32: {prompt[:50]}...")
        r = _generate_single_turn(fp32_engine, tokenizer, prompt, max_tokens, stop_ids)
        fp32_results.append(r)
    fp32_engine.cleanup()
    gc.collect()

    # ── Compare ──
    comparison = []
    for v4r, fp32r in zip(v4_results, fp32_results):
        v4_toks = v4r["tokens"]
        fp32_toks = fp32r["tokens"]
        matches = sum(1 for a, b in zip(v4_toks, fp32_toks) if a == b)
        total = min(len(v4_toks), len(fp32_toks))
        pct = 100 * matches / total if total > 0 else 0

        rep_v4, rep_substr = _check_repetition(v4r["text"])
        rep_fp32, _ = _check_repetition(fp32r["text"])

        comparison.append({
            "prompt": v4r["prompt"],
            "v4_text": v4r["text"],
            "fp32_text": fp32r["text"],
            "v4_tokens": v4_toks,
            "fp32_tokens": fp32_toks,
            "token_matches": matches,
            "token_total": total,
            "token_pct": pct,
            "v4_len": len(v4_toks),
            "fp32_len": len(fp32_toks),
            "v4_repetition": rep_v4,
            "fp32_repetition": rep_fp32,
            "v4_prefill_ms": v4r["prefill_ms"],
            "v4_decode_ms": v4r["decode_ms"],
            "fp32_prefill_ms": fp32r["prefill_ms"],
            "fp32_decode_ms": fp32r["decode_ms"],
        })

        status = "PASS" if pct == 100 else f"DRIFT ({pct:.0f}%)"
        rep_flag = " [REPETITION]" if rep_v4 else ""
        print(f"      {v4r['prompt'][:40]:40s}: {matches}/{total} ({pct:.0f}%) [{status}]{rep_flag}")

    return {"prompts": PROMPTS, "comparison": comparison,
            "v4_results": v4_results, "fp32_results": fp32_results}


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="All-chunks V4 precision policy — Qwen3.5-4B")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if mlpackage already exists")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export phase, only assemble/validate")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip validation phase")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Comma-separated chunk indices to export (default: all)")
    parser.add_argument("--tokens", type=int, default=40,
                        help="Max tokens per turn for standard validation")
    parser.add_argument("--custom-tokens", type=int, default=120,
                        help="Max tokens for custom prompt generation")
    parser.add_argument("--fp16-attn", action="store_true",
                        help="D2_fp16_attn_all: keep F-layer attention Q/K/V/O in FP16 (better quality, +1.2%% latency)")
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    chunk_indices = list(range(NUM_CHUNKS))
    if args.chunks:
        chunk_indices = [int(x) for x in args.chunks.split(",")]

    print("=" * 70)
    print("  ALL-CHUNKS V4 PRECISION POLICY — Qwen3.5-4B")
    policy = "V4 + D2_fp16_attn_all (F-layer attn Q/K/V/O → FP16)" if args.fp16_attn else \
              "V4 (F-layer kv_cache_state → FP32, rest FP16)"
    print(f"  Policy:     {policy}")
    print(f"  Artifacts:  {ARTIFACT_DIR}")
    print(f"  V2 source:  {V2_ARTIFACT_DIR}")
    print(f"  FP32 ref:   {FP32_MODEL_DIR}")
    print(f"  Chunks:     {chunk_indices}")
    print(f"  Config:     CTX={CTX} BATCH={BATCH_SIZE} NUM_CHUNKS={NUM_CHUNKS}")
    print("=" * 70)

    # ═══════════════════════════
    #  PHASE 1 — Export
    # ═══════════════════════════

    export_results = {}

    if not args.skip_export:
        print("\n── Phase 1: Export all chunks with V4 policy ──")
        model = load_model()

        for ci in chunk_indices:
            sl, el = CHUNK_RANGES[ci]
            fp32_layers, fp16_layers = get_chunk_fl(ci)

            kind = "all-L" if not fp32_layers else \
                   "all-F" if not fp16_layers else \
                   f"F={fp32_layers} L={fp16_layers}"

            print(f"\n  chunk {ci} (layers {sl}-{el-1}): {kind}")
            chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")
            os.makedirs(chunk_dir, exist_ok=True)

            result = export_chunk(model, ci, chunk_dir,
                                  skip_existing=args.skip_existing,
                                  fp16_attn=args.fp16_attn)
            export_results[ci] = result

        del model
        gc.collect()

        # Save export summary
        summary = {}
        for ci, r in export_results.items():
            summary[str(ci)] = {k: {kk: vv for kk, vv in v.items() if kk != "path"}
                                for k, v in r.items()}
        with open(os.path.join(ARTIFACT_DIR, "export_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    else:
        print("\n── Phase 1: Export SKIPPED ──")
        # Verify chunks exist
        for ci in chunk_indices:
            chunk_dir = os.path.join(ARTIFACT_DIR, f"chunk_{ci}")
            dec = os.path.exists(os.path.join(chunk_dir, "decode.mlpackage"))
            pf = os.path.exists(os.path.join(chunk_dir, "prefill.mlpackage"))
            if not dec or not pf:
                print(f"  WARNING: chunk {ci} missing: decode={dec} prefill={pf}")

    # ═══════════════════════════
    #  PHASE 2 — Assemble + Combine
    # ═══════════════════════════

    assembled_dir = os.path.join(ARTIFACT_DIR, "assembled")
    print("\n── Phase 2: Assemble + Combine ──")

    print("  Assembling pipeline...")
    assemble_pipeline(assembled_dir)

    print("  Combining chunks (dedup)...")
    combined_dir = combine_all(assembled_dir, skip_existing=args.skip_existing)

    # ── Cast analysis ──
    print("\n  Cast analysis (combined models):")
    total_casts = {"fp16": 0, "fp32": 0, "total": 0}
    per_chunk_casts = {}
    for ci in range(NUM_CHUNKS):
        pkg = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        infer_c = count_combined_casts(pkg, "infer")
        prefill_c = count_combined_casts(pkg, "prefill")
        per_chunk_casts[ci] = {"infer": infer_c, "prefill": prefill_c}
        total_casts["fp16"] += infer_c["fp16"] + prefill_c["fp16"]
        total_casts["fp32"] += infer_c["fp32"] + prefill_c["fp32"]
        total_casts["total"] += infer_c["total"] + prefill_c["total"]
        print(f"    chunk {ci}: infer={infer_c['total']} (fp16={infer_c['fp16']}, "
              f"fp32={infer_c['fp32']})  prefill={prefill_c['total']} "
              f"(fp16={prefill_c['fp16']}, fp32={prefill_c['fp32']})")

    print(f"\n    TOTAL: {total_casts['total']} casts "
          f"(fp16={total_casts['fp16']}, fp32={total_casts['fp32']})")

    # Compare with FP32 reference
    fp32_total = {"fp16": 0, "fp32": 0, "total": 0}
    fp32_combined = os.path.join(FP32_MODEL_DIR, "combined_LUT4_dedup")
    for ci in range(NUM_CHUNKS):
        pkg = os.path.join(fp32_combined, f"chunk{ci}.mlpackage")
        if os.path.exists(pkg):
            c = count_combined_casts(pkg, "infer")
            fp32_total["fp16"] += c["fp16"]
            fp32_total["fp32"] += c["fp32"]
            fp32_total["total"] += c["total"]

    print(f"    FP32 ref total: {fp32_total['total']} casts "
          f"(fp16={fp32_total['fp16']}, fp32={fp32_total['fp32']})")

    cast_data = {
        "v4": total_casts,
        "fp32_ref": fp32_total,
        "per_chunk": {str(k): v for k, v in per_chunk_casts.items()},
    }
    with open(os.path.join(ARTIFACT_DIR, "cast_analysis.json"), "w") as f:
        json.dump(cast_data, f, indent=2)

    if args.skip_validate:
        print("\n── Validation SKIPPED ──")
        print("Done.")
        return

    # ═══════════════════════════
    #  PHASE 3 — Standard validation
    # ═══════════════════════════

    print("\n── Phase 3: Standard 3-turn validation ──")
    std_results = validate_standard(assembled_dir, max_tokens=args.tokens)

    with open(os.path.join(ARTIFACT_DIR, "standard_validation.json"), "w") as f:
        # Convert tokens to plain lists for JSON
        serializable = {
            "fresh_vs_inc_pass": std_results["fresh_vs_inc_pass"],
            "turns": std_results["turns"],
        }
        for mode in ["fresh", "incremental"]:
            serializable[mode] = []
            for r in std_results[mode]:
                serializable[mode].append({
                    "turn": r["turn"],
                    "prompt_len": r["prompt_len"],
                    "tokens": r["tokens"],
                    "text": r["text"],
                    "end_pos": r["end_pos"],
                })
        json.dump(serializable, f, indent=2)

    # ═══════════════════════════
    #  PHASE 4 — Custom prompts + FP32 comparison
    # ═══════════════════════════

    print("\n── Phase 4: Custom prompts + FP32 comparison ──")
    custom_results = validate_custom(assembled_dir, max_tokens=args.custom_tokens)

    with open(os.path.join(ARTIFACT_DIR, "custom_validation.json"), "w") as f:
        # Serialize
        out = {"prompts": custom_results["prompts"], "comparison": []}
        for c in custom_results["comparison"]:
            out["comparison"].append({
                "prompt": c["prompt"],
                "v4_text": c["v4_text"],
                "fp32_text": c["fp32_text"],
                "token_matches": c["token_matches"],
                "token_total": c["token_total"],
                "token_pct": c["token_pct"],
                "v4_len": c["v4_len"],
                "fp32_len": c["fp32_len"],
                "v4_repetition": c["v4_repetition"],
                "fp32_repetition": c["fp32_repetition"],
            })
        json.dump(out, f, indent=2)

    # ═══════════════════════════
    #  PHASE 5 — Final Report
    # ═══════════════════════════

    print("\n" + "=" * 70)
    print("  FINAL REPORT — V4 Model-Wide Precision Policy")
    print("=" * 70)

    # 1. Deployment result
    print("\n  1. DEPLOYMENT RESULT")
    print(f"     Chunks modified: all {NUM_CHUNKS} (0-{NUM_CHUNKS-1})")
    all_exported = all(
        os.path.exists(os.path.join(ARTIFACT_DIR, f"chunk_{ci}", "decode.mlpackage"))
        and os.path.exists(os.path.join(ARTIFACT_DIR, f"chunk_{ci}", "prefill.mlpackage"))
        for ci in range(NUM_CHUNKS)
    )
    print(f"     All chunks exported: {'YES' if all_exported else 'NO'}")
    all_combined = all(
        os.path.exists(os.path.join(combined_dir, f"chunk{ci}.mlpackage"))
        for ci in range(NUM_CHUNKS)
    )
    print(f"     Combined model built: {'YES' if all_combined else 'NO'}")
    print(f"     Prefill + decode: {'WORKING' if std_results['fresh_vs_inc_pass'] else 'ISSUE'}")

    # 2. Correctness result
    print("\n  2. CORRECTNESS RESULT")
    print(f"     Fresh vs incremental: {'PASS (100%)' if std_results['fresh_vs_inc_pass'] else 'FAIL'}")

    custom_comp = custom_results["comparison"]
    any_repetition = any(c["v4_repetition"] for c in custom_comp)
    avg_match = np.mean([c["token_pct"] for c in custom_comp])

    for c in custom_comp:
        status = "MATCH" if c["token_pct"] == 100 else f"DRIFT {c['token_pct']:.0f}%"
        rep = " [REPETITION]" if c["v4_repetition"] else ""
        print(f"     {c['prompt'][:45]:45s} {status}{rep}")
        print(f"       V4:   {c['v4_text'][:120]}")
        print(f"       FP32: {c['fp32_text'][:120]}")

    print(f"\n     Average token match vs FP32: {avg_match:.1f}%")
    print(f"     Repetition detected: {'YES' if any_repetition else 'NO'}")

    # 3. Performance result
    print("\n  3. PERFORMANCE RESULT")
    print(f"     V4  total casts: {total_casts['total']} "
          f"(fp16={total_casts['fp16']}, fp32={total_casts['fp32']})")
    print(f"     FP32 total casts: {fp32_total['total']} "
          f"(fp16={fp32_total['fp16']}, fp32={fp32_total['fp32']})")
    if fp32_total['total'] > 0:
        reduction = 100 * (1 - total_casts['total'] / fp32_total['total'])
        print(f"     Cast reduction vs FP32: {reduction:.1f}%")
    print(f"     ANE usage: preserved (CPU_AND_NE compute unit)")

    # 4. Final recommendation
    safe = (std_results["fresh_vs_inc_pass"]
            and avg_match >= 95
            and not any_repetition)

    print("\n  4. FINAL RECOMMENDATION")
    if safe:
        print("     ✓ V4 precision policy is SAFE to deploy model-wide.")
        print("     ✓ All chunks pass fresh/incremental consistency.")
        print(f"     ✓ Average {avg_match:.0f}% token match vs FP32 reference.")
        print("     ✓ No repetition detected.")
    else:
        print("     ✗ V4 precision policy shows issues:")
        if not std_results["fresh_vs_inc_pass"]:
            print("       - Fresh vs incremental mismatch")
        if avg_match < 95:
            print(f"       - Token match vs FP32 only {avg_match:.0f}%")
        if any_repetition:
            print("       - Repetition detected")

    print("\nDone.")

    # Save full report data
    report = {
        "deployment": {
            "chunks_modified": list(range(NUM_CHUNKS)),
            "all_exported": all_exported,
            "all_combined": all_combined,
        },
        "correctness": {
            "fresh_vs_inc_pass": std_results["fresh_vs_inc_pass"],
            "avg_match_vs_fp32": avg_match,
            "any_repetition": any_repetition,
        },
        "performance": {
            "v4_casts": total_casts,
            "fp32_casts": fp32_total,
        },
        "recommendation": "SAFE" if safe else "INVESTIGATE",
    }
    with open(os.path.join(ARTIFACT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
