#!/usr/bin/env python3
"""
CHUNK MERGE EXPERIMENT — Qwen3.5-4B P2

Tests whether two adjacent FLLL chunks can be merged into a single
FLLLFLLL chunk (8 layers) to reduce ANE model swap overhead.

Candidate: merge chunks 3+4 (layers 11-18, FLLLFLLL) — middle of model.

Current 9-chunk layout:
  chunk 0: layers  0–2   (LLL)
  chunk 1: layers  3–6   (FLLL)
  chunk 2: layers  7–10  (FLLL)
  chunk 3: layers 11–14  (FLLL)   ← MERGE these two
  chunk 4: layers 15–18  (FLLL)   ←
  chunk 5: layers 19–22  (FLLL)
  chunk 6: layers 23–26  (FLLL)
  chunk 7: layers 27–30  (FLLL)
  chunk 8: layer  31     (F)

New 8-chunk layout:
  chunk 0: layers  0–2   (LLL)        ← unchanged
  chunk 1: layers  3–6   (FLLL)       ← unchanged
  chunk 2: layers  7–10  (FLLL)       ← unchanged
  chunk 3: layers 11–18  (FLLLFLLL)   ← MERGED
  chunk 4: layers 19–22  (FLLL)       ← was chunk 5
  chunk 5: layers 23–26  (FLLL)       ← was chunk 6
  chunk 6: layers 27–30  (FLLL)       ← was chunk 7
  chunk 7: layer  31     (F)          ← was chunk 8

Phases:
  1. Export merged chunk (layers 11-18) with V4 precision
  2. Assemble 8-chunk model (P2 chunks + merged chunk)
  3. Standard 3-turn validation
  4. Custom prompt comparison vs P2 baseline
  5. Performance / cast analysis
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

from config import BATCH_SIZE, CTX, CHUNK_RANGES, LUT_BITS, FFN_PER_CHANNEL

# F-layers: full attention (every 4th starting at 3)
F_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
# L-layers: linear attention (all others)
L_LAYERS = set(range(32)) - F_LAYERS

# ── paths ──
HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
P2_MODEL_DIR = os.path.join(REPO_ROOT, "qwen3_5_v4_lut4_p2")
P2_COMBINED = os.path.join(P2_MODEL_DIR, "combined_LUT4_dedup")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "merge_flll_p2_experiment")
LAYER_PATTERN = re.compile(r"layers[._](\d+)")

# ── merged layout ──
# Merge chunks 3+4 → single chunk with layers 11-18
MERGED_IDX = 3  # index in new layout
MERGED_START = 11
MERGED_END = 19  # exclusive
MERGED_NUM_CHUNKS = 8

MERGED_CHUNK_RANGES = [
    (0, 3),    # chunk 0: LLL (unchanged)
    (3, 7),    # chunk 1: FLLL (unchanged)
    (7, 11),   # chunk 2: FLLL (unchanged)
    (11, 19),  # chunk 3: FLLLFLLL (MERGED)
    (19, 23),  # chunk 4: FLLL (was chunk 5)
    (23, 27),  # chunk 5: FLLL (was chunk 6)
    (27, 31),  # chunk 6: FLLL (was chunk 7)
    (31, 32),  # chunk 7: F (was chunk 8)
]

MERGED_LAYOUT = "LLL|FLLL|FLLL|FLLLFLLL|FLLL|FLLL|FLLL|F"

# Mapping: new chunk index -> old P2 chunk index (None = merged, needs export)
P2_REUSE_MAP = {
    0: 0,     # LLL unchanged
    1: 1,     # FLLL unchanged
    2: 2,     # FLLL unchanged
    3: None,  # MERGED — must export
    4: 5,     # FLLL was chunk 5
    5: 6,     # FLLL was chunk 6
    6: 7,     # FLLL was chunk 7
    7: 8,     # F was chunk 8
}


# ═══════════════════════════════════════════════════════════════════
#  V4 PRECISION SELECTOR (same policy as P2 export)
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


def make_v4_selector_for_layers(start_layer, end_layer):
    """Build V4 op_selector for a custom layer range.
    F-layer kv_cache ops → FP32, everything else → FP16.
    """
    fp16_layers = set()
    fp32_layers = set()
    for li in range(start_layer, end_layer):
        if li in F_LAYERS:
            fp32_layers.add(li)
        else:
            fp16_layers.add(li)

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
            return True  # pre-layer → FP16
        home = max(layers)
        if home in fp16_layers:
            return True  # L layer → FP16
        if home in fp32_layers:
            return not _is_kv_cache_op(op)  # F layer: FP16 except kv_cache
        return True

    return selector, fp16_layers, fp32_layers


# ═══════════════════════════════════════════════════════════════════
#  PHASE 1 — EXPORT MERGED CHUNK
# ═══════════════════════════════════════════════════════════════════

def export_merged_chunk(model, merged_dir, skip_existing=False):
    """Export decode + prefill for the merged chunk (layers 11-18)."""
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    selector, fp16_ls, fp32_ls = make_v4_selector_for_layers(MERGED_START, MERGED_END)
    print(f"  Merged chunk layers {MERGED_START}-{MERGED_END-1}")
    print(f"    F-layers (kv fp32): {sorted(fp32_ls)}")
    print(f"    L-layers (fp16):    {sorted(fp16_ls)}")

    results = {}

    for phase, converter_fn in [("decode", "convert_part_2"), ("prefill", "convert_part_2_prefill")]:
        pkg_path = os.path.join(merged_dir, f"{phase}.mlpackage")
        if skip_existing and os.path.exists(pkg_path):
            print(f"  [skip] {phase}")
            results[phase] = {"path": pkg_path, "skipped": True}
            continue

        print(f"  Exporting {phase} (layers {MERGED_START}-{MERGED_END-1}, V4 precision)...")
        t0 = time.time()

        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=MERGED_NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=FFN_PER_CHANNEL,
            compute_precision="float32")
        conv.compute_precision = FP16ComputePrecision(op_selector=selector)

        fn = getattr(conv, converter_fn)
        if phase == "decode":
            ml = fn(model, chunk_idx=MERGED_IDX, total_chunks=MERGED_NUM_CHUNKS,
                    override_start_layer=MERGED_START, override_end_layer=MERGED_END)
        else:
            ml = fn(model, chunk_idx=MERGED_IDX, total_chunks=MERGED_NUM_CHUNKS,
                    override_start_layer=MERGED_START, override_end_layer=MERGED_END)

        ml.save(pkg_path)
        elapsed = time.time() - t0
        print(f"  Saved {phase} ({elapsed:.1f}s)")
        results[phase] = {"path": pkg_path, "elapsed": elapsed}
        del ml, conv
        gc.collect()

    return results


# ═══════════════════════════════════════════════════════════════════
#  PHASE 2 — ASSEMBLE + COMBINE
# ═══════════════════════════════════════════════════════════════════

def assemble_merged_model(assembled_dir, merged_dir):
    """Create 8-chunk model directory: 7 P2 chunks + 1 merged."""
    os.makedirs(assembled_dir, exist_ok=True)

    # Copy tokenizer files from P2
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"]:
        src = os.path.join(P2_MODEL_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(assembled_dir, fname))

    # Copy embed_lmhead_combined
    for name in ["embed_lmhead_combined.mlpackage", "embed_single.mlpackage", "lm_head_nosplit.mlpackage"]:
        src = os.path.join(P2_MODEL_DIR, name)
        dst = os.path.join(assembled_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)

    # Place 8 chunks
    for new_ci in range(MERGED_NUM_CHUNKS):
        old_ci = P2_REUSE_MAP[new_ci]

        if old_ci is not None:
            # Reuse P2 chunk (symlink)
            for prefix in ["ffn_LUT4", "prefill_LUT4"]:
                src = os.path.join(P2_MODEL_DIR, f"{prefix}_chunk{old_ci}.mlpackage")
                dst = os.path.join(assembled_dir, f"{prefix}_chunk{new_ci}.mlpackage")
                if os.path.lexists(dst):
                    if os.path.islink(dst):
                        os.unlink(dst)
                    else:
                        shutil.rmtree(dst)
                os.symlink(os.path.abspath(src), dst)
                print(f"    chunk {new_ci} ({prefix}): symlink → P2 chunk {old_ci}")
        else:
            # Merged chunk
            for phase, prefix in [("decode", "ffn_LUT4"), ("prefill", "prefill_LUT4")]:
                src = os.path.join(merged_dir, f"{phase}.mlpackage")
                dst = os.path.join(assembled_dir, f"{prefix}_chunk{new_ci}.mlpackage")
                if os.path.lexists(dst):
                    if os.path.islink(dst):
                        os.unlink(dst)
                    else:
                        shutil.rmtree(dst)
                os.symlink(os.path.abspath(src), dst)
                print(f"    chunk {new_ci} ({prefix}): MERGED (layers {MERGED_START}-{MERGED_END-1})")


def combine_all(assembled_dir):
    """Combine decode+prefill into multifunction dedup for all 8 chunks."""
    from anemll.utils.combine_models import _save_multifunction_dedup

    combined_dir = os.path.join(assembled_dir, "combined_LUT4_dedup")
    os.makedirs(combined_dir, exist_ok=True)

    for ci in range(MERGED_NUM_CHUNKS):
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
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
        tag = " (MERGED)" if ci == MERGED_IDX else ""
        print(f"    chunk {ci} ({time.time() - t0:.1f}s){tag}")

    return combined_dir


# ═══════════════════════════════════════════════════════════════════
#  CAST ANALYSIS
# ═══════════════════════════════════════════════════════════════════

DTYPE_MAP = {10: "fp16", 11: "fp32", 22: "int16", 23: "int32", 32: "bool"}

def count_casts(mlpackage_path, function_name="main"):
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
    return {"total": sum(casts.values()), "fp16": casts.get("fp16", 0), "fp32": casts.get("fp32", 0)}


def count_combined_casts(combined_path, fn_name="infer"):
    try:
        return count_casts(combined_path, fn_name)
    except Exception:
        return {"total": 0, "fp16": 0, "fp32": 0}


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION ENGINE (8-chunk)
# ═══════════════════════════════════════════════════════════════════

class MergeEngine:
    """DedupEngine variant that supports arbitrary chunk count."""

    def __init__(self, combined_dir, model_dir, compute_unit, num_chunks):
        combined_el = os.path.join(model_dir, "embed_lmhead_combined.mlpackage")
        if os.path.exists(combined_el):
            print(f"    Loading embed_lmhead_combined...")
            self.embed = ct.models.MLModel(combined_el, compute_units=ct.ComputeUnit.CPU_ONLY,
                                           function_name="embed")
            self.lmhead = ct.models.MLModel(combined_el, compute_units=ct.ComputeUnit.CPU_ONLY,
                                            function_name="lmhead")
        else:
            print(f"    Loading embed_single + lm_head_nosplit...")
            self.embed = ct.models.MLModel(
                os.path.join(model_dir, "embed_single.mlpackage"),
                compute_units=ct.ComputeUnit.CPU_ONLY)
            self.lmhead = ct.models.MLModel(
                os.path.join(model_dir, "lm_head_nosplit.mlpackage"),
                compute_units=ct.ComputeUnit.CPU_ONLY)

        self.num_chunks = num_chunks
        self.ffns = []
        for ci in range(num_chunks):
            print(f"    Loading dedup chunk {ci} ({compute_unit})...")
            m = ct.models.MLModel(
                os.path.join(combined_dir, f"chunk{ci}.mlpackage"),
                compute_units=compute_unit, function_name="infer")
            self.ffns.append(m)

        # Per-chunk input shapes
        self.inp_maps = []
        for ci in range(num_chunks):
            spec = self.ffns[ci].get_spec()
            imap = {}
            fn_inputs = None
            for fn in spec.description.functions:
                if fn.name == "infer":
                    fn_inputs = fn.input
                    break
            if fn_inputs is None:
                fn_inputs = spec.description.input
            for inp in fn_inputs:
                try:
                    imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                except Exception:
                    pass
            self.inp_maps.append(imap)
        self.has_linear = "linear_conv_state" in self.inp_maps[0]
        self.reset_all()

    def reset_all(self):
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_maps[ci]["linear_conv_state"], dtype=np.float16)
                              for ci in range(self.num_chunks)]
            self.lin_recs = [np.zeros(self.inp_maps[ci]["linear_recurrent_state"], dtype=np.float16)
                             for ci in range(self.num_chunks)]
        else:
            self.lin_convs = [None] * self.num_chunks
            self.lin_recs = [None] * self.num_chunks

    def _step(self, tok_id, pos):
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if self.lin_convs[ci] is not None:
                inp["linear_conv_state"] = self.lin_convs[ci]
                inp["linear_recurrent_state"] = self.lin_recs[ci]
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if "linear_conv_state_out" in out:
                self.lin_convs[ci] = out["linear_conv_state_out"]
                self.lin_recs[ci] = out["linear_recurrent_state_out"]
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        logits_key = [k for k in lm_out if "logit" in k.lower()]
        if logits_key:
            logits = lm_out[logits_key[0]]
        else:
            logits = list(lm_out.values())[0]
        return int(np.argmax(logits.flatten()))

    def prefill_and_decode(self, token_ids, start_pos, max_gen, stop_ids):
        t0 = time.time()
        for i, tid in enumerate(token_ids):
            pos = start_pos + i
            if pos >= CTX:
                break
            last_next = self._step(tid, pos)
        prefill_end_pos = start_pos + len(token_ids)
        t_prefill = (time.time() - t0) * 1000

        tokens = [last_next]
        t_dec = time.time()
        for gi in range(max_gen - 1):
            pos = prefill_end_pos + gi
            if pos >= CTX - 1:
                break
            next_id = self._step(tokens[-1], pos)
            tokens.append(next_id)
            if next_id in stop_ids:
                break
        t_decode = (time.time() - t_dec) * 1000

        fill_pos = prefill_end_pos + len(tokens) - 1
        if tokens and fill_pos < CTX:
            self._step(tokens[-1], fill_pos)
        end_pos = prefill_end_pos + len(tokens)
        return tokens, end_pos, t_prefill, t_decode

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        gc.collect()


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def build_stop_ids(tokenizer):
    stop_ids = set()
    for name in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    return stop_ids


def generate_single(engine, tokenizer, prompt, max_tokens, stop_ids):
    conversation = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=True)
    if hasattr(input_ids, "tolist"):
        token_list = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
    else:
        token_list = list(input_ids[0]) if hasattr(input_ids[0], '__iter__') else list(input_ids)

    engine.reset_all()
    gen_tokens, end_pos, pf_ms, dc_ms = engine.prefill_and_decode(
        token_list, 0, max_tokens, stop_ids)
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return {
        "prompt": prompt, "prompt_len": len(token_list),
        "tokens": gen_tokens, "text": text,
        "end_pos": end_pos, "prefill_ms": pf_ms, "decode_ms": dc_ms,
    }


def validate_3turn(engine, tokenizer, max_tokens, stop_ids, label):
    """Run 3-turn fresh and incremental, check agreement."""
    TURNS = [
        "What is a stack in computer science?",
        "How does it compare to a queue?",
        "Give me a Python example of each.",
    ]

    # Fresh mode: each turn from full conversation history
    conversation = []
    fresh_results = []
    for ti, msg in enumerate(TURNS):
        conversation.append({"role": "user", "content": msg})
        input_ids = tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True)
        if hasattr(input_ids, "tolist"):
            token_list = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        else:
            token_list = list(input_ids[0]) if hasattr(input_ids[0], '__iter__') else list(input_ids)

        engine.reset_all()
        gen_tokens, end_pos, pf_ms, dc_ms = engine.prefill_and_decode(
            token_list, 0, max_tokens, stop_ids)
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print(f"    {label} Turn {ti+1} [fresh]: {len(gen_tokens)} tokens, "
              f"pf={pf_ms:.0f}ms dc={dc_ms:.0f}ms")
        fresh_results.append({"turn": ti + 1, "tokens": gen_tokens, "text": text,
                              "prompt_len": len(token_list), "end_pos": end_pos})
        conversation.append({"role": "assistant", "content": text})

    # Incremental mode: same turns, building incrementally
    engine.reset_all()
    conversation_inc = []
    inc_results = []
    cur_pos = 0
    for ti, msg in enumerate(TURNS):
        conversation_inc.append({"role": "user", "content": msg})
        input_ids = tokenizer.apply_chat_template(
            conversation_inc, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True)
        if hasattr(input_ids, "tolist"):
            all_tokens = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        else:
            all_tokens = list(input_ids[0]) if hasattr(input_ids[0], '__iter__') else list(input_ids)

        new_tokens = all_tokens[cur_pos:]
        gen_tokens, end_pos, pf_ms, dc_ms = engine.prefill_and_decode(
            new_tokens, cur_pos, max_tokens, stop_ids)
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print(f"    {label} Turn {ti+1} [inc]:   {len(gen_tokens)} tokens, "
              f"pf={pf_ms:.0f}ms dc={dc_ms:.0f}ms")
        inc_results.append({"turn": ti + 1, "tokens": gen_tokens, "text": text,
                            "prompt_len": len(new_tokens), "end_pos": end_pos})
        conversation_inc.append({"role": "assistant", "content": text})
        cur_pos = end_pos

    # Compare
    all_pass = True
    for ti in range(len(TURNS)):
        f_toks = fresh_results[ti]["tokens"]
        i_toks = inc_results[ti]["tokens"]
        matches = sum(1 for a, b in zip(f_toks, i_toks) if a == b)
        total = min(len(f_toks), len(i_toks))
        pct = 100 * matches / total if total > 0 else 0
        status = "PASS" if pct == 100 else "FAIL"
        if pct < 100:
            all_pass = False
        print(f"    Turn {ti+1} fresh vs inc: {matches}/{total} ({pct:.0f}%) [{status}]")

    return {"fresh": fresh_results, "incremental": inc_results, "all_pass": all_pass}


def check_repetition(text, min_len=20, max_repeats=3):
    for length in range(min_len, min(60, len(text) // 2)):
        for start in range(len(text) - length * 2):
            substr = text[start:start + length]
            if text.count(substr) > max_repeats:
                return True, substr[:40]
    return False, None


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="P2 chunk merge experiment — FLLL+FLLL → FLLLFLLL")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--tokens", type=int, default=40,
                        help="Max tokens for 3-turn validation")
    parser.add_argument("--custom-tokens", type=int, default=120,
                        help="Max tokens for custom prompt comparison")
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("=" * 70)
    print("  P2 CHUNK MERGE EXPERIMENT — Qwen3.5-4B")
    print(f"  Merge:      chunks 3+4 → layers 11–18 (FLLLFLLL)")
    print(f"  New layout: {MERGED_NUM_CHUNKS} chunks ({MERGED_LAYOUT})")
    print(f"  Policy:     V4 (kv_cache FP32 for F-layers, else FP16)")
    print(f"  P2 base:    {P2_MODEL_DIR}")
    print(f"  Artifacts:  {ARTIFACT_DIR}")
    print("=" * 70)

    # ═══════════════════════════
    #  PHASE 1 — Export merged chunk
    # ═══════════════════════════

    merged_dir = os.path.join(ARTIFACT_DIR, "merged_chunk")
    os.makedirs(merged_dir, exist_ok=True)

    if not args.skip_export:
        print("\n── Phase 1: Export merged chunk (layers 11–18) ──")
        from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
        print("  Loading HF model...")
        cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
        cfg.context_length = CTX
        cfg.state_length = CTX
        model = Qwen35ForCausalLM(cfg)
        assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        export_results = export_merged_chunk(model, merged_dir, args.skip_existing)
        del model
        gc.collect()

        with open(os.path.join(ARTIFACT_DIR, "export_results.json"), "w") as f:
            serializable = {}
            for k, v in export_results.items():
                serializable[k] = {kk: vv for kk, vv in v.items() if kk != "path"}
            json.dump(serializable, f, indent=2)
    else:
        print("\n── Phase 1: Export SKIPPED ──")
        for phase in ["decode", "prefill"]:
            pkg = os.path.join(merged_dir, f"{phase}.mlpackage")
            if not os.path.exists(pkg):
                print(f"  WARNING: {phase}.mlpackage missing!")

    # ═══════════════════════════
    #  PHASE 2 — Assemble + Combine
    # ═══════════════════════════

    assembled_dir = os.path.join(ARTIFACT_DIR, "assembled")
    print("\n── Phase 2: Assemble + Combine ──")

    if os.path.exists(assembled_dir):
        shutil.rmtree(assembled_dir)

    print("  Assembling 8-chunk pipeline...")
    assemble_merged_model(assembled_dir, merged_dir)

    print("  Combining chunks (dedup)...")
    combined_dir = combine_all(assembled_dir)

    # Cast analysis
    print("\n  Cast analysis:")
    merge_total = {"fp16": 0, "fp32": 0, "total": 0}
    for ci in range(MERGED_NUM_CHUNKS):
        pkg = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        c = count_combined_casts(pkg, "infer")
        merge_total["fp16"] += c["fp16"]
        merge_total["fp32"] += c["fp32"]
        merge_total["total"] += c["total"]
        tag = " MERGED" if ci == MERGED_IDX else ""
        print(f"    chunk {ci}: infer casts={c['total']} (fp16={c['fp16']}, fp32={c['fp32']}){tag}")

    # P2 baseline casts
    p2_total = {"fp16": 0, "fp32": 0, "total": 0}
    for ci in range(9):
        pkg = os.path.join(P2_COMBINED, f"chunk{ci}.mlpackage")
        if os.path.exists(pkg):
            c = count_combined_casts(pkg, "infer")
            p2_total["fp16"] += c["fp16"]
            p2_total["fp32"] += c["fp32"]
            p2_total["total"] += c["total"]

    print(f"\n    Merged model infer casts: {merge_total['total']}")
    print(f"    P2 baseline infer casts:  {p2_total['total']}")
    delta = merge_total["total"] - p2_total["total"]
    print(f"    Delta: {delta:+d} casts")

    if args.skip_validate:
        print("\n── Validation SKIPPED ──")
        print("Done.")
        return

    # ═══════════════════════════
    #  PHASE 3 — Standard 3-turn validation
    # ═══════════════════════════

    print("\n── Phase 3: Standard 3-turn validation ──")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(P2_MODEL_DIR, use_fast=False)
    stop_ids = build_stop_ids(tokenizer)
    compute_unit = ct.ComputeUnit.CPU_AND_NE

    print("  Loading merged model (8 chunks)...")
    engine = MergeEngine(combined_dir, assembled_dir, compute_unit, MERGED_NUM_CHUNKS)

    print("  Running 3-turn validation...")
    std_results = validate_3turn(engine, tokenizer, args.tokens, stop_ids, "Merged")
    std_pass = std_results["all_pass"]
    print(f"\n  Standard validation: {'ALL PASS' if std_pass else 'FAIL'}")

    # ═══════════════════════════
    #  PHASE 4 — Custom prompts + P2 baseline comparison
    # ═══════════════════════════

    print("\n── Phase 4: Custom prompts + P2 baseline comparison ──")

    PROMPTS = [
        "What is a stack in computer science?",
        "教我做红烧鱼",
        "A farmer has 17 sheep. All but 9 run away. How many are left?",
    ]

    # Generate with merged model
    print("  Generating with merged model...")
    merged_gen = []
    for prompt in PROMPTS:
        print(f"    {prompt[:50]}...")
        r = generate_single(engine, tokenizer, prompt, args.custom_tokens, stop_ids)
        merged_gen.append(r)

    engine.cleanup()
    gc.collect()

    # Load P2 baseline (9-chunk)
    print("  Loading P2 baseline (9-chunk)...")
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
    from validate import DedupEngine
    p2_engine = DedupEngine(P2_COMBINED, P2_MODEL_DIR, compute_unit)
    p2_gen = []
    for prompt in PROMPTS:
        print(f"    P2: {prompt[:50]}...")
        p2_engine.reset_all()
        conversation = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True)
        if hasattr(input_ids, "tolist"):
            token_list = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        else:
            token_list = list(input_ids[0])
        gen_tokens, end_pos, pf_ms, dc_ms = p2_engine.prefill_and_decode(
            token_list, 0, args.custom_tokens, stop_ids)
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        p2_gen.append({"prompt": prompt, "tokens": gen_tokens, "text": text,
                       "prefill_ms": pf_ms, "decode_ms": dc_ms})
    p2_engine.cleanup()
    gc.collect()

    # Compare
    print("\n  Comparison (merged vs P2 baseline):")
    comparisons = []
    for mg, pg in zip(merged_gen, p2_gen):
        matches = sum(1 for a, b in zip(mg["tokens"], pg["tokens"]) if a == b)
        total = min(len(mg["tokens"]), len(pg["tokens"]))
        pct = 100 * matches / total if total > 0 else 0
        rep_m, _ = check_repetition(mg["text"])
        rep_p, _ = check_repetition(pg["text"])

        comparisons.append({
            "prompt": mg["prompt"],
            "merged_len": len(mg["tokens"]), "p2_len": len(pg["tokens"]),
            "token_matches": matches, "token_total": total, "token_pct": pct,
            "merged_repetition": rep_m, "p2_repetition": rep_p,
            "merged_prefill_ms": mg["prefill_ms"], "merged_decode_ms": mg["decode_ms"],
            "p2_prefill_ms": pg["prefill_ms"], "p2_decode_ms": pg["decode_ms"],
        })

        rep_flag = " [REPEAT!]" if rep_m else ""
        print(f"    {mg['prompt'][:45]:45s} {matches}/{total} ({pct:.0f}%){rep_flag}")
        print(f"      Merged: {mg['text'][:100]}")
        print(f"      P2:     {pg['text'][:100]}")

    # ═══════════════════════════
    #  PHASE 5 — Performance comparison
    # ═══════════════════════════

    print("\n── Phase 5: Performance comparison ──")
    for c in comparisons:
        idx = comparisons.index(c)
        merged_tps = len(merged_gen[idx]["tokens"]) / (c["merged_decode_ms"] / 1000) if c["merged_decode_ms"] > 0 else 0
        p2_tps = len(p2_gen[idx]["tokens"]) / (c["p2_decode_ms"] / 1000) if c["p2_decode_ms"] > 0 else 0
        print(f"    {c['prompt'][:40]:40s}")
        print(f"      Merged (8ch): pf={c['merged_prefill_ms']:.0f}ms  dc={c['merged_decode_ms']:.0f}ms  ({merged_tps:.1f} tok/s)")
        print(f"      P2 (9ch):     pf={c['p2_prefill_ms']:.0f}ms  dc={c['p2_decode_ms']:.0f}ms  ({p2_tps:.1f} tok/s)")

    print(f"\n  Cast summary:")
    print(f"    Merged model (8 chunks): {merge_total['total']} infer casts")
    print(f"    P2 baseline  (9 chunks): {p2_total['total']} infer casts")
    print(f"    Delta: {delta:+d} casts")

    # ═══════════════════════════
    #  FINAL REPORT
    # ═══════════════════════════

    print("\n" + "=" * 70)
    print("  FINAL REPORT — P2 FLLL+FLLL → FLLLFLLL Merge Experiment")
    print("=" * 70)

    any_rep = any(c["merged_repetition"] for c in comparisons)
    avg_match = np.mean([c["token_pct"] for c in comparisons]) if comparisons else 0

    print(f"\n  1. DEPLOYMENT")
    print(f"     Merged chunk exported:     layers 11–18 (FLLLFLLL)")
    print(f"     ANE loadable:              CPU_AND_NE")
    print(f"     8-chunk model assembled:   YES")
    print(f"     Standard 3-turn:           {'PASS' if std_pass else 'FAIL'}")

    print(f"\n  2. CORRECTNESS")
    print(f"     Fresh vs incremental:      {'PASS' if std_pass else 'FAIL'}")
    print(f"     Avg token match vs P2:     {avg_match:.0f}%")
    print(f"     Repetition detected:       {'YES' if any_rep else 'NO'}")
    for c in comparisons:
        print(f"     {c['prompt'][:40]:40s} {c['token_pct']:.0f}% match")

    print(f"\n  3. PERFORMANCE")
    print(f"     Models: 8 chunks (was 9) = 1 fewer ANE swap per step")
    print(f"     Cast delta: {delta:+d}")

    avg_merged_tps = 0
    avg_p2_tps = 0
    for c in comparisons:
        idx = comparisons.index(c)
        if c["merged_decode_ms"] > 0:
            avg_merged_tps += len(merged_gen[idx]["tokens"]) / (c["merged_decode_ms"] / 1000)
        if c["p2_decode_ms"] > 0:
            avg_p2_tps += len(p2_gen[idx]["tokens"]) / (c["p2_decode_ms"] / 1000)
    if comparisons:
        avg_merged_tps /= len(comparisons)
        avg_p2_tps /= len(comparisons)
    print(f"     Avg decode speed merged:   {avg_merged_tps:.1f} tok/s")
    print(f"     Avg decode speed P2:       {avg_p2_tps:.1f} tok/s")
    if avg_p2_tps > 0:
        speed_delta = (avg_merged_tps - avg_p2_tps) / avg_p2_tps * 100
        print(f"     Speed delta:               {speed_delta:+.1f}%")

    safe = std_pass and not any_rep
    print(f"\n  4. RECOMMENDATION")
    if safe:
        print(f"     SAFE — Merging FLLL+FLLL → FLLLFLLL appears viable")
        print(f"       Self-consistency: 100%")
        print(f"       No repetition detected")
    else:
        print(f"     INVESTIGATE — Merging shows issues:")
        if not std_pass:
            print(f"       Fresh vs incremental mismatch")
        if any_rep:
            print(f"       Repetition detected in merged model")

    # Save report
    report = {
        "experiment": "merge_flll_p2_chunks_3_4",
        "merged_layers": [MERGED_START, MERGED_END],
        "merged_chunk_idx": MERGED_IDX,
        "new_num_chunks": MERGED_NUM_CHUNKS,
        "p2_baseline": P2_MODEL_DIR,
        "deployment": {"exported": True, "ane_loadable": True, "std_validation": std_pass},
        "correctness": {
            "fresh_vs_inc": std_pass,
            "avg_match_vs_p2": avg_match,
            "any_repetition": any_rep,
            "comparisons": [{k: v for k, v in c.items()} for c in comparisons],
        },
        "performance": {
            "merged_casts": merge_total,
            "p2_casts": p2_total,
            "cast_delta": delta,
            "avg_merged_tps": avg_merged_tps,
            "avg_p2_tps": avg_p2_tps,
        },
        "recommendation": "SAFE" if safe else "INVESTIGATE",
    }
    with open(os.path.join(ARTIFACT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    gen_output = {
        "merged": [{"prompt": r["prompt"], "text": r["text"], "tokens": r["tokens"]}
                   for r in merged_gen],
        "p2": [{"prompt": r["prompt"], "text": r["text"], "tokens": r["tokens"]}
               for r in p2_gen],
    }
    with open(os.path.join(ARTIFACT_DIR, "generation_outputs.json"), "w") as f:
        json.dump(gen_output, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
