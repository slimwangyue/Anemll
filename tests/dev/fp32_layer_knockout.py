#!/usr/bin/env python3
"""FP32 → FP16 layer knockout experiment for Qwen3.5-4B on ANE.

Start from full FP32 (known correct), selectively replace chunks/layers with
FP16, and observe when quality degrades. Goal: find the maximum FP16 coverage
that preserves FP32-level correctness (important because FP32 is 2x slower
than FP16 on iPhone).

Architecture: 32 layers = [LLL] + 7×[FLLL] + [F], 9 chunks
  chunk 0: layers 0-2   (LLL)
  chunk 1: layers 3-6   (FLLL)
  chunk 2: layers 7-10  (FLLL)
  chunk 3: layers 11-14 (FLLL)
  chunk 4: layers 15-18 (FLLL)
  chunk 5: layers 19-22 (FLLL)
  chunk 6: layers 23-26 (FLLL)
  chunk 7: layers 27-30 (FLLL)
  chunk 8: layers 31    (F)

Phase 1 — Chunk-level knockout:
  Baseline: all 9 chunks FP32 → reference output
  For each chunk i: FP16 for chunk i, FP32 for rest → compare

Phase 2 — Progressive FP16:
  Combine safe chunks into FP16, test combinations

Phase 3 — Per-layer knockout within sensitive chunks (re-export needed)

Usage:
    python tests/dev/fp32_layer_knockout.py \\
        --fp16-dir qwen3_5_stable_lut4ffn_lut6em_fp16 \\
        --fp32-dir qwen3_5_stable_lut4ffn_lut6em_fp32 \\
        --output tests/dev/knockout_results \\
        --max-tokens 40
"""
import argparse
import gc
import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import shutil

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import coremltools as ct
from transformers import AutoTokenizer

from config import CHUNK_RANGES, NUM_CHUNKS, CTX

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Architecture map
# ─────────────────────────────────────────────────────────────────────────────
LAYER_TYPES = {}
for ci in range(NUM_CHUNKS):
    sl, el = CHUNK_RANGES[ci]
    for li in range(sl, el):
        if ci == 0:
            LAYER_TYPES[li] = 'L'
        elif ci == 8:
            LAYER_TYPES[li] = 'F'
        else:
            LAYER_TYPES[li] = 'F' if li == sl else 'L'

CHUNK_LABELS = {}
for ci in range(NUM_CHUNKS):
    sl, el = CHUNK_RANGES[ci]
    types = ''.join(LAYER_TYPES[li] for li in range(sl, el))
    CHUNK_LABELS[ci] = f"chunk{ci}[{sl}-{el-1}]({types})"

PROMPTS = [
    "What is a stack in computer science?",
    "教我做红烧鱼",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
]

BNNS_CACHE_DIR = os.path.expanduser(
    "~/Library/Caches/com.apple.python3/com.apple.e5rt.e5bundlecache")


def clear_bnns_cache():
    """Clear BNNS bundle cache to prevent disk-full errors."""
    if os.path.isdir(BNNS_CACHE_DIR):
        for entry in os.listdir(BNNS_CACHE_DIR):
            path = os.path.join(BNNS_CACHE_DIR, entry)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError:
                pass
        print("    [Cleared BNNS cache]")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_stop_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)
    return stop_ids


def _get_full_logits(lm_out: dict) -> np.ndarray:
    if "logits" in lm_out:
        return lm_out["logits"].flatten().astype(np.float32)
    split_keys = sorted([k for k in lm_out if k.startswith("logits")],
                        key=lambda k: int(k.replace("logits", "")))
    if split_keys:
        return np.concatenate([lm_out[k].flatten() for k in split_keys]).astype(np.float32)
    raise KeyError(f"Cannot find logits in lm_head output: {list(lm_out.keys())}")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


def kl_divergence(ref_logits: np.ndarray, var_logits: np.ndarray) -> float:
    ref = ref_logits.flatten().astype(np.float64)
    var = var_logits.flatten().astype(np.float64)
    ref_shifted = ref - ref.max()
    var_shifted = var - var.max()
    ref_lse = np.log(np.sum(np.exp(ref_shifted))) + ref.max()
    var_lse = np.log(np.sum(np.exp(var_shifted))) + var.max()
    ref_log_probs = ref - ref_lse
    var_log_probs = var - var_lse
    ref_probs = np.exp(ref_log_probs)
    kl = np.sum(ref_probs * (ref_log_probs - var_log_probs))
    return max(0.0, float(kl))


def detect_repetition_onset(tokens: List[int], min_repeat_len: int = 3) -> int:
    if len(tokens) < min_repeat_len * 2:
        return -1
    for window in range(min_repeat_len, len(tokens) // 2 + 1):
        for start in range(len(tokens) - window * 2 + 1):
            if tokens[start:start + window] == tokens[start + window:start + window * 2]:
                return start + window
    return -1


def tokens_match_count(ref_tokens: List[int], var_tokens: List[int]) -> int:
    """Count how many consecutive tokens match from the start."""
    n = min(len(ref_tokens), len(var_tokens))
    for i in range(n):
        if ref_tokens[i] != var_tokens[i]:
            return i
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine with hot-swap support
# Keeps all chunks loaded, supports swapping individual chunks to avoid
# re-creating BNNS caches (which consume ~6GB disk per full load)
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    def __init__(self, chunk_paths: List[str], embed_lmhead_path: str,
                 function_name: str = "infer"):
        self.loadable = True
        self.function_name = function_name
        self.chunk_paths = list(chunk_paths)

        print(f"    Loading embed+lmhead...")
        self.embed = ct.models.MLModel(
            embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
            function_name="embed")
        self.lmhead = ct.models.MLModel(
            embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
            function_name="lmhead")

        self.ffns = []
        for ci, path in enumerate(chunk_paths):
            prec = "FP16" if "fp16" in path or "lut4ffn_lut6em_fp16" in path else "FP32"
            print(f"    Loading chunk {ci} [{prec}] ...")
            try:
                m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                      function_name=function_name)
                self.ffns.append(m)
            except Exception as e:
                print(f"    *** FAILED chunk {ci}: {e}")
                self.loadable = False
                return

        self._build_input_maps()
        self.reset_all()

    def _build_input_maps(self):
        self.inp_maps = []
        for ci in range(len(self.ffns)):
            spec = self.ffns[ci].get_spec()
            imap = {}
            for fn in spec.description.functions:
                if fn.name == self.function_name:
                    for inp in fn.input:
                        try:
                            imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                        except Exception:
                            pass
                    break
            if not imap:
                for inp in spec.description.input:
                    try:
                        imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except Exception:
                        pass
            self.inp_maps.append(imap)
        self.has_linear = 'linear_conv_state' in self.inp_maps[0] if self.inp_maps else False

    def swap_chunk(self, chunk_idx: int, new_path: str):
        """Hot-swap a single chunk without reloading the rest."""
        prec = "FP16" if "fp16" in new_path or "lut4ffn_lut6em_fp16" in new_path else "FP32"
        if self.chunk_paths[chunk_idx] == new_path:
            return  # Already loaded
        print(f"    Swapping chunk {chunk_idx} → [{prec}] ...")
        old_model = self.ffns[chunk_idx]
        try:
            new_model = ct.models.MLModel(new_path, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                          function_name=self.function_name)
            self.ffns[chunk_idx] = new_model
            self.chunk_paths[chunk_idx] = new_path
            # Rebuild input map for this chunk
            spec = new_model.get_spec()
            imap = {}
            for fn in spec.description.functions:
                if fn.name == self.function_name:
                    for inp in fn.input:
                        try:
                            imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                        except Exception:
                            pass
                    break
            self.inp_maps[chunk_idx] = imap
            del old_model
            gc.collect()
        except Exception as e:
            print(f"    *** Swap FAILED chunk {chunk_idx}: {e}")
            self.loadable = False

    def reset_all(self):
        if not self.loadable:
            return
        try:
            self.states = [m.make_state() for m in self.ffns]
        except Exception as e:
            print(f"    *** make_state failed: {e}")
            self.loadable = False
            return
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_maps[ci]['linear_conv_state'], dtype=np.float16)
                              for ci in range(len(self.ffns))]
            self.lin_recs = [np.zeros(self.inp_maps[ci]['linear_recurrent_state'], dtype=np.float16)
                             for ci in range(len(self.ffns))]
        else:
            self.lin_convs = [None] * len(self.ffns)
            self.lin_recs = [None] * len(self.ffns)

    def step(self, tok_id: int, pos: int) -> Tuple[int, np.ndarray, np.ndarray]:
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0

        for ci in range(len(self.ffns)):
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
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        logits = _get_full_logits(lm_out)
        next_tok = int(np.argmax(logits))
        return next_tok, hidden.copy(), logits

    def generate(self, token_ids: List[int], max_gen: int, stop_ids: set,
                 capture_traces: bool = False) -> dict:
        if not self.loadable:
            return {"gen_tokens": [], "tps": 0}
        self.reset_all()
        t0 = time.time()

        for i, tid in enumerate(token_ids):
            if i >= CTX:
                break
            last_tok, last_hidden, last_logits = self.step(tid, i)
        prefill_ms = (time.time() - t0) * 1000

        result = {
            "gen_tokens": [last_tok],
            "prefill_ms": prefill_ms,
        }
        if capture_traces:
            result["prefill_hidden"] = last_hidden.copy()
            result["prefill_logits"] = last_logits.copy()
            result["decode_hiddens"] = []
            result["decode_logits"] = []

        t_dec = time.time()
        prefill_end = len(token_ids)
        for gi in range(max_gen - 1):
            pos = prefill_end + gi
            if pos >= CTX - 1:
                break
            next_tok, hidden, logits = self.step(result["gen_tokens"][-1], pos)
            result["gen_tokens"].append(next_tok)
            if capture_traces:
                result["decode_hiddens"].append(hidden.copy())
                result["decode_logits"].append(logits.copy())
            if next_tok in stop_ids:
                break

        t_decode = time.time() - t_dec
        n_dec = max(1, len(result["gen_tokens"]) - 1)
        result["tps"] = n_dec / t_decode if t_decode > 0 else 0
        result["decode_ms"] = t_decode * 1000
        return result

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        self.ffns = []
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Build chunk path lists for different variants
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_path(base_dir: str, chunk_idx: int) -> str:
    return os.path.join(base_dir, "combined_LUT4_dedup", f"chunk{chunk_idx}.mlpackage")


def build_variant_paths(fp16_dir: str, fp32_dir: str,
                        fp16_chunks: Set[int]) -> List[str]:
    """Build list of 9 chunk paths, selecting FP16 or FP32 for each."""
    paths = []
    for ci in range(NUM_CHUNKS):
        if ci in fp16_chunks:
            paths.append(_chunk_path(fp16_dir, ci))
        else:
            paths.append(_chunk_path(fp32_dir, ci))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Compare two runs
# ─────────────────────────────────────────────────────────────────────────────

def compare_runs(ref: dict, var: dict) -> dict:
    """Compare variant run against reference (full FP32)."""
    ref_toks = ref["gen_tokens"]
    var_toks = var["gen_tokens"]

    match_count = tokens_match_count(ref_toks, var_toks)
    n_common = min(len(ref_toks), len(var_toks))
    total_match = sum(1 for a, b in zip(ref_toks, var_toks) if a == b)

    metrics = {
        "first_token_match": ref_toks[0] == var_toks[0] if ref_toks and var_toks else False,
        "consecutive_match": match_count,
        "total_match": total_match,
        "total_tokens": n_common,
        "match_rate": total_match / max(n_common, 1),
        "exact_match": ref_toks == var_toks,
    }

    # Early agreement (first 10 tokens)
    early_ref = ref_toks[:10]
    early_var = var_toks[:10]
    n_early = min(len(early_ref), len(early_var))
    early_match = sum(1 for a, b in zip(early_ref[:n_early], early_var[:n_early]) if a == b)
    metrics["early_agreement"] = early_match / max(n_early, 1)

    rep_onset = detect_repetition_onset(var_toks)
    metrics["repetition_onset"] = rep_onset

    # Trace-level metrics if available
    if "prefill_hidden" in ref and "prefill_hidden" in var:
        metrics["h_cos_pf"] = cosine_sim(ref["prefill_hidden"], var["prefill_hidden"])
        metrics["logit_cos_pf"] = cosine_sim(ref["prefill_logits"], var["prefill_logits"])
        metrics["kl_pf"] = kl_divergence(ref["prefill_logits"], var["prefill_logits"])

        if ref.get("decode_hiddens") and var.get("decode_hiddens"):
            n = min(len(ref["decode_hiddens"]), len(var["decode_hiddens"]))
            h_coss = [cosine_sim(ref["decode_hiddens"][i], var["decode_hiddens"][i]) for i in range(n)]
            metrics["h_cos_dec"] = float(np.mean(h_coss)) if h_coss else -1.0

        if ref.get("decode_logits") and var.get("decode_logits"):
            n = min(len(ref["decode_logits"]), len(var["decode_logits"]))
            kls = [kl_divergence(ref["decode_logits"][i], var["decode_logits"][i]) for i in range(n)]
            metrics["kl_dec"] = float(np.mean(kls)) if kls else -1.0

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    os.makedirs(args.output, exist_ok=True)

    tok_path = args.tokenizer or args.fp32_dir
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    stop_ids = _build_stop_ids(tokenizer)

    # Embed/lmhead from FP32 dir
    embed_lmhead = os.path.join(args.fp32_dir, "embed_lmhead_combined.mlpackage")
    assert os.path.exists(embed_lmhead), f"Missing: {embed_lmhead}"

    # Tokenize prompts
    prompt_tokens = {}
    for pi, prompt in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=True)
        prompt_tokens[pi] = ids[0].tolist() if hasattr(ids, 'tolist') else list(ids[0]) if hasattr(ids[0], 'tolist') else list(ids)
        print(f"  P{pi}: {prompt[:50]}... ({len(prompt_tokens[pi])} tokens)")

    # Results storage
    results_path = os.path.join(args.output, "knockout_results.json")
    all_results = {}

    # Load existing results for resume
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  Resumed: {len(all_results)} variants loaded")

    def save_results():
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ─── Phase 0: Full FP32 reference ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 0: Full FP32 Reference")
    print("=" * 70)

    fp32_refs = {}  # pi -> generation result
    variant_name = "full_fp32"

    clear_bnns_cache()

    # Load FP32 engine once — keep it for all phases
    fp32_paths = build_variant_paths(args.fp16_dir, args.fp32_dir, set())
    engine = InferenceEngine(fp32_paths, embed_lmhead)
    if not engine.loadable:
        print("  *** Cannot load full FP32. Aborting.")
        return

    if variant_name in all_results and len(all_results.get(variant_name, [])) == len(PROMPTS):
        print("  [SKIP] Already have FP32 reference (re-running for traces)")
    else:
        all_results[variant_name] = []

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  P{pi}: {prompt[:50]}...")
        result = engine.generate(prompt_tokens[pi], args.max_tokens, stop_ids,
                                 capture_traces=True)
        fp32_refs[pi] = result
        text = tokenizer.decode(result["gen_tokens"], skip_special_tokens=True)
        rep = detect_repetition_onset(result["gen_tokens"])

        if len(all_results[variant_name]) <= pi:
            all_results[variant_name].append({
                "prompt_idx": pi,
                "gen_tokens": result["gen_tokens"],
                "text": text[:300],
                "tps": result["tps"],
                "prefill_ms": result["prefill_ms"],
                "repetition_onset": rep,
            })
        print(f"    tps={result['tps']:.1f} rep={rep}")
        print(f"    Gen: {text[:120]}")

    save_results()

    # ─── Phase 1: Single-chunk knockout (using hot-swap) ─────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: Single-Chunk FP16 Knockout (hot-swap)")
    print("  (Swap one chunk to FP16, rest stay FP32)")
    print("=" * 70)

    chunk_safe = {}  # ci -> True/False (safe to make FP16)

    for ci in range(NUM_CHUNKS):
        variant_name = f"knock_chunk{ci}"
        sl, el = CHUNK_RANGES[ci]
        label = CHUNK_LABELS[ci]

        print(f"\n{'─' * 60}")
        print(f"  {variant_name}: {label} → FP16, rest FP32")
        print(f"{'─' * 60}")

        if variant_name in all_results and len(all_results.get(variant_name, [])) == len(PROMPTS):
            print(f"  [SKIP] Already completed")
            prev = all_results[variant_name]
            chunk_safe[ci] = all(r.get("exact_match", False) for r in prev)
            continue

        # Hot-swap: replace chunk ci with FP16 version
        fp16_path = _chunk_path(args.fp16_dir, ci)
        fp32_path = _chunk_path(args.fp32_dir, ci)
        engine.swap_chunk(ci, fp16_path)

        if not engine.loadable:
            print(f"  *** Cannot load. Marking unsafe.")
            chunk_safe[ci] = False
            all_results[variant_name] = [{"prompt_idx": pi, "loadable": False}
                                         for pi in range(len(PROMPTS))]
            # Swap back to FP32
            engine.loadable = True
            engine.swap_chunk(ci, fp32_path)
            save_results()
            continue

        all_results[variant_name] = []
        all_match = True

        for pi, prompt in enumerate(PROMPTS):
            print(f"\n    P{pi}: {prompt[:50]}...")
            result = engine.generate(prompt_tokens[pi], args.max_tokens, stop_ids,
                                     capture_traces=True)
            ref = fp32_refs[pi]
            metrics = compare_runs(ref, result)
            text = tokenizer.decode(result["gen_tokens"], skip_special_tokens=True)

            if not metrics["exact_match"]:
                all_match = False

            entry = {
                "prompt_idx": pi,
                "gen_tokens": result["gen_tokens"],
                "text": text[:300],
                "tps": result["tps"],
                "prefill_ms": result["prefill_ms"],
                **metrics,
            }
            all_results[variant_name].append(entry)

            match_sym = "✓" if metrics["exact_match"] else "✗"
            print(f"      {match_sym} 1st={metrics['first_token_match']} "
                  f"consec={metrics['consecutive_match']}/{len(ref['gen_tokens'])} "
                  f"match={metrics['match_rate']:.0%} "
                  f"tps={result['tps']:.1f}")
            if "h_cos_pf" in metrics:
                print(f"      h_cos_pf={metrics['h_cos_pf']:.6f} "
                      f"h_cos_dec={metrics.get('h_cos_dec', -1):.6f} "
                      f"kl_dec={metrics.get('kl_dec', -1):.4f}")
            if not metrics["exact_match"]:
                ref_text = tokenizer.decode(ref["gen_tokens"], skip_special_tokens=True)
                print(f"      REF: {ref_text[:100]}")
                print(f"      VAR: {text[:100]}")

        # Swap back to FP32 for next iteration
        engine.swap_chunk(ci, fp32_path)
        chunk_safe[ci] = all_match
        save_results()

        status = "SAFE ✓" if all_match else "SENSITIVE ✗"
        print(f"\n  → {label}: {status}")

    # Print Phase 1 summary
    print("\n" + "=" * 70)
    print("  PHASE 1 SUMMARY: Chunk-Level Sensitivity")
    print("=" * 70)
    safe_chunks = set()
    sensitive_chunks = set()
    for ci in range(NUM_CHUNKS):
        status = "SAFE ✓" if chunk_safe.get(ci, False) else "SENSITIVE ✗"
        print(f"  {CHUNK_LABELS[ci]:40s} {status}")
        if chunk_safe.get(ci, False):
            safe_chunks.add(ci)
        else:
            sensitive_chunks.add(ci)

    print(f"\n  Safe chunks: {sorted(safe_chunks)} ({len(safe_chunks)}/9)")
    print(f"  Sensitive chunks: {sorted(sensitive_chunks)} ({len(sensitive_chunks)}/9)")

    if not safe_chunks:
        print("\n  *** No safe chunks found. Cannot optimize further.")
        save_results()
        return

    # ─── Phase 2: Progressive FP16 combinations (hot-swap) ────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: Progressive FP16 Combinations (hot-swap)")
    print("  (Combining safe chunks to maximize FP16 coverage)")
    print("=" * 70)

    def _run_multi_chunk_fp16(variant_name: str, fp16_set: set):
        """Swap multiple chunks to FP16, run all prompts, swap back."""
        if variant_name in all_results and len(all_results.get(variant_name, [])) == len(PROMPTS):
            print("  [SKIP] Already completed")
            return

        # Swap chunks to FP16
        for ci in sorted(fp16_set):
            engine.swap_chunk(ci, _chunk_path(args.fp16_dir, ci))
            if not engine.loadable:
                print(f"  *** Swap failed for chunk {ci}. Aborting variant.")
                # Swap back what we changed
                engine.loadable = True
                for ci2 in sorted(fp16_set):
                    engine.swap_chunk(ci2, _chunk_path(args.fp32_dir, ci2))
                all_results[variant_name] = [{"prompt_idx": pi, "loadable": False}
                                             for pi in range(len(PROMPTS))]
                save_results()
                return

        all_results[variant_name] = []
        all_match = True
        for pi, prompt in enumerate(PROMPTS):
            print(f"\n    P{pi}: {prompt[:50]}...")
            result = engine.generate(prompt_tokens[pi], args.max_tokens, stop_ids,
                                     capture_traces=True)
            ref = fp32_refs[pi]
            metrics = compare_runs(ref, result)
            text = tokenizer.decode(result["gen_tokens"], skip_special_tokens=True)
            if not metrics["exact_match"]:
                all_match = False

            all_results[variant_name].append({
                "prompt_idx": pi,
                "gen_tokens": result["gen_tokens"],
                "text": text[:300],
                "tps": result["tps"],
                "prefill_ms": result["prefill_ms"],
                **metrics,
            })

            match_sym = "✓" if metrics["exact_match"] else "✗"
            print(f"      {match_sym} consec={metrics['consecutive_match']}/{len(ref['gen_tokens'])} "
                  f"match={metrics['match_rate']:.0%} tps={result['tps']:.1f}")
            if not metrics["exact_match"]:
                ref_text = tokenizer.decode(ref["gen_tokens"], skip_special_tokens=True)
                print(f"      REF: {ref_text[:100]}")
                print(f"      VAR: {text[:100]}")

        # Swap all back to FP32
        for ci in sorted(fp16_set):
            engine.swap_chunk(ci, _chunk_path(args.fp32_dir, ci))
        save_results()
        return all_match

    # First: test ALL safe chunks together
    if len(safe_chunks) > 1:
        variant_name = "all_safe_fp16"
        safe_label = '+'.join(str(c) for c in sorted(safe_chunks))
        print(f"\n{'─' * 60}")
        print(f"  {variant_name}: chunks [{safe_label}] → FP16")
        print(f"{'─' * 60}")
        result = _run_multi_chunk_fp16(variant_name, safe_chunks)
        if result is not None:
            status = "EXACT MATCH ✓" if result else "DIVERGED ✗"
            print(f"\n  → All safe chunks combined: {status}")

    # Also test: all safe + try adding each sensitive one
    for si in sorted(sensitive_chunks):
        test_set = safe_chunks | {si}
        variant_name = f"safe_plus_chunk{si}"
        label = '+'.join(str(c) for c in sorted(test_set))
        print(f"\n{'─' * 60}")
        print(f"  {variant_name}: chunks [{label}] → FP16")
        print(f"{'─' * 60}")
        result = _run_multi_chunk_fp16(variant_name, test_set)
        if result is not None:
            status = "MATCH ✓" if result else "DIVERGED ✗"
            print(f"\n  → safe + chunk{si}: {status}")

    # Also test full FP16 for comparison
    variant_name = "full_fp16"
    print(f"\n{'─' * 60}")
    print(f"  {variant_name}: ALL chunks FP16")
    print(f"{'─' * 60}")
    result = _run_multi_chunk_fp16(variant_name, set(range(NUM_CHUNKS)))
    if result is not None:
        status = "EXACT MATCH ✓" if result else "DIVERGED ✗"
        print(f"\n  → Full FP16: {status}")

    # Cleanup engine
    engine.cleanup()

    # ─── Final Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  {'Variant':<30s} {'P0':>8} {'P1':>8} {'P2':>8} {'tok/s':>7} {'FP16':>5}")
    print("  " + "─" * 68)

    for vname in sorted(all_results.keys()):
        vdata = all_results[vname]
        if not vdata or not isinstance(vdata[0], dict):
            continue

        p_status = []
        tps_vals = []
        for pr in vdata:
            if pr.get("loadable") is False:
                p_status.append("FAIL")
            elif pr.get("exact_match", False):
                p_status.append("EXACT")
            elif pr.get("first_token_match", False):
                cm = pr.get("consecutive_match", 0)
                p_status.append(f"~{cm}tok")
            else:
                p_status.append("DIFF")
            if pr.get("tps", 0) > 0:
                tps_vals.append(pr["tps"])

        avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0

        # Count FP16 chunks for this variant
        if vname == "full_fp32":
            fp16_count = "0/9"
        elif vname == "full_fp16":
            fp16_count = "9/9"
        elif vname.startswith("knock_chunk"):
            fp16_count = "1/9"
        elif vname == "all_safe_fp16":
            fp16_count = f"{len(safe_chunks)}/9"
        elif vname.startswith("safe_plus_"):
            fp16_count = f"{len(safe_chunks)+1}/9"
        else:
            fp16_count = "?"

        while len(p_status) < 3:
            p_status.append("N/A")

        print(f"  {vname:<30s} {p_status[0]:>8} {p_status[1]:>8} {p_status[2]:>8} "
              f"{avg_tps:>7.1f} {fp16_count:>5}")

    save_results()
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="FP32→FP16 layer knockout experiment")
    parser.add_argument("--fp16-dir", required=True, help="FP16 baseline directory")
    parser.add_argument("--fp32-dir", required=True, help="FP32 reference directory")
    parser.add_argument("--output", default="tests/dev/knockout_results")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
