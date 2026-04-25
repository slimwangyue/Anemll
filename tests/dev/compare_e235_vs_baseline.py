#!/usr/bin/env python3
"""Head-to-head accuracy comparison: E235 vs Baseline (V4+D2) models.

Loads both compiled CoreML models and the HF reference, runs identical
prompts through all three, and reports:
  - Cosine similarity (CoreML vs HF logits)
  - KL divergence (CoreML vs HF probability distributions)
  - Top-1 / Top-5 / Top-10 agreement rates
  - Greedy-decoded text comparison
  - Per-position token match rate

Usage:
    cd /Volumes/MySSD/Anemll
    .venv/bin/python3 tests/dev/compare_e235_vs_baseline.py

Requires both model dirs to have .mlmodelc compiled models.
"""

import gc
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_QW_SCRIPTS = os.path.join(_REPO_ROOT, "scripts_qwen3_5")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _QW_SCRIPTS not in sys.path:
    sys.path.insert(0, _QW_SCRIPTS)

import coremltools as ct
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE

# ── Configuration ──
HF_MODEL = os.path.join(_REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
E235_DIR = os.path.join(_REPO_ROOT, "qwen3_5_4B_E235")
BASELINE_DIR = os.path.join(_REPO_ROOT, "qwen3_5_4B_milestone_3.4_ctx4096_fkv")
NUM_CHUNKS = 9
CTX = 4096
BATCH_SIZE = 256
COMPUTE_UNIT = ct.ComputeUnit.CPU_AND_NE

# Prompts for evaluation (diverse: factual, reasoning, creative, code)
EVAL_PROMPTS = [
    "The capital of France is",
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about the ocean.",
    "What is the Fibonacci sequence? The first 10 numbers are:",
    "def factorial(n):\n    ",
    "The three laws of thermodynamics are:",
    "Translate 'hello world' into Japanese:",
    "If a train travels at 60 mph for 2.5 hours, the distance covered is",
    "The difference between a list and a tuple in Python is",
    "Once upon a time, in a land far away,",
    "The chemical formula for water is",
    "What are the benefits of exercise?",
]

MAX_GENERATE = 30  # tokens to generate for text comparison


# ═══════════════════════════════════════════════════════════════════
# Model loaders
# ═══════════════════════════════════════════════════════════════════

def _find_model(directory, name):
    """Find .mlmodelc or .mlpackage in directory.
    Prefer .mlmodelc (pre-compiled, loads instantly).
    """
    for ext in [".mlmodelc", ".mlpackage"]:
        p = os.path.join(directory, name + ext)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"Model '{name}' not found in {directory}")


def _load_coreml(path, cu, function_name=None):
    """Load CoreML model. Prefer CompiledMLModel for .mlmodelc (instant load)."""
    if path.endswith(".mlmodelc"):
        if function_name:
            return ct.models.CompiledMLModel(path, cu, function_name=function_name)
        return ct.models.CompiledMLModel(path, cu)
    else:
        kwargs = {"compute_units": cu}
        if function_name:
            kwargs["function_name"] = function_name
        return ct.models.MLModel(path, **kwargs)


class CoreMLModel:
    """Wraps a CoreML model dir for sequential inference."""

    def __init__(self, model_dir, label="model"):
        self.label = label
        self.model_dir = model_dir
        self.num_chunks = NUM_CHUNKS
        self.ctx = CTX
        self.embed = None
        self.lmhead = None
        self.ffns = []
        self.states = []
        self.lin_convs = []
        self.lin_recs = []
        # Detect combined dedup directory
        self.combined_dir = None
        for d in sorted(os.listdir(model_dir)):
            if d.startswith("combined_") and d.endswith("_dedup"):
                self.combined_dir = os.path.join(model_dir, d)
                break

    def load(self):
        t0 = time.time()
        cu = COMPUTE_UNIT
        print(f"  [{self.label}] Loading models from {os.path.basename(self.model_dir)}...")

        # Embed + lmhead — try separate .mlmodelc first (no function_name needed)
        sep_embed = os.path.join(self.model_dir, "embed_single.mlmodelc")
        sep_lmhead = os.path.join(self.model_dir, "lm_head_nosplit.mlmodelc")
        if os.path.isdir(sep_embed) and os.path.isdir(sep_lmhead):
            self.embed = ct.models.CompiledMLModel(sep_embed, cu)
            self.lmhead = ct.models.CompiledMLModel(sep_lmhead, cu)
            print(f"  [{self.label}] Loaded embed_single + lm_head_nosplit (.mlmodelc)")
        else:
            # Try combined .mlpackage (supports function_name)
            combined_path = _find_model(self.model_dir, "embed_lmhead_combined")
            self.embed = _load_coreml(combined_path, cu, function_name="embed")
            self.lmhead = _load_coreml(combined_path, cu, function_name="lmhead")
            print(f"  [{self.label}] Loaded embed_lmhead_combined")

        # FFN chunks — prefer combined_LUT4_dedup .mlmodelc (pre-compiled)
        self.ffns = []
        for ci in range(self.num_chunks):
            if self.combined_dir:
                path = _find_model(self.combined_dir, f"chunk{ci}")
                m = _load_coreml(path, cu, function_name="infer")
            else:
                path = _find_model(self.model_dir, f"ffn_LUT4_chunk{ci}")
                m = _load_coreml(path, cu)
            self.ffns.append(m)
            print(f"  [{self.label}] chunk{ci} loaded")

        # Detect shapes from first chunk
        self._detect_shapes()
        self._reset_states()
        elapsed = time.time() - t0
        print(f"  [{self.label}] All models loaded ({elapsed:.1f}s)")

    def _detect_shapes(self):
        """Detect per-chunk conv/rec state shapes from metadata.json."""
        import json as _json
        self.conv_shapes = []
        self.rec_shapes = []
        for ci in range(len(self.ffns)):
            conv_shape = (6, 1024, 32)  # fallback
            rec_shape = (6, 2, 128, 128)

            # Try metadata.json from compiled model directories
            meta_found = False
            for base_dir in [self.combined_dir, self.model_dir]:
                if base_dir is None:
                    continue
                for pattern in [f"chunk{ci}.mlmodelc", f"ffn_LUT4_chunk{ci}.mlmodelc"]:
                    meta_path = os.path.join(base_dir, pattern, "metadata.json")
                    if os.path.isfile(meta_path):
                        try:
                            with open(meta_path) as f:
                                meta_list = _json.load(f)
                            for entry in meta_list:
                                for inp in entry.get("inputSchema", []):
                                    name = inp.get("name", "")
                                    shp_str = inp.get("shape", "")
                                    if name == "linear_conv_state" and shp_str:
                                        conv_shape = tuple(
                                            int(x) for x in shp_str.strip("[]").split(","))
                                    elif name == "linear_recurrent_state" and shp_str:
                                        rec_shape = tuple(
                                            int(x) for x in shp_str.strip("[]").split(","))
                            meta_found = True
                        except Exception:
                            pass
                    if meta_found:
                        break
                if meta_found:
                    break

            self.conv_shapes.append(conv_shape)
            self.rec_shapes.append(rec_shape)
            print(f"  [{self.label}] chunk{ci}: conv={conv_shape} rec={rec_shape}")

    def _reset_states(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [
            np.zeros(self.conv_shapes[ci], dtype=np.float16)
            for ci in range(self.num_chunks)]
        self.lin_recs = [
            np.zeros(self.rec_shapes[ci], dtype=np.float16)
            for ci in range(self.num_chunks)]
        self.pos = 0

    def step(self, tok_id, pos):
        """Run one token. Returns (next_token_id, logits_fp32)."""
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]

        mask = np.full((1, 1, 1, self.ctx), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0

        pos_arr = np.array([pos], dtype=np.int32)

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if "linear_conv_state_out" in out:
                self.lin_convs[ci] = out["linear_conv_state_out"]
                self.lin_recs[ci] = out["linear_recurrent_state_out"]

        lm_out = self.lmhead.predict(
            {"hidden_states": hidden.astype(np.float16)})
        logits = lm_out["logits"].flatten().astype(np.float32)
        return int(np.argmax(logits)), logits

    def generate_with_logits(self, token_ids):
        """Process prompt tokens and collect per-position logits.
        Returns list of (next_token_id, logits) for each position.
        """
        self._reset_states()
        results = []
        for i, tid in enumerate(token_ids):
            next_tok, logits = self.step(tid, i)
            results.append((next_tok, logits))
        return results


# ═══════════════════════════════════════════════════════════════════
# HF Reference
# ═══════════════════════════════════════════════════════════════════

class HFReference:
    """Custom Qwen3.5 reference model for ground-truth logits."""

    def __init__(self, model_path):
        self.model_path = model_path

    def load(self):
        print("  [HF] Loading reference model...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        cfg = Qwen35Config.from_json(os.path.join(self.model_path, "config.json"))
        cfg.context_length = CTX
        cfg.state_length = CTX
        self.model = Qwen35ForCausalLM(cfg)
        assert self.model.load_pretrained_weights(self.model_path), \
            f"Failed to load weights from {self.model_path}"
        self.model.half()  # ensure all params are float16 (safetensors may load as float32)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        elapsed = time.time() - t0
        print(f"  [HF] Loaded in {elapsed:.1f}s")

    def reset(self):
        """Reset KV cache and linear states to zero."""
        self.model.model.kv_cache_0.zero_()
        if hasattr(self.model.model, "linear_conv_state"):
            self.model.model.linear_conv_state.zero_()
        if hasattr(self.model.model, "linear_recurrent_state"):
            self.model.model.linear_recurrent_state.zero_()

    def get_logits(self, token_ids):
        """Get logits for all positions using sequential inference.
        Returns list of logits arrays (one per position).
        """
        self.reset()
        results = []
        for i, tid in enumerate(token_ids):
            input_ids = torch.tensor([[tid]], dtype=torch.long)
            position_ids = torch.tensor([i], dtype=torch.long)
            current_pos = torch.tensor([i], dtype=torch.long)
            causal_mask = torch.zeros((1, 1, 1, CTX), dtype=MODEL_DTYPE)
            causal_mask[:, :, :, i + 1:] = float("-inf")
            with torch.no_grad():
                logits = self.model(
                    input_ids, position_ids=position_ids,
                    causal_mask=causal_mask, current_pos=current_pos)
            # logits shape: (1, 1, vocab_size)
            logits_np = logits.squeeze().float().numpy()
            results.append(logits_np)
        return results

    def generate_greedy(self, token_ids, max_new=30):
        """Generate tokens greedily. Returns list of generated token IDs."""
        self.reset()
        # Prefill
        for i, tid in enumerate(token_ids):
            input_ids = torch.tensor([[tid]], dtype=torch.long)
            position_ids = torch.tensor([i], dtype=torch.long)
            current_pos = torch.tensor([i], dtype=torch.long)
            causal_mask = torch.zeros((1, 1, 1, CTX), dtype=MODEL_DTYPE)
            causal_mask[:, :, :, i + 1:] = float("-inf")
            with torch.no_grad():
                logits = self.model(
                    input_ids, position_ids=position_ids,
                    causal_mask=causal_mask, current_pos=current_pos)
        # First generated token
        next_tok = int(logits.squeeze().argmax(-1).item())
        generated = [next_tok]
        pos = len(token_ids)
        for _ in range(max_new - 1):
            input_ids = torch.tensor([[next_tok]], dtype=torch.long)
            position_ids = torch.tensor([pos], dtype=torch.long)
            current_pos = torch.tensor([pos], dtype=torch.long)
            causal_mask = torch.zeros((1, 1, 1, CTX), dtype=MODEL_DTYPE)
            causal_mask[:, :, :, pos + 1:] = float("-inf")
            with torch.no_grad():
                logits = self.model(
                    input_ids, position_ids=position_ids,
                    causal_mask=causal_mask, current_pos=current_pos)
            next_tok = int(logits.squeeze().argmax(-1).item())
            generated.append(next_tok)
            pos += 1
        return generated


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def kl_divergence(logits_p, logits_q):
    """KL(P || Q) where P=reference, Q=model. Uses log-softmax."""
    p = torch.tensor(logits_p, dtype=torch.float32)
    q = torch.tensor(logits_q, dtype=torch.float32)
    log_p = F.log_softmax(p, dim=-1)
    log_q = F.log_softmax(q, dim=-1)
    p_probs = F.softmax(p, dim=-1)
    kl = (p_probs * (log_p - log_q)).sum().item()
    return max(kl, 0.0)  # clamp numerical noise


def top_k_agreement(logits_ref, logits_model, k):
    """Fraction of top-k tokens in ref that appear in model's top-k."""
    ref_topk = set(np.argsort(logits_ref)[-k:])
    mod_topk = set(np.argsort(logits_model)[-k:])
    return len(ref_topk & mod_topk) / k


# ═══════════════════════════════════════════════════════════════════
# Main comparison
# ═══════════════════════════════════════════════════════════════════

def run_comparison():
    print("=" * 70)
    print("  E235 vs Baseline (V4+D2) Accuracy Comparison")
    print("=" * 70)
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    # Load all models
    print("Loading models...")
    hf_ref = HFReference(HF_MODEL)
    hf_ref.load()

    e235 = CoreMLModel(E235_DIR, "E235")
    e235.load()

    baseline = CoreMLModel(BASELINE_DIR, "Baseline")
    baseline.load()

    print()
    print("=" * 70)
    print("  Running evaluation on %d prompts" % len(EVAL_PROMPTS))
    print("=" * 70)
    print()

    # Accumulators
    all_results = []

    for pi, prompt in enumerate(EVAL_PROMPTS):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) < 1:
            continue

        prompt_short = prompt[:50] + ("..." if len(prompt) > 50 else "")
        print(f"[{pi+1}/{len(EVAL_PROMPTS)}] \"{prompt_short}\" ({len(token_ids)} tokens)")

        # Get HF reference logits (all positions at once)
        hf_logits = hf_ref.get_logits(token_ids)

        # Get CoreML logits (sequential)
        e235_results = e235.generate_with_logits(token_ids)
        baseline_results = baseline.generate_with_logits(token_ids)

        # Per-position metrics (use logits at each position to predict next token)
        n_pos = len(token_ids)
        prompt_metrics = {
            "prompt": prompt_short,
            "n_tokens": n_pos,
            "e235": {"cos": [], "kl": [], "top1": [], "top5": [], "top10": []},
            "baseline": {"cos": [], "kl": [], "top1": [], "top5": [], "top10": []},
        }

        for pos in range(n_pos):
            ref_logits = hf_logits[pos]  # logits predicting token at pos+1
            e235_logits = e235_results[pos][1]
            base_logits = baseline_results[pos][1]

            # E235 metrics
            prompt_metrics["e235"]["cos"].append(cosine_similarity(ref_logits, e235_logits))
            prompt_metrics["e235"]["kl"].append(kl_divergence(ref_logits, e235_logits))
            prompt_metrics["e235"]["top1"].append(top_k_agreement(ref_logits, e235_logits, 1))
            prompt_metrics["e235"]["top5"].append(top_k_agreement(ref_logits, e235_logits, 5))
            prompt_metrics["e235"]["top10"].append(top_k_agreement(ref_logits, e235_logits, 10))

            # Baseline metrics
            prompt_metrics["baseline"]["cos"].append(cosine_similarity(ref_logits, base_logits))
            prompt_metrics["baseline"]["kl"].append(kl_divergence(ref_logits, base_logits))
            prompt_metrics["baseline"]["top1"].append(top_k_agreement(ref_logits, base_logits, 1))
            prompt_metrics["baseline"]["top5"].append(top_k_agreement(ref_logits, base_logits, 5))
            prompt_metrics["baseline"]["top10"].append(top_k_agreement(ref_logits, base_logits, 10))

        # Average metrics for this prompt
        e_cos = np.mean(prompt_metrics["e235"]["cos"])
        b_cos = np.mean(prompt_metrics["baseline"]["cos"])
        e_kl = np.mean(prompt_metrics["e235"]["kl"])
        b_kl = np.mean(prompt_metrics["baseline"]["kl"])
        e_t1 = np.mean(prompt_metrics["e235"]["top1"]) * 100
        b_t1 = np.mean(prompt_metrics["baseline"]["top1"]) * 100
        e_t5 = np.mean(prompt_metrics["e235"]["top5"]) * 100
        b_t5 = np.mean(prompt_metrics["baseline"]["top5"]) * 100

        delta_cos = e_cos - b_cos
        delta_kl = e_kl - b_kl  # lower is better, so negative = E235 better

        print(f"  E235:     cos={e_cos:.6f}  KL={e_kl:.4f}  top1={e_t1:.1f}%  top5={e_t5:.1f}%")
        print(f"  Baseline: cos={b_cos:.6f}  KL={b_kl:.4f}  top1={b_t1:.1f}%  top5={b_t5:.1f}%")
        print(f"  Δ(E235-Base): cos={delta_cos:+.6f}  KL={delta_kl:+.4f}")
        print()

        all_results.append(prompt_metrics)

    # ── Greedy generation comparison ──
    print("=" * 70)
    print("  Greedy text generation comparison (%d tokens)" % MAX_GENERATE)
    print("=" * 70)
    print()

    gen_prompts = EVAL_PROMPTS[:4]  # Use first 4 prompts for generation
    gen_results = []

    for pi, prompt in enumerate(gen_prompts):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_short = prompt[:50] + ("..." if len(prompt) > 50 else "")
        print(f"[{pi+1}/{len(gen_prompts)}] \"{prompt_short}\"")

        # Generate with both models
        for model, label in [(e235, "E235"), (baseline, "Baseline")]:
            model._reset_states()
            generated = list(token_ids)
            for i, tid in enumerate(token_ids):
                next_tok, _ = model.step(tid, i)
            # Now decode
            pos = len(token_ids)
            current_tok = next_tok
            gen_tokens = [current_tok]
            for _ in range(MAX_GENERATE - 1):
                current_tok, _ = model.step(current_tok, pos)
                gen_tokens.append(current_tok)
                pos += 1
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            print(f"  {label:>10}: {text[:120]}")

        # HF reference generation (greedy, sequential)
        hf_gen = hf_ref.generate_greedy(token_ids, max_new=MAX_GENERATE)
        hf_text = tokenizer.decode(hf_gen, skip_special_tokens=True)
        print(f"  {'HF ref':>10}: {hf_text[:120]}")

        # Token match rates
        e235_m = e235
        base_m = baseline
        e235_m._reset_states()
        base_m._reset_states()
        # Re-generate to get tokens
        e235_toks = []
        base_toks = []
        for i, tid in enumerate(token_ids):
            e_next, _ = e235_m.step(tid, i)
            b_next, _ = base_m.step(tid, i)
        e_pos = len(token_ids)
        b_pos = len(token_ids)
        e_cur = e_next
        b_cur = b_next
        e235_toks.append(e_cur)
        base_toks.append(b_cur)
        for _ in range(MAX_GENERATE - 1):
            e_cur, _ = e235_m.step(e_cur, e_pos)
            b_cur, _ = base_m.step(b_cur, b_pos)
            e235_toks.append(e_cur)
            base_toks.append(b_cur)
            e_pos += 1
            b_pos += 1

        hf_gen_arr = hf_gen[:MAX_GENERATE]
        match_len = min(len(e235_toks), len(base_toks), len(hf_gen_arr))
        e_match = sum(1 for i in range(match_len) if e235_toks[i] == hf_gen_arr[i])
        b_match = sum(1 for i in range(match_len) if base_toks[i] == hf_gen_arr[i])
        eb_match = sum(1 for i in range(match_len) if e235_toks[i] == base_toks[i])

        print(f"  Token match vs HF: E235={e_match}/{match_len} ({e_match/match_len*100:.0f}%)  Baseline={b_match}/{match_len} ({b_match/match_len*100:.0f}%)")
        print(f"  E235 vs Baseline:  {eb_match}/{match_len} ({eb_match/match_len*100:.0f}%) identical")
        print()

        gen_results.append({
            "prompt": prompt_short,
            "e235_match_hf": e_match / match_len if match_len > 0 else 0,
            "base_match_hf": b_match / match_len if match_len > 0 else 0,
            "e235_vs_base": eb_match / match_len if match_len > 0 else 0,
        })

    # ── Summary ──
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    # Aggregate all per-position metrics
    e_all_cos = [v for r in all_results for v in r["e235"]["cos"]]
    b_all_cos = [v for r in all_results for v in r["baseline"]["cos"]]
    e_all_kl = [v for r in all_results for v in r["e235"]["kl"]]
    b_all_kl = [v for r in all_results for v in r["baseline"]["kl"]]
    e_all_t1 = [v for r in all_results for v in r["e235"]["top1"]]
    b_all_t1 = [v for r in all_results for v in r["baseline"]["top1"]]
    e_all_t5 = [v for r in all_results for v in r["e235"]["top5"]]
    b_all_t5 = [v for r in all_results for v in r["baseline"]["top5"]]
    e_all_t10 = [v for r in all_results for v in r["e235"]["top10"]]
    b_all_t10 = [v for r in all_results for v in r["baseline"]["top10"]]

    n_total = len(e_all_cos)
    print(f"  Total positions evaluated: {n_total}")
    print()

    header = f"  {'Metric':<25} {'E235':>12} {'Baseline':>12} {'Δ(E235-Base)':>14} {'Winner':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = [
        ("Cosine Similarity", np.mean(e_all_cos), np.mean(b_all_cos), True),   # higher is better
        ("KL Divergence", np.mean(e_all_kl), np.mean(b_all_kl), False),        # lower is better
        ("Top-1 Agreement (%)", np.mean(e_all_t1)*100, np.mean(b_all_t1)*100, True),
        ("Top-5 Agreement (%)", np.mean(e_all_t5)*100, np.mean(b_all_t5)*100, True),
        ("Top-10 Agreement (%)", np.mean(e_all_t10)*100, np.mean(b_all_t10)*100, True),
    ]

    for name, e_val, b_val, higher_is_better in rows:
        delta = e_val - b_val
        if higher_is_better:
            winner = "E235" if delta > 0.0001 else ("Baseline" if delta < -0.0001 else "Tie")
        else:
            winner = "E235" if delta < -0.0001 else ("Baseline" if delta > 0.0001 else "Tie")

        if name.startswith("Top"):
            print(f"  {name:<25} {e_val:>11.1f}% {b_val:>11.1f}% {delta:>+13.2f}% {winner:>10}")
        elif "KL" in name:
            print(f"  {name:<25} {e_val:>12.4f} {b_val:>12.4f} {delta:>+14.4f} {winner:>10}")
        else:
            print(f"  {name:<25} {e_val:>12.6f} {b_val:>12.6f} {delta:>+14.6f} {winner:>10}")

    # Generation summary
    if gen_results:
        e_match_avg = np.mean([r["e235_match_hf"] for r in gen_results]) * 100
        b_match_avg = np.mean([r["base_match_hf"] for r in gen_results]) * 100
        eb_avg = np.mean([r["e235_vs_base"] for r in gen_results]) * 100
        delta_match = e_match_avg - b_match_avg
        winner = "E235" if delta_match > 0.5 else ("Baseline" if delta_match < -0.5 else "Tie")
        print(f"  {'Greedy Token Match (%)':<25} {e_match_avg:>11.1f}% {b_match_avg:>11.1f}% {delta_match:>+13.1f}% {winner:>10}")
        print(f"  {'E235 vs Baseline Match':<25} {eb_avg:>11.1f}%")

    print()

    # Per-prompt breakdown
    print("  Per-prompt cosine similarity:")
    print(f"  {'#':<4} {'Prompt':<45} {'E235':>10} {'Baseline':>10} {'Δ':>10}")
    print("  " + "-" * 82)
    for i, r in enumerate(all_results):
        e_cos = np.mean(r["e235"]["cos"])
        b_cos = np.mean(r["baseline"]["cos"])
        delta = e_cos - b_cos
        print(f"  {i+1:<4} {r['prompt']:<45} {e_cos:>10.6f} {b_cos:>10.6f} {delta:>+10.6f}")

    overall_e = np.mean(e_all_cos)
    overall_b = np.mean(b_all_cos)
    overall_d = overall_e - overall_b
    print("  " + "-" * 82)
    print(f"  {'':4} {'OVERALL':45} {overall_e:>10.6f} {overall_b:>10.6f} {overall_d:>+10.6f}")
    print()

    # Size comparison
    print("  Model sizes (combined_LUT4_dedup/):")
    try:
        e_mb = int(os.popen(f"du -sm '{os.path.join(E235_DIR, 'combined_LUT4_dedup')}'").read().split()[0])
        b_mb = int(os.popen(f"du -sm '{os.path.join(BASELINE_DIR, 'combined_LUT4_dedup')}'").read().split()[0])
        print(f"    E235:     {e_mb} MB")
        print(f"    Baseline: {b_mb} MB")
        print(f"    Delta:    {e_mb - b_mb:+d} MB ({(e_mb - b_mb) / b_mb * 100:+.1f}%)")
    except Exception:
        print("    (could not determine sizes)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
