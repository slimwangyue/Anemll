#!/usr/bin/env python3
"""Sweep LUT4 lm_head group_size to find smallest that recovers accuracy.

Prior result: LUT4 with group_size=8 → 70% accuracy (REJECTED).
Hypothesis: smaller group_size → more LUT codebooks → better accuracy.

Tests group sizes: 1, 2, 4, 8 (and LUT6 gs=8 as control).
Each config runs 3-turn conversation, compares vs fp16 baseline.

Key question: can we get 100% greedy-token match at group_size=1 or 2,
while still saving substantial model size vs LUT6 (462 MB)?

Expected sizes (rough):
  fp16:  ~1213 MB (baseline)
  LUT6:  ~ 462 MB (current production, group_size=8)
  LUT4:  varies by group_size
    gs=1:  ~320-350 MB (most codebooks, highest accuracy)
    gs=2:  ~310-330 MB
    gs=4:  ~300-320 MB
    gs=8:  ~300 MB (fewest codebooks, lowest accuracy)

Usage:
    python tests/dev/test_lmhead_lut4_groupsize_sweep.py
    python tests/dev/test_lmhead_lut4_groupsize_sweep.py --tokens 40 --skip-export
    python tests/dev/test_lmhead_lut4_groupsize_sweep.py --group-sizes 1 2
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, time, argparse, warnings
import numpy as np
import torch
import coremltools as ct
import coremltools.optimize as cto
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
STABLE_DIR = "/Users/yw68/Anemll/qwen3_5_stable_models"
CTX = 1024  # must match compiled FFN models
NUM_CHUNKS = 4

CONVERSATION_TURNS = [
    "What is a stack in computer science?",
    "How does it compare to a queue?",
    "Give me a Python example of each.",
]


# ── Helpers ──────────────────────────────────────────────────────────

def _ensure_ids(tpl_out):
    if hasattr(tpl_out, 'input_ids'):
        ids = tpl_out.input_ids
    elif isinstance(tpl_out, torch.Tensor):
        ids = tpl_out
    else:
        ids = torch.tensor(tpl_out)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.to(torch.int32)


def _build_stop_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)
    return stop_ids


def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


# ── Model loading helpers (matching chat_server.py) ──────────────────

def _load_model(path, compute_unit, function_name=None):
    """Load a CoreML model from .mlpackage or .mlmodelc."""
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, compute_unit)
    kwargs = {"compute_units": compute_unit}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def _find_model(base_dir, name):
    """Find model path, preferring .mlmodelc over .mlpackage."""
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def _detect_shapes(ffn_model, use_combined):
    """Read input shapes from model spec (handles combined multi-function models)."""
    inp_map = {}
    spec = ffn_model.get_spec()
    fn_inputs = None
    if use_combined and hasattr(spec.description, 'functions'):
        for fn in spec.description.functions:
            if fn.name == "infer":
                fn_inputs = fn.input
                break
    if fn_inputs is None:
        fn_inputs = spec.description.input
    for inp in fn_inputs:
        try:
            inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
        except Exception:
            pass
    return inp_map


# ── FlexEngine ───────────────────────────────────────────────────────

class FlexEngine:
    """Engine with swappable lm_head, shared FFN chunks from stable_models.

    Pass shared_models=(embed, lmhead, ffns, prefills, inp_map) to reuse
    already-loaded models across configs (only LM head is swapped).

    CRITICAL: prefill functions MUST be loaded alongside infer functions.
    Without prefill, lm_head.predict() segfaults on ANE.
    """

    def __init__(self, lmhead_path, compute_unit, name="engine",
                 shared_models=None):
        self.name = name
        if shared_models:
            self.embed, self.lmhead, self.ffns, self.prefills, self.inp_map = shared_models
            self._owns_shared = False
        else:
            # Load EXACTLY like chat_server.py: embed → lm_head → FFN(infer+prefill)
            cu = compute_unit
            self.embed = _load_model(_find_model(STABLE_DIR, "embeddings"), cu)
            print(f"    Loading lm_head for {name}...", end="", flush=True)
            import time as _t; _t0 = _t.time()
            self.lmhead = _load_model(lmhead_path, cu)
            print(f" {_t.time()-_t0:.0f}s")
            self.ffns = []
            self.prefills = []
            combined_dir = os.path.join(STABLE_DIR, "combined_LUT4_dedup")
            use_combined = os.path.isdir(combined_dir)
            for ci in range(NUM_CHUNKS):
                if use_combined:
                    path = _find_model(combined_dir, f"chunk{ci}")
                    if path.endswith(".mlmodelc"):
                        use_combined = False
                if use_combined:
                    m_infer = _load_model(path, cu, function_name="infer")
                    m_prefill = _load_model(path, cu, function_name="prefill")
                else:
                    path = _find_model(STABLE_DIR, f"ffn_LUT4_chunk{ci}")
                    m_infer = _load_model(path, cu)
                    m_prefill = None
                self.ffns.append(m_infer)
                self.prefills.append(m_prefill)
            self.inp_map = _detect_shapes(self.ffns[0], use_combined)
            self._owns_shared = True
        self.reset_all()

    def reset_all(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [
            np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
            for _ in range(NUM_CHUNKS)]
        self.lin_recs = [
            np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
            for _ in range(NUM_CHUNKS)]
        # Pre-allocate reusable buffers (same as chat_server.py)
        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)

    def _step(self, tok_id, pos):
        self._tok_buf[0, 0] = tok_id
        hidden = list(self.embed.predict({"input_ids": self._tok_buf}).values())[0]

        self._mask_buf[:, :, :, :] = -65504.0
        self._mask_buf[:, :, :, :pos + 1] = 0
        self._pos_buf[0] = pos

        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": self._pos_buf,
                "causal_mask": self._mask_buf,
                "current_pos": self._pos_buf,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out or "output_logits" in lm_out:
            key = "output_logits" if "output_logits" in lm_out else "logits"
            return int(np.argmax(lm_out[key].flatten())), lm_out[key].flatten()
        return int(lm_out["argmax_idx"].flatten()[0]), None

    def decode(self, token_ids, start_pos, max_gen, stop_ids):
        for i, tid in enumerate(token_ids):
            pos = start_pos + i
            if pos >= CTX:
                break
            last_next, _ = self._step(tid, pos)
        prefill_end_pos = start_pos + len(token_ids)
        tokens = [last_next]
        logits_list = []
        for gi in range(max_gen - 1):
            pos = prefill_end_pos + gi
            if pos >= CTX - 1:
                break
            next_id, logits = self._step(tokens[-1], pos)
            tokens.append(next_id)
            if logits is not None:
                logits_list.append(logits)
            if next_id in stop_ids:
                break
        return tokens, logits_list

    def cleanup(self):
        if self._owns_shared:
            del self.lmhead
            del self.embed
            for m in self.ffns:
                del m
            for m in self.prefills:
                if m is not None:
                    del m
            self.ffns = []
            self.prefills = []
        gc.collect()


# ── Export lm_head with specific LUT bits and group_size ─────────────

def export_lmhead(lut_bits, group_size, out_dir):
    """Export lm_head.mlpackage with given LUT bits and group_size."""
    tag = f"lm_head_LUT{lut_bits}_gs{group_size}"
    path = os.path.join(out_dir, f"{tag}.mlpackage")
    if os.path.exists(path):
        sz = dir_size_mb(path)
        print(f"  {tag} already exists ({sz:.1f} MB), skipping.")
        return path

    print(f"  Exporting {tag}...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(MODEL_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                           num_chunks=NUM_CHUNKS, lut_bits=lut_bits,
                           per_channel=group_size)
    ml = conv.convert_part_3(model, argmax_in_model=False)
    ml.save(path)
    sz = dir_size_mb(path)
    print(f"    Done ({time.time()-t0:.1f}s) — {sz:.1f} MB")
    del ml, conv, model
    gc.collect()
    return path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep LUT4 lm_head group_size for accuracy recovery")
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--export-dir", type=str,
                        default="/tmp/lmhead_groupsize_sweep")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--group-sizes", type=int, nargs="+",
                        default=[1, 2, 4, 8])
    args = parser.parse_args()

    max_gen = args.tokens
    out_dir = args.export_dir
    group_sizes = args.group_sizes
    compute_unit = ct.ComputeUnit.CPU_AND_NE
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)

    print("=" * 85)
    print("  LUT4 lm_head Group-Size Sweep — Qwen3.5-4B")
    print(f"  Tokens/turn: {max_gen}, Turns: {len(CONVERSATION_TURNS)}, CTX={CTX}")
    print(f"  Group sizes: {group_sizes}")
    print(f"  Baseline: LUT6 gs=8 (production lm_head_logits.mlpackage)")
    print("=" * 85)

    # ── 1. Export all LUT4 variants ──
    if not args.skip_export:
        print(f"\n  STEP 1: Export LUT4 lm_head variants")
        print(f"  {'-'*60}")
        lmhead_paths = {}
        for gs in group_sizes:
            lmhead_paths[f"LUT4_gs{gs}"] = export_lmhead(4, gs, out_dir)
    else:
        lmhead_paths = {}
        for gs in group_sizes:
            tag = f"lm_head_LUT4_gs{gs}"
            lmhead_paths[f"LUT4_gs{gs}"] = os.path.join(out_dir, f"{tag}.mlpackage")

    # Add production baseline (LUT6 gs=8)
    prod_lmhead = os.path.join(STABLE_DIR, "lm_head_logits.mlpackage")
    lmhead_paths["LUT6_gs8 (production)"] = prod_lmhead

    # ── 2. Size comparison ──
    print(f"\n{'='*85}")
    print("  SIZE COMPARISON")
    print(f"{'='*85}")
    print(f"\n  {'Config':<30} {'Size (MB)':>10} {'vs LUT6':>10}")
    print(f"  {'-'*52}")
    prod_sz = dir_size_mb(prod_lmhead)
    for label, path in sorted(lmhead_paths.items()):
        if os.path.exists(path):
            sz = dir_size_mb(path)
            saving = prod_sz - sz
            print(f"  {label:<30} {sz:>9.1f}M {saving:>+9.1f}M")
        else:
            print(f"  {label:<30} {'MISSING':>10}")

    # ── 3. Run all configs ──
    print(f"\n{'='*85}")
    print("  ACCURACY COMPARISON (3-turn fresh mode)")
    print(f"{'='*85}")

    # Load models matching chat_server.py order: embed → lm_head → FFN(infer+prefill)
    # CRITICAL: prefill must be loaded alongside infer — without it, lm_head segfaults
    print(f"\n  Loading shared models (chat_server.py order)...")
    import time as _t; _t0 = _t.time()
    cu = compute_unit
    embed = _load_model(_find_model(STABLE_DIR, "embeddings"), cu)
    print(f"    embed loaded in {_t.time()-_t0:.0f}s", flush=True)

    # Load baseline lm_head BEFORE FFN (critical for ANE)
    _tc = _t.time()
    baseline_lm = _load_model(prod_lmhead, cu)
    print(f"    lm_head (baseline) loaded in {_t.time()-_tc:.0f}s", flush=True)

    ffns = []
    prefills = []
    combined_dir = os.path.join(STABLE_DIR, "combined_LUT4_dedup")
    use_combined = os.path.isdir(combined_dir)
    for ci in range(NUM_CHUNKS):
        if use_combined:
            path = _find_model(combined_dir, f"chunk{ci}")
            if path.endswith(".mlmodelc"):
                use_combined = False
        if use_combined:
            _tc = _t.time()
            m_infer = _load_model(path, cu, function_name="infer")
            print(f"    chunk {ci} infer  (combined) {_t.time()-_tc:.0f}s", flush=True)
            _tc = _t.time()
            m_prefill = _load_model(path, cu, function_name="prefill")
            print(f"    chunk {ci} prefill (combined) {_t.time()-_tc:.0f}s", flush=True)
        else:
            path = _find_model(STABLE_DIR, f"ffn_LUT4_chunk{ci}")
            _tc = _t.time()
            m_infer = _load_model(path, cu)
            print(f"    chunk {ci} infer  (separate) {_t.time()-_tc:.0f}s", flush=True)
            m_prefill = None
        ffns.append(m_infer)
        prefills.append(m_prefill)
    inp_map = _detect_shapes(ffns[0], use_combined)
    print(f"  All shared models loaded in {_t.time()-_t0:.0f}s", flush=True)

    all_results = {}

    # Run baseline first (LUT6 production) — lm_head already loaded
    shared = (embed, baseline_lm, ffns, prefills, inp_map)
    print(f"\n  --- LUT6_gs8 (production baseline) ---")
    engine = FlexEngine(prod_lmhead, compute_unit, name="LUT6_gs8",
                        shared_models=shared)
    ref_results = run_conversation(engine, tokenizer, max_gen, stop_ids)
    all_results["LUT6_gs8 (production)"] = ref_results
    engine.cleanup()

    # Run each LUT4 variant — swap lm_head, reuse embed + FFN + prefill
    for label in sorted(k for k in lmhead_paths if k.startswith("LUT4")):
        path = lmhead_paths[label]
        if not os.path.exists(path):
            print(f"\n  --- {label} --- SKIPPED (not found)")
            continue
        print(f"\n  --- {label} ---")
        # Load new lm_head, swap into shared tuple
        print(f"    Loading lm_head for {label}...", end="", flush=True)
        _tc = _t.time()
        new_lm = _load_model(path, cu)
        print(f" {_t.time()-_tc:.0f}s", flush=True)
        shared_variant = (embed, new_lm, ffns, prefills, inp_map)
        engine = FlexEngine(path, compute_unit, name=label,
                            shared_models=shared_variant)
        results = run_conversation(engine, tokenizer, max_gen, stop_ids)
        all_results[label] = results
        engine.cleanup()
        del new_lm
        gc.collect()

    # ── 4. Results table ──
    print(f"\n{'='*85}")
    print("  RESULTS SUMMARY")
    print(f"{'='*85}")

    ref_key = "LUT6_gs8 (production)"
    ref = all_results[ref_key]

    for ti in range(len(CONVERSATION_TURNS)):
        print(f"\n  Turn {ti+1}: \"{CONVERSATION_TURNS[ti]}\"")
        print(f"    {'Config':<30} {'Match':>16} {'1st Diff':>10}")
        print(f"    {'-'*58}")

        ref_toks = ref[ti]['tokens']
        for label, results in all_results.items():
            toks = results[ti]['tokens']
            if label == ref_key:
                print(f"    {label:<30} {'baseline':>16} {'---':>10}")
            else:
                matches = sum(1 for a, b in zip(ref_toks, toks) if a == b)
                total = min(len(ref_toks), len(toks))
                pct = 100 * matches / total if total > 0 else 0
                first_diff = "none"
                for pos, (a, b) in enumerate(zip(ref_toks, toks)):
                    if a != b:
                        first_diff = f"pos {pos}"
                        break
                print(f"    {label:<30} {matches:>3}/{total} ({pct:>5.1f}%) {first_diff:>10}")

    # ── 5. Overall verdict ──
    print(f"\n{'='*85}")
    print("  VERDICT")
    print(f"{'='*85}")
    print(f"\n  {'Config':<30} {'Accuracy':>12} {'Size (MB)':>10} {'vs LUT6':>10}  Status")
    print(f"  {'-'*78}")

    for label, results in all_results.items():
        path = lmhead_paths.get(label)
        if path is None or not os.path.exists(path):
            continue
        sz = dir_size_mb(path)
        saving = prod_sz - sz

        total_match = 0
        total_tok = 0
        for ti in range(len(CONVERSATION_TURNS)):
            ref_toks = ref[ti]['tokens']
            toks = results[ti]['tokens']
            matches = sum(1 for a, b in zip(ref_toks, toks) if a == b)
            total = min(len(ref_toks), len(toks))
            total_match += matches
            total_tok += total

        if label == ref_key:
            print(f"  {label:<30} {'baseline':>12} {sz:>9.1f}M {'---':>10}  PRODUCTION")
        else:
            pct = 100 * total_match / total_tok if total_tok > 0 else 0
            status = "✅ PASS" if pct >= 99.0 else ("⚠️  MARGINAL" if pct >= 90.0 else "❌ FAIL")
            print(f"  {label:<30} {pct:>11.1f}% {sz:>9.1f}M {saving:>+9.1f}M  {status}")

    # ── 6. Token-level divergence detail ──
    print(f"\n{'='*85}")
    print("  DIVERGENCE DETAILS (first 5 mismatches per config)")
    print(f"{'='*85}")

    for label, results in all_results.items():
        if label == ref_key:
            continue
        mismatches = []
        for ti in range(len(CONVERSATION_TURNS)):
            ref_toks = ref[ti]['tokens']
            toks = results[ti]['tokens']
            for pos, (a, b) in enumerate(zip(ref_toks, toks)):
                if a != b:
                    a_str = tokenizer.decode([a]).replace('\n', '\\n')
                    b_str = tokenizer.decode([b]).replace('\n', '\\n')
                    mismatches.append(
                        f"Turn {ti+1} pos {pos}: "
                        f"ref=[{a_str}]({a}) vs [{b_str}]({b})")
        if mismatches:
            print(f"\n  {label}:")
            for m in mismatches[:5]:
                print(f"    {m}")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches)-5} more")
        else:
            print(f"\n  {label}: No mismatches! ✅")

    # ── 7. Generated text ──
    print(f"\n{'='*85}")
    print("  GENERATED TEXT (first turn only)")
    print(f"{'='*85}")
    for label, results in all_results.items():
        text = results[0]['text'][:200].replace('\n', ' ')
        print(f"  [{label}] {text}")

    # Cleanup shared models
    del embed, baseline_lm, ffns, prefills

    print(f"\nDone.")


def run_conversation(engine, tokenizer, max_gen, stop_ids):
    """Run 3-turn conversation with fresh KV cache each turn."""
    conversation = []
    results = []
    for ti, user_msg in enumerate(CONVERSATION_TURNS):
        print(f"    Turn {ti+1} [{engine.name}]: {user_msg[:50]}")
        conversation.append({"role": "user", "content": user_msg})
        input_ids = _ensure_ids(tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True))
        prompt_len = input_ids.shape[1]
        if prompt_len + max_gen > CTX:
            input_ids = input_ids[:, -(CTX - max_gen):]
            prompt_len = input_ids.shape[1]
        engine.reset_all()
        token_list = input_ids[0].tolist()
        gen_tokens, logits_list = engine.decode(token_list, 0, max_gen, stop_ids)
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": "<think>\n" + raw_text})
        results.append({
            'turn': ti + 1, 'prompt_len': prompt_len,
            'tokens': gen_tokens, 'text': raw_text,
        })
        print(f"      [{prompt_len} tok] {raw_text[:100]}")
    return results


if __name__ == "__main__":
    main()
