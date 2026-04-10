#!/usr/bin/env python3
"""Chunk2 INTRA-layer FP16 knockout analysis for Qwen3.5-4B.

True per-layer compute precision control: uses FP16ComputePrecision(op_selector=...)
to apply FP16 only to MIL ops belonging to a specific transformer layer INSIDE
one single chunk2 model — not boundary casts, not model splitting.

The op_selector uses transitive input dependency tracing: if an op's inputs
ultimately come from weight ops named 'model_model_layers_N_...', that op
belongs to layer N.  Only ops attributed to the target layer get the fp16 pass.

Configurations tested:
  1. chunk2_all_fp32        — baseline: all fp32 compute
  2. chunk2_all_fp16        — swap chunk2 from fp16 pipeline
  3. chunk2_L7_fp16_inside  — layer 7 internal compute in fp16, rest fp32
  4. chunk2_L8_fp16_inside  — layer 8 internal compute in fp16, rest fp32
  5. chunk2_L9_fp16_inside  — layer 9 internal compute in fp16, rest fp32
  6. chunk2_L10_fp16_inside — layer 10 internal compute in fp16, rest fp32

Usage:
    python tests/dev/chunk2_intra_layer_knockout.py
    python tests/dev/chunk2_intra_layer_knockout.py --skip-export
    python tests/dev/chunk2_intra_layer_knockout.py --tokens 60
"""
import sys, os, gc, time, argparse, json, re
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import coremltools as ct
from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision
import torch
from transformers import AutoTokenizer

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

# ── Paths ──
FP32_DIR = os.path.join(_REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp32")
FP16_DIR = os.path.join(_REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp16")
FP32_DEDUP = os.path.join(FP32_DIR, "combined_LUT4_dedup")
FP16_DEDUP = os.path.join(FP16_DIR, "combined_LUT4_dedup")
HF_MODEL = os.path.join(_REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
KNOCKOUT_DIR = os.path.join(FP32_DIR, "knockout_intra_layer")
COMPILED_DIR = os.path.join(FP32_DIR, "ct_compiled")

# Chunk2 covers layers 7-10
CHUNK2_START, CHUNK2_END = CHUNK_RANGES[2]  # (7, 11)
CHUNK2_LAYERS = list(range(CHUNK2_START, CHUNK2_END))  # [7, 8, 9, 10]

PROMPTS = [
    "What is a stack in computer science?",
    "教我做红烧鱼",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
]


# ======================================================================
#  Layer-aware op_selector for FP16ComputePrecision
# ======================================================================

def make_layer_op_selector(target_layers):
    """Return an op_selector callable for FP16ComputePrecision.

    Uses the "max layer in transitive set" strategy:
    - Trace each op's inputs transitively to find all layer-named weight ops
    - The op's "home layer" is the HIGHEST layer index in its transitive set
      (since layers are sequential: layer 7 feeds layer 8 feeds layer 9, etc.)
    - An op is selected for fp16 only if its home layer is in target_layers

    Example for chunk2 (layers 7-10):
      - Op with transitive={7}      → home=7
      - Op with transitive={7,8}    → home=8 (layer 8 op consuming layer 7 output)
      - Op with transitive={7,8,9}  → home=9
      - Op with transitive={}       → no layer (shared infra, stays fp32)
    """
    target_set = set(target_layers)
    _cache = {}  # op identity (id) -> set of layer indices
    # Pattern matches both: model.model.layers.7 and model_model_layers_7
    _pattern = re.compile(r'layers[._](\d+)')

    def _get_layers(op, visited=None):
        """Recursively trace input ops to find layer attributions."""
        op_id = id(op)
        if op_id in _cache:
            return _cache[op_id]
        if visited is None:
            visited = set()
        if op_id in visited:
            return set()
        visited.add(op_id)

        layers = set()
        # Check this op's own name for layer pattern
        m = _pattern.search(op.name)
        if m:
            layers.add(int(m.group(1)))

        # Check input ops (transitive)
        for inp_val in op.inputs.values():
            if isinstance(inp_val, (list, tuple)):
                for v in inp_val:
                    if hasattr(v, 'op') and v.op is not None:
                        layers |= _get_layers(v.op, visited)
            elif hasattr(inp_val, 'op') and inp_val.op is not None:
                layers |= _get_layers(inp_val.op, visited)

        _cache[op_id] = layers
        return layers

    def selector(op):
        layers = _get_layers(op)
        if not layers:
            return False  # Not attributed to any layer → keep fp32
        # Home layer = highest layer index (sequential processing)
        home_layer = max(layers)
        return home_layer in target_set

    return selector


# ======================================================================
#  Step 1: Export chunk2 variants with per-layer intra-layer FP16
# ======================================================================

def export_knockout_chunk2(model, fp16_layers, label, skip_existing=False):
    """Export chunk2 as a dedup (infer+prefill) mlpackage with one target layer
    assigned FP16 compute precision via FP16ComputePrecision(op_selector=...).
    All other layers remain FP32 compute.
    """
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
    from anemll.utils.combine_models import _save_multifunction_dedup

    dedup_path = os.path.join(KNOCKOUT_DIR, f"{label}.mlpackage")
    if skip_existing and os.path.exists(dedup_path):
        print(f"  [skip] {label}")
        return

    dec_path = os.path.join(KNOCKOUT_DIR, f"{label}_decode.mlpackage")
    pf_path = os.path.join(KNOCKOUT_DIR, f"{label}_prefill.mlpackage")

    # Build the layer-aware op_selector
    op_selector = make_layer_op_selector(fp16_layers)
    custom_precision = FP16ComputePrecision(op_selector=op_selector)

    # Export decode
    if not (skip_existing and os.path.exists(dec_path)):
        print(f"  Exporting {label} decode (layers {CHUNK2_START}-{CHUNK2_END-1}, "
              f"fp16-compute layers: {sorted(fp16_layers)})...")
        t0 = time.time()
        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
            compute_precision="float32",  # base is fp32
        )
        # Override compute_precision with our layer-aware FP16ComputePrecision
        conv.compute_precision = custom_precision
        ml = conv.convert_part_2(
            model, chunk_idx=2, total_chunks=NUM_CHUNKS,
            override_start_layer=CHUNK2_START, override_end_layer=CHUNK2_END,
        )
        ml.save(dec_path)
        del ml, conv; gc.collect()
        print(f"    Saved decode ({time.time()-t0:.1f}s)")

    # Export prefill
    if not (skip_existing and os.path.exists(pf_path)):
        print(f"  Exporting {label} prefill...")
        t0 = time.time()
        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
            compute_precision="float32",  # base is fp32
        )
        # Override compute_precision with our layer-aware FP16ComputePrecision
        conv.compute_precision = custom_precision
        ml = conv.convert_part_2_prefill(
            model, chunk_idx=2, total_chunks=NUM_CHUNKS,
            override_start_layer=CHUNK2_START, override_end_layer=CHUNK2_END,
        )
        ml.save(pf_path)
        del ml, conv; gc.collect()
        print(f"    Saved prefill ({time.time()-t0:.1f}s)")

    # Combine into dedup multifunction model
    print(f"  Combining into dedup: {label}.mlpackage...")
    t0 = time.time()
    sources = [
        (dec_path, "main", "infer"),
        (pf_path, "main", "prefill"),
    ]
    _save_multifunction_dedup(sources, dedup_path, dedup_weights=True, verbose=False)
    print(f"    Saved dedup ({time.time()-t0:.1f}s)")


def export_all_variants(skip_existing=False):
    """Export all per-layer knockout variants of chunk2."""
    os.makedirs(KNOCKOUT_DIR, exist_ok=True)

    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config

    print(f"\n  Loading model weights...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), f"Failed to load weights from {HF_MODEL}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Single-layer knockouts
    for layer in CHUNK2_LAYERS:
        export_knockout_chunk2(model, {layer}, f"L{layer}_fp16_inside", skip_existing)

    del model; gc.collect()


# ======================================================================
#  Step 2: Inference engine (DedupEngine with chunk2 override)
# ======================================================================

def _argmax_from_lm_out(lm_out):
    if "logits" in lm_out:
        return int(np.argmax(lm_out["logits"].flatten()))
    if "argmax_idx" in lm_out:
        return int(lm_out["argmax_idx"].flatten()[0])
    split_keys = sorted([k for k in lm_out if k.startswith("logits")],
                        key=lambda k: int(k.replace("logits", "")))
    if split_keys:
        full = np.concatenate([lm_out[k].flatten() for k in split_keys])
        return int(np.argmax(full))
    raise KeyError(f"Cannot find logits in lm_head output: {list(lm_out.keys())}")


def _build_stop_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)
    return stop_ids


class DedupEngine:
    """Loads 9 dedup chunks with hot-swappable chunk2.

    Shared models (embed, lmhead, chunks 0-1, 3-8) are loaded ONCE.
    Only chunk2 is reloaded per-config to save boot volume disk space.
    """

    _shared = None  # class-level cache for shared models

    @classmethod
    def _get_inp_map(cls, mlpackage_path, function_name="infer"):
        """Extract input shape map from an mlpackage spec (no compilation needed)."""
        spec = ct.utils.load_spec(mlpackage_path)
        fn_inputs = None
        for fn in spec.description.functions:
            if fn.name == function_name:
                fn_inputs = fn.input
                break
        if fn_inputs is None:
            fn_inputs = spec.description.input
        imap = {}
        for inp in fn_inputs:
            try:
                imap[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass
        return imap

    @classmethod
    def _load_shared(cls):
        """Load all models EXCEPT chunk2 (once, cached)."""
        if cls._shared is not None:
            return
        embed_lmhead = os.path.join(COMPILED_DIR, "embed_lmhead_combined.mlmodelc")
        cu = ct.ComputeUnit.CPU_ONLY

        print(f"  Loading shared models (embed, lmhead, chunks 0-1, 3-8) [CPU_ONLY, pre-compiled]...")
        embed = ct.models.CompiledMLModel(embed_lmhead, compute_units=cu,
                                          function_name="embed")
        lmhead = ct.models.CompiledMLModel(embed_lmhead, compute_units=cu,
                                           function_name="lmhead")
        shared_ffns = {}
        shared_inp_maps = {}
        for ci in range(NUM_CHUNKS):
            if ci == 2:
                continue  # chunk2 loaded per-config
            compiled_path = os.path.join(COMPILED_DIR, f"chunk{ci}.mlmodelc")
            print(f"    Loading chunk {ci}: ct_compiled/chunk{ci}.mlmodelc ({cu})")
            m = ct.models.CompiledMLModel(compiled_path, compute_units=cu, function_name="infer")
            shared_ffns[ci] = m
            # Get input shapes from original mlpackage spec
            mlpackage_path = os.path.join(FP32_DEDUP, f"chunk{ci}.mlpackage")
            shared_inp_maps[ci] = cls._get_inp_map(mlpackage_path)
        print(f"  Shared models loaded.")
        cls._shared = {
            'embed': embed, 'lmhead': lmhead,
            'ffns': shared_ffns, 'inp_maps': shared_inp_maps,
        }

    def __init__(self, chunk2_override=None, chunk2_mlpackage_for_spec=None):
        DedupEngine._load_shared()
        s = DedupEngine._shared
        self.embed = s['embed']
        self.lmhead = s['lmhead']

        cu = ct.ComputeUnit.CPU_ONLY
        if chunk2_override:
            path = chunk2_override
            src = os.path.basename(chunk2_override)
        else:
            path = os.path.join(COMPILED_DIR, "chunk2.mlmodelc")
            src = "ct_compiled/chunk2.mlmodelc"
        print(f"  Loading chunk 2: {src} ({cu})")
        chunk2_model = ct.models.CompiledMLModel(path, compute_units=cu, function_name="infer")
        # Get input shapes from original mlpackage
        if chunk2_mlpackage_for_spec:
            imap = self._get_inp_map(chunk2_mlpackage_for_spec)
        else:
            imap = self._get_inp_map(os.path.join(FP32_DEDUP, "chunk2.mlpackage"))

        # Assemble full model list
        self.ffns = []
        self.inp_maps = []
        for ci in range(NUM_CHUNKS):
            if ci == 2:
                self.ffns.append(chunk2_model)
                self.inp_maps.append(imap)
            else:
                self.ffns.append(s['ffns'][ci])
                self.inp_maps.append(s['inp_maps'][ci])

        self.has_linear = 'linear_conv_state' in self.inp_maps[0]
        self._chunk2_model = chunk2_model  # keep ref for cleanup
        self.reset_all()

    def reset_all(self):
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_maps[ci]['linear_conv_state'], dtype=np.float16)
                              for ci in range(NUM_CHUNKS)]
            self.lin_recs = [np.zeros(self.inp_maps[ci]['linear_recurrent_state'], dtype=np.float16)
                             for ci in range(NUM_CHUNKS)]
        else:
            self.lin_convs = [None] * NUM_CHUNKS
            self.lin_recs = [None] * NUM_CHUNKS

    def _step(self, tok_id, pos):
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        for ci in range(NUM_CHUNKS):
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
        return _argmax_from_lm_out(lm_out)

    def generate(self, token_ids, max_gen, stop_ids):
        """Fresh generation: reset state, prefill, decode."""
        self.reset_all()
        t0 = time.time()
        for i, tid in enumerate(token_ids):
            if i >= CTX:
                break
            last_next = self._step(tid, i)
        prefill_end = len(token_ids)
        t_prefill = time.time() - t0

        tokens = [last_next]
        t_dec = time.time()
        for gi in range(max_gen - 1):
            pos = prefill_end + gi
            if pos >= CTX - 1:
                break
            next_id = self._step(tokens[-1], pos)
            tokens.append(next_id)
            if next_id in stop_ids:
                break
        t_decode = time.time() - t_dec
        n_gen = len(tokens)
        tok_per_sec = n_gen / t_decode if t_decode > 0 else 0
        return tokens, t_prefill, t_decode, tok_per_sec

    def cleanup(self):
        """Release only chunk2 model (shared models stay loaded)."""
        if hasattr(self, '_chunk2_model'):
            del self._chunk2_model
        self.ffns = []
        gc.collect()


# ======================================================================
#  Step 3: Evaluation
# ======================================================================

def detect_repetition_onset(tokens, tokenizer, window=10):
    if len(tokens) < window * 2:
        return None
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    words = text.split()
    for w in range(window, 3, -1):
        for i in range(len(words) - w):
            ngram = " ".join(words[i:i+w])
            rest = " ".join(words[i+w:])
            if ngram in rest:
                prefix = " ".join(words[:i+w])
                return len(tokenizer.encode(prefix, add_special_tokens=False))
    return None


def coherence_score(text):
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def evaluate_config(name, chunk2_override, tokenizer, stop_ids, max_gen):
    print(f"\n{'='*70}")
    print(f"  Config: {name}")
    if chunk2_override:
        print(f"  Chunk2: {os.path.basename(chunk2_override)}")
    else:
        print(f"  Chunk2: fp32 reference (no override)")
    print(f"{'='*70}")

    engine = DedupEngine(chunk2_override=chunk2_override)
    results = []

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {pi+1}: {prompt[:60]}")
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True)
        if isinstance(input_ids, torch.Tensor):
            token_list = input_ids[0].tolist()
        else:
            token_list = list(input_ids)

        tokens, t_pf, t_dec, tok_s = engine.generate(token_list, max_gen, stop_ids)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        rep_onset = detect_repetition_onset(tokens, tokenizer)
        coh = coherence_score(text)

        results.append({
            "prompt_idx": pi,
            "prompt": prompt,
            "tokens": tokens,
            "text": text,
            "n_tokens": len(tokens),
            "first_token": tokens[0] if tokens else None,
            "tok_per_sec": tok_s,
            "prefill_s": t_pf,
            "decode_s": t_dec,
            "repetition_onset": rep_onset,
            "coherence": coh,
        })
        print(f"    Generated {len(tokens)} tokens ({tok_s:.1f} tok/s)")
        print(f"    Coherence: {coh:.3f} | Rep onset: {rep_onset}")
        print(f"    Text: {text[:200]}")

    engine.cleanup()
    return results


# ======================================================================
#  Step 4: Analysis and reporting
# ======================================================================

def compare_results(ref_results, test_results):
    comparison = []
    for pi in range(len(PROMPTS)):
        ref = ref_results[pi]
        test = test_results[pi]
        first_match = ref["first_token"] == test["first_token"]
        n_early = min(10, len(ref["tokens"]), len(test["tokens"]))
        early_matches = sum(1 for i in range(n_early)
                           if ref["tokens"][i] == test["tokens"][i])
        early_agreement = early_matches / n_early if n_early > 0 else 0
        n_full = min(len(ref["tokens"]), len(test["tokens"]))
        full_matches = sum(1 for i in range(n_full)
                          if ref["tokens"][i] == test["tokens"][i])
        full_agreement = full_matches / n_full if n_full > 0 else 0
        first_div = n_full
        for i in range(n_full):
            if ref["tokens"][i] != test["tokens"][i]:
                first_div = i
                break
        comparison.append({
            "prompt_idx": pi,
            "first_token_match": first_match,
            "early_agreement": early_agreement,
            "full_agreement": full_agreement,
            "first_divergence_pos": first_div,
            "ref_rep_onset": ref["repetition_onset"],
            "test_rep_onset": test["repetition_onset"],
            "ref_coherence": ref["coherence"],
            "test_coherence": test["coherence"],
            "test_tok_per_sec": test["tok_per_sec"],
            "ref_n_tokens": ref["n_tokens"],
            "test_n_tokens": test["n_tokens"],
        })
    return comparison


def print_sensitivity_table(all_comparisons):
    print("\n" + "=" * 100)
    print("  PER-LAYER INTRA-LAYER SENSITIVITY TABLE")
    print("=" * 100)
    header = (f"{'Config':<25s} | {'P':>1s} | {'1st':>3s} | {'Early10':>7s} | "
              f"{'Full':>6s} | {'1stDiv':>6s} | {'Rep':>4s} | {'Coh':>5s} | {'tok/s':>6s}")
    print(header)
    print("-" * 100)
    for config_name, comparisons, results in all_comparisons:
        for pi, (comp, res) in enumerate(zip(comparisons, results)):
            first_tok = "Y" if comp["first_token_match"] else "N"
            early = f"{comp['early_agreement']:.0%}"
            full = f"{comp['full_agreement']:.0%}"
            div_pos = f"{comp['first_divergence_pos']}" if comp["first_divergence_pos"] < comp["ref_n_tokens"] else "none"
            rep = f"{comp['test_rep_onset']}" if comp["test_rep_onset"] else "none"
            coh = f"{comp['test_coherence']:.3f}"
            tok_s = f"{comp['test_tok_per_sec']:.1f}"
            print(f"{config_name:<25s} | {pi+1} | {first_tok:>3s} | {early:>7s} | "
                  f"{full:>6s} | {div_pos:>6s} | {rep:>4s} | {coh:>5s} | {tok_s:>6s}")
        avg_early = np.mean([c["early_agreement"] for c in comparisons])
        avg_full = np.mean([c["full_agreement"] for c in comparisons])
        avg_coh = np.mean([c["test_coherence"] for c in comparisons])
        avg_toks = np.mean([r["tok_per_sec"] for r in results])
        n_first = sum(1 for c in comparisons if c["first_token_match"])
        n_rep = sum(1 for c in comparisons if c["test_rep_onset"] is not None)
        print(f"{'  (avg)':<25s} |   | {n_first}/3 | {avg_early:>6.0%} | "
              f"{avg_full:>5.0%} |        | {n_rep:>3d}r | {avg_coh:>5.3f} | {avg_toks:>6.1f}")
        print("-" * 100)


def print_ranking(all_comparisons):
    print("\n" + "=" * 70)
    print("  RANKED SENSITIVITY ORDER (most degraded first)")
    print("=" * 70)
    scores = []
    for config_name, comparisons, results in all_comparisons:
        avg_early = np.mean([c["early_agreement"] for c in comparisons])
        avg_full = np.mean([c["full_agreement"] for c in comparisons])
        n_first = sum(1 for c in comparisons if c["first_token_match"])
        n_rep = sum(1 for c in comparisons if c["test_rep_onset"] is not None)
        avg_coh = np.mean([c["test_coherence"] for c in comparisons])
        sensitivity = (
            (1 - n_first / 3) * 30 +
            (1 - avg_early) * 30 +
            (1 - avg_full) * 20 +
            (1 - avg_coh) * 10 +
            (n_rep / 3) * 10
        )
        scores.append((config_name, sensitivity, avg_early, avg_full, n_first, n_rep, avg_coh))
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Rank':>4s}  {'Config':<25s}  {'Score':>5s}  {'1stTok':>6s}  {'Early':>6s}  {'Full':>6s}  {'Reps':>4s}  {'Coh':>5s}")
    for rank, (name, score, ae, af, nf, nr, ac) in enumerate(scores, 1):
        print(f"{rank:>4d}  {name:<25s}  {score:>5.1f}  {nf}/3     {ae:>5.0%}  {af:>5.0%}  {nr:>4d}  {ac:>5.3f}")


def main():
    parser = argparse.ArgumentParser(description="Chunk2 intra-layer FP16 knockout")
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing knockout models")
    parser.add_argument("--tokens", type=int, default=60, help="Max tokens per prompt")
    args = parser.parse_args()

    print("=" * 70)
    print("  Chunk2 INTRA-Layer FP16 Knockout — Qwen3.5-4B")
    print(f"  Method:   FP16ComputePrecision(op_selector=layer_N_only)")
    print(f"  FP32 ref: {FP32_DEDUP}")
    print(f"  FP16 src: {FP16_DEDUP}")
    print(f"  Knockout: {KNOCKOUT_DIR}")
    print(f"  Tokens:   {args.tokens}")
    print(f"  Chunk2:   layers {CHUNK2_LAYERS} (CHUNK_RANGES[2]={CHUNK_RANGES[2]})")
    print("=" * 70)

    # ── Export knockout variants ──
    print("\n-- Step 1: Export chunk2 knockout variants --")
    if not args.skip_export:
        export_all_variants(skip_existing=True)
    else:
        print("  [--skip-export] Reusing existing models")

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(FP32_DIR, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)

    # ── Define configurations ──
    configs = [
        ("chunk2_all_fp32",        None),  # default: compiled/chunk2.mlmodelc
        ("chunk2_all_fp16",        os.path.join(COMPILED_DIR, "chunk2_fp16_ref.mlmodelc")),
        ("chunk2_L7_fp16_inside",  os.path.join(COMPILED_DIR, "L7_fp16_inside.mlmodelc")),
        ("chunk2_L8_fp16_inside",  os.path.join(COMPILED_DIR, "L8_fp16_inside.mlmodelc")),
        ("chunk2_L9_fp16_inside",  os.path.join(COMPILED_DIR, "L9_fp16_inside.mlmodelc")),
        ("chunk2_L10_fp16_inside", os.path.join(COMPILED_DIR, "L10_fp16_inside.mlmodelc")),
    ]

    # ── Run evaluations ──
    print("\n-- Step 2: Run evaluations --")
    ref_name, ref_override = configs[0]
    ref_results = evaluate_config(ref_name, ref_override, tokenizer, stop_ids, args.tokens)

    all_comparisons = []
    all_raw = {ref_name: ref_results}

    for config_name, override_path in configs[1:]:
        results = evaluate_config(config_name, override_path, tokenizer, stop_ids, args.tokens)
        all_raw[config_name] = results
        comparisons = compare_results(ref_results, results)
        all_comparisons.append((config_name, comparisons, results))

    # ── Print results ──
    print("\n-- Step 3: Analysis --")
    print_sensitivity_table(all_comparisons)
    print_ranking(all_comparisons)

    # ── Recommendation ──
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    single_layer_scores = {}
    for config_name, comparisons, results in all_comparisons:
        if "inside" in config_name:
            m = re.search(r'L(\d+)_fp16_inside', config_name)
            if m:
                layer_num = int(m.group(1))
                avg_full = np.mean([c["full_agreement"] for c in comparisons])
                avg_early = np.mean([c["early_agreement"] for c in comparisons])
                n_first = sum(1 for c in comparisons if c["first_token_match"])
                single_layer_scores[layer_num] = {"full": avg_full, "early": avg_early, "first": n_first}

    if single_layer_scores:
        ranked = sorted(single_layer_scores.items(), key=lambda x: x[1]["full"])
        print(f"\n  Layer sensitivity (most → least):")
        for layer, scores in ranked:
            print(f"    Layer {layer}: full={scores['full']:.0%} early={scores['early']:.0%} first={scores['first']}/3")

        most_sensitive = ranked[0][0]
        least_sensitive = ranked[-1][0]
        print(f"\n  Most sensitive layer:  {most_sensitive}")
        print(f"  Least sensitive layer: {least_sensitive}")

        chunk2_fp16_data = None
        for cn, comps, _ in all_comparisons:
            if cn == "chunk2_all_fp16":
                chunk2_fp16_data = comps
                break
        if chunk2_fp16_data:
            chunk2_full = np.mean([c["full_agreement"] for c in chunk2_fp16_data])
            worst_single_full = ranked[0][1]["full"]
            print(f"\n  Full chunk2 fp16 agreement:   {chunk2_full:.0%}")
            print(f"  Worst single-layer agreement: {worst_single_full:.0%}")
            if worst_single_full < 0.90:
                print(f"  → Layer {most_sensitive} is clearly sensitive: keep it FP32, others can be FP16")
            elif chunk2_full < worst_single_full - 0.1:
                print(f"  → Multiple layers compound: whole chunk2 should stay FP32")
            else:
                print(f"  → All layers tolerate FP16 well: chunk2 can safely use FP16")

    # ── Save raw data ──
    out_path = os.path.join(KNOCKOUT_DIR, "knockout_intra_results.json")
    save_data = {}
    for name, res_list in all_raw.items():
        save_data[name] = [
            {k: v for k, v in r.items() if k != "tokens"}
            | {"tokens_list": r["tokens"]}
            for r in res_list
        ]
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Raw results saved to {out_path}")
    print("\n  Done!")


if __name__ == "__main__":
    main()
