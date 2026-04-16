#!/usr/bin/env python3
"""Gemma4 E4B: Multi-round conversation validation on CoreML.

Runs a 3-turn conversation through the chunked CoreML pipeline
(embed → 14 FFN chunks → lm_head) with KV-cache state, validates
that generation is coherent across turns.

Usage:
    cd /path/to/Anemll
    python scripts_gemma4/validate_multiround.py \
        --model-dir gemma4_E4B_lut4ffn_lut6em \
        --hf-model models/google__gemma-4-E4B-it \
        --tokens 80
"""
import argparse
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(1, os.path.dirname(_SCRIPT_DIR))

import coremltools as ct
from config import CTX, NUM_CHUNKS, FFN_LABEL, DEFAULT_OUTPUT, DEFAULT_HF_MODEL

# ── Conversation turns ──

CONVERSATION_TURNS = [
    "What is a stack in computer science?",
    "How does it compare to a queue?",
    "Give me a Python example of each.",
]


# ── Helpers ──

def _argmax_from_lm_out(lm_out):
    """Extract argmax token from lm_head output (split logits or single)."""
    if "logits" in lm_out:
        return int(np.argmax(lm_out["logits"].flatten()))
    split_keys = sorted(
        [k for k in lm_out if k.startswith("logits")],
        key=lambda k: int(k.replace("logits", "")) if k != "logits" else 0,
    )
    if split_keys:
        full = np.concatenate([lm_out[k].flatten() for k in split_keys])
        return int(np.argmax(full))
    raise KeyError(f"Cannot find logits in lm_head output: {list(lm_out.keys())}")


class Gemma4Engine:
    """CoreML inference engine for Gemma4 chunked pipeline.

    Uses separate compiled .mlmodelc for embed/lm_head (no function_name needed),
    and .mlpackage with function_name="infer" for combined FFN chunks (need state).
    """

    def __init__(self, model_dir, label, num_chunks, compute_unit):
        combined_dir = os.path.join(model_dir, f"combined_{label}_dedup")

        # --- Load embed + lm_head (separate compiled models, no function_name) ---
        embed_mlc = os.path.join(model_dir, "embeddings.mlmodelc")
        embed_pkg = os.path.join(model_dir, "embeddings.mlpackage")

        if os.path.exists(embed_mlc):
            print(f"  Loading embed from {embed_mlc} (compiled, CPU_ONLY)...")
            self.embed = ct.models.CompiledMLModel(embed_mlc, ct.ComputeUnit.CPU_ONLY)
        elif os.path.exists(embed_pkg):
            print(f"  Loading embed from {embed_pkg} (CPU_ONLY)...")
            self.embed = ct.models.MLModel(embed_pkg, compute_units=ct.ComputeUnit.CPU_ONLY)
        else:
            raise FileNotFoundError("No embeddings model found")

        # LM head
        lm_mlc = sorted([f for f in os.listdir(model_dir)
                         if f.startswith("lm_head_") and f.endswith(".mlmodelc")])
        lm_pkg = sorted([f for f in os.listdir(model_dir)
                         if f.startswith("lm_head_") and f.endswith(".mlpackage")])
        if lm_mlc:
            lm_path = os.path.join(model_dir, lm_mlc[0])
            print(f"  Loading lm_head from {lm_path} (compiled, CPU_ONLY)...")
            self.lmhead = ct.models.CompiledMLModel(lm_path, ct.ComputeUnit.CPU_ONLY)
        elif lm_pkg:
            lm_path = os.path.join(model_dir, lm_pkg[0])
            print(f"  Loading lm_head from {lm_path} (CPU_ONLY)...")
            self.lmhead = ct.models.MLModel(lm_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        else:
            raise FileNotFoundError("No lm_head model found")

        # --- Load FFN chunks (separate decode .mlpackage, single function, no function_name) ---
        self.ffns = []
        self.num_chunks = num_chunks
        for ci in range(num_chunks):
            pkg_path = os.path.join(model_dir, f"decode_{label}_chunk{ci:02d}.mlpackage")
            if os.path.exists(pkg_path):
                print(f"  Loading chunk {ci:02d} from {pkg_path} ({compute_unit})...")
                m = ct.models.MLModel(pkg_path, compute_units=compute_unit)
                self.ffns.append(m)
            else:
                raise FileNotFoundError(f"decode_{label}_chunk{ci:02d}.mlpackage not found in {model_dir}")

        self.has_ple = True  # Gemma4 always has PLE
        print(f"  per_layer_emb: {self.has_ple} (Gemma4 always uses PLE)")

        self.reset_all()

    def reset_all(self):
        """Reset KV cache states for all chunks."""
        self.states = [m.make_state() for m in self.ffns]

    def _step(self, tok_id, pos, debug=False):
        """Run one token through embed → all chunks → lm_head. Return argmax token."""
        tok = np.array([[tok_id]], dtype=np.int32)

        # Embed
        embed_out = self.embed.predict({"input_ids": tok})
        if debug:
            print(f"    [DEBUG] embed outputs: {', '.join(f'{k}={v.shape}' for k, v in embed_out.items())}")

        # Get hidden states and PLE by name
        hidden = embed_out["hidden_states"]
        ple = embed_out.get("per_layer_emb", None)

        # Causal mask: [1, 1, 1, CTX]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, : pos + 1] = 0.0

        pos_arr = np.array([pos], dtype=np.int32)

        # FFN chunks
        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
            }
            if self.has_ple and ple is not None:
                inp["per_layer_emb"] = ple.astype(np.float16)
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]

        if debug:
            print(f"    [DEBUG] final hidden range: [{hidden.min():.4f}, {hidden.max():.4f}]")

        # LM head
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        return _argmax_from_lm_out(lm_out)


def build_gemma_prompt(tokenizer, messages):
    """Build Gemma chat prompt from conversation messages."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
    # Add generation prompt for model
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def run_conversation(engine, tokenizer, max_gen):
    """Run multi-turn conversation and report results."""
    # Gemma uses: 1 = <eos>, 107 = <end_of_turn>
    stop_ids = {1, 107}
    print(f"  Stop token IDs: {stop_ids}")

    conversation = []
    all_turn_results = []

    for ti, user_msg in enumerate(CONVERSATION_TURNS):
        print(f"\n{'=' * 60}")
        print(f"Turn {ti + 1}/{len(CONVERSATION_TURNS)}: {user_msg}")
        print("=" * 60)

        conversation.append({"role": "user", "content": user_msg})

        # Build full prompt
        prompt_text = build_gemma_prompt(tokenizer, conversation)
        encoded = tokenizer.encode(prompt_text)
        # BOS (token 2 for Gemma) + encoded tokens
        token_list = [2] + encoded.ids

        prompt_len = len(token_list)

        # Truncate if too long
        if prompt_len + max_gen > CTX:
            trim_amount = prompt_len + max_gen - CTX
            token_list = token_list[trim_amount:]
            prompt_len = len(token_list)
            print(f"  [Trimmed {trim_amount} tokens to fit CTX={CTX}]")

        # Reset KV caches for fresh prefill each turn
        engine.reset_all()

        # --- Prefill ---
        t0 = time.time()
        last_next = None
        for i, tid in enumerate(token_list):
            if i >= CTX:
                break
            debug = (ti == 0 and i == 0)  # Debug first token of first turn
            last_next = engine._step(tid, i, debug=debug)
        prefill_pos = len(token_list)
        t_prefill = time.time() - t0

        # --- Decode ---
        gen_tokens = [last_next]
        t_dec = time.time()
        for gi in range(max_gen - 1):
            pos = prefill_pos + gi
            if pos >= CTX - 1:
                print(f"  [Hit context limit at pos {pos}]")
                break
            next_id = engine._step(gen_tokens[-1], pos)
            gen_tokens.append(next_id)
            if next_id in stop_ids:
                break
        t_decode = time.time() - t_dec

        # Decode text (filter out special tokens manually)
        filtered = [t for t in gen_tokens if t not in stop_ids]
        raw_text = tokenizer.decode(filtered)
        conversation.append({"role": "model", "content": raw_text})

        tok_sec = len(gen_tokens) / t_decode if t_decode > 0 else 0
        print(f"  Prompt: {prompt_len} tokens")
        print(f"  Generated: {len(gen_tokens)} tokens in {t_decode * 1000:.0f}ms ({tok_sec:.1f} tok/s)")
        print(f"  Prefill: {t_prefill * 1000:.0f}ms ({prompt_len / t_prefill:.1f} tok/s)")
        print(f"\n  Response:\n  {raw_text[:500]}")
        if len(raw_text) > 500:
            print(f"  ...[truncated, {len(raw_text)} chars total]")

        all_turn_results.append({
            "turn": ti + 1,
            "user": user_msg,
            "response": raw_text,
            "gen_tokens": len(gen_tokens),
            "prompt_tokens": prompt_len,
            "decode_ms": t_decode * 1000,
            "prefill_ms": t_prefill * 1000,
            "tok_sec": tok_sec,
        })

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("MULTI-ROUND VALIDATION SUMMARY")
    print("=" * 60)
    all_ok = True
    for r in all_turn_results:
        has_content = len(r["response"].strip()) > 5
        is_coherent = not _is_repetitive(r["response"])
        status = "PASS" if (has_content and is_coherent) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(
            f"  Turn {r['turn']}: {status} | "
            f"{r['gen_tokens']} tokens @ {r['tok_sec']:.1f} tok/s | "
            f"prefill {r['prefill_ms']:.0f}ms | "
            f"decode {r['decode_ms']:.0f}ms"
        )
        if not has_content:
            print(f"    ⚠ Empty or too-short response")
        if not is_coherent:
            print(f"    ⚠ Repetitive output detected")

    if all_ok:
        print("\n  ✅ All turns passed")
    else:
        print("\n  ❌ Some turns failed — check output above")

    return all_ok


def _is_repetitive(text, threshold=0.4):
    """Heuristic: check if text is heavily repetitive."""
    words = text.split()
    if len(words) < 10:
        return False
    # Check 3-gram repetition
    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return False
    unique_ratio = len(set(trigrams)) / len(trigrams)
    return unique_ratio < threshold


def main():
    parser = argparse.ArgumentParser(
        description="Gemma4 E4B multi-round conversation validation"
    )
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT, help="Model directory")
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="HF model for tokenizer")
    parser.add_argument("--tokens", type=int, default=80, help="Max tokens per turn")
    parser.add_argument("--label", default=FFN_LABEL, help="FFN label")
    parser.add_argument("--chunks", type=int, default=None, help="Override chunk count")
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Use CPU_ONLY for FFN chunks (skip ANE). Note: may not support state on all coremltools versions",
    )
    args = parser.parse_args()

    num_chunks = args.chunks if args.chunks else NUM_CHUNKS
    compute_unit = (
        ct.ComputeUnit.CPU_ONLY if args.cpu_only else ct.ComputeUnit.CPU_AND_NE
    )
    if args.cpu_only:
        print("WARNING: CPU_ONLY may fail with make_state(). Use CPU_AND_NE if you get state errors.")

    print(f"Model dir: {args.model_dir}")
    print(f"HF model:  {args.hf_model}")
    print(f"Chunks:    {num_chunks}, Label: {args.label}")
    print(f"CTX:       {CTX}, Max tokens/turn: {args.tokens}")
    print(f"Compute:   {compute_unit}")

    # Load tokenizer
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(os.path.join(args.hf_model, "tokenizer.json"))

    # Load engine
    print(f"\nLoading models...")
    t0 = time.time()
    engine = Gemma4Engine(args.model_dir, args.label, num_chunks, compute_unit)
    print(f"Models loaded in {time.time() - t0:.1f}s")

    # Run conversation
    success = run_conversation(engine, tokenizer, args.tokens)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
