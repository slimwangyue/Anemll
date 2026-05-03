#!/usr/bin/env python3
"""A/B multi-round comparison script for warm-path KV cache diagnostics.

Drives the ChatEngine from chat_server.py for a 2-round text-only conversation
and logs detailed diagnostics for comparison with the iOS app's A/B/C modes.

Logs per round:
  - round index
  - position before prefill
  - incremental prompt tokens (first 5 + full length)
  - first token after prefill
  - top-5 logits
  - ropeDelta (rope_offset)
  - model identity info

Usage:
    cd /Volumes/MySSD/Anemll
    source .venv_qwen35/bin/activate
    python tests/dev/test_multiround_ab_compare.py \\
        --model-dir qwen3_5_stable_lut4ffn_lut6em_fp16 \\
        --num-chunks 9 --ctx 1024

Compare the output against iOS app logs filtered by [AB-TEST].
"""

import sys, os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts_qwen3_5")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

import argparse
import numpy as np
import coremltools as ct

# Import ChatEngine from chat_server
from chat_server import ChatEngine, BATCH_SIZE, BLOCK_SIZE

# ── Test prompts (same for both Python server and iOS app) ──
ROUND1_MSG = "What is 2+2?"
ROUND2_MSG = "What is 3+3?"


def top5_logits(logits):
    """Return list of (token_id, logit_value) for top 5."""
    if logits is None:
        return []
    top_idx = np.argpartition(logits.ravel(), -5)[-5:]
    top_idx = top_idx[np.argsort(-logits.ravel()[top_idx])]
    return [(int(idx), float(logits.ravel()[idx])) for idx in top_idx]


def log_round_info(label, engine, prompt_tokens, first_token_id, logits):
    """Print diagnostics matching iOS [AB-TEST] log format."""
    print(f"\n{'='*60}")
    print(f"[AB-TEST] {label}")
    print(f"  position before prefill : {engine.pos}")
    print(f"  ropeDelta (rope_offset) : {engine.rope_offset}")
    print(f"  prompt token count      : {len(prompt_tokens)}")
    print(f"  first 5 token IDs       : {prompt_tokens[:5]}")
    print(f"  prompt decoded          : {repr(engine.tokenizer.decode(prompt_tokens)[:200])}")
    print(f"  first token after prefill: {first_token_id}")
    if first_token_id is not None:
        print(f"  first token decoded     : {repr(engine.tokenizer.decode([first_token_id]))}")
    t5 = top5_logits(logits)
    if t5:
        print(f"  top-5 logits:")
        for rank, (tid, val) in enumerate(t5):
            decoded = repr(engine.tokenizer.decode([tid]))
            print(f"    #{rank+1}: id={tid:6d} logit={val:8.3f} -> {decoded}")
    else:
        print(f"  top-5 logits: N/A (argmax mode)")

    # Model identity (Python uses same instances throughout)
    print(f"  prefill model count     : {len(engine.prefills)}")
    print(f"  prefill model ids       : {[id(m) for m in engine.prefills[:3]]}...")
    print(f"  infer model count       : {len(engine.ffns)}")
    print(f"  infer model ids         : {[id(m) for m in engine.ffns[:3]]}...")
    print(f"  state count             : {len(engine.states)}")
    print(f"{'='*60}")


def run_round(engine, user_msg, round_label, enable_thinking=False, max_decode=30):
    """Run one chat round, collect first-token info and full response."""
    is_first = not any(m["role"] == "user" for m in engine.messages)

    engine.messages.append({"role": "user", "content": user_msg})

    if is_first:
        prompt_tokens = engine._tokenize_messages(
            engine.messages, enable_thinking)
    else:
        prompt_tokens = engine._get_continuation_delta(
            user_msg, enable_thinking)

    pos_before = engine.pos
    rope_before = engine.rope_offset

    # Process prompt
    first_token_id = engine._process_prompt(prompt_tokens)

    # Get logits for the first decode token by re-running lm_head
    # (process_prompt already ran it; we can get logits from a step)
    # Actually, _batch_prefill / _process_prompt already computed the first token.
    # For logits, we need to grab them from _step.  Let's just do one step to get logits.
    # But that would advance position.  Instead, let's use the approach:
    # After _process_prompt returns first_token_id, that's the argmax.
    # For top-5 logits, we re-run lm_head on the hidden state.
    # Simpler: just log the first token and decode a few tokens.

    # For now, log position info and first token
    print(f"\n[AB-TEST] {round_label}: pos_before={pos_before} rope_offset={rope_before}")
    print(f"[AB-TEST] {round_label}: pos_after_prefill={engine.pos}")
    print(f"[AB-TEST] {round_label}: first_token={first_token_id}")

    # Decode a short response
    response_ids = [first_token_id] if first_token_id is not None else []
    if first_token_id is not None:
        im_end = engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
        for _ in range(max_decode):
            if engine.pos >= engine.ctx - 1:
                break
            tok = response_ids[-1]
            if tok == im_end:
                break
            next_tok, logits = engine._step(tok, engine.pos)
            engine.pos += 1
            response_ids.append(next_tok)

            # Log top-5 logits for the FIRST decode step only
            if len(response_ids) == 2 and logits is not None:
                t5 = top5_logits(logits)
                print(f"[AB-TEST] {round_label}: first decode step top-5: "
                      + ", ".join(f"id={tid} logit={val:.3f}" for tid, val in t5))

    response_text = engine.tokenizer.decode(response_ids, skip_special_tokens=True)
    engine.messages.append({"role": "assistant", "content": response_text})
    engine.token_history.extend(prompt_tokens)
    engine.token_history.extend(response_ids)

    print(f"[AB-TEST] {round_label}: response ({len(response_ids)} tokens): {repr(response_text[:200])}")
    print(f"[AB-TEST] {round_label}: final pos={engine.pos}")

    return prompt_tokens, first_token_id, response_text


def main():
    parser = argparse.ArgumentParser(
        description="Multi-round A/B comparison for warm-path KV cache diagnostics")
    parser.add_argument("--model-dir", required=True,
                        help="Path to compiled model directory")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer dir (default: same as --model-dir)")
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--embed-lmhead", default=None)
    parser.add_argument("--ffn-dir", default=None)
    parser.add_argument("--round1", default=ROUND1_MSG,
                        help="Round 1 user message")
    parser.add_argument("--round2", default=ROUND2_MSG,
                        help="Round 2 user message")
    parser.add_argument("--max-decode", type=int, default=30,
                        help="Max tokens to decode per round")
    parser.add_argument("--compute-unit", default="cpu_and_ne",
                        choices=["all", "cpu", "cpu_and_ne"])
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model_dir

    cu_map = {
        "all": ct.ComputeUnit.CPU_AND_NE,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }

    # Auto-detect num_chunks
    num_chunks = args.num_chunks
    if num_chunks is None:
        combined_dir = os.path.join(args.model_dir, "combined_LUT4_dedup")
        if os.path.isdir(combined_dir):
            num_chunks = sum(
                1 for f in os.listdir(combined_dir)
                if f.startswith("chunk") and f.endswith(".mlpackage")
            )
        if not num_chunks:
            num_chunks = 9
        print(f"  Auto-detected {num_chunks} FFN chunks")

    print(f"\n{'#'*60}")
    print(f"# Multi-Round A/B Comparison (Python Reference)")
    print(f"# model_dir : {args.model_dir}")
    print(f"# ctx       : {args.ctx}")
    print(f"# chunks    : {num_chunks}")
    print(f"# compute   : {args.compute_unit}")
    print(f"# round1    : {repr(args.round1)}")
    print(f"# round2    : {repr(args.round2)}")
    print(f"# thinking  : {args.enable_thinking}")
    print(f"{'#'*60}\n")

    engine = ChatEngine(
        args.model_dir, args.tokenizer, ctx=args.ctx,
        num_chunks=num_chunks,
        embed_lmhead_path=args.embed_lmhead,
        ffn_dir=args.ffn_dir,
        compute_unit=cu_map[args.compute_unit],
    )
    engine.load()

    print(f"\n[AB-TEST] Python server: prefill models are NEVER unloaded between rounds")
    print(f"[AB-TEST] This is the reference behavior that iOS Mode B should match\n")

    # ── Round 1 ──
    print(f"\n{'*'*60}")
    print(f"  ROUND 1: {repr(args.round1)}")
    print(f"{'*'*60}")
    r1_tokens, r1_first, r1_text = run_round(
        engine, args.round1, "R1",
        enable_thinking=args.enable_thinking,
        max_decode=args.max_decode,
    )

    # ── Round 2 ──
    print(f"\n{'*'*60}")
    print(f"  ROUND 2: {repr(args.round2)}")
    print(f"{'*'*60}")
    r2_tokens, r2_first, r2_text = run_round(
        engine, args.round2, "R2",
        enable_thinking=args.enable_thinking,
        max_decode=args.max_decode,
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY (copy these values to compare with iOS app)")
    print(f"{'='*60}")
    print(f"Round 1:")
    print(f"  prompt tokens  : {len(r1_tokens)}")
    print(f"  first 5 tokens : {r1_tokens[:5]}")
    print(f"  first token    : {r1_first}")
    print(f"  answer         : {repr(r1_text[:100])}")
    print(f"Round 2:")
    print(f"  prompt tokens  : {len(r2_tokens)}")
    print(f"  first 5 tokens : {r2_tokens[:5]}")
    print(f"  first token    : {r2_first}")
    print(f"  answer         : {repr(r2_text[:100])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
