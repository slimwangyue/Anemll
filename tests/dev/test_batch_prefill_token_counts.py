#!/usr/bin/env python3
"""Test batch prefill with different token counts to validate the NaN fix.

Tests 5 token-count scenarios that stress different padding configurations:
  1. 256 tokens — exactly 1 full block (0 padding)
  2. 257 tokens — 1 full + 1 tail token (255 padding = worst case)
  3. 378 tokens — original Andorra test (134 padding)
  4. 511 tokens — 1 full + 255 tail (1 padding = minimal)
  5. 600 tokens — 2 full + 88 tail (3+ blocks)

Usage:
    python tests/dev/test_batch_prefill_token_counts.py \\
        --model-dir /path/to/compiled/models \\
        --tokenizer /path/to/hf/tokenizer
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))

from chat_server import ChatEngine

# ── Andorra prompt (produces ~378 tokens with ChatML wrapping) ──
ANDORRA_TEXT = """\
Andorra
Geography
Andorra is a small, landlocked country in southwestern Europe, located in the eastern Pyrenees mountain range and bordered by Spain and France. It is the sixth-smallest nation in Europe, having an area of 468 square kilometres (181 sq mi) and a population of approximately 79,034. The Andorran people are a Romance ethnic group of originally Catalan descent. Andorra is the 16th-smallest country in the world by land and the 11th-smallest by population. Its capital, Andorra la Vella, is the highest capital city in Europe, at an elevation of 1,023 metres (3,356 ft) above sea level. The official language is Catalan, but Spanish, Portuguese, and French are also commonly spoken.
Economy
Tourism, the mainstay of Andorra s tiny, well-to-do economy, accounts for roughly 80% of GDP. An estimated 9 million tourists visit annually, attracted by Andorra s duty-free status and by its summer and winter resorts. Andorra s comparative advantage has eroded as the economies of neighbouring France and Spain have been opened up, providing broader availability of goods and lower tariffs. The banking sector, with its tax haven status, also contributes substantially to the economy. Agricultural production is limited only 2% of the land is arable and most food has to be imported. The principal livestock activity is sheep raising. Manufacturing output consists mainly of cigarettes, cigars, and furniture. Andorra is a member of the EU Customs Union and is treated as an EU member for trade in manufactured goods (no tariffs) and as a non-EU member for agricultural products. Translate to Chinese"""

# Test token counts and their descriptions
TEST_CASES = [
    (256,  "1 full block, 0 padding"),
    (257,  "1 full + 1 tail, 255 padding (WORST CASE)"),
    (378,  "original Andorra (122 tail, 134 padding)"),
    (511,  "1 full + 255 tail, 1 padding (MINIMAL)"),
    (600,  "2 full + 88 tail, 168 padding (3 blocks)"),
]


def run_test(engine, target_tokens, desc, max_gen=128):
    """Run one test case with truncated tokens."""
    print(f"\n{'='*70}")
    print(f"TEST: {target_tokens} tokens — {desc}")
    print(f"{'='*70}")

    # Build ChatML prompt
    messages = [{"role": "user", "content": ANDORRA_TEXT}]
    full_tokens = engine._tokenize_messages(messages, enable_thinking=False)

    if len(full_tokens) < target_tokens:
        # Pad by repeating the prompt
        extra_messages = [{"role": "user", "content": ANDORRA_TEXT + " " + ANDORRA_TEXT}]
        full_tokens = engine._tokenize_messages(extra_messages, enable_thinking=False)
        if len(full_tokens) < target_tokens:
            print(f"  SKIP: Cannot generate {target_tokens} tokens "
                  f"(max={len(full_tokens)} from doubled prompt)")
            return None

    # Truncate to target length
    tokens = full_tokens[:target_tokens]
    print(f"  Prompt tokens: {len(tokens)} (truncated from {len(full_tokens)})")

    # Reset engine state
    engine._reset_states()
    engine.pos = 0
    engine.rope_offset = 0
    engine.token_history.clear()

    bs = engine._prefill_bs
    n_blocks = (len(tokens) + bs - 1) // bs
    tail_len = len(tokens) % bs or bs
    padding = bs - tail_len if tail_len < bs else 0
    print(f"  Batch size: {bs}, blocks: {n_blocks}, "
          f"tail: {tail_len}, padding: {padding}")

    # Prefill
    t0 = time.time()
    first_token = engine._process_prompt(tokens)
    prefill_ms = (time.time() - t0) * 1000

    if first_token is None:
        print(f"  FAIL: _process_prompt returned None")
        return {"tokens": target_tokens, "status": "FAIL", "reason": "None"}

    print(f"  Prefill done in {prefill_ms:.0f}ms, "
          f"first_token={first_token}, pos={engine.pos}")

    # Decode
    generated = [first_token]
    current = first_token
    for i in range(max_gen - 1):
        if engine.pos >= engine.ctx - 1:
            break
        if current in engine.stop_ids and i > 0:
            break
        next_tok, _ = engine._step(current, engine.pos)
        engine.pos += 1
        current = next_tok
        if current in engine.stop_ids:
            generated.append(current)
            break
        generated.append(current)

    # Decode text
    text = engine.tokenizer.decode(generated, skip_special_tokens=True)
    hit_eos = generated[-1] in engine.stop_ids

    print(f"  Generated: {len(generated)} tokens, EOS={hit_eos}")
    print(f"  Output:\n---\n{text}\n---")

    # Check for NaN indicators (garbage output)
    has_garbage = False
    if len(text.strip()) == 0:
        has_garbage = True
    if any(c in text for c in ['\x00', '\ufffd']):
        has_garbage = True
    # Check for extreme repetition (sign of NaN corruption)
    words = text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.1:
            has_garbage = True

    status = "PASS" if not has_garbage else "FAIL (garbage)"
    print(f"  Status: {status}")

    return {
        "tokens": target_tokens,
        "padding": padding,
        "gen_tokens": len(generated),
        "hit_eos": hit_eos,
        "status": status,
        "text_preview": text[:100],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test batch prefill with different token counts")
    parser.add_argument("--model-dir", required=True,
                        help="Path to compiled model directory")
    parser.add_argument("--tokenizer", required=True,
                        help="Path to HF tokenizer directory")
    parser.add_argument("--ctx", type=int, default=1024,
                        help="Context length (default: 1024)")
    parser.add_argument("--chunks", type=int, default=9,
                        help="Number of FFN chunks (default: 9)")
    parser.add_argument("--ffn-dir", default=None,
                        help="Override FFN chunk directory (to force separate .mlmodelc)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use CPU_ONLY compute unit (avoids ANE compilation)")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated token counts to test (default: all)")
    args = parser.parse_args()

    # Parse specific cases if provided
    if args.cases:
        requested = set(int(x) for x in args.cases.split(","))
        cases = [(n, d) for n, d in TEST_CASES if n in requested]
        if not cases:
            cases = [(int(args.cases), "custom")]
    else:
        cases = TEST_CASES

    print(f"[test] Loading engine from {args.model_dir}")
    compute_unit = None
    if args.cpu_only:
        import coremltools as ct
        compute_unit = ct.ComputeUnit.CPU_ONLY
        print("[test] Using CPU_ONLY compute unit")
    engine = ChatEngine(
        model_dir=args.model_dir,
        hf_path=args.tokenizer,
        ctx=args.ctx,
        num_chunks=args.chunks,
        ffn_dir=args.ffn_dir,
        compute_unit=compute_unit,
    )
    engine.load()

    results = []
    for target, desc in cases:
        r = run_test(engine, target, desc)
        if r:
            results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Tokens':>6} {'Padding':>8} {'Gen':>5} {'EOS':>5} {'Status':<20} Preview")
    print(f"{'-'*6:>6} {'-'*8:>8} {'-'*5:>5} {'-'*5:>5} {'-'*20:<20} {'-'*30}")
    for r in results:
        print(f"{r['tokens']:>6} {r['padding']:>8} "
              f"{r['gen_tokens']:>5} {str(r['hit_eos']):>5} "
              f"{r['status']:<20} {r['text_preview'][:30]}")

    failures = [r for r in results if "FAIL" in r["status"]]
    if failures:
        print(f"\n  {len(failures)} FAILED test(s)")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} tests PASSED")


if __name__ == "__main__":
    main()
