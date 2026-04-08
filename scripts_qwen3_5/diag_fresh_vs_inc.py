#!/usr/bin/env python3
"""Diagnose fresh-vs-incremental parity gap.

Proves whether the divergence is caused by token sequence differences
(round-trip encode→decode→re-encode) or by numerical state drift.
"""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from transformers import AutoTokenizer
from config import DEFAULT_HF_MODEL

CONVERSATION_TURNS = [
    "What is a stack in computer science?",
    "How does it compare to a queue?",
]

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL, use_fast=False)

# -- Template tokens (same as validate.py) --
def _get_template_tokens(tokenizer):
    return {
        "im_start": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end":   tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "think":    tokenizer.convert_tokens_to_ids("<think>"),
        "nl":       198,
        "user":     tokenizer.encode("user", add_special_tokens=False),
        "assistant": tokenizer.encode("assistant", add_special_tokens=False),
    }

def _build_continuation(tokenizer, tpl_tokens, user_msg, has_stop_token):
    t = tpl_tokens
    msg_tokens = tokenizer.encode(user_msg, add_special_tokens=False)
    continuation = []
    if not has_stop_token:
        continuation += [t["im_end"], t["nl"]]
    else:
        continuation += [t["nl"]]
    continuation += [t["im_start"]] + t["user"] + [t["nl"]]
    continuation += msg_tokens
    continuation += [t["im_end"], t["nl"]]
    continuation += [t["im_start"]] + t["assistant"] + [t["nl"]]
    continuation += [t["think"], t["nl"]]
    return continuation

tpl_tokens = _get_template_tokens(tokenizer)
stop_ids = set()
if tokenizer.eos_token_id is not None:
    stop_ids.add(tokenizer.eos_token_id)
for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
    tok = tokenizer.convert_tokens_to_ids(name)
    if tok is not None and tok != tokenizer.unk_token_id:
        stop_ids.add(tok)

# Simulate turn 1 generating some response tokens (fake response for analysis)
fake_turn1_gen_ids = [1236, 594, 64, 7301, 1882, 429, 686, 1492, 752,
                       4226, 279, 1196, 594, 1681, 1447, 382, 16, 13,
                       3070, 334, 46607, 279, 5765, 25, 334, 262, 198,
                       262, 262, 353, 262, 334, 1474, 594, 16693, 25,
                       334, 576, 1196, 6801, 311]
fake_turn1_text = tokenizer.decode(fake_turn1_gen_ids, skip_special_tokens=True)

print("=" * 70)
print("  FRESH vs INCREMENTAL TOKEN SEQUENCE COMPARISON")
print("=" * 70)

# -- FRESH mode for turn 2 --
conversation_fresh = [
    {"role": "user", "content": CONVERSATION_TURNS[0]},
    {"role": "assistant", "content": "<think>\n" + fake_turn1_text},
    {"role": "user", "content": CONVERSATION_TURNS[1]},
]
fresh_ids = tokenizer.apply_chat_template(
    conversation_fresh, return_tensors="pt", add_generation_prompt=True,
    enable_thinking=True)[0].tolist()

# -- INCREMENTAL mode for turn 2 --
# First, turn 1 tokens:
conversation_t1 = [{"role": "user", "content": CONVERSATION_TURNS[0]}]
t1_ids = tokenizer.apply_chat_template(
    conversation_t1, return_tensors="pt", add_generation_prompt=True,
    enable_thinking=True)[0].tolist()

# Simulate: turn 1 processed t1_ids, then generated fake_turn1_gen_ids
# Now for turn 2, incremental builds continuation:
has_stop = any(t in stop_ids for t in fake_turn1_gen_ids[-1:])
print(f"\nLast gen token: {fake_turn1_gen_ids[-1]} "
      f"({tokenizer.decode([fake_turn1_gen_ids[-1]])}), "
      f"has_stop={has_stop}")

inc_continuation = _build_continuation(tokenizer, tpl_tokens, CONVERSATION_TURNS[1], has_stop)

# The full incremental sequence would be: t1_ids + fake_turn1_gen_ids + inc_continuation
inc_full = t1_ids + fake_turn1_gen_ids + inc_continuation

print(f"\n-- Turn 1 prompt tokens: {len(t1_ids)}")
print(f"-- Turn 1 generated tokens: {len(fake_turn1_gen_ids)}")
print(f"-- Fresh turn 2 total tokens: {len(fresh_ids)}")
print(f"-- Incremental turn 2 total tokens: {len(inc_full)}")
print(f"   (t1={len(t1_ids)} + gen={len(fake_turn1_gen_ids)} + cont={len(inc_continuation)})")

# Compare token by token
min_len = min(len(fresh_ids), len(inc_full))
first_diff = None
for i in range(min_len):
    if fresh_ids[i] != inc_full[i]:
        first_diff = i
        break

if first_diff is not None:
    print(f"\n*** FIRST TOKEN MISMATCH at position {first_diff} ***")
    context = 5
    start = max(0, first_diff - context)
    for i in range(start, min(first_diff + context + 1, min_len)):
        marker = ">>>" if i == first_diff else "   "
        f_tok = fresh_ids[i] if i < len(fresh_ids) else "N/A"
        i_tok = inc_full[i] if i < len(inc_full) else "N/A"
        match = "OK" if f_tok == i_tok else "DIFF"
        f_text = tokenizer.decode([f_tok]) if isinstance(f_tok, int) else ""
        i_text = tokenizer.decode([i_tok]) if isinstance(i_tok, int) else ""
        print(f"  {marker} pos {i:4d}: fresh={f_tok:>8} ({f_text!r:>20})  "
              f"inc={i_tok:>8} ({i_text!r:>20})  [{match}]")
elif len(fresh_ids) != len(inc_full):
    print(f"\n*** LENGTH MISMATCH: fresh={len(fresh_ids)}, inc={len(inc_full)} ***")
    print("  (first {min_len} tokens match)")
else:
    print(f"\n*** ALL {min_len} TOKENS MATCH EXACTLY ***")
    print("  Divergence is NOT caused by token mismatch — must be numerical.")

# Show the boundary region in detail
print(f"\n-- BOUNDARY REGION (end of turn 1 / start of turn 2) --")
# In fresh, find where turn 1 assistant response ends
# In fresh, the chat template re-encodes everything — find the turn 2 user tokens
t2_user_tokens = tokenizer.encode(CONVERSATION_TURNS[1], add_special_tokens=False)
print(f"  Turn 2 user message tokens: {t2_user_tokens}")
print(f"  Turn 2 user message: {tokenizer.decode(t2_user_tokens)!r}")

# Show fresh token sequence around the boundary
t1_end_in_fresh = len(t1_ids)  # approximate
print(f"\n  Fresh tokens around pos {t1_end_in_fresh} (near end of t1 prompt):")
for i in range(max(0, t1_end_in_fresh - 3), min(len(fresh_ids), t1_end_in_fresh + 20)):
    f_tok = fresh_ids[i]
    text = tokenizer.decode([f_tok])
    print(f"    pos {i:4d}: {f_tok:>8} ({text!r})")

# Show incremental boundary
t1_gen_end = len(t1_ids) + len(fake_turn1_gen_ids)
print(f"\n  Incremental tokens around pos {t1_gen_end} (boundary):")
for i in range(max(0, t1_gen_end - 3), min(len(inc_full), t1_gen_end + 20)):
    i_tok = inc_full[i]
    text = tokenizer.decode([i_tok])
    print(f"    pos {i:4d}: {i_tok:>8} ({text!r})")

# Check: does re-encoding the generated text produce the same tokens?
print(f"\n-- ROUND-TRIP ENCODE CHECK --")
reencoded = tokenizer.encode(fake_turn1_text, add_special_tokens=False)
print(f"  Original gen tokens ({len(fake_turn1_gen_ids)}): {fake_turn1_gen_ids[:20]}...")
print(f"  Re-encoded  tokens  ({len(reencoded)}): {reencoded[:20]}...")
if reencoded == fake_turn1_gen_ids:
    print("  *** ROUND-TRIP MATCHES — re-encoding produces identical tokens ***")
else:
    first_rt_diff = next((i for i in range(min(len(reencoded), len(fake_turn1_gen_ids)))
                          if reencoded[i] != fake_turn1_gen_ids[i]), None)
    if first_rt_diff is not None:
        print(f"  *** ROUND-TRIP DIFFERS at position {first_rt_diff} ***")
        print(f"    Original: {fake_turn1_gen_ids[first_rt_diff]} "
              f"({tokenizer.decode([fake_turn1_gen_ids[first_rt_diff]])})")
        print(f"    Re-encoded: {reencoded[first_rt_diff]} "
              f"({tokenizer.decode([reencoded[first_rt_diff]])})")
    elif len(reencoded) != len(fake_turn1_gen_ids):
        print(f"  *** ROUND-TRIP LENGTH DIFFERS: original={len(fake_turn1_gen_ids)}, "
              f"reencoded={len(reencoded)} ***")
