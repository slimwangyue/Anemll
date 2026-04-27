#!/usr/bin/env python3
"""Diagnose multi-block vs single-block batch prefill divergence.

Tests:
  A) 100-token prompt as single batch block → first token
  B) 356-token prompt (256+100) via sequential → reference first token
  C) 356-token prompt (256+100) via multi-block batch → first token
  D) 356-token prompt: batch block1 + sequential tail → first token

If A works and C fails (vs B), the issue is multi-block specific.
If D matches B but C doesn't, block1 state is fine but batch-tail is wrong.
If D also fails, block1 state is corrupted.
"""
import sys, os, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))

import argparse, numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--ffn-dir", default=None)
parser.add_argument("--chunks", type=int, default=9)
parser.add_argument("--ctx", type=int, default=4096)
args = parser.parse_args()

import coremltools as ct
from chat_server import ChatEngine, _chunk_tokens

engine = ChatEngine(
    model_dir=args.model_dir,
    hf_path=args.tokenizer,
    ctx=args.ctx,
    num_chunks=args.chunks,
    ffn_dir=args.ffn_dir,
    compute_unit=ct.ComputeUnit.CPU_ONLY,
)
engine.load()
bs = engine._prefill_bs
print(f"Batch size: {bs}")

# Build a prompt with exactly 256+100 = 356 tokens
# Use a simple English prompt and pad/trim to exact length
BASE_PROMPT = "Explain in detail the history of artificial intelligence from its inception to modern day. " * 20
messages = [{"role": "user", "content": BASE_PROMPT}]
all_tokens = engine._tokenize_messages(messages, enable_thinking=False)

# Ensure we have enough tokens; if not, repeat
while len(all_tokens) < bs + 100:
    BASE_PROMPT += " Describe the future of AI. " * 10
    messages = [{"role": "user", "content": BASE_PROMPT}]
    all_tokens = engine._tokenize_messages(messages, enable_thinking=False)

# Trim to exactly bs + 100
target_len = bs + 100
tokens_356 = all_tokens[:target_len]
# Also get the last 100 tokens as a standalone prompt
tokens_100 = all_tokens[:100]

print(f"\n{'='*70}")
print(f"Token counts: tokens_356={len(tokens_356)}, tokens_100={len(tokens_100)}")
print(f"Multi-block: {(len(tokens_356)+bs-1)//bs} blocks, tail={len(tokens_356)%bs or bs}")
print(f"{'='*70}")


def state_norms(engine):
    """Return L2 norms of conv/rec states for first 2 chunks."""
    norms = {}
    for ci in range(min(2, engine.num_chunks)):
        conv = engine.lin_convs[ci]
        rec = engine.lin_recs[ci]
        norms[f'conv[{ci}]'] = float(np.sqrt(np.sum(conv.astype(np.float32)**2)))
        norms[f'rec[{ci}]'] = float(np.sqrt(np.sum(rec.astype(np.float32)**2)))
    return norms


def reset_engine(engine):
    engine._reset_states()
    engine.pos = 0
    engine.rope_offset = 0
    engine.token_history.clear()


# ══════════════════════════════════════════════════════════════════════
# TEST A: 100 tokens as single batch block
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TEST A: 100 tokens — SINGLE batch block")
print(f"{'='*70}")
reset_engine(engine)
tok_a = engine._batch_prefill(tokens_100, 0)
print(f"  First token: {tok_a} = '{engine.tokenizer.decode([tok_a])}'")
print(f"  pos after: {engine.pos}")

# Also run 100 tokens sequential as reference
reset_engine(engine)
orig = engine.has_prefill
engine.has_prefill = False
tok_a_seq = engine._process_prompt(tokens_100)
engine.has_prefill = orig
print(f"  Sequential ref: {tok_a_seq} = '{engine.tokenizer.decode([tok_a_seq])}'")
print(f"  MATCH: {tok_a == tok_a_seq}")


# ══════════════════════════════════════════════════════════════════════
# TEST B: 356 tokens — full sequential (reference)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TEST B: 356 tokens — SEQUENTIAL (reference)")
print(f"{'='*70}")
reset_engine(engine)
orig = engine.has_prefill
engine.has_prefill = False
tok_b = engine._process_prompt(tokens_356)
engine.has_prefill = orig
print(f"  First token: {tok_b} = '{engine.tokenizer.decode([tok_b])}'")
print(f"  pos after: {engine.pos}")
norms_b = state_norms(engine)
print(f"  State norms: {norms_b}")


# ══════════════════════════════════════════════════════════════════════
# TEST C: 356 tokens — multi-block batch (current code)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TEST C: 356 tokens — MULTI-BLOCK batch")
print(f"{'='*70}")
reset_engine(engine)
tok_c = engine._process_prompt(tokens_356)
print(f"  First token: {tok_c} = '{engine.tokenizer.decode([tok_c])}'")
print(f"  pos after: {engine.pos}")
norms_c = state_norms(engine)
print(f"  State norms: {norms_c}")


# ══════════════════════════════════════════════════════════════════════
# TEST D: 356 tokens — batch block1 + sequential tail
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TEST D: 356 tokens — batch block1 + SEQUENTIAL tail")
print(f"{'='*70}")
reset_engine(engine)
# Block 1: batch prefill first 256 tokens
_ = engine._batch_prefill(tokens_356[:bs], 0)
print(f"  After block1: pos={engine.pos}")
norms_d1 = state_norms(engine)
print(f"  State norms after block1: {norms_d1}")
# Tail: sequential remaining 100 tokens
remaining = tokens_356[bs:]
for ti, tok_id in enumerate(remaining):
    is_last = (ti == len(remaining) - 1)
    if is_last:
        tok_d, _ = engine._step(tok_id, engine.pos)
    else:
        engine._step_kv_only(tok_id, engine.pos)
    engine.pos += 1
print(f"  First token: {tok_d} = '{engine.tokenizer.decode([tok_d])}'")
print(f"  pos after: {engine.pos}")


# ══════════════════════════════════════════════════════════════════════
# TEST E: Compare state after block1 (batch vs sequential)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TEST E: State after first 256 tokens — batch vs sequential")
print(f"{'='*70}")

# E1: batch first 256 tokens
reset_engine(engine)
_ = engine._batch_prefill(tokens_356[:bs], 0)
conv_batch = [engine.lin_convs[ci].copy() for ci in range(engine.num_chunks)]
rec_batch = [engine.lin_recs[ci].copy() for ci in range(engine.num_chunks)]
print(f"  Batch: pos={engine.pos}")

# E2: sequential first 256 tokens
reset_engine(engine)
for ti, tok_id in enumerate(tokens_356[:bs]):
    engine._step_kv_only(tok_id, engine.pos)
    engine.pos += 1
conv_seq = [engine.lin_convs[ci].copy() for ci in range(engine.num_chunks)]
rec_seq = [engine.lin_recs[ci].copy() for ci in range(engine.num_chunks)]
print(f"  Sequential: pos={engine.pos}")

# Compare
for ci in range(min(3, engine.num_chunks)):
    conv_b = conv_batch[ci].astype(np.float32)
    conv_s = conv_seq[ci].astype(np.float32)
    rec_b = rec_batch[ci].astype(np.float32)
    rec_s = rec_seq[ci].astype(np.float32)
    
    conv_diff = np.max(np.abs(conv_b - conv_s))
    conv_cos = np.dot(conv_b.flatten(), conv_s.flatten()) / (
        np.linalg.norm(conv_b) * np.linalg.norm(conv_s) + 1e-12)
    rec_diff = np.max(np.abs(rec_b - rec_s))
    rec_cos = np.dot(rec_b.flatten(), rec_s.flatten()) / (
        np.linalg.norm(rec_b) * np.linalg.norm(rec_s) + 1e-12)
    
    print(f"  chunk[{ci}]: conv max_diff={conv_diff:.6f} cos={conv_cos:.6f}"
          f"  rec max_diff={rec_diff:.6f} cos={rec_cos:.6f}")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  A (single-block 100):     {tok_a} = '{engine.tokenizer.decode([tok_a])}'  "
      f"{'✓ matches seq' if tok_a == tok_a_seq else '✗ DIFFERS from seq'}")
print(f"  B (sequential 356, ref):  {tok_b} = '{engine.tokenizer.decode([tok_b])}'")
print(f"  C (multi-block 356):      {tok_c} = '{engine.tokenizer.decode([tok_c])}'  "
      f"{'✓ matches B' if tok_c == tok_b else '✗ DIFFERS from B'}")
print(f"  D (batch-blk1 + seq-tail):{tok_d} = '{engine.tokenizer.decode([tok_d])}'  "
      f"{'✓ matches B' if tok_d == tok_b else '✗ DIFFERS from B'}")
print()
if tok_c != tok_b and tok_d == tok_b:
    print("  → Block1 state is fine. Issue is in batch tail processing.")
elif tok_c != tok_b and tok_d != tok_b:
    print("  → Block1 state is CORRUPTED by batch prefill. Issue is in block1.")
elif tok_c == tok_b:
    print("  → Multi-block batch matches sequential. No issue detected!")
