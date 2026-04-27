#!/usr/bin/env python3
"""Per-chunk hidden state comparison: batch-tail vs sequential-tail.

Runs the 356-token prompt (256+100) two ways after batch block1:
  1) Batch tail (100 tokens padded to 256) via prefill model
  2) Sequential tail (100 tokens one at a time) via infer model

After each chunk, compares the hidden states for the LAST valid token
(position 99 in batch, or the last sequential token) to find exactly
which chunk first diverges.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))

import argparse, numpy as np, copy
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
hd = 2560  # hidden_dim for Qwen3.5-4B

# Build prompt with exactly bs+100 tokens
BASE_PROMPT = "Explain in detail the history of artificial intelligence from its inception to modern day. " * 20
messages = [{"role": "user", "content": BASE_PROMPT}]
all_tokens = engine._tokenize_messages(messages, enable_thinking=False)
while len(all_tokens) < bs + 100:
    BASE_PROMPT += " Describe the future of AI. " * 10
    messages = [{"role": "user", "content": BASE_PROMPT}]
    all_tokens = engine._tokenize_messages(messages, enable_thinking=False)
tokens = all_tokens[:bs + 100]
block1_tokens = tokens[:bs]
tail_tokens = tokens[bs:]
print(f"Total: {len(tokens)}, block1: {len(block1_tokens)}, tail: {len(tail_tokens)}, bs: {bs}")


def reset_engine(engine):
    engine._reset_states()
    engine.pos = 0
    engine.rope_offset = 0
    engine.token_history.clear()


def save_states(engine):
    """Deep copy all state arrays."""
    return {
        'lin_convs': [c.copy() for c in engine.lin_convs],
        'lin_recs': [r.copy() for r in engine.lin_recs],
        'states': engine.states,  # CoreML states (can't deep copy, but immutable between runs)
        'pos': engine.pos,
    }


def restore_states(engine, saved):
    """Restore states (shallow for CoreML states)."""
    for ci in range(engine.num_chunks):
        engine.lin_convs[ci][:] = saved['lin_convs'][ci]
        engine.lin_recs[ci][:] = saved['lin_recs'][ci]
    engine.pos = saved['pos']


# ══════════════════════════════════════════════════════════════════════
# Step 1: Process block1 via batch and save state
# ══════════════════════════════════════════════════════════════════════
print("\n=== Processing block1 (256 full tokens) via batch ===")
reset_engine(engine)
_ = engine._batch_prefill(block1_tokens, 0)
print(f"After block1: pos={engine.pos}")

# Save state after block1 — we'll restore before each tail test
# Note: CoreML states can't be deep-copied easily, so we'll re-run block1 each time
block1_conv = [c.copy() for c in engine.lin_convs]
block1_rec = [r.copy() for r in engine.lin_recs]


# ══════════════════════════════════════════════════════════════════════
# Step 2: BATCH TAIL — instrument per-chunk
# ══════════════════════════════════════════════════════════════════════
print("\n=== BATCH TAIL: 100 tokens through prefill model, per-chunk ===")
# Re-run block1 to get clean state
reset_engine(engine)
_ = engine._batch_prefill(block1_tokens, 0)

valid_len = len(tail_tokens)
block_start = engine.pos  # 256

# Build batch inputs (same as _batch_prefill but we intercept chunk outputs)
input_ids = engine._batch_tok_buf
input_ids[0, :] = 0
input_ids[0, :valid_len] = tail_tokens
hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
if valid_len < bs:
    hidden[:, valid_len:, :] = 0.0

mask = engine._batch_mask_buf
mask[:, :, :, :] = -65504.0
for i in range(valid_len):
    mask[0, 0, i, :block_start + i + 1] = 0
for i in range(valid_len, bs):
    mask[0, 0, i, 0] = 0.0

pos_ids = engine._batch_pos_buf
pos_ids[:valid_len] = np.arange(block_start, block_start + valid_len, dtype=np.int32)
pos_ids[valid_len:] = 0

cur_pos = engine._batch_cur_buf
cur_pos[0] = block_start

valid_len_arr = engine._valid_len_buf
valid_len_arr[0] = valid_len

batch_chunk_hiddens = []  # per-chunk hidden state at position valid_len-1
for ci in range(engine.num_chunks):
    inp = {
        "hidden_states": hidden.astype(np.float16),
        "position_ids": pos_ids,
        "causal_mask": mask,
        "current_pos": cur_pos,
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
        "valid_len": valid_len_arr,
    }
    out = engine.prefills[ci].predict(inp, state=engine.states[ci])
    hidden = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']

    # Extract hidden state at the LAST valid position
    if hidden.ndim >= 3 and hidden.shape[1] > 1:
        h_last = hidden[0, valid_len - 1, :].copy().astype(np.float32)
    else:
        h_last = hidden[0, 0, :].copy().astype(np.float32)

    h_norm = np.linalg.norm(h_last)
    batch_chunk_hiddens.append(h_last)
    print(f"  chunk[{ci}]: hidden_shape={hidden.shape}, last_tok_L2={h_norm:.4f}, "
          f"h[0:4]=[{h_last[0]:.4f},{h_last[1]:.4f},{h_last[2]:.4f},{h_last[3]:.4f}]")


# ══════════════════════════════════════════════════════════════════════
# Step 3: SEQUENTIAL TAIL — instrument per-token
# ══════════════════════════════════════════════════════════════════════
print("\n=== SEQUENTIAL TAIL: 100 tokens through infer model, one at a time ===")
# Re-run block1 to get clean state
reset_engine(engine)
_ = engine._batch_prefill(block1_tokens, 0)

seq_chunk_hiddens = []  # after all tail tokens, what's the hidden at each "chunk" layer?
# For sequential, we can only get the FINAL hidden state after all chunks for each token.
# We process all 100 tokens, tracking the last token's hidden state after each chunk.
# But the infer model processes all chunks in one call...
# So instead, let's process the first 99 tokens normally (kv_only),
# then for the LAST token, instrument chunk by chunk.

for ti in range(len(tail_tokens) - 1):
    engine._step_kv_only(tail_tokens[ti], engine.pos)
    engine.pos += 1

# Last token: process chunk by chunk through infer model
last_tok = tail_tokens[-1]
last_pos = engine.pos

# Embed the last token
embed_inp = np.zeros((1, 1), dtype=np.int32)
embed_inp[0, 0] = last_tok
hidden_seq = list(engine.embed.predict({"input_ids": embed_inp}).values())[0]

# Build single-token inputs for infer
seq_mask = np.full((1, 1, 1, engine.ctx), -65504.0, dtype=np.float16)
seq_mask[0, 0, 0, :last_pos + 1] = 0.0
seq_pos = np.array([last_pos], dtype=np.int32)
seq_cur = np.array([last_pos], dtype=np.int32)

for ci in range(engine.num_chunks):
    inp = {
        "hidden_states": hidden_seq.astype(np.float16),
        "position_ids": seq_pos,
        "causal_mask": seq_mask,
        "current_pos": seq_cur,
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
    }
    out = engine.ffns[ci].predict(inp, state=engine.states[ci])
    hidden_seq = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']

    h_last = hidden_seq.flatten()[:hd].astype(np.float32).copy()
    h_norm = np.linalg.norm(h_last)
    seq_chunk_hiddens.append(h_last)
    print(f"  chunk[{ci}]: hidden_shape={hidden_seq.shape}, L2={h_norm:.4f}, "
          f"h[0:4]=[{h_last[0]:.4f},{h_last[1]:.4f},{h_last[2]:.4f},{h_last[3]:.4f}]")


# ══════════════════════════════════════════════════════════════════════
# Step 4: Compare per-chunk
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PER-CHUNK COMPARISON (batch-tail vs seq-tail for last valid token)")
print(f"{'='*70}")
for ci in range(engine.num_chunks):
    b = batch_chunk_hiddens[ci]
    s = seq_chunk_hiddens[ci]
    n = min(len(b), len(s))
    b, s = b[:n], s[:n]
    
    max_diff = np.max(np.abs(b - s))
    cos = np.dot(b, s) / (np.linalg.norm(b) * np.linalg.norm(s) + 1e-12)
    rel_diff = max_diff / (np.max(np.abs(s)) + 1e-12)
    
    status = "✓" if cos > 0.99 else ("⚠" if cos > 0.95 else "✗")
    print(f"  chunk[{ci}]: cos={cos:.6f} max_diff={max_diff:.4f} rel_diff={rel_diff:.4f} {status}")
