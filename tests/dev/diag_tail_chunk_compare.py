#!/usr/bin/env python3
"""Compare batch-tail vs sequential-tail CHUNK BY CHUNK after block1.

Finds exactly which chunk first diverges between:
  A) block1(batch) + tail(batch)   — the broken path
  B) block1(batch) + tail(seq)     — the working path

Also checks embed output shapes for mismatches.
"""
import sys, os, copy, time
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

# Build prompt: block1 (bs tokens) + tail (100 tokens)
BASE_PROMPT = "Explain in detail the history of artificial intelligence from its inception to modern day. " * 20
messages = [{"role": "user", "content": BASE_PROMPT}]
all_tokens = engine._tokenize_messages(messages, enable_thinking=False)
while len(all_tokens) < bs + 100:
    BASE_PROMPT += " Describe the future of AI. " * 10
    messages = [{"role": "user", "content": BASE_PROMPT}]
    all_tokens = engine._tokenize_messages(messages, enable_thinking=False)

block1_tokens = all_tokens[:bs]
tail_tokens = all_tokens[bs:bs+100]
print(f"block1: {len(block1_tokens)} tokens, tail: {len(tail_tokens)} tokens")

def reset(e):
    e._reset_states()
    e.pos = 0
    e.rope_offset = 0
    e.token_history.clear()

def cosine(a, b):
    a_f = a.flatten().astype(np.float32)
    b_f = b.flatten().astype(np.float32)
    dot = np.dot(a_f, b_f)
    na = np.linalg.norm(a_f)
    nb = np.linalg.norm(b_f)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)

# ═══════════════════════════════════════════════════════════════════
# CHECK 0: Verify embed_prefill output shape
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CHECK 0: embed_prefill output shape")
print(f"{'='*70}")
input_ids = np.zeros((1, bs), dtype=np.int32)
input_ids[0, :len(tail_tokens)] = tail_tokens
embed_out = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
print(f"  input_ids shape: {input_ids.shape}")
print(f"  embed_prefill output shape: {embed_out.shape}")
if embed_out.shape[1] != bs:
    print(f"  *** SHAPE MISMATCH: embed outputs {embed_out.shape[1]} but chunk models expect {bs}! ***")
else:
    print(f"  OK: output seq_len={embed_out.shape[1]} matches batch_size={bs}")

# ═══════════════════════════════════════════════════════════════════
# PATH A: block1(batch) + tail(batch) — the BROKEN path
# Capture hidden states at each chunk boundary for the tail block
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PATH A: block1(batch) + tail(batch)")
print(f"{'='*70}")
reset(engine)
# Block1
_ = engine._batch_prefill(block1_tokens, 0)
print(f"  After block1: pos={engine.pos}")

# Save states before tail
conv_after_b1_A = [c.copy() for c in engine.lin_convs]
rec_after_b1_A = [r.copy() for r in engine.lin_recs]

# Tail batch: manually run chunk-by-chunk to capture intermediates
valid_len = len(tail_tokens)
block_start = engine.pos  # should be bs=256

# Embed
input_ids_tail = engine._batch_tok_buf.copy()
input_ids_tail[0, :] = 0
input_ids_tail[0, :valid_len] = tail_tokens
hidden_A = list(engine.embed_prefill.predict({"input_ids": input_ids_tail}).values())[0]
print(f"  Embed output shape: {hidden_A.shape}")
if valid_len < bs:
    hidden_A[:, valid_len:, :] = 0.0

# Build mask
mask_A = np.full((1, 1, bs, engine.ctx), -65504.0, dtype=np.float16)
for i in range(valid_len):
    mask_A[0, 0, i, :block_start + i + 1] = 0
for i in range(valid_len, bs):
    mask_A[0, 0, i, 0] = 0.0

# Position IDs
pos_ids_A = np.zeros(bs, dtype=np.int32)
pos_ids_A[:valid_len] = np.arange(block_start, block_start + valid_len, dtype=np.int32)

cur_pos_A = np.array([block_start], dtype=np.int32)
valid_len_arr = np.array([valid_len], dtype=np.int32)

hidden_per_chunk_A = [hidden_A.copy()]
conv_per_chunk_A = []
rec_per_chunk_A = []

for ci in range(engine.num_chunks):
    inp = {
        "hidden_states": hidden_A.astype(np.float16),
        "position_ids": pos_ids_A,
        "causal_mask": mask_A,
        "current_pos": cur_pos_A,
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
        "valid_len": valid_len_arr,
    }
    out = engine.prefills[ci].predict(inp, state=engine.states[ci])
    hidden_A = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
    hidden_per_chunk_A.append(hidden_A.copy())
    conv_per_chunk_A.append(engine.lin_convs[ci].copy())
    rec_per_chunk_A.append(engine.lin_recs[ci].copy())

# Extract last valid token
if hidden_A.ndim >= 3 and hidden_A.shape[1] > 1:
    final_hidden_A = hidden_A[:, valid_len - 1:valid_len, :]
else:
    final_hidden_A = hidden_A

lm_out_A = engine.lmhead.predict({"hidden_states": final_hidden_A.astype(np.float16)})
logits_A = list(lm_out_A.values())[0]
tok_A = int(np.argmax(logits_A.flatten()))
print(f"  Tail batch token: {tok_A} = '{engine.tokenizer.decode([tok_A])}'")

# ═══════════════════════════════════════════════════════════════════
# PATH B: block1(batch) + tail(sequential) — the CORRECT path
# Capture hidden states at each chunk boundary for FIRST tail token
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PATH B: block1(batch) + tail(sequential)")
print(f"{'='*70}")
reset(engine)
# Block1 (same as path A)
_ = engine._batch_prefill(block1_tokens, 0)
print(f"  After block1: pos={engine.pos}")

# Save states before tail
conv_after_b1_B = [c.copy() for c in engine.lin_convs]
rec_after_b1_B = [r.copy() for r in engine.lin_recs]

# Verify states match between A and B after block1
print("\n  State comparison after block1 (A vs B):")
for ci in range(min(3, engine.num_chunks)):
    cos_conv = cosine(conv_after_b1_A[ci], conv_after_b1_B[ci])
    cos_rec = cosine(rec_after_b1_A[ci], rec_after_b1_B[ci])
    print(f"    chunk{ci}: conv cos={cos_conv:.6f}, rec cos={cos_rec:.6f}")

# Run first tail token through each chunk sequentially, capturing intermediates
first_tok = tail_tokens[0]
# Embed single token
tok_buf = np.array([[first_tok]], dtype=np.int32)
hidden_B_single = list(engine.embed.predict({"input_ids": tok_buf}).values())[0]
print(f"  embed_single output shape: {hidden_B_single.shape}")

# Also embed via embed_prefill for comparison
input_ids_pf = np.zeros((1, bs), dtype=np.int32)
input_ids_pf[0, 0] = first_tok
hidden_B_prefill = list(engine.embed_prefill.predict({"input_ids": input_ids_pf}).values())[0]
print(f"  embed_prefill output shape: {hidden_B_prefill.shape}")
print(f"  Embed cos (single vs prefill[0]): {cosine(hidden_B_single[:, 0:1, :], hidden_B_prefill[:, 0:1, :]):.6f}")

# Build single-token mask
pos_B = engine.pos  # should be bs=256
mask_B = np.full((1, 1, 1, engine.ctx), -65504.0, dtype=np.float16)
mask_B[0, 0, 0, :pos_B + 1] = 0

pos_ids_B = np.array([pos_B], dtype=np.int32)

hidden_per_chunk_B = [hidden_B_single.copy()]

for ci in range(engine.num_chunks):
    inp = {
        "hidden_states": hidden_B_single.astype(np.float16),
        "position_ids": pos_ids_B,
        "causal_mask": mask_B,
        "current_pos": np.array([pos_B], dtype=np.int32),
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
    }
    out = engine.ffns[ci].predict(inp, state=engine.states[ci])
    hidden_B_single = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
    hidden_per_chunk_B.append(hidden_B_single.copy())

lm_out_B = engine.lmhead.predict({"hidden_states": hidden_B_single.astype(np.float16)})
logits_B = list(lm_out_B.values())[0]
tok_B = int(np.argmax(logits_B.flatten()))
print(f"  First seq token: {tok_B} = '{engine.tokenizer.decode([tok_B])}'")

# Now do full sequential for all tail tokens
reset(engine)
_ = engine._batch_prefill(block1_tokens, 0)
for ti, t in enumerate(tail_tokens):
    is_last = (ti == len(tail_tokens) - 1)
    if is_last:
        tok_B_full, _ = engine._step(t, engine.pos)
    else:
        engine._step_kv_only(t, engine.pos)
    engine.pos += 1
print(f"  Full seq token: {tok_B_full} = '{engine.tokenizer.decode([tok_B_full])}'")

# ═══════════════════════════════════════════════════════════════════
# COMPARISON: chunk-by-chunk hidden state divergence
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("COMPARISON: Per-chunk hidden state (first token only)")
print(f"{'='*70}")
print(f"  Path A (batch tail) token: {tok_A}")
print(f"  Path B (seq first tok) token: {tok_B}")
print(f"  Path B (seq all tail) token: {tok_B_full}")
print()

for ci in range(engine.num_chunks + 1):
    label = f"chunk{ci}" if ci < engine.num_chunks else "final"
    if ci == 0:
        label = "embed"

    h_A = hidden_per_chunk_A[ci]
    h_B = hidden_per_chunk_B[ci]

    # Extract first token from batch hidden (path A)
    if h_A.ndim >= 3 and h_A.shape[1] > 1:
        h_A_tok0 = h_A[:, 0:1, :]
    else:
        h_A_tok0 = h_A

    cos = cosine(h_A_tok0, h_B)
    l2_A = np.linalg.norm(h_A_tok0.flatten().astype(np.float32))
    l2_B = np.linalg.norm(h_B.flatten().astype(np.float32))
    diff = np.abs(h_A_tok0.flatten().astype(np.float32) - h_B.flatten().astype(np.float32))
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    status = "OK" if cos > 0.99 else ("DIVERGED" if cos < 0.95 else "drifting")
    if ci == 0:
        label_str = f"  After {label:>8s}"
    else:
        label_str = f"  After {label:>8s}"
    print(f"{label_str}: cos={cos:.6f} L2_A={l2_A:.4f} L2_B={l2_B:.4f} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} [{status}]")

    if cos < 0.95 and ci > 0:
        print(f"  *** FIRST MAJOR DIVERGENCE at chunk{ci-1} output! ***")
        # Show some values
        a_flat = h_A_tok0.flatten().astype(np.float32)[:10]
        b_flat = h_B.flatten().astype(np.float32)[:10]
        print(f"      A[0:10]: {a_flat}")
        print(f"      B[0:10]: {b_flat}")

# Also compare linear attention states after tail processing
print(f"\n{'='*70}")
print("Linear attention states after tail (A=batch, B=seq)")
print(f"{'='*70}")
# Note: path B states were modified by the sequential processing of first token only
# Re-run path B to get states after first token
# (We already have them from the loop above since we saved them)
# Actually we need to compare after all tail tokens for a fair comparison
# Skip this for now, the chunk-by-chunk comparison above is more informative

print("\nDONE")
