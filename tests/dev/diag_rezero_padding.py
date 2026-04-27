#!/usr/bin/env python3
"""Test whether re-zeroing padding hidden states after each chunk fixes tail batch.

Hypothesis: padding hidden states accumulate non-zero values through residual
connections and MLP, eventually causing numerical issues at valid positions.
Fix: zero padding hidden after each chunk output.
"""
import sys, os
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

# ═════════════════════════════════════════════════════════════════
# REFERENCE: block1(batch) + tail(sequential)
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("REFERENCE: block1(batch) + tail(sequential)")
print(f"{'='*70}")
reset(engine)
_ = engine._batch_prefill(block1_tokens, 0)
for ti, t in enumerate(tail_tokens):
    is_last = (ti == len(tail_tokens) - 1)
    if is_last:
        tok_ref, _ = engine._step(t, engine.pos)
    else:
        engine._step_kv_only(t, engine.pos)
    engine.pos += 1
print(f"  Token: {tok_ref} = '{engine.tokenizer.decode([tok_ref])}'")

# ═════════════════════════════════════════════════════════════════
# TEST 1: block1(batch) + tail(batch) — NO re-zeroing (current code)
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TEST 1: block1(batch) + tail(batch) — NO re-zeroing")
print(f"{'='*70}")
reset(engine)
_ = engine._batch_prefill(block1_tokens, 0)
tok_no_zero = engine._batch_prefill(tail_tokens, engine.pos)
print(f"  Token: {tok_no_zero} = '{engine.tokenizer.decode([tok_no_zero])}'")
print(f"  Match: {tok_no_zero == tok_ref}")

# ═════════════════════════════════════════════════════════════════
# TEST 2: block1(batch) + tail(batch) — WITH re-zeroing after each chunk
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TEST 2: block1(batch) + tail(batch) — WITH re-zeroing after each chunk")
print(f"{'='*70}")
reset(engine)
_ = engine._batch_prefill(block1_tokens, 0)

# Manual tail batch with re-zeroing
valid_len = len(tail_tokens)
block_start = engine.pos

# Embed
input_ids = engine._batch_tok_buf.copy()
input_ids[0, :] = 0
input_ids[0, :valid_len] = tail_tokens
hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
if valid_len < bs:
    hidden[:, valid_len:, :] = 0.0

# Build mask
mask = np.full((1, 1, bs, engine.ctx), -65504.0, dtype=np.float16)
for i in range(valid_len):
    mask[0, 0, i, :block_start + i + 1] = 0
for i in range(valid_len, bs):
    mask[0, 0, i, 0] = 0.0

pos_ids = np.zeros(bs, dtype=np.int32)
pos_ids[:valid_len] = np.arange(block_start, block_start + valid_len, dtype=np.int32)
cur_pos = np.array([block_start], dtype=np.int32)
valid_len_arr = np.array([valid_len], dtype=np.int32)

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

    # RE-ZERO padding hidden states after each chunk
    if valid_len < bs:
        hidden[:, valid_len:, :] = 0.0

# Extract last valid token
if hidden.ndim >= 3 and hidden.shape[1] > 1:
    hidden = hidden[:, valid_len - 1:valid_len, :]

lm_out = engine.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
logits = list(lm_out.values())[0]
tok_rezeroed = int(np.argmax(logits.flatten()))
print(f"  Token: {tok_rezeroed} = '{engine.tokenizer.decode([tok_rezeroed])}'")
print(f"  Match ref: {tok_rezeroed == tok_ref}")

# ═════════════════════════════════════════════════════════════════
# TEST 3: Same as test 2, also check padding norm growth without zeroing
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TEST 3: Padding hidden state norms (NO re-zeroing)")
print(f"{'='*70}")
reset(engine)
_ = engine._batch_prefill(block1_tokens, 0)

# Embed
input_ids[0, :] = 0
input_ids[0, :valid_len] = tail_tokens
hidden_nz = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
if valid_len < bs:
    hidden_nz[:, valid_len:, :] = 0.0

for ci in range(engine.num_chunks):
    inp = {
        "hidden_states": hidden_nz.astype(np.float16),
        "position_ids": pos_ids,
        "causal_mask": mask,
        "current_pos": cur_pos,
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
        "valid_len": valid_len_arr,
    }
    out = engine.prefills[ci].predict(inp, state=engine.states[ci])
    hidden_nz = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']

    # Measure norms
    valid_norm = np.linalg.norm(hidden_nz[:, :valid_len, :].flatten().astype(np.float32))
    pad_norm = np.linalg.norm(hidden_nz[:, valid_len:, :].flatten().astype(np.float32))
    pad_max = np.abs(hidden_nz[:, valid_len:, :].astype(np.float32)).max()
    has_nan = np.any(np.isnan(hidden_nz))
    has_inf = np.any(np.isinf(hidden_nz))
    print(f"  chunk{ci}: valid_L2={valid_norm:.1f} pad_L2={pad_norm:.1f} pad_max={pad_max:.1f} nan={has_nan} inf={has_inf}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Reference (seq): {tok_ref} = '{engine.tokenizer.decode([tok_ref])}'")
print(f"  No re-zero:      {tok_no_zero} = '{engine.tokenizer.decode([tok_no_zero])}' match={tok_no_zero==tok_ref}")
print(f"  With re-zero:    {tok_rezeroed} = '{engine.tokenizer.decode([tok_rezeroed])}' match={tok_rezeroed==tok_ref}")
