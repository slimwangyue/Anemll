#!/usr/bin/env python3
"""
KV cache write range analysis: does batch prefill write garbage KV entries
beyond valid_len that contaminate subsequent inference?

For a 35-token tail (valid_len=35) with batch_size=256:
  - Batch prefill writes KV cache at positions 256..511 (256 entries)
  - But only 35 entries are valid (positions 256..290)
  - Positions 291..511 contain KV entries from padding hidden states
  - Does sequential decode ever attend to these positions?
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine

MODEL_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3'
HF_PATH   = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'

engine = ChatEngine(model_dir=MODEL_DIR, hf_path=HF_PATH, ctx=4096, num_chunks=9,
                    compute_unit=ct.ComputeUnit.CPU_ONLY)
engine.load()
bs = engine._prefill_bs

# Build prompt
prompt = (
    "Please translate the following passage into Chinese.\n\n"
    "In recent work on efficient transformer inference, a method referred to as "
    "Hierarchical Context Distillation has been proposed to address the growing "
    "cost of long-context processing."
)
messages = [{'role': 'user', 'content': prompt}]
tokens = engine._tokenize_messages(messages, enable_thinking=False)
target = bs + 35
while len(tokens) < target:
    prompt += " Additional context for longer input."
    messages = [{'role': 'user', 'content': prompt}]
    tokens = engine._tokenize_messages(messages, enable_thinking=False)
tokens = tokens[:target]
block1, tail = tokens[:bs], tokens[bs:]
print(f"Tokens: {len(tokens)}, block1: {len(block1)}, tail: {len(tail)}")

# ═══ Case A: 256 batch + 35 sequential ═══
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
for ti, tok_id in enumerate(tail[:-1]):
    engine._step_kv_only(tok_id, engine.pos)
    engine.pos += 1
engine._step(tail[-1], engine.pos)
engine.pos += 1
snap_A = {}
for ci in range(engine.num_chunks):
    snap_A[ci] = {}
    for sn in engine.kv_state_names:
        snap_A[ci][sn] = engine.states[ci].read_state(name=sn).copy()

# ═══ Case B: 256 batch + 35 batch ═══
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
_ = engine._batch_prefill(tail, engine.pos)
snap_B = {}
for ci in range(engine.num_chunks):
    snap_B[ci] = {}
    for sn in engine.kv_state_names:
        snap_B[ci][sn] = engine.states[ci].read_state(name=sn).copy()

# ═══ Compare KV cache in the WRITTEN region ═══
valid_end = bs + len(tail)  # position 291 (last valid)
batch_end = bs + bs          # position 512 (batch writes up to here)
print(f"\nKV cache write range: pos {bs}..{batch_end-1}")
print(f"Valid range:          pos {bs}..{valid_end-1}")
print(f"Garbage range:        pos {valid_end}..{batch_end-1}")

for ci in range(engine.num_chunks):
    for sn in engine.kv_state_names:
        kv_A = snap_A[ci][sn]
        kv_B = snap_B[ci][sn]
        # KV shape: (layers, heads, ctx, head_dim) or similar
        # Find the sequence axis (second-to-last)
        ndim = kv_A.ndim
        if ndim < 3:
            continue
        seq_axis = ndim - 2
        
        # Compare different regions
        def slice_region(arr, start, end):
            slc = [slice(None)] * ndim
            slc[seq_axis] = slice(start, end)
            return arr[tuple(slc)]
        
        # Valid region (positions 256..290)
        val_A = slice_region(kv_A, bs, valid_end)
        val_B = slice_region(kv_B, bs, valid_end)
        val_diff = np.abs(val_A.astype(np.float64) - val_B.astype(np.float64))
        val_max = float(np.max(val_diff)) if val_diff.size else 0
        val_mean = float(np.mean(val_diff)) if val_diff.size else 0
        
        # Garbage region (positions 291..511) — only exists in B
        garb_B = slice_region(kv_B, valid_end, batch_end)
        garb_norm = float(np.linalg.norm(garb_B.astype(np.float64)))
        garb_max = float(np.max(np.abs(garb_B.astype(np.float64)))) if garb_B.size else 0
        
        # In A, this region should be all zeros (never written)
        garb_A = slice_region(kv_A, valid_end, batch_end)
        garb_A_norm = float(np.linalg.norm(garb_A.astype(np.float64)))
        
        if val_max > 0.001 or garb_max > 0.001:
            print(f"\n  chunk{ci}/{sn}:")
            print(f"    Valid region ({bs}..{valid_end-1}): max_diff={val_max:.4f}, mean_diff={val_mean:.6f}")
            print(f"    Garbage region ({valid_end}..{batch_end-1}):")
            print(f"      In A (seq): norm={garb_A_norm:.4f}, max_abs={float(np.max(np.abs(garb_A.astype(np.float64)))):.4f}")
            print(f"      In B (batch): norm={garb_norm:.4f}, max_abs={garb_max:.4f}")

# ═══ Check: does the first decode token ATTEND to garbage positions? ═══
print(f"\n{'='*60}")
print(f"  First decode step after B: does causal mask include garbage?")
print(f"{'='*60}")
# After batch prefill, pos = 256+35 = 291
# When we call _step(next_tok, pos=291), the mask unmasks 0..291
# But batch wrote KV at 256..511, including garbage at 291..511
# The decode mask unmasks 0..291, so position 291 is right at the boundary.
# Garbage starts at 291 — the FIRST garbage position is exactly where
# the decode step writes its own KV entry!
# Actually no — after batch prefill, self.pos = bs + valid_len = 291.
# The decode step writes at pos=291 and unmasks 0..291.
# The garbage is at positions 292..511 which are NOT unmasked.
# So the causal mask should protect against garbage.
#
# BUT: the batch wrote KV entries at positions 256..511 (all 256 batch positions).
# These entries are computed from hidden states of padding tokens.
# Even though positions > 291 won't be attended to now, they persist in the cache.
# Future tokens at pos 292, 293, ... will gradually unmask more positions,
# but by then the positions are being overwritten by real tokens.
print(f"  After batch tail prefill, pos={bs + len(tail)}")
print(f"  First decode unmasks 0..{bs + len(tail)}")
print(f"  Garbage positions: {bs + len(tail)}..{bs + bs - 1}")
print(f"  These are BEYOND the unmasked range → causal mask PROTECTS")
print(f"  But garbage KV entries will be overwritten one-by-one as decode progresses")
print(f"  At pos=292, the decode step writes at 292 and unmasks 0..292")
print(f"  But the garbage at 292 was written by batch, so the decode step")
print(f"  overwrites it with correct data. Same for 293, 294, ...")
print(f"  → Garbage KV entries are overwritten before they can be attended to!")
print(f"\n  CONCLUSION: KV cache garbage is NOT the root cause")

# ═══ Check hidden state through final norm ═══
print(f"\n{'='*60}")
print(f"  Checking: does the final norm amplify differences?")
print(f"{'='*60}")
# Run block1+tail batch, collect hidden BEFORE and AFTER the final chunk
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)

# Batch tail, but capture hidden before lm_head
vl = len(tail)
input_ids = engine._batch_tok_buf
input_ids[0, :] = 0
input_ids[0, :vl] = tail
hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
hidden[:, vl:, :] = 0.0

mask = engine._batch_mask_buf
mask[:, :, :, :] = -65504.0
for i in range(vl):
    mask[0, 0, i, :bs + i + 1] = 0
for i in range(vl, bs):
    mask[0, 0, i, 0] = 0.0

pos_ids = engine._batch_pos_buf
pos_ids[:vl] = np.arange(bs + engine.rope_offset, bs + engine.rope_offset + vl, dtype=np.int32)
pos_ids[vl:] = 0
cur_pos = engine._batch_cur_buf
cur_pos[0] = bs
valid_len_arr = engine._valid_len_buf
valid_len_arr[0] = vl

# Check hidden norm at padding positions through each chunk
for ci in range(min(3, engine.num_chunks)):
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
    hidden_out = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
    
    # Check padding hidden norms BEFORE re-zero
    valid_h = hidden_out[:, vl-1:vl, :].astype(np.float64)
    pad_h = hidden_out[:, vl:, :].astype(np.float64)
    print(f"\n  chunk{ci} output (BEFORE re-zero):")
    print(f"    valid last-token norm: {np.linalg.norm(valid_h):.4f}")
    print(f"    padding mean norm:     {np.mean([np.linalg.norm(pad_h[:, i, :]) for i in range(pad_h.shape[1])]):.4f}")
    print(f"    padding max abs:       {np.max(np.abs(pad_h)):.4f}")
    
    # Re-zero
    hidden_out[:, vl:, :] = 0.0
    hidden = hidden_out
