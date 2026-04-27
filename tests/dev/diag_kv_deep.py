#!/usr/bin/env python3
"""Deep-dive into chunk3 k_cache write-position bug.

Confirmed: block2 batch prefill modifies chunk3 k_cache positions 0-255
while leaving positions 256-296 at zero. Hypothesis: k_cache is written
at position 0 instead of current_pos=256. This script verifies.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine
from tokenizers import Tokenizer

MODEL_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3'
HF_PATH = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'
COMPUTE = ct.ComputeUnit.CPU_AND_NE

tokenizer = Tokenizer.from_file(os.path.join(HF_PATH, 'tokenizer.json'))
engine = ChatEngine(MODEL_DIR, HF_PATH, ctx=4096, num_chunks=9, compute_unit=COMPUTE)
engine.load()
BS = engine._prefill_bs
print(f'BS={BS}', flush=True)

base = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
paragraph = 'Explain the theory of general relativity in simple terms. How does mass curve spacetime and what are the observable consequences for light, time, and gravity near massive objects? Discuss gravitational lensing, time dilation, and frame dragging. '
prompt = base
while True:
    enc = tokenizer.encode(prompt + '<|im_end|>\n<|im_start|>assistant\n')
    if len(enc.ids) > BS + 30:
        break
    prompt += paragraph
prompt += '<|im_end|>\n<|im_start|>assistant\n'
all_ids = tokenizer.encode(prompt).ids
block1_ids = all_ids[:BS]
block2_ids = all_ids[BS:]
valid_len2 = len(block2_ids)
print(f'block1={len(block1_ids)}, block2={valid_len2}', flush=True)

engine._reset_states()

# ---- Block1 batch prefill (full batch) ----
input_ids = engine._batch_tok_buf.copy()
input_ids[0, :] = 0
input_ids[0, :BS] = block1_ids
hidden = list(engine.embed_prefill.predict({'input_ids': input_ids}).values())[0]
mask = engine._batch_mask_buf.copy()
mask[:,:,:,:] = -65504.0
for i in range(BS):
    mask[0,0,i,:i+1] = 0
pos_ids = engine._batch_pos_buf.copy()
pos_ids[:BS] = np.arange(engine.rope_offset, engine.rope_offset+BS, dtype=np.int32)
cur_pos = engine._batch_cur_buf.copy()
cur_pos[0] = 0
vl = engine._valid_len_buf.copy()
vl[0] = BS

for ci in range(engine.num_chunks):
    inp = {'hidden_states': hidden.astype(np.float16), 'position_ids': pos_ids, 'causal_mask': mask, 'current_pos': cur_pos, 'linear_conv_state': engine.lin_convs[ci], 'linear_recurrent_state': engine.lin_recs[ci], 'valid_len': vl}
    out = engine.prefills[ci].predict(inp, state=engine.states[ci])
    hidden = out['output_hidden_states']
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']

# Snapshot all of chunk3 k/v cache after block1
k3_after_b1 = engine.states[3].read_state(name='k_cache').copy()
v3_after_b1 = engine.states[3].read_state(name='v_cache').copy()
print(f'\nchunk3 k_cache shape: {k3_after_b1.shape}', flush=True)
print(f'chunk3 v_cache shape: {v3_after_b1.shape}', flush=True)
print(f'After block1: k[0,:,0:5,0:4] sample:\n{k3_after_b1[0,:,:5,:4]}', flush=True)
print(f'After block1: k[0,:,250:260,0:4] sample:\n{k3_after_b1[0,:,250:260,:4]}', flush=True)

# ---- Block2 batch prefill (41 valid + padding) ----
input_ids2 = engine._batch_tok_buf.copy()
input_ids2[0,:] = 0
input_ids2[0,:valid_len2] = block2_ids
hidden2 = list(engine.embed_prefill.predict({'input_ids': input_ids2}).values())[0]
hidden2[:, valid_len2:, :] = 0.0
mask2 = engine._batch_mask_buf.copy()
mask2[:,:,:,:] = -65504.0
for i in range(valid_len2):
    mask2[0,0,i,:BS+i+1] = 0
for i in range(valid_len2, BS):
    mask2[0,0,i,0] = 0.0
pos_ids2 = engine._batch_pos_buf.copy()
pos_ids2[:valid_len2] = np.arange(BS+engine.rope_offset, BS+engine.rope_offset+valid_len2, dtype=np.int32)
pos_ids2[valid_len2:] = 0
cur_pos2 = engine._batch_cur_buf.copy()
cur_pos2[0] = BS
vl2 = engine._valid_len_buf.copy()
vl2[0] = valid_len2

for ci in range(engine.num_chunks):
    inp2 = {'hidden_states': hidden2.astype(np.float16), 'position_ids': pos_ids2, 'causal_mask': mask2, 'current_pos': cur_pos2, 'linear_conv_state': engine.lin_convs[ci], 'linear_recurrent_state': engine.lin_recs[ci], 'valid_len': vl2}
    out2 = engine.prefills[ci].predict(inp2, state=engine.states[ci])
    hidden2 = out2['output_hidden_states']
    if 'linear_conv_state_out' in out2:
        engine.lin_convs[ci] = out2['linear_conv_state_out']
        engine.lin_recs[ci] = out2['linear_recurrent_state_out']
    if valid_len2 < BS:
        hidden2[:, valid_len2:, :] = 0.0

k3_after_b2 = engine.states[3].read_state(name='k_cache').copy()
v3_after_b2 = engine.states[3].read_state(name='v_cache').copy()

print(f'\n=== chunk3 k_cache analysis ===', flush=True)
print(f'After block2: k[0,:,0:5,0:4] sample:\n{k3_after_b2[0,:,:5,:4]}', flush=True)
print(f'After block2: k[0,:,250:260,0:4] sample (boundary):\n{k3_after_b2[0,:,250:260,:4]}', flush=True)
print(f'After block2: k[0,:,256:261,0:4] (should be block2 valid):\n{k3_after_b2[0,:,256:261,:4]}', flush=True)

# Check: did block2 data land at position 0?
# If write went to pos 0, then positions 0:valid_len2 should have block2's keys
# and positions valid_len2:BS should be zero (padding).
print(f'\n=== Hypothesis: k_cache write went to pos 0 instead of pos {BS} ===', flush=True)
# Check if positions 0-40 changed (these were block1's keys)
k_b1_first41 = k3_after_b1[0,:,:valid_len2,:]
k_b2_first41 = k3_after_b2[0,:,:valid_len2,:]
print(f'Positions 0-{valid_len2-1} changed? {not np.array_equal(k_b1_first41, k_b2_first41)}', flush=True)
print(f'  diff norm: {np.linalg.norm((k_b2_first41-k_b1_first41).astype(np.float64)):.4f}', flush=True)

# Check if positions valid_len2:BS are now zero (padding overwrote block1)
k_b2_padding_in_b1 = k3_after_b2[0,:,valid_len2:BS,:]
print(f'Positions {valid_len2}-{BS-1} all zero? {np.all(k_b2_padding_in_b1 == 0)}', flush=True)
print(f'  norm: {np.linalg.norm(k_b2_padding_in_b1.astype(np.float64)):.4f}', flush=True)

# Compare with block1's original data at those positions
k_b1_padding_range = k3_after_b1[0,:,valid_len2:BS,:]
print(f'  (was norm {np.linalg.norm(k_b1_padding_range.astype(np.float64)):.4f} in block1)', flush=True)

# v_cache control: same analysis
print(f'\n=== chunk3 v_cache control ===', flush=True)
v_b1_first41 = v3_after_b1[0,:,:valid_len2,:]
v_b2_first41 = v3_after_b2[0,:,:valid_len2,:]
print(f'Positions 0-{valid_len2-1} changed? {not np.array_equal(v_b1_first41, v_b2_first41)}', flush=True)
v_b2_at_block2 = v3_after_b2[0,:,BS:BS+valid_len2,:]
print(f'Positions {BS}-{BS+valid_len2-1} (block2 valid) norm: {np.linalg.norm(v_b2_at_block2.astype(np.float64)):.4f}', flush=True)

# Also check: how many F layers does chunk3 have?
print(f'\n=== chunk3 cache dimensions ===', flush=True)
print(f'k_cache shape: {k3_after_b2.shape} (dim0=num_kv_layers)', flush=True)
for layer_idx in range(k3_after_b2.shape[0]):
    b1_norm = np.linalg.norm(k3_after_b1[layer_idx].astype(np.float64))
    b2_norm = np.linalg.norm(k3_after_b2[layer_idx].astype(np.float64))
    changed = not np.array_equal(k3_after_b1[layer_idx,:,:BS,:], k3_after_b2[layer_idx,:,:BS,:])
    print(f'  layer{layer_idx}: b1_norm={b1_norm:.2f} b2_norm={b2_norm:.2f} block1_modified={changed}', flush=True)

# For the modified layer(s), check ALL chunks to see if the bug is chunk3-specific
print(f'\n=== All chunks: num_kv_layers and block1 integrity ===', flush=True)
for ci in range(engine.num_chunks):
    k = engine.states[ci].read_state(name='k_cache')
    print(f'  chunk{ci} k_cache shape: {k.shape}', flush=True)

print("\nDone.", flush=True)
