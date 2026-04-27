#!/usr/bin/env python3
"""Check if block2 batch prefill correctly preserves block1's KV cache."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine
from tokenizers import Tokenizer

MODEL_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.4_fix'
HF_PATH = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'
COMPUTE = ct.ComputeUnit.CPU_AND_NE

tokenizer = Tokenizer.from_file(os.path.join(HF_PATH, 'tokenizer.json'))
engine = ChatEngine(MODEL_DIR, HF_PATH, ctx=4096, num_chunks=9, compute_unit=COMPUTE)
engine.load()
BS = engine._prefill_bs
print(f'Using BS={BS}', flush=True)

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

# Block1 batch prefill (full batch, no padding)
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

print("Running block1 prefill...", flush=True)
for ci in range(engine.num_chunks):
    inp = {'hidden_states': hidden.astype(np.float16), 'position_ids': pos_ids, 'causal_mask': mask, 'current_pos': cur_pos, 'linear_conv_state': engine.lin_convs[ci], 'linear_recurrent_state': engine.lin_recs[ci], 'valid_len': vl}
    out = engine.prefills[ci].predict(inp, state=engine.states[ci])
    hidden = out['output_hidden_states']
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
engine.pos = BS

# Snapshot KV after block1
kv_b1 = {}
for ci in range(engine.num_chunks):
    kv_b1[ci] = {}
    for sn in engine.kv_state_names:
        kv_b1[ci][sn] = engine.states[ci].read_state(name=sn).copy()
print("Block1 done, KV snapshot taken.", flush=True)

# Block2 batch prefill (41 tokens, padded to 256)
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

print("Running block2 prefill...", flush=True)
for ci in range(engine.num_chunks):
    inp2 = {'hidden_states': hidden2.astype(np.float16), 'position_ids': pos_ids2, 'causal_mask': mask2, 'current_pos': cur_pos2, 'linear_conv_state': engine.lin_convs[ci], 'linear_recurrent_state': engine.lin_recs[ci], 'valid_len': vl2}
    out2 = engine.prefills[ci].predict(inp2, state=engine.states[ci])
    hidden2 = out2['output_hidden_states']
    if 'linear_conv_state_out' in out2:
        engine.lin_convs[ci] = out2['linear_conv_state_out']
        engine.lin_recs[ci] = out2['linear_recurrent_state_out']
    if valid_len2 < BS:
        hidden2[:, valid_len2:, :] = 0.0
print("Block2 done.", flush=True)

print()
print('=== Check 1: Did block2 modify block1 KV (pos 0-255)? ===', flush=True)
for ci in range(engine.num_chunks):
    for sn in engine.kv_state_names:
        before = kv_b1[ci][sn]
        after = engine.states[ci].read_state(name=sn)
        if before.ndim == 4:
            b = before[:,:,:BS,:]
            a = after[:,:,:BS,:]
        else:
            b, a = before, after
        eq = np.array_equal(b, a)
        if not eq:
            d = np.abs(b.astype(np.float64)-a.astype(np.float64))
            print(f'  chunk{ci} {sn}: MODIFIED! max={d.max():.6f} changed={np.count_nonzero(d>0)}', flush=True)
        else:
            print(f'  chunk{ci} {sn}: unchanged', flush=True)

print()
print('=== Check 2: Padding KV (pos 297-511) zero? ===', flush=True)
for ci in range(engine.num_chunks):
    for sn in engine.kv_state_names:
        kv = engine.states[ci].read_state(name=sn)
        if kv.ndim == 4:
            pad = kv[:,:,BS+valid_len2:BS+BS,:]
            mx = float(np.abs(pad).max())
            nz = int(np.count_nonzero(pad))
            print(f'  chunk{ci} {sn}: max_abs={mx:.6f} nonzero={nz}/{pad.size}', flush=True)

print()
print('=== Check 3: Valid KV norms ===', flush=True)
for ci in range(engine.num_chunks):
    for sn in engine.kv_state_names:
        kv = engine.states[ci].read_state(name=sn)
        if kv.ndim == 4:
            valid_part = kv[:,:,BS:BS+valid_len2,:]
            norm = float(np.linalg.norm(valid_part.astype(np.float64)))
            block1_part = kv[:,:,:BS,:]
            norm1 = float(np.linalg.norm(block1_part.astype(np.float64)))
            print(f'  chunk{ci} {sn}: block1_norm={norm1:.2f} block2_valid_norm={norm:.2f}', flush=True)

print("\nDone.", flush=True)
