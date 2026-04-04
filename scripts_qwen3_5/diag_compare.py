#!/usr/bin/env python3
"""Diagnostic: compare sequential vs batch prefill paths token-by-token."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CTX, NUM_CHUNKS
import coremltools as ct
from transformers import AutoTokenizer

from config import CTX, NUM_CHUNKS, BATCH_SIZE  # BATCH_SIZE=512 (compiled model size)
MODEL_DIR = '/Users/yw68/Anemll/qwen3_5_stable_models_6chunk'
FFN_DIR = os.path.join(MODEL_DIR, 'combined_LUT6_dedup')
TOKENIZER = '/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B'
EMBED = os.path.join(MODEL_DIR, 'embeddings.mlpackage')
LMHEAD = os.path.join(MODEL_DIR, 'lm_head_logits.mlpackage')

tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
cu = ct.ComputeUnit.CPU_AND_NE

print('Loading models...')
embed = ct.models.MLModel(EMBED, compute_units=cu)
lmhead = ct.models.MLModel(LMHEAD, compute_units=cu)

ffns_infer = []
ffns_prefill = []
for ci in range(NUM_CHUNKS):
    p = os.path.join(FFN_DIR, f'chunk{ci}.mlpackage')
    ffns_infer.append(ct.models.MLModel(p, compute_units=cu, function_name='infer'))
    ffns_prefill.append(ct.models.MLModel(p, compute_units=cu, function_name='prefill'))
    print(f'  chunk{ci} loaded')

# Detect shapes
conv_shapes = []
rec_shapes = []
for ci in range(NUM_CHUNKS):
    spec = ffns_infer[ci].get_spec()
    for fn in spec.description.functions:
        if fn.name == 'infer':
            for inp in fn.input:
                if inp.name == 'linear_conv_state':
                    conv_shapes.append(tuple(inp.type.multiArrayType.shape))
                elif inp.name == 'linear_recurrent_state':
                    rec_shapes.append(tuple(inp.type.multiArrayType.shape))
            break

# Sort logits keys
spec = lmhead.get_spec()
out_names = [o.name for o in spec.description.output]
logits_keys = sorted([k for k in out_names if k.startswith('logits')],
                     key=lambda x: int(x[6:]))
print(f'Split logits: {len(logits_keys)}-way')

# Prompt
msgs = [{'role': 'user', 'content': '1+1等于几？'}]
prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      enable_thinking=False, return_dict=False)
prompt_ids = list(prompt_ids)
print(f'Prompt: {len(prompt_ids)} tokens: {prompt_ids}')

def extract_logits(lm_out):
    parts = [lm_out[k].flatten().astype(np.float32) for k in logits_keys]
    return np.concatenate(parts)

# ========== PATH A: Sequential (test_e2e style) ==========
print('\n' + '='*60)
print('PATH A: Sequential prefill (test_e2e style)')
print('='*60)

states_a = [m.make_state() for m in ffns_infer]
lc_a = [np.zeros(conv_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
lr_a = [np.zeros(rec_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
pos_a = 0

for ti, tid in enumerate(prompt_ids):
    t = np.array([[tid]], dtype=np.int32)
    hidden = list(embed.predict({'input_ids': t}).values())[0]
    mask = np.full((1,1,1,CTX), -65504.0, dtype=np.float16)
    mask[:,:,:,:pos_a+1] = 0
    pos_arr = np.array([pos_a], dtype=np.int32)

    for ci in range(NUM_CHUNKS):
        inp = {
            'hidden_states': hidden.astype(np.float16),
            'position_ids': pos_arr,
            'causal_mask': mask,
            'current_pos': pos_arr,
            'linear_conv_state': lc_a[ci],
            'linear_recurrent_state': lr_a[ci],
        }
        out = ffns_infer[ci].predict(inp, state=states_a[ci])
        hidden = out['output_hidden_states']
        if 'linear_conv_state_out' in out:
            lc_a[ci] = out['linear_conv_state_out']
            lr_a[ci] = out['linear_recurrent_state_out']
    pos_a += 1

hidden_a = hidden.astype(np.float16)
lm_out_a = lmhead.predict({'hidden_states': hidden_a})
logits_a = extract_logits(lm_out_a)
tok_a = int(np.argmax(logits_a))
print(f'Last hidden shape: {hidden_a.shape}, '
      f'min={hidden_a.min():.4f}, max={hidden_a.max():.4f}')
print(f'Logits shape={logits_a.shape}, '
      f'min={logits_a.min():.4f}, max={logits_a.max():.4f}')
print(f'First token: {tok_a} = {repr(tok.decode([tok_a]))}')
top5a = np.argsort(-logits_a)[:5].tolist()
print(f'Top-5: {[(t, f"{logits_a[t]:.2f}") for t in top5a]}')
print(f'Hidden[0,:5]: {hidden_a.flatten()[:5]}')

# ========== PATH B: Batch prefill (chat_server style) ==========
print('\n' + '='*60)
print('PATH B: Batch prefill (chat_server style)')
print('='*60)

# IMPORTANT: prefill uses shared states with infer. Must use infer model's state.
states_b = [m.make_state() for m in ffns_infer]
lc_b = [np.zeros(conv_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
lr_b = [np.zeros(rec_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]

valid_len = len(prompt_ids)
input_ids_batch = np.zeros((1, BATCH_SIZE), dtype=np.int32)
input_ids_batch[0, :valid_len] = prompt_ids

hidden_b = list(embed.predict({'input_ids': input_ids_batch}).values())[0]
print(f'Embed output shape: {hidden_b.shape}')

mask_b = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
for i in range(valid_len):
    mask_b[0, 0, i, :i+1] = 0

pos_ids_b = np.zeros(BATCH_SIZE, dtype=np.int32)
pos_ids_b[:valid_len] = np.arange(valid_len, dtype=np.int32)

cur_pos_b = np.array([0], dtype=np.int32)
valid_len_arr = np.array([valid_len], dtype=np.int32)

for ci in range(NUM_CHUNKS):
    inp = {
        'hidden_states': hidden_b.astype(np.float16),
        'position_ids': pos_ids_b,
        'causal_mask': mask_b,
        'current_pos': cur_pos_b,
        'linear_conv_state': lc_b[ci],
        'linear_recurrent_state': lr_b[ci],
        'valid_len': valid_len_arr,
    }
    out = ffns_prefill[ci].predict(inp, state=states_b[ci])
    hidden_b = out['output_hidden_states']
    if 'linear_conv_state_out' in out:
        lc_b[ci] = out['linear_conv_state_out']
        lr_b[ci] = out['linear_recurrent_state_out']

print(f'Prefill output shape: {hidden_b.shape}')
# Model may extract last-token internally (returns [1,1,2560]) or return full batch [1,512,2560]
if hidden_b.ndim >= 3 and hidden_b.shape[1] > 1:
    hidden_last_b = hidden_b[:, valid_len-1:valid_len, :].astype(np.float16)
else:
    hidden_last_b = hidden_b.astype(np.float16)
print(f'Last hidden shape: {hidden_last_b.shape}, '
      f'min={hidden_last_b.min():.4f}, max={hidden_last_b.max():.4f}')

lm_out_b = lmhead.predict({'hidden_states': hidden_last_b})
logits_b = extract_logits(lm_out_b)
tok_b = int(np.argmax(logits_b))
print(f'Logits shape={logits_b.shape}, '
      f'min={logits_b.min():.4f}, max={logits_b.max():.4f}')
print(f'First token: {tok_b} = {repr(tok.decode([tok_b]))}')
top5b = np.argsort(-logits_b)[:5].tolist()
print(f'Top-5: {[(t, f"{logits_b[t]:.2f}") for t in top5b]}')
print(f'Hidden[0,:5]: {hidden_last_b.flatten()[:5]}')

# ========== COMPARISON ==========
print('\n' + '='*60)
print('COMPARISON: Sequential vs Batch')
print('='*60)
hid_diff = np.abs(hidden_a.flatten().astype(np.float32) -
                  hidden_last_b.flatten().astype(np.float32))
log_diff = np.abs(logits_a - logits_b)
print(f'Hidden diff: max={hid_diff.max():.6f}, mean={hid_diff.mean():.6f}')
print(f'Logits diff: max={log_diff.max():.4f}, mean={log_diff.mean():.6f}')
print(f'Token A={tok_a} B={tok_b} MATCH={tok_a==tok_b}')
print(f'Top-10 A: {np.argsort(-logits_a)[:10].tolist()}')
print(f'Top-10 B: {np.argsort(-logits_b)[:10].tolist()}')
print(f'Top-10 MATCH: {np.argsort(-logits_a)[:10].tolist() == np.argsort(-logits_b)[:10].tolist()}')

# Generate 10 more tokens from each
print('\nGenerating 10 decode tokens from each...')
gen_a = [tok_a]
for _ in range(10):
    t = np.array([[gen_a[-1]]], dtype=np.int32)
    h = list(embed.predict({'input_ids': t}).values())[0]
    m = np.full((1,1,1,CTX), -65504.0, dtype=np.float16)
    m[:,:,:,:pos_a+1] = 0
    p = np.array([pos_a], dtype=np.int32)
    for ci in range(NUM_CHUNKS):
        inp = {'hidden_states': h.astype(np.float16), 'position_ids': p,
               'causal_mask': m, 'current_pos': p,
               'linear_conv_state': lc_a[ci], 'linear_recurrent_state': lr_a[ci]}
        out = ffns_infer[ci].predict(inp, state=states_a[ci])
        h = out['output_hidden_states']
        if 'linear_conv_state_out' in out:
            lc_a[ci] = out['linear_conv_state_out']
            lr_a[ci] = out['linear_recurrent_state_out']
    pos_a += 1
    logits = extract_logits(lmhead.predict({'hidden_states': h.astype(np.float16)}))
    gen_a.append(int(np.argmax(logits)))

pos_b = valid_len
gen_b = [tok_b]
for _ in range(10):
    t = np.array([[gen_b[-1]]], dtype=np.int32)
    h = list(embed.predict({'input_ids': t}).values())[0]
    m = np.full((1,1,1,CTX), -65504.0, dtype=np.float16)
    m[:,:,:,:pos_b+1] = 0
    p = np.array([pos_b], dtype=np.int32)
    for ci in range(NUM_CHUNKS):
        inp = {'hidden_states': h.astype(np.float16), 'position_ids': p,
               'causal_mask': m, 'current_pos': p,
               'linear_conv_state': lc_b[ci], 'linear_recurrent_state': lr_b[ci]}
        out = ffns_infer[ci].predict(inp, state=states_b[ci])
        h = out['output_hidden_states']
        if 'linear_conv_state_out' in out:
            lc_b[ci] = out['linear_conv_state_out']
            lr_b[ci] = out['linear_recurrent_state_out']
    pos_b += 1
    logits = extract_logits(lmhead.predict({'hidden_states': h.astype(np.float16)}))
    gen_b.append(int(np.argmax(logits)))

print(f'Sequential: {gen_a}')
print(f'  => {repr(tok.decode(gen_a))}')
print(f'Batch:      {gen_b}')
print(f'  => {repr(tok.decode(gen_b))}')
print(f'Token MATCH: {gen_a == gen_b}')
if gen_a != gen_b:
    for i, (a, b) in enumerate(zip(gen_a, gen_b)):
        if a != b:
            print(f'  First divergence at decode step {i}: '
                  f'A={a}({repr(tok.decode([a]))}) vs B={b}({repr(tok.decode([b]))})')
            break
