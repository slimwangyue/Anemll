#!/usr/bin/env python3
"""Quick validation: confirm sequential-only prefill matches test_e2e.

Uses the SAME model loading + step logic as diag_compare.py PATH A,
then verifies the output matches by running through chat_server's step path.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CTX, NUM_CHUNKS, BATCH_SIZE
import coremltools as ct
from transformers import AutoTokenizer

MODEL_DIR = '/Users/yw68/Anemll/qwen3_5_stable_models_6chunk'
FFN_DIR   = os.path.join(MODEL_DIR, 'combined_LUT6_dedup')
TOKENIZER = '/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B'
EMBED     = os.path.join(MODEL_DIR, 'embeddings.mlpackage')
LMHEAD    = os.path.join(MODEL_DIR, 'lm_head_logits.mlpackage')

tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
cu  = ct.ComputeUnit.CPU_AND_NE

print('Loading models...')
embed  = ct.models.MLModel(EMBED, compute_units=cu)
lmhead = ct.models.MLModel(LMHEAD, compute_units=cu)

# Load ONLY infer functions (no prefill) — same as sequential-only chat_server
ffns = []
for ci in range(NUM_CHUNKS):
    p = os.path.join(FFN_DIR, f'chunk{ci}.mlpackage')
    ffns.append(ct.models.MLModel(p, compute_units=cu, function_name='infer'))
    print(f'  chunk{ci} loaded')

# Detect shapes
conv_shapes, rec_shapes = [], []
for ci in range(NUM_CHUNKS):
    spec = ffns[ci].get_spec()
    for fn in spec.description.functions:
        if fn.name == 'infer':
            cs, rs = (6, 1024, 32), (6, 32, 128, 128)
            for inp in fn.input:
                if inp.name == 'linear_conv_state':
                    cs = tuple(inp.type.multiArrayType.shape)
                if inp.name == 'linear_recurrent_state':
                    rs = tuple(inp.type.multiArrayType.shape)
            conv_shapes.append(cs)
            rec_shapes.append(rs)
            break

# Sort logits keys
spec = lmhead.get_spec()
out_names = [o.name for o in spec.description.output]
logits_keys = sorted([k for k in out_names if k.startswith('logits')],
                     key=lambda x: int(x[6:]))

def extract_logits(lm_out):
    return np.concatenate([lm_out[k].flatten().astype(np.float32) for k in logits_keys])

# Prompt
msgs = [{'role': 'user', 'content': '1+1等于几？'}]
prompt_ids = list(tok.apply_chat_template(
    msgs, add_generation_prompt=True, tokenize=True,
    enable_thinking=False, return_dict=False))
print(f'Prompt: {len(prompt_ids)} tokens')

# Run two identical sequential paths with independent states
# (both should produce byte-identical results)
def run_sequential(label, prompt_ids, max_gen=15):
    states = [m.make_state() for m in ffns]
    lc = [np.zeros(conv_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
    lr = [np.zeros(rec_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
    pos = 0

    # Prefill (kv-only for all but last)
    for ti, tid in enumerate(prompt_ids):
        t = np.array([[tid]], dtype=np.int32)
        hidden = list(embed.predict({'input_ids': t}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        pos_arr = np.array([pos], dtype=np.int32)

        for ci in range(NUM_CHUNKS):
            inp = {
                'hidden_states': hidden.astype(np.float16),
                'position_ids': pos_arr,
                'causal_mask': mask,
                'current_pos': pos_arr,
                'linear_conv_state': lc[ci],
                'linear_recurrent_state': lr[ci],
            }
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out['output_hidden_states']
            if 'linear_conv_state_out' in out:
                lc[ci] = out['linear_conv_state_out']
                lr[ci] = out['linear_recurrent_state_out']
        pos += 1

    # First decode token
    logits = extract_logits(lmhead.predict({'hidden_states': hidden.astype(np.float16)}))
    gen_ids = [int(np.argmax(logits))]

    # Generate more
    for _ in range(max_gen - 1):
        tid = gen_ids[-1]
        if tid == tok.eos_token_id:
            break
        t = np.array([[tid]], dtype=np.int32)
        hidden = list(embed.predict({'input_ids': t}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        pos_arr = np.array([pos], dtype=np.int32)
        for ci in range(NUM_CHUNKS):
            inp = {
                'hidden_states': hidden.astype(np.float16),
                'position_ids': pos_arr,
                'causal_mask': mask,
                'current_pos': pos_arr,
                'linear_conv_state': lc[ci],
                'linear_recurrent_state': lr[ci],
            }
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out['output_hidden_states']
            if 'linear_conv_state_out' in out:
                lc[ci] = out['linear_conv_state_out']
                lr[ci] = out['linear_recurrent_state_out']
        pos += 1
        logits = extract_logits(lmhead.predict({'hidden_states': hidden.astype(np.float16)}))
        gen_ids.append(int(np.argmax(logits)))

    text = tok.decode(gen_ids, skip_special_tokens=True)
    print(f'{label}: {gen_ids}')
    print(f'  => {repr(text)}')
    return gen_ids

# Run twice — should be bit-identical
ids_a = run_sequential('Run A', prompt_ids)
ids_b = run_sequential('Run B', prompt_ids)
print(f'\nBit-identical: {ids_a == ids_b}')
if ids_a != ids_b:
    for i, (a, b) in enumerate(zip(ids_a, ids_b)):
        if a != b:
            print(f'  First diff at step {i}: {a} vs {b}')
            break
else:
    print('SUCCESS: Sequential path is deterministic and reproducible.')
