#!/usr/bin/env python3
"""Chunk-by-chunk comparison: sequential infer vs batch prefill.

For each chunk, compares:
  - output_hidden_states (last-token for seq, token[valid_len-1] for batch)
  - linear_conv_state_out
  - linear_recurrent_state_out
Identifies the FIRST chunk and FIRST tensor where divergence appears.
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

tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
cu  = ct.ComputeUnit.CPU_AND_NE

# ── Load models ──────────────────────────────────────────────────
print('Loading models...')
embed = ct.models.MLModel(EMBED, compute_units=cu)

ffns_infer   = []
ffns_prefill = []
for ci in range(NUM_CHUNKS):
    p = os.path.join(FFN_DIR, f'chunk{ci}.mlpackage')
    ffns_infer.append(ct.models.MLModel(p, compute_units=cu, function_name='infer'))
    ffns_prefill.append(ct.models.MLModel(p, compute_units=cu, function_name='prefill'))
    print(f'  chunk{ci} loaded')

# ── Detect per-chunk shapes ──────────────────────────────────────
conv_shapes, rec_shapes = [], []
for ci in range(NUM_CHUNKS):
    spec = ffns_infer[ci].get_spec()
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
    print(f'  chunk{ci}: conv={conv_shapes[-1]}, rec={rec_shapes[-1]}')

# ── Prompt ───────────────────────────────────────────────────────
msgs = [{'role': 'user', 'content': '1+1等于几？'}]
prompt_ids = list(tok.apply_chat_template(
    msgs, add_generation_prompt=True, tokenize=True,
    enable_thinking=False, return_dict=False))
valid_len = len(prompt_ids)
print(f'\nPrompt: {valid_len} tokens: {prompt_ids}')

# ── Helper ───────────────────────────────────────────────────────
def tensor_stats(name, a, b):
    """Compare two tensors, return dict of stats."""
    af = a.flatten().astype(np.float32)
    bf = b.flatten().astype(np.float32)
    diff = np.abs(af - bf)
    mx = float(diff.max())
    mn = float(diff.mean())
    cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    return {'name': name, 'max_diff': mx, 'mean_diff': mn, 'cosine': cos,
            'a_range': (float(a.min()), float(a.max())),
            'b_range': (float(b.min()), float(b.max()))}

# ══════════════════════════════════════════════════════════════════
# PATH A: Sequential infer — process all tokens one-by-one,
# capture hidden/state after every chunk at the LAST token.
# ══════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('Running PATH A: sequential infer (all tokens)...')
print('='*70)

states_a = [m.make_state() for m in ffns_infer]
lc_a = [np.zeros(conv_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
lr_a = [np.zeros(rec_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]

# We need per-chunk outputs only for the LAST token of the prompt
seq_hidden_after = {}  # ci -> hidden after chunk ci for last token
seq_conv_after   = {}  # ci -> conv state after processing last token
seq_rec_after    = {}  # ci -> rec state after processing last token

for ti, tid in enumerate(prompt_ids):
    t = np.array([[tid]], dtype=np.int32)
    hidden = list(embed.predict({'input_ids': t}).values())[0]
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :ti + 1] = 0
    pos_arr = np.array([ti], dtype=np.int32)

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

        # Capture after last prompt token
        if ti == valid_len - 1:
            seq_hidden_after[ci] = hidden.copy()
            seq_conv_after[ci]   = lc_a[ci].copy()
            seq_rec_after[ci]    = lr_a[ci].copy()

    if ti % 5 == 0 or ti == valid_len - 1:
        print(f'  token {ti}/{valid_len-1}', flush=True)

print('Sequential prefill done.')

# ══════════════════════════════════════════════════════════════════
# PATH B: Batch prefill — one call per chunk, capture after each
# ══════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('Running PATH B: batch prefill (one pass per chunk)...')
print('='*70)

states_b = [m.make_state() for m in ffns_infer]  # fresh states from infer model
lc_b = [np.zeros(conv_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]
lr_b = [np.zeros(rec_shapes[ci], dtype=np.float16) for ci in range(NUM_CHUNKS)]

# Batch embedding
input_ids_batch = np.zeros((1, BATCH_SIZE), dtype=np.int32)
input_ids_batch[0, :valid_len] = prompt_ids
hidden_b = list(embed.predict({'input_ids': input_ids_batch}).values())[0]

# Also get sequential embedding of last token for comparison
last_tok_embed = list(embed.predict(
    {'input_ids': np.array([[prompt_ids[-1]]], dtype=np.int32)}).values())[0]
batch_last_embed = hidden_b[:, valid_len-1:valid_len, :]
embed_diff = np.abs(last_tok_embed.flatten().astype(np.float32) -
                    batch_last_embed.flatten().astype(np.float32))
print(f'Embed last-token diff: max={embed_diff.max():.8f} mean={embed_diff.mean():.8f}')

# Causal mask
mask_b = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
for i in range(valid_len):
    mask_b[0, 0, i, :i + 1] = 0

pos_ids_b = np.zeros(BATCH_SIZE, dtype=np.int32)
pos_ids_b[:valid_len] = np.arange(valid_len, dtype=np.int32)

cur_pos_b  = np.array([0], dtype=np.int32)
valid_arr  = np.array([valid_len], dtype=np.int32)

bat_hidden_after = {}
bat_conv_after   = {}
bat_rec_after    = {}

for ci in range(NUM_CHUNKS):
    inp = {
        'hidden_states':          hidden_b.astype(np.float16),
        'position_ids':           pos_ids_b,
        'causal_mask':            mask_b,
        'current_pos':            cur_pos_b,
        'linear_conv_state':      lc_b[ci],
        'linear_recurrent_state': lr_b[ci],
        'valid_len':              valid_arr,
    }
    out = ffns_prefill[ci].predict(inp, state=states_b[ci])
    hidden_b = out['output_hidden_states']
    if 'linear_conv_state_out' in out:
        lc_b[ci] = out['linear_conv_state_out']
        lr_b[ci] = out['linear_recurrent_state_out']

    bat_hidden_after[ci] = hidden_b.copy()
    bat_conv_after[ci]   = lc_b[ci].copy()
    bat_rec_after[ci]    = lr_b[ci].copy()
    print(f'  chunk{ci} done, hidden shape={hidden_b.shape}', flush=True)

print('Batch prefill done.')

# ══════════════════════════════════════════════════════════════════
# CHUNK-BY-CHUNK COMPARISON
# ══════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('CHUNK-BY-CHUNK COMPARISON')
print('='*70)

first_diverge_chunk = None
first_diverge_tensor = None
TOLERANCE = 0.01  # fp16 noise threshold

for ci in range(NUM_CHUNKS):
    print(f'\n── chunk{ci} ────────────────────────────────────')

    # Hidden states: seq is [1,1,2560]; batch is [1,512,2560] for ci<5, [1,1,2560] for ci=5
    h_seq = seq_hidden_after[ci]  # (1, 1, 2560)
    h_bat_raw = bat_hidden_after[ci]
    if h_bat_raw.ndim >= 3 and h_bat_raw.shape[1] > 1:
        # Extract last valid token
        h_bat = h_bat_raw[:, valid_len-1:valid_len, :]
    else:
        h_bat = h_bat_raw

    s = tensor_stats('hidden_states', h_seq, h_bat)
    flag = ' *** DIVERGED ***' if s['max_diff'] > TOLERANCE else ''
    print(f"  hidden_states: max_diff={s['max_diff']:.6f}  mean={s['mean_diff']:.6f}  "
          f"cos={s['cosine']:.8f}{flag}")
    print(f"    seq range: [{s['a_range'][0]:.4f}, {s['a_range'][1]:.4f}]  "
          f"bat range: [{s['b_range'][0]:.4f}, {s['b_range'][1]:.4f}]")
    if s['max_diff'] > TOLERANCE and first_diverge_chunk is None:
        first_diverge_chunk = ci
        first_diverge_tensor = 'hidden_states'

    # Conv state
    c_seq = seq_conv_after[ci]
    c_bat = bat_conv_after[ci]
    s = tensor_stats('conv_state', c_seq, c_bat)
    flag = ' *** DIVERGED ***' if s['max_diff'] > TOLERANCE else ''
    print(f"  conv_state:    max_diff={s['max_diff']:.6f}  mean={s['mean_diff']:.6f}  "
          f"cos={s['cosine']:.8f}{flag}")
    if s['max_diff'] > TOLERANCE and first_diverge_chunk is None:
        first_diverge_chunk = ci
        first_diverge_tensor = 'conv_state'

    # Rec state
    r_seq = seq_rec_after[ci]
    r_bat = bat_rec_after[ci]
    s = tensor_stats('rec_state', r_seq, r_bat)
    flag = ' *** DIVERGED ***' if s['max_diff'] > TOLERANCE else ''
    print(f"  rec_state:     max_diff={s['max_diff']:.6f}  mean={s['mean_diff']:.6f}  "
          f"cos={s['cosine']:.8f}{flag}")
    if s['max_diff'] > TOLERANCE and first_diverge_chunk is None:
        first_diverge_chunk = ci
        first_diverge_tensor = 'rec_state'

    # Also check if batch hidden has NaN/inf in padding region
    if h_bat_raw.ndim >= 3 and h_bat_raw.shape[1] > 1:
        pad_region = h_bat_raw[:, valid_len:, :]
        has_nan = np.any(np.isnan(pad_region))
        has_inf = np.any(np.isinf(pad_region))
        pad_max = float(np.abs(pad_region).max()) if pad_region.size > 0 else 0
        print(f"  batch padding region: nan={has_nan}, inf={has_inf}, absmax={pad_max:.4f}")

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
if first_diverge_chunk is not None:
    print(f'FIRST DIVERGENCE: chunk{first_diverge_chunk}, tensor={first_diverge_tensor}')
    print(f'  All chunks before chunk{first_diverge_chunk} matched within tolerance={TOLERANCE}')
else:
    print(f'No divergence found above tolerance={TOLERANCE}!')

# Show per-layer conv/rec breakdown for the first diverging chunk
if first_diverge_chunk is not None:
    ci = first_diverge_chunk
    n_layers = conv_shapes[ci][0]  # first dim = layer count
    print(f'\nPer-layer breakdown for chunk{ci} ({n_layers} layers):')

    c_seq = seq_conv_after[ci]
    c_bat = bat_conv_after[ci]
    r_seq = seq_rec_after[ci]
    r_bat = bat_rec_after[ci]

    for li in range(n_layers):
        cd = np.abs(c_seq[li].flatten().astype(np.float32) -
                    c_bat[li].flatten().astype(np.float32))
        rd = np.abs(r_seq[li].flatten().astype(np.float32) -
                    r_bat[li].flatten().astype(np.float32))
        cc = np.dot(c_seq[li].flatten().astype(np.float32),
                    c_bat[li].flatten().astype(np.float32)) / (
                    np.linalg.norm(c_seq[li].flatten().astype(np.float32)) *
                    np.linalg.norm(c_bat[li].flatten().astype(np.float32)) + 1e-12)
        rc = np.dot(r_seq[li].flatten().astype(np.float32),
                    r_bat[li].flatten().astype(np.float32)) / (
                    np.linalg.norm(r_seq[li].flatten().astype(np.float32)) *
                    np.linalg.norm(r_bat[li].flatten().astype(np.float32)) + 1e-12)
        print(f'  layer {li}: conv max={cd.max():.6f} mean={cd.mean():.6f} cos={cc:.6f}'
              f'  |  rec max={rd.max():.6f} mean={rd.mean():.6f} cos={rc:.6f}')
