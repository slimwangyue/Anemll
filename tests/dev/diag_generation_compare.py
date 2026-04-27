#!/usr/bin/env python3
"""
Compare full generation from Path A (batch tail prefill) vs Path B (sequential tail prefill).
Generates N tokens from each path and compares the output text.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine

MODEL_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3'
HF_PATH   = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'
COMPUTE   = ct.ComputeUnit.CPU_AND_NE
NUM_GEN   = 100  # tokens to generate after prefill
BS        = 256


def decode_one(engine, tok_id, pos):
    """Decode a single token, return next_id."""
    tok = engine._tok_buf
    tok[0, 0] = tok_id
    hidden = list(engine.embed.predict({"input_ids": tok}).values())[0]

    mask = engine._mask_buf
    mask[:, :, :, :] = -65504.0
    mask[:, :, :, :pos + 1] = 0

    pos_arr = engine._pos_buf
    pos_arr[0] = pos

    rope_arr = engine._rope_buf
    rope_arr[0] = pos + engine.rope_offset

    for ci in range(engine.num_chunks):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": rope_arr,
            "causal_mask": mask,
            "current_pos": pos_arr,
            "linear_conv_state": engine.lin_convs[ci],
            "linear_recurrent_state": engine.lin_recs[ci],
        }
        out = engine.ffns[ci].predict(inp, state=engine.states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            engine.lin_convs[ci] = out['linear_conv_state_out']
            engine.lin_recs[ci]  = out['linear_recurrent_state_out']

    lm_out = engine.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits = engine._extract_logits(lm_out)
        next_id = int(np.argmax(logits))
    else:
        next_id = int(lm_out["argmax_idx"].flatten()[0])
    engine.pos = pos + 1
    return next_id


def batch_prefill(engine, token_ids, block_start):
    """Run batch prefill, return first generated token id."""
    valid_len = len(token_ids)
    bs = engine._prefill_bs

    input_ids = engine._batch_tok_buf.copy()
    input_ids[0, :] = 0
    input_ids[0, :valid_len] = token_ids
    hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
    if valid_len < bs:
        hidden[:, valid_len:, :] = 0.0

    mask = engine._batch_mask_buf.copy()
    mask[:, :, :, :] = -65504.0
    for i in range(valid_len):
        mask[0, 0, i, :block_start + i + 1] = 0
    for i in range(valid_len, bs):
        mask[0, 0, i, 0] = 0.0

    pos_ids = engine._batch_pos_buf.copy()
    pos_ids[:valid_len] = np.arange(
        block_start + engine.rope_offset,
        block_start + engine.rope_offset + valid_len, dtype=np.int32)
    pos_ids[valid_len:] = 0

    cur_pos = engine._batch_cur_buf.copy()
    cur_pos[0] = block_start

    valid_len_arr = engine._valid_len_buf.copy()
    valid_len_arr[0] = valid_len

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
            engine.lin_recs[ci]  = out['linear_recurrent_state_out']
        if valid_len < bs:
            hidden[:, valid_len:, :] = 0.0

    if hidden.ndim >= 3 and hidden.shape[1] > 1:
        last_h = hidden[:, valid_len-1:valid_len, :]
    else:
        last_h = hidden

    lm_out = engine.lmhead.predict({"hidden_states": last_h.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits = engine._extract_logits(lm_out)
        next_id = int(np.argmax(logits))
    else:
        next_id = int(lm_out["argmax_idx"].flatten()[0])
    engine.pos = block_start + valid_len
    return next_id


def sequential_infer(engine, token_ids, start_pos):
    """Run sequential infer for each token, return first generated token id."""
    for ti, tok_id in enumerate(token_ids):
        pos = start_pos + ti
        tok = engine._tok_buf
        tok[0, 0] = tok_id
        hidden = list(engine.embed.predict({"input_ids": tok}).values())[0]

        mask = engine._mask_buf
        mask[:, :, :, :] = -65504.0
        mask[:, :, :, :pos + 1] = 0

        pos_arr = engine._pos_buf
        pos_arr[0] = pos

        rope_arr = engine._rope_buf
        rope_arr[0] = pos + engine.rope_offset

        for ci in range(engine.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": rope_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": engine.lin_convs[ci],
                "linear_recurrent_state": engine.lin_recs[ci],
            }
            out = engine.ffns[ci].predict(inp, state=engine.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                engine.lin_convs[ci] = out['linear_conv_state_out']
                engine.lin_recs[ci]  = out['linear_recurrent_state_out']
        engine.pos = pos + 1

    lm_out = engine.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits = engine._extract_logits(lm_out)
        next_id = int(np.argmax(logits))
    else:
        next_id = int(lm_out["argmax_idx"].flatten()[0])
    return next_id


def generate_n(engine, first_tok, start_pos, n, tokenizer, label):
    """Generate n tokens starting from first_tok at start_pos."""
    ids = [first_tok]
    stop_ids = {248044, 248046}  # <|endoftext|>, <|im_end|>
    for i in range(n - 1):
        if ids[-1] in stop_ids:
            break
        nxt = decode_one(engine, ids[-1], start_pos + i)
        ids.append(nxt)
    text = tokenizer.decode(ids)
    return ids, text


def snapshot_states(engine):
    """Snapshot all states for later restore."""
    snap = {'lin': {}, 'kv': {}}
    for ci in range(engine.num_chunks):
        snap['lin'][ci] = {
            'conv': engine.lin_convs[ci].copy(),
            'rec':  engine.lin_recs[ci].copy(),
        }
        snap['kv'][ci] = {}
        for sn in engine.kv_state_names:
            snap['kv'][ci][sn] = engine.states[ci].read_state(name=sn).copy()
    return snap


def restore_states(engine, snap):
    """Restore all states from snapshot."""
    for ci in range(engine.num_chunks):
        engine.lin_convs[ci] = snap['lin'][ci]['conv'].copy()
        engine.lin_recs[ci]  = snap['lin'][ci]['rec'].copy()
        for sn in engine.kv_state_names:
            engine.states[ci].write_state(name=sn, value=snap['kv'][ci][sn])


def main():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(HF_PATH, 'tokenizer.json'))

    # Build prompt > 1 BS
    base = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
    )
    paragraph = (
        "Explain the theory of general relativity in simple terms. "
        "How does mass curve spacetime and what are the observable consequences "
        "for light, time, and gravity near massive objects? "
        "Discuss gravitational lensing, time dilation, and frame dragging. "
    )
    prompt = base
    while True:
        enc = tokenizer.encode(prompt + "<|im_end|>\n<|im_start|>assistant\n")
        if len(enc.ids) > BS + 30:
            break
        prompt += paragraph
    prompt += "<|im_end|>\n<|im_start|>assistant\n"
    all_ids = tokenizer.encode(prompt).ids
    block1_ids = all_ids[:BS]
    block2_ids = all_ids[BS:]

    print(f"Prompt: {len(all_ids)} tokens = block1({len(block1_ids)}) + block2({len(block2_ids)})")
    print(f"Generating {NUM_GEN} tokens from each path\n")

    # Load engine once
    engine = ChatEngine(MODEL_DIR, HF_PATH, ctx=4096, num_chunks=9,
                        compute_unit=COMPUTE)
    engine.load()
    print(f"Engine ready: BS={engine._prefill_bs}, ctx={engine.ctx}\n")

    # ── Path A: block1 batch + block2 batch (padded) ──
    print("=" * 60)
    print("PATH A: block1(batch) + block2(batch-padded)")
    print("=" * 60)
    t0 = time.time()
    engine._reset_states()
    _ = batch_prefill(engine, block1_ids, block_start=0)
    first_a = batch_prefill(engine, block2_ids, block_start=BS)
    pos_after_prefill = engine.pos

    ids_a, text_a = generate_n(engine, first_a, pos_after_prefill, NUM_GEN, tokenizer, "A")
    t_a = time.time() - t0
    print(f"  First token: {first_a}")
    print(f"  Generated {len(ids_a)} tokens in {t_a:.1f}s")
    print(f"  Text: {text_a[:500]}")
    print()

    # ── Path B: block1 batch + block2 sequential ──
    print("=" * 60)
    print("PATH B: block1(batch) + block2(sequential)")
    print("=" * 60)
    t0 = time.time()
    engine._reset_states()
    _ = batch_prefill(engine, block1_ids, block_start=0)
    first_b = sequential_infer(engine, block2_ids, start_pos=BS)
    pos_after_prefill_b = engine.pos

    ids_b, text_b = generate_n(engine, first_b, pos_after_prefill_b, NUM_GEN, tokenizer, "B")
    t_b = time.time() - t0
    print(f"  First token: {first_b}")
    print(f"  Generated {len(ids_b)} tokens in {t_b:.1f}s")
    print(f"  Text: {text_b[:500]}")
    print()

    # ── Comparison ──
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    min_len = min(len(ids_a), len(ids_b))
    first_diff = None
    for i in range(min_len):
        if ids_a[i] != ids_b[i]:
            first_diff = i
            break
    if first_diff is not None:
        print(f"  First divergence at token {first_diff}/{min_len}")
        print(f"    A[{first_diff}] = {ids_a[first_diff]}  B[{first_diff}] = {ids_b[first_diff]}")
        match_pct = first_diff / min_len * 100
        print(f"  Matching: {first_diff}/{min_len} tokens ({match_pct:.1f}%)")
    else:
        print(f"  ALL {min_len} tokens match!")

    print(f"\n  len(A)={len(ids_a)}  len(B)={len(ids_b)}")
    print(f"\n--- PATH A TEXT ---")
    print(text_a)
    print(f"\n--- PATH B TEXT ---")
    print(text_b)

    del engine


if __name__ == '__main__':
    main()
