#!/usr/bin/env python3
"""Diagnostic: Compare batch-prefill+decode vs sequential-only.

Uses per-chunk input maps (shapes differ between chunks).
"""
import sys, os, time
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
from config import BATCH_SIZE, CTX, NUM_CHUNKS, FFN_LABEL

MODEL_DIR = "/Users/yw68/Anemll/qwen3_5_stable_models_6chunk"
FFN_DIR = os.path.join(MODEL_DIR, "combined_LUT6_dedup")
TOKENIZER_DIR = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"
GEN_TOKENS = 30


def load_models():
    cu = ct.ComputeUnit.CPU_AND_NE
    embed = ct.models.MLModel(os.path.join(MODEL_DIR, "embeddings.mlpackage"),
                              compute_units=ct.ComputeUnit.CPU_ONLY)
    lmhead = ct.models.MLModel(os.path.join(MODEL_DIR, "lm_head_logits.mlpackage"),
                               compute_units=ct.ComputeUnit.CPU_ONLY)
    ffns = []
    prefills = []
    for ci in range(NUM_CHUNKS):
        path = os.path.join(FFN_DIR, f"chunk{ci}.mlpackage")
        print(f"  Loading chunk {ci}...", end="", flush=True)
        t0 = time.time()
        m_infer = ct.models.MLModel(path, compute_units=cu, function_name="infer")
        m_prefill = ct.models.MLModel(path, compute_units=cu, function_name="prefill")
        ffns.append(m_infer)
        prefills.append(m_prefill)
        print(f" {time.time()-t0:.0f}s")
    return embed, lmhead, ffns, prefills


def get_inp_maps(ffns):
    """Get per-chunk input shape maps (shapes differ between chunks!)."""
    maps = []
    for m in ffns:
        spec = m.get_spec()
        imap = {}
        fn_inputs = None
        for fn in spec.description.functions:
            if fn.name == "infer":
                fn_inputs = fn.input
                break
        if fn_inputs is None:
            fn_inputs = spec.description.input
        for inp in fn_inputs:
            try:
                imap[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass
        maps.append(imap)
    return maps


def make_states(ffns):
    return [m.make_state() for m in ffns]


def init_linear(inp_maps):
    convs = [np.zeros(inp_maps[ci]['linear_conv_state'], dtype=np.float16)
             for ci in range(NUM_CHUNKS)]
    recs = [np.zeros(inp_maps[ci]['linear_recurrent_state'], dtype=np.float16)
            for ci in range(NUM_CHUNKS)]
    return convs, recs


def step_infer(embed, lmhead, ffns, states, convs, recs, tok_id, pos):
    tok = np.array([[tok_id]], dtype=np.int32)
    hidden = list(embed.predict({"input_ids": tok}).values())[0]
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :pos + 1] = 0
    pos_arr = np.array([pos], dtype=np.int32)
    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_arr,
            "causal_mask": mask,
            "current_pos": pos_arr,
            "linear_conv_state": convs[ci],
            "linear_recurrent_state": recs[ci],
        }
        out = ffns[ci].predict(inp, state=states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            convs[ci] = out['linear_conv_state_out']
            recs[ci] = out['linear_recurrent_state_out']
    lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    logits = np.concatenate([lm_out[k].flatten() for k in sorted(lm_out.keys())])
    return int(np.argmax(logits)), logits


def batch_prefill(embed, lmhead, prefills, states, convs, recs, token_ids, block_start):
    valid_len = len(token_ids)
    input_ids = np.zeros((1, BATCH_SIZE), dtype=np.int32)
    input_ids[0, :valid_len] = token_ids
    hidden = list(embed.predict({"input_ids": input_ids}).values())[0]

    mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
    for i in range(valid_len):
        mask[0, 0, i, :block_start + i + 1] = 0

    pos_ids = np.zeros(BATCH_SIZE, dtype=np.int32)
    pos_ids[:valid_len] = np.arange(block_start, block_start + valid_len, dtype=np.int32)
    cur_pos = np.array([block_start], dtype=np.int32)
    valid_len_arr = np.array([valid_len], dtype=np.int32)

    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": cur_pos,
            "linear_conv_state": convs[ci],
            "linear_recurrent_state": recs[ci],
            "valid_len": valid_len_arr,
        }
        out = prefills[ci].predict(inp, state=states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            convs[ci] = out['linear_conv_state_out']
            recs[ci] = out['linear_recurrent_state_out']

    if hidden.ndim >= 3 and hidden.shape[1] > 1:
        hidden = hidden[:, valid_len - 1:valid_len, :]

    lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    logits = np.concatenate([lm_out[k].flatten() for k in sorted(lm_out.keys())])
    return int(np.argmax(logits)), logits


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=False)
    prompt = "教我做红烧鱼"
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                         add_generation_prompt=True,
                                         enable_thinking=True)
    prompt_tokens = ids[0].tolist()
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
    print(f"Decoded: {repr(tokenizer.decode(prompt_tokens))}")

    print("\nLoading models...")
    embed, lmhead, ffns, prefills = load_models()
    inp_maps = get_inp_maps(ffns)

    for ci in range(NUM_CHUNKS):
        print(f"  chunk{ci}: conv={inp_maps[ci]['linear_conv_state']}, "
              f"rec={inp_maps[ci]['linear_recurrent_state']}")

    # ── Mode A: Sequential only ──
    print("\n" + "="*60)
    print("  MODE A: Sequential prefill + decode")
    print("="*60)
    states_a = make_states(ffns)
    convs_a, recs_a = init_linear(inp_maps)
    pos = 0
    for tok_id in prompt_tokens[:-1]:
        step_infer(embed, lmhead, ffns, states_a, convs_a, recs_a, tok_id, pos)
        pos += 1
    next_a, logits_a_first = step_infer(embed, lmhead, ffns, states_a, convs_a, recs_a,
                                        prompt_tokens[-1], pos)
    pos += 1
    gen_a = [next_a]
    for gi in range(GEN_TOKENS - 1):
        nxt, _ = step_infer(embed, lmhead, ffns, states_a, convs_a, recs_a, gen_a[-1], pos)
        gen_a.append(nxt)
        pos += 1
    print(f"  First token: {next_a} = {repr(tokenizer.decode([next_a]))}")
    print(f"  Generated: {tokenizer.decode(gen_a)[:200]}")

    # ── Mode B: Batch prefill + decode ──
    print("\n" + "="*60)
    print("  MODE B: Batch prefill + decode")
    print("="*60)
    states_b = make_states(ffns)
    convs_b, recs_b = init_linear(inp_maps)
    next_b, logits_b_first = batch_prefill(embed, lmhead, prefills, states_b,
                                           convs_b, recs_b, prompt_tokens, 0)
    pos_b = len(prompt_tokens)
    gen_b = [next_b]
    for gi in range(GEN_TOKENS - 1):
        nxt, _ = step_infer(embed, lmhead, ffns, states_b, convs_b, recs_b, gen_b[-1], pos_b)
        gen_b.append(nxt)
        pos_b += 1
    print(f"  First token: {next_b} = {repr(tokenizer.decode([next_b]))}")
    print(f"  Generated: {tokenizer.decode(gen_b)[:200]}")

    # ── Compare ──
    print("\n" + "="*60)
    print("  COMPARISON")
    print("="*60)
    matches = sum(1 for a, b in zip(gen_a, gen_b) if a == b)
    total = min(len(gen_a), len(gen_b))
    print(f"  Token match: {matches}/{total} ({100*matches/total:.0f}%)")

    diff = np.abs(logits_a_first.astype(np.float64) - logits_b_first.astype(np.float64))
    print(f"  First-token logit diff: max={diff.max():.4f}, mean={diff.mean():.6f}")
    top5_a = np.argsort(logits_a_first)[-5:][::-1]
    top5_b = np.argsort(logits_b_first)[-5:][::-1]
    print(f"  Top5 A: {[(int(i), tokenizer.decode([int(i)])) for i in top5_a]}")
    print(f"  Top5 B: {[(int(i), tokenizer.decode([int(i)])) for i in top5_b]}")

    # Compare linear state after prefill (before any decode)
    for ci in range(NUM_CHUNKS):
        cd = np.abs(convs_a[ci].astype(np.float64) - convs_b[ci].astype(np.float64))
        rd = np.abs(recs_a[ci].astype(np.float64) - recs_b[ci].astype(np.float64))
        print(f"  chunk{ci} conv: max={cd.max():.4f} mean={cd.mean():.6f} "
              f"| rec: max={rd.max():.4f} mean={rd.mean():.6f}")

    # Token-by-token
    first_diff = None
    for i in range(total):
        if gen_a[i] != gen_b[i]:
            if first_diff is None:
                first_diff = i
            print(f"    [{i}] A={gen_a[i]} '{tokenizer.decode([gen_a[i]])}' "
                  f"vs B={gen_b[i]} '{tokenizer.decode([gen_b[i]])}'")
            if i > first_diff + 5:
                print("    ...")
                break
    if first_diff is None:
        print("  ALL MATCH!")


if __name__ == "__main__":
    main()
