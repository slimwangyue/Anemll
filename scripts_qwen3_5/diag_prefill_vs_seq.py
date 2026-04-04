#!/usr/bin/env python3
"""Diagnostic: compare batched prefill vs sequential decode for the same prompt.

Loads the combined model, runs the same 16-token prompt through:
  A) Batched prefill (single 512-wide call with valid_len=16)
  B) Sequential token-by-token (16 calls with seq_len=1)

Then compares:
  - Linear conv state after each path
  - Linear recurrent state after each path
  - First decode token from lm_head
  - First 5 decode tokens
"""
import sys, os, time
import numpy as np
import coremltools as ct

MODEL_DIR = "/Users/yw68/Anemll/qwen3_5_stable_models"
NUM_CHUNKS = 6
CTX = 2048
BATCH_SIZE = 512

# Prompt: "教我做红烧鱼" with Qwen3.5 chat template
PROMPT_TOKENS = [248045, 846, 198, 95975, 120282, 126114, 97255, 248046,
                 198, 248045, 74455, 198, 248068, 271, 248069, 271]

cu = ct.ComputeUnit.CPU_AND_NE

def find_model(base, prefix):
    for ext in (".mlpackage", ".mlmodelc"):
        p = os.path.join(base, prefix + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model matching {prefix} in {base}")

def load_models():
    combined_dir = os.path.join(MODEL_DIR, "combined_LUT6_dedup")
    embed_path = find_model(MODEL_DIR, "embeddings")
    lm_path = find_model(MODEL_DIR, "lm_head_logits")

    print("Loading embeddings...", flush=True)
    embed = ct.models.MLModel(embed_path, compute_units=cu)
    print("Loading lm_head...", flush=True)
    lm_head = ct.models.MLModel(lm_path, compute_units=cu)

    ffns = []
    prefills = []
    for ci in range(NUM_CHUNKS):
        path = find_model(combined_dir, f"chunk{ci}")
        print(f"Loading chunk {ci}...", flush=True)
        t0 = time.time()
        m_infer = ct.models.MLModel(path, compute_units=cu, function_name="infer")
        m_prefill = ct.models.MLModel(path, compute_units=cu, function_name="prefill")
        print(f"  loaded in {time.time()-t0:.0f}s")
        ffns.append(m_infer)
        prefills.append(m_prefill)
    return embed, lm_head, ffns, prefills

def get_state_shapes(ffns):
    """Detect conv/rec shapes from chunk0 infer outputs."""
    conv_shapes = []
    rec_shapes = []
    for ci in range(NUM_CHUNKS):
        spec = ffns[ci].get_spec()
        fn_desc = spec.description
        if hasattr(fn_desc, 'functions'):
            for fn in fn_desc.functions:
                if fn.name == 'infer':
                    fn_desc = fn
                    break
        out_dict = {}
        for o in fn_desc.output:
            s = list(o.type.multiArrayType.shape)
            out_dict[o.name] = s
        if 'linear_conv_state_out' in out_dict:
            conv_shapes.append(tuple(out_dict['linear_conv_state_out']))
            rec_shapes.append(tuple(out_dict['linear_recurrent_state_out']))
        else:
            conv_shapes.append(None)
            rec_shapes.append(None)
    return conv_shapes, rec_shapes

def make_fresh_states(ffns, conv_shapes, rec_shapes):
    states = [m.make_state() for m in ffns]
    lin_convs = [np.zeros(s, dtype=np.float16) if s else None for s in conv_shapes]
    lin_recs = [np.zeros(s, dtype=np.float16) if s else None for s in rec_shapes]
    return states, lin_convs, lin_recs

def run_sequential(embed, lm_head, ffns, conv_shapes, rec_shapes):
    """Run prompt token-by-token (sequential path)."""
    states, lin_convs, lin_recs = make_fresh_states(ffns, conv_shapes, rec_shapes)
    tok_buf = np.zeros((1, 1), dtype=np.int32)
    mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    pos_buf = np.zeros((1,), dtype=np.int32)

    for i, tok_id in enumerate(PROMPT_TOKENS):
        tok_buf[0, 0] = tok_id
        hidden = list(embed.predict({"input_ids": tok_buf}).values())[0]

        mask_buf[:, :, :, :] = -65504.0
        mask_buf[:, :, :, :i + 1] = 0
        pos_buf[0] = i

        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_buf,
                "causal_mask": mask_buf,
                "current_pos": pos_buf,
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
            }
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']

    # Get first decode token
    lm_out = lm_head.predict({"hidden_states": hidden.astype(np.float16)})
    # Extract logits
    logits_parts = []
    for k in sorted(lm_out.keys()):
        if 'logit' in k.lower():
            logits_parts.append(lm_out[k].flatten().astype(np.float32))
    if logits_parts:
        logits = np.concatenate(logits_parts)
        first_tok = int(np.argmax(logits))
    else:
        first_tok = int(lm_out.get("argmax_idx", lm_out[list(lm_out.keys())[0]]).flatten()[0])
        logits = None

    pos = len(PROMPT_TOKENS)

    # Decode 5 more tokens
    decode_toks = [first_tok]
    for step in range(4):
        tok_buf[0, 0] = decode_toks[-1]
        hidden = list(embed.predict({"input_ids": tok_buf}).values())[0]
        mask_buf[:, :, :, :] = -65504.0
        mask_buf[:, :, :, :pos + 1] = 0
        pos_buf[0] = pos

        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_buf,
                "causal_mask": mask_buf,
                "current_pos": pos_buf,
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
            }
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
        lm_out = lm_head.predict({"hidden_states": hidden.astype(np.float16)})
        logits_parts = []
        for k in sorted(lm_out.keys()):
            if 'logit' in k.lower():
                logits_parts.append(lm_out[k].flatten().astype(np.float32))
        if logits_parts:
            next_tok = int(np.argmax(np.concatenate(logits_parts)))
        else:
            next_tok = int(lm_out[list(lm_out.keys())[0]].flatten()[0])
        decode_toks.append(next_tok)
        pos += 1

    return lin_convs, lin_recs, decode_toks

def run_batched(embed, lm_head, ffns, prefills, conv_shapes, rec_shapes):
    """Run prompt via batched prefill (single 512-wide call)."""
    states, lin_convs, lin_recs = make_fresh_states(ffns, conv_shapes, rec_shapes)

    valid_len = len(PROMPT_TOKENS)
    # Batch embedding
    input_ids = np.zeros((1, BATCH_SIZE), dtype=np.int32)
    input_ids[0, :valid_len] = PROMPT_TOKENS
    hidden = list(embed.predict({"input_ids": input_ids}).values())[0]

    # Causal mask
    mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
    for i in range(valid_len):
        mask[0, 0, i, :i + 1] = 0
    # Padding rows stay all -inf

    pos_ids = np.zeros((BATCH_SIZE,), dtype=np.int32)
    pos_ids[:valid_len] = np.arange(0, valid_len, dtype=np.int32)

    cur_pos = np.array([0], dtype=np.int32)
    valid_len_arr = np.array([valid_len], dtype=np.int32)

    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": cur_pos,
            "linear_conv_state": lin_convs[ci],
            "linear_recurrent_state": lin_recs[ci],
            "valid_len": valid_len_arr,
        }
        out = prefills[ci].predict(inp, state=states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_convs[ci] = out['linear_conv_state_out']
            lin_recs[ci] = out['linear_recurrent_state_out']

    # Extract last valid token
    if hidden.ndim >= 3 and hidden.shape[1] > 1:
        hidden = hidden[:, valid_len - 1:valid_len, :]

    lm_out = lm_head.predict({"hidden_states": hidden.astype(np.float16)})
    logits_parts = []
    for k in sorted(lm_out.keys()):
        if 'logit' in k.lower():
            logits_parts.append(lm_out[k].flatten().astype(np.float32))
    if logits_parts:
        logits = np.concatenate(logits_parts)
        first_tok = int(np.argmax(logits))
    else:
        first_tok = int(lm_out.get("argmax_idx", lm_out[list(lm_out.keys())[0]]).flatten()[0])
        logits = None

    pos = valid_len

    # Decode 5 more tokens using infer (sequential)
    tok_buf = np.zeros((1, 1), dtype=np.int32)
    mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    pos_buf = np.zeros((1,), dtype=np.int32)

    decode_toks = [first_tok]
    for step in range(4):
        tok_buf[0, 0] = decode_toks[-1]
        hidden = list(embed.predict({"input_ids": tok_buf}).values())[0]
        mask_buf[:, :, :, :] = -65504.0
        mask_buf[:, :, :, :pos + 1] = 0
        pos_buf[0] = pos

        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_buf,
                "causal_mask": mask_buf,
                "current_pos": pos_buf,
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
            }
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
        lm_out = lm_head.predict({"hidden_states": hidden.astype(np.float16)})
        logits_parts = []
        for k in sorted(lm_out.keys()):
            if 'logit' in k.lower():
                logits_parts.append(lm_out[k].flatten().astype(np.float32))
        if logits_parts:
            next_tok = int(np.argmax(np.concatenate(logits_parts)))
        else:
            next_tok = int(lm_out[list(lm_out.keys())[0]].flatten()[0])
        decode_toks.append(next_tok)
        pos += 1

    return lin_convs, lin_recs, decode_toks

def compare_states(seq_convs, seq_recs, bat_convs, bat_recs):
    print("\n" + "="*60)
    print("STATE COMPARISON: Sequential vs Batched Prefill")
    print("="*60)
    for ci in range(NUM_CHUNKS):
        if seq_convs[ci] is None:
            continue
        sc = seq_convs[ci].astype(np.float32)
        bc = bat_convs[ci].astype(np.float32)
        diff_c = np.abs(sc - bc)
        print(f"\n  chunk{ci} conv_state:")
        print(f"    max_abs_diff = {diff_c.max():.6e}")
        print(f"    mean_abs_diff = {diff_c.mean():.6e}")
        print(f"    seq max = {np.abs(sc).max():.4f}, bat max = {np.abs(bc).max():.4f}")

        sr = seq_recs[ci].astype(np.float32)
        br = bat_recs[ci].astype(np.float32)
        diff_r = np.abs(sr - br)
        print(f"  chunk{ci} rec_state:")
        print(f"    max_abs_diff = {diff_r.max():.6e}")
        print(f"    mean_abs_diff = {diff_r.mean():.6e}")
        print(f"    seq max = {np.abs(sr).max():.4f}, bat max = {np.abs(br).max():.4f}")

        # Check if any recurrent state entries are NaN or extremely large
        if np.isnan(br).any():
            print(f"    *** BATCHED REC STATE HAS NaN! ***")
        if np.abs(br).max() > 100:
            print(f"    *** BATCHED REC STATE HAS LARGE VALUES (>{np.abs(br).max():.1f}) ***")

def main():
    print("Loading models...")
    embed, lm_head, ffns, prefills = load_models()
    conv_shapes, rec_shapes = get_state_shapes(ffns)
    print(f"Conv shapes: {conv_shapes}")
    print(f"Rec shapes: {rec_shapes}")

    print("\n--- Running SEQUENTIAL prefill ---")
    t0 = time.time()
    seq_convs, seq_recs, seq_toks = run_sequential(embed, lm_head, ffns, conv_shapes, rec_shapes)
    print(f"Sequential took {time.time()-t0:.1f}s")
    print(f"Sequential decode tokens: {seq_toks}")

    print("\n--- Running BATCHED prefill ---")
    t0 = time.time()
    bat_convs, bat_recs, bat_toks = run_batched(embed, lm_head, ffns, prefills, conv_shapes, rec_shapes)
    print(f"Batched took {time.time()-t0:.1f}s")
    print(f"Batched decode tokens: {bat_toks}")

    compare_states(seq_convs, seq_recs, bat_convs, bat_recs)

    print("\n" + "="*60)
    print("DECODE TOKEN COMPARISON")
    print("="*60)
    match = seq_toks == bat_toks
    print(f"  Sequential: {seq_toks}")
    print(f"  Batched:    {bat_toks}")
    print(f"  Match: {match}")
    if not match:
        for i, (s, b) in enumerate(zip(seq_toks, bat_toks)):
            marker = "✓" if s == b else "✗"
            print(f"    tok[{i}]: seq={s}, bat={b} {marker}")

if __name__ == "__main__":
    main()
