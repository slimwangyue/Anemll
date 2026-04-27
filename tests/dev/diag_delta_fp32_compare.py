#!/usr/bin/env python3
"""
Delta-FP32 A/B comparison: baseline vs patched prefill models.

For BOTH baseline and patched model directories, runs:
  Path A: block1 full batch + block2 tail batch   (the divergent path)
  Path B: block1 full batch + block2 tail sequential (the reference path)

Compares baseline-A-vs-B divergence against patched-A-vs-B divergence
to determine if fp32 delta rule accumulations reduce tail-batch error.

Metrics:
  - block1 state equality (A vs B should be identical in both)
  - block2 chunk0 hidden cosine / max abs diff (A vs B)
  - per-chunk hidden divergence through block2 (A vs B)
  - per-chunk linear state divergence (conv + rec)
  - per-chunk KV divergence at full-attention chunks
  - final lastTokenHidden cosine
  - final logits cosine
  - first generated token match/mismatch
"""

import sys, os, copy, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine

# ─── Configuration ───────────────────────────────────────────────────
BASELINE_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3'
PATCHED_DIR  = '/Volumes/MySSD/Anemll/qwen3_5_4B_delta_fp32_test'
HF_PATH      = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'
COMPUTE       = ct.ComputeUnit.CPU_ONLY

# ─── Helpers ─────────────────────────────────────────────────────────
def cosine(a, b):
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    if na < 1e-30 and nb < 1e-30: return 1.0
    if na < 1e-30 or nb < 1e-30:  return 0.0
    return float(np.dot(a64, b64) / (na * nb))

def maxabs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))

def l2norm(a):
    return float(np.linalg.norm(a.astype(np.float64).ravel()))

def snapshot_kv(engine):
    kv = {}
    for ci in range(engine.num_chunks):
        kv[ci] = {}
        for sn in engine.kv_state_names:
            kv[ci][sn] = engine.states[ci].read_state(name=sn).copy()
    return kv

def snapshot_linear(engine):
    lin = {}
    for ci in range(engine.num_chunks):
        lin[ci] = {
            'conv': engine.lin_convs[ci].copy(),
            'rec':  engine.lin_recs[ci].copy(),
        }
    return lin

def snapshot_all(engine):
    return {'kv': snapshot_kv(engine), 'lin': snapshot_linear(engine)}

def restore_kv(engine, kv_snap):
    for ci in range(engine.num_chunks):
        for sn in engine.kv_state_names:
            engine.states[ci].write_state(name=sn, value=kv_snap[ci][sn])

def restore_linear(engine, lin_snap):
    for ci in range(engine.num_chunks):
        engine.lin_convs[ci] = lin_snap[ci]['conv'].copy()
        engine.lin_recs[ci]  = lin_snap[ci]['rec'].copy()

def restore_all(engine, snap):
    restore_kv(engine, snap['kv'])
    restore_linear(engine, snap['lin'])


# ─── Instrumented batch prefill (per-chunk capture) ──────────────────
def batch_prefill_instrumented(engine, token_ids, block_start):
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

    result = {'per_chunk': {}}

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

        chunk_data = {}
        if hidden.ndim >= 3 and hidden.shape[1] > 1:
            chunk_data['hidden_last_valid'] = hidden[:, valid_len-1:valid_len, :].copy()
        else:
            chunk_data['hidden_last_valid'] = hidden.copy()
        chunk_data['hidden_full'] = hidden.copy()
        for sn in engine.kv_state_names:
            chunk_data[sn] = engine.states[ci].read_state(name=sn).copy()
        chunk_data['conv'] = engine.lin_convs[ci].copy()
        chunk_data['rec']  = engine.lin_recs[ci].copy()
        result['per_chunk'][ci] = chunk_data

        if valid_len < bs:
            hidden[:, valid_len:, :] = 0.0

    if hidden.ndim >= 3 and hidden.shape[1] > 1:
        last_h = hidden[:, valid_len-1:valid_len, :]
    else:
        last_h = hidden
    result['last_hidden'] = last_h.copy()

    lm_out = engine.lmhead.predict({"hidden_states": last_h.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits = engine._extract_logits(lm_out)
        next_id = int(np.argmax(logits))
        result['logits'] = logits.copy() if isinstance(logits, np.ndarray) else np.array(logits)
    else:
        next_id = int(lm_out["argmax_idx"].flatten()[0])
        result['logits'] = None
    result['token'] = next_id
    engine.pos = block_start + valid_len
    return result


def sequential_tail_instrumented(engine, token_ids, start_pos):
    n = len(token_ids)
    result = {'per_chunk': {}}

    for ti, tok_id in enumerate(token_ids):
        is_last = (ti == n - 1)
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

            if is_last:
                chunk_data = {}
                chunk_data['hidden_last_valid'] = hidden.copy()
                chunk_data['hidden_full'] = hidden.copy()
                for sn in engine.kv_state_names:
                    chunk_data[sn] = engine.states[ci].read_state(name=sn).copy()
                chunk_data['conv'] = engine.lin_convs[ci].copy()
                chunk_data['rec']  = engine.lin_recs[ci].copy()
                result['per_chunk'][ci] = chunk_data

        engine.pos = pos + 1

    result['last_hidden'] = hidden.copy()
    lm_out = engine.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits = engine._extract_logits(lm_out)
        next_id = int(np.argmax(logits))
        result['logits'] = logits.copy() if isinstance(logits, np.ndarray) else np.array(logits)
    else:
        next_id = int(lm_out["argmax_idx"].flatten()[0])
        result['logits'] = None
    result['token'] = next_id
    return result


def decode_one_token(engine, tok_id, pos):
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
        logits = None
        next_id = int(lm_out["argmax_idx"].flatten()[0])

    engine.pos = pos + 1
    return next_id, logits, hidden


# ─── Run one model variant (baseline or patched) ────────────────────
def run_variant(model_dir, label, prompt_ids, block1_ids, block2_ids, bs,
                force_separate=False, compute_unit=None):
    """
    Run both paths (A=batch tail, B=seq tail) for one model variant.
    Returns dict of results.
    """
    if compute_unit is None:
        compute_unit = COMPUTE
    # Monkey-patch _find_model: prefer .mlpackage for prefill (needs make_state),
    # prefer .mlmodelc for everything else
    import scripts_qwen3_5.chat_server as cs_module

    # If force_separate, temporarily hide combined_*_dedup so ChatEngine
    # loads from separate ffn_LUT4_chunk*.mlmodelc + prefill_LUT4_chunk*.mlmodelc
    combined_hidden = None
    if force_separate:
        combined_dir = cs_module._find_combined_dir(model_dir)
        if combined_dir:
            combined_hidden = combined_dir + '.__hidden__'
            os.rename(combined_dir, combined_hidden)
            print(f"  [force_separate] Temporarily hid {os.path.basename(combined_dir)}")

    print(f"\n{'='*72}")
    print(f"  VARIANT: {label}")
    print(f"  Model dir: {model_dir}")
    print(f"{'='*72}")

    try:
        engine = ChatEngine(model_dir, HF_PATH, ctx=4096, num_chunks=9,
                            compute_unit=compute_unit)
        engine.load()
    finally:
        # Restore combined dir
        if combined_hidden and os.path.exists(combined_hidden):
            os.rename(combined_hidden, combined_hidden.replace('.__hidden__', ''))

    print(f"  Loaded. BS={engine._prefill_bs}, ctx={engine.ctx}, chunks={engine.num_chunks}")
    assert engine._prefill_bs == bs, f"Expected BS={bs}, got {engine._prefill_bs}"

    # In separate mode (non-combined), prefill and ffn are different model instances
    # whose states are not interchangeable. Create separate ffn states for decode.
    use_separate_ffn_states = not engine.use_combined

    # ── Path A: block1 batch + block2 batch tail ──
    print(f"\n  --- Path A: block1 batch + block2 BATCH tail ---")
    engine._reset_states()
    block1_a = batch_prefill_instrumented(engine, block1_ids, block_start=0)
    snap_after_block1_a = snapshot_all(engine)
    block2_a = batch_prefill_instrumented(engine, block2_ids, block_start=bs)
    snap_after_block2_a = snapshot_all(engine)
    print(f"  Path A: block1 tok={block1_a['token']}, block2 tok={block2_a['token']}")

    if use_separate_ffn_states:
        # Can't run sequential decode with prefill-created states.
        # Return batch-only results.
        print(f"\n  --- Path B: SKIPPED (separate mode, cross-state incompatible) ---")
        block1_b = block1_a  # same as A (batch uses same code path)
        snap_after_block1_b = snap_after_block1_a
        block2_b = block2_a  # use batch results as placeholder
        snap_after_block2_b = snap_after_block2_a
    else:
        # ── Path B: block1 batch + block2 sequential tail ──
        print(f"\n  --- Path B: block1 batch + block2 SEQUENTIAL tail ---")
        engine._reset_states()
        block1_b = batch_prefill_instrumented(engine, block1_ids, block_start=0)
        snap_after_block1_b = snapshot_all(engine)
        block2_b = sequential_tail_instrumented(engine, block2_ids, start_pos=bs)
        snap_after_block2_b = snapshot_all(engine)
        print(f"  Path B: block1 tok={block1_b['token']}, block2 tok={block2_b['token']}")
    print(f"  Path B: block1 tok={block1_b['token']}, block2 tok={block2_b['token']}")

    # ── Checkpoint 1: Block1 identity ──
    print(f"\n  --- Checkpoint 1: Block1 state equality ---")
    block1_identical = True
    for ci in range(engine.num_chunks):
        for sn in engine.kv_state_names:
            kv_a = snap_after_block1_a['kv'][ci][sn]
            kv_b = snap_after_block1_b['kv'][ci][sn]
            if not np.array_equal(kv_a, kv_b):
                block1_identical = False
                print(f"    !! chunk{ci} {sn} DIFFERS (cos={cosine(kv_a, kv_b):.6f})")
        ca = snap_after_block1_a['lin'][ci]['conv']
        cb = snap_after_block1_b['lin'][ci]['conv']
        if not np.array_equal(ca, cb):
            block1_identical = False
            print(f"    !! chunk{ci} conv DIFFERS")
        ra = snap_after_block1_a['lin'][ci]['rec']
        rb = snap_after_block1_b['lin'][ci]['rec']
        if not np.array_equal(ra, rb):
            block1_identical = False
            print(f"    !! chunk{ci} rec DIFFERS")
    print(f"  Block1 identical: {block1_identical}")

    # ── Checkpoint 2: Per-chunk divergence in block2 ──
    print(f"\n  --- Checkpoint 2: Per-chunk divergence in block2 ---")
    print(f"  {'chunk':>6} | {'hidden_cos':>10} | {'hidden_max':>10} | {'conv_cos':>9} | {'rec_cos':>9} | {'kv_cos':>9}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

    per_chunk_metrics = {}
    for ci in range(engine.num_chunks):
        ha = block2_a['per_chunk'][ci]['hidden_last_valid']
        hb = block2_b['per_chunk'][ci]['hidden_last_valid']
        h_cos = cosine(ha, hb)
        h_max = maxabs(ha, hb)

        ca = block2_a['per_chunk'][ci]['conv']
        cb = block2_b['per_chunk'][ci]['conv']
        c_cos = cosine(ca, cb)

        ra_arr = block2_a['per_chunk'][ci]['rec']
        rb_arr = block2_b['per_chunk'][ci]['rec']
        r_cos = cosine(ra_arr, rb_arr)

        # KV divergence (average across all state names)
        kv_cosines = []
        for sn in engine.kv_state_names:
            if sn in block2_a['per_chunk'][ci] and sn in block2_b['per_chunk'][ci]:
                ka = block2_a['per_chunk'][ci][sn]
                kb = block2_b['per_chunk'][ci][sn]
                kv_cosines.append(cosine(ka, kb))
        kv_cos = np.mean(kv_cosines) if kv_cosines else 1.0

        per_chunk_metrics[ci] = {
            'hidden_cos': h_cos, 'hidden_max': h_max,
            'conv_cos': c_cos, 'rec_cos': r_cos, 'kv_cos': kv_cos,
        }
        print(f"  {ci:>6} | {h_cos:>10.6f} | {h_max:>10.4f} | {c_cos:>9.6f} | {r_cos:>9.6f} | {kv_cos:>9.6f}")

    # ── Checkpoint 3: Final comparison ──
    print(f"\n  --- Checkpoint 3: Final comparison ---")
    h_cos_final = cosine(block2_a['last_hidden'], block2_b['last_hidden'])
    h_max_final = maxabs(block2_a['last_hidden'], block2_b['last_hidden'])
    print(f"  last_hidden cos={h_cos_final:.6f}  max_abs={h_max_final:.4f}")

    if block2_a['logits'] is not None and block2_b['logits'] is not None:
        l_cos = cosine(block2_a['logits'], block2_b['logits'])
        print(f"  logits cos={l_cos:.6f}")
    else:
        l_cos = None
        print(f"  logits: argmax mode")

    tok_match = block2_a['token'] == block2_b['token']
    print(f"  token A={block2_a['token']}  B={block2_b['token']}  match={tok_match}")

    # ── Experiment X: Replace bad hidden with good → lm_head ──
    print(f"\n  --- Experiment X: good hidden → lm_head ---")
    good_hidden = block2_b['last_hidden'].copy()
    lm_out_x = engine.lmhead.predict({"hidden_states": good_hidden.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits_x = engine._extract_logits(lm_out_x)
        tok_x = int(np.argmax(logits_x))
    else:
        tok_x = int(lm_out_x["argmax_idx"].flatten()[0])
    print(f"  good hidden → lm_head → token={tok_x}  (B_token={block2_b['token']})")
    print(f"  Experiment X fixes token: {tok_x == block2_b['token']}")

    # ── Experiment Y: Replace bad states with good → decode 1 token ──
    # ── Experiment Z: Replace bad linear with good → decode 1 token ──
    tok_y = None
    tok_z = None
    if not use_separate_ffn_states:
        print(f"\n  --- Experiment Y: good KV → decode ---")
        engine._reset_states()
        restore_all(engine, snap_after_block2_a)  # start from path A final state
        restore_kv(engine, snap_after_block2_b['kv'])  # replace KV with path B
        engine.pos = bs + len(block2_ids)
        tok_y, _, _ = decode_one_token(engine, block2_a['token'], pos=engine.pos)
        print(f"  A-states + B-KV → decode → token={tok_y}")

        print(f"\n  --- Experiment Z: good linear → decode ---")
        engine._reset_states()
        restore_all(engine, snap_after_block2_a)
        restore_linear(engine, snap_after_block2_b['lin'])
        engine.pos = bs + len(block2_ids)
        tok_z, _, _ = decode_one_token(engine, block2_a['token'], pos=engine.pos)
        print(f"  A-states + B-linear → decode → token={tok_z}")
    else:
        print(f"\n  --- Experiments Y, Z: SKIPPED (separate mode) ---")

    del engine
    return {
        'label': label,
        'block1_identical': block1_identical,
        'per_chunk_metrics': per_chunk_metrics,
        'hidden_cos_final': h_cos_final,
        'hidden_max_final': h_max_final,
        'logits_cos': l_cos,
        'token_a': block2_a['token'],
        'token_b': block2_b['token'],
        'token_match': tok_match,
        'tok_x': tok_x,
        'tok_y': tok_y,
        'tok_z': tok_z,
    }


# ======================================================================
#                           MAIN
# ======================================================================
def main():
    print("=" * 72)
    print("  DELTA-FP32 A/B COMPARISON")
    print("  Baseline: V4 (all fp16 delta rule)")
    print("  Patched:  V4 + delta-fp32 (exp/matmul/reduce in fp32 for prefill)")
    print("=" * 72)

    # Build prompt > 1 BS to force 2 blocks
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(HF_PATH, 'tokenizer.json'))
    BS = 256

    # Prompt template
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
    print(f"\n  Total prompt tokens: {len(all_ids)}")

    block1_ids = all_ids[:BS]
    block2_ids = all_ids[BS:]
    print(f"  Block 1: {BS} tokens (full batch)")
    print(f"  Block 2: {len(block2_ids)} tokens (tail)")

    # ── Run baseline ──
    print("\n" + "=" * 72)
    print("  PHASE 1: BASELINE")
    print("=" * 72)
    t0 = time.time()
    baseline = run_variant(BASELINE_DIR, "BASELINE", all_ids, block1_ids, block2_ids, BS)
    t_baseline = time.time() - t0
    print(f"\n  Baseline completed in {t_baseline:.1f}s")

    # ── Run patched ──
    print("\n" + "=" * 72)
    print("  PHASE 2: PATCHED (delta-fp32)")
    print("=" * 72)
    t0 = time.time()
    patched = run_variant(PATCHED_DIR, "PATCHED (delta-fp32)", all_ids, block1_ids, block2_ids, BS)
    t_patched = time.time() - t0
    print(f"\n  Patched completed in {t_patched:.1f}s")

    # ── Summary comparison ──
    print("\n" + "=" * 72)
    print("  SUMMARY: BASELINE vs PATCHED")
    print("=" * 72)

    print(f"\n  {'Metric':<35} | {'Baseline':>12} | {'Patched':>12} | {'Better?':>8}")
    print(f"  {'-'*35}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    # Block1 identity
    print(f"  {'block1_identical':<35} | {str(baseline['block1_identical']):>12} | {str(patched['block1_identical']):>12} |")

    # Chunk0 hidden divergence
    b_c0 = baseline['per_chunk_metrics'].get(0, {}).get('hidden_cos', 'N/A')
    p_c0 = patched['per_chunk_metrics'].get(0, {}).get('hidden_cos', 'N/A')
    if isinstance(b_c0, float) and isinstance(p_c0, float):
        better_c0 = "YES" if p_c0 > b_c0 + 0.001 else ("no" if p_c0 < b_c0 - 0.001 else "same")
        print(f"  {'chunk0 hidden cos':<35} | {b_c0:>12.6f} | {p_c0:>12.6f} | {better_c0:>8}")
    else:
        print(f"  {'chunk0 hidden cos':<35} | {str(b_c0):>12} | {str(p_c0):>12} |")

    # Per-chunk hidden divergence
    print(f"\n  Per-chunk hidden cosine (A vs B):")
    print(f"  {'chunk':>6} | {'Baseline':>12} | {'Patched':>12} | {'Delta':>10}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for ci in range(9):
        bm = baseline['per_chunk_metrics'].get(ci, {})
        pm = patched['per_chunk_metrics'].get(ci, {})
        bc = bm.get('hidden_cos', None)
        pc = pm.get('hidden_cos', None)
        if bc is not None and pc is not None:
            delta = pc - bc
            print(f"  {ci:>6} | {bc:>12.6f} | {pc:>12.6f} | {delta:>+10.6f}")

    # Per-chunk conv/rec divergence
    print(f"\n  Per-chunk linear state cosine (A vs B):")
    print(f"  {'chunk':>6} | {'B conv':>10} | {'P conv':>10} | {'B rec':>10} | {'P rec':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for ci in range(9):
        bm = baseline['per_chunk_metrics'].get(ci, {})
        pm = patched['per_chunk_metrics'].get(ci, {})
        bc = bm.get('conv_cos', None)
        pc = pm.get('conv_cos', None)
        br = bm.get('rec_cos', None)
        pr = pm.get('rec_cos', None)
        if bc is not None and pc is not None:
            print(f"  {ci:>6} | {bc:>10.6f} | {pc:>10.6f} | {br:>10.6f} | {pr:>10.6f}")

    # Per-chunk KV divergence
    print(f"\n  Per-chunk KV cosine (A vs B):")
    print(f"  {'chunk':>6} | {'Baseline':>12} | {'Patched':>12} | {'Delta':>10}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for ci in range(9):
        bm = baseline['per_chunk_metrics'].get(ci, {})
        pm = patched['per_chunk_metrics'].get(ci, {})
        bk = bm.get('kv_cos', None)
        pk = pm.get('kv_cos', None)
        if bk is not None and pk is not None:
            delta = pk - bk
            print(f"  {ci:>6} | {bk:>12.6f} | {pk:>12.6f} | {delta:>+10.6f}")

    # Final metrics
    b_hc = baseline['hidden_cos_final']
    p_hc = patched['hidden_cos_final']
    better_hc = "YES" if p_hc > b_hc + 0.001 else ("no" if p_hc < b_hc - 0.001 else "same")
    print(f"\n  {'final hidden cos':<35} | {b_hc:>12.6f} | {p_hc:>12.6f} | {better_hc:>8}")
    print(f"  {'final hidden maxabs':<35} | {baseline['hidden_max_final']:>12.4f} | {patched['hidden_max_final']:>12.4f} |")

    if baseline['logits_cos'] is not None and patched['logits_cos'] is not None:
        b_lc = baseline['logits_cos']
        p_lc = patched['logits_cos']
        better_lc = "YES" if p_lc > b_lc + 0.001 else ("no" if p_lc < b_lc - 0.001 else "same")
        print(f"  {'final logits cos':<35} | {b_lc:>12.6f} | {p_lc:>12.6f} | {better_lc:>8}")

    b_match = baseline['token_match']
    p_match = patched['token_match']
    print(f"  {'token A==B match':<35} | {str(b_match):>12} | {str(p_match):>12} | {'YES' if p_match and not b_match else ''}")
    print(f"  {'token A':<35} | {baseline['token_a']:>12} | {patched['token_a']:>12} |")
    print(f"  {'token B':<35} | {baseline['token_b']:>12} | {patched['token_b']:>12} |")

    print(f"\n  Experiment results:")
    print(f"  {'exp X (good hidden→lmhead)':<35} | {baseline['tok_x']:>12} | {patched['tok_x']:>12} |")
    b_y = baseline['tok_y'] if baseline['tok_y'] is not None else 'N/A'
    p_y = patched['tok_y'] if patched['tok_y'] is not None else 'N/A'
    b_z = baseline['tok_z'] if baseline['tok_z'] is not None else 'N/A'
    p_z = patched['tok_z'] if patched['tok_z'] is not None else 'N/A'
    print(f"  {'exp Y (good KV→decode)':<35} | {str(b_y):>12} | {str(p_y):>12} |")
    print(f"  {'exp Z (good linear→decode)':<35} | {str(b_z):>12} | {str(p_z):>12} |")

    # ── Conclusion ──
    print(f"\n  {'='*60}")
    improvements = 0
    if isinstance(p_c0, float) and isinstance(b_c0, float) and p_c0 > b_c0 + 0.001:
        improvements += 1
        print(f"  ✓ Chunk0 hidden similarity improved: {b_c0:.6f} → {p_c0:.6f}")
    if p_hc > b_hc + 0.001:
        improvements += 1
        print(f"  ✓ Final hidden similarity improved: {b_hc:.6f} → {p_hc:.6f}")
    if baseline['logits_cos'] is not None and patched['logits_cos'] is not None:
        if patched['logits_cos'] > baseline['logits_cos'] + 0.001:
            improvements += 1
            print(f"  ✓ Logits similarity improved: {baseline['logits_cos']:.6f} → {patched['logits_cos']:.6f}")
    if p_match and not b_match:
        improvements += 1
        print(f"  ✓ Token mismatch FIXED!")
    elif p_match == b_match:
        print(f"  · Token match unchanged: {b_match}")

    if improvements > 0:
        print(f"\n  CONCLUSION: Delta-FP32 provides {improvements} improvement(s)")
    else:
        print(f"\n  CONCLUSION: Delta-FP32 does NOT materially help")

    print(f"\n  ALL COMPARISONS COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
