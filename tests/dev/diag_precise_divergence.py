#!/usr/bin/env python3
"""
Precise per-chunk divergence localization for batch tail prefill bug.

Compares:
  Path A: block1 batch prefill + block2 batch tail prefill  (BAD)
  Path B: block1 batch prefill + block2 sequential tail     (GOOD)

Instrumentation at three checkpoints:
  1) After block1 — verify A and B are identical
  2) During block2, after EACH chunk — find first divergence
  3) After block2 — final hidden, logits, token

Swap experiments:
  X) Replace bad final hidden with good → run lm_head
  Y) Replace bad KV states with good → decode 1 token
  Z) Replace bad linear states with good → decode 1 token
  Per-chunk KV/linear swaps to find dominant source chunk

current_pos semantic test:
  Run path A with current_pos = blockStart + validLen - 1

All on CPU_AND_NE (ANE path).
"""

import sys, os, copy, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine

# ─── Configuration ───────────────────────────────────────────────────
MODEL_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3'
HF_PATH   = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'
COMPUTE   = ct.ComputeUnit.CPU_AND_NE

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
    """Snapshot all KV cache states (MLState)."""
    kv = {}
    for ci in range(engine.num_chunks):
        kv[ci] = {}
        for sn in engine.kv_state_names:
            kv[ci][sn] = engine.states[ci].read_state(name=sn).copy()
    return kv

def snapshot_linear(engine):
    """Snapshot all linear conv/rec states."""
    lin = {}
    for ci in range(engine.num_chunks):
        lin[ci] = {
            'conv': engine.lin_convs[ci].copy(),
            'rec':  engine.lin_recs[ci].copy(),
        }
    return lin

def snapshot_all(engine):
    """Full state snapshot: KV + linear."""
    return {'kv': snapshot_kv(engine), 'lin': snapshot_linear(engine)}

def restore_kv(engine, kv_snap):
    """Restore KV cache states from snapshot."""
    for ci in range(engine.num_chunks):
        for sn in engine.kv_state_names:
            engine.states[ci].write_state(name=sn, value=kv_snap[ci][sn])

def restore_linear(engine, lin_snap):
    """Restore linear states from snapshot."""
    for ci in range(engine.num_chunks):
        engine.lin_convs[ci] = lin_snap[ci]['conv'].copy()
        engine.lin_recs[ci]  = lin_snap[ci]['rec'].copy()

def restore_all(engine, snap):
    """Restore full state."""
    restore_kv(engine, snap['kv'])
    restore_linear(engine, snap['lin'])


# ─── Instrumented batch prefill (per-chunk capture) ──────────────────
def batch_prefill_instrumented(engine, token_ids, block_start, cur_pos_override=None):
    """
    Run batch prefill with per-chunk state capture.
    Returns dict with per-chunk snapshots + final token + logits.
    """
    valid_len = len(token_ids)
    bs = engine._prefill_bs

    # Embed
    input_ids = engine._batch_tok_buf.copy()
    input_ids[0, :] = 0
    input_ids[0, :valid_len] = token_ids
    hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
    if valid_len < bs:
        hidden[:, valid_len:, :] = 0.0

    # Causal mask
    mask = engine._batch_mask_buf.copy()
    mask[:, :, :, :] = -65504.0
    for i in range(valid_len):
        mask[0, 0, i, :block_start + i + 1] = 0
    for i in range(valid_len, bs):
        mask[0, 0, i, 0] = 0.0

    # Position IDs
    pos_ids = engine._batch_pos_buf.copy()
    pos_ids[:valid_len] = np.arange(
        block_start + engine.rope_offset,
        block_start + engine.rope_offset + valid_len, dtype=np.int32)
    pos_ids[valid_len:] = 0

    # current_pos
    cur_pos = engine._batch_cur_buf.copy()
    if cur_pos_override is not None:
        cur_pos[0] = cur_pos_override
    else:
        cur_pos[0] = block_start

    # valid_len
    valid_len_arr = engine._valid_len_buf.copy()
    valid_len_arr[0] = valid_len

    result = {
        'inputs': {
            'current_pos': int(cur_pos[0]),
            'valid_len': valid_len,
            'block_start': block_start,
            'position_ids_first': int(pos_ids[0]),
            'position_ids_last_valid': int(pos_ids[valid_len-1]),
        },
        'per_chunk': {},
    }

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

        # Capture per-chunk data
        chunk_data = {}
        # Hidden for the LAST VALID token only
        if hidden.ndim >= 3 and hidden.shape[1] > 1:
            chunk_data['hidden_last_valid'] = hidden[:, valid_len-1:valid_len, :].copy()
        else:
            chunk_data['hidden_last_valid'] = hidden.copy()
        # Full hidden (for detailed analysis of first diverging chunk)
        chunk_data['hidden_full'] = hidden.copy()
        # KV state after this chunk
        for sn in engine.kv_state_names:
            chunk_data[sn] = engine.states[ci].read_state(name=sn).copy()
        # Linear states after this chunk
        chunk_data['conv'] = engine.lin_convs[ci].copy()
        chunk_data['rec']  = engine.lin_recs[ci].copy()

        result['per_chunk'][ci] = chunk_data

        # Re-zero padding between chunks
        if valid_len < bs:
            hidden[:, valid_len:, :] = 0.0

    # Extract last valid token for lm_head
    if hidden.ndim >= 3 and hidden.shape[1] > 1:
        last_h = hidden[:, valid_len-1:valid_len, :]
    else:
        last_h = hidden

    result['last_hidden'] = last_h.copy()

    # LM head
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


# ─── Instrumented sequential tail (per-chunk capture) ────────────────
def sequential_tail_instrumented(engine, token_ids, start_pos):
    """
    Run sequential prefill via infer models with per-chunk state capture
    for the LAST token only (to match what batch prefill produces).
    """
    n = len(token_ids)
    result = {
        'inputs': {
            'start_pos': start_pos,
            'num_tokens': n,
        },
        'per_chunk': {},
    }

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

            # Capture per-chunk data for LAST token only
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

    # LM head on final hidden
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


# ─── Decode one token (for swap experiments) ─────────────────────────
def decode_one_token(engine, tok_id, pos):
    """Run one step through infer models + lm_head. Returns (next_id, logits, hidden)."""
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


# ======================================================================
#                           MAIN
# ======================================================================
def main():
    # Monkey-patch module-level _find_model to prefer .mlmodelc
    import scripts_qwen3_5.chat_server as cs_module
    def _find_model_mlmodelc(base_dir, name):
        for ext in ('.mlmodelc', '.mlpackage'):
            p = os.path.join(base_dir, name + ext)
            if os.path.exists(p):
                return p
        return None
    cs_module._find_model = _find_model_mlmodelc

    print("=" * 72)
    print("  PRECISE DIVERGENCE LOCALIZATION")
    print("  Path A: block1 batch + block2 batch tail")
    print("  Path B: block1 batch + block2 sequential tail")
    print("=" * 72)

    # ── Load engine ──
    print("\n  Loading engine (CPU_AND_NE)...")
    t0 = time.time()
    engine = ChatEngine(MODEL_DIR, HF_PATH, ctx=4096, num_chunks=9, compute_unit=COMPUTE)
    engine.use_combined = False
    engine.combined_dir = None
    engine.load()
    print(f"  Engine loaded in {time.time()-t0:.1f}s")
    print(f"  BS={engine._prefill_bs} CTX={engine.ctx} chunks={engine.num_chunks}")

    nc = engine.num_chunks
    bs = engine._prefill_bs

    # ── Tokenize ──
    # Need a prompt that produces >256 tokens so block1=256 + tail>0
    prompt = "What is the capital of France?" 
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    all_tokens = tokenizer.encode(text)
    n_total = len(all_tokens)
    print(f"\n  Prompt: {prompt!r}")
    print(f"  Total tokens from template: {n_total}")

    # If prompt is too short, generate enough tokens by repeating content
    if n_total <= bs:
        # Build prompt by repeating paragraphs until we exceed bs tokens
        para = ("Please explain in detail the history geography culture economy "
                "and political system of France. Cover the major historical events "
                "from the Roman era through the Revolution to modern times. ")
        long_prompt = ""
        while True:
            long_prompt += para
            test_msgs = [{"role": "user", "content": long_prompt}]
            test_text = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
            test_toks = tokenizer.encode(test_text)
            if len(test_toks) > bs + 30:  # enough for block1 + meaningful tail
                break
        messages = [{"role": "user", "content": long_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_tokens = tokenizer.encode(text)
        n_total = len(all_tokens)
        print(f"  Extended prompt tokens: {n_total}")

    block1_tokens = all_tokens[:bs]
    tail_tokens = all_tokens[bs:]
    print(f"  Block1: {len(block1_tokens)}, Tail: {len(tail_tokens)}")
    assert len(tail_tokens) > 0, f"Need >BS tokens but got {n_total}"

    # ══════════════════════════════════════════════════════════════════
    #  CHECKPOINT 1: Run block1, verify A and B start identically
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  CHECKPOINT 1: Block1 (identical for both paths)")
    print("=" * 72)

    # --- Path A: block1 ---
    engine._reset_states()
    engine.pos = 0
    _ = batch_prefill_instrumented(engine, block1_tokens, 0)
    snap_after_block1_A = snapshot_all(engine)
    pos_after_block1_A = engine.pos
    print(f"\n  Path A block1 done: pos={pos_after_block1_A}")

    # --- Path B: block1 (reset + run again) ---
    engine._reset_states()
    engine.pos = 0
    _ = batch_prefill_instrumented(engine, block1_tokens, 0)
    snap_after_block1_B = snapshot_all(engine)
    pos_after_block1_B = engine.pos
    print(f"  Path B block1 done: pos={pos_after_block1_B}")

    # Compare
    print("\n  Block1 state comparison (A vs B):")
    identical = True
    for ci in range(nc):
        for sn in engine.kv_state_names:
            c = cosine(snap_after_block1_A['kv'][ci][sn], snap_after_block1_B['kv'][ci][sn])
            mx = maxabs(snap_after_block1_A['kv'][ci][sn], snap_after_block1_B['kv'][ci][sn])
            if c < 0.999999 or mx > 1e-6:
                print(f"    chunk{ci}/{sn}: cos={c:.6f} max_abs={mx:.6f} ← DIFFERENT!")
                identical = False
        c_conv = cosine(snap_after_block1_A['lin'][ci]['conv'], snap_after_block1_B['lin'][ci]['conv'])
        c_rec  = cosine(snap_after_block1_A['lin'][ci]['rec'],  snap_after_block1_B['lin'][ci]['rec'])
        mx_conv = maxabs(snap_after_block1_A['lin'][ci]['conv'], snap_after_block1_B['lin'][ci]['conv'])
        mx_rec  = maxabs(snap_after_block1_A['lin'][ci]['rec'],  snap_after_block1_B['lin'][ci]['rec'])
        if c_conv < 0.999999 or mx_conv > 1e-6:
            print(f"    chunk{ci}/conv: cos={c_conv:.6f} max_abs={mx_conv:.6f} ← DIFFERENT!")
            identical = False
        if c_rec < 0.999999 or mx_rec > 1e-6:
            print(f"    chunk{ci}/rec:  cos={c_rec:.6f} max_abs={mx_rec:.6f} ← DIFFERENT!")
            identical = False

    if identical:
        print("    ✓ All states BIT-IDENTICAL after block1")
    else:
        print("    ✗ States DIFFER after block1 — block1 itself is non-deterministic!")

    # ══════════════════════════════════════════════════════════════════
    #  CHECKPOINT 2: Block2 per-chunk comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  CHECKPOINT 2: Block2 per-chunk divergence")
    print("=" * 72)

    # Save the post-block1 state for re-use
    block1_snap = snap_after_block1_B  # B's state (just ran)
    block1_pos = pos_after_block1_B

    # --- Path A: block2 batch tail ---
    print("\n  Running Path A (batch tail)...")
    # State is already at post-block1 from path B run. Just use it.
    result_A = batch_prefill_instrumented(engine, tail_tokens, block1_pos)
    snap_after_A = snapshot_all(engine)
    pos_A = engine.pos

    # --- Restore state for Path B ---
    print("  Restoring block1 state for Path B...")
    restore_all(engine, block1_snap)
    engine.pos = block1_pos

    # --- Path B: block2 sequential tail ---
    print("  Running Path B (sequential tail)...")
    result_B = sequential_tail_instrumented(engine, tail_tokens, block1_pos)
    snap_after_B = snapshot_all(engine)
    pos_B = engine.pos

    # --- Per-chunk comparison ---
    print(f"\n  Path A token: {result_A['token']}")
    print(f"  Path B token: {result_B['token']}")
    print(f"  Token match: {result_A['token'] == result_B['token']}")

    print(f"\n  Control inputs for Path A (batch):")
    for k, v in result_A['inputs'].items():
        print(f"    {k}: {v}")
    print(f"  Control inputs for Path B (sequential):")
    for k, v in result_B['inputs'].items():
        print(f"    {k}: {v}")

    print(f"\n  Per-chunk divergence (A=batch vs B=sequential):")
    print(f"  {'chunk':>5}  {'hidden_cos':>11}  {'hidden_mx':>10}  {'k_cos':>8}  {'v_cos':>8}  {'conv_cos':>9}  {'rec_cos':>8}  {'k_L2ratio':>10}  {'conv_L2ratio':>12}")
    print(f"  {'─'*5}  {'─'*11}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*10}  {'─'*12}")

    first_diverge_chunk = None
    first_diverge_state = None

    for ci in range(nc):
        a_chunk = result_A['per_chunk'][ci]
        b_chunk = result_B['per_chunk'][ci]

        h_cos = cosine(a_chunk['hidden_last_valid'], b_chunk['hidden_last_valid'])
        h_mx  = maxabs(a_chunk['hidden_last_valid'], b_chunk['hidden_last_valid'])

        # KV (will be empty for linear-only chunks)
        k_cos = v_cos = 1.0
        k_l2r = 1.0
        for sn in engine.kv_state_names:
            if sn in a_chunk and sn in b_chunk:
                c = cosine(a_chunk[sn], b_chunk[sn])
                if sn == 'k_cache':
                    k_cos = c
                    la, lb = l2norm(a_chunk[sn]), l2norm(b_chunk[sn])
                    k_l2r = la / lb if lb > 0 else 0
                elif sn == 'v_cache':
                    v_cos = c

        conv_cos = cosine(a_chunk['conv'], b_chunk['conv'])
        rec_cos  = cosine(a_chunk['rec'],  b_chunk['rec'])

        la_c, lb_c = l2norm(a_chunk['conv']), l2norm(b_chunk['conv'])
        conv_l2r = la_c / lb_c if lb_c > 0 else 0

        print(f"  {ci:>5}  {h_cos:>11.6f}  {h_mx:>10.4f}  {k_cos:>8.4f}  {v_cos:>8.4f}  {conv_cos:>9.6f}  {rec_cos:>8.6f}  {k_l2r:>10.4f}  {conv_l2r:>12.4f}")

        # Determine first divergence
        if first_diverge_chunk is None:
            threshold = 0.9999
            diverged = []
            if h_cos < threshold: diverged.append(('hidden', h_cos))
            if k_cos < threshold: diverged.append(('k_cache', k_cos))
            if v_cos < threshold: diverged.append(('v_cache', v_cos))
            if conv_cos < threshold: diverged.append(('conv', conv_cos))
            if rec_cos < threshold: diverged.append(('rec', rec_cos))
            if diverged:
                first_diverge_chunk = ci
                # Sort by cosine ascending (worst first)
                diverged.sort(key=lambda x: x[1])
                first_diverge_state = diverged[0][0]

    print(f"\n  First diverging chunk: {first_diverge_chunk}")
    if first_diverge_chunk is not None:
        print(f"  First diverging state type: {first_diverge_state}")
        # Print detailed first diverging chunk info
        ci = first_diverge_chunk
        a_c, b_c = result_A['per_chunk'][ci], result_B['per_chunk'][ci]
        print(f"\n  Detailed chunk {ci} comparison:")
        print(f"    hidden:   cos={cosine(a_c['hidden_last_valid'], b_c['hidden_last_valid']):.8f}  max_abs={maxabs(a_c['hidden_last_valid'], b_c['hidden_last_valid']):.6f}  A_L2={l2norm(a_c['hidden_last_valid']):.4f}  B_L2={l2norm(b_c['hidden_last_valid']):.4f}")
        for sn in engine.kv_state_names:
            if sn in a_c:
                print(f"    {sn:8s}: cos={cosine(a_c[sn], b_c[sn]):.8f}  max_abs={maxabs(a_c[sn], b_c[sn]):.6f}  A_L2={l2norm(a_c[sn]):.4f}  B_L2={l2norm(b_c[sn]):.4f}")
        print(f"    conv:     cos={cosine(a_c['conv'], b_c['conv']):.8f}  max_abs={maxabs(a_c['conv'], b_c['conv']):.6f}  A_L2={l2norm(a_c['conv']):.4f}  B_L2={l2norm(b_c['conv']):.4f}")
        print(f"    rec:      cos={cosine(a_c['rec'], b_c['rec']):.8f}  max_abs={maxabs(a_c['rec'], b_c['rec']):.6f}  A_L2={l2norm(a_c['rec']):.4f}  B_L2={l2norm(b_c['rec']):.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  CHECKPOINT 3: Final state comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  CHECKPOINT 3: Final state after block2")
    print("=" * 72)

    h_A = result_A['last_hidden']
    h_B = result_B['last_hidden']
    print(f"\n  Final hidden: cos={cosine(h_A, h_B):.8f}  max_abs={maxabs(h_A, h_B):.6f}")
    print(f"    A_L2={l2norm(h_A):.4f}  B_L2={l2norm(h_B):.4f}")

    if result_A['logits'] is not None and result_B['logits'] is not None:
        logA, logB = np.array(result_A['logits']), np.array(result_B['logits'])
        print(f"  Logits:       cos={cosine(logA, logB):.8f}  max_abs={maxabs(logA, logB):.6f}")
        topA = np.argsort(logA)[-5:][::-1]
        topB = np.argsort(logB)[-5:][::-1]
        print(f"  Top-5 A: {list(topA)}")
        print(f"  Top-5 B: {list(topB)}")

    print(f"  Token A: {result_A['token']}")
    print(f"  Token B: {result_B['token']}")

    # Per-chunk final state summary
    print(f"\n  Final state divergence per chunk:")
    print(f"  {'chunk':>5}  {'k_cos':>8}  {'v_cos':>8}  {'conv_cos':>9}  {'rec_cos':>8}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")
    for ci in range(nc):
        k_cos = v_cos = 1.0
        for sn in engine.kv_state_names:
            c = cosine(snap_after_A['kv'][ci][sn], snap_after_B['kv'][ci][sn])
            if sn == 'k_cache': k_cos = c
            elif sn == 'v_cache': v_cos = c
        conv_cos = cosine(snap_after_A['lin'][ci]['conv'], snap_after_B['lin'][ci]['conv'])
        rec_cos  = cosine(snap_after_A['lin'][ci]['rec'],  snap_after_B['lin'][ci]['rec'])
        print(f"  {ci:>5}  {k_cos:>8.4f}  {v_cos:>8.4f}  {conv_cos:>9.6f}  {rec_cos:>8.6f}")

    # ══════════════════════════════════════════════════════════════════
    #  EXPERIMENT X: Final-hidden swap
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  EXPERIMENT X: Replace BAD final hidden with GOOD → lm_head")
    print("=" * 72)

    # Use GOOD path B's hidden with lm_head
    good_h = result_B['last_hidden'].copy()
    lm_out = engine.lmhead.predict({"hidden_states": good_h.astype(np.float16)})
    if engine.lmhead_mode == "logits":
        logits_X = engine._extract_logits(lm_out)
        tok_X = int(np.argmax(logits_X))
    else:
        tok_X = int(lm_out["argmax_idx"].flatten()[0])

    print(f"  BAD path token:                {result_A['token']}")
    print(f"  GOOD path token:               {result_B['token']}")
    print(f"  BAD path + GOOD hidden → token: {tok_X}")
    print(f"  Does swapping hidden fix token? {tok_X == result_B['token']}")

    # ══════════════════════════════════════════════════════════════════
    #  EXPERIMENT Y: KV-only swap (good KV, bad linear, decode 1 token)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  EXPERIMENT Y: Good KV states + Bad linear → decode 1 token")
    print("=" * 72)

    # Restore bad path A states
    restore_all(engine, snap_after_A)
    engine.pos = pos_A
    # Replace KV with good path B
    restore_kv(engine, snap_after_B['kv'])
    # Decode using path A's predicted token
    tok_to_decode = result_A['token']
    tok_Y, logits_Y, _ = decode_one_token(engine, tok_to_decode, pos_A)

    # Also decode from good path for reference
    restore_all(engine, snap_after_B)
    engine.pos = pos_B
    tok_ref, logits_ref, _ = decode_one_token(engine, result_B['token'], pos_B)

    print(f"  Decoded from BAD (all bad states):  (would need separate run)")
    print(f"  Decoded from GOOD (all good states): {tok_ref}")
    print(f"  Decoded from MIXED (good KV + bad linear): {tok_Y}")
    if logits_Y is not None and logits_ref is not None:
        print(f"  Logits cos (mixed vs good): {cosine(np.array(logits_Y), np.array(logits_ref)):.6f}")

    # Also run pure bad decode for comparison
    restore_all(engine, snap_after_A)
    engine.pos = pos_A
    tok_bad_decode, logits_bad_decode, _ = decode_one_token(engine, result_A['token'], pos_A)
    print(f"  Decoded from BAD (all bad states):   {tok_bad_decode}")
    print(f"  Does KV swap fix decode? {tok_Y == tok_ref}")

    # ══════════════════════════════════════════════════════════════════
    #  EXPERIMENT Z: Linear-only swap (good linear, bad KV, decode 1 token)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  EXPERIMENT Z: Good linear states + Bad KV → decode 1 token")
    print("=" * 72)

    restore_all(engine, snap_after_A)
    engine.pos = pos_A
    restore_linear(engine, snap_after_B['lin'])
    tok_Z, logits_Z, _ = decode_one_token(engine, result_A['token'], pos_A)

    print(f"  Decoded from BAD (all bad states):    {tok_bad_decode}")
    print(f"  Decoded from GOOD (all good states):  {tok_ref}")
    print(f"  Decoded from MIXED (good linear + bad KV): {tok_Z}")
    if logits_Z is not None and logits_ref is not None:
        print(f"  Logits cos (mixed vs good): {cosine(np.array(logits_Z), np.array(logits_ref)):.6f}")
    print(f"  Does linear swap fix decode? {tok_Z == tok_ref}")

    # ══════════════════════════════════════════════════════════════════
    #  PER-CHUNK KV SWAP: Find dominant source chunk
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PER-CHUNK KV SWAP: Good KV one chunk at a time")
    print("=" * 72)
    print(f"  {'swap_chunk':>10}  {'decode_tok':>10}  {'logits_cos':>11}  {'matches_good':>12}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*11}  {'─'*12}")

    for swap_ci in range(nc):
        restore_all(engine, snap_after_A)  # start from BAD
        engine.pos = pos_A
        # Swap just this chunk's KV
        for sn in engine.kv_state_names:
            engine.states[swap_ci].write_state(name=sn, value=snap_after_B['kv'][swap_ci][sn])
        tok_swap, logits_swap, _ = decode_one_token(engine, result_A['token'], pos_A)
        lc = cosine(np.array(logits_swap), np.array(logits_ref)) if logits_swap is not None and logits_ref is not None else -1
        print(f"  {swap_ci:>10}  {tok_swap:>10}  {lc:>11.6f}  {tok_swap == tok_ref:>12}")

    # ══════════════════════════════════════════════════════════════════
    #  PER-CHUNK LINEAR SWAP: Good linear one chunk at a time
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PER-CHUNK LINEAR SWAP: Good linear one chunk at a time")
    print("=" * 72)
    print(f"  {'swap_chunk':>10}  {'decode_tok':>10}  {'logits_cos':>11}  {'matches_good':>12}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*11}  {'─'*12}")

    for swap_ci in range(nc):
        restore_all(engine, snap_after_A)  # start from BAD
        engine.pos = pos_A
        # Swap just this chunk's linear states
        engine.lin_convs[swap_ci] = snap_after_B['lin'][swap_ci]['conv'].copy()
        engine.lin_recs[swap_ci]  = snap_after_B['lin'][swap_ci]['rec'].copy()
        tok_swap, logits_swap, _ = decode_one_token(engine, result_A['token'], pos_A)
        lc = cosine(np.array(logits_swap), np.array(logits_ref)) if logits_swap is not None and logits_ref is not None else -1
        print(f"  {swap_ci:>10}  {tok_swap:>10}  {lc:>11.6f}  {tok_swap == tok_ref:>12}")

    # ══════════════════════════════════════════════════════════════════
    #  CURRENT_POS SEMANTIC TEST
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  CURRENT_POS SEMANTIC TEST")
    print("  Default: current_pos = blockStart")
    print("  Alternative: current_pos = blockStart + validLen - 1")
    print("=" * 72)

    # Restore block1 state
    restore_all(engine, block1_snap)
    engine.pos = block1_pos

    # Run block2 with alternative current_pos
    alt_cur_pos = block1_pos + len(tail_tokens) - 1
    print(f"\n  Default current_pos: {block1_pos}")
    print(f"  Alternative current_pos: {alt_cur_pos}")

    result_A_alt = batch_prefill_instrumented(
        engine, tail_tokens, block1_pos, cur_pos_override=alt_cur_pos)

    print(f"\n  Token (default current_pos):     {result_A['token']}")
    print(f"  Token (alt current_pos):          {result_A_alt['token']}")
    print(f"  Token (sequential/GOOD):          {result_B['token']}")
    print(f"  Does alt current_pos match GOOD?  {result_A_alt['token'] == result_B['token']}")

    if result_A_alt['logits'] is not None and result_B['logits'] is not None:
        logA_alt = np.array(result_A_alt['logits'])
        logB = np.array(result_B['logits'])
        print(f"  Logits cos (alt vs good): {cosine(logA_alt, logB):.8f}")
        print(f"  Logits cos (def vs good): {cosine(np.array(result_A['logits']), logB):.8f}")

    # Per-chunk comparison: alt vs good
    print(f"\n  Per-chunk divergence (alt_cur_pos A' vs sequential B):")
    print(f"  {'chunk':>5}  {'hidden_cos':>11}  {'k_cos':>8}  {'v_cos':>8}  {'conv_cos':>9}  {'rec_cos':>8}")
    print(f"  {'─'*5}  {'─'*11}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")
    for ci in range(nc):
        a_c = result_A_alt['per_chunk'][ci]
        b_c = result_B['per_chunk'][ci]
        h_cos = cosine(a_c['hidden_last_valid'], b_c['hidden_last_valid'])
        k_cos = v_cos = 1.0
        for sn in engine.kv_state_names:
            if sn in a_c and sn in b_c:
                c = cosine(a_c[sn], b_c[sn])
                if sn == 'k_cache': k_cos = c
                elif sn == 'v_cache': v_cos = c
        conv_cos = cosine(a_c['conv'], b_c['conv'])
        rec_cos  = cosine(a_c['rec'],  b_c['rec'])
        print(f"  {ci:>5}  {h_cos:>11.6f}  {k_cos:>8.4f}  {v_cos:>8.4f}  {conv_cos:>9.6f}  {rec_cos:>8.6f}")

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
