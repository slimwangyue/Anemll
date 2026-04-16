#!/usr/bin/env python3
"""Raw KV-cache persistence experiment for Qwen3.5-4B on ANE.

Tests whether raw runtime state (MLState KV cache, linear states, position)
can be persisted to disk and restored in a fresh process to continue decoding
with exact token-by-token equality against an uninterrupted golden run.

Usage:
    # Full golden run (saves golden tokens + state snapshots)
    python tests/dev/test_kv_persist_raw.py golden

    # Resume from snapshot in same process (sanity check)
    python tests/dev/test_kv_persist_raw.py resume-same

    # Resume from snapshot in fresh process (true restart)
    # Step 1: save snapshot
    python tests/dev/test_kv_persist_raw.py snapshot --snapshot-at prefill
    # Step 2: resume in new process
    python tests/dev/test_kv_persist_raw.py resume-fresh --snapshot-at prefill

    # Full automated test (runs golden + snapshot + resume-fresh via subprocess)
    python tests/dev/test_kv_persist_raw.py auto [--snapshot-at prefill,8,16]

Architecture:
    Wraps chat_server.ChatEngine to instrument the inference pipeline.
    Uses greedy decode (argmax only) for deterministic comparison.
    Persists ALL runtime state: MLState KV cache buffers, linear conv/recurrent
    states, position, rope_offset, token_history, and model metadata.
"""
import sys, os, argparse, json, time, subprocess, struct, hashlib
from pathlib import Path
from collections import deque

# Setup paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_QWEN_SCRIPTS = os.path.join(_REPO_ROOT, "scripts_qwen3_5")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _QWEN_SCRIPTS not in sys.path:
    sys.path.insert(0, _QWEN_SCRIPTS)

import numpy as np
import coremltools as ct  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────

SNAPSHOT_DIR = os.path.join(_SCRIPT_DIR, "kv_persist_snapshots")
GOLDEN_DIR = os.path.join(_SCRIPT_DIR, "kv_persist_golden")

# Fixed test prompts for deterministic comparison
TEST_PROMPTS = {
    "short": "What is the capital of France?",
    "medium": "Explain the difference between a compiler and an interpreter in simple terms.",
    "multi_turn": [
        "What is Python?",
        "How does it compare to Java?",
    ],
}

# Default: use fp32 model dir (has combined embed_lmhead + combined chunks)
DEFAULT_MODEL_DIR = os.path.join(
    _REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp32")
DEFAULT_HF_PATH = os.path.join(_REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
DEFAULT_NUM_CHUNKS = 9
DEFAULT_CTX = 2048
DEFAULT_MAX_TOKENS = 32  # short decode for testing


# ── Snapshot format ──────────────────────────────────────────────────

def save_snapshot(engine, snapshot_path, metadata=None):
    """Save complete inference state to disk.

    Persists:
      - MLState KV cache buffers (via read_state for each state name)
      - Linear conv states (numpy arrays per chunk)
      - Linear recurrent states (numpy arrays per chunk)
      - Position (pos), rope_offset
      - Token history (for validation)
      - Model metadata (num_chunks, ctx, state names, shapes)
    """
    os.makedirs(snapshot_path, exist_ok=True)
    print(f"\n[snapshot] Saving to {snapshot_path}")

    # 1. Save metadata
    meta = {
        "num_chunks": engine.num_chunks,
        "ctx": engine.ctx,
        "pos": engine.pos,
        "rope_offset": engine.rope_offset,
        "kv_state_names": engine.kv_state_names,
        "per_chunk_conv_shapes": [list(s) for s in engine.per_chunk_conv_shapes],
        "per_chunk_rec_shapes": [list(s) for s in engine.per_chunk_rec_shapes],
        "token_history": list(engine.token_history),
        "timestamp": time.time(),
    }
    if metadata:
        meta.update(metadata)
    with open(os.path.join(snapshot_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta: pos={engine.pos}, rope_offset={engine.rope_offset}, "
          f"history_len={len(engine.token_history)}")

    # 2. Save MLState KV cache buffers
    for ci in range(engine.num_chunks):
        state = engine.states[ci]
        for sname in engine.kv_state_names:
            arr = state.read_state(name=sname)
            fname = f"chunk{ci}_{sname}.npy"
            np.save(os.path.join(snapshot_path, fname), arr)
            print(f"  {fname}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"hash={_array_hash(arr)[:12]}")

    # 3. Save linear states
    for ci in range(engine.num_chunks):
        conv = engine.lin_convs[ci]
        rec = engine.lin_recs[ci]
        np.save(os.path.join(snapshot_path, f"chunk{ci}_lin_conv.npy"), conv)
        np.save(os.path.join(snapshot_path, f"chunk{ci}_lin_rec.npy"), rec)
        print(f"  chunk{ci}_lin_conv: shape={conv.shape}, hash={_array_hash(conv)[:12]}")
        print(f"  chunk{ci}_lin_rec:  shape={rec.shape}, hash={_array_hash(rec)[:12]}")

    print(f"[snapshot] Done — {sum(1 for _ in Path(snapshot_path).iterdir())} files")


def load_snapshot(snapshot_path):
    """Load snapshot metadata and arrays from disk.

    Returns (meta_dict, kv_arrays, lin_conv_arrays, lin_rec_arrays).
    kv_arrays: dict mapping (chunk_idx, state_name) -> numpy array
    lin_conv_arrays: list of numpy arrays per chunk
    lin_rec_arrays: list of numpy arrays per chunk
    """
    print(f"\n[restore] Loading from {snapshot_path}")
    with open(os.path.join(snapshot_path, "meta.json"), "r") as f:
        meta = json.load(f)

    kv_arrays = {}
    for ci in range(meta["num_chunks"]):
        for sname in meta["kv_state_names"]:
            fname = f"chunk{ci}_{sname}.npy"
            arr = np.load(os.path.join(snapshot_path, fname))
            kv_arrays[(ci, sname)] = arr
            print(f"  {fname}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"hash={_array_hash(arr)[:12]}")

    lin_convs = []
    lin_recs = []
    for ci in range(meta["num_chunks"]):
        conv = np.load(os.path.join(snapshot_path, f"chunk{ci}_lin_conv.npy"))
        rec = np.load(os.path.join(snapshot_path, f"chunk{ci}_lin_rec.npy"))
        lin_convs.append(conv)
        lin_recs.append(rec)
        print(f"  chunk{ci}_lin_conv: hash={_array_hash(conv)[:12]}")
        print(f"  chunk{ci}_lin_rec:  hash={_array_hash(rec)[:12]}")

    print(f"  pos={meta['pos']}, rope_offset={meta['rope_offset']}, "
          f"history_len={len(meta['token_history'])}")
    return meta, kv_arrays, lin_convs, lin_recs


def restore_state(engine, snapshot_path):
    """Restore complete inference state from disk snapshot.

    Writes KV cache via MLState.write_state(), sets linear states,
    restores position/offset/history.
    """
    meta, kv_arrays, lin_convs, lin_recs = load_snapshot(snapshot_path)

    # Validate model compatibility
    assert engine.num_chunks == meta["num_chunks"], \
        f"Chunk count mismatch: engine={engine.num_chunks} vs snapshot={meta['num_chunks']}"
    assert engine.ctx == meta["ctx"], \
        f"CTX mismatch: engine={engine.ctx} vs snapshot={meta['ctx']}"
    assert engine.kv_state_names == meta["kv_state_names"], \
        f"KV state names mismatch: {engine.kv_state_names} vs {meta['kv_state_names']}"

    # 1. Create fresh MLState objects and write saved data
    engine.states = [m.make_state() for m in engine.ffns]
    for ci in range(engine.num_chunks):
        for sname in engine.kv_state_names:
            arr = kv_arrays[(ci, sname)]
            engine.states[ci].write_state(name=sname, value=arr)
            # Verify round-trip
            readback = engine.states[ci].read_state(name=sname)
            if not np.array_equal(arr, readback):
                maxdiff = np.max(np.abs(arr.astype(np.float32) - readback.astype(np.float32)))
                print(f"  WARNING: chunk{ci}/{sname} write_state round-trip mismatch! "
                      f"max_diff={maxdiff}")
            else:
                print(f"  chunk{ci}/{sname}: write_state round-trip OK")

    # 2. Restore linear states
    engine.lin_convs = lin_convs
    engine.lin_recs = lin_recs

    # 3. Restore position and history
    engine.pos = meta["pos"]
    engine.rope_offset = meta["rope_offset"]
    engine.token_history = deque(meta["token_history"], maxlen=engine.ctx * 2)

    print(f"[restore] State restored: pos={engine.pos}, "
          f"rope_offset={engine.rope_offset}")


# ── Deterministic decode ─────────────────────────────────────────────

def deterministic_prefill(engine, prompt_tokens):
    """Prefill using the same path as chat_server._process_prompt.

    Always uses greedy argmax. Returns first_token_id.
    """
    return engine._process_prompt(prompt_tokens)


def deterministic_decode(engine, first_token, max_tokens, stop_ids):
    """Greedy decode using engine._step, exactly matching chat_server decode loop.

    Returns list of generated token IDs (including first_token).
    """
    generated = [first_token]
    for gi in range(max_tokens - 1):
        if engine.pos >= engine.ctx:
            print(f"[decode] Context full at pos={engine.pos}")
            break
        fed_tok = generated[-1]
        next_id, logits = engine._step(fed_tok, engine.pos)
        engine.pos += 1
        engine.token_history.append(fed_tok)
        # Pure greedy — always argmax, no penalties/sampling
        # (logits already give argmax via _step)
        generated.append(next_id)
        if next_id in stop_ids:
            break
    return generated


def build_prompt_tokens(engine, prompt_text, enable_thinking=False):
    """Build prompt tokens using the same path as chat_server."""
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = engine.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking)
    if hasattr(input_ids, 'input_ids'):
        input_ids = input_ids.input_ids
        return input_ids[0].tolist()
    return list(input_ids)


# ── Engine creation ──────────────────────────────────────────────────

def create_engine(model_dir=None, hf_path=None, num_chunks=None, ctx=None):
    """Create and load a ChatEngine."""
    from chat_server import ChatEngine
    model_dir = model_dir or DEFAULT_MODEL_DIR
    hf_path = hf_path or DEFAULT_HF_PATH
    num_chunks = num_chunks or DEFAULT_NUM_CHUNKS
    ctx = ctx or DEFAULT_CTX

    engine = ChatEngine(
        model_dir, hf_path, ctx=ctx,
        num_chunks=num_chunks,
        compute_unit=ct.ComputeUnit.CPU_AND_NE,
    )
    engine.load()
    return engine


# ── Golden run ───────────────────────────────────────────────────────

def run_golden(engine, prompt_text, max_tokens, snapshot_points,
               golden_dir, prompt_name="short"):
    """Run a complete inference and save golden tokens + state snapshots.

    snapshot_points: list of points like ["prefill", 1, 8, 16]
      - "prefill": snapshot after prefill, before any decode
      - int N: snapshot after generating N decode tokens
    """
    os.makedirs(golden_dir, exist_ok=True)

    prompt_tokens = build_prompt_tokens(engine, prompt_text,
                                         enable_thinking=False)
    print(f"\n{'='*60}")
    print(f"[golden] Prompt: {repr(prompt_text[:80])}")
    print(f"[golden] Tokens: {len(prompt_tokens)}")
    print(f"[golden] Snapshot points: {snapshot_points}")

    # Reset engine state
    engine._reset_states()

    # Prefill
    print(f"\n[golden] Prefilling {len(prompt_tokens)} tokens...")
    t0 = time.time()
    first_token = deterministic_prefill(engine, prompt_tokens)
    prefill_time = time.time() - t0
    if first_token is None:
        print("[golden] PREFILL FAILED — context overflow")
        return None
    print(f"[golden] Prefill done in {prefill_time*1000:.0f}ms, "
          f"first_token={first_token}, pos={engine.pos}")
    # Record prompt tokens in history (matches chat_server)
    engine.token_history.extend(prompt_tokens)

    # Snapshot after prefill
    if "prefill" in snapshot_points:
        snap_path = os.path.join(golden_dir, f"{prompt_name}_snap_prefill")
        save_snapshot(engine, snap_path, metadata={
            "snapshot_point": "prefill",
            "prompt_name": prompt_name,
            "first_token": first_token,
            "prompt_tokens": prompt_tokens,
            "decode_tokens_so_far": [first_token],
        })

    # Decode
    print(f"\n[golden] Decoding up to {max_tokens} tokens (greedy)...")
    generated = [first_token]
    t0 = time.time()
    for gi in range(max_tokens - 1):
        if engine.pos >= engine.ctx:
            print(f"[golden] Context full at pos={engine.pos}")
            break
        fed_tok = generated[-1]
        next_id, logits = engine._step(fed_tok, engine.pos)
        engine.pos += 1
        engine.token_history.append(fed_tok)
        generated.append(next_id)

        decode_count = len(generated)

        # Check for snapshot point
        for sp in snapshot_points:
            if isinstance(sp, int) and sp == decode_count:
                snap_path = os.path.join(golden_dir,
                                          f"{prompt_name}_snap_tok{sp}")
                save_snapshot(engine, snap_path, metadata={
                    "snapshot_point": f"token_{sp}",
                    "prompt_name": prompt_name,
                    "first_token": first_token,
                    "prompt_tokens": prompt_tokens,
                    "decode_tokens_so_far": generated[:],
                })

        if next_id in engine.stop_ids:
            print(f"[golden] EOS at token {decode_count}")
            break

    decode_time = time.time() - t0
    tps = len(generated) / max(decode_time, 1e-9)
    print(f"[golden] Generated {len(generated)} tokens in "
          f"{decode_time:.1f}s ({tps:.1f} tok/s)")

    # Save golden result
    golden_text = engine.tokenizer.decode(generated, skip_special_tokens=True)
    result = {
        "prompt_name": prompt_name,
        "prompt_text": prompt_text,
        "prompt_tokens": prompt_tokens,
        "generated_ids": generated,
        "generated_text": golden_text,
        "final_pos": engine.pos,
        "final_rope_offset": engine.rope_offset,
        "max_tokens": max_tokens,
        "snapshot_points": [str(s) for s in snapshot_points],
    }
    result_path = os.path.join(golden_dir, f"{prompt_name}_golden.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[golden] Text: {repr(golden_text[:200])}")
    print(f"[golden] Saved to {result_path}")
    return result


# ── Resume and compare ───────────────────────────────────────────────

def resume_and_compare(engine, snapshot_path, golden_path, max_tokens):
    """Restore state from snapshot, continue decoding, compare with golden.

    Returns (success: bool, report: dict).
    """
    # Load golden
    with open(golden_path, "r") as f:
        golden = json.load(f)
    golden_ids = golden["generated_ids"]

    # Load snapshot metadata to know where we left off
    with open(os.path.join(snapshot_path, "meta.json"), "r") as f:
        snap_meta = json.load(f)
    decode_so_far = snap_meta.get("decode_tokens_so_far", [])
    first_token = snap_meta.get("first_token")
    snapshot_point = snap_meta.get("snapshot_point", "unknown")
    resume_offset = len(decode_so_far)

    print(f"\n{'='*60}")
    print(f"[resume] Snapshot: {snapshot_point}")
    print(f"[resume] Tokens already decoded: {resume_offset}")
    print(f"[resume] Golden total tokens: {len(golden_ids)}")
    print(f"[resume] Will decode {len(golden_ids) - resume_offset} more tokens")

    # Restore state
    restore_state(engine, snapshot_path)

    # Validate decode_so_far matches golden prefix
    if decode_so_far != golden_ids[:resume_offset]:
        print(f"[resume] ERROR: decode_so_far doesn't match golden prefix!")
        for i, (a, b) in enumerate(zip(decode_so_far, golden_ids)):
            if a != b:
                print(f"  First mismatch at index {i}: snapshot={a} vs golden={b}")
                break
        return False, {"error": "prefix_mismatch"}

    # Continue decoding from where snapshot left off
    remaining_tokens = len(golden_ids) - resume_offset
    if remaining_tokens <= 0:
        print("[resume] Nothing to decode — snapshot is at end")
        return True, {"match": True, "tokens_compared": 0}

    # The last token in decode_so_far is the one we need to feed next
    if resume_offset > 0:
        current_token = decode_so_far[-1]
    else:
        current_token = first_token

    resumed_ids = list(decode_so_far)
    mismatches = []

    print(f"[resume] Starting decode at pos={engine.pos}, "
          f"feeding token={current_token}...")

    t0 = time.time()
    for gi in range(remaining_tokens):
        if engine.pos >= engine.ctx:
            print(f"[resume] Context full at pos={engine.pos}")
            break

        fed_tok = current_token
        next_id, logits = engine._step(fed_tok, engine.pos)
        engine.pos += 1
        engine.token_history.append(fed_tok)
        resumed_ids.append(next_id)

        # Compare with golden
        golden_idx = len(resumed_ids) - 1
        if golden_idx < len(golden_ids):
            expected = golden_ids[golden_idx]
            if next_id != expected:
                mismatches.append({
                    "step": golden_idx,
                    "expected": expected,
                    "got": next_id,
                    "expected_text": engine.tokenizer.decode([expected]),
                    "got_text": engine.tokenizer.decode([next_id]),
                })
                if len(mismatches) <= 5:
                    print(f"  MISMATCH at step {golden_idx}: "
                          f"expected={expected} ({repr(engine.tokenizer.decode([expected]))}) "
                          f"got={next_id} ({repr(engine.tokenizer.decode([next_id]))})")

        current_token = next_id
        if next_id in engine.stop_ids:
            break

    decode_time = time.time() - t0
    tokens_compared = len(resumed_ids) - resume_offset
    tps = tokens_compared / max(decode_time, 1e-9)

    # Build report
    resumed_text = engine.tokenizer.decode(resumed_ids, skip_special_tokens=True)
    golden_text = golden["generated_text"]

    exact_match = (resumed_ids == golden_ids[:len(resumed_ids)])
    text_match = (resumed_text == golden_text[:len(resumed_text)])

    report = {
        "snapshot_point": snapshot_point,
        "tokens_compared": tokens_compared,
        "total_resumed": len(resumed_ids),
        "total_golden": len(golden_ids),
        "exact_id_match": exact_match,
        "text_match": text_match,
        "num_mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "mismatches": mismatches[:10],
        "decode_tps": round(tps, 1),
        "resumed_text_preview": resumed_text[:200],
        "golden_text_preview": golden_text[:200],
    }

    if exact_match:
        print(f"\n[resume] ✅ EXACT MATCH — {tokens_compared} tokens compared, "
              f"all identical")
    else:
        print(f"\n[resume] ❌ MISMATCH — {len(mismatches)} mismatches in "
              f"{tokens_compared} tokens")
        if mismatches:
            m = mismatches[0]
            print(f"  First divergence at step {m['step']}: "
                  f"expected token {m['expected']} got {m['got']}")

    print(f"[resume] Decode: {tokens_compared} tok in {decode_time:.1f}s "
          f"({tps:.1f} tok/s)")

    return exact_match, report


# ── Helpers ──────────────────────────────────────────────────────────

def _array_hash(arr):
    """Short hash of numpy array for quick comparison."""
    return hashlib.md5(arr.tobytes()).hexdigest()


def _parse_snapshot_points(s):
    """Parse comma-separated snapshot points like 'prefill,1,8,16'."""
    points = []
    for p in s.split(","):
        p = p.strip()
        if p == "prefill":
            points.append("prefill")
        else:
            points.append(int(p))
    return points


# ── Commands ─────────────────────────────────────────────────────────

def cmd_golden(args):
    """Run golden reference and save snapshots."""
    engine = create_engine(args.model_dir, args.hf_path,
                           args.num_chunks, args.ctx)
    snapshot_points = _parse_snapshot_points(args.snapshot_at)
    prompt = TEST_PROMPTS[args.prompt]
    if isinstance(prompt, list):
        prompt = prompt[0]  # use first message for golden
    result = run_golden(engine, prompt, args.max_tokens,
                        snapshot_points, GOLDEN_DIR, args.prompt)
    if result:
        print(f"\n[golden] Complete. {len(result['generated_ids'])} tokens generated.")


def cmd_resume_same(args):
    """Resume from snapshot in same process (sanity check)."""
    engine = create_engine(args.model_dir, args.hf_path,
                           args.num_chunks, args.ctx)
    snapshot_points = _parse_snapshot_points(args.snapshot_at)
    prompt = TEST_PROMPTS[args.prompt]
    if isinstance(prompt, list):
        prompt = prompt[0]

    # Run golden first
    result = run_golden(engine, prompt, args.max_tokens,
                        snapshot_points, GOLDEN_DIR, args.prompt)
    if not result:
        print("[error] Golden run failed")
        return

    # Now resume from each snapshot point
    all_pass = True
    for sp in snapshot_points:
        sp_str = sp if sp == "prefill" else f"tok{sp}"
        snap_path = os.path.join(GOLDEN_DIR, f"{args.prompt}_snap_{sp_str}")
        golden_path = os.path.join(GOLDEN_DIR, f"{args.prompt}_golden.json")
        if not os.path.exists(snap_path):
            print(f"\n[skip] No snapshot at {sp_str}")
            continue

        # Reset engine to fresh state, then restore from snapshot
        engine._reset_states()
        success, report = resume_and_compare(
            engine, snap_path, golden_path, args.max_tokens)
        all_pass = all_pass and success

        # Save report
        report_path = os.path.join(GOLDEN_DIR,
                                    f"{args.prompt}_resume_same_{sp_str}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    if all_pass:
        print("✅ ALL SAME-PROCESS RESUME TESTS PASSED")
    else:
        print("❌ SOME SAME-PROCESS RESUME TESTS FAILED")


def cmd_snapshot(args):
    """Run golden and save snapshots (for use by resume-fresh)."""
    # Same as golden but ensures snapshot is saved
    cmd_golden(args)


def cmd_resume_fresh(args):
    """Resume from previously saved snapshot in a FRESH process."""
    engine = create_engine(args.model_dir, args.hf_path,
                           args.num_chunks, args.ctx)

    snapshot_points = _parse_snapshot_points(args.snapshot_at)
    all_pass = True

    for sp in snapshot_points:
        sp_str = sp if sp == "prefill" else f"tok{sp}"
        snap_path = os.path.join(GOLDEN_DIR, f"{args.prompt}_snap_{sp_str}")
        golden_path = os.path.join(GOLDEN_DIR, f"{args.prompt}_golden.json")
        if not os.path.exists(snap_path):
            print(f"\n[skip] No snapshot at {sp_str}")
            continue
        if not os.path.exists(golden_path):
            print(f"\n[error] No golden reference at {golden_path}")
            return

        success, report = resume_and_compare(
            engine, snap_path, golden_path, args.max_tokens)
        all_pass = all_pass and success

        report_path = os.path.join(GOLDEN_DIR,
                                    f"{args.prompt}_resume_fresh_{sp_str}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    if all_pass:
        print("✅ ALL FRESH-PROCESS RESUME TESTS PASSED")
    else:
        print("❌ SOME FRESH-PROCESS RESUME TESTS FAILED")


def cmd_auto(args):
    """Automated test: golden → same-process resume → fresh-process resume."""
    # Build common CLI args shared across all phases
    common_args = [
        "--model-dir", args.model_dir or DEFAULT_MODEL_DIR,
        "--hf-path", args.hf_path or DEFAULT_HF_PATH,
        "--num-chunks", str(args.num_chunks or DEFAULT_NUM_CHUNKS),
        "--ctx", str(args.ctx or DEFAULT_CTX),
        "--max-tokens", str(args.max_tokens),
        "--prompt", args.prompt,
        "--snapshot-at", args.snapshot_at,
    ]

    def _run_phase(phase_name, command):
        print(f"\n{'='*60}")
        print(f"[auto] {phase_name}")
        print(f"{'='*60}")
        cmd = [sys.executable, __file__, command] + common_args
        r = subprocess.run(cmd, cwd=_REPO_ROOT)
        if r.returncode != 0:
            print(f"[auto] {phase_name} FAILED (exit={r.returncode})")
        return r.returncode == 0

    ok = _run_phase("PHASE 1: Golden run + snapshots", "golden")
    if not ok:
        return

    # Phase 2 loads model fresh, runs golden internally, then resumes
    # from snapshot in same process — a sanity check.
    # Skip this for speed; the key test is phase 3.
    # _run_phase("PHASE 2: Same-process resume", "resume-same")

    _run_phase("PHASE 3: Fresh-process resume (TRUE RESTART)", "resume-fresh")

    print(f"\n{'='*60}")
    print(f"[auto] All phases complete. Check reports in {GOLDEN_DIR}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Raw KV-cache persistence experiment")
    parser.add_argument("command",
                        choices=["golden", "resume-same", "resume-fresh",
                                 "snapshot", "auto"],
                        help="Command to run")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf-path", default=DEFAULT_HF_PATH)
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--prompt", default="short",
                        choices=list(TEST_PROMPTS.keys()),
                        help="Test prompt name")
    parser.add_argument("--snapshot-at", default="prefill,1,8,16",
                        help="Comma-separated snapshot points "
                             "(prefill, or token count like 1,8,16)")
    args = parser.parse_args()

    {
        "golden": cmd_golden,
        "resume-same": cmd_resume_same,
        "resume-fresh": cmd_resume_fresh,
        "snapshot": cmd_snapshot,
        "auto": cmd_auto,
    }[args.command](args)


if __name__ == "__main__":
    main()
