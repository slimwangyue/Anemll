#!/usr/bin/env python3
"""Verify KV cache state persistence across prefill calls.

Hypothesis: The prefill model may read initial zeros instead of the 
current state on the second call, losing block1's KV entries.

Tests:
1) After block1 prefill, read state → check positions 0..255 non-zero
2) After block2 prefill, read state → check positions 0..255 STILL non-zero
   If block1's data at 0..255 becomes zero after block2, state is corrupted.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))

import argparse, numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--ffn-dir", default=None)
parser.add_argument("--chunks", type=int, default=9)
parser.add_argument("--ctx", type=int, default=4096)
args = parser.parse_args()

import coremltools as ct
from chat_server import ChatEngine

engine = ChatEngine(
    model_dir=args.model_dir,
    hf_path=args.tokenizer,
    ctx=args.ctx,
    num_chunks=args.chunks,
    ffn_dir=args.ffn_dir,
    compute_unit=ct.ComputeUnit.CPU_ONLY,
)
engine.load()
bs = engine._prefill_bs

# Build prompt with bs+100 tokens
BASE_PROMPT = "Explain in detail the history of artificial intelligence from its inception to modern day. " * 20
messages = [{"role": "user", "content": BASE_PROMPT}]
all_tokens = engine._tokenize_messages(messages, enable_thinking=False)
while len(all_tokens) < bs + 100:
    BASE_PROMPT += " Describe the future of AI. " * 10
    messages = [{"role": "user", "content": BASE_PROMPT}]
    all_tokens = engine._tokenize_messages(messages, enable_thinking=False)
tokens = all_tokens[:bs + 100]
block1 = tokens[:bs]
tail = tokens[bs:]
print(f"Total: {len(tokens)}, block1: {len(block1)}, tail: {len(tail)}, bs: {bs}")


def read_kv_state(engine, chunk_idx=0):
    """Read KV cache from CoreML state and report norms at key positions."""
    state = engine.states[chunk_idx]
    k_cache = None
    v_cache = None
    for name in ['k_cache', 'v_cache']:
        try:
            arr = state.read_state(name=name)
            arr_np = np.array(arr).astype(np.float32)
            if name == 'k_cache':
                k_cache = arr_np
            else:
                v_cache = arr_np
        except Exception as e:
            print(f"  Cannot read state '{name}': {e}")
    return k_cache, v_cache


def report_kv(k_cache, label, positions_to_check):
    """Report L2 norms of KV cache at specified position ranges."""
    if k_cache is None:
        print(f"  {label}: could not read KV cache")
        return
    print(f"  {label}: k_cache shape={k_cache.shape}")
    # k_cache shape: (num_kv_layers, num_heads, state_length, head_dim)
    for start, end, name in positions_to_check:
        slice_data = k_cache[:, :, start:end, :]
        norm = np.linalg.norm(slice_data)
        max_abs = np.max(np.abs(slice_data))
        nonzero = np.count_nonzero(slice_data)
        total = slice_data.size
        print(f"    positions [{start}:{end}] ({name}): L2={norm:.4f} max_abs={max_abs:.6f} "
              f"nonzero={nonzero}/{total} ({100*nonzero/total:.1f}%)")


engine._reset_states()

# ══════════════════════════════════════════════════════════════════════
# Check 1: State after reset (should be all zeros)
# ══════════════════════════════════════════════════════════════════════
print("\n=== CHECK 1: State after reset ===")
k_cache, _ = read_kv_state(engine, 0)
if k_cache is not None:
    report_kv(k_cache, "After reset", [
        (0, 256, "block1 region"),
        (256, 512, "block2 region"),
    ])
else:
    print("  Could not read state directly. Trying predict-based probe...")
    # Alternate: run a simple prediction and check if output makes sense
    
# ══════════════════════════════════════════════════════════════════════
# BLOCK 1: Process first 256 tokens
# ══════════════════════════════════════════════════════════════════════
print("\n=== BLOCK 1: Processing 256 tokens via prefill ===")
_ = engine._batch_prefill(block1, 0)
print(f"  pos after block1: {engine.pos}")

print("\n=== CHECK 2: State after block1 ===")
k_cache_after_b1, _ = read_kv_state(engine, 0)
if k_cache_after_b1 is not None:
    report_kv(k_cache_after_b1, "After block1", [
        (0, 256, "block1 region - should be NON-ZERO"),
        (256, 512, "block2 region - should be zero"),
    ])
    # Save block1's KV data for later comparison
    block1_kv_data = k_cache_after_b1[:, :, :256, :].copy()

# ══════════════════════════════════════════════════════════════════════
# BLOCK 2 (TAIL): Process next 100 tokens
# ══════════════════════════════════════════════════════════════════════
print("\n=== BLOCK 2: Processing 100 tokens (tail) via prefill ===")
tok = engine._batch_prefill(tail, engine.pos)
print(f"  pos after block2: {engine.pos}")
print(f"  First token: {tok} = '{engine.tokenizer.decode([tok])}'")

print("\n=== CHECK 3: State after block2 (CRITICAL) ===")
k_cache_after_b2, _ = read_kv_state(engine, 0)
if k_cache_after_b2 is not None:
    report_kv(k_cache_after_b2, "After block2", [
        (0, 256, "block1 region - MUST still be NON-ZERO!"),
        (256, 356, "block2 valid region - should be non-zero"),
        (356, 512, "block2 padding region - may be garbage"),
    ])
    
    # Compare block1 region before and after block2
    block1_kv_after = k_cache_after_b2[:, :, :256, :].copy()
    diff = np.max(np.abs(block1_kv_data - block1_kv_after))
    cos = np.dot(block1_kv_data.flatten(), block1_kv_after.flatten()) / (
        np.linalg.norm(block1_kv_data) * np.linalg.norm(block1_kv_after) + 1e-12)
    is_zero = np.all(block1_kv_after == 0)
    print(f"\n  *** BLOCK1 KV PRESERVATION CHECK ***")
    print(f"  block1 region max_diff: {diff:.6f}")
    print(f"  block1 region cosine: {cos:.6f}")
    print(f"  block1 region all zeros: {is_zero}")
    if is_zero:
        print(f"  *** CONFIRMED: Block2 prefill WIPED block1's KV cache! ***")
        print(f"  *** Root cause: prefill model reads initial zeros, not current state ***")
    elif diff < 0.001:
        print(f"  Block1 KV preserved correctly. Issue is elsewhere.")
    else:
        print(f"  Block1 KV partially corrupted (diff={diff:.6f})")
else:
    print("  Could not read state. Cannot verify hypothesis.")
    print("  Consider using PyTorch model for verification.")
