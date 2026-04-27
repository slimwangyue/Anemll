#!/usr/bin/env python3
"""Quick check: does the INFER function update the KV state?"""
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

engine._reset_states()

# Check initial state
k0 = np.array(engine.states[0].read_state(name='k_cache')).astype(np.float32)
print(f"After reset: k_cache[0][0:5] L2={np.linalg.norm(k0[:,:,:5,:]):.6f}")

# Run one token via INFER (step_kv_only)
engine._step_kv_only(1234, 0)
k1 = np.array(engine.states[0].read_state(name='k_cache')).astype(np.float32)
pos0_l2 = np.linalg.norm(k1[:, :, 0, :])
pos1_l2 = np.linalg.norm(k1[:, :, 1, :])
print(f"After 1 infer token: pos[0] L2={pos0_l2:.6f}  pos[1] L2={pos1_l2:.6f}")

# Run one more token
engine._step_kv_only(5678, 1)
k2 = np.array(engine.states[0].read_state(name='k_cache')).astype(np.float32)
pos0_l2 = np.linalg.norm(k2[:, :, 0, :])
pos1_l2 = np.linalg.norm(k2[:, :, 1, :])
pos2_l2 = np.linalg.norm(k2[:, :, 2, :])
print(f"After 2 infer tokens: pos[0] L2={pos0_l2:.6f}  pos[1] L2={pos1_l2:.6f}  pos[2] L2={pos2_l2:.6f}")

# Now test PREFILL
engine._reset_states()
messages = [{"role": "user", "content": "Hello world"}]
tokens = engine._tokenize_messages(messages, enable_thinking=False)
_ = engine._batch_prefill(tokens[:10], 0)
k3 = np.array(engine.states[0].read_state(name='k_cache')).astype(np.float32)
pos0_l2 = np.linalg.norm(k3[:, :, 0, :])
pos5_l2 = np.linalg.norm(k3[:, :, 5, :])
pos255_l2 = np.linalg.norm(k3[:, :, 255, :])
print(f"After prefill 10 tokens: pos[0] L2={pos0_l2:.6f}  pos[5] L2={pos5_l2:.6f}  pos[255] L2={pos255_l2:.6f}")

# Now try infer on SAME state
engine._step_kv_only(1234, engine.pos)
k4 = np.array(engine.states[0].read_state(name='k_cache')).astype(np.float32)
pos0_l2 = np.linalg.norm(k4[:, :, 0, :])
pos10_l2 = np.linalg.norm(k4[:, :, 10, :])
print(f"After prefill+infer: pos[0] L2={pos0_l2:.6f}  pos[10] L2={pos10_l2:.6f} (infer wrote here)")

print("\nConclusion:")
if pos0_l2 > 0 and pos10_l2 > 0:
    print("  Both prefill and infer update the state correctly.")
elif pos0_l2 == 0 and pos10_l2 > 0:
    print("  PREFILL does NOT update state, but INFER does!")
    print("  This is the root cause of multi-block batch prefill failure.")
else:
    print(f"  Unexpected: pos0={pos0_l2:.6f} pos10={pos10_l2:.6f}")
