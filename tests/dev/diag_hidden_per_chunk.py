#!/usr/bin/env python3
"""Compare hidden states chunk-by-chunk between batch-tail and sequential-tail.
Shows where the divergence actually happens."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct
from chat_server import ChatEngine

engine = ChatEngine(
    model_dir='/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.3',
    hf_path='/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B',
    ctx=4096, num_chunks=9,
    compute_unit=ct.ComputeUnit.CPU_ONLY,
)
engine.load()

prompt = (
    "Please translate the following passage into Chinese. After translation, "
    "explicitly identify any terms or concepts that are unclear, undefined, or "
    "potentially non-standard, and explain why.\n\nText:\n\n"
    "In recent work on efficient transformer inference, a method referred to as "
    "Hierarchical Context Distillation (HCD) has been proposed to address the "
    "growing cost of long-context processing. The central idea behind HCD is to "
    "iteratively compress intermediate representations across layers, such that "
    "only a subset of context-critical tokens are propagated forward. Unlike "
    "standard token pruning methods, HCD claims to preserve global coherence by "
    "maintaining a secondary structure known as the Residual Context Graph (RCG), "
    "which encodes long-range dependencies in a compressed form.\n\n"
    "The RCG is constructed during the prefill stage by computing pairwise "
    "affinity scores between tokens using a function termed bidirectional "
    "semantic alignment (BSA). Tokens with high mutual affinity are grouped into "
    "clusters, and each cluster is represented by a centroid embedding. During "
    "decoding, these centroid embeddings are selectively expanded back into "
    "token-level representations through a process called context unfolding, "
    "which is guided by a learned gating mechanism.\n\n"
    "Additionally, HCD introduces a normalization technique called "
    "variance-preserving attention scaling (VPAS). This mechanism adjusts "
    "attention weights based on the estimated variance of token embeddings "
    "within each cluster, ensuring that clusters with higher internal diversity "
    "are not underrepresented. According to the authors, VPAS stabilizes "
    "training and improves performance on tasks requiring fine-grained reasoning.\n\n"
    "However, several aspects of the method remain unclear. The exact "
    "mathematical formulation of BSA is not provided, and it is not specified "
    "whether the clustering process is differentiable. Furthermore, the "
    "interaction between RCG and standard attention layers is only described at "
    "a high level, leaving open questions about implementation feasibility.\n\n"
    "Despite these ambiguities, preliminary experiments suggest that HCD "
    "achieves up to 1.8x speedup on long-sequence benchmarks while maintaining "
    "comparable accuracy to baseline transformer models. The authors note that "
    "further investigation is required to validate the robustness of the "
    "approach under different input distributions."
)

messages = [{'role': 'user', 'content': prompt}]
tokens = engine._tokenize_messages(messages, enable_thinking=False)
bs = engine._prefill_bs
print(f"Tokens: {len(tokens)}, bs={bs}")
assert len(tokens) > bs, f"Need > {bs} tokens for a tail, got {len(tokens)}"

tail = tokens[bs:]
valid_len = len(tail)
print(f"Block1: {bs}, tail: {valid_len}")

def cos_sim(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    return np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-30)

# ═══ Step 1: Block 1 via batch (shared) ═══
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(tokens[:bs], 0)

# Save all state
import copy
saved_convs = [c.copy() for c in engine.lin_convs]
saved_recs = [r.copy() for r in engine.lin_recs]
saved_pos = engine.pos

# ═══ Path C: tail via BATCH prefill ═══
# Manually run batch prefill but capture hidden after each chunk
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(tokens[:bs], 0)

input_ids = engine._batch_tok_buf
input_ids[0, :] = 0
input_ids[0, :valid_len] = tail
hidden_C = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
if valid_len < bs:
    hidden_C[:, valid_len:, :] = 0.0

mask = engine._batch_mask_buf
mask[:, :, :, :] = -65504.0
for i in range(valid_len):
    mask[0, 0, i, :bs + i + 1] = 0
for i in range(valid_len, bs):
    mask[0, 0, i, 0] = 0.0

pos_ids = engine._batch_pos_buf
pos_ids[:valid_len] = np.arange(bs + engine.rope_offset, bs + engine.rope_offset + valid_len, dtype=np.int32)
pos_ids[valid_len:] = 0
cur_pos = engine._batch_cur_buf
cur_pos[0] = bs
valid_len_arr = engine._valid_len_buf
valid_len_arr[0] = valid_len

hiddens_C = [hidden_C[:, valid_len-1:valid_len, :].copy()]
print(f"\n{'='*60}")
print(f"  BATCH TAIL: hidden states after each chunk (last valid token)")
print(f"{'='*60}")
for ci in range(engine.num_chunks):
    inp = {
        "hidden_states": hidden_C.astype(np.float16),
        "position_ids": pos_ids,
        "causal_mask": mask,
        "current_pos": cur_pos,
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
        "valid_len": valid_len_arr,
    }
    out = engine.prefills[ci].predict(inp, state=engine.states[ci])
    hidden_C = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
    if valid_len < bs:
        hidden_C[:, valid_len:, :] = 0.0
    h_last = hidden_C[:, valid_len-1:valid_len, :].copy()
    norm_h = np.linalg.norm(h_last.astype(np.float64))
    print(f"  chunk{ci}: hidden norm={norm_h:.4f}")
    hiddens_C.append(h_last)

# ═══ Path A: tail via SEQUENTIAL ═══
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(tokens[:bs], 0)

# Process tail tokens one by one, capture hidden after each chunk
# But we can't easily get per-chunk hidden from _step. 
# Instead, let's process the ENTIRE tail sequentially and just compare final result.
for ti, tok_id in enumerate(tail):
    is_last = (ti == len(tail) - 1)
    if is_last:
        tok_A, _ = engine._step(tok_id, engine.pos)
    else:
        engine._step_kv_only(tok_id, engine.pos)
    engine.pos += 1

# Now do a per-chunk comparison by running JUST the last token through each chunk
# after sequential has primed the state
# This is complicated. Instead, let's compare: run last token of tail through 
# single-token inference after block1+sequential-tail-minus-1, and capture hidden per chunk.
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(tokens[:bs], 0)

# Process all but last tail token sequentially
for ti, tok_id in enumerate(tail[:-1]):
    engine._step_kv_only(tok_id, engine.pos)
    engine.pos += 1

# Now run last tail token and capture per-chunk hidden
last_tok = tail[-1]
embed_out = list(engine.embed.predict({"input_ids": np.array([[last_tok]], dtype=np.int32)}).values())[0]
hidden_A = embed_out
last_pos = engine.pos

hiddens_A = [hidden_A.copy()]
print(f"\n{'='*60}")
print(f"  SEQUENTIAL TAIL: hidden of last token after each chunk")
print(f"{'='*60}")

for ci in range(engine.num_chunks):
    # Build single-token inputs
    mask_s = engine._mask_buf
    mask_s[:, :, :, :] = -65504.0
    mask_s[0, 0, 0, :last_pos+1] = 0.0
    
    inp = {
        "hidden_states": hidden_A.astype(np.float16),
        "position_ids": np.array([last_pos + engine.rope_offset], dtype=np.int32),
        "causal_mask": mask_s,
        "current_pos": np.array([last_pos], dtype=np.int32),
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
    }
    out = engine.ffns[ci].predict(inp, state=engine.states[ci])
    hidden_A = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
    h_a = hidden_A.copy()
    norm_a = np.linalg.norm(h_a.astype(np.float64))
    print(f"  chunk{ci}: hidden norm={norm_a:.4f}")
    hiddens_A.append(h_a)

# Compare
print(f"\n{'='*60}")
print(f"  COMPARISON: batch vs sequential hidden of last valid token")
print(f"{'='*60}")
for i, (ha, hc) in enumerate(zip(hiddens_A, hiddens_C)):
    cos = cos_sim(ha, hc)
    mse = np.mean((ha.astype(np.float64) - hc.astype(np.float64))**2)
    mad = np.max(np.abs(ha.astype(np.float64) - hc.astype(np.float64)))
    label = f"embed" if i == 0 else f"chunk{i-1}"
    print(f"  {label}: cos={cos:.6f}  MSE={mse:.2e}  max_abs_diff={mad:.4f}")
