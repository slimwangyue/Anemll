#!/usr/bin/env python3
"""Test: does valid_len masking in the compiled CoreML model work correctly?

Compares the hidden states at valid positions for the SAME tokens processed via:
- Run A: valid_len=256 (all 256 tokens valid, no padding)
- Run B: valid_len=164 (first 164 tokens valid, rest padding zeros)

If valid_len masking works correctly, the hidden states at positions 0-163
should be IDENTICAL between Run A and Run B.

If they differ, the valid_len masking is corrupting valid tokens."""
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
tail_tokens = tokens[bs:]
valid_len = len(tail_tokens)
print(f"Tokens: {len(tokens)}, bs={bs}, tail: {valid_len}")

def cos_sim(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    if a_f.shape[0] == 0 or b_f.shape[0] == 0:
        return float('nan')
    return np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-30)

# Build the input for the tail block (same for both runs)
input_ids = engine._batch_tok_buf
pos_ids = engine._batch_pos_buf
cur_pos = engine._batch_cur_buf
valid_len_arr = engine._valid_len_buf
mask = engine._batch_mask_buf

# Same tokens for first valid_len positions
input_ids[0, :] = 0
input_ids[0, :valid_len] = tail_tokens

# ═══ RUN A: valid_len=256 (full batch, fill remaining with dummy tokens) ═══
print(f"\n{'='*60}")
print(f"  RUN A: valid_len={bs} (FULL, fill remaining with repeat tokens)")
print(f"{'='*60}")
engine._reset_states()
engine.pos = 0
# First process block1 via batch prefill
_ = engine._batch_prefill(tokens[:bs], 0)
# Save states after block1
import copy
saved_convs_A = [c.copy() for c in engine.lin_convs]
saved_recs_A = [r.copy() for r in engine.lin_recs]
# Also need to save MLState — but we can't easily copy MLState.
# Instead, we'll reset and re-process block1 for each run.

# Now process the tail with valid_len=256 (all valid)
# Fill padding positions with the same tokens repeated
input_ids[0, valid_len:] = tail_tokens[:bs - valid_len] if valid_len < bs else 0

# Embed
hidden_A = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
# NO re-zero — all positions are "valid"

# Build mask for ALL 256 valid
mask[:, :, :, :] = -65504.0
for i in range(bs):
    mask[0, 0, i, :bs + i + 1] = 0

# Position IDs for ALL 256
pos_ids[:bs] = np.arange(bs + engine.rope_offset, bs + engine.rope_offset + bs, dtype=np.int32)

cur_pos[0] = bs
valid_len_arr[0] = bs  # all valid

hiddens_A = []
for ci in range(engine.num_chunks):
    inp = {
        "hidden_states": hidden_A.astype(np.float16),
        "position_ids": pos_ids,
        "causal_mask": mask,
        "current_pos": cur_pos,
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
        "valid_len": valid_len_arr,
    }
    out = engine.prefills[ci].predict(inp, state=engine.states[ci])
    hidden_A = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
    # No re-zero — all valid
    # Save hidden at the LAST of the original valid positions (163)
    h = hidden_A[:, valid_len-1:valid_len, :].copy()
    norm_h = np.linalg.norm(h.astype(np.float64))
    print(f"  chunk{ci}: hidden[{valid_len-1}] norm={norm_h:.4f}")
    hiddens_A.append(h)

# ═══ RUN B: valid_len=164 (partial batch, padding zeros) ═══
print(f"\n{'='*60}")
print(f"  RUN B: valid_len={valid_len} (PARTIAL, padding zeros)")
print(f"{'='*60}")
engine._reset_states()
engine.pos = 0
# Re-process block1
_ = engine._batch_prefill(tokens[:bs], 0)

# Now process the tail with valid_len=164
input_ids[0, :] = 0
input_ids[0, :valid_len] = tail_tokens

# Embed
hidden_B = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
# Re-zero padding
hidden_B[:, valid_len:, :] = 0.0

# Build mask for 164 valid
mask[:, :, :, :] = -65504.0
for i in range(valid_len):
    mask[0, 0, i, :bs + i + 1] = 0
for i in range(valid_len, bs):
    mask[0, 0, i, 0] = 0.0

# Position IDs
pos_ids[:valid_len] = np.arange(bs + engine.rope_offset, bs + engine.rope_offset + valid_len, dtype=np.int32)
pos_ids[valid_len:] = 0

cur_pos[0] = bs
valid_len_arr[0] = valid_len

hiddens_B = []
for ci in range(engine.num_chunks):
    inp = {
        "hidden_states": hidden_B.astype(np.float16),
        "position_ids": pos_ids,
        "causal_mask": mask,
        "current_pos": cur_pos,
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
        "valid_len": valid_len_arr,
    }
    out = engine.prefills[ci].predict(inp, state=engine.states[ci])
    hidden_B = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
    # Re-zero padding
    if valid_len < bs:
        hidden_B[:, valid_len:, :] = 0.0
    h = hidden_B[:, valid_len-1:valid_len, :].copy()
    norm_h = np.linalg.norm(h.astype(np.float64))
    print(f"  chunk{ci}: hidden[{valid_len-1}] norm={norm_h:.4f}")
    hiddens_B.append(h)

# ═══ COMPARISON ═══
print(f"\n{'='*60}")
print(f"  COMPARISON: full-valid vs partial-padded at position {valid_len-1}")
print(f"{'='*60}")
for ci, (ha, hb) in enumerate(zip(hiddens_A, hiddens_B)):
    cos = cos_sim(ha, hb)
    mse = np.mean((ha.astype(np.float64) - hb.astype(np.float64))**2)
    mad = np.max(np.abs(ha.astype(np.float64) - hb.astype(np.float64)))
    print(f"  chunk{ci}: cos={cos:.6f}  MSE={mse:.2e}  max_abs_diff={mad:.4f}")
