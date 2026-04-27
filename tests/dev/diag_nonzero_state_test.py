#!/usr/bin/env python3
"""Test: does NON-ZERO initial state amplify chunked-vs-recurrent divergence?

Compare PREFILL vs INFER for a FULL batch (no padding), but WITH block1
processing first (non-zero initial recurrent states).

If cos drops to ~0.7 (like the partial batch), the root cause is the feedback
amplification in the delta rule with non-zero initial state.
If cos stays at ~0.999 (like the zero-state full batch), then the issue is
specific to partial batches."""
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

# Need 512+ tokens (2 full batches)
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
    "approach under different input distributions.\n\n"
    "A particularly noteworthy aspect of this work is the claim that HCD can be "
    "applied to any decoder-only transformer architecture without significant "
    "modifications to the training procedure. The authors demonstrate this by "
    "applying the method to both GPT-style and Llama-style models, reporting "
    "consistent improvements in inference latency. However, the experiments "
    "are limited to models with fewer than 7 billion parameters, and it remains "
    "unclear whether the benefits scale to larger models.\n\n"
    "The paper also introduces a new benchmark called LongReason, which "
    "specifically tests a model's ability to maintain logical consistency "
    "over extended contexts. The benchmark consists of multi-hop reasoning "
    "tasks where the relevant information is distributed across different "
    "segments of a long input sequence. The authors report that HCD "
    "outperforms vanilla attention and sliding-window attention on LongReason "
    "while using significantly less memory.\n\n"
    "Critics of the approach have noted that the Residual Context Graph "
    "introduces additional computational overhead during the prefill phase, "
    "which may offset some of the latency gains during decoding. Additionally, "
    "the reliance on learned gating mechanisms raises concerns about "
    "generalization to out-of-distribution inputs. These concerns remain "
    "unaddressed in the current version of the paper."
)

messages = [{'role': 'user', 'content': prompt}]
tokens = engine._tokenize_messages(messages, enable_thinking=False)
bs = engine._prefill_bs
print(f"Total tokens: {len(tokens)}, bs={bs}")

# Need at least 2 * bs tokens
if len(tokens) < 2 * bs:
    # Duplicate to get enough tokens
    tokens = tokens * 3
    tokens = tokens[:2 * bs + 100]
    print(f"Extended to: {len(tokens)} tokens")

block1 = tokens[:bs]
block2 = tokens[bs:2*bs]
print(f"Block1: {bs} tokens, Block2: {bs} tokens")

def cos_sim(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    if a_f.shape[0] == 0 or b_f.shape[0] == 0:
        return float('nan')
    return np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-30)

# ═══ PATH P: block1 via batch, then block2 via PREFILL ═══
print(f"\n{'='*60}")
print(f"  PATH P: block1 + block2 via PREFILL (non-zero initial state)")
print(f"{'='*60}")
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
print(f"  Block1 done. pos={engine.pos}")

# Now process block2 via PREFILL model
input_ids = engine._batch_tok_buf
input_ids[0, :] = 0
input_ids[0, :bs] = block2
hidden_P = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]

mask = engine._batch_mask_buf
mask[:, :, :, :] = -65504.0
for i in range(bs):
    mask[0, 0, i, :bs + i + 1] = 0

pos_ids = engine._batch_pos_buf
pos_ids[:bs] = np.arange(bs + engine.rope_offset, 2*bs + engine.rope_offset, dtype=np.int32)

cur_pos = engine._batch_cur_buf
cur_pos[0] = bs
valid_len_arr = engine._valid_len_buf
valid_len_arr[0] = bs  # all valid

hiddens_P = []
for ci in range(engine.num_chunks):
    inp = {
        "hidden_states": hidden_P.astype(np.float16),
        "position_ids": pos_ids,
        "causal_mask": mask,
        "current_pos": cur_pos,
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
        "valid_len": valid_len_arr,
    }
    out = engine.prefills[ci].predict(inp, state=engine.states[ci])
    hidden_P = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
    h = hidden_P[:, bs-1:bs, :].copy()
    norm_h = np.linalg.norm(h.astype(np.float64))
    print(f"  chunk{ci}: hidden[{bs-1}] norm={norm_h:.4f}")
    hiddens_P.append(h)

# ═══ PATH I: block1 via batch, then block2 via INFER (sequential) ═══
print(f"\n{'='*60}")
print(f"  PATH I: block1 + block2 via INFER sequential (non-zero initial state)")
print(f"{'='*60}")
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(block1, 0)
print(f"  Block1 done. pos={engine.pos}")

# Process block2 tokens sequentially
for ti, tok_id in enumerate(block2[:-1]):
    engine._step_kv_only(tok_id, engine.pos)
    engine.pos += 1

# Last token through individual chunks
last_tok = block2[-1]
embed_out = list(engine.embed.predict({"input_ids": np.array([[last_tok]], dtype=np.int32)}).values())[0]
hidden_I = embed_out
last_pos = engine.pos

hiddens_I = []
for ci in range(engine.num_chunks):
    mask_s = engine._mask_buf
    mask_s[:, :, :, :] = -65504.0
    mask_s[0, 0, 0, :last_pos+1] = 0.0

    rope_arr = engine._rope_buf
    rope_arr[0] = last_pos + engine.rope_offset

    inp = {
        "hidden_states": hidden_I.astype(np.float16),
        "position_ids": rope_arr,
        "causal_mask": mask_s,
        "current_pos": np.array([last_pos], dtype=np.int32),
        "linear_conv_state": engine.lin_convs[ci],
        "linear_recurrent_state": engine.lin_recs[ci],
    }
    out = engine.ffns[ci].predict(inp, state=engine.states[ci])
    hidden_I = out["output_hidden_states"]
    if 'linear_conv_state_out' in out:
        engine.lin_convs[ci] = out['linear_conv_state_out']
        engine.lin_recs[ci] = out['linear_recurrent_state_out']
    h = hidden_I.copy()
    norm_h = np.linalg.norm(h.astype(np.float64))
    print(f"  chunk{ci}: hidden norm={norm_h:.4f}")
    hiddens_I.append(h)

# ═══ COMPARISON ═══
print(f"\n{'='*60}")
print(f"  COMPARISON: PREFILL vs INFER for block2 (WITH non-zero initial state)")
print(f"{'='*60}")
for i, (hp, hi) in enumerate(zip(hiddens_P, hiddens_I)):
    cos = cos_sim(hp, hi)
    mse = np.mean((hp.astype(np.float64) - hi.astype(np.float64))**2)
    mad = np.max(np.abs(hp.astype(np.float64) - hi.astype(np.float64)))
    print(f"  chunk{i}: cos={cos:.6f}  MSE={mse:.2e}  max_abs_diff={mad:.4f}")
