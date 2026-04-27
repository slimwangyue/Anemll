#!/usr/bin/env python3
"""DEFINITIVE TEST: Does valid_len masking isolate valid tokens from padding?

Both runs use valid_len=164. The ONLY difference is the padding content:
- Run Z: padding positions filled with ZEROS
- Run R: padding positions filled with RANDOM embeddings

If valid_len masking works correctly, the hidden states at valid positions
(specifically position 163) should be IDENTICAL between Run Z and Run R.

Any difference proves the masking leaks padding into valid positions."""
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
print(f"Padding positions: {valid_len}..{bs-1} (count: {bs - valid_len})")

def cos_sim(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    if a_f.shape[0] == 0 or b_f.shape[0] == 0:
        return float('nan')
    return np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-30)

# Build shared inputs
mask = engine._batch_mask_buf
pos_ids = engine._batch_pos_buf
cur_pos = engine._batch_cur_buf
valid_len_arr = engine._valid_len_buf
input_ids = engine._batch_tok_buf

# Common mask for valid_len=164
mask[:, :, :, :] = -65504.0
for i in range(valid_len):
    mask[0, 0, i, :bs + i + 1] = 0
for i in range(valid_len, bs):
    mask[0, 0, i, 0] = 0.0

# Common position IDs
pos_ids[:valid_len] = np.arange(bs + engine.rope_offset, bs + engine.rope_offset + valid_len, dtype=np.int32)
pos_ids[valid_len:] = 0

# Common current_pos and valid_len
cur_pos[0] = bs
valid_len_arr[0] = valid_len

def run_tail_prefill(padding_hidden, label=""):
    """Run tail through prefill model with specified padding hidden states.
    Returns list of hidden[163] after each chunk."""
    engine._reset_states()
    engine.pos = 0
    # Process block1
    _ = engine._batch_prefill(tokens[:bs], 0)
    
    # Embed the tail tokens
    input_ids[0, :] = 0
    input_ids[0, :valid_len] = tail_tokens
    hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]
    
    # Set padding hidden states to the specified values
    if hidden.shape[1] > valid_len:
        hidden[:, valid_len:, :] = padding_hidden[:, :hidden.shape[1]-valid_len, :]
    
    results = []
    for ci in range(engine.num_chunks):
        if label and ci == 0:
            print(f"    [DEBUG] hidden shape before chunk0: {hidden.shape}")
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
        if label and ci == 0:
            print(f"    [DEBUG] hidden shape after chunk0: {hidden.shape}")
        if 'linear_conv_state_out' in out:
            engine.lin_convs[ci] = out['linear_conv_state_out']
            engine.lin_recs[ci] = out['linear_recurrent_state_out']
        
        # Re-zero padding (only if hidden has padding positions)
        if hidden.shape[1] > valid_len:
            hidden[:, valid_len:, :] = padding_hidden[:, :hidden.shape[1]-valid_len, :]
        
        # Extract last valid token
        if hidden.shape[1] >= valid_len:
            h = hidden[:, valid_len-1:valid_len, :].copy()
        elif hidden.shape[1] == 1:
            h = hidden.copy()
        else:
            h = hidden[:, -1:, :].copy()
        results.append(h)
    return results

# ═══ RUN Z: padding = ZEROS ═══
print(f"\n{'='*60}")
print(f"  RUN Z: padding = ZEROS (standard)")
print(f"{'='*60}")
zeros = np.zeros((1, bs, 2560), dtype=np.float16)
hiddens_Z = run_tail_prefill(zeros, label="Z")
for ci, h in enumerate(hiddens_Z):
    norm = np.linalg.norm(h.astype(np.float64))
    print(f"  chunk{ci}: hidden[163] norm={norm:.4f}")

# ═══ RUN R: padding = RANDOM ═══
print(f"\n{'='*60}")
print(f"  RUN R: padding = RANDOM VALUES")
print(f"{'='*60}")
np.random.seed(42)
random_pad = np.random.randn(1, bs, 2560).astype(np.float16)
hiddens_R = run_tail_prefill(random_pad, label="R")
for ci, h in enumerate(hiddens_R):
    norm = np.linalg.norm(h.astype(np.float64))
    print(f"  chunk{ci}: hidden[163] norm={norm:.4f}")

# ═══ RUN T: padding = ONES (constant non-zero) ═══
print(f"\n{'='*60}")
print(f"  RUN T: padding = ONES (constant non-zero)")
print(f"{'='*60}")
ones = np.ones((1, bs, 2560), dtype=np.float16)
hiddens_T = run_tail_prefill(ones, label="T")
for ci, h in enumerate(hiddens_T):
    norm = np.linalg.norm(h.astype(np.float64))
    print(f"  chunk{ci}: hidden[163] norm={norm:.4f}")

# ═══ COMPARISON ═══
print(f"\n{'='*60}")
print(f"  COMPARISON at position {valid_len-1} (last valid token)")
print(f"{'='*60}")
print(f"\n  ZEROS vs RANDOM:")
for ci, (hz, hr) in enumerate(zip(hiddens_Z, hiddens_R)):
    cos = cos_sim(hz, hr)
    mad = np.max(np.abs(hz.astype(np.float64) - hr.astype(np.float64)))
    print(f"    chunk{ci}: cos={cos:.6f}  max_abs_diff={mad:.4f}")

print(f"\n  ZEROS vs ONES:")
for ci, (hz, ht) in enumerate(zip(hiddens_Z, hiddens_T)):
    cos = cos_sim(hz, ht)
    mad = np.max(np.abs(hz.astype(np.float64) - ht.astype(np.float64)))
    print(f"    chunk{ci}: cos={cos:.6f}  max_abs_diff={mad:.4f}")

print(f"\n  RANDOM vs ONES:")
for ci, (hr, ht) in enumerate(zip(hiddens_R, hiddens_T)):
    cos = cos_sim(hr, ht)
    mad = np.max(np.abs(hr.astype(np.float64) - ht.astype(np.float64)))
    print(f"    chunk{ci}: cos={cos:.6f}  max_abs_diff={mad:.4f}")
