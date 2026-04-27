#!/usr/bin/env python3
"""
Diagnostic: Compare conv_state and recurrent_state after batch-tail vs sequential-tail
to pinpoint what causes the total output break.

After block1 (batch), we compare:
  Path A: 164 tokens via sequential infer  (correct)  
  Path C: 164 tokens via batch prefill     (broken)

For each chunk, compare: hidden, conv_state, recurrent_state
"""
import sys, os, copy
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
tail = tokens[bs:]
print(f"Total tokens: {len(tokens)}, block1: {bs}, tail: {len(tail)}")

def cos_sim(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    dot = np.dot(a_f, b_f)
    na = np.linalg.norm(a_f)
    nb = np.linalg.norm(b_f)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def max_abs_diff(a, b):
    return np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))


# ═══ Step 1: Process block 1 via batch (shared starting point) ═══
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(tokens[:bs], 0)
# Save state after block1
blk1_convs = [c.copy() for c in engine.lin_convs]
blk1_recs  = [r.copy() for r in engine.lin_recs]
blk1_states = []
for s in engine.states:
    # MLState objects can't be easily deep-copied; we'll reset and re-run
    pass
blk1_pos = engine.pos
print(f"Block1 done: pos={blk1_pos}")

# ═══ Path A: tail via sequential ═══
# Reset to post-block1 state
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(tokens[:bs], 0)
print("Path A: sequential tail...")
# Run tail tokens one by one, capturing state after each chunk's last token
for ti, tok_id in enumerate(tail):
    is_last = (ti == len(tail) - 1)
    if is_last:
        tok_A, _ = engine._step(tok_id, engine.pos)
    else:
        engine._step_kv_only(tok_id, engine.pos)
    engine.pos += 1
convs_A = [c.copy() for c in engine.lin_convs]
recs_A  = [r.copy() for r in engine.lin_recs]
print(f"  tok_A = {tok_A} = {repr(engine.tokenizer.decode([tok_A]))}")

# ═══ Path C: tail via batch prefill ═══
engine._reset_states()
engine.pos = 0
_ = engine._batch_prefill(tokens[:bs], 0)
print("Path C: batch tail...")
tok_C = engine._batch_prefill(tail, bs)
convs_C = [c.copy() for c in engine.lin_convs]
recs_C  = [r.copy() for r in engine.lin_recs]
print(f"  tok_C = {tok_C} = {repr(engine.tokenizer.decode([tok_C]))}")

# ═══ Compare states ═══
print(f"\n{'='*60}")
print(f"  COMPARISON: Path A (seq) vs Path C (batch) after tail")
print(f"{'='*60}")
print(f"  Token A={tok_A} ({repr(engine.tokenizer.decode([tok_A]))})")
print(f"  Token C={tok_C} ({repr(engine.tokenizer.decode([tok_C]))})")
print(f"  Match: {tok_A == tok_C}")

for ci in range(engine.num_chunks):
    conv_cos = cos_sim(convs_A[ci], convs_C[ci])
    conv_mad = max_abs_diff(convs_A[ci], convs_C[ci])
    rec_cos = cos_sim(recs_A[ci], recs_C[ci])
    rec_mad = max_abs_diff(recs_A[ci], recs_C[ci])
    print(f"  chunk{ci}: conv cos={conv_cos:.6f} mad={conv_mad:.4f}  |  rec cos={rec_cos:.6f} mad={rec_mad:.4f}")

# Also check: what does block1 conv_state look like vs what batch-tail starts with?
print(f"\n--- Block1 vs post-tail conv states ---")
for ci in range(engine.num_chunks):
    norm_blk1 = np.linalg.norm(blk1_convs[ci].astype(np.float64))
    norm_A = np.linalg.norm(convs_A[ci].astype(np.float64))
    norm_C = np.linalg.norm(convs_C[ci].astype(np.float64))
    print(f"  chunk{ci}: blk1 conv norm={norm_blk1:.2f}, A conv norm={norm_A:.2f}, C conv norm={norm_C:.2f}")
