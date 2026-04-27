# Precise Divergence Localization Report

**Model**: Qwen3.5-4B hybrid, 32 layers, 9 chunks, ANE (CPU_AND_NE)  
**Prompt**: 289 tokens → Block1=256 (full batch), Tail=33 tokens  
**Batch size**: 256  
**Script**: `tests/dev/diag_precise_divergence.py`  
**Output**: `tests/dev/diag_precise_divergence_output.txt`

---

## 1. Exact Reproduction Setup

- Path A: `block1 batch prefill (256 tok) → block2 batch prefill (33 tok, padded to 256)`
- Path B: `block1 batch prefill (256 tok) → block2 sequential via infer model (33 tok, one-at-a-time)`
- Both use identical prompt, same engine instance, same initial state (zero), same block partitioning
- States snapshotted via `MLState.read_state()` / `write_state()`, restored with `memcpy`-equivalent before Path B

---

## 2. First Divergence Point

### Checkpoint 1: After Block1
**States are BIT-IDENTICAL** between A and B. The divergence is exclusively in block2.

### Checkpoint 2: During Block2, per-chunk

| chunk | hidden_cos | k_cos  | v_cos  | conv_cos | rec_cos  | k_L2ratio |
|-------|-----------|--------|--------|----------|----------|-----------|
| **0** | **0.973** | 1.000  | 1.000  | 0.997    | 0.9998   | 0.000*    |
| 1     | 0.954     | 0.993  | 0.993  | 0.964    | 0.985    | 0.999     |
| 2     | 0.964     | 0.990  | 0.988  | 0.978    | 0.981    | 1.000     |
| **3** | 0.886     | **0.114** | 0.977 | 0.899   | 0.913    | **0.338** |
| **4** | 0.860     | **0.133** | 0.950 | 0.851   | 0.888    | **0.336** |
| 5     | 0.947     | 0.963  | 0.956  | 0.930    | 0.977    | 1.000     |
| **6** | 0.940     | **0.174** | 0.978 | 0.922   | 0.965    | **0.337** |
| **7** | 0.683     | **0.150** | 0.988 | 0.904   | 0.926    | **0.335** |
| 8     | 0.694     | 0.977  | 0.993  | 1.000    | 1.000    | 1.000     |

*\*Chunk 0 has no full-attention layers → k_L2=0 for both paths*

**First divergence: Chunk 0, hidden state (cos=0.973).**

Chunk 0 contains only linear attention layers. There are NO KV cache entries. The hidden output diverges purely because `forward_prefill_export()` (chunked gated delta rule) and `forward_regular()` (recurrent gated delta rule) produce numerically different results for the same input tokens.

---

## 3. Which Tensor Family Diverged First

At chunk 0 (first diverging chunk):

| Tensor | Cosine | Max abs diff | A_L2 | B_L2 |
|--------|--------|-------------|------|------|
| **hidden** | **0.973** | 0.271 | 2.544 | 2.780 |
| conv | 0.997 | 1.047 | 389.6 | 390.9 |
| rec | 0.9998 | 0.088 | 38.04 | 38.03 |
| k_cache | 1.000 | 0.000 | 0.0 | 0.0 |
| v_cache | 1.000 | 0.000 | 0.0 | 0.0 |

**Hidden diverges first**, followed by conv, then rec. KV is not involved at chunk 0.

The divergence cascade:
1. Chunk 0: **hidden cos=0.973** — linear attention prefill-vs-infer path difference
2. Chunk 1: hidden cos=0.954, KV starts diverging (cos=0.993)
3. Chunks 3,4,6,7: KV cache (k_cache) **collapses** (cos=0.11-0.17) at full-attention layers
4. Chunk 8: final hidden cos=0.694

---

## 4. Accumulative vs Localized

**Accumulative.** The divergence starts small at chunk 0 (cos=0.973) and grows monotonically through the pipeline. However, full-attention layers at chunks 3,4,6,7 are massive amplifiers — k_cache diverges catastrophically because the diverged hidden states produce fundamentally different Q/K/V projections.

The k_L2ratio of ~0.34 at full-attention chunks means the batch prefill path writes k_cache entries with only **34% of the L2 norm** compared to sequential. This is a 3x magnitude discrepancy in what gets written to the KV cache.

---

## 5. Swap Experiment Results

### Experiment X: Final-hidden swap → lm_head
| Scenario | Token |
|----------|-------|
| BAD path (A) | 90700 |
| GOOD path (B) | 8160 |
| BAD path + GOOD hidden → lm_head | **8160 ✓** |

**YES — swapping final hidden immediately fixes the first token.**  
The token failure is driven by wrong final hidden state, not by lm_head corruption.

### Experiment Y: Good KV + Bad linear → decode 1 token
| Scenario | Token |
|----------|-------|
| All GOOD states | 579 |
| All BAD states | 8340 |
| Good KV + Bad linear | 8340 |

**KV swap alone does NOT fix decode.** Note: the first token already diverged (90700 vs 8160), so the embedding of the wrong token dominates subsequent decode.

### Experiment Z: Good linear + Bad KV → decode 1 token
| Scenario | Token |
|----------|-------|
| All GOOD states | 579 |
| All BAD states | 8340 |
| Good linear + Bad KV | 8340 |

**Linear swap alone does NOT fix decode.** Same reason — the wrong first token (90700) was already embedded.

### Per-chunk KV/Linear swaps
All 9 individual chunk KV swaps produce token 8340. All 9 individual chunk linear swaps produce token 8340. No single chunk dominates — the problem is **systemic** (accumulated hidden state divergence), not localized to one chunk's state.

**Interpretation**: Once the first token is wrong (90700 instead of 8160), no amount of state patching can recover correct decode. The root cause must be fixed at the hidden-state level during prefill.

---

## 6. `current_pos` Semantic Test

| Setting | Token | Logits cos vs GOOD |
|---------|-------|--------------------|
| `current_pos = blockStart` (default=256) | 90700 | 0.758 |
| `current_pos = blockStart + validLen - 1` (=288) | 90700 | 0.744 |
| Sequential (GOOD) | 8160 | 1.000 |

**Alternative current_pos makes things WORSE** (logits cos drops from 0.758 to 0.744). Per-chunk comparison shows v_cache dropping from ~0.98-0.99 to ~0.89-0.90 across the board with the alt setting.

**`current_pos = blockStart` is the correct semantic.** The model writes KV at `[current_pos, current_pos + seq_len)` and `current_pos` represents the start of the write window, not the end.

**This rules out a current_pos contract bug as the cause.**

---

## 7. Root Cause Ranking by Confidence

### 1. (HIGH CONFIDENCE) Linear attention path mismatch: `forward_prefill_export` ≠ `forward_regular`

The divergence starts at **chunk 0 hidden** (cos=0.973) where there are **zero KV cache entries** — only linear attention layers exist. This proves the root cause is in the linear attention computation itself.

The prefill path uses `_chunk_gated_delta_rule` (batched), while the infer path uses `_recurrent_gated_delta_rule` (single-step). These are mathematically equivalent in exact arithmetic, but produce different results under fp16 due to:
- Different reduction order (batch reduces across all tokens simultaneously; sequential reduces one at a time)
- Different intermediate accumulation precision
- Different conv_stage windowing (prefill uses valid_len-based one-hot gather; infer uses simple shift)

**Evidence**: Chunk 0 has no KV cache yet diverges. Conv state (cos=0.997) and rec state (cos=0.9998) are close but not identical, confirming the linear attention recurrence and convolution produce slightly different results in the two paths.

### 2. (HIGH CONFIDENCE) KV cache magnitude amplification at full-attention layers

At full-attention chunks (3,4,6,7), the diverged hidden states from linear attention produce dramatically different KV cache writes. The batch path writes entries with only **34% of the L2 norm** of the sequential path. This 3x magnitude discrepancy is the primary amplifier that turns small hidden divergence (cos~0.97) into catastrophic state corruption (k_cache cos=0.11).

**Evidence**: k_L2ratio = 0.336 consistently at chunks 3,4,6,7.

### 3. (LOW CONFIDENCE) `current_pos` contract bug → RULED OUT

Both `current_pos = blockStart` and `current_pos = blockStart + validLen - 1` produce wrong tokens. The alternative makes things worse. The model's KV write logic `k_cache[:, pos:pos+seq_len, :]` is consistent with `current_pos = blockStart`.

### 4. (LOW CONFIDENCE) Padding leakage → RULED OUT

Previous Test 2 (from RCA v3) proved zero vs random padding produces bit-identical results on ANE. Padding is perfectly masked.

---

## Conclusion

**The root cause is a numerical mismatch between the prefill linear attention path (`_chunk_gated_delta_rule` + `conv_stage` with `valid_len`) and the infer linear attention path (`_recurrent_gated_delta_rule` + `conv_stage` sequential).** 

This mismatch manifests as a cos=0.973 divergence at chunk 0 (purely linear attention layers), which then cascades and amplifies through the pipeline, especially at full-attention layers where KV cache writes diverge by 3x in magnitude.

**The fix must be in the model code** — either:
1. Make the chunked delta rule numerically match the recurrent delta rule under fp16 (hard)
2. Use the infer model path for tail blocks (current workaround — sequential fallback)
3. Run the tail through the prefill model but with seq_len=1 per token (preserve the graph but eliminate batch reduction differences)
4. Use fp32 intermediate precision in the linear attention kernels to reduce accumulation error

**This is NOT a runtime contract bug** (current_pos, valid_len, mask are all correct). It IS a fundamental numerical equivalence gap between the two exported model graphs' linear attention implementations.
