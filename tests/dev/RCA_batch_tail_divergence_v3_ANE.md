# RCA v3: Batch Tail Prefill Divergence — ANE Evidence

**Model**: Qwen3.5-4B hybrid, 32 layers, 9 chunks  
**Hardware**: CPU_AND_NE (Apple Neural Engine)  
**Prompt**: "What is the capital of France?" → 291 tokens → Block1=256, Tail=35  
**Batch size**: 256  

---

## Executive Summary

**Root Cause: The prefill model graph computes fundamentally different results from the infer model graph for the same valid tokens. This is NOT caused by padding leakage.**

On ANE:
- Padding is perfectly isolated (Test 2: zero vs random = BIT-IDENTICAL for all states)
- Yet batch prefill vs sequential infer produces severe divergence (cos=0.08 at k_cache chunk3)
- The divergence originates at chunk0 conv state (cos=0.996) and amplifies through every chunk
- k_cache at full-attention layers (chunks 3,4,6,7) diverges catastrophically (cos=0.08-0.17)
- This is WORSE than CPU_ONLY, and ANE vs CPU themselves differ significantly (logits cos=0.776)

**The fix must be in the model export/graph, not in padding handling.**

---

## A. What CPU_ONLY Proved vs What It Did NOT Prove About ANE

### Proved (valid on both CPU_ONLY and ANE):
1. **Divergence exists between prefill-graph and infer-graph processing of tail tokens** — confirmed on both
2. **Block1 (256 tokens) produces identical states** — confirmed on both
3. **Divergence starts at chunk0** and grows through the pipeline — confirmed on both
4. **k_cache at full-attention chunks diverges worst** — confirmed on both (chunks 3,4,6,7)
5. **First token still matches** despite severe state divergence — confirmed on both

### Did NOT Prove (different on ANE):
1. **Padding is NOT the cause on ANE** — CPU_ONLY showed padding had some effect; ANE shows ZERO effect
2. **Divergence magnitude is WORSE on ANE** — CPU_ONLY logits cos ~0.97; ANE logits cos = 0.834
3. **ANE itself adds additional divergence vs CPU** — ANE vs CPU_ONLY batch states diverge (logits cos=0.776)
4. **Non-zero initial state makes divergence even worse on ANE** — condition 3 produces token MISMATCH

---

## B. ANE Evidence Table

### Test 0: ANE Determinism
| Metric | Value |
|--------|-------|
| Token match | ✓ (198=198) |
| State identity | cos=1.0, max_abs=0.0 everywhere |
| Verdict | **ANE is deterministic** (sub-bit differences are IEEE754 NaN handling) |

### Test 1: Sequential Tail (A) vs Batch Tail (B) — ANE
| Chunk | k_cache cos | v_cache cos | conv cos | rec cos | hidden cos |
|-------|------------|------------|----------|---------|------------|
| 0 | 1.000 | 1.000 | 0.996 | 0.9996 | 0.982 |
| 1 | 0.990 | 0.990 | 0.964 | 0.984 | 0.969 |
| 2 | 0.979 | 0.979 | 0.964 | 0.967 | 0.947 |
| 3 | **0.081** | 0.963 | 0.879 | 0.895 | 0.859 |
| 4 | **0.128** | 0.933 | 0.824 | 0.877 | 0.733 |
| 5 | 0.947 | 0.951 | 0.915 | 0.974 | 0.811 |
| 6 | **0.167** | 0.980 | 0.911 | 0.964 | 0.937 |
| 7 | **0.146** | 0.988 | 0.912 | 0.956 | 0.483 |
| 8 | 0.971 | 0.993 | 1.000 | 1.000 | 0.420 |

**Logits cosine: 0.834** | First token: A=198, B=198 (match)  
**Pattern**: k_cache diverges catastrophically at full-attention chunks (3,4,6,7 = layers 15,19,27,31)

### Test 2: Padding Leakage — ANE
| Metric | Value |
|--------|-------|
| All states zero vs random | **cos=1.0, max_abs=0.0 for ALL 36 state pairs** |
| Valid hidden states | cos=1.0 for all chunks |
| Padding hidden states | cos=1.0 for chunks 1-7 (chunk0 padding differs but doesn't leak) |
| Token match | ✓ (198=198) |
| Logits | cos=1.0, max_abs=0.0 |
| **Verdict** | **Padding is NOT the cause. valid_len masking is perfect on ANE.** |

Note: Chunk0 padding hidden states show cos=-0.028, max_abs=91.2 between zero/random — but this does NOT leak into valid tokens or states. From chunk1 onward, even padding outputs are identical.

### Test 3: 4-Condition Matrix — ANE
| Condition | Worst cos | Worst location | Token match |
|-----------|-----------|----------------|-------------|
| 1. zero_state + full_batch | 0.730 | chunk4/v_cache | ✓ (41453=41453) |
| 2. zero_state + partial_batch | 0.934 | chunk1/conv | ✓ (198=198) |
| 3. nonzero_state + full_batch | 0.565 | chunk3/k_cache | **✗ (41453≠248046)** |
| 4. nonzero_state + partial_batch | **0.081** | chunk3/k_cache | ✓ (198=198) |

**Key findings**:
- **Condition 3 (nonzero+full) produces TOKEN MISMATCH** — the only condition where first token differs
- **Condition 4 (nonzero+partial = real scenario) has worst divergence** (cos=0.081) but tokens still match
- Full-batch (256 tokens) divergence is worse than partial-batch (35 tokens) in absolute terms
- Non-zero initial states amplify divergence compared to zero states

### Test 4: Divergence Timeline — ANE
| Stage | Token A | Token B | Worst cos | Location |
|-------|---------|---------|-----------|----------|
| After tail | 198 | 198 | 0.081 | chunk3/k_cache |
| After decode1 | 248068 | 248068 | 0.088 | chunk3/k_cache |
| Logits cos tail | 0.834 | | | |
| Logits cos decode1 | 0.739 | | | |

**Divergence persists and worsens during autoregressive decoding** — logits cos degrades from 0.834 → 0.739.

### Test 5: State Change Magnitudes — ANE
| Chunk | k_cache A_change | k_cache B_change | k_cache cos |
|-------|-----------------|-----------------|-------------|
| 0 | — | — | 1.000 |
| 1 | 264.1 | 264.8 | 0.990 |
| 2 | 280.2 | 280.0 | 0.979 |
| 3 | **291.0** | **808.0** | **0.081** |
| 4 | **292.9** | **794.2** | **0.128** |
| 5 | 308.5 | 306.5 | 0.947 |
| 6 | **282.6** | **773.2** | **0.167** |
| 7 | **274.5** | **752.6** | **0.146** |
| 8 | 270.4 | 271.9 | 0.971 |

**CRITICAL FINDING**: At full-attention chunks (3,4,6,7), the batch prefill k_cache change is **2.5-3x larger** than sequential (e.g., 808 vs 291). The prefill model is writing dramatically different values into the KV cache at full-attention layers.

### Test 6: Padding Effect on Conv/Rec — ANE
| All chunks | Conv cos | Rec cos |
|------------|----------|---------|
| 0-8 | 1.000000 | 1.000000 |

**Zero padding vs random padding: ALL conv and rec states BIT-IDENTICAL.** Padding has no effect whatsoever on ANE.

### BONUS: ANE vs CPU_ONLY Cross-Comparison
| Metric | ANE batch | CPU batch | ANE vs CPU cos |
|--------|-----------|-----------|----------------|
| Token | 198 | 198 | match |
| Logits | — | — | **0.776** |

ANE and CPU_ONLY produce substantially different results even for the SAME batch tail operation:

| Chunk | k_cache ANEvsCPU | v_cache ANEvsCPU |
|-------|-----------------|-----------------|
| 0 | 1.000 | 1.000 |
| 1 | 0.908 | 0.901 |
| 2 | 0.852 | **0.091** |
| 3 | 0.716 | 0.759 |
| 4 | 0.734 | **0.050** |
| 5 | 0.796 | **0.046** |
| 6 | 0.755 | 0.872 |
| 7 | 0.801 | **0.157** |
| 8 | 0.889 | 0.977 |

**ANE adds its own layer of numerical divergence** — especially in v_cache at chunks 2,4,5,7 (cos < 0.16).

---

## C. Which Conclusions Remain Valid on ANE vs Changed

### VALID on both CPU_ONLY and ANE:
1. ✅ Divergence is in the prefill→infer graph mismatch, not padding
2. ✅ k_cache at full-attention layers is the worst divergence point
3. ✅ Divergence starts at chunk0 and cascades
4. ✅ First token usually still matches despite massive state divergence
5. ✅ Divergence persists/worsens during autoregressive decoding

### CHANGED on ANE:
1. ❌ **Padding is NOT a contributing factor on ANE** (was marginal on CPU_ONLY, is ZERO on ANE)
2. ❌ **Divergence is MUCH WORSE on ANE** — logits cos 0.834 vs CPU_ONLY ~0.97
3. ❌ **ANE adds its own numerical divergence layer** — ANE vs CPU differ (logits cos 0.776)
4. ❌ **Non-zero state + full batch can cause TOKEN MISMATCH on ANE** (condition 3)
5. ❌ **k_cache magnitudes diverge 2.5-3x at full-attention chunks on ANE** (808 vs 291)

### NEW on ANE:
1. 🆕 Full-attention layer k_cache in prefill model receives **2.5-3x larger updates** than infer model
2. 🆕 ANE fp16 quantization may amplify the prefill/infer graph difference
3. 🆕 The problem is architectural: the prefill graph processes 256 tokens simultaneously with attention, while the infer graph processes 1 token — the attention computation itself differs

---

## D. Root Cause Analysis and Recommendation

### Root Cause

The batch prefill model and the sequential infer model implement **numerically different computations** for the same tokens. This is NOT a padding issue — it's a fundamental graph mismatch:

1. **The prefill model** processes 256 tokens simultaneously. At full-attention layers (3,7,11,15,19,23,27,31), it computes self-attention across the full batch — each token can attend to all 256 tokens.

2. **The infer model** processes 1 token at a time. At full-attention layers, it computes attention against the KV cache built up token-by-token.

3. **The difference**: When processing the tail (35 tokens) after a block of 256:
   - **Sequential (A)**: Each token sees all previous tokens in the KV cache
   - **Batch prefill (B)**: All 35 tokens see each other simultaneously, BUT the KV cache state from the block1 is used as initial state

4. **The amplification**: The batch prefill graph writes k_cache values that are **2.5-3x larger in magnitude** at full-attention chunks. This suggests the attention computation in the prefill graph accumulates differently (possibly due to the larger attention window or different numerical path through the batch dimension).

5. **ANE worsens it**: The Neural Engine's fp16 computation adds another layer of numerical divergence compared to CPU fp32, making the graph mismatch even more severe.

### Why Sequential Fallback Works But Is Not The Solution

Sequential fallback bypasses the bug by only using the infer graph for tail tokens. But:
- It's 35x slower for the tail (1 token at a time vs 35 in batch)
- It doesn't fix the underlying graph mismatch
- For prompts that are exact multiples of batch_size, there IS no tail, so the bug doesn't manifest

### Recommendations to Fix Batch Prefill

1. **IMMEDIATE: Investigate the prefill model's full-attention layer implementation**
   - Compare the attention computation graph between `prefill_LUT4_chunk{3,4,6,7}` and `ffn_LUT4_chunk{3,4,6,7}`
   - The k_cache writes are 2.5-3x larger in prefill — this is the smoking gun
   - Focus on how `valid_len` affects the attention mask in prefill vs how position affects it in infer

2. **INVESTIGATE: fp32 re-export**
   - Re-export with `--fp32-compute --no-v4` to see if fp32 precision reduces the divergence
   - If fp32 divergence is much smaller, the issue is precision-related amplification
   - If fp32 divergence is similar, the issue is in the graph structure itself

3. **INVESTIGATE: Single-chunk isolation**
   - Run prefill and infer on just chunk3 (first catastrophic chunk) with identical inputs
   - Compare the intermediate attention weights, Q, K, V projections
   - This will pinpoint whether the divergence is in the attention mask, positional encoding, or weight application

4. **LONG-TERM: Ensure prefill and infer graphs compute equivalent results**
   - The prefill graph may need to use causal masking that exactly matches the sequential behavior
   - Or the KV cache update logic in the prefill graph may need to match the sequential update pattern
   - This is an export/conversion issue, not a runtime issue

---

## Appendix: Raw Test Output

Full output saved to: `tests/dev/diag_ane_rca_output.txt`  
Diagnostic script: `tests/dev/diag_ane_rca.py`  
Compute unit: `ComputeUnit.CPU_AND_NE`  
