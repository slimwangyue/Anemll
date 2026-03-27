# Qwen3.5-4B Milestone 2.1 — LUT6 FFN Quantization Upgrade

**Date**: 2026-03-27  
**Model**: Qwen3.5-4B (28 layers, 4 FFN chunks × 7 layers)  
**Previous**: Milestone 2.0 (Batch Prefill, 2026-03-22)

## Summary

Upgraded FFN chunk quantization from **LUT4 gs=8** to **LUT6 gs=4** (6-bit palette, per-channel group size 4). This single change raises first-token match rate from 50% to 88% against HuggingFace reference, with only 6% latency overhead — all on Apple Neural Engine.

## Problem Statement

Milestone 2.0 models, while fast, exhibited significant quality drift on ANE compared to the HuggingFace (HF) PyTorch reference. On a diverse set of 8 prompts (Chinese + English), **only 50% of first tokens matched HF**. In the worst cases, the model refused to answer or asked clarifying questions instead of responding (e.g., `zh_recipe`: "作为AI助手，我不能教你做红烧肉" instead of answering).

Root cause analysis (see Mitigation Report below) showed:
1. ANE's FP16-only ALUs amplify recurrence error ~45× per layer through the RMSNormGated + out_proj chain
2. LUT4 weight reconstruction error compounds this further across 28 layers
3. FP32 compute precision fully recovers quality but runs as CPU/ANE hybrid at 2× latency — not viable for production

## Investigation & Selection

Five quantization configurations were systematically tested across 8 diverse prompts:

| Config | 1st Match | Avg Quality | Chunk Latency | Decode Time | FFN Size (×4) | Overhead |
|--------|-----------|-------------|---------------|-------------|---------------|----------|
| LUT4 gs=8 (baseline) | 50% (4/8) | 1.88/3 | 21.4 ms | 7,005 ms | 428 MB | 1.00× |
| LUT6 gs=16 | 88% (7/8) | 2.25/3 | 23.7 ms | 7,817 ms | 642 MB | 1.11× |
| LUT6 gs=8 | 75% (6/8) | 2.12/3 | 24.8 ms | 7,610 ms | 644 MB | 1.16× |
| **LUT6 gs=4** ★ | **88% (7/8)** | **2.25/3** | **22.7 ms** | **7,215 ms** | **649 MB** | **1.06×** |
| LUT6 gs=1 | 75% (6/8) | 2.25/3 | 46.0 ms | 13,520 ms | 680 MB | 2.14× |

**LUT6 gs=4** was selected as the winner:
- **Best quality**: 88% first-token match, tied with gs=16
- **Lowest latency** among LUT6 variants: only 1.06× vs LUT4 baseline
- **Stays on ANE**: Unlike gs=1 which spills to CPU (2.14× slower) because per-tensor LUT tables are too large for ANE's local memory
- **Modest size increase**: +221 MB per chunk (649 vs 428 MB), +884 MB total for 4 chunks

### Why Not gs=16 or gs=8?

- **gs=16** achieves the same 88% match rate but is 1.11× latency (vs 1.06× for gs=4)
- **gs=8** is surprisingly worse at 75% — the group size sweet spot for this model is gs=4
- **gs=1** (per-tensor) causes ANE to spill large LUT ops to CPU, doubling latency with no quality gain

### Why Not FP32?

Full FP32 compute precision achieves 100% first-token match but:
- 2.0× latency (40 ms/chunk vs 20 ms) — ANE has FP16-only ALUs, FP32 ops spill to CPU
- 132 extra `cast` ops inserted at fp16↔fp32 boundaries
- Selective FP32 (norm ops only) had zero quality improvement at 1.15× cost

## Per-Prompt Breakdown

| Prompt | LUT4 gs=8 | LUT6 gs=4 | HF Ref |
|--------|-----------|-----------|--------|
| 教我做红烧肉 (zh_recipe) | 109266 ✗ | 126114 ✓ | 126114 |
| 解释量子纠缠 (zh_physics) | 95772 ✗ | 332 ✓ | 332 |
| 写一首关于春天的诗 (zh_poem) | 332 ✓ | 332 ✓ | 332 |
| Explain neural network (en_nn) | 1597 ✓ | 1597 ✓ | 1597 |
| Python sort function (en_code) | 38493 ✗ | 8160 ✓ | 8160 |
| Capital of France (en_factoid) | 760 ✓ | 760 ✓ | 760 |
| Train distance math (en_math) | 1206 ✓ | 1206 ✓ | 1206 |
| Robot cook story (en_story) | 332 ✗ | 760 ✗ | 4413 |

Notable improvements:
- **zh_recipe**: LUT4 refused to answer ("作为AI助手，我不能…"); LUT6 gs=4 correctly teaches the recipe
- **en_code**: LUT4 asked clarifying questions; LUT6 gs=4 directly writes sort functions
- **zh_physics**: LUT4 produced wrong leading token; LUT6 gs=4 matches HF

The only remaining miss (`en_story`) produces a coherent, well-written story — just with different narrative framing than HF.

## Updated Model Files

`qwen3_5_stable_models/` now contains:
- `embeddings.mlpackage` — 304 MB (unchanged, LUT6 gs=8)
- `lm_head.mlpackage` — 462 MB (unchanged, LUT6 gs=8 + argmax)
- `ffn_LUT4_chunk{0..3}.mlpackage` — 4 × **649 MB** (upgraded to **LUT6 gs=4**)
- `prefill_LUT4_chunk{0..3}.mlpackage` — 4 × 437 MB (unchanged, LUT4 gs=8, to be upgraded separately)
- `combined_LUT4_dedup/` — combined infer+prefill (LUT4, to be re-combined with LUT6)

| Component | Milestone 2.0 | Milestone 2.1 |
|-----------|--------------|---------------|
| FFN Quant | LUT4 gs=8 | **LUT6 gs=4** |
| FFN Size (×4) | 1,712 MB | **2,596 MB** |
| Total Deploy | ~4.0 GB | ~4.9 GB |
| 1st Token Match | 50% | **88%** |
| Quality Score | 1.88/3 | **2.25/3** |
| Chunk Latency | 21.4 ms | **22.7 ms** |
| Decode (8-prompt avg) | 7,005 ms | **7,215 ms** |
| Latency Overhead | — | **+6%** |

## Technical Details

### What is LUT6 gs=4?

- **LUT6**: 6-bit lookup table quantization. Each weight is encoded as a 6-bit index into a palette of $2^6 = 64$ float16 centroids per output channel group.
- **gs=4**: Per-channel group size of 4. Every 4 output channels share one palette. Finer-grained than gs=8 (fewer channels share centroids → better weight fidelity), but coarser than gs=1 (per-tensor, which overflows ANE local memory).
- **ANE compatibility**: The `constexpr_lut_to_dense` ops (68 per chunk) reconstruct weights on-device. With gs=4, the palette tables fit within ANE's SRAM, keeping execution on-chip.

### Why 6-bit helps

Moving from 4-bit to 6-bit increases palette size from $2^4 = 16$ to $2^6 = 64$ centroids. This 4× increase in representable weight values dramatically reduces quantization error, especially for the large linear projections in the Mamba-2 hybrid attention layers where weight distribution has heavy tails.

### ANE execution verification

Both LUT4 and LUT6 gs=4 run entirely on ANE (no CPU spillover):
- LUT4 gs=8: 21.4 ms/chunk ← pure ANE
- LUT6 gs=4: 22.7 ms/chunk ← pure ANE (6% overhead from larger weight decompression)
- LUT6 gs=1: 46.0 ms/chunk ← **CPU spillover** (palette table too large for ANE SRAM)

## Rollback

LUT4 gs=8 originals are preserved at:
- `/Users/yw68/Anemll_remote_run/qwen35_milestone1_3/ffn_LUT4_chunk{0..3}.mlpackage`

## Related Files

- Comparison script: `tests/dev/debug_lut6_comparison.py`
- Results JSON: `tests/dev/lut6_comparison_report.json`
- FP32 investigation: `tests/dev/debug_fp32_execution_device.py`
- Selective FP32: `tests/dev/debug_selective_fp32_quality.py`
- Amplification report: `tests/dev/MITIGATION_AMPLIFICATION_REPORT.md`
- HF reference cache: `tests/dev/fp32_multiprompt_hf_cache.json`

## Next Steps

- [ ] Re-export prefill chunks with LUT6 gs=4
- [ ] Re-combine infer+prefill into combined models
- [ ] Extended repetition testing with diverse prompts
- [ ] Upload updated models to HuggingFace
