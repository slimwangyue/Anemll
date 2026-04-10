# Chunk2 Per-Layer Precision Knockout — Analysis

## Experiment Setup

**Model**: Qwen3.5-4B on ANE (Apple Neural Engine)  
**Chunk2**: Layers 7-10 = [FLLL] (F=full-attention, L=linear-attention/recurrence)  
**Method**: Export single chunk2 with `FP16ComputePrecision(op_selector=...)` that keeps specified layers FP32 while casting others to FP16. The custom chunk is hot-swapped into the FP16 pipeline for testing.  
**3 test prompts**: CS question (EN), cooking (ZH), math riddle (EN)

## Results Table

| Variant | P0 vs FP32-c2 | P1 vs FP32-c2 | P2 vs FP32-c2 | Avg h_cos_pf | Avg h_cos_dec | Avg KL_dec |
|---------|---------------|---------------|---------------|-------------|--------------|-----------|
| **FP32=[7]** (F only) | 0/40 0% | 20/40 50% | 18/40 45% | 0.839 | 0.423 | 9.94 |
| **FP32=[8]** (L only) | 0/40 0% | 0/40 0% | 0/40 0% | 0.005 | 0.003 | 14.20 |
| **FP32=[9]** (L only) | 0/31 0% | 0/40 0% | 0/40 2% | 0.008 | 0.066 | 11.35 |
| **FP32=[10]** (L only) | 0/40 0% | 0/40 0% | 0/40 2% | 0.014 | 0.036 | 11.45 |
| **Reference: all-FP32** | 27/40 72% | 20/40 50% | 18/40 45% | — | — | — |

### vs FP16 Baseline

| Variant | P0 match | P1 match | P2 match | Output Quality |
|---------|---------|---------|---------|---------------|
| FP32=[7] | 2% | **100%** | **100%** | Coherent, mostly = FP16 |
| FP32=[8] | 0% | 0% | 0% | **Gibberish** (repetitive: "halungeappen阿姆斯") |
| FP32=[9] | 0% | 0% | 0% | **Gibberish** (random: "bane. sniff{0" . %") |
| FP32=[10] | 0% | 0% | 0% | **Gibberish** (random: "niacyss -4 (,3-Con") |

## Key Findings

### 1. Single-layer FP32 for linear-attention layers (8-10) causes CATASTROPHIC failure

When any single L-layer is kept FP32 while all others are FP16:
- **Hidden cosine similarity drops to ~0** (0.003–0.066), indicating random/uncorrelated activations
- **KL divergence >10** (vs ~6 for whole-chunk FP32), indicating complete distribution mismatch
- **Output is unintelligible gibberish** — the model produces random token sequences

### 2. Layer 7 (full-attention) FP32 has minimal effect

FP32=[7] produces output that mostly matches the all-FP16 baseline:
- P1 and P2 are 100% token-match with FP16
- P0 diverges but produces coherent text
- Hidden similarity is moderate (h_cos_pf ≈ 0.84), much better than L-layers

### 3. The op_selector has ~350 unresolved ops (~20% of graph)

Export statistics show:
- Layer 7 FP32: FP16:1407, **FP32:35**, unresolved→FP16:353
- Layers 8-10 FP32: FP16:1394, **FP32:51**, unresolved→FP16:350

The 350+ unresolved ops cannot be traced to any specific layer via BFS. These ops are cast to FP16 by default, creating **precision mismatches** at layer boundaries.

## Root Cause

The `FP16ComputePrecision(op_selector)` approach creates **intra-graph precision boundaries** that corrupt computation:

1. **Layer boundary ops** (residual connections, layer norms, KV-cache ops) span multiple layers in the MIL graph
2. The BFS-based `_find_layer_from_inputs` can only trace ~80% of ops to their source layer
3. ~20% of ops default to FP16, creating FP32→FP16 cast points at semantically wrong locations
4. For linear-attention layers (L), the recurrence state involves cross-layer ops that are particularly sensitive to precision mismatches
5. Full-attention layer (F) is more tolerant because its computation is more localized (Q·K^T attention is within-layer)

## Conclusions

1. **Per-layer mixed precision within a single MIL chunk is NOT viable** for Qwen3.5-4B's linear attention layers
2. **Chunk2 must be entirely FP32 or entirely FP16** — granular control requires separate model exports per layer
3. The previous chunk-level knockout correctly identified chunk2 as the most sensitive (32.3% avg match)
4. **Recommendation**: Keep chunk2 entirely in FP32 compute precision for production quality

## Implications for FP32 Optimization

Since per-layer knockout within a chunk doesn't work due to MIL graph precision boundary limitations:

- The minimum granularity for precision control is the **chunk level** (4 layers per chunk)
- To reduce FP32 model size, investigate whether the FP32 requirement is specific to:
  - The **linear attention** mechanism (layers 8-10, L-type)
  - The **full attention** mechanism (layer 7, F-type)
  - Or the **chunk boundary** behavior (inter-chunk vs intra-chunk)
- Alternative approaches:
  1. **Two-chunk split**: Split chunk2 into chunk2a (layer 7, F) and chunk2b (layers 8-10, L), export each with independent precision
  2. **Higher-precision quantization**: Use LUT8 instead of LUT4 for chunk2's weights while keeping FP16 compute
  3. **Selective FP32 at the coremltools compile level** instead of the MIL graph level
