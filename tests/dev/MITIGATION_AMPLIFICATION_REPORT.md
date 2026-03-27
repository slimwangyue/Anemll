# Mitigation Report: Linear-Attention Amplification

**Date**: 2025-03-26  
**Model**: Qwen3.5-4B, layer 0, SEQ_LEN=256, CTX=1024  
**Baseline**: ANE recurrence → RMSNormGated → out_proj, cosine=0.9956 vs PyTorch fp16  

## Experiment Summary

| Strategy | cosine | max_abs | ch0_mean | Recovery | Verdict |
|---|---|---|---|---|---|
| **A) Baseline** (current code) | 0.995559 | 0.3027 | 0.1380 | — | baseline |
| **B) Pre-norm scaling S=8** | 0.999841 | 0.4531 | 0.3459 | — | ❌ breaks semantics (0.86 cos to original) |
| **C) Direct RMSNorm** (no doubled-LN trick) | 0.995559 | 0.3027 | 0.1380 | 0% | ❌ no improvement |
| **D) FP32 compute precision** | **0.999950** | **0.0186** | **0.0089** | **98.9%** | ✅ **winner** |
| **E) FP32 recurrence math** (trace-time fp32) | 0.995559 | 0.3027 | 0.1380 | 0% | ❌ ANE ignores |
| **F) Direct RMSNorm + FP32** | **0.999950** | **0.0176** | **0.0090** | **98.9%** | ✅ same as D |

## Key Findings

### 1. FP32 compute precision is the only effective mitigation (D, F)
- Cosine: 0.9956 → 0.99995 (+0.0044, closing 98.9% of the gap to 1.0)
- max_abs error: 0.303 → 0.019 (16× reduction)
- ch0 systematic bias: 0.138 → 0.009 (15× reduction)

### 2. The doubled-LayerNorm trick is NOT the amplification source (C)
Replacing `torch.cat([x, -x]) → LayerNorm → slice[:hidden_size]` with standard
`rsqrt(mean(x²) + eps) * x` gives **identical** cosine (0.995559 vs 0.995559).
The norm formulation doesn't matter — both amplify fp16 perturbation equally.

### 3. Trace-time fp32 math is ineffective (E)
Setting `math_dtype=torch.float32` in `_chunk_gated_delta_rule` at trace time
produces identical ANE output to baseline. CoreML/ANE re-casts all ops to fp16
when `compute_precision=FLOAT16`, ignoring the trace dtype.

### 4. Pre-norm scaling changes function semantics (B)
Scaling recurrence output by 1/S before RMSNormGated improves CoreML self-consistency
(cos=0.9998 against its own PyTorch), but changes the mathematical function
(only 0.86 cosine to the original PyTorch output). RMSNorm is scale-invariant
for the norm part, but the SiLU gate pathway `z` is also scaled, breaking equivalence.

## Root Cause Confirmed

The amplification chain is:
1. ANE fp16 recurrence accumulates small numerical error (cos=0.9999, max_abs≈0.003)
2. With `compute_precision=FLOAT16`, the norm + projection chain amplifies this ~45×
3. With `compute_precision=FLOAT32`, CoreML keeps intermediate computations in fp32,
   and the amplification is effectively neutralized

## Recommended Action

**Use `ct.precision.FLOAT32` for the linear-attention CoreNorm stage exports.**

This requires changing `qwen3_5_converter.py`'s `compute_precision` parameter for
the FFN/prefill chunk conversions (or at minimum, for the CoreNorm sub-graph).

### Trade-offs
- **Accuracy**: 98.9% gap recovery (cosine 0.9956 → 0.99995)
- **Model size**: ~2× larger for fp32 intermediates (mitigated by LUT quantization of weights)
- **Latency**: May be slightly slower on ANE if fp32 ops can't be fused as efficiently
- **Alternative**: Apply fp32 precision only to the linear-attention chunks, keep
  full-attention chunks at fp16 (since full-attention already has cos≈0.9999)

### Implementation Path
```python
# In qwen3_5_converter.py, for linear-attention chunks only:
mlm = ct.convert(
    traced,
    ...
    compute_precision=ct.precision.FLOAT32,  # was FLOAT16
    ...
)
```
