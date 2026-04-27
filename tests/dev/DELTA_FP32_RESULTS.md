# Delta-FP32 Experiment Results

**Date**: 2025-07-17  
**Model**: Qwen3.5-4B (32 layers, 9 chunks, BS=256, CTX=4096)  
**Prompt**: 297 tokens → block1=256 (full batch) + block2=41 (tail batch)

## Objective

Test whether assigning fp32 compute precision to the chunked delta-rule
accumulation ops (`matmul`, `exp`, `reduce_sum`, `reduce_mean`, `rsqrt`,
`log`, `cumsum`, `softmax`) during CoreML export materially reduces or
removes the tail-batch divergence observed in Qwen3.5-4B prefill.

## Experimental Setup

| Variant  | Model dir | Compute unit | Mode |
|----------|-----------|--------------|------|
| Baseline | `qwen3_5_4B_milestone_3.3` | CPU_AND_NE | combined-dedup |
| Patched  | `qwen3_5_4B_delta_fp32_test` | CPU_AND_GPU | separate |

**Test protocol** (for each variant):
- **Path A**: block1 full-batch prefill → block2 **batch** tail prefill (the "fast" path)
- **Path B**: block1 full-batch prefill → block2 **sequential** tail prefill (the "reference" path)
- Compare A vs B to measure how much batch tail diverges from sequential reference

## Key Limitation: Patched Path B Could Not Run

The patched variant was forced into **separate mode** (individual `.mlpackage` per
chunk) because the delta-fp32 models were not compiled into combined multifunction
packages. In separate mode, CoreML's `MLState` objects created by one model instance
**cannot** be shared with another model instance's `predict()` call — the call returns
empty `{}` output. This means Path B (sequential decode using prefill-created states)
could not execute. The patched column therefore shows **A vs A = 1.000000** (trivially
identical), not a real measurement.

Additionally, the delta-fp32 models are **ANE-incompatible** — loading with
`CPU_AND_NE` causes `make_state()` to fail because fp32 ops cannot compile for
the Apple Neural Engine. The workaround (`CPU_AND_GPU`) works but defeats the purpose
of ANE-accelerated inference.

## Results

### Baseline: Per-Chunk Hidden Divergence (Path A vs Path B)

| Chunk | Hidden cos | Hidden maxabs | Conv cos | Rec cos | KV cos |
|------:|-----------:|--------------:|---------:|--------:|-------:|
| 0     | 0.963      | 0.078         | 0.997    | 1.000   | 1.000  |
| 1     | 0.965      | 0.272         | 0.956    | 0.981   | 0.992  |
| 2     | 0.967      | 0.844         | 0.953    | 0.989   | 0.985  |
| 3     | 0.847      | 0.840         | 0.829    | 0.885   | 0.535  |
| 4     | 0.685      | 0.768         | 0.801    | 0.844   | 0.538  |
| 5     | 0.590      | 1.873         | 0.896    | 0.963   | 0.951  |
| 6     | 0.752      | 2.506         | 0.905    | 0.955   | 0.580  |
| 7     | 0.646      | 3.153         | 0.917    | 0.932   | 0.576  |
| 8     | 0.663      | 8.731         | 1.000    | 1.000   | 0.983  |

**Pattern**: Divergence grows across chunks. Full-attention chunks (3, 4, 6, 7)
show the most severe KV divergence (cosine 0.535–0.580). Chunk 8 (single full-attention
layer) inherits accumulated error: hidden cos=0.663, maxabs=8.73.

### Baseline: Final Metrics

| Metric | Value |
|--------|------:|
| Final hidden cosine (A vs B) | 0.6627 |
| Final hidden max abs diff | 8.7305 |
| Final logits cosine | 0.6836 |
| Token A (batch tail) | 248068 |
| Token B (sequential tail) | 248068 |
| Token match | **Yes** |

### Baseline: Injection Experiments

| Experiment | Token | Description |
|-----------|------:|-------------|
| X: good hidden → lm_head | 248068 | Path-B hidden through lm_head. Same token. |
| Y: good KV → decode | **271** | Path-A states + Path-B KV cache → decode. Different token! |
| Z: good linear → decode | **271** | Path-A states + Path-B linear states → decode. Different token! |

Experiments Y and Z reveal that even though the final token matches, the **internal
representations are severely diverged**. Injecting "correct" KV or linear states into
the diverged model state produces a completely different token (271 vs 248068), showing
the model has adapted its full computation around the drift.

### Patched: All Metrics

All patched metrics show 1.000000 — **this is an artifact of comparing Path A to itself**
(Path B was skipped). The patched models do produce token 248068, matching baseline.

## Conclusions

### 1. Baseline Divergence Is Real and Severe
Batch-tail prefill diverges substantially from sequential reference:
- Hidden cosine drops to **0.663** by the final chunk
- KV cache cosine at full-attention layers drops to **0.535**
- Max absolute difference reaches **8.73**
- Despite this, the token still matches on this prompt — but internal states are deeply diverged

### 2. Delta-FP32 Cannot Be Evaluated on ANE
The fp32 precision ops (`exp`, `matmul`, `reduce_sum`, etc.) are **not supported by
the Apple Neural Engine**. Models load but `make_state()` fails. The workaround
(CPU_AND_GPU) bypasses ANE entirely, making the comparison meaningless for
production deployment.

### 3. CoreML State Isolation Prevents Cross-Model Comparison
`MLState` objects are bound to the model instance that created them. In separate mode
(where prefill and FFN are distinct `.mlpackage` files), states cannot be transferred
between model instances. This prevents Path B from running on the patched models.

### 4. Recommendations

1. **FP32 delta rule on ANE is not viable** with current CoreML tooling. The op
   set required for delta-rule accumulation cannot be forced to fp32 without
   breaking ANE compilation.

2. **Alternative approaches to consider**:
   - Mixed-precision at the **MIL graph level** (quantize only non-delta ops)
   - Restructure the delta-rule computation to use ANE-compatible ops
   - Accept fp16 delta rule and mitigate divergence through other means
     (e.g., periodic state reset, reduced batch size, attention normalization)

3. **The baseline divergence characterization is itself valuable**: the per-chunk
   breakdown pinpoints full-attention layers (chunks 3,4,6,7) as the primary
   divergence amplifiers through their KV caches.

## Files

- Test script: `tests/dev/diag_delta_fp32_compare.py`
- Full output: `tests/dev/diag_delta_fp32_compare_output_final.txt`
- Export patch: `scripts_qwen3_5/export.py` (`--delta-fp32`, `--prefill-only` flags)
- Patched models: `qwen3_5_4B_delta_fp32_test/` (symlinks + new prefill .mlpackage)
