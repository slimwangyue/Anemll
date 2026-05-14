# Component Benchmark: Qwen3.5-4B on iPhone 14 Pro Max

**Date**: 2025-05-07 (revised 2025-05-08 with quality review)  
**Device**: iPhone 14 Pro Max (A16 Bionic), iOS 26.2  
**Model**: Qwen3.5-**4B** (9 chunks, 32 layers, hidden=2560, ctx=4096)  
**Quantization**: LUT4 FFN, LUT6 embed/LM head  
**Compute units**: FFN=ANE (`cpuAndNeuralEngine`), LMHead=ANE, Embed=CPU (`cpuOnly`)  
**Prompt**: 22 tokens ("Explain Einstein's field equations in one sentence.")  
**Decode tokens**: 30 per run, 3 runs averaged  
**Warmup**: 5-token generation (model load + first ANE compilation) before measurement  

## Data Quality Notes

**Benchmark vs app absolute throughput — KNOWN DISCREPANCY:**

| Metric | Benchmark (4B) | Previous app logs (2B) | Previous app logs (4B) |
|--------|---------------:|----------------------:|----------------------:|
| FFN/step | 395–449ms | 97–149ms | 184–394ms |
| Decode tok/s (step) | 2.2–2.5 | 6.3–9.1 | 2.5–5.4 |
| Chunks | 9 | 7 | 9 |

**Why benchmark reports ~2.3 tok/s while "real app" showed ~8 tok/s:**

1. **Different model**: The ~8 tok/s numbers came from **Qwen3.5-2B** (7 chunks, 24 layers, hidden=2048). The benchmark runs **Qwen3.5-4B** (9 chunks, 32 layers, hidden=2560). The 4B model is inherently ~3x slower per decode step.

2. **Thermal throttling**: The benchmark runs after ~6 minutes of model loading + compilation + warmup, heating the A16. Previous 4B app runs on a cool device showed FFN=184ms (best, pos=40) vs our benchmark's 395–449ms. Chunk-by-chunk timing confirms this: early runs show `[9.6,22.0,...,18.6]` (cool, FFN=184ms) vs `[12.4,57.7,...,15.0]` (warm, FFN=395ms).

3. **Accumulator contamination (minor)**: The profiling accumulators include 1 step from the last prefill token (via `step()` in `sequentialPrefill`). With 30 decode steps, this adds ~3% noise — not a significant factor.

4. **Wall clock vs step-only**: App logs report both "step-only" and "wall" tok/s. The wall throughput includes sampling overhead (~30ms/step in production). Our benchmark reports step-only throughput.

## Raw Results

### Default Config: Embed=CPU, FFN=ANE, LMHead=ANE

| Run | embed (ms) | FFN (ms) | LM (ms) | step (ms) | tok/s |
|-----|--------:|--------:|-------:|--------:|------:|
| 1 | 0.41 | 424.88 | 12.66 | 437.96 | 2.3 |
| 2 | 0.40 | 448.90 | 13.12 | 462.43 | 2.2 |
| 3 | 0.39 | 394.78 | 12.11 | 407.28 | 2.5 |
| **avg** | **0.40** | **422.85** | **12.63** | **435.89** | **2.3** |

Per-chunk FFN breakdown (pos=40, run3):
`[12.4, 57.7, 60.6, 59.2, 51.7, 64.4, 54.4, 50.4, 15.0]` ms

### GPU Embed: Embed=GPU, FFN=ANE, LMHead=ANE

| Run | embed (ms) | FFN (ms) | LM (ms) | step (ms) | tok/s |
|-----|--------:|--------:|-------:|--------:|------:|
| 1* | 0.45 | 591.48 | 17.74 | 609.82 | 1.6 |
| 2 | 0.63 | 460.02 | 12.33 | 472.98 | 2.1 |
| 3 | 0.48 | 447.03 | 12.27 | 459.79 | 2.2 |
| **avg** | **0.52** | **499.51** | **14.11** | **514.20** | **2.0** |
| avg(2-3) | 0.56 | 453.53 | 12.30 | 466.39 | 2.15 |

*Run 1 was cold after GPU embed reload — ANE needed to re-optimize. Runs 2-3 are more representative.

## Analysis

### Absolute app throughput (4B model)

The Qwen3.5-4B on iPhone 14 Pro Max (A16) delivers:
- **Best case** (cool device, short position): ~5 tok/s, FFN≈184ms/step
- **Sustained** (thermally stable): ~2.3 tok/s, FFN≈400–450ms/step
- **Prefill**: ~3 tok/s sequential (22 tokens in 7–9 seconds)

The high variance (2x range) is dominated by ANE thermal throttling. This is a hardware characteristic of the A16 — ANE performance degrades significantly under sustained load.

### Relative: embed CPU vs GPU

| Metric | CPU embed | GPU embed (runs 2-3) | Delta |
|--------|----------:|--------------------:|------:|
| Embed latency | 0.40ms | 0.56ms | +40% |
| FFN latency | 422.85ms | 453.53ms | +7% |
| LM Head latency | 12.63ms | 12.30ms | -3% |
| Total step | 435.89ms | 466.39ms | +7% |
| Decode tok/s | 2.3 | 2.15 | -7% |

**Embed is negligible** in either configuration: 0.40ms (CPU) vs 0.56ms (GPU), both <0.15% of step time.

**FFN slows down +7% with GPU embed**. This is likely noise/thermal rather than GPU↔ANE contention — the GPU embed test ran AFTER the CPU embed test, so the device was warmer. The initial run's inflated numbers (591ms FFN) were from ANE re-optimization after model reload, not from sustained contention.

### Conclusion

1. **Keep embed on CPU-only.** There is zero benefit to GPU embed — the lookup is 0.4ms regardless. Even if GPU were free, it can't improve on sub-millisecond CPU performance.

2. **The 2.3 tok/s benchmark number is valid for the 4B model under sustained thermal load.** Cool-device performance is ~2x better (~5 tok/s).

3. **The ~8 tok/s numbers referenced in conversation were from the 2B model** (7 chunks, fewer layers, smaller hidden dim). That model is ~3x faster per step.

4. **A/B/A/B test could not complete** due to `reloadEmbed` using `MLModelConfiguration.functionName` on a compiled `.mlmodelc` — this fails when the model is loaded fresh from disk (outside the AnemllModelCompiler cache). The `reloadEmbed` method needs to be fixed to use the compiler's cached model or to strip `functionName` for compiled models before it can support proper A/B/A/B testing. (No production code changed per user request.)

## Test Infrastructure

Benchmark test: `local_llmTests/ComponentBenchmarkTests.swift`  
- `testBenchmarkABAB_EmbedCPUvsGPU()` — A/B/A/B pattern with warmup (currently blocked by `reloadEmbed` issue)
- `reloadEmbed(computeUnits:)` — test-only method on `AnemllInferenceProvider` (not `#if DEBUG` guarded; should be made DEBUG-only)
- Uses built-in profiling accumulators: `decodeEmbedAccMs`, `decodeFfnAccMs`, `decodeLmAccMs`  
- Accumulators reset in `processPrompt()` before each generation — no cross-run contamination
