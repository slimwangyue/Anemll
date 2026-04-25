# Qwen3.5-4B Quality Improvement Plan

## Current Model Summary

| Parameter | Value |
|-----------|-------|
| Architecture | Qwen3.5-4B, 32 layers, `[L L L F] × 8` |
| Hidden dim | 2560, FFN intermediate 9216 |
| L-layers (Gated DeltaNet) | 24 layers, 32 V-heads / 16 QK-heads, head_dim=128 |
| F-layers (Full Attention) | 8 layers at {3,7,11,15,19,23,27,31}, 16 Q-heads / 4 KV-heads (GQA 4:1), head_dim=256 |
| Vocabulary | 248,320 (tied embed/lm_head) |
| Chunk partition | 9 chunks, FLLL pattern |
| Context / Batch | 4096 / 256 |
| FFN quantization | **LUT4** (K-Means), group_size=4 |
| Embed/LM_head | **LUT6** (K-Means), group_size=8 |
| V4 precision | FP32 for KV-cache state ops in F-layers |
| D2 precision | F-layer attn Q/K/V/O weights kept FP16 (skip LUT4) |
| Selective FP32 ops | softmax, reduce_sum, reduce_mean, rsqrt, exp, log, cumsum |
| Model size (combined dedup) | ~4 GB |
| coremltools version | **9.0** |
| Decode speed | 6–7 tok/s steady state (iPhone A18 Pro) |
| Quality (single-turn) | 19/19 pass, cos_sim 0.91–0.97 per chunk |
| Quality (multi-turn) | Repetition after ~1000 tokens (sampling issue) |

---

## Research Findings

### 1. Unsloth Qwen3.5 GGUF Benchmarks (Key Takeaways)

- **Hybrid architectures have unique sensitivity patterns**: `ssm_out` dramatically increases KLD when quantized. `attn_*` tensors are especially sensitive. Leaving attention weights at higher precision works well.
- **Imatrix (importance matrix) dramatically improves quantization quality** — especially at lower bit widths. Reduces 99.9% KLD significantly for sensitive tensors like `ssm_out`.
- **ffn_down is more sensitive than ffn_up/ffn_gate** — quantizing FFN layers too aggressively (below 3-bit) is harmful, but at 3–4 bit the tradeoff is acceptable.
- **Perplexity/KLD can be misleading** — real-world evals (LiveCodeBench, MMLU Pro) don't always correlate. Unsloth's Dynamic IQ2_XXS outperformed AesSedai's IQ3_S despite being 11GB smaller and having worse PPL.
- **MXFP4 is worse than Q4_K** on many tensors (attn_gate, attn_q, ssm_beta, ssm_alpha).

### 2. CoreML Palettization Algorithms (coremltools 9.0)

| Algorithm | Data Required | Quality | Speed | Available |
|-----------|--------------|---------|-------|-----------|
| **K-Means** (current) | None (data-free) | Baseline | Fast | ✅ |
| **Sensitive K-Means (SKM)** | ~128 calibration samples | **Best post-training** | Moderate | ✅ |
| **PostTrainingPalettizer** (Torch) | None | Same as K-Means | Fast | ✅ |
| **Differentiable K-Means (DKM)** | Full training data | Best overall | Very slow | ✅ |

**SKM** uses Fisher information (Hessian approximation via squared gradients) to weight K-Means clusters toward sensitive weight values. Based on SqueezeLLM paper. "Works well, better than data-free K-Means, for large transformer-based architectures." Requires only ~128 calibration samples.

### 3. Additional CoreML Quality Knobs

| Knob | Current | Potential | Size Impact |
|------|---------|-----------|-------------|
| `enable_per_channel_scale` | `False` | `True` | None |
| `cluster_dim` (vector palettization) | `1` (scalar) | `2` or `4` | None |
| `group_size` for FFN | `4` | `2` (more LUTs) | +~50% per weight |
| `nbits` for FFN | `4` (LUT4) | `6` (LUT6) | +50% per weight |

### 4. Existing Ablation Data (from fp16_ablation.py)

**Weight sensitivity ranking (lowest SNR = most sensitive to LUT4):**

| Family | SNR (dB) | Params (M) | FP16 Overhead (MB) |
|--------|----------|------------|---------------------|
| ssm_alpha | **16.0** | 1.97 | 2.94 |
| ssm_beta | **16.3** | 1.97 | 2.94 |
| attn_kv | 18.0 | 47.19 | 70.63 |
| attn_o | 18.6 | 94.37 | 141.37 |
| attn_q | 18.9 | 188.74 | 282.53 |
| ssm_out | **18.9** | 251.66 | 377.00 |
| ssm_z | 19.0 | 251.66 | 376.70 |
| ssm_qkv | 19.3 | 503.32 | 753.40 |
| mlp | 19.4 | 2335.70 | 3498.01 |

**D2 (F-layer attention FP16) per-chunk quality:**

| Chunk | Layers | Baseline cos_sim | D2 cos_sim | Delta |
|-------|--------|-----------------|------------|-------|
| 0 | LLL | 0.933 | N/A | — |
| 1 | FLLL | 0.932 | 0.954 | **+2.2%** |
| 4 | FLLL (mid) | 0.912 | 0.931 | **+1.9%** |
| 8 | F (final) | 0.970 | 0.981 | **+1.1%** |

### 5. Official Qwen3.5 Sampling Recommendations

| Mode | temp | top_p | top_k | presence_penalty | rep_penalty |
|------|------|-------|-------|-----------------|-------------|
| Think-on general | 1.0 | 0.95 | 20 | **1.5** | 1.0 |
| Think-off general | 0.7 | 0.8 | 20 | **1.5** | 1.0 |
| Think-off reasoning | 1.0 | 1.0 | 40 | **2.0** | 1.0 |

**Note**: Official Qwen3.5 uses `presence_penalty=1.5` for non-thinking mode, while our iOS app uses `1.0` for think-off. The official config also uses `repetition_penalty=1.0` (not 1.1) — relying on `presence_penalty` alone to control repetition.

---

## Proposed Experiments

### Experiment 1: Sensitive K-Means (SKM) for FFN Chunks ⭐ HIGH IMPACT

**Rationale**: This is the single highest-impact quality improvement available. SKM uses Fisher information to weight K-Means clusters toward sensitive weight values, producing significantly better LUT assignments than plain K-Means. Apple's docs state it "works well, better than data-free K-Means, for large transformer-based architectures." Requires only ~128 calibration samples.

**What changes**: Replace post-conversion `cto.coreml.palettize_weights()` (K-Means) with `SKMPalettizer` (calibration-based) applied before CoreML conversion.

**Implementation sketch**:
```python
from coremltools.optimize.torch.palettization import SKMPalettizer, SKMPalettizerConfig

skm_config = SKMPalettizerConfig.from_dict({
    "global_config": {
        "n_bits": 4,
        "granularity": "per_grouped_channel",
        "group_size": 4,
    },
    "calibration_nsamples": 128,
})

# calibration_data = load_calibration_data()  # ~128 diverse prompts
# loss_fn = lambda model, data: F.cross_entropy(model(data[0]), data[1])
palettizer = SKMPalettizer(torch_model, skm_config)
palettized_model = palettizer.compress(dataloader=calibration_data, loss_fn=loss_fn)
```

**Expected impact**: +2–5% cos_sim across all chunks, no size increase, no speed decrease.  
**Risk**: Requires calibration data pipeline and torch model forward pass integration.  
**Effort**: Medium (need to wire up calibration data loading + modify export.py).  
**Size impact**: None.

---

### Experiment 2: enable_per_channel_scale ⭐ LOW EFFORT

**Rationale**: Normalizes weights along output channels using per-channel scales before palettization. This is essentially free — it's a single flag in `OpPalettizerConfig`. Can significantly improve LUT quality for weights with large dynamic range (which SSM projections have, given SNR as low as 16 dB).

**What changes**: Add `enable_per_channel_scale=True` to the `OpPalettizerConfig` in export.py.

**Implementation**:
```python
op_config = cto.coreml.OpPalettizerConfig(
    nbits=4,
    granularity=cto.coreml._config.CompressionGranularity.PER_GROUPED_CHANNEL,
    group_size=4,
    enable_per_channel_scale=True,  # <-- ADD THIS
)
```

**Expected impact**: +0.5–2% cos_sim, especially for SSM and attention weights.  
**Risk**: Very low — worst case no change.  
**Effort**: Trivial (one flag).  
**Size impact**: Negligible (per-channel scales are tiny FP16 vectors).

---

### Experiment 3: Upgrade ssm_alpha + ssm_beta to FP16 ⭐ FREE QUALITY

**Rationale**: The ablation shows these are the most sensitive tensor families (16.0/16.3 dB SNR) but only 1.97M params each (~6 MB total overhead). This is essentially free quality — the ablation report identified this as "Policy 1 (latency-first): free bonus."

**What changes**: Add `"ssm_alpha"` and `"ssm_beta"` to the `D2_FP16_ATTN_FAMILIES` list in export.py, or create a new D3 policy.

**Implementation**:
```python
# In export.py, expand FP16 families:
D3_FP16_FAMILIES = ["attn_q", "attn_kv", "attn_o", "ssm_alpha", "ssm_beta"]
```

**Expected impact**: Small but non-zero quality improvement for ~6 MB cost.  
**Risk**: None.  
**Effort**: Trivial (one line change + re-export).  
**Size impact**: +6 MB (~0.15% of total).

---

### Experiment 4: LUT6 for FFN Chunks (instead of LUT4) 📊 SIZE TRADEOFF

**Rationale**: LUT6 gives 64 centroids vs LUT4's 16 centroids — 4× better weight representation. The ablation showed LUT6 was never tested. Unsloth data confirms that 3–4 bit is the "sweet spot" for FFN but the quality gap between 4-bit and 6-bit is significant for sensitive tensors like `ffn_down`.

**What changes**: Change `LUT_BITS=4` to `LUT_BITS=6` in config.py.

**Size impact analysis**:
- Current FFN weights at LUT4 gs=4: each weight stored as 4-bit index + LUT overhead
- At LUT6 gs=4: each weight stored as 6-bit index + larger LUT
- Approximate size increase: **+50% for FFN weights** (~+1 GB total model)
- Total model: ~4 GB → ~5 GB

**Expected impact**: Significant quality improvement (+3–8% cos_sim based on K-Means accuracy benchmarks).  
**Risk**: Model may not fit in 8GB iPhone memory with both infer+prefill loaded (~785 MB → ~1.3 GB).  
**Effort**: Trivial (one config change + full re-export).  
**Decision criteria**: Run memory profiling first to determine headroom.

---

### Experiment 5: Mixed LUT4/LUT6 — LUT6 for Sensitive Layers 📊 SMART TRADEOFF

**Rationale**: Error compounds through layers. Middle chunks show lowest cos_sim (0.912). Instead of LUT6 everywhere, use LUT6 selectively for the most sensitive or highest-impact chunks/layers. Unsloth's "Dynamic" quantization uses exactly this approach.

**Variant A — LUT6 for edge chunks** (first + last):
- Chunks 0, 1, 8 → LUT6 (first 7 layers + final layer = 8 layers)
- Chunks 2–7 → LUT4 (24 layers)
- Size: +~300 MB

**Variant B — LUT6 for ssm projections only**:
- `ssm_qkv`, `ssm_out`, `ssm_z` → LUT6 (1006M params)
- Everything else → LUT4
- Size: +~300 MB

**Variant C — LUT6 for middle chunks** (where cos_sim is lowest):
- Chunks 3–5 (layers 11–22) → LUT6
- Rest → LUT4
- Size: +~400 MB

**Implementation**: Modify `_apply_selective_lut4()` to accept per-family nbits overrides.

**Expected impact**: +1–3% cos_sim for targeted layers.  
**Risk**: Low. Needs per-op nbits configuration in export.py.  
**Effort**: Medium.

---

### Experiment 6: Smaller group_size (gs=2) for SSM Projections 📊 MODERATE

**Rationale**: Smaller group_size = more LUTs = better approximation. Current gs=4 means each 4-channel group shares one 16-entry LUT. At gs=2, each 2-channel group gets its own LUT — doubles the number of LUTs but each fits its local weight distribution better. Particularly beneficial for SSM projections (lowest SNR).

**What changes**: Modify export.py to use gs=2 for SSM families, keep gs=4 for MLP.

**Expected impact**: +0.5–2% cos_sim for L-layer chunks.  
**Risk**: Low — group_size=8 works fine for embed/lm_head, gs=2 should be at least as good as gs=4.  
**Effort**: Medium (need per-op group_size in palettization config).  
**Size impact**: +~10–15% for SSM weights (small total).

---

### Experiment 7: Vector Palettization (cluster_dim=2) 🔬 EXPERIMENTAL

**Rationale**: 2D clustering (`cluster_dim=2`) captures correlations between adjacent weight values, producing better approximations than scalar K-Means. Available in coremltools 9.0.

**What changes**: Set `cluster_dim=2` in `OpPalettizerConfig`.

**Expected impact**: Unknown — not benchmarked for LLMs. Could be +1–3%.  
**Risk**: May not work well for all weight patterns. May have ANE performance implications.  
**Effort**: Low (one parameter change).  
**Size impact**: None.

---

### Experiment 8: Sampling Parameter Alignment 🎯 ZERO COST

**Rationale**: Official Qwen3.5 recommends `presence_penalty=1.5` for non-thinking mode general tasks. Our iOS app uses `1.0`. The official config also uses `repetition_penalty=1.0` (relies on presence_penalty alone). Aligning with official recommendations could improve output quality significantly, especially for the multi-turn repetition issue.

**What changes in InferenceConfig.swift**:
```swift
// Think-off (non-thinking) mode:
SamplingConfig(temperature: 0.7, topP: 0.8, topK: 20,
    presencePenalty: 1.5,   // was 1.0
    repetitionPenalty: 1.0, // was 1.1
    frequencyPenalty: 0.0)  // was 0.05
```

**Expected impact**: Better diversity, less repetition, potentially better output quality.  
**Risk**: May need tuning — `presence_penalty=1.5` could cause language mixing at lower bit quantization.  
**Effort**: Trivial (Swift config change only, no re-export needed).  
**Size impact**: None.

---

## Recommended Execution Order

| Priority | Experiment | Effort | Risk | Impact |
|----------|-----------|--------|------|--------|
| 🥇 1 | **Exp 8**: Sampling alignment | Trivial | Low | Medium |
| 🥈 2 | **Exp 2**: enable_per_channel_scale | Trivial | None | Low–Med |
| 🥉 3 | **Exp 3**: ssm_alpha/beta FP16 | Trivial | None | Low |
| 4 | **Exp 5**: Mixed LUT4/LUT6 | Medium | Low | Medium |
| 5 | **Exp 1**: SKM palettization | Medium | Medium | **High** |
| 6 | **Exp 6**: gs=2 for SSM | Medium | Low | Low–Med |
| 7 | **Exp 4**: Full LUT6 FFN | Trivial | **High (memory)** | High |
| 8 | **Exp 7**: Vector palettization | Low | Unknown | Unknown |

**Recommended first batch** (can be done in one re-export cycle):
- Exp 2 + Exp 3 together: `enable_per_channel_scale=True` + ssm_alpha/beta FP16
- Exp 8 independently (Swift-only change)

**Recommended second batch** (requires more infrastructure):
- Exp 5 (mixed LUT4/LUT6) or Exp 1 (SKM)

---

## Measurement Plan

For each experiment, measure:
1. **Per-chunk cos_sim** (vs FP16 reference) — using fp16_ablation.py framework
2. **End-to-end text quality** — 5 grading prompts, human evaluation
3. **Model size** — total .mlpackage + .mlmodelc
4. **Decode latency** — on-device tok/s (iPhone A18 Pro)
5. **Memory footprint** — jetsam headroom with both infer+prefill loaded

Baseline to beat:
- cos_sim: 0.912–0.970 per chunk (D2 policy)
- Quality: 19/19 single-turn, 1/3 multi-turn
- Size: ~4 GB combined dedup
- Speed: 6–7 tok/s
- Memory: ~785 MB both models loaded
