#!/usr/bin/env python3
"""Qwen3.5 4B — FP16 Ablation Results Report.

Compile and display final results from the ablation experiment.
Run after all ablation phases are complete.
"""

# ══════════════════════════════════════════════════════════════════════
# QWEN3.5 4B — FP16 TENSOR FAMILY ABLATION: FINAL REPORT
# ══════════════════════════════════════════════════════════════════════

report = """
═══════════════════════════════════════════════════════════════════════════════
 QWEN3.5 4B — FP16 TENSOR FAMILY ABLATION EXPERIMENT: FINAL REPORT
═══════════════════════════════════════════════════════════════════════════════

1. TENSOR FAMILY MAPPING
─────────────────────────────────────────────────────────────────────────────

 Concept          │ HF Weight Key Pattern               │ ANEMLL Code           │ Layer Type  │ CoreML MIL Weight Name Pattern
 ─────────────────┼──────────────────────────────────────┼───────────────────────┼─────────────┼────────────────────────────────────────
 attn_q (+ gate)  │ self_attn.q_proj.weight              │ self_attn.q_proj      │ F-layers    │ model_model_layers_N_self_attn_q_proj_weight
 attn_kv          │ self_attn.{k,v}_proj.weight          │ self_attn.{k,v}_proj  │ F-layers    │ model_model_layers_N_self_attn_{k,v}_proj_weight
 attn_o           │ self_attn.o_proj.weight              │ self_attn.o_proj      │ F-layers    │ model_model_layers_N_self_attn_o_proj_weight (*)
 ssm_alpha        │ linear_attn.in_proj_a.weight         │ self_attn.in_proj_a   │ L-layers    │ model_model_layers_N_self_attn_in_proj_a_weight
 ssm_beta         │ linear_attn.in_proj_b.weight         │ self_attn.in_proj_b   │ L-layers    │ model_model_layers_N_self_attn_in_proj_b_weight
 ssm_qkv          │ linear_attn.in_proj_qkv.weight       │ self_attn.in_proj_qkv │ L-layers    │ model_model_layers_N_self_attn_in_proj_qkv_weight
 ssm_z            │ linear_attn.in_proj_z.weight         │ self_attn.in_proj_z   │ L-layers    │ model_model_layers_N_self_attn_in_proj_z_weight
 ssm_conv         │ linear_attn.conv1d.weight            │ self_attn.conv2d      │ L-layers    │ model_model_layers_N_self_attn_conv2d_weight
 ssm_out          │ linear_attn.out_proj.weight          │ self_attn.out_proj    │ L-layers    │ squeeze_N (reshaped o_proj)
 mlp              │ mlp.{gate,up,down}_proj.weight        │ mlp.*_proj            │ All layers  │ model_model_layers_N_mlp_{gate,up,down}_proj_weight

 (*) F-layer o_proj appears as "squeeze_N" in some chunks after MIL optimization.

 CRITICAL NOTE: "attn_gate" is NOT a separate weight in Qwen3.5. The attention output
 gate is extracted from the second half of q_proj's output channels (q_head_dim = 2 × head_dim).
 Keeping attn_q in FP16 automatically keeps the gate in FP16.

 Also: A_log and dt_bias are ALWAYS FP32 (scalar parameters, never quantized).


2. WEIGHT ANALYSIS — SIZE AND QUANTIZATION SENSITIVITY
─────────────────────────────────────────────────────────────────────────────

 LUT4 quantization: 4-bit KMeans palettization, per_grouped_channel, group_size=4

 Family          Tensors  Params(M)  FP16(MB)  LUT4(MB)  Overhead(MB)  %Model   SNR(dB)  RMSE
 ─────────────── ──────── ───────── ───────── ───────── ──────────── ─────── ──────── ────────
 ssm_alpha           24       1.97      3.93      0.99         2.94    0.2%    16.0   0.002858
 ssm_beta            24       1.97      3.93      0.99         2.94    0.2%    16.3   0.001884
 attn_kv             18      47.19     94.37     23.74        70.63    3.8%    18.0   0.001771
 attn_o               9      94.37    188.74     47.37       141.37    7.6%    18.6   0.001553
 attn_q               9     188.74    377.49     94.96       282.53   15.3%    18.9   0.001443
 ssm_out             24     251.66    503.32    126.32       377.00   20.4%    18.9   0.001348
 ssm_z               24     251.66    503.32    126.62       376.70   20.4%    19.0   0.001549
 ssm_qkv             24     503.32   1006.63    253.23       753.40   40.7%    19.3   0.001524
 mlp                 99    2335.70   4671.41   1173.39      3498.01  189.0%    19.4   0.001022
 ─────────────── ──────── ───────── ───────── ───────── ──────────── ─────── ──────── ────────
 TOTAL                              7355.07   1851.10

 KEY: SNR = Signal-to-Noise Ratio (lower = MORE sensitive to quantization)
      RMSE = Root Mean Square Error of simulated LUT4 quantization
      Overhead = additional bytes when keeping family in FP16 vs LUT4


3. ABLATION EXPERIMENT RESULTS
─────────────────────────────────────────────────────────────────────────────

 Setup: 4 evaluation samples per chunk, random hidden state inputs,
        cosine similarity between FP16 reference and quantized model outputs.
        Deterministic seed (42). CPU-only inference.

 │ Chunk 0 (layers 0-2, LLL — pure linear attention)
 │
 │  Config                         CosSim     Δ vs LUT4
 │  ──────────────────────────── ────────── ──────────
 │  A1_all_lut4                   0.933341   baseline
 │  B2_fp16_ssm_alpha             0.933874   +0.000533
 │  B3_fp16_ssm_beta              0.933331   −0.000010
 │  C1_fp16_ssm_alpha+beta        0.933885   +0.000544
 │
 │  → SSM small projections: negligible improvement on pure L-layer chunks.

 │ Chunk 1 (layers 3-6, FLLL — early, first F-layer)
 │
 │  Config                         CosSim     Δ vs LUT4
 │  ──────────────────────────── ────────── ──────────
 │  A1_all_lut4                   0.932488   baseline
 │  C1_fp16_ssm_alpha+beta        0.933213   +0.000725
 │  B1_fp16_attn_q                0.938568   +0.006080
 │  D1_fp16_all_primary           0.939318   +0.006830
 │  D2_fp16_attn_all              0.954311   +0.021823
 │
 │  → attn_q gives +0.6%, D2 (all attention) gives +2.2%. Clear winner.

 │ Chunk 4 (layers 15-18, FLLL — middle of model)
 │
 │  Config                         CosSim     Δ vs LUT4
 │  ──────────────────────────── ────────── ──────────
 │  A1_all_lut4                   0.911871   baseline
 │  D2_fp16_attn_all              0.930542   +0.018671
 │
 │  → Middle chunk most sensitive. D2 gives +1.9%. Baseline cos lower (0.91).

 │ Chunk 8 (layer 31, F — pure full attention, final layer)
 │
 │  Config                         CosSim     Δ vs LUT4
 │  ──────────────────────────── ────────── ──────────
 │  A1_all_lut4                   0.970399   baseline
 │  C1_fp16_ssm_alpha+beta        0.970399   +0.000000
 │  B1_fp16_attn_q                0.971117   +0.000718
 │  D1_fp16_all_primary           0.971117   +0.000718
 │  D2_fp16_attn_all              0.981125   +0.010726
 │
 │  → No SSM in F-layer (as expected). D2 gives +1.1%. Baseline already high (0.97).


4. PARETO ANALYSIS
─────────────────────────────────────────────────────────────────────────────

 Config                   FP16 Families           Overhead(MB)  %Model  Avg Δcos   Efficiency
 ──────────────────────── ────────────────────── ──────────── ─────── ──────── ────────────
 A1  all_lut4             (none)                        0.00    0.0%  baseline
 C1  ssm_alpha+beta       alpha, beta                   5.89    0.3%  +0.0004  0.07/GB
 B1  fp16_attn_q          q_proj                      282.53   15.3%  +0.0034  0.01/GB
 D1  all_primary          q, alpha, beta              288.41   15.6%  +0.0038  0.01/GB
 D2  fp16_attn_all        q, kv, o (all F-attn)      494.53   26.7%  +0.0170  0.03/GB

 PARETO FRONTIER (non-dominated configs):
   ★ A1_all_lut4           — 0 overhead, baseline quality
   ★ C1_ssm_alpha+beta     — ~0 overhead, tiny quality gain  (DOMINATED by noise)
   ★ D2_fp16_attn_all      — 27% overhead, +1.7% avg cos_sim (CLEAR WINNER)

 B1 and D1 are DOMINATED by D2:
   D2 gives 5× the quality improvement for only 1.7× the overhead of B1.

 There is NO intermediate Pareto point between C1 and D2:
   - attn_q alone (B1) is inefficient — kv and o_proj contribute most of the quality gain.
   - The jump from B1 (+0.3%) to D2 (+1.7%) costs only 212 MB more, for 5× the benefit.


5. ANSWERS TO KEY QUESTIONS
─────────────────────────────────────────────────────────────────────────────

 Q: Which single tensor family gives the best accuracy gain per latency cost?
 A: ssm_alpha — essentially zero cost (2.94 MB / 0.2%), but also near-zero benefit.
    Among meaningful families: attn_kv or attn_o (not tested individually but D2
    minus B1 indicates they contribute most of the +1.4% delta).

 Q: Which combination gives the best overall tradeoff?
 A: D2_fp16_attn_all (q+kv+o projections in FP16). It is the only config that
    meaningfully improves quality while keeping overhead under 30%.

 Q: Are attn_q and attn_gate worth FP16 globally, or only in attention blocks?
 A: Only in F-layers (full attention), which is where they exist. Qwen3.5 has
    attn_q only in F-layers (8 out of 32 layers). The gate is part of q_proj
    and cannot be separated. Keeping attn_q alone gives only +0.3% — not enough
    to justify 282 MB overhead. Better to keep ALL F-layer attention in FP16.

 Q: Are ssm_alpha and ssm_beta worth FP16 globally?
 A: No. Despite having the HIGHEST quantization sensitivity (lowest SNR: 16.0 dB),
    they are so small (2M params each) that their impact on model output is negligible.
    Keeping them in FP16 is free but gives < 0.1% quality improvement.

 Q: Is there a clearly dominant mixed-precision policy?
 A: Yes. The Pareto analysis shows exactly TWO practical policies:
    1) All LUT4 (maximum speed, acceptable quality)
    2) D2 (all F-layer attention in FP16, +27% size, +1.7% quality)
    Nothing between these points is efficient.


6. DEPLOYMENT RECOMMENDATIONS
─────────────────────────────────────────────────────────────────────────────

 ┌─────────────────────────────────────────────────────────────────────────┐
 │ POLICY 1: LATENCY-FIRST (recommended for memory-constrained devices)   │
 │                                                                         │
 │   All weights: LUT4 per_grouped_channel gs=4                           │
 │   ssm_alpha + ssm_beta: FP16 (free, +0.04% quality)                   │
 │   A_log + dt_bias: FP32 (already FP32, never quantized)               │
 │                                                                         │
 │   Model size: ~1851 MB (LUT4 baseline)                                │
 │   Quality: cos_sim ≈ 0.93 (LUT4 baseline, per-chunk average)          │
 │                                                                         │
 │   Implementation:                                                       │
 │     In postprocess(), set op_name_configs with None for all ops        │
 │     matching "in_proj_a_weight" and "in_proj_b_weight"                 │
 └─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────┐
 │ POLICY 2: BALANCED (recommended for quality-sensitive deployment)       │
 │                                                                         │
 │   F-layer attention weights: FP16                                      │
 │     - self_attn.q_proj (8 layers × [8192, 2560])                      │
 │     - self_attn.k_proj (8 layers × [1024, 2560])                      │
 │     - self_attn.v_proj (8 layers × [1024, 2560])                      │
 │     - self_attn.o_proj (8 layers × [2560, 4096])                      │
 │   Everything else: LUT4 per_grouped_channel gs=4                       │
 │   ssm_alpha + ssm_beta: FP16 (free bonus)                             │
 │   A_log + dt_bias: FP32 (already FP32)                                │
 │                                                                         │
 │   Model size: ~2346 MB (+495 MB, +26.7% vs LUT4 baseline)             │
 │   Quality: cos_sim ≈ 0.95 (+1.7% vs LUT4 baseline, per-chunk avg)    │
 │                                                                         │
 │   Implementation:                                                       │
 │     In postprocess(), set op_name_configs with None for all ops        │
 │     matching: q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight│
 │     in F-layers (layers 3,7,11,15,19,23,27,31)                        │
 │     Also skip: in_proj_a_weight, in_proj_b_weight (free bonus)         │
 └─────────────────────────────────────────────────────────────────────────┘

 There is no good ACCURACY-FIRST policy between D2 and all-FP16.
 The next step (keeping SSM projections in FP16) adds 1.5 GB for uncertain benefit.
 Going all-FP16 adds 5.5 GB (+297%), which defeats the purpose of quantization.


7. CAVEATS
─────────────────────────────────────────────────────────────────────────────

 1. The "attn_gate" concept cannot be isolated from attn_q in Qwen3.5 because
    the gate weights are the second half of q_proj's output channels. They
    share the same Conv2d weight matrix. Testing attn_q effectively tests both.

 2. Evaluation used random hidden state inputs (seeded), not actual model
    activations. Real-world quality differences may be larger or smaller
    depending on the activation distribution during inference.

 3. Per-chunk cosine similarity is a proxy for end-to-end generation quality.
    Errors compound across 9 chunks, so the actual generation quality difference
    between LUT4 and D2 is likely LARGER than the per-chunk measurements suggest.

 4. SSM state parameters (A_log, dt_bias) were already FP32 and excluded from
    the experiment. These are tiny scalar parameters that are never quantized.

 5. Middle chunks (chunk 4, layers 15-18) showed the lowest baseline quality
    (cos_sim=0.91), suggesting quantization errors accumulate through layers.
    The D2 policy particularly helps these chunks.

 6. The experiment did not test LUT6 as an alternative to LUT4+FP16.
    LUT6 for all weights might achieve similar quality to D2 with smaller
    or comparable overhead, but was out of scope for this experiment.

 7. For Qwen3.5-2B (24 layers, 6 F-layers), the same policy structure applies
    but with proportionally smaller overhead.

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(report)
