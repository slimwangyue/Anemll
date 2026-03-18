# Qwen3.5 ANE-Friendliness Audit

Model file: `anemll/models/qwen3_5_model.py`
Date: 2026-03-18

## Summary
- Full-attention path: **ANE-friendly yes** (Conv2d projections, fixed-width cache mask contract).
- Linear-attention projection/state path: **ANE-friendly mostly yes** (Conv2d projections, Conv2d temporal conv, fixed-shape state buffers).
- Complex linear kernels (`_chunk_gated_delta_rule`, `_recurrent_gated_delta_rule`): **ANE friendly not sure**.

## Component Check
- `Qwen35RMSNorm`: ANE-friendly yes (LayerNorm kernel trick, static affine scale).
- `Qwen35MLP`: ANE-friendly yes (Conv2d 1x1, SiLU, elementwise mul).
- `Qwen35FullAttention` projections (`q/k/v/o`): ANE-friendly yes (Conv2d 1x1).
- `Qwen35FullAttention` cache mask: ANE-friendly yes (fixed width = `state_length`).
- `Qwen35LinearAttention` projections (`in_proj_qkv/a/b/z`, `out_proj`): ANE-friendly yes (Conv2d 1x1).
- `Qwen35LinearAttention` temporal conv (`conv2d`): ANE-friendly yes (depthwise Conv2d + static slice update).
- `Qwen35LinearAttention` state buffers (`linear_conv_state`, `linear_recurrent_state`): ANE-friendly yes (fixed shapes).
- `_chunk_gated_delta_rule`: ANE friendly not sure (triangular recurrence + dynamic pad/chunk ops).
- `_recurrent_gated_delta_rule`: ANE friendly not sure (per-token Python loop recurrence).

## Recommended Next ANE-Lowering Step
- Replace `_chunk_gated_delta_rule` and `_recurrent_gated_delta_rule` with MIL-lowerable blocks using static loop bounds/chunk schedule and avoid dynamic pad where possible.
