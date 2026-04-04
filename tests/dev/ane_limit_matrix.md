# ANE Limit Matrix — Qwen3.5-4B Prefill

> **Last updated:** 2026-04-01  
> **Commit:** aa7e7c3  
> **Model:** Qwen3.5-4B, 32 hidden layers, hidden=2560, 16 attn heads, 4 KV heads, head_dim=128  
> **Attention pattern:** `full_attention_interval=4` (every 4th layer full SDPA, others linear with chunked gated delta rule)  
> **chunk_size (recurrence):** 16 (set in `qwen3_5_model.py:978`, reduced from original 64)

---

## Best-Known Empirical ANE Limits

| Device | Chip | RAM | iOS | ANE op limit (MIL ops) | Confirmed pass | Confirmed fail | Notes |
|--------|------|-----|-----|------------------------|---------------|---------------|-------|
| iPhone 14 Pro Max | A16 | 6 GB | 26.5 beta | 18,101 ≤ limit < 21,656 | 18,101 (6-chunk LUT6) | 21,656 (5-chunk LUT6) | Error -14, no CPU fallback either |
| Mac Studio | M1 Ultra | 36 GB | macOS (Sequoia) | ≥ 21,922 | 21,922 (8L fp16) | — | ANE compile warns on 8L+ but CPU fallback works |

**Production recommendation:** **6 chunks** (6 or 5 layers/chunk, ≤18,101 ops) for iPhone A16.  
8 chunks (4 layers/chunk, ~11K ops) is the conservative safe choice.

---

## Test Matrix

### Legend
- **CU**: Compute units used for loading (`ANE` = cpuAndNeuralEngine, `CPU` = cpuOnly, `ALL` = all)
- **Isolated**: Whether the model was the only one loaded in that app session (avoids resource contention)
- **Result**: `PASS` = loaded+ran successfully, `FAIL` = error during load/compile/run

### A. Individual Op Tests (iPhone 14 Pro Max, A16, iOS 26.5 beta)

All tested isolated on device. All PASS on ANE.

| # | Model | Ops | Quant | Weight MB | CU | Result | Notes |
|---|-------|-----|-------|-----------|-----|--------|-------|
| 1 | test_baseline | 4 | FP16 | 0 | ANE | PASS | Identity-like |
| 2 | test_dw_conv | 21 | FP16 | 0 | ANE | PASS | Depthwise conv (groups=1536) |
| 3 | test_gather | 17 | FP16 | 0.5 | ANE | PASS | Gather/scatter ops |
| 4 | test_less | 7 | FP16 | 0 | ANE | PASS | Comparison ops |
| 5 | test_less_and_one_hot | ~10 | FP16 | 0 | ANE | PASS | one_hot + less |
| 6 | test_one_hot | ~5 | FP16 | 0 | ANE | PASS | one_hot standalone |
| 7 | test_one_hot_conv | ~8 | FP16 | 0 | ANE | PASS | one_hot + conv1d |
| 8 | test_repeat_interleave | ~10 | FP16 | 0 | ANE | PASS | repeat_interleave pattern |
| 9 | test_slice_update | ~12 | FP16 | 0.3 | ANE | PASS | Slice + update (state-like) |
| 10 | test_softplus | ~15 | FP16 | 2.5 | ANE | PASS | Softplus activation |
| 11 | test_tril_matmul | ~10 | FP16 | 0 | ANE | PASS | Lower-triangular matmul |
| 12 | test_state | ~8 | FP16 | 0 | ANE | PASS | CoreML State read/write |
| 13 | test_state_lut6 | ~10 | LUT6 | 0 | ANE | PASS | State + LUT6 weight |
| 14 | test_lut6 | ~8 | LUT6 | 0 | ANE | PASS | constexpr_lut_to_dense |
| 15 | test_multi_state_lut6 | ~15 | LUT6 | 0 | ANE | PASS | Multiple states + LUT6 |
| 16 | test_linear_attn_combined | ~60 | FP16 | 10 | ANE | PASS | Linear attn subgraph |

**Conclusion:** All individual ops used by Qwen3.5 are ANE-compatible on A16. The issue is total op count, not op type.

### B. Real Qwen3.5 Prefill Chunks (iPhone 14 Pro Max, A16, iOS 26.5 beta)

All use: `batch_size=512, context_length=2048, chunk_size=16, valid_len=ON, exact prefill`

| # | Model | Chunks | Chunk ID | Layers | Quant | MIL Ops | Weight MB | Pkg MB | CU | Isolated | Result | Time | Error |
|---|-------|--------|----------|--------|-------|---------|-----------|--------|-----|----------|--------|------|-------|
| B1 | test_mixed_attn_1layer | — | — | 1 | FP16 | 92 | 73 | 77 | ANE | Y | PASS | <1s | — |
| B2 | test_real_9layer_state | — | — | 9 | FP16 | 580 | 397 | 410 | ANE | Y | PASS | ~2s | — |
| B3 | test_real_qwen_prefill_2L | 16 | 0 | 2 | FP16 | 7,144 | 431 | 453 | ANE | Y | PASS | 22.97s | — |
| B4 | test_real_qwen_prefill_2L | 16 | 0 | 2 | FP16 | 7,144 | 431 | 453 | ANE | N (w/ 4L) | FAIL | 4.30s | -14 (resource contention) |
| B5 | test_real_qwen_prefill_4L | 8 | 0 | 4 | FP16 | 10,993 | 852 | 895 | ANE | Y | PASS | 49.47s | — |
| B6 | test_real_qwen_prefill_4L | 8 | 0 | 4 | FP16 | 10,993 | 852 | 895 | ANE | N (w/ 2L) | PASS | 50.41s | 2L failed instead |
| B7 | test_real_qwen_prefill_4L_LUT6 | 8 | 0 | 4 | LUT6 | 10,991 | 325 | 343 | ANE | Y | PASS | 66.92s | — |
| B8 | test_real_qwen_prefill_4L_LUT6 | 8 | 0 | 4 | LUT6 | 10,991 | 325 | 343 | CPU | Y | PASS | 7.76s | — |
| B9 | prefill_LUT6_chunk4_bs512 | 8 | 4 | 4 | LUT6 | 10,991 | 327 | 343 | ANE | Y | PASS | 38.93s | — |
| B10 | prefill_LUT6_chunk4_bs512 | 8 | 4 | 4 | LUT6 | 10,991 | 327 | 343 | CPU | Y | PASS | 4.23s | — |
| B11 | test_prefill_7chunks_LUT6 | 7 | 0 | 5 | LUT6 | 14,546 | 407 | 429 | ANE | — | — | — | Not yet tested on device |
| B12 | test_prefill_6chunks_LUT6 | 6 | 0 | 6 | LUT6 | 18,101 | 490 | 516 | ANE | Y | PASS | 66.37s | — |
| B13 | test_prefill_6chunks_LUT6 | 6 | 0 | 6 | LUT6 | 18,101 | 490 | 516 | CPU | Y | PASS | 34.18s | — |
| B14 | test_prefill_5chunks_LUT6 | 5 | 0 | 7 | LUT6 | 21,656 | 572 | 603 | ANE | Y | FAIL | 66.76s | -14: Failed to build execution plan |
| B15 | test_prefill_5chunks_LUT6 | 5 | 0 | 7 | LUT6 | 21,656 | 572 | 603 | CPU | Y | PASS | 20.14s | CPU fallback works |
| B16 | test_real_qwen_prefill_8L | 4 | 0 | 8 | FP16 | 21,922 | 1,700 | 1,789 | ANE | — | — | — | Not tested on device (too large) |
| B17 | prefill_LUT6_chunk0 (4-chunk) | 4 | 0 | 8 | LUT6 | 21,920 | 618 | 650 | ANE | Y | FAIL | — | -14 (early session, cs=16) |

### C. Mac Loading Tests (Mac Studio M1 Ultra, 36 GB, macOS)

| # | Model | Ops | Quant | CU | Result | Time | Notes |
|---|-------|-----|-------|-----|--------|------|-------|
| C1 | test_real_qwen_prefill_2L | 7,144 | FP16 | CPU_AND_NE | PASS | ~5s | No ANE compile warnings |
| C2 | test_real_qwen_prefill_4L | 10,993 | FP16 | CPU_AND_NE | PASS | ~8s | No ANE compile warnings |
| C3 | test_real_qwen_prefill_4L_LUT6 | 10,991 | LUT6 | CPU_AND_NE | PASS | 32.0s | No ANE compile warnings |
| C4 | test_real_qwen_prefill_4L_LUT6 | 10,991 | LUT6 | CPU_ONLY | PASS | 2.5s | — |
| C5 | test_real_qwen_prefill_8L | 21,922 | FP16 | CPU_AND_NE | PASS | ~15s | MILCompilerForANE error, CPU fallback |
| C6 | test_real_qwen_prefill_8L | 21,922 | FP16 | CPU_ONLY | PASS | ~12s | — |
| C7 | test_real_qwen_prefill_8L | 21,922 | FP16 | ALL | PASS | ~15s | ANE compile fails, CPU fallback |
| C8 | test_prefill_5chunks_LUT6 | 21,656 | LUT6 | CPU_AND_NE | PASS | 67.1s | — |
| C9 | test_prefill_6chunks_LUT6 | 18,101 | LUT6 | CPU_AND_NE | PASS | 55.7s | — |
| C10 | test_prefill_7chunks_LUT6 | 14,546 | LUT6 | CPU_AND_NE | PASS | 82.7s | — |

---

## Chunk Config Reference

For `Qwen3.5-4B` with 32 hidden layers, `batch_size=512`, `context_length=2048`, `chunk_size=16`:

| Chunks | Layers/chunk (chunk0) | MIL Ops (chunk0, LUT6) | Pkg MB (LUT6) | iPhone A16 ANE | Total model size |
|--------|----------------------|------------------------|---------------|----------------|-----------------|
| 4 | 8 | 21,920 | 650 | FAIL ❌ | 2.6 GB |
| 5 | 7 | 21,656 | 603 | FAIL ❌ | ~3.0 GB |
| 6 | 6 | 18,101 | 516 | PASS ✅ | ~3.1 GB |
| 7 | 5 | 14,546 | 429 | untested | ~3.0 GB |
| 8 | 4 | 10,991 | 343 | PASS ✅ | 2.6 GB |

Note: Layer distribution uses `divmod(32, N)` — some chunks get 1 extra layer.  
Chunk0 always gets the most layers (and thus the most ops).

---

## Key Observations

1. **Op count is the primary factor.** All individual ops pass; only when total MIL ops exceed ~18-21K does error -14 occur.
2. **LUT6 does NOT add ops.** `constexpr_lut_to_dense` replaces weight `const` ops 1:1. LUT6 model has ~same op count as FP16.
3. **Error -14 is absolute on iPhone.** When it fails on ANE, CPU fallback ALSO fails (unlike Mac which handles gracefully).
4. **Resource contention matters.** Loading multiple models in one app session can cause models that pass in isolation to fail (B3/B4 vs B5/B6).
5. **iPhone ANE compile time scales with ops.** ~11K ops → ~40-67s, ~18K ops → ~66s, ~22K ops → 67s then fail.
6. **Mac ANE compile is more lenient.** 22K ops causes ANE compile warnings but CPU fallback works seamlessly.

---

## Abandoned Approaches

| Approach | Result | Reason abandoned |
|----------|--------|-----------------|
| Neumann series (matrix doubling) | 10,993 → 8,227 ops | BNNS compilation errors ("map::at: key not found") |
| chunk_size sweep (8, 12, 16, 32, 64) | cs=16 best trade-off | Loop1 O(cs²) vs Loop2 O(seq/cs), can't reduce enough |
| Stateless export | N/A | coremltools crashes without states |
| Identity pass for unused State inputs | crash | Monkey-patched `handle_unused_inputs` to skip State-type |

---

## Export Configs Tested

| Config | chunk_size | valid_len | seq_len | bucket | target | coremltools |
|--------|-----------|-----------|---------|--------|--------|-------------|
| Standard | 16 | ON | 512 | exact | iOS18 | 9.0 |

---

## How to reproduce

```bash
# Export chunk0 for N-chunk config with LUT6
cd /Users/yw68/Anemll
.venv/bin/python3 tests/dev/export_chunk_sweep_lut6.py <N>

# Export all chunks for 8-chunk production
.venv/bin/python3 tests/dev/export_8chunks_lut6.py

# Push to iPhone and test
xcrun devicectl device copy to --device 00008120-0011554C3C90C01E \
  --domain-type appDataContainer --domain-identifier com.anemll.ANETestLoader \
  --source <model.mlpackage> --destination Documents/<model.mlpackage>
xcrun devicectl device process launch --device 00008120-0011554C3C90C01E \
  --terminate-existing com.anemll.ANETestLoader
sleep 120
xcrun devicectl device copy from --device 00008120-0011554C3C90C01E \
  --domain-type appDataContainer --domain-identifier com.anemll.ANETestLoader \
  --source Documents/results.txt --destination /tmp/results.txt
cat /tmp/results.txt
```
