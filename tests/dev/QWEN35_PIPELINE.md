# Qwen3.5-4B ANE Pipeline — Reproducible Guide

> **Last verified**: March 2026  
> **Machine**: macOS (Apple Silicon, ≥16 GB RAM)  
> **Model**: Qwen3.5-4B (HuggingFace)  
> **Output**: CoreML models optimized for Apple Neural Engine  

---

## Overview

This guide converts a HuggingFace Qwen3.5-4B model to CoreML format for
on-device ANE inference with multi-round conversation support. The pipeline
produces a **2.5 GB** deployable model set running at **~10 tok/s** on ANE.

### Pipeline Steps

```
Step 1: Export       → 10 separate .mlpackage files (embeddings, lm_head, 4×decode, 4×prefill)
Step 2: Combine      → 4 dedup .mlpackage files (decode+prefill share weights)
Step 3: Compile      → .mlmodelc for fast loading
Step 4: Validate     → multi-round conversation correctness
Step 5: Chat         → browser-based multi-round chat UI
```

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Hidden size | 2560 |
| Layers | 32 (hybrid: full + linear attention) |
| Attention heads | 20 (QKV) / 4 (KV) |
| Head dim | 128 |
| Intermediate size | 9728 |
| Vocab size | 248320 |
| Context length | 1024 |

### Quantization Config

| Component | Quantization | Pieces | Approx Size |
|-----------|-------------|--------|-------------|
| Embeddings | LUT4 | 1 | 304 MB |
| FFN decode | LUT4 | 4 chunks | 428 MB each |
| FFN prefill | LUT4 | 4 chunks | 437 MB each |
| LM head | LUT6 + argmax | 1 | 462 MB |
| **Total (separate)** | | **10 files** | **~4.2 GB** |
| **Total (dedup)** | | **6 files** | **~2.5 GB** |

---

## Prerequisites

### System Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- ≥16 GB RAM
- ≥10 GB free disk space
- Xcode Command Line Tools installed

```bash
# Verify Xcode CLI tools
xcrun --find coremlcompiler
```

### HuggingFace Model

Download the Qwen3.5-4B model from HuggingFace:

```bash
# Set your model path (adjust as needed)
export HF_MODEL="/path/to/Qwen__Qwen3.5-4B"

# Download (if not already available)
# pip install huggingface_hub
# huggingface-cli download Qwen/Qwen3.5-4B --local-dir "$HF_MODEL"
```

### Python Environment

```bash
cd /path/to/Anemll    # Clone of the ANEMLL repo

# Create venv (Python 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify key packages
python -c "import coremltools; print(f'coremltools {coremltools.__version__}')"
python -c "import torch; print(f'torch {torch.__version__}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"
```

Required versions (tested):
- `coremltools >= 9.0`
- `torch >= 2.5.0`
- `transformers >= 4.39.0`
- `scikit-learn <= 1.5.1`

---

## Step 1: Export (.mlpackage)

Export all 10 model components. This is the longest step (~30–60 min depending on hardware).

```bash
source .venv/bin/activate
cd /path/to/Anemll

export HF_MODEL="/path/to/Qwen__Qwen3.5-4B"
export OUTPUT_DIR="/path/to/output"

python tests/dev/qwen35_export.py \
    --model "$HF_MODEL" \
    --output "$OUTPUT_DIR"
```

Use `--skip-existing` to resume if interrupted:
```bash
python tests/dev/qwen35_export.py \
    --model "$HF_MODEL" \
    --output "$OUTPUT_DIR" \
    --skip-existing
```

### What it does

1. Loads the full HF model
2. Exports embeddings (LUT4) → `embeddings.mlpackage`
3. Exports LM head (LUT6) → `lm_head.mlpackage`
4. Exports 4 FFN decode chunks (LUT4) → `ffn_LUT4_chunk{0..3}.mlpackage`
5. Exports 4 FFN prefill chunks (LUT4) → `prefill_LUT4_chunk{0..3}.mlpackage`

### Export config (hardcoded in script)

| Constant | Value | Meaning |
|----------|-------|---------|
| `BATCH_SIZE` | 256 | Prefill input length |
| `CTX` | 1024 | KV cache / context length |
| `NUM_CHUNKS` | 4 | Number of FFN chunks |
| `LUT_BITS` | 4 | FFN/embed quantization bits |
| `LM_HEAD_LUT` | 6 | LM head quantization bits |
| `PER_CHANNEL` | 8 | LUT group size |

### Expected output

```
$OUTPUT_DIR/
├── embeddings.mlpackage         (304 MB)
├── lm_head.mlpackage            (462 MB)
├── ffn_LUT4_chunk0.mlpackage    (428 MB)
├── ffn_LUT4_chunk1.mlpackage    (428 MB)
├── ffn_LUT4_chunk2.mlpackage    (428 MB)
├── ffn_LUT4_chunk3.mlpackage    (428 MB)
├── prefill_LUT4_chunk0.mlpackage (437 MB)
├── prefill_LUT4_chunk1.mlpackage (437 MB)
├── prefill_LUT4_chunk2.mlpackage (437 MB)
└── prefill_LUT4_chunk3.mlpackage (437 MB)
```

### Alternative: LM Head with Fused Argmax (16GB machines)

If `qwen35_export.py` OOMs on the LM head step (the 248K vocab is large),
use the lightweight export script that loads only the LM head weight:

```bash
# Step 1a: Export LM head with argmax (lightweight, avoids OOM)
python tests/dev/qwen35_export_lut6_argmax.py \
    --model "$HF_MODEL" \
    --output "$OUTPUT_DIR"
```

This produces a `lm_head.mlpackage` with fused argmax on ANE — outputs
`argmax_idx` (int32) and `argmax_val` (fp16) instead of full logits.

---

## Step 2: Combine (Dedup)

Merge each decode + prefill pair into a single multi-function `.mlpackage`
with shared (deduplicated) weights. This cuts the deploy size by ~40%.

```bash
python tests/dev/qwen35_combine.py \
    --input "$OUTPUT_DIR"
```

### What it does

For each of the 4 chunk pairs:
- Combines `ffn_LUT4_chunk{i}.mlpackage` (decode) + `prefill_LUT4_chunk{i}.mlpackage`
- Into `combined_LUT4_dedup/chunk{i}.mlpackage` with functions `infer` + `prefill`
- Weights are deduplicated (shared between functions)

### Expected output

```
$OUTPUT_DIR/
├── embeddings.mlpackage
├── lm_head.mlpackage
├── combined_LUT4_dedup/
│   ├── chunk0.mlpackage    (437 MB)
│   ├── chunk1.mlpackage    (437 MB)
│   ├── chunk2.mlpackage    (437 MB)
│   └── chunk3.mlpackage    (437 MB)
├── ffn_LUT4_chunk*.mlpackage     (kept, not needed for deploy)
└── prefill_LUT4_chunk*.mlpackage (kept, not needed for deploy)
```

**Deployable set** (6 files, ~2.5 GB):
- `embeddings.mlpackage`
- `lm_head.mlpackage`
- `combined_LUT4_dedup/chunk{0..3}.mlpackage`

### Expected log (clean, no warnings)

```
[dedup] Loading anchor: ffn_LUT4_chunk0.mlpackage
[dedup] Anchor has 143 weight tensors
[dedup] Processing target 1/1: prefill_LUT4_chunk0.mlpackage -> prefill
[dedup]   No replacements needed
[dedup] Done in 26.5s: 0 total ops replaced across 1 non-anchor sources
```

"No replacements needed" means the weights are already byte-identical —
CoreML's `save_multifunction` handles the blob-level dedup automatically.

---

## Step 3: Compile (.mlmodelc)

Compile `.mlpackage` → `.mlmodelc` for faster model loading (2–3× faster
first load, cached subsequently).

```bash
python tests/dev/qwen35_compile.py \
    --model-dir "$OUTPUT_DIR" \
    --separate-only
```

Without `--separate-only`, it also compiles the combined dedup models:
```bash
python tests/dev/qwen35_compile.py \
    --model-dir "$OUTPUT_DIR"
```

### Expected output

Each `.mlpackage` gets a corresponding `.mlmodelc` in the same directory.

> **Note**: Multi-function `.mlmodelc` (from combined dedup) may not support
> `function_name` on ANE. For production, use separate `.mlmodelc` files.

---

## Step 4: Validate (Multi-Round Conversation)

### Option A: Full validation (dedup + separate, 3-turn conversation)

Validates that dedup combined models produce **identical tokens** to separate
models across 3 conversation turns.

```bash
python tests/dev/qwen35_validate.py \
    --model-dir "$OUTPUT_DIR" \
    --tokenizer "$HF_MODEL" \
    --tokens 40
```

Tests 4 configurations:
1. Separate LUT4 fresh (reference)
2. Separate LUT4 incremental
3. Dedup LUT4 fresh
4. Dedup LUT4 incremental

All must produce 100% identical tokens.

Skip separate model testing (faster, dedup-only):
```bash
python tests/dev/qwen35_validate.py \
    --model-dir "$OUTPUT_DIR" \
    --tokenizer "$HF_MODEL" \
    --tokens 40 \
    --skip-separate
```

### Option B: Lightweight E2E test (memory-efficient)

Loads models one-at-a-time (works on 16GB). Tests 3 prompts:

```bash
python tests/dev/qwen35_e2e_lut6.py
```

> **Note**: This script has hardcoded paths. Edit `EXPORT_DIR` and
> `MODEL_PATH` at the top of the file if your paths differ.

### Expected results

- All prompts generate coherent text
- Decode speed: ~9.9 tok/s on ANE
- No ANE errors (error -14, function_name issues)

---

## Step 5: Chat (Multi-Round Conversation)

Launch a browser-based chat interface for interactive testing:

```bash
python tests/dev/qwen35_chat_server.py \
    --model-dir "$OUTPUT_DIR" \
    --tokenizer "$HF_MODEL" \
    --port 8080
```

Then open http://localhost:8080 in your browser.

Features:
- Full multi-round conversation with context history
- Prefill acceleration (batch_size=256)
- Token-per-second display
- Reset conversation button
- KV cache state management

---

## Profiling (Optional)

Run comprehensive ANE profiling:

```bash
python tests/dev/qwen35_profile.py \
    --tokens 40 \
    --skip-cpu-compare \
    --export-dir "$OUTPUT_DIR"
```

> **Note**: `MODEL_PATH` is hardcoded at the top of the script. Edit if needed.

### Expected performance

| Metric | Value |
|--------|-------|
| Decode speed | ~9.9 tok/s |
| ANE ops | 99.7% |
| Embed latency | 0.2 ms |
| FFN latency | 77.9 ms |
| LM head latency | 24.9 ms |
| Total step | ~101 ms |

---

## Quick Reference: Full Pipeline Commands

```bash
# ── Setup ──
source .venv/bin/activate
cd /path/to/Anemll

export HF_MODEL="/path/to/Qwen__Qwen3.5-4B"
export OUTPUT_DIR="/path/to/output"

# ── Step 1: Export (30-60 min) ──
python tests/dev/qwen35_export.py \
    --model "$HF_MODEL" \
    --output "$OUTPUT_DIR" \
    --skip-existing

# ── Step 2: Combine / Dedup (5 min) ──
python tests/dev/qwen35_combine.py \
    --input "$OUTPUT_DIR"

# ── Step 3: Compile (5 min) ──
python tests/dev/qwen35_compile.py \
    --model-dir "$OUTPUT_DIR" \
    --separate-only

# ── Step 4: Validate (5-10 min) ──
python tests/dev/qwen35_validate.py \
    --model-dir "$OUTPUT_DIR" \
    --tokenizer "$HF_MODEL" \
    --tokens 40

# ── Step 5: Chat ──
python tests/dev/qwen35_chat_server.py \
    --model-dir "$OUTPUT_DIR" \
    --tokenizer "$HF_MODEL" \
    --port 8080
# Open http://localhost:8080
```

---

## Troubleshooting

### OOM during export

The LM head export can OOM on 16GB machines (248K vocab × 2560 hidden):
```bash
# Use the lightweight LM head export instead:
python tests/dev/qwen35_export_lut6_argmax.py \
    --model "$HF_MODEL" \
    --output "$OUTPUT_DIR"
```

### CoreML e5rt cache filling disk

CoreML caches compiled models at `~/Library/Caches/org.python.python/com.apple.e5rt.e5bundlecache/`.
This can grow to 50+ GB. Safe to clean:
```bash
rm -rf ~/Library/Caches/org.python.python/com.apple.e5rt.e5bundlecache/
```

### ANE error -14

This means a model op isn't ANE-compatible. The current pipeline is validated
to run 99.7% of ops on ANE. If you see error -14:
- Verify you're using the correct model files (not modified)
- Check that `--separate-only` was used for compile (multi-function .mlmodelc
  can fail on ANE)

### Dedup combine warning: "Weight count mismatch"

If you see `"Weight count mismatch: anchor has 888, target has 42640"`, your
`anemll/utils/dedup_weights.py` needs the `min_size` filter fix. The preflight
should show ~143 tensors for both decode and prefill. The warning is cosmetic —
CoreML still deduplicates at the blob level — but the fix is already in the
current codebase.

### Slow first model load

First load from `.mlpackage` triggers JIT compilation. Use Step 3 (compile to
`.mlmodelc`) for 2–3× faster loading. Subsequent loads use the cached
compilation.

---

## File Reference

| Script | Purpose |
|--------|---------|
| `tests/dev/qwen35_export.py` | Export all 10 .mlpackage models |
| `tests/dev/qwen35_export_lut6_argmax.py` | Lightweight LM head export (16GB-safe) |
| `tests/dev/qwen35_quantize_lmhead.py` | Standalone LUT6 quantization step |
| `tests/dev/qwen35_combine.py` | Dedup combine decode+prefill |
| `tests/dev/qwen35_compile.py` | Compile .mlpackage → .mlmodelc |
| `tests/dev/qwen35_validate.py` | Multi-round conversation validation |
| `tests/dev/qwen35_e2e_lut6.py` | Lightweight E2E generation test |
| `tests/dev/qwen35_chat_server.py` | Browser-based chat UI |
| `tests/dev/qwen35_profile.py` | ANE profiling and benchmarks |
