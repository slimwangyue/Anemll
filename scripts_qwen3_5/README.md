# Qwen3.5-4B ANE Pipeline — Milestone 2.0

Self-contained pipeline for exporting, combining, compiling, validating,
and chatting with Qwen3.5-4B on Apple Neural Engine (ANE).

**Milestone 2.0** adds true batched prefill (256-token blocks with `valid_len`
gating), dual-instance model loading (separate infer/prefill MLModel objects
sharing KV-cache state), and a validated crossover threshold of 32 tokens.

**Scripts directory**: `scripts_qwen3_5/`  
**Stable model output**: `qwen3_5_stable_models/`  
**Total deploy size**: ~2.5 GB (after dedup)

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Embeddings | LUT4 | 4-bit quantized |
| LM Head | **LUT6** | 6-bit quantized + fused argmax |
| FFN | LUT4 × 4 chunks | 4-bit quantized, 4-way chunked |
| Batch size | 256 | Prefill input length |
| Context | 1024 | KV cache / context length |
| per_channel | 8 | LUT group size |

## Prerequisites

```bash
# macOS with Apple Silicon, ≥16 GB RAM
# Xcode Command Line Tools
xcrun --find coremlcompiler

# Python venv (from repo root)
cd /path/to/Anemll
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download HuggingFace model (if needed)
# huggingface-cli download Qwen/Qwen3.5-4B --local-dir /path/to/Qwen3.5-4B
```

## Quick Start — Full Pipeline

```bash
source .venv/bin/activate
cd /path/to/Anemll

# One-shot pipeline (export → combine → compile → validate)
./scripts_qwen3_5/run_pipeline.sh --model /path/to/Qwen3.5-4B

# Output goes to qwen3_5_stable_models/ by default
# Override with: --output /path/to/output
```

## Step-by-Step Commands

```bash
source .venv/bin/activate
cd /path/to/Anemll

export HF_MODEL="/path/to/Qwen3.5-4B"
export OUTPUT="qwen3_5_stable_models"

# Step 1: Export (30-60 min)
python scripts_qwen3_5/export.py --model "$HF_MODEL" --output "$OUTPUT" --skip-existing

# Step 2: Combine / Dedup (5 min)
python scripts_qwen3_5/combine.py --input "$OUTPUT"

# Step 3: Compile (5 min)
python scripts_qwen3_5/compile.py --model-dir "$OUTPUT"

# Step 4: Validate — multi-round conversation (5-10 min)
python scripts_qwen3_5/validate.py --model-dir "$OUTPUT" --tokens 40

# Step 5: Chat server
python scripts_qwen3_5/chat_server.py --model-dir "$OUTPUT"
# Open http://localhost:8080
```

## Test Commands

```bash
# Full multi-round validation (3-turn conversation, 100% token match)
python scripts_qwen3_5/validate.py --model-dir "$OUTPUT"

# Lightweight E2E smoke test (memory-efficient, 3 prompts)
python scripts_qwen3_5/test_e2e.py --model-dir "$OUTPUT"

# ANE profiling / benchmarks
python scripts_qwen3_5/profile.py --export-dir "$OUTPUT" --skip-cpu-compare
```

## Models Produced

### Combined Dedup — Recommended (6 files, ~2.5 GB)
- `embeddings.mlpackage` (304 MB)
- `lm_head.mlpackage` (462 MB, LUT6 + argmax)
- `combined_LUT4_dedup/chunk{0..3}.mlpackage` — Each contains:
  - `infer` function (decode, single-token)
  - `prefill` function (batch prefill, 256 tokens with `valid_len`)
- Tokenizer: `tokenizer.json`, `tokenizer_config.json`, `vocab.json`

### Separate Models (10 files, ~4.2 GB)
- `embeddings.mlpackage` — Token embeddings (LUT4, 304 MB)
- `lm_head.mlpackage` — Language model head (LUT6 + argmax, 462 MB)
- `ffn_LUT4_chunk{0..3}.mlpackage` — Decode (4 × 428 MB)
- `prefill_LUT4_chunk{0..3}.mlpackage` — Prefill (4 × 437 MB)

## File Reference

| Script | Purpose |
|--------|---------|
| `config.py` | Shared constants (batch size, ctx, LUT bits, paths) |
| `export.py` | Step 1: Export HF model → 10 .mlpackage files |
| `combine.py` | Step 2: Dedup combine decode+prefill chunks |
| `compile.py` | Step 3: Compile .mlpackage → .mlmodelc |
| `validate.py` | Step 4: Multi-round conversation validation |
| `test_e2e.py` | Lightweight E2E generation smoke test |
| `validate_pipeline.py` | Full validation + benchmark suite (5 tests) |
| `chat_server.py` | Browser-based multi-round chat UI |
| `profile.py` | ANE profiling and benchmarks |
| `run_pipeline.sh` | One-shot pipeline runner |
| `README.md` | This file |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN35_HF_MODEL` | `<repo>/models/Qwen__Qwen3.5-4B` | HuggingFace model path |
| `QWEN35_OUTPUT` | `<repo>/qwen3_5_stable_models` | Output directory |

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Decode speed | ~13 tok/s | Single-token generation |
| **Batch prefill** | **96 tok/s** | **256-token blocks (7.4x vs sequential)** |
| Prefill crossover | 32 tokens | Batch beats sequential above this |
| ANE ops | 99.7% | |
| Embed latency | 0.2 ms | |
| FFN latency | 77.9 ms | Per-chunk decode step |
| LM head latency | 24.9 ms | |
| Total decode step | ~101 ms | |
| Deploy size | 2.5 GB | Combined dedup (6 models) |

### Crossover Benchmark (Batch vs Sequential Prefill)

| Tokens | Batch (ms) | Sequential (ms) | Speedup |
|--------|-----------|-----------------|---------|
| 16 | 1,517 | 1,287 | 0.85x |
| **32** | **1,608** | **2,866** | **1.78x** |
| 64 | 1,419 | 4,987 | 3.52x |
| 128 | 1,738 | 10,085 | 5.80x |
| 256 | 1,685 | 19,801 | 11.75x |

## Dependencies

All scripts use only:
- `anemll/` — the ANEMLL package (from repo root)
- `scripts_qwen3_5/config.py` — shared pipeline configuration

No external dependencies beyond `requirements.txt`.
