# Qwen3.5-4B Pipeline — Milestone 1.1

Self-contained pipeline for exporting, combining, compiling, and running
Qwen3.5-4B on Apple Neural Engine (ANE).

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Embeddings | LUT4 | 4-bit quantized |
| LM Head | **LUT6** | 6-bit quantized (62% size reduction, 98.9% top-1 match) |
| FFN | LUT4 × 4 chunks | 4-bit quantized, 4-way chunked |
| Batch size | 256 | Prefill input length |
| Context | 1024 | KV cache / context length |
| Prefill blocks | 4 | Full-prompt batch prefill (4 × 256 = 1024) |

## Quick Start

```bash
# Full pipeline (from repo root)
./scripts_qwen3_5/run_pipeline.sh --model /path/to/Qwen3.5-4B --output /path/to/output

# Or step by step
python scripts_qwen3_5/export.py --model /path/to/Qwen3.5-4B --output /path/to/output
python scripts_qwen3_5/combine.py --input /path/to/output
python scripts_qwen3_5/compile.py --model-dir /path/to/output
python tests/dev/qwen35_chat_server.py --model-dir /path/to/output --hf-model /path/to/Qwen3.5-4B
```

## Models Produced

### Separate Models (24 total)
- `embeddings.mlpackage` — Token embeddings (LUT4)
- `lm_head.mlpackage` — Language model head (LUT6)
- `ffn_LUT4_chunk{0..3}.mlpackage` — Decode (single-token inference)
- `prefill_LUT4_chunk{0..3}_block{0..3}.mlpackage` — Multi-block prefill

### Combined Dedup Models
- `combined_LUT4_dedup/chunk{0..3}.mlpackage` — Each contains:
  - `infer` function (decode)
  - `prefill_0`, `prefill_256`, `prefill_512`, `prefill_768` functions
  - Weights shared via ANEMLL-Dedup

## Multi-Block Prefill

Each prefill block model writes KV cache entries at a fixed position range,
determined at trace time for ANE compatibility (static slice bounds):

| Block | Write Range | Function Name |
|-------|-------------|---------------|
| 0 | positions 0–255 | `prefill_0` |
| 1 | positions 256–511 | `prefill_256` |
| 2 | positions 512–767 | `prefill_512` |
| 3 | positions 768–1023 | `prefill_768` |

For a 700-token prompt:
1. Block 0: tokens 0–255 via batch prefill (~109 tok/s)
2. Block 1: tokens 256–511 via batch prefill (~109 tok/s)
3. Tail: tokens 512–699 via sequential decode (~11 tok/s)

## Dependencies

Requires the `anemll` package from the repo root:
```bash
cd /path/to/Anemll
pip install -e .
# or: PYTHONPATH=. python scripts_qwen3_5/export.py ...
```
