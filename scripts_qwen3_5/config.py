"""Qwen3.5 — Shared pipeline configuration.

Supports 4B (default) and 2B via QWEN35_MODEL_SIZE env var.
"""
import os

# ── Model-size preset (set QWEN35_MODEL_SIZE=2B for Qwen3.5-2B) ──
_MODEL_SIZE = os.environ.get("QWEN35_MODEL_SIZE", "4B").upper()

BATCH_SIZE = 256   # prefill input length
CTX = 4096         # KV cache / context length
LUT_BITS = 4       # FFN quantization (LUT4)
LM_HEAD_LUT = 6    # LM head quantization (LUT6)
PER_CHANNEL = 8    # per-channel group size for embeddings & lm_head
FFN_PER_CHANNEL = 4  # per-channel group size for FFN chunks (gs=4)

# ── Per-model chunking ──
# [FLLL] partition — F layers START each chunk (except chunk 0).
# This avoids L→F transitions within a single CoreML model, which cause
# catastrophic FP16 MIL error amplification (cos 0.333 → 0.994).

if _MODEL_SIZE == "2B":
    # Qwen3.5-2B: 24 layers, full_attention_interval=4
    # layer_types: [L L L F] × 6  →  F at {3,7,11,15,19,23}
    # Pattern: [LLL, FLLL, FLLL, FLLL, FLLL, FLLL, F]
    NUM_CHUNKS = 7
    CHUNK_RANGES = [
        (0, 3),    # chunk 0: layers 0-2   (LLL)
        (3, 7),    # chunk 1: layers 3-6   (FLLL)
        (7, 11),   # chunk 2: layers 7-10  (FLLL)
        (11, 15),  # chunk 3: layers 11-14 (FLLL)
        (15, 19),  # chunk 4: layers 15-18 (FLLL)
        (19, 23),  # chunk 5: layers 19-22 (FLLL)
        (23, 24),  # chunk 6: layer  23    (F)
    ]
    _DEFAULT_MODEL_NAME = "Qwen__Qwen3.5-2B"
    _DEFAULT_OUTPUT_NAME = "qwen3_5_2b_stable_test"
else:
    # Qwen3.5-4B: 32 layers, full_attention_interval=4
    # layer_types: [L L L F] × 8  →  F at {3,7,11,15,19,23,27,31}
    # Pattern: [LLL, FLLL, FLLL, FLLL, FLLL, FLLL, FLLL, FLLL, F]
    NUM_CHUNKS = 9
    CHUNK_RANGES = [
        (0, 3),    # chunk 0: layers 0-2   (LLL)
        (3, 7),    # chunk 1: layers 3-6   (FLLL)
        (7, 11),   # chunk 2: layers 7-10  (FLLL)
        (11, 15),  # chunk 3: layers 11-14 (FLLL)
        (15, 19),  # chunk 4: layers 15-18 (FLLL)
        (19, 23),  # chunk 5: layers 19-22 (FLLL)
        (23, 27),  # chunk 6: layers 23-26 (FLLL)
        (27, 31),  # chunk 7: layers 27-30 (FLLL)
        (31, 32),  # chunk 8: layer  31    (F)
    ]
    _DEFAULT_MODEL_NAME = "Qwen__Qwen3.5-4B"
    _DEFAULT_OUTPUT_NAME = "qwen3_5_stable_lut4ffn_lut6em_test"

# Derived label used for file naming: ffn_LUT6_chunk{i}, combined_LUT6_dedup/
FFN_LABEL = f"LUT{LUT_BITS}"

# Repo root (parent of scripts_qwen3_5/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default paths (override via CLI args or env vars)
DEFAULT_HF_MODEL = os.environ.get(
    "QWEN35_HF_MODEL",
    os.path.join(REPO_ROOT, "models", _DEFAULT_MODEL_NAME),
)
DEFAULT_OUTPUT = os.environ.get(
    "QWEN35_OUTPUT",
    os.path.join(REPO_ROOT, _DEFAULT_OUTPUT_NAME),
)
