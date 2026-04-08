"""Qwen3.5-4B — Shared pipeline configuration (Milestone 2.1)."""
import os

BATCH_SIZE = 512   # prefill input length
CTX = 2048         # KV cache / context length
NUM_CHUNKS = 9  # FFN layer chunks ([FLLL] 9-chunk partition)
LUT_BITS = 6       # FFN quantization (LUT6)
LM_HEAD_LUT = 6    # LM head quantization (LUT6)
PER_CHANNEL = 8    # per-channel group size for embeddings & lm_head
FFN_PER_CHANNEL = 4  # per-channel group size for FFN chunks (gs=4, Milestone 2.1)

# [FLLL] 9-chunk partition — F layers START each chunk (except chunk 0).
# This avoids L→F transitions within a single CoreML model, which cause
# catastrophic FP16 MIL error amplification (cos 0.333 → 0.994).
# Pattern: [LLL, FLLL, FLLL, FLLL, FLLL, FLLL, FLLL, FLLL, F]
# Boundaries after layers: [2, 6, 10, 14, 18, 22, 26, 30]
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

# Derived label used for file naming: ffn_LUT6_chunk{i}, combined_LUT6_dedup/
FFN_LABEL = f"LUT{LUT_BITS}"

# Repo root (parent of scripts_qwen3_5/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default paths (override via CLI args or env vars)
DEFAULT_HF_MODEL = os.environ.get(
    "QWEN35_HF_MODEL",
    os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B"),
)
DEFAULT_OUTPUT = os.environ.get(
    "QWEN35_OUTPUT",
    os.path.join(REPO_ROOT, "qwen3_5_stable_models_testing"),
)
