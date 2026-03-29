"""Qwen3.5-4B — Shared pipeline configuration (Milestone 2.1)."""
import os

BATCH_SIZE = 512   # prefill input length
CTX = 4096         # KV cache / context length
NUM_CHUNKS = int(os.environ.get("QWEN35_NUM_CHUNKS", 4))  # FFN layer chunks
LUT_BITS = 6       # FFN quantization (LUT6)
LM_HEAD_LUT = 6    # LM head quantization (LUT6)
PER_CHANNEL = 8    # per-channel group size for embeddings & lm_head
FFN_PER_CHANNEL = 4  # per-channel group size for FFN chunks (gs=4, Milestone 2.1)

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
