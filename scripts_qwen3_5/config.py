"""Qwen3.5-4B Milestone 1.2 — Shared configuration."""

BATCH_SIZE = 256   # prefill input length
CTX = 1024         # KV cache / context length
NUM_CHUNKS = 4     # FFN layer chunks
LUT_BITS = 4       # FFN quantization
LM_HEAD_LUT = 6    # LM head quantization (LUT6)
PER_CHANNEL = 8    # per-channel group size

# Default paths (override via CLI args)
DEFAULT_HF_MODEL = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
DEFAULT_OUTPUT = "/Users/yw68/qwen35_milestone1_2"
