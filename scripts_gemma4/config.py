"""Gemma4 E4B pipeline configuration.

Central config imported by all pipeline scripts.
"""
import os

# ─── Architecture ────────────────────────────────────────────────────────────
NUM_LAYERS = 42          # Gemma4 E4B total layers
HIDDEN_SIZE = 2560       # hidden_size
HEAD_DIM_LOCAL = 256     # head_dim for sliding-window layers
HEAD_DIM_GLOBAL = 512    # global_head_dim for full-attention layers
NUM_KV_HEADS = 2         # num_key_value_heads
VOCAB_SIZE = 262144      # vocabulary size
SLIDING_WINDOW = 512     # sliding_window size

# ─── Pipeline settings ───────────────────────────────────────────────────────
BATCH_SIZE = 64          # prefill batch size
CTX = 512                # context / state length
NUM_CHUNKS = 14          # FFN chunk count

# ANE compiler (ANEF) fails when a chunk mixes local and global attention layers.
# Each chunk type works alone (local-only or global-only), but mixing them
# causes the ANE compiler to reject the model with error -14.
# Solution: isolate global layers (5, 11, 17, 23) into their own 1-layer chunks,
# group local/shared layers in chunks up to 5 layers.
CHUNK_RANGES = [
    # Source local layers
    (0, 5),    # chunk 0:  layers 0-4   (5 local)
    (5, 6),    # chunk 1:  layer  5     (1 global)
    (6, 11),   # chunk 2:  layers 6-10  (5 local)
    (11, 12),  # chunk 3:  layer  11    (1 global)
    (12, 17),  # chunk 4:  layers 12-16 (5 local)
    (17, 18),  # chunk 5:  layer  17    (1 global)
    (18, 23),  # chunk 6:  layers 18-22 (5 local)
    (23, 24),  # chunk 7:  layer  23    (1 global)
    # Shared KV layers (24-41) — no KV writes, grouped by 3
    (24, 27),  # chunk 8:  layers 24-26 (3 shared)
    (27, 30),  # chunk 9:  layers 27-29 (3 shared)
    (30, 33),  # chunk 10: layers 30-32 (3 shared)
    (33, 36),  # chunk 11: layers 33-35 (3 shared)
    (36, 39),  # chunk 12: layers 36-38 (3 shared)
    (39, 42),  # chunk 13: layers 39-41 (3 shared)
]

# ─── Quantization ────────────────────────────────────────────────────────────
LUT_BITS = 4             # FFN quantization bits
LM_HEAD_LUT = 6          # LM head quantization bits
PER_CHANNEL = 8          # per-channel group size for embeddings & lm_head
FFN_PER_CHANNEL = 4      # per-channel group size for FFN chunks
FFN_LABEL = f"LUT{LUT_BITS}"

# ─── Paths ───────────────────────────────────────────────────────────────────
DEFAULT_HF_MODEL = os.environ.get(
    "GEMMA4_HF_MODEL",
    os.path.expanduser("~/local_llm/models/google__gemma-4-E4B-it"),
)
DEFAULT_OUTPUT = os.environ.get(
    "GEMMA4_OUTPUT",
    "gemma4_E4B_lut4ffn_lut6em",
)

# ─── Imports (all scripts share these) ───────────────────────────────────────
from anemll.models.gemma4_model import Gemma4ForCausalLM, Gemma4Config, MODEL_DTYPE, TEST_DEVICE
from anemll.ane_converter.gemma4_converter import Gemma4Converter
