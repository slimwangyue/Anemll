"""Qwen3.5-4B — Exploratory configurations.

Each configuration is self-contained and does NOT modify the stable baseline.
Models export to separate output folders under OUTPUT_ROOT.

Usage:
    from scripts_qwen3_5.explore.explore_config import CONFIGS, get_config
    cfg = get_config("batch512_ctx1024")
"""
import os

# ── Paths ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

HF_MODEL = os.environ.get(
    "QWEN35_HF_MODEL",
    os.path.join(_REPO_ROOT, "models", "Qwen__Qwen3.5-4B"),
)

# Separate root for exploratory models — never touches qwen3_5_stable_models/
OUTPUT_ROOT = os.environ.get(
    "QWEN35_EXPLORE_OUTPUT",
    os.path.join(_REPO_ROOT, "qwen3_5_explore"),
)

# ── Stable baseline (for reference only, never modified) ──
STABLE = {
    "name": "stable_batch256_ctx1024",
    "BATCH_SIZE": 256,
    "CTX": 1024,
    "NUM_CHUNKS": 4,
    "LUT_BITS": 6,
    "LM_HEAD_LUT": 6,
    "PER_CHANNEL": 8,
    "FFN_PER_CHANNEL": 4,
    "output_dir": os.path.join(_REPO_ROOT, "qwen3_5_stable_models"),
}

# ── Exploratory configurations ──
CONFIGS = {
    # Experiment 1: Larger prefill batch (512 tokens), same cache
    "batch512_ctx1024": {
        "name": "batch512_ctx1024",
        "BATCH_SIZE": 512,
        "CTX": 1024,
        "NUM_CHUNKS": 4,
        "LUT_BITS": 4,
        "LM_HEAD_LUT": 6,
        "PER_CHANNEL": 8,
        "output_dir": os.path.join(OUTPUT_ROOT, "batch512_ctx1024"),
        "notes": "Larger prefill input (512 vs 256). Same KV cache size.",
    },
    # Experiment 2: Larger cache (2048), same prefill batch
    "batch256_ctx2048": {
        "name": "batch256_ctx2048",
        "BATCH_SIZE": 256,
        "CTX": 2048,
        "NUM_CHUNKS": 4,
        "LUT_BITS": 4,
        "LM_HEAD_LUT": 6,
        "PER_CHANNEL": 8,
        "output_dir": os.path.join(OUTPUT_ROOT, "batch256_ctx2048"),
        "notes": "2x KV cache (2048). Doubles memory for KV states.",
    },
    # Experiment 3: Larger cache (4096), same prefill batch
    "batch256_ctx4096": {
        "name": "batch256_ctx4096",
        "BATCH_SIZE": 256,
        "CTX": 4096,
        "NUM_CHUNKS": 4,
        "LUT_BITS": 4,
        "LM_HEAD_LUT": 6,
        "PER_CHANNEL": 8,
        "output_dir": os.path.join(OUTPUT_ROOT, "batch256_ctx4096"),
        "notes": "4x KV cache (4096). Significant memory increase.",
    },
    # Experiment 4: Larger batch + cache together
    "batch512_ctx2048": {
        "name": "batch512_ctx2048",
        "BATCH_SIZE": 512,
        "CTX": 2048,
        "NUM_CHUNKS": 4,
        "LUT_BITS": 4,
        "LM_HEAD_LUT": 6,
        "PER_CHANNEL": 8,
        "output_dir": os.path.join(OUTPUT_ROOT, "batch512_ctx2048"),
        "notes": "Combined: larger prefill + 2x cache.",
    },
}


def get_config(name):
    """Get an exploration config by name. Returns a dict."""
    if name == "stable":
        return STABLE.copy()
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name].copy()


def list_configs():
    """Print all available configurations."""
    print(f"{'Name':<25} {'Batch':>6} {'CTX':>6} {'Notes'}")
    print("-" * 70)
    print(f"{'stable (baseline)':<25} {STABLE['BATCH_SIZE']:>6} {STABLE['CTX']:>6} Current production config")
    for name, cfg in CONFIGS.items():
        print(f"{name:<25} {cfg['BATCH_SIZE']:>6} {cfg['CTX']:>6} {cfg.get('notes', '')}")
