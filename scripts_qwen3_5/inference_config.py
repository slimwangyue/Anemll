"""
Unified inference configuration for Qwen3.5 models on ANE.

Sampling presets per model size (4B / 2B) and mode (think-on / think-off).
Used by chat_server.py and run_validation.py.
"""

# ── Model definitions ────────────────────────────────────────────────
MODELS = {
    "4B": {
        "description": "Qwen3.5-4B",
        "num_layers": 32,
        "default_chunks": 9,
        "default_ctx": 4096,
        "default_input_length": 256,
    },
    "2B": {
        "description": "Qwen3.5-2B",
        "num_layers": 28,
        "default_chunks": 7,
        "default_ctx": 2048,
        "default_input_length": 256,
    },
}

# ── Sampling presets ─────────────────────────────────────────────────
# Keys: (model_size, enable_thinking) → dict of sampling parameters
#
# Optimized via systematic sweep (sweep_sampling.py) 2026-04-15.
# Primary goal: minimum repetition while preserving coherence.
# Validated on 10 single-turn + 4 multi-turn prompts (EN, ZH, code, reasoning, long-form).
SAMPLING_PRESETS = {
    ("4B", True): {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.08,
    },
    ("4B", False): {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.0,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.05,
    },
    ("2B", True): {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.08,
    },
    ("2B", False): {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.0,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.05,
    },
}

# ── Server defaults ──────────────────────────────────────────────────
DEFAULT_PORT = 8080
DEFAULT_COMPUTE_UNIT = "all"
DEFAULT_MAX_TOKENS = 2048


def get_sampling_config(model_size="4B", enable_thinking=True):
    """Return sampling preset dict for given model size and thinking mode.

    Falls back to 4B think-on if (model_size, enable_thinking) is unknown.
    """
    key = (model_size.upper(), bool(enable_thinking))
    return dict(SAMPLING_PRESETS.get(key, SAMPLING_PRESETS[("4B", True)]))


def get_model_info(model_size="4B"):
    """Return model metadata dict for given size. Falls back to 4B."""
    return dict(MODELS.get(model_size.upper(), MODELS["4B"]))
