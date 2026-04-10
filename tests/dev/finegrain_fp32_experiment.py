#!/usr/bin/env python3
"""Fine-grained selective FP32 experiment for Qwen3.5-4B on ANE.

Instead of global op-type sweeps, this experiment applies FP32 to specific
**structural precision islands**: per-layer recurrence subgraphs, per-layer
full-attention paths, and specific sensitive transitions.

Uses coremltools' `FP16ComputePrecision(op_selector=fn)` to select individual
ops by name pattern within the MIL graph, keeping FP32 only for targeted
structural regions while leaving the rest in FP16.

Candidates:
  C1: Recurrence state update (all L layers) — the exp/mul/sub/add loop
  C2: Softplus gate path (all L layers) — relu/abs/exp/log that compute g
  C3: Full recurrence + softplus (all L layers) — C1 ∪ C2
  C4: Full-attention QKV path (F layer only) — matmul/softmax/sigmoid
  C5: C3 + C4 combined  
  C6: Last L layer recurrence only (single-layer test)
  C7: L2-norm + recurrence core (tightest targeting)

Usage:
    python tests/dev/finegrain_fp32_experiment.py \\
        --model models/Qwen__Qwen3.5-4B \\
        --baseline-dir qwen3_5_stable_lut4ffn_lut6em_fp16 \\
        --fp32-dir qwen3_5_stable_lut4ffn_lut6em_fp32 \\
        --output tests/dev/finegrain_fp32_results \\
        --max-tokens 40
"""
import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import coremltools as ct
from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision
from transformers import AutoTokenizer

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES, FFN_PER_CHANNEL
from anemll.models.qwen3_5_model import (
    Qwen35Config, Qwen35ForCausalLM, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Architecture map: layer types per chunk
# [LLL] + 7×[FLLL] + [F]
# ─────────────────────────────────────────────────────────────────────────────
LAYER_TYPES = {}  # layer_idx -> 'F' or 'L'
for ci in range(NUM_CHUNKS):
    sl, el = CHUNK_RANGES[ci]
    for li in range(sl, el):
        # Chunk 0: all L; Chunk 8: all F; Chunks 1-7: first=F, rest=L
        if ci == 0:
            LAYER_TYPES[li] = 'L'
        elif ci == 8:
            LAYER_TYPES[li] = 'F'
        else:
            LAYER_TYPES[li] = 'F' if li == sl else 'L'

PROMPTS = [
    "What is a stack in computer science?",
    "教我做红烧鱼",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Precision island definitions
# Each returns an op_selector function: True = cast to FP16, False = keep FP32
# ─────────────────────────────────────────────────────────────────────────────

# MIL op naming patterns discovered from graph dump of chunk 5 (layers 19-22):
#
# Recurrence state update ops (per L layer):
#   state_* (mul)     — state *= exp(g)
#   kv_mem* (reduce_sum) — memory lookup
#   op_XXXX (sub)     — delta = v - kv_mem
#   delta* (mul)      — delta *= beta
#   state_workaround / state_N (add) — state += k⊗delta
#   op_XXXX (exp)     — exp(g) for state decay (the one with shape (1,32))
#   g_t* (expand_dims) — reshape of exp(g)
#   beta_t* (expand_dims) — beta gating
#   query_N (mul)     — L2-normed query * state
#   key / key_N (mul) — L2-normed key
#   sq_sum* (reduce_sum) — L2 norm computation
#   op_XXXX (rsqrt)   — 1/sqrt for normalization
#
# Softplus gate ops (per L layer):
#   op_XXXX (relu)    — softplus: relu(x)
#   op_XXXX (abs)     — softplus: |x|
#   op_XXXX (exp) (1,1,32) — softplus: exp(-|x|) 
#   op_XXXX (log)     — softplus: log(1+exp(-|x|))
#   sp / sp_N (add)   — softplus sum
#   x_N (sigmoid)     — A_log gate sigmoid (for L layers, shape (1,32,1))
#   x_N (mul)         — final gate multiplication
#
# Full-attention ops (per F layer):
#   op_751 (matmul)   — Q @ K^T 
#   attn_weights* (mul/add/softmax) — attention scoring
#   attn_output* (matmul/reshape/mul) — attn @ V + gate
#   op_783 (sigmoid)  — attention output gate
#
# Layer boundary indicators:
#   hidden_states_N (add) — residual connection (end of attention block)
#   normed_N (layer_norm) — RMSNorm


def _make_recurrence_selector(chunk_idx: int, layer_filter: str = "all_L"):
    """Create op_selector that keeps recurrence state update ops in FP32.
    
    The recurrence update in each L layer consists of:
      exp(g) → state*g → kv_mem(reduce_sum) → sub(v-kv_mem) → delta*beta →
      state+k⊗delta → q*state(reduce_sum)
    
    These are identified by name patterns in the MIL graph.
    
    layer_filter:
      "all_L" — all linear attention layers in the chunk
      "last_L" — only the last linear attention layer
    """
    sl, el = CHUNK_RANGES[chunk_idx]
    local_layers = list(range(sl, el))
    l_layers = [li for li in local_layers if LAYER_TYPES.get(li) == 'L']
    
    if layer_filter == "last_L" and l_layers:
        target_layers = [l_layers[-1]]
    else:
        target_layers = l_layers
    
    # Recurrence op name patterns (discovered from MIL dump):
    # These names are consistent across layers within a chunk
    recurrence_name_patterns = [
        r'^state_\d+$', r'^state_workaround$', r'^state$',   # state mul/add
        r'^g_t', r'^beta_t',                                   # gate/beta expand_dims
        r'^kv_mem',                                             # reduce_sum (memory lookup)
        r'^delta',                                              # mul (delta)
        r'^query_\d+$', r'^query$',                            # L2-normed query mul  
        r'^key_\d*$', r'^key$',                                # L2-normed key mul
        r'^sq_sum',                                             # reduce_sum (L2 norm)
    ]
    # Also match by type+shape for ops with generic names (op_XXXX):
    recurrence_type_shapes = {
        'exp': [(1, 32)],           # exp(g) for state decay
        'sub': [(1, 32, 128)],      # v - kv_mem
        'rsqrt': [(1, 32, 1, 1)],   # 1/sqrt for L2 norm
    }
    
    compiled_patterns = [re.compile(p) for p in recurrence_name_patterns]
    
    def op_selector(op):
        """Returns True to cast to FP16, False to keep FP32."""
        # Always cast const ops (weights stay in their native precision)
        if op.op_type == 'const':
            return True
        
        # Check name patterns
        for pat in compiled_patterns:
            if pat.match(op.name):
                return False  # Keep FP32
        
        # Check type+shape patterns
        if op.op_type in recurrence_type_shapes:
            for out in op.outputs:
                try:
                    shape = tuple(out.shape)
                    if shape in recurrence_type_shapes[op.op_type]:
                        return False  # Keep FP32
                except:
                    pass
        
        return True  # Cast to FP16
    
    return op_selector


def _make_softplus_selector(chunk_idx: int):
    """Keep softplus gate computation in FP32.
    
    Softplus in each L layer:
      relu(x) + log(1 + exp(-|x|))
    Plus the sigmoid gate and final multiplication.
    """
    softplus_name_patterns = [
        r'^sp$', r'^sp_\d+$',      # softplus add result
        r'^x_\d+$',                 # sigmoid gate + final mul
    ]
    softplus_types = {'relu', 'abs', 'log'}
    # exp with shape (1,1,32) is softplus exp, not recurrence exp
    
    compiled_patterns = [re.compile(p) for p in softplus_name_patterns]
    
    def op_selector(op):
        if op.op_type == 'const':
            return True
        # Named softplus ops
        for pat in compiled_patterns:
            if pat.match(op.name):
                return False
        # Type-based softplus ops
        if op.op_type in softplus_types:
            return False
        # exp with shape (1,1,32) = softplus exp
        if op.op_type == 'exp':
            for out in op.outputs:
                try:
                    if tuple(out.shape) == (1, 1, 32):
                        return False
                except:
                    pass
        return True
    
    return op_selector


def _make_full_recurrence_selector(chunk_idx: int, layer_filter: str = "all_L"):
    """Keep both recurrence update AND softplus gate in FP32."""
    rec_sel = _make_recurrence_selector(chunk_idx, layer_filter)
    sp_sel = _make_softplus_selector(chunk_idx)
    
    def op_selector(op):
        # If either says "keep FP32", keep FP32
        rec = rec_sel(op)
        sp = sp_sel(op)
        if not rec or not sp:
            return False  # Keep FP32
        return True  # Cast to FP16
    
    return op_selector


def _make_full_attn_selector(chunk_idx: int):
    """Keep full-attention QKV scoring path in FP32.
    
    matmul (Q@K^T) → mul (scale) → add (mask) → softmax → matmul (attn@V) →
    sigmoid (gate) → mul (gated output)
    """
    attn_name_patterns = [
        r'^attn_weights',   # mul, add, softmax
        r'^attn_output',    # matmul, reshape, mul
    ]
    compiled_patterns = [re.compile(p) for p in attn_name_patterns]
    
    def op_selector(op):
        if op.op_type == 'const':
            return True
        for pat in compiled_patterns:
            if pat.match(op.name):
                return False
        # matmul in full-attention (shape (1,16,1,2048) for Q@K^T)
        if op.op_type == 'matmul':
            return False
        # softmax is always full-attention
        if op.op_type == 'softmax':
            return False
        # sigmoid with shape (1,1,4096) is full-attention gate
        if op.op_type == 'sigmoid':
            for out in op.outputs:
                try:
                    if tuple(out.shape) == (1, 1, 4096):
                        return False
                except:
                    pass
        return True

    return op_selector


def _make_combined_selector(chunk_idx: int):
    """C3 + C4: full recurrence + softplus + full-attention scoring."""
    rec_sel = _make_full_recurrence_selector(chunk_idx)
    attn_sel = _make_full_attn_selector(chunk_idx)
    
    def op_selector(op):
        rec = rec_sel(op)
        attn = attn_sel(op)
        if not rec or not attn:
            return False
        return True
    
    return op_selector


def _make_recurrence_core_selector(chunk_idx: int):
    """C7: Tightest targeting — just the core state evolution.
    
    Only: exp(g), state*g, kv_mem, sub, delta, state_add
    Excludes L2-norm, softplus, convolutions, etc.
    """
    core_name_patterns = [
        r'^state_\d+$', r'^state_workaround$', r'^state$',
        r'^g_t',
        r'^beta_t',
        r'^kv_mem',
        r'^delta',
    ]
    core_type_shapes = {
        'exp': [(1, 32)],
        'sub': [(1, 32, 128)],
    }
    compiled_patterns = [re.compile(p) for p in core_name_patterns]
    
    def op_selector(op):
        if op.op_type == 'const':
            return True
        for pat in compiled_patterns:
            if pat.match(op.name):
                return False
        if op.op_type in core_type_shapes:
            for out in op.outputs:
                try:
                    if tuple(out.shape) in core_type_shapes[op.op_type]:
                        return False
                except:
                    pass
        return True
    
    return op_selector


# ─────────────────────────────────────────────────────────────────────────────
# Candidate definitions
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES = {
    "C1_recurrence_allL": {
        "desc": "Recurrence state update (all L layers): exp/mul/sub/add loop + L2-norm",
        "selector_fn": lambda ci: _make_recurrence_selector(ci, "all_L"),
        "priority": "A",
    },
    "C2_softplus_allL": {
        "desc": "Softplus gate path (all L layers): relu/abs/exp/log + sigmoid",
        "selector_fn": lambda ci: _make_softplus_selector(ci),
        "priority": "A",
    },
    "C3_full_recurrence": {
        "desc": "Full recurrence + softplus (all L layers): C1 ∪ C2",
        "selector_fn": lambda ci: _make_full_recurrence_selector(ci),
        "priority": "A",
    },
    "C4_full_attn": {
        "desc": "Full-attention QKV path (F layer): matmul/softmax/sigmoid",
        "selector_fn": lambda ci: _make_full_attn_selector(ci),
        "priority": "B",
    },
    "C5_combined": {
        "desc": "C3 + C4: all sensitive paths (recurrence + fullAttn)",
        "selector_fn": lambda ci: _make_combined_selector(ci),
        "priority": "AB",
    },
    "C6_last_L_recurrence": {
        "desc": "Last L layer recurrence only (single-layer test)",
        "selector_fn": lambda ci: _make_recurrence_selector(ci, "last_L"),
        "priority": "A",
    },
    "C7_recurrence_core": {
        "desc": "Tightest: just exp/state/kv_mem/sub/delta (no L2-norm, no softplus)",
        "selector_fn": lambda ci: _make_recurrence_core_selector(ci),
        "priority": "A",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    candidate: str
    prompt_idx: int = 0
    # Quality metrics
    first_token_match: bool = False
    early_token_agreement: float = -1.0   # fraction of first 10 tokens matching ref
    repetition_onset: int = -1
    generated_text: str = ""
    coherent: bool = False
    # Cosine similarity
    hidden_cosine_prefill: float = -1.0
    hidden_cosine_decode_avg: float = -1.0
    logit_cosine_prefill: float = -1.0
    kl_divergence_prefill: float = -1.0
    kl_divergence_decode_avg: float = -1.0
    top1_match_rate: float = -1.0
    top5_match_rate: float = -1.0
    # Performance
    tokens_per_sec: float = -1.0
    export_time_s: float = -1.0
    prefill_ms: float = -1.0
    decode_ms: float = -1.0
    loadable: bool = False
    # Op stats
    fp32_ops_count: int = 0
    fp16_ops_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (from op_sensitivity_experiment.py)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def kl_divergence(ref_logits: np.ndarray, var_logits: np.ndarray) -> float:
    ref = ref_logits.flatten().astype(np.float64)
    var = var_logits.flatten().astype(np.float64)
    ref_shifted = ref - ref.max()
    var_shifted = var - var.max()
    ref_lse = np.log(np.sum(np.exp(ref_shifted))) + ref.max()
    var_lse = np.log(np.sum(np.exp(var_shifted))) + var.max()
    ref_log_probs = ref - ref_lse
    var_log_probs = var - var_lse
    ref_probs = np.exp(ref_log_probs)
    kl = np.sum(ref_probs * (ref_log_probs - var_log_probs))
    return max(0.0, float(kl))


def top_k_in(ref_logits: np.ndarray, var_logits: np.ndarray, k: int = 5) -> bool:
    ref_top = int(np.argmax(ref_logits.flatten()))
    var_topk = set(np.argsort(var_logits.flatten())[-k:])
    return ref_top in var_topk


def detect_repetition_onset(tokens: List[int], min_repeat_len: int = 3) -> int:
    if len(tokens) < min_repeat_len * 2:
        return -1
    for window in range(min_repeat_len, len(tokens) // 2 + 1):
        for start in range(len(tokens) - window * 2 + 1):
            segment = tokens[start:start + window]
            next_seg = tokens[start + window:start + window * 2]
            if segment == next_seg:
                return start + window
    return -1


def _build_stop_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)
    return stop_ids


def _get_full_logits(lm_out: dict) -> np.ndarray:
    if "logits" in lm_out:
        return lm_out["logits"].flatten().astype(np.float32)
    split_keys = sorted([k for k in lm_out if k.startswith("logits")],
                        key=lambda k: int(k.replace("logits", "")))
    if split_keys:
        return np.concatenate([lm_out[k].flatten() for k in split_keys]).astype(np.float32)
    raise KeyError(f"Cannot find logits in lm_head output: {list(lm_out.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Export with fine-grained op_selector
# ─────────────────────────────────────────────────────────────────────────────

def export_candidate_variant(
    model: Qwen35ForCausalLM,
    candidate_name: str,
    selector_factory: Callable,
    out_dir: str,
) -> Tuple[List[str], float, Dict[int, Tuple[int, int]]]:
    """Export all 9 chunks with fine-grained op_selector.

    Returns (chunk_paths, total_export_time, op_stats_per_chunk).
    op_stats_per_chunk[ci] = (fp32_count, fp16_count)
    """
    variant_dir = os.path.join(out_dir, candidate_name)
    os.makedirs(variant_dir, exist_ok=True)

    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=None,
        per_channel=FFN_PER_CHANNEL,
        compute_precision="float16",
    )

    paths = []
    op_stats = {}
    t_total = time.time()

    for chunk_idx in range(NUM_CHUNKS):
        sl, el = CHUNK_RANGES[chunk_idx]
        dec_path = os.path.join(variant_dir, f"ffn_{candidate_name}_chunk{chunk_idx}.mlpackage")

        if os.path.exists(dec_path):
            print(f"      [skip] {dec_path} exists")
            paths.append(dec_path)
            op_stats[chunk_idx] = (0, 0)
            continue

        print(f"      Exporting chunk {chunk_idx} [{sl}-{el-1}]...")
        t0 = time.time()

        local_num_layers = el - sl
        cfg = model.config

        class FFNWrapper(torch.nn.Module):
            def __init__(self, model, start_layer, end_layer):
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.local_num_layers = end_layer - start_layer
                self.register_buffer("k_cache", torch.zeros(
                    (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                    dtype=MODEL_DTYPE, device=TEST_DEVICE))
                self.register_buffer("v_cache", torch.zeros(
                    (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                    dtype=MODEL_DTYPE, device=TEST_DEVICE))
                if cfg.has_linear_attention():
                    conv_dim = (
                        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
                    )
                    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
                    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                    self._lin_conv_shape = (self.local_num_layers, ane_dim1, ane_dim2)
                    self._lin_rec_shape = (
                        self.local_num_layers,
                        cfg.text_config.linear_num_value_heads,
                        cfg.text_config.linear_key_head_dim,
                        cfg.text_config.linear_value_head_dim,
                    )
                    self._has_linear = True
                else:
                    self._has_linear = False
                self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                    model, self.local_num_layers, prefix="", split_full_attention_kv=True)

            def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                        linear_conv_state, linear_recurrent_state):
                out = self.model.model.process_layers_regular_single_token_export_local_state(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    kv_cache_0=None,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    linear_conv_state=linear_conv_state,
                    linear_recurrent_state=linear_recurrent_state,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    apply_final_norm=False,
                )
                if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                    out = self.model.model.norm(out)
                return out, linear_conv_state, linear_recurrent_state

        wrapper = FFNWrapper(model, sl, el).eval()
        hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
        position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        if wrapper._has_linear:
            lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
            lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
        else:
            lin_conv = torch.zeros((local_num_layers, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
            lin_rec = torch.zeros((local_num_layers, 1, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)

        conv._reset_state_buffers(wrapper)
        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec),
            check_trace=False,
        )
        conv._reset_state_buffers(wrapper)
        conv._reset_state_buffers(traced)

        # Create per-chunk op_selector
        op_sel = selector_factory(chunk_idx)

        # Count ops that will be FP32 vs FP16 using a wrapper
        fp32_count = 0
        fp16_count = 0
        def counting_selector(op):
            nonlocal fp32_count, fp16_count
            result = op_sel(op)
            if result:  # True = cast to FP16
                fp16_count += 1
            else:
                fp32_count += 1
            return result

        compute_prec = FP16ComputePrecision(op_selector=counting_selector)

        convert_kwargs = dict(
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
                ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
                ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="output_hidden_states", dtype=np.float16),
                ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
                ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
            ],
            states=wrapper.states,
            compute_precision=compute_prec,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        mlmodel = ct.convert(traced, **convert_kwargs)
        mlmodel.save(dec_path)
        elapsed = time.time() - t0

        op_stats[chunk_idx] = (fp32_count, fp16_count)
        print(f"      Saved chunk {chunk_idx} ({elapsed:.1f}s) — FP32:{fp32_count} FP16:{fp16_count}")

        del mlmodel, traced, wrapper
        gc.collect()
        paths.append(dec_path)

    total_time = time.time() - t_total
    return paths, total_time, op_stats


# ─────────────────────────────────────────────────────────────────────────────
# Inference engine (reused from op_sensitivity_experiment)
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """Loads 9 chunks + embed/lmhead and runs step-by-step inference."""

    def __init__(self, chunk_paths, embed_lmhead_path, compute_unit=ct.ComputeUnit.CPU_AND_NE,
                 mode="separate", function_name=None):
        self.compute_unit = compute_unit
        self.loadable = True
        self.mode = mode

        print(f"    Loading embed+lmhead...")
        self.embed = ct.models.MLModel(
            embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
            function_name="embed")
        self.lmhead = ct.models.MLModel(
            embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
            function_name="lmhead")

        self.ffns = []
        for ci, path in enumerate(chunk_paths):
            print(f"    Loading chunk {ci} ({compute_unit})...")
            try:
                kwargs = dict(compute_units=compute_unit)
                if function_name:
                    kwargs["function_name"] = function_name
                m = ct.models.MLModel(path, **kwargs)
                self.ffns.append(m)
            except Exception as e:
                print(f"    *** FAILED chunk {ci}: {e}")
                self.loadable = False
                return

        self.inp_maps = []
        for ci in range(len(self.ffns)):
            spec = self.ffns[ci].get_spec()
            imap = {}
            if function_name:
                for fn in spec.description.functions:
                    if fn.name == function_name:
                        for inp in fn.input:
                            try:
                                imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                            except Exception:
                                pass
                        break
            if not imap:
                for inp in spec.description.input:
                    try:
                        imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except Exception:
                        pass
            self.inp_maps.append(imap)

        self.has_linear = 'linear_conv_state' in self.inp_maps[0] if self.inp_maps else False
        self.reset_all()

    def reset_all(self):
        if not self.loadable:
            return
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_maps[ci]['linear_conv_state'], dtype=np.float16)
                              for ci in range(len(self.ffns))]
            self.lin_recs = [np.zeros(self.inp_maps[ci]['linear_recurrent_state'], dtype=np.float16)
                             for ci in range(len(self.ffns))]
        else:
            self.lin_convs = [None] * len(self.ffns)
            self.lin_recs = [None] * len(self.ffns)

    def step(self, tok_id: int, pos: int) -> Tuple[int, np.ndarray, np.ndarray]:
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0

        for ci in range(len(self.ffns)):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if self.lin_convs[ci] is not None:
                inp["linear_conv_state"] = self.lin_convs[ci]
                inp["linear_recurrent_state"] = self.lin_recs[ci]
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        logits = _get_full_logits(lm_out)
        next_tok = int(np.argmax(logits))
        return next_tok, hidden.copy(), logits

    def generate_with_traces(self, token_ids: List[int], max_gen: int, stop_ids: set):
        if not self.loadable:
            return None
        self.reset_all()
        t0 = time.time()

        for i, tid in enumerate(token_ids):
            if i >= CTX:
                break
            last_tok, last_hidden, last_logits = self.step(tid, i)
        prefill_ms = (time.time() - t0) * 1000

        result = {
            "prefill_hidden": last_hidden.copy(),
            "prefill_logits": last_logits.copy(),
            "decode_hiddens": [],
            "decode_logits": [],
            "gen_tokens": [last_tok],
            "prefill_ms": prefill_ms,
        }

        t_dec = time.time()
        prefill_end = len(token_ids)
        for gi in range(max_gen - 1):
            pos = prefill_end + gi
            if pos >= CTX - 1:
                break
            next_tok, hidden, logits = self.step(result["gen_tokens"][-1], pos)
            result["decode_hiddens"].append(hidden.copy())
            result["decode_logits"].append(logits.copy())
            result["gen_tokens"].append(next_tok)
            if next_tok in stop_ids:
                break
        t_decode = (time.time() - t_dec) * 1000
        n_dec = max(1, len(result["gen_tokens"]) - 1)
        result["tps"] = n_dec / (t_decode / 1000) if t_decode > 0 else 0
        result["decode_ms"] = t_decode
        return result

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        self.ffns = []
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch reference
# ─────────────────────────────────────────────────────────────────────────────

def pytorch_reference_with_traces(model, tokenizer, prompt, max_gen, stop_ids):
    model.eval()
    msgs = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=True
    )
    token_list = input_ids[0].tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)

    model.model.kv_cache_0.zero_()
    if hasattr(model.model, 'linear_conv_state'):
        model.model.linear_conv_state.zero_()
    if hasattr(model.model, 'linear_recurrent_state'):
        model.model.linear_recurrent_state.zero_()

    def _forward_one(tid, pos):
        inp = torch.tensor([[tid]], dtype=torch.int32, device=TEST_DEVICE)
        pos_t = torch.tensor([pos], dtype=torch.int32, device=TEST_DEVICE)
        mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=torch.float16, device=TEST_DEVICE)
        mask[:, :, :, :pos + 1] = 0
        hidden = model.model.embed_tokens(inp).to(MODEL_DTYPE)
        for li in range(len(model.model.layers)):
            hidden = model.model._process_layer_regular(li, hidden, pos_t, mask, pos_t)
        hidden = model.model.norm(hidden)
        h = hidden.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        parts = [getattr(model, f"lm_head16_{j+1}")(h).squeeze(2).permute(0, 2, 1)
                 for j in range(model.lm_head_split)]
        logits = torch.cat(parts, dim=-1)
        next_tok = int(logits.argmax(-1).flatten()[0])
        return (hidden.detach().cpu().numpy().astype(np.float32),
                logits.detach().cpu().numpy().flatten().astype(np.float32),
                next_tok)

    result = {"token_list": token_list}
    with torch.no_grad():
        for i, tid in enumerate(token_list):
            hidden, logits, next_tok = _forward_one(tid, i)
        result["prefill_hidden"] = hidden
        result["prefill_logits"] = logits

        result["gen_tokens"] = [next_tok]
        result["decode_hiddens"] = []
        result["decode_logits"] = []
        prefill_end = len(token_list)
        for gi in range(max_gen - 1):
            pos = prefill_end + gi
            if pos >= CTX - 1:
                break
            hidden, logits, next_tok = _forward_one(result["gen_tokens"][-1], pos)
            result["decode_hiddens"].append(hidden)
            result["decode_logits"].append(logits)
            result["gen_tokens"].append(next_tok)
            if next_tok in stop_ids:
                break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Compare traces
# ─────────────────────────────────────────────────────────────────────────────

def compare_traces(ref_trace: dict, var_trace: dict) -> dict:
    metrics = {}
    metrics["hidden_cos_prefill"] = cosine_sim(
        ref_trace["prefill_hidden"], var_trace["prefill_hidden"])
    metrics["logit_cos_prefill"] = cosine_sim(
        ref_trace["prefill_logits"], var_trace["prefill_logits"])
    metrics["kl_prefill"] = kl_divergence(
        ref_trace["prefill_logits"], var_trace["prefill_logits"])

    n_steps = min(len(ref_trace["decode_hiddens"]), len(var_trace["decode_hiddens"]))
    n_tok_steps = min(len(ref_trace["decode_logits"]), len(var_trace["decode_logits"]))

    h_coss, kls = [], []
    top1_matches = 0
    top5_matches = 0

    for i in range(max(n_steps, n_tok_steps)):
        if i < n_steps:
            h_coss.append(cosine_sim(
                ref_trace["decode_hiddens"][i], var_trace["decode_hiddens"][i]))
        if i < n_tok_steps:
            kls.append(kl_divergence(
                ref_trace["decode_logits"][i], var_trace["decode_logits"][i]))
            ref_top1 = int(np.argmax(ref_trace["decode_logits"][i]))
            var_top1 = int(np.argmax(var_trace["decode_logits"][i]))
            if ref_top1 == var_top1:
                top1_matches += 1
            if top_k_in(ref_trace["decode_logits"][i], var_trace["decode_logits"][i], k=5):
                top5_matches += 1

    metrics["hidden_cos_decode_avg"] = float(np.mean(h_coss)) if h_coss else -1.0
    metrics["kl_decode_avg"] = float(np.mean(kls)) if kls else -1.0
    metrics["top1_match_rate"] = top1_matches / max(n_tok_steps, 1)
    metrics["top5_match_rate"] = top5_matches / max(n_tok_steps, 1)

    # Early token agreement (first 10 decode tokens)
    ref_toks = ref_trace["gen_tokens"][:10]
    var_toks = var_trace["gen_tokens"][:10]
    n_early = min(len(ref_toks), len(var_toks))
    early_match = sum(1 for a, b in zip(ref_toks[:n_early], var_toks[:n_early]) if a == b)
    metrics["early_token_agreement"] = early_match / max(n_early, 1)

    ref_first = ref_trace["gen_tokens"][0] if ref_trace["gen_tokens"] else -1
    var_first = var_trace["gen_tokens"][0] if var_trace["gen_tokens"] else -1
    metrics["first_token_match"] = ref_first == var_first

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Incremental save / resume
# ─────────────────────────────────────────────────────────────────────────────

def _save_incremental(results: List[CandidateResult], output_dir: str):
    path = os.path.join(output_dir, "finegrain_results.json")
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)


def _load_existing_results(output_dir: str) -> List[CandidateResult]:
    path = os.path.join(output_dir, "finegrain_results.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        results = []
        for d in data:
            r = CandidateResult(candidate=d["candidate"])
            for k, v in d.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            results.append(r)
        return results
    except Exception as e:
        print(f"  Warning: could not load existing results: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    os.makedirs(args.output, exist_ok=True)

    tok_path = args.tokenizer or args.model
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    stop_ids = _build_stop_ids(tokenizer)

    print(f"Loading PyTorch model from {args.model}...")
    cfg = Qwen35Config.from_json(os.path.join(args.model, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    pt_model = Qwen35ForCausalLM(cfg)
    assert pt_model.load_pretrained_weights(args.model), "Failed to load weights"
    pt_model.eval()
    for p in pt_model.parameters():
        p.requires_grad = False

    # Phase 1: PyTorch references
    print("\n" + "=" * 70)
    print("  PHASE 1: PyTorch Reference Traces")
    print("=" * 70)
    pt_refs = {}
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {pi}: {prompt[:60]}...")
        ref = pytorch_reference_with_traces(pt_model, tokenizer, prompt, args.max_tokens, stop_ids)
        pt_refs[pi] = ref
        text = tokenizer.decode(ref["gen_tokens"], skip_special_tokens=True)
        print(f"    [{len(ref['token_list'])} prompt tok] → {text[:120]}")

    # Embed/lmhead path
    embed_lmhead = os.path.join(args.fp32_dir, "embed_lmhead_combined.mlpackage")
    if not os.path.exists(embed_lmhead):
        embed_lmhead = os.path.join(args.baseline_dir, "embed_lmhead_combined.mlpackage")
    assert os.path.exists(embed_lmhead), f"Cannot find embed_lmhead_combined.mlpackage"

    # Determine candidates
    if args.candidates:
        candidate_names = args.candidates.split(",")
    else:
        candidate_names = list(CANDIDATES.keys())

    # Build variant order: baseline_fp16, candidates, full_fp32
    variant_order = ["baseline_fp16"] + candidate_names + ["full_fp32"]

    # Resume support
    all_results: List[CandidateResult] = []
    completed_cands = set()
    if args.resume:
        existing = _load_existing_results(args.output)
        if existing:
            all_results.extend(existing)
            op_counts = Counter(r.candidate for r in existing if r.loadable)
            for cand, cnt in op_counts.items():
                if cnt >= len(PROMPTS):
                    completed_cands.add(cand)
            print(f"  Resuming: {len(existing)} results loaded, completed: {sorted(completed_cands)}")

    # Phase 2: Export & Inference
    print("\n" + "=" * 70)
    print("  PHASE 2: Fine-Grained Export & Inference")
    print("=" * 70)

    for variant_label in variant_order:
        print(f"\n{'─' * 60}")
        print(f"  CANDIDATE: {variant_label}")
        if variant_label in CANDIDATES:
            print(f"  {CANDIDATES[variant_label]['desc']}")
        print(f"{'─' * 60}")

        if variant_label in completed_cands:
            print(f"    [SKIP] Already completed in previous run")
            continue

        chunk_paths = None
        export_time = 0.0
        function_name = None
        mode = "separate"
        total_fp32 = 0
        total_fp16 = 0

        if variant_label == "baseline_fp16":
            dedup = os.path.join(args.baseline_dir, "combined_LUT4_dedup")
            if os.path.isdir(dedup):
                chunk_paths = [os.path.join(dedup, f"chunk{ci}.mlpackage") for ci in range(NUM_CHUNKS)]
                function_name = "infer"
                mode = "dedup"
            else:
                chunk_paths = [os.path.join(args.baseline_dir, f"ffn_LUT4_chunk{ci}.mlpackage")
                               for ci in range(NUM_CHUNKS)]
            print(f"    Using existing FP16: {chunk_paths[0]}")

        elif variant_label == "full_fp32":
            dedup = os.path.join(args.fp32_dir, "combined_LUT4_dedup")
            if os.path.isdir(dedup):
                chunk_paths = [os.path.join(dedup, f"chunk{ci}.mlpackage") for ci in range(NUM_CHUNKS)]
                function_name = "infer"
                mode = "dedup"
            else:
                chunk_paths = [os.path.join(args.fp32_dir, f"ffn_LUT4_chunk{ci}.mlpackage")
                               for ci in range(NUM_CHUNKS)]
            print(f"    Using existing FP32: {chunk_paths[0]}")

        elif variant_label in CANDIDATES:
            cand = CANDIDATES[variant_label]
            print(f"    Exporting 9 chunks with {variant_label} precision islands...")
            try:
                chunk_paths, export_time, op_stats = export_candidate_variant(
                    pt_model, variant_label, cand["selector_fn"], args.output)
                total_fp32 = sum(s[0] for s in op_stats.values())
                total_fp16 = sum(s[1] for s in op_stats.values())
                print(f"    Total export: {export_time:.1f}s — FP32:{total_fp32} FP16:{total_fp16}")
            except Exception as e:
                print(f"    *** Export FAILED: {e}")
                import traceback; traceback.print_exc()
                variant_dir = os.path.join(args.output, variant_label)
                if os.path.isdir(variant_dir):
                    shutil.rmtree(variant_dir)
                for pi in range(len(PROMPTS)):
                    all_results.append(CandidateResult(candidate=variant_label, prompt_idx=pi))
                _save_incremental(all_results, args.output)
                continue
        else:
            print(f"    *** Unknown candidate: {variant_label}")
            continue

        # Verify paths exist
        if not chunk_paths or not all(os.path.exists(p) for p in chunk_paths):
            print(f"    *** Missing chunk files, skipping")
            for pi in range(len(PROMPTS)):
                all_results.append(CandidateResult(candidate=variant_label, prompt_idx=pi))
            _save_incremental(all_results, args.output)
            continue

        # Load & inference
        try:
            engine = InferenceEngine(chunk_paths, embed_lmhead, mode=mode,
                                     function_name=function_name)
        except Exception as e:
            print(f"    *** Engine load failed: {e}")
            for pi in range(len(PROMPTS)):
                all_results.append(CandidateResult(candidate=variant_label, prompt_idx=pi))
            _save_incremental(all_results, args.output)
            continue

        if not engine.loadable:
            print(f"    *** Not loadable on ANE")
            for pi in range(len(PROMPTS)):
                all_results.append(CandidateResult(candidate=variant_label, prompt_idx=pi,
                                                   loadable=False))
            engine.cleanup()
            _save_incremental(all_results, args.output)
            continue

        for pi, prompt in enumerate(PROMPTS):
            print(f"\n    Prompt {pi}: {prompt[:50]}...")
            ref = pt_refs[pi]
            var_trace = engine.generate_with_traces(ref["token_list"], args.max_tokens, stop_ids)

            if var_trace is None:
                all_results.append(CandidateResult(candidate=variant_label, prompt_idx=pi))
                continue

            metrics = compare_traces(ref, var_trace)
            text = tokenizer.decode(var_trace["gen_tokens"], skip_special_tokens=True)
            rep_onset = detect_repetition_onset(var_trace["gen_tokens"])

            # Coherence: no repetition within first 20 tokens + first token matches
            coherent = (rep_onset < 0 or rep_onset > 20) and metrics["first_token_match"]

            r = CandidateResult(
                candidate=variant_label,
                prompt_idx=pi,
                first_token_match=metrics["first_token_match"],
                early_token_agreement=metrics["early_token_agreement"],
                repetition_onset=rep_onset,
                generated_text=text[:300],
                coherent=coherent,
                hidden_cosine_prefill=metrics["hidden_cos_prefill"],
                hidden_cosine_decode_avg=metrics["hidden_cos_decode_avg"],
                logit_cosine_prefill=metrics["logit_cos_prefill"],
                kl_divergence_prefill=metrics["kl_prefill"],
                kl_divergence_decode_avg=metrics["kl_decode_avg"],
                top1_match_rate=metrics["top1_match_rate"],
                top5_match_rate=metrics["top5_match_rate"],
                tokens_per_sec=var_trace["tps"],
                export_time_s=export_time,
                prefill_ms=var_trace.get("prefill_ms", -1),
                decode_ms=var_trace.get("decode_ms", -1),
                loadable=True,
                fp32_ops_count=total_fp32,
                fp16_ops_count=total_fp16,
            )
            all_results.append(r)

            print(f"      h_cos_pf={metrics['hidden_cos_prefill']:.6f} "
                  f"h_cos_dec={metrics['hidden_cos_decode_avg']:.6f} "
                  f"KL_dec={metrics['kl_decode_avg']:.4f}")
            print(f"      top1={metrics['top1_match_rate']:.0%} "
                  f"early_agree={metrics['early_token_agreement']:.0%} "
                  f"rep={rep_onset} 1st={metrics['first_token_match']}")
            print(f"      tps={var_trace['tps']:.1f} "
                  f"pf_ms={var_trace.get('prefill_ms', -1):.0f} "
                  f"dec_ms={var_trace.get('decode_ms', -1):.0f}")
            print(f"      Gen: {text[:100]}")

        engine.cleanup()
        gc.collect()

        # Cleanup exported chunks (not baselines)
        if variant_label not in ("baseline_fp16", "full_fp32"):
            variant_dir = os.path.join(args.output, variant_label)
            if os.path.isdir(variant_dir):
                print(f"    Cleaning up {variant_dir}...")
                shutil.rmtree(variant_dir)

        _save_incremental(all_results, args.output)
        print(f"    [Saved {len(all_results)} results]")

    # Free PyTorch model
    del pt_model
    gc.collect()

    # Phase 3: Results
    print("\n" + "=" * 70)
    print("  PHASE 3: Fine-Grained Precision Island Ranking")
    print("=" * 70)
    _print_results_table(all_results)

    results_path = os.path.join(args.output, "finegrain_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


def _print_results_table(results: List[CandidateResult]):
    """Print results sorted by quality."""
    by_cand = {}
    for r in results:
        by_cand.setdefault(r.candidate, []).append(r)

    rows = []
    for cand, rlist in by_cand.items():
        valid = [r for r in rlist if r.loadable and r.hidden_cosine_prefill >= 0]
        if not valid:
            rows.append({"cand": cand, "broken": True})
            continue
        n = len(valid)
        rows.append({
            "cand": cand,
            "h_cos_pf": sum(r.hidden_cosine_prefill for r in valid) / n,
            "h_cos_dec": sum(r.hidden_cosine_decode_avg for r in valid if r.hidden_cosine_decode_avg >= 0) / max(1, sum(1 for r in valid if r.hidden_cosine_decode_avg >= 0)),
            "kl_dec": sum(r.kl_divergence_decode_avg for r in valid if r.kl_divergence_decode_avg >= 0) / max(1, sum(1 for r in valid if r.kl_divergence_decode_avg >= 0)),
            "top1": sum(r.top1_match_rate for r in valid) / n,
            "early": sum(r.early_token_agreement for r in valid) / n,
            "1st_match": sum(1 for r in valid if r.first_token_match),
            "coherent": sum(1 for r in valid if r.coherent),
            "rep_count": sum(1 for r in valid if 0 <= r.repetition_onset <= 20),
            "tps": sum(r.tokens_per_sec for r in valid if r.tokens_per_sec > 0) / max(1, sum(1 for r in valid if r.tokens_per_sec > 0)),
            "pf_ms": sum(r.prefill_ms for r in valid if r.prefill_ms > 0) / max(1, sum(1 for r in valid if r.prefill_ms > 0)),
            "fp32": valid[0].fp32_ops_count,
            "broken": False,
        })

    # Sort: baselines at edges, candidates by hidden cos decode (descending = better)
    baselines = [r for r in rows if r["cand"] in ("baseline_fp16", "full_fp32")]
    cands = [r for r in rows if r["cand"] not in ("baseline_fp16", "full_fp32")]
    cands.sort(key=lambda r: r.get("h_cos_dec", -1), reverse=True)

    print(f"\n  {'Candidate':<26} {'h_pf':>7} {'h_dec':>7} {'KL_dec':>8} "
          f"{'top1%':>6} {'early%':>6} {'1st':>4} {'coh':>4} {'rep':>4} "
          f"{'tok/s':>6} {'pf_ms':>6} {'FP32':>5}")
    print("  " + "─" * 110)

    def _fmt(r):
        if r["broken"]:
            print(f"  {r['cand']:<26} {'BROKEN':>7}")
            return
        def f(v, w=7, d=4):
            return f"{v:{w}.{d}f}" if v >= 0 else f"{'N/A':>{w}}"
        def fp(v, w=6):
            return f"{v:{w}.0%}" if v >= 0 else f"{'N/A':>{w}}"
        print(f"  {r['cand']:<26} {f(r['h_cos_pf'])} {f(r['h_cos_dec'])} "
              f"{f(r['kl_dec'], 8)} {fp(r['top1'])} {fp(r['early'])} "
              f"{r['1st_match']:>4}/3 {r['coherent']:>4}/3 {r['rep_count']:>4} "
              f"{r['tps']:>6.1f} {r['pf_ms']:>6.0f} {r['fp32']:>5}")

    for r in baselines:
        if r["cand"] == "baseline_fp16":
            _fmt(r)
    for r in cands:
        _fmt(r)
    for r in baselines:
        if r["cand"] == "full_fp32":
            _fmt(r)

    # Decision summary
    print(f"\n  DECISION SUMMARY:")
    valid_cands = [r for r in cands if not r["broken"]]
    if valid_cands:
        best = valid_cands[0]
        fp32_row = next((r for r in baselines if r["cand"] == "full_fp32" and not r.get("broken", True)), None)
        fp16_row = next((r for r in baselines if r["cand"] == "baseline_fp16" and not r.get("broken", True)), None)

        print(f"    Best local candidate: {best['cand']}")
        print(f"      h_cos_dec={best['h_cos_dec']:.4f}  top1={best['top1']:.0%}  "
              f"coherent={best['coherent']}/3  tok/s={best['tps']:.1f}")
        if fp32_row:
            quality_ratio = best['h_cos_dec'] / max(fp32_row['h_cos_dec'], 0.001)
            speed_ratio = best['tps'] / max(fp32_row['tps'], 0.001)
            print(f"    vs full FP32: quality={quality_ratio:.1%}  speed={speed_ratio:.1%}")
        if fp16_row:
            improvement = best['h_cos_dec'] - fp16_row['h_cos_dec']
            print(f"    vs baseline FP16: hidden_cos improvement=+{improvement:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-grained selective FP32 experiment")
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--baseline-dir", required=True, help="FP16 baseline directory")
    parser.add_argument("--fp32-dir", required=True, help="FP32 reference directory")
    parser.add_argument("--output", default="tests/dev/finegrain_fp32_results")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--candidates", type=str, default=None,
                        help="Comma-separated candidate names to test (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")
    args = parser.parse_args()
    print(f"  Model: {args.model}")
    print(f"  Baseline: {args.baseline_dir}")
    print(f"  FP32 ref: {args.fp32_dir}")
    print(f"  Max tokens: {args.max_tokens}")
    run_experiment(args)


if __name__ == "__main__":
    main()
