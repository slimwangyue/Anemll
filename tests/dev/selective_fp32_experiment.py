#!/usr/bin/env python3
"""Targeted selective-FP32 ablation experiment for Qwen3.5-4B on ANE.

Investigates two sensitive regions:
  1. Recurrent-state update path (linear attention core loop)
  2. Full-attention input path (hidden_before_F, L→F boundary)

Tests 6 selective-FP32 variants against baseline FP16, full FP32, and PyTorch
reference across 3 prompts.

Metrics: hidden cosine, recurrent-state cosine/L2, first-token agreement,
repetition onset, ANE loadability, fallback signs, prefill latency, decode latency.

Usage:
    python tests/dev/selective_fp32_experiment.py \
        --model models/Qwen__Qwen3.5-4B \
        --baseline-dir qwen3_5_stable_lut4ffn_lut6em_fp16 \
        --fp32-dir qwen3_5_stable_lut4ffn_lut6em_fp32 \
        --output tests/dev/selective_fp32_results \
        --max-tokens 60
"""
import argparse
import gc
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import coremltools as ct
from transformers import AutoTokenizer

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES, LUT_BITS, LM_HEAD_LUT, PER_CHANNEL, FFN_PER_CHANNEL
from anemll.models.qwen3_5_model import (
    Qwen35Config, Qwen35ForCausalLM, MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CANDIDATE OP MAP
# ─────────────────────────────────────────────────────────────────────────────
# Recurrent-state update path (_recurrent_gated_delta_rule):
#   g_t = g[:,:,i].exp()                 → exp
#   state = state * g_t                  → mul
#   kv_mem = (state * k_t).sum()         → mul, reduce_sum
#   delta = (v_t - kv_mem) * beta_t      → sub, mul
#   state = state + k_t * delta          → mul, add
#   out = (state * q_t).sum()            → mul, reduce_sum
#
# Layout stage (softplus for g):
#   sp = relu(x) + log(1 + exp(-|x|))   → relu, abs, exp, log, add
#   g = -A_log.exp() * sp               → exp, mul
#
# Full-attention input path:
#   q/k/v = Conv2d projections           → conv (linear)
#   qk = q @ k^T / sqrt(dk)             → matmul, mul
#   attn = softmax(qk + mask)            → add, softmax
#   out = attn @ v                       → matmul
#
# Key: "skip_ops_by_type" in add_fp16_cast keeps those MIL ops in FP32.

VARIANTS: Dict[str, Dict] = {
    "V0_baseline_fp16": {
        "label": "Baseline FP16",
        "skip_ops": None,  # standard FP16 — no custom pipeline
        "use_full_fp32": False,
    },
    "V1_exp_only": {
        "label": "exp only → FP32",
        "skip_ops": "exp",
        "use_full_fp32": False,
    },
    "V2_exp_mul_add": {
        "label": "exp+mul+add → FP32",
        "skip_ops": "exp,mul,add",
        "use_full_fp32": False,
    },
    "V3_exp_mul_add_matmul": {
        "label": "exp+mul+add+matmul → FP32",
        "skip_ops": "exp,mul,add,matmul",
        "use_full_fp32": False,
    },
    "V4_recurrence_full": {
        "label": "Recurrence path full (exp+mul+add+sub+reduce_sum+log+abs+relu)",
        "skip_ops": "exp,mul,add,sub,reduce_sum,log,abs,relu",
        "use_full_fp32": False,
    },
    "V5_full_attn_input": {
        "label": "Full-attn input (matmul+softmax+add+mul+conv)",
        "skip_ops": "matmul,softmax,add,mul,conv",
        "use_full_fp32": False,
    },
    "V6_recurrence_plus_attn": {
        "label": "Best recurrence + best attn (exp+mul+add+sub+reduce_sum+log+abs+relu+matmul+softmax+conv)",
        "skip_ops": "exp,mul,add,sub,reduce_sum,log,abs,relu,matmul,softmax,conv",
        "use_full_fp32": False,
    },
    "V7_full_fp32": {
        "label": "Full FLOAT32 compute",
        "skip_ops": None,
        "use_full_fp32": True,
    },
}

PROMPTS = [
    "What is a stack in computer science?",
    "教我做红烧鱼",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VariantResult:
    variant_id: str = ""
    label: str = ""
    skip_ops: str = ""
    prompt_idx: int = 0
    prompt_text: str = ""
    # Quality metrics
    hidden_cosine: float = -1.0
    recurrent_state_cosine: float = -1.0
    recurrent_state_l2: float = -1.0
    first_token_match: bool = False
    early_token_agreement: float = 0.0  # fraction of first 10 tokens matching PyTorch
    generated_text: str = ""
    repetition_onset: int = -1  # token index where repetition starts (-1 = none)
    # ANE metrics
    loadable_cpu_and_ne: bool = False
    fallback_signs: str = ""  # "none", "partial", "full"
    # Latency
    prefill_latency_ms: float = -1.0
    decode_latency_ms: float = -1.0
    tokens_per_sec: float = -1.0
    # Export
    export_time_s: float = -1.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.flatten().astype(np.float64) - b.flatten().astype(np.float64)))


def detect_repetition_onset(tokens: List[int], min_repeat_len: int = 3) -> int:
    """Detect where repetitive patterns begin. Returns token index or -1."""
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


def _argmax_from_lm_out(lm_out):
    if "logits" in lm_out:
        return int(np.argmax(lm_out["logits"].flatten()))
    if "argmax_idx" in lm_out:
        return int(lm_out["argmax_idx"].flatten()[0])
    split_keys = sorted([k for k in lm_out if k.startswith("logits")],
                        key=lambda k: int(k.replace("logits", "")))
    if split_keys:
        full = np.concatenate([lm_out[k].flatten() for k in split_keys])
        return int(np.argmax(full))
    raise KeyError(f"Cannot find logits in lm_head output: {list(lm_out.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELECTIVE-FP32 EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def make_pass_pipeline(skip_ops: Optional[str], use_full_fp32: bool):
    """Create a PassPipeline for selective FP32.

    Args:
        skip_ops: Comma-separated MIL op types to keep in FP32
                  (e.g. "exp,mul,add"). None = standard FP16.
        use_full_fp32: If True, use FLOAT32 compute precision globally.
    Returns:
        (compute_precision, pass_pipeline) tuple.
    """
    if use_full_fp32:
        return ct.precision.FLOAT32, None

    if skip_ops is None:
        return ct.precision.FLOAT16, None

    # Selective: base is FP16, but skip certain ops from the FP16 cast
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": skip_ops})
    return ct.precision.FLOAT16, pipeline


def export_selective_chunk(
    model: Qwen35ForCausalLM,
    chunk_idx: int,
    out_dir: str,
    variant_id: str,
    skip_ops: Optional[str],
    use_full_fp32: bool,
) -> Tuple[str, float]:
    """Export a single FFN decode chunk with selective FP32 ops.

    Returns (path_to_mlpackage, export_time_seconds).
    """
    sl, el = CHUNK_RANGES[chunk_idx]
    tag = variant_id
    dec_path = os.path.join(out_dir, f"ffn_{tag}_chunk{chunk_idx}.mlpackage")

    if os.path.exists(dec_path):
        print(f"    [skip] {dec_path} exists")
        return dec_path, 0.0

    compute_prec, pipeline = make_pass_pipeline(skip_ops, use_full_fp32)

    print(f"    Exporting chunk {chunk_idx} [{sl}-{el-1}] variant={variant_id}...")
    t0 = time.time()

    # Build the FFNWrapper the same way as the converter, but inject our pipeline
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=None,  # no LUT for experiments
        per_channel=FFN_PER_CHANNEL,
        compute_precision="float32" if use_full_fp32 else "float16",
    )

    total_layers = model.config.num_hidden_layers
    local_num_layers = el - sl

    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter as _Conv
    from anemll.models.qwen3_5_model import ane_conv_state_shape

    class FFNWrapper(torch.nn.Module):
        def __init__(self, model, start_layer, end_layer):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.local_num_layers = end_layer - start_layer
            cfg = model.config
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
            self.states = _Conv.GetChunkLocalTransformerStates(
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
    cfg = model.config
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
    if pipeline is not None:
        convert_kwargs["pass_pipeline"] = pipeline

    mlmodel = ct.convert(traced, **convert_kwargs)
    mlmodel.save(dec_path)

    elapsed = time.time() - t0
    del mlmodel, traced, wrapper
    gc.collect()
    print(f"    Saved {dec_path} ({elapsed:.1f}s)")
    return dec_path, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE ENGINE (per-chunk decode, reuses embed/lmhead from baselines)
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentEngine:
    """Minimal inference engine loading variant chunks + shared embed/lmhead."""

    def __init__(
        self,
        chunk_paths: List[str],
        embed_lmhead_path: str,
        compute_unit=ct.ComputeUnit.CPU_AND_NE,
    ):
        self.compute_unit = compute_unit
        # embed + lmhead from combined (always CPU_ONLY for safety)
        print(f"    Loading embed from {embed_lmhead_path} ...")
        self.embed = ct.models.MLModel(
            embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
            function_name="embed")
        print(f"    Loading lmhead from {embed_lmhead_path} ...")
        self.lmhead = ct.models.MLModel(
            embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
            function_name="lmhead")

        self.ffns = []
        self.loadable = True
        self.fallback = "none"  # Track ANE fallback
        for ci, path in enumerate(chunk_paths):
            print(f"    Loading chunk {ci} from {path} ({compute_unit})...")
            try:
                m = ct.models.MLModel(path, compute_units=compute_unit)
                self.ffns.append(m)
            except Exception as e:
                print(f"    *** FAILED to load chunk {ci} on {compute_unit}: {e}")
                print(f"    Falling back to CPU_AND_NE...")
                try:
                    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
                    self.ffns.append(m)
                    self.fallback = "partial"
                except Exception as e2:
                    print(f"    *** FAILED on CPU_AND_NE too: {e2}")
                    self.loadable = False
                    return

        # Detect input shapes
        self.inp_maps = []
        for ci in range(len(self.ffns)):
            spec = self.ffns[ci].get_spec()
            imap = {}
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

    def _step(self, tok_id, pos):
        """Single-token decode step. Returns (next_token_id, hidden_out, lin_rec_out)."""
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        lin_recs_out = []
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
                lin_recs_out.append(out['linear_recurrent_state_out'])
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        next_tok = _argmax_from_lm_out(lm_out)
        return next_tok, hidden, lin_recs_out

    def generate(self, token_ids, max_gen, stop_ids):
        """Prefill (token-by-token) + decode.

        Returns: (gen_tokens, hidden_at_first_gen, lin_recs_at_first_gen,
                  prefill_ms, decode_ms, tokens_per_sec)
        """
        if not self.loadable:
            return [], None, None, -1, -1, -1

        self.reset_all()
        t0 = time.time()
        # Prefill: feed all prompt tokens
        for i, tid in enumerate(token_ids):
            if i >= CTX:
                break
            last_tok, last_hidden, last_recs = self._step(tid, i)
        prefill_end_pos = len(token_ids)
        t_prefill = (time.time() - t0) * 1000

        first_hidden = last_hidden.copy()
        first_recs = [r.copy() for r in last_recs] if last_recs else None

        # Decode
        gen_tokens = [last_tok]
        t_dec = time.time()
        for gi in range(max_gen - 1):
            pos = prefill_end_pos + gi
            if pos >= CTX - 1:
                break
            next_id, _, _ = self._step(gen_tokens[-1], pos)
            gen_tokens.append(next_id)
            if next_id in stop_ids:
                break
        t_decode = (time.time() - t_dec) * 1000
        n_decode_tokens = max(1, len(gen_tokens) - 1)
        tps = n_decode_tokens / (t_decode / 1000) if t_decode > 0 else 0
        return gen_tokens, first_hidden, first_recs, t_prefill, t_decode, tps

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        self.ffns = []
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 5. PYTORCH REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def pytorch_reference_generate(model, tokenizer, prompt, max_gen, stop_ids):
    """Generate with PyTorch model (FP16 weights, FP32 math in recurrence).

    Calls the model in token-by-token decode mode matching the CoreML path.
    """
    model.eval()
    msgs = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=True
    )
    if isinstance(input_ids, torch.Tensor):
        token_list = input_ids[0].tolist()
    else:
        token_list = list(input_ids)

    # Reset all states
    model.model.kv_cache_0.zero_()
    if hasattr(model.model, 'linear_conv_state'):
        model.model.linear_conv_state.zero_()
    if hasattr(model.model, 'linear_recurrent_state'):
        model.model.linear_recurrent_state.zero_()

    gen_tokens = []
    with torch.no_grad():
        # Prefill token-by-token (matching CoreML decode path)
        for i, tid in enumerate(token_list):
            inp = torch.tensor([[tid]], dtype=torch.int32, device=TEST_DEVICE)
            pos = torch.tensor([i], dtype=torch.int32, device=TEST_DEVICE)
            mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=torch.float16, device=TEST_DEVICE)
            mask[:, :, :, :i + 1] = 0
            # Use _process_layer_regular for each layer (the runtime path)
            hidden = model.model.embed_tokens(inp).to(MODEL_DTYPE)
            for li in range(len(model.model.layers)):
                hidden = model.model._process_layer_regular(
                    li, hidden, pos, mask, pos)
            hidden = model.model.norm(hidden)
            # LM head
            h = hidden.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
            parts = [getattr(model, f"lm_head16_{j+1}")(h).squeeze(2).permute(0, 2, 1)
                     for j in range(model.lm_head_split)]
            logits = torch.cat(parts, dim=-1)
            next_tok = int(logits.argmax(-1).flatten()[0])

        gen_tokens.append(next_tok)
        prefill_end = len(token_list)

        # Decode
        for gi in range(max_gen - 1):
            pos_val = prefill_end + gi
            if pos_val >= CTX - 1:
                break
            inp = torch.tensor([[gen_tokens[-1]]], dtype=torch.int32, device=TEST_DEVICE)
            pos = torch.tensor([pos_val], dtype=torch.int32, device=TEST_DEVICE)
            mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=torch.float16, device=TEST_DEVICE)
            mask[:, :, :, :pos_val + 1] = 0
            hidden = model.model.embed_tokens(inp).to(MODEL_DTYPE)
            for li in range(len(model.model.layers)):
                hidden = model.model._process_layer_regular(
                    li, hidden, pos, mask, pos)
            hidden = model.model.norm(hidden)
            h = hidden.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
            parts = [getattr(model, f"lm_head16_{j+1}")(h).squeeze(2).permute(0, 2, 1)
                     for j in range(model.lm_head_split)]
            logits = torch.cat(parts, dim=-1)
            next_tok = int(logits.argmax(-1).flatten()[0])
            gen_tokens.append(next_tok)
            if next_tok in stop_ids:
                break

    return token_list, gen_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    os.makedirs(args.output, exist_ok=True)

    # ── Load tokenizer ──
    tok_path = args.tokenizer or args.model
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    stop_ids = _build_stop_ids(tokenizer)

    # ── Load PyTorch model ──
    print(f"Loading PyTorch model from {args.model}...")
    cfg = Qwen35Config.from_json(os.path.join(args.model, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    pt_model = Qwen35ForCausalLM(cfg)
    assert pt_model.load_pretrained_weights(args.model), f"Failed to load weights"
    pt_model.eval()
    for p in pt_model.parameters():
        p.requires_grad = False

    # ── Generate PyTorch references ──
    print("\n" + "=" * 70)
    print("  PHASE 1: PyTorch Reference Generation")
    print("=" * 70)
    pt_refs = {}
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {pi}: {prompt[:60]}...")
        token_list, gen_tokens = pytorch_reference_generate(
            pt_model, tokenizer, prompt, args.max_tokens, stop_ids)
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pt_refs[pi] = {
            "prompt_tokens": token_list,
            "gen_tokens": gen_tokens,
            "text": text,
        }
        print(f"    [{len(token_list)} prompt tok] → {text[:120]}")

    # ── Find embed_lmhead ──
    # Prefer FP32 dir's embed_lmhead (it exists). Embed/lmhead are not the
    # bottleneck; they're small and the same regardless of FFN precision.
    embed_lmhead = os.path.join(args.fp32_dir, "embed_lmhead_combined.mlpackage")
    if not os.path.exists(embed_lmhead):
        # Fallback: look in baseline dir
        embed_lmhead = os.path.join(args.baseline_dir, "embed_lmhead_combined.mlpackage")
    if not os.path.exists(embed_lmhead):
        print("ERROR: Cannot find embed_lmhead_combined.mlpackage")
        print("  Checked:", args.fp32_dir, args.baseline_dir)
        sys.exit(1)
    print(f"\nUsing embed+lmhead: {embed_lmhead}")

    # ── Phase 2: Export selective-FP32 variants ──
    print("\n" + "=" * 70)
    print("  PHASE 2: Selective-FP32 Export (all 9 chunks per variant)")
    print("=" * 70)

    variant_chunk_paths = {}
    variant_export_times = {}

    for vid, vcfg in VARIANTS.items():
        skip_ops = vcfg["skip_ops"]
        use_full_fp32 = vcfg["use_full_fp32"]

        # Check if we can reuse existing models
        if vid == "V0_baseline_fp16":
            # Use existing FP16 baseline models
            base = args.baseline_dir
            dedup_dir = os.path.join(base, "combined_LUT4_dedup")
            if os.path.isdir(dedup_dir):
                paths = [os.path.join(dedup_dir, f"chunk{ci}.mlpackage") for ci in range(NUM_CHUNKS)]
                if all(os.path.exists(p) for p in paths):
                    variant_chunk_paths[vid] = ("dedup", paths)
                    variant_export_times[vid] = 0.0
                    print(f"\n  {vid}: Using existing dedup models from {dedup_dir}")
                    continue
            # Fallback to separate
            paths = [os.path.join(base, f"ffn_LUT4_chunk{ci}.mlpackage") for ci in range(NUM_CHUNKS)]
            if all(os.path.exists(p) for p in paths):
                variant_chunk_paths[vid] = ("separate", paths)
                variant_export_times[vid] = 0.0
                print(f"\n  {vid}: Using existing separate models from {base}")
                continue

        if vid == "V7_full_fp32":
            # Use existing FP32 models
            dedup_dir = os.path.join(args.fp32_dir, "combined_LUT4_dedup")
            if os.path.isdir(dedup_dir):
                paths = [os.path.join(dedup_dir, f"chunk{ci}.mlpackage") for ci in range(NUM_CHUNKS)]
                if all(os.path.exists(p) for p in paths):
                    variant_chunk_paths[vid] = ("dedup", paths)
                    variant_export_times[vid] = 0.0
                    print(f"\n  {vid}: Using existing dedup models from {dedup_dir}")
                    continue

        # Export new variant
        variant_dir = os.path.join(args.output, vid)
        os.makedirs(variant_dir, exist_ok=True)
        print(f"\n  {vid} ({vcfg['label']}):")
        print(f"    skip_ops = {skip_ops}")

        t_total = time.time()
        paths = []
        for ci in range(NUM_CHUNKS):
            path, _ = export_selective_chunk(
                pt_model, ci, variant_dir, vid, skip_ops, use_full_fp32)
            paths.append(path)
        variant_chunk_paths[vid] = ("separate", paths)
        variant_export_times[vid] = time.time() - t_total
        print(f"    Total export: {variant_export_times[vid]:.1f}s")

    # Free the model from memory before loading CoreML
    del pt_model
    gc.collect()

    # ── Phase 3: Inference & Measurement ──
    print("\n" + "=" * 70)
    print("  PHASE 3: Inference & Metrics (3 prompts × 8 variants)")
    print("=" * 70)

    all_results: List[VariantResult] = []

    for vid, vcfg in VARIANTS.items():
        print(f"\n{'─' * 60}")
        print(f"  Variant: {vid} — {vcfg['label']}")
        print(f"{'─' * 60}")

        mode, paths = variant_chunk_paths[vid]

        # Load engine
        try:
            if mode == "dedup":
                # Load via dedup (multifunction) — use "infer" function
                engine_ffns = []
                load_ok = True
                for ci, p in enumerate(paths):
                    print(f"    Loading dedup chunk {ci} (CPU_AND_NE)...")
                    try:
                        m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                              function_name="infer")
                        engine_ffns.append(m)
                    except Exception as e:
                        print(f"    *** FAILED loading chunk {ci}: {e}")
                        load_ok = False
                        break
                if not load_ok:
                    for pi in range(len(PROMPTS)):
                        r = VariantResult(
                            variant_id=vid, label=vcfg["label"],
                            skip_ops=vcfg["skip_ops"] or "",
                            prompt_idx=pi, prompt_text=PROMPTS[pi],
                            loadable_cpu_and_ne=False, fallback_signs="failed",
                        )
                        all_results.append(r)
                    continue
                # Build a simple wrapper
                engine = _build_dedup_engine(engine_ffns, embed_lmhead, paths)
            else:
                engine = ExperimentEngine(paths, embed_lmhead, ct.ComputeUnit.CPU_AND_NE)
                if not engine.loadable:
                    for pi in range(len(PROMPTS)):
                        r = VariantResult(
                            variant_id=vid, label=vcfg["label"],
                            skip_ops=vcfg["skip_ops"] or "",
                            prompt_idx=pi, prompt_text=PROMPTS[pi],
                            loadable_cpu_and_ne=False, fallback_signs="failed",
                        )
                        all_results.append(r)
                    continue
        except Exception as e:
            print(f"    *** Engine load failed: {e}")
            for pi in range(len(PROMPTS)):
                r = VariantResult(
                    variant_id=vid, label=vcfg["label"],
                    skip_ops=vcfg["skip_ops"] or "",
                    prompt_idx=pi, prompt_text=PROMPTS[pi],
                    loadable_cpu_and_ne=False, fallback_signs="failed",
                )
                all_results.append(r)
            continue

        for pi, prompt in enumerate(PROMPTS):
            print(f"\n    Prompt {pi}: {prompt[:50]}...")
            ref = pt_refs[pi]
            token_list = ref["prompt_tokens"]
            ref_gen = ref["gen_tokens"]

            gen_tokens, first_hidden, first_recs, pf_ms, dc_ms, tps = engine.generate(
                token_list, args.max_tokens, stop_ids)
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # Metrics
            first_match = (len(gen_tokens) > 0 and len(ref_gen) > 0
                           and gen_tokens[0] == ref_gen[0])
            n_compare = min(10, len(gen_tokens), len(ref_gen))
            early_agree = (sum(1 for j in range(n_compare)
                               if gen_tokens[j] == ref_gen[j]) / max(1, n_compare))
            rep_onset = detect_repetition_onset(gen_tokens)

            # Hidden cosine — compare against baseline FP16 (populated later)
            # For now store the hidden
            r = VariantResult(
                variant_id=vid,
                label=vcfg["label"],
                skip_ops=vcfg["skip_ops"] or "",
                prompt_idx=pi,
                prompt_text=prompt,
                first_token_match=first_match,
                early_token_agreement=early_agree,
                generated_text=text[:300],
                repetition_onset=rep_onset,
                loadable_cpu_and_ne=True,
                fallback_signs=engine.fallback if hasattr(engine, 'fallback') else "none",
                prefill_latency_ms=pf_ms,
                decode_latency_ms=dc_ms,
                tokens_per_sec=tps,
                export_time_s=variant_export_times.get(vid, 0),
            )
            all_results.append(r)
            print(f"      1st-tok match={first_match} | early-agree={early_agree:.0%} "
                  f"| rep-onset={rep_onset} | pf={pf_ms:.0f}ms dc={dc_ms:.0f}ms tps={tps:.1f}")
            print(f"      Gen: {text[:120]}")

        # Cleanup engine
        if mode == "dedup":
            for m in engine_ffns:
                del m
            del engine_ffns
        else:
            engine.cleanup()
        gc.collect()

        # Cleanup exported mlpackages to save disk space (V1-V6 only)
        if mode == "separate" and vid not in ("V0_baseline_fp16", "V7_full_fp32"):
            variant_dir = os.path.join(args.output, vid)
            if os.path.isdir(variant_dir):
                import shutil
                print(f"    Cleaning up {variant_dir} to save disk space...")
                shutil.rmtree(variant_dir)
                gc.collect()

    # ── Phase 4: Report ──
    print("\n" + "=" * 70)
    print("  PHASE 4: Ablation Results")
    print("=" * 70)

    _print_ablation_table(all_results, pt_refs, tokenizer)

    # Save raw results
    results_path = os.path.join(args.output, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nRaw results saved to {results_path}")

    # Save recommendation
    _print_recommendation(all_results)


def _build_dedup_engine(ffn_models, embed_lmhead_path, chunk_paths):
    """Build an ExperimentEngine-like object from pre-loaded dedup models."""

    class DedupWrapper:
        def __init__(self, ffn_models, embed_lmhead_path):
            self.embed = ct.models.MLModel(
                embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
                function_name="embed")
            self.lmhead = ct.models.MLModel(
                embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
                function_name="lmhead")
            self.ffns = ffn_models
            self.loadable = True
            self.fallback = "none"
            self.inp_maps = []
            for ci in range(len(self.ffns)):
                spec = self.ffns[ci].get_spec()
                imap = {}
                fn_inputs = None
                for fn in spec.description.functions:
                    if fn.name == "infer":
                        fn_inputs = fn.input
                        break
                if fn_inputs is None:
                    fn_inputs = spec.description.input
                for inp in fn_inputs:
                    try:
                        imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except Exception:
                        pass
                self.inp_maps.append(imap)
            self.has_linear = 'linear_conv_state' in self.inp_maps[0] if self.inp_maps else False
            self.reset_all()

        def reset_all(self):
            self.states = [m.make_state() for m in self.ffns]
            if self.has_linear:
                self.lin_convs = [np.zeros(self.inp_maps[ci]['linear_conv_state'], dtype=np.float16)
                                  for ci in range(len(self.ffns))]
                self.lin_recs = [np.zeros(self.inp_maps[ci]['linear_recurrent_state'], dtype=np.float16)
                                 for ci in range(len(self.ffns))]
            else:
                self.lin_convs = [None] * len(self.ffns)
                self.lin_recs = [None] * len(self.ffns)

        def _step(self, tok_id, pos):
            tok = np.array([[tok_id]], dtype=np.int32)
            hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
            mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
            mask[:, :, :, :pos + 1] = 0
            lin_recs_out = []
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
                    lin_recs_out.append(out['linear_recurrent_state_out'])
            lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
            next_tok = _argmax_from_lm_out(lm_out)
            return next_tok, hidden, lin_recs_out

        def generate(self, token_ids, max_gen, stop_ids):
            if not self.loadable:
                return [], None, None, -1, -1, -1
            self.reset_all()
            t0 = time.time()
            for i, tid in enumerate(token_ids):
                if i >= CTX:
                    break
                last_tok, last_hidden, last_recs = self._step(tid, i)
            t_prefill = (time.time() - t0) * 1000
            first_hidden = last_hidden.copy() if last_hidden is not None else None
            first_recs = [r.copy() for r in last_recs] if last_recs else None
            gen_tokens = [last_tok]
            t_dec = time.time()
            prefill_end = len(token_ids)
            for gi in range(max_gen - 1):
                pos = prefill_end + gi
                if pos >= CTX - 1:
                    break
                next_id, _, _ = self._step(gen_tokens[-1], pos)
                gen_tokens.append(next_id)
                if next_id in stop_ids:
                    break
            t_decode = (time.time() - t_dec) * 1000
            n_dec = max(1, len(gen_tokens) - 1)
            tps = n_dec / (t_decode / 1000) if t_decode > 0 else 0
            return gen_tokens, first_hidden, first_recs, t_prefill, t_decode, tps

    return DedupWrapper(ffn_models, embed_lmhead_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _print_ablation_table(results: List[VariantResult], pt_refs, tokenizer):
    """Print a formatted ablation table."""
    # Group by variant
    by_variant = {}
    for r in results:
        by_variant.setdefault(r.variant_id, []).append(r)

    header = (
        f"{'Variant':<40} "
        f"{'1st-tok':>7} {'Early%':>7} {'RepOn':>6} {'#Rep':>5} "
        f"{'ANE':>4} {'Fall':>6} "
        f"{'PF(ms)':>8} {'DC(ms)':>8} {'tok/s':>7} "
        f"{'Export':>7}"
    )
    print(f"\n{header}")
    print("─" * len(header))

    for vid in VARIANTS:
        if vid not in by_variant:
            continue
        rlist = by_variant[vid]
        # Average metrics across prompts
        first_tok_rate = sum(1 for r in rlist if r.first_token_match) / len(rlist)
        early_agree = sum(r.early_token_agreement for r in rlist) / len(rlist)
        rep_onsets = [r.repetition_onset for r in rlist if r.repetition_onset >= 0]
        avg_rep = (sum(rep_onsets) / len(rep_onsets)) if rep_onsets else -1
        n_rep = len(rep_onsets)  # number of prompts with repetition
        ane_ok = all(r.loadable_cpu_and_ne for r in rlist)
        fallbacks = set(r.fallback_signs for r in rlist)
        fallback_str = "none" if fallbacks == {"none"} else ",".join(sorted(fallbacks))
        pf_avg = sum(r.prefill_latency_ms for r in rlist if r.prefill_latency_ms > 0) / max(1, sum(1 for r in rlist if r.prefill_latency_ms > 0))
        dc_avg = sum(r.decode_latency_ms for r in rlist if r.decode_latency_ms > 0) / max(1, sum(1 for r in rlist if r.decode_latency_ms > 0))
        tps_avg = sum(r.tokens_per_sec for r in rlist if r.tokens_per_sec > 0) / max(1, sum(1 for r in rlist if r.tokens_per_sec > 0))
        export_s = rlist[0].export_time_s

        label = VARIANTS[vid]["label"][:38]
        print(
            f"  {label:<38} "
            f"{first_tok_rate:>6.0%} {early_agree:>6.0%} "
            f"{avg_rep:>6.0f} {n_rep:>4}/{len(rlist)} "
            f"{'✓' if ane_ok else '✗':>4} {fallback_str:>6} "
            f"{pf_avg:>8.0f} {dc_avg:>8.0f} {tps_avg:>7.1f} "
            f"{export_s:>6.0f}s"
        )

    # Per-prompt details
    print(f"\n\n{'─' * 80}")
    print("PER-PROMPT DETAIL")
    print(f"{'─' * 80}")
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {pi}: {prompt}")
        print(f"  PyTorch ref: {pt_refs[pi]['text'][:120]}")
        print(f"  {'Variant':<35} {'1st':>4} {'Erly%':>6} {'Rep':>5} {'tok/s':>7}  Generated text")
        print(f"  {'─' * 100}")
        for vid in VARIANTS:
            if vid not in by_variant:
                continue
            matches = [r for r in by_variant[vid] if r.prompt_idx == pi]
            if not matches:
                continue
            r = matches[0]
            label = VARIANTS[vid]["label"][:33]
            print(
                f"  {label:<35} "
                f"{'Y' if r.first_token_match else 'N':>4} "
                f"{r.early_token_agreement:>5.0%} "
                f"{r.repetition_onset:>5} "
                f"{r.tokens_per_sec:>7.1f}  "
                f"{r.generated_text[:80]}"
            )


def _print_recommendation(results: List[VariantResult]):
    """Print the final recommendation."""
    by_variant = {}
    for r in results:
        by_variant.setdefault(r.variant_id, []).append(r)

    print(f"\n{'=' * 70}")
    print("  RECOMMENDATION")
    print(f"{'=' * 70}")

    # Score each variant: quality + speed
    scores = {}
    for vid, rlist in by_variant.items():
        if not all(r.loadable_cpu_and_ne for r in rlist):
            scores[vid] = -999  # not loadable
            continue
        first_tok = sum(1 for r in rlist if r.first_token_match) / len(rlist)
        early = sum(r.early_token_agreement for r in rlist) / len(rlist)
        # Repetition penalty: linear scale based on how early repetition starts
        # No repetition → 0 penalty; earlier onset → higher penalty
        rep_scores = []
        for r in rlist:
            if r.repetition_onset < 0:
                rep_scores.append(0.0)  # no repetition
            else:
                # Penalize proportionally: rep at token 10/40 → 0.75 penalty, at 35/40 → 0.125
                rep_scores.append(max(0.0, 1.0 - r.repetition_onset / 60.0))
        avg_rep_penalty = sum(rep_scores) / max(len(rep_scores), 1)
        tps = sum(r.tokens_per_sec for r in rlist if r.tokens_per_sec > 0)
        tps_avg = tps / max(1, sum(1 for r in rlist if r.tokens_per_sec > 0))

        # Quality score (0-100): repetition gets heavy weight since it's the main signal
        q = first_tok * 25 + early * 25 + (1 - avg_rep_penalty) * 50
        # Speed score (normalize by baseline)
        scores[vid] = (q, tps_avg)

    print("\n  Quality/Speed ranking:")
    ranked = sorted(scores.items(), key=lambda x: x[1][0] if isinstance(x[1], tuple) else x[1], reverse=True)
    for i, (vid, score) in enumerate(ranked):
        if isinstance(score, tuple):
            q, tps = score
            label = VARIANTS[vid]["label"]
            print(f"    {i+1}. {label:<50} Q={q:.1f}  tok/s={tps:.1f}")
        else:
            label = VARIANTS[vid]["label"]
            print(f"    {i+1}. {label:<50} NOT LOADABLE ON ANE")

    best_vid = ranked[0][0] if ranked else "V0_baseline_fp16"
    fp32_vid = "V7_full_fp32"
    baseline_vid = "V0_baseline_fp16"

    print(f"\n  Best variant: {VARIANTS[best_vid]['label']}")
    if best_vid == fp32_vid:
        print("  → Full FP32 compute is the best. Consider using --fp32-compute globally.")
    elif best_vid == baseline_vid:
        print("  → Baseline FP16 is already optimal. No selective FP32 needed.")
    else:
        skip = VARIANTS[best_vid]["skip_ops"]
        print(f"  → Selective FP32 improves quality. Recommended skip_ops: {skip}")
        print(f"  → Apply via: pipeline.set_options('common::add_fp16_cast', " +
              f"{{'skip_ops_by_type': '{skip}'}})")

    # Compare selective vs full FP32
    if best_vid != fp32_vid and fp32_vid in scores and isinstance(scores[fp32_vid], tuple):
        best_q = scores[best_vid][0] if isinstance(scores[best_vid], tuple) else 0
        best_tps = scores[best_vid][1] if isinstance(scores[best_vid], tuple) else 0
        fp32_q = scores[fp32_vid][0]
        fp32_tps = scores[fp32_vid][1]
        print(f"\n  Selective FP32 vs Full FP32:")
        print(f"    Quality: {best_q:.1f} vs {fp32_q:.1f} ({'better' if best_q >= fp32_q else 'worse'})")
        print(f"    Speed:   {best_tps:.1f} vs {fp32_tps:.1f} tok/s "
              f"({'faster' if best_tps >= fp32_tps else 'slower'})")
        if best_q >= fp32_q * 0.95 and best_tps > fp32_tps:
            print(f"  → RECOMMENDATION: Use selective FP32 (better speed/quality tradeoff)")
        elif fp32_q > best_q:
            print(f"  → RECOMMENDATION: Use full FP32 compute (better quality)")
        else:
            print(f"  → RECOMMENDATION: Use selective FP32 (comparable quality, better speed)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Targeted selective-FP32 ablation for Qwen3.5-4B on ANE")
    parser.add_argument("--model", default=os.path.join(_REPO_ROOT, "models", "Qwen__Qwen3.5-4B"),
                        help="Path to HuggingFace model")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer path (default: same as --model)")
    parser.add_argument("--baseline-dir",
                        default=os.path.join(_REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp16"),
                        help="Directory with baseline FP16 .mlpackage models")
    parser.add_argument("--fp32-dir",
                        default=os.path.join(_REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp32"),
                        help="Directory with full FP32 .mlpackage models")
    parser.add_argument("--output",
                        default=os.path.join(_REPO_ROOT, "tests", "dev", "selective_fp32_results"),
                        help="Output directory for experiment results")
    parser.add_argument("--max-tokens", type=int, default=60,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export phase (use existing models)")
    parser.add_argument("--only-variants", type=str, default=None,
                        help="Comma-separated variant IDs to test (e.g. V0_baseline_fp16,V1_exp_only)")
    args = parser.parse_args()

    if args.only_variants:
        selected = set(args.only_variants.split(","))
        global VARIANTS
        VARIANTS = {k: v for k, v in VARIANTS.items() if k in selected}
        print(f"Testing only: {list(VARIANTS.keys())}")

    print("=" * 70)
    print("  Selective-FP32 Ablation Experiment — Qwen3.5-4B on ANE")
    print(f"  Model: {args.model}")
    print(f"  Baseline: {args.baseline_dir}")
    print(f"  FP32: {args.fp32_dir}")
    print(f"  Output: {args.output}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Variants: {len(VARIANTS)}")
    print("=" * 70)

    run_experiment(args)


if __name__ == "__main__":
    main()
