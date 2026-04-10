#!/usr/bin/env python3
"""Per-op FP16 sensitivity analysis for Qwen3.5-4B on ANE.

For each MIL op type individually, exports the full 9-chunk model with just
that one op type kept in FP32 (all others in FP16). Then runs end-to-end
inference through ALL chunks and compares the **whole-model output** against
PyTorch reference using:
  - Hidden-state cosine similarity (pre-lm_head, after all chunks + final norm)
  - Logit cosine similarity (full output distribution)
  - KL divergence of output distribution
  - Top-1 / Top-5 agreement
  - Repetition onset tracking

This gives a per-op sensitivity ranking showing exactly which ops lose the
most precision when cast to FP16.

Usage:
    python tests/dev/op_sensitivity_experiment.py \\
        --model models/Qwen__Qwen3.5-4B \\
        --baseline-dir qwen3_5_stable_lut4ffn_lut6em_fp16 \\
        --fp32-dir qwen3_5_stable_lut4ffn_lut6em_fp32 \\
        --output tests/dev/op_sensitivity_results \\
        --max-tokens 40
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import special as sp_special

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import coremltools as ct
from transformers import AutoTokenizer

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES, FFN_PER_CHANNEL
from anemll.models.qwen3_5_model import (
    Qwen35Config, Qwen35ForCausalLM, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Op types to test individually
# ─────────────────────────────────────────────────────────────────────────────
# Each will be tested as: that single op kept FP32, everything else FP16.
CANDIDATE_OPS = [
    "exp",          # recurrent gate g_t = exp(g)
    "mul",          # state*g_t, k_t*delta, q_t*state, attention scaling
    "add",          # state accumulation, residual connections, mask addition
    "sub",          # v_t - kv_mem delta computation
    "reduce_sum",   # dot products in recurrence
    "log",          # softplus: log(1+exp(-|x|))
    "abs",          # softplus: exp(-|x|)
    "relu",         # softplus: relu(x) + ...
    "matmul",       # full-attention QK^T, attn@V
    "softmax",      # full-attention softmax
    "conv",         # Q/K/V/O projections (Conv2d with kernel=1)
]

PROMPTS = [
    "What is a stack in computer science?",
    "教我做红烧鱼",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpResult:
    op_type: str                    # "exp", "mul", etc. or "baseline_fp16" / "full_fp32"
    prompt_idx: int = 0
    # Whole-model metrics (averaged over decode steps)
    hidden_cosine_prefill: float = -1.0   # cos(hidden, ref_hidden) at end of prefill
    hidden_cosine_decode_avg: float = -1.0  # average cos over decode steps
    hidden_cosine_decode_min: float = -1.0  # worst cos over decode steps
    logit_cosine_prefill: float = -1.0    # cos(logits, ref_logits) at end of prefill
    logit_cosine_decode_avg: float = -1.0
    kl_divergence_prefill: float = -1.0   # KL(ref || variant) at prefill end
    kl_divergence_decode_avg: float = -1.0
    top1_match_rate: float = -1.0         # fraction of decode steps with same argmax
    top5_match_rate: float = -1.0         # fraction where ref top-1 is in variant top-5
    # Generation quality
    first_token_match: bool = False
    repetition_onset: int = -1
    generated_text: str = ""
    tokens_per_sec: float = -1.0
    export_time_s: float = -1.0
    loadable: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def kl_divergence(ref_logits: np.ndarray, var_logits: np.ndarray) -> float:
    """KL(ref || variant) using log-softmax for numerical stability."""
    ref = ref_logits.flatten().astype(np.float64)
    var = var_logits.flatten().astype(np.float64)
    # Subtract max for numerical stability
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
    """Check if ref's argmax is in variant's top-k."""
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
    """Extract full logit vector from lm_head output dict."""
    if "logits" in lm_out:
        return lm_out["logits"].flatten().astype(np.float32)
    split_keys = sorted([k for k in lm_out if k.startswith("logits")],
                        key=lambda k: int(k.replace("logits", "")))
    if split_keys:
        return np.concatenate([lm_out[k].flatten() for k in split_keys]).astype(np.float32)
    raise KeyError(f"Cannot find logits in lm_head output: {list(lm_out.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Export (reused from selective_fp32_experiment.py with minor changes)
# ─────────────────────────────────────────────────────────────────────────────

def export_single_op_variant(
    model: Qwen35ForCausalLM,
    op_type: str,
    out_dir: str,
) -> Tuple[List[str], float]:
    """Export all 9 chunks with a single op type kept in FP32.

    Returns (list_of_chunk_paths, total_export_time).
    """
    variant_dir = os.path.join(out_dir, f"op_{op_type}")
    os.makedirs(variant_dir, exist_ok=True)

    pipeline = ct.PassPipeline.DEFAULT
    pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": op_type})

    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=None,
        per_channel=FFN_PER_CHANNEL,
        compute_precision="float16",
    )

    paths = []
    t_total = time.time()

    for chunk_idx in range(NUM_CHUNKS):
        sl, el = CHUNK_RANGES[chunk_idx]
        dec_path = os.path.join(variant_dir, f"ffn_op_{op_type}_chunk{chunk_idx}.mlpackage")

        if os.path.exists(dec_path):
            print(f"      [skip] {dec_path} exists")
            paths.append(dec_path)
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
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            pass_pipeline=pipeline,
        )

        mlmodel = ct.convert(traced, **convert_kwargs)
        mlmodel.save(dec_path)
        elapsed = time.time() - t0
        del mlmodel, traced, wrapper
        gc.collect()
        print(f"      Saved chunk {chunk_idx} ({elapsed:.1f}s)")
        paths.append(dec_path)

    total_time = time.time() - t_total
    return paths, total_time


# ─────────────────────────────────────────────────────────────────────────────
# Inference engine with hidden+logit capture
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """Loads 9 chunks + embed/lmhead and runs step-by-step inference,
    capturing hidden states and full logits at every decode step."""

    def __init__(self, chunk_paths, embed_lmhead_path, compute_unit=ct.ComputeUnit.CPU_AND_NE,
                 mode="separate", function_name=None):
        self.compute_unit = compute_unit
        self.loadable = True
        self.mode = mode

        # Embed + lmhead
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

        # Parse input shapes
        self.inp_maps = []
        for ci in range(len(self.ffns)):
            spec = self.ffns[ci].get_spec()
            imap = {}
            # Try function-specific inputs first
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
        """Single-token step through the full pipeline.

        Returns (next_token_id, hidden_state, full_logits).
        hidden_state: the pre-lm_head output after all 9 chunks.
        full_logits: the full vocabulary logit vector.
        """
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

        # LM head → full logits
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        logits = _get_full_logits(lm_out)
        next_tok = int(np.argmax(logits))
        return next_tok, hidden.copy(), logits

    def generate_with_traces(self, token_ids: List[int], max_gen: int, stop_ids: set):
        """Prefill + decode, capturing hidden states and logits at every step.

        Returns dict with:
          prefill_hidden, prefill_logits: at end of prompt
          decode_hiddens, decode_logits: list per decode step
          gen_tokens: generated token IDs
          tps: tokens per second
        """
        if not self.loadable:
            return None

        self.reset_all()
        t0 = time.time()

        # Prefill
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

        # Decode
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
# PyTorch reference with trace capture
# ─────────────────────────────────────────────────────────────────────────────

def pytorch_reference_with_traces(model, tokenizer, prompt, max_gen, stop_ids):
    """Run PyTorch inference capturing hidden+logits at every step."""
    model.eval()
    msgs = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=True
    )
    token_list = input_ids[0].tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)

    # Reset states
    model.model.kv_cache_0.zero_()
    if hasattr(model.model, 'linear_conv_state'):
        model.model.linear_conv_state.zero_()
    if hasattr(model.model, 'linear_recurrent_state'):
        model.model.linear_recurrent_state.zero_()

    def _forward_one(tid, pos):
        """Single-token forward, returns (hidden, logits, next_tok)."""
        inp = torch.tensor([[tid]], dtype=torch.int32, device=TEST_DEVICE)
        pos_t = torch.tensor([pos], dtype=torch.int32, device=TEST_DEVICE)
        mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=torch.float16, device=TEST_DEVICE)
        mask[:, :, :, :pos + 1] = 0
        hidden = model.model.embed_tokens(inp).to(MODEL_DTYPE)
        for li in range(len(model.model.layers)):
            hidden = model.model._process_layer_regular(li, hidden, pos_t, mask, pos_t)
        hidden = model.model.norm(hidden)
        # LM head
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
        # Prefill
        for i, tid in enumerate(token_list):
            hidden, logits, next_tok = _forward_one(tid, i)
        result["prefill_hidden"] = hidden
        result["prefill_logits"] = logits

        # Decode
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
    """Compare reference and variant traces, computing all metrics."""
    metrics = {}

    # Prefill hidden cosine
    metrics["hidden_cos_prefill"] = cosine_sim(
        ref_trace["prefill_hidden"], var_trace["prefill_hidden"])

    # Prefill logit cosine
    metrics["logit_cos_prefill"] = cosine_sim(
        ref_trace["prefill_logits"], var_trace["prefill_logits"])

    # Prefill KL
    metrics["kl_prefill"] = kl_divergence(
        ref_trace["prefill_logits"], var_trace["prefill_logits"])

    # Decode step metrics
    n_steps = min(len(ref_trace["decode_hiddens"]), len(var_trace["decode_hiddens"]))
    n_tok_steps = min(len(ref_trace["decode_logits"]), len(var_trace["decode_logits"]))

    h_coss = []
    l_coss = []
    kls = []
    top1_matches = 0
    top5_matches = 0

    for i in range(max(n_steps, n_tok_steps)):
        if i < n_steps:
            h_coss.append(cosine_sim(
                ref_trace["decode_hiddens"][i], var_trace["decode_hiddens"][i]))
        if i < n_tok_steps:
            l_coss.append(cosine_sim(
                ref_trace["decode_logits"][i], var_trace["decode_logits"][i]))
            kls.append(kl_divergence(
                ref_trace["decode_logits"][i], var_trace["decode_logits"][i]))
            ref_top1 = int(np.argmax(ref_trace["decode_logits"][i]))
            var_top1 = int(np.argmax(var_trace["decode_logits"][i]))
            if ref_top1 == var_top1:
                top1_matches += 1
            if top_k_in(ref_trace["decode_logits"][i], var_trace["decode_logits"][i], k=5):
                top5_matches += 1

    metrics["hidden_cos_decode_avg"] = float(np.mean(h_coss)) if h_coss else -1.0
    metrics["hidden_cos_decode_min"] = float(np.min(h_coss)) if h_coss else -1.0
    metrics["logit_cos_decode_avg"] = float(np.mean(l_coss)) if l_coss else -1.0
    metrics["kl_decode_avg"] = float(np.mean(kls)) if kls else -1.0
    metrics["top1_match_rate"] = top1_matches / max(n_tok_steps, 1)
    metrics["top5_match_rate"] = top5_matches / max(n_tok_steps, 1)

    # Token-level
    ref_first = ref_trace["gen_tokens"][0] if ref_trace["gen_tokens"] else -1
    var_first = var_trace["gen_tokens"][0] if var_trace["gen_tokens"] else -1
    metrics["first_token_match"] = ref_first == var_first

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def _save_incremental(results: List[OpResult], output_dir: str):
    """Save results incrementally after each op completes."""
    path = os.path.join(output_dir, "op_sensitivity_results.json")
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)


def _load_existing_results(output_dir: str) -> List[OpResult]:
    """Load previously saved results for resume support."""
    path = os.path.join(output_dir, "op_sensitivity_results.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        results = []
        for d in data:
            r = OpResult(op_type=d["op_type"])
            for k, v in d.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            results.append(r)
        return results
    except Exception as e:
        print(f"  Warning: could not load existing results: {e}")
        return []


def run_experiment(args):
    os.makedirs(args.output, exist_ok=True)

    # Load tokenizer
    tok_path = args.tokenizer or args.model
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    stop_ids = _build_stop_ids(tokenizer)

    # Load PyTorch model
    print(f"Loading PyTorch model from {args.model}...")
    cfg = Qwen35Config.from_json(os.path.join(args.model, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    pt_model = Qwen35ForCausalLM(cfg)
    assert pt_model.load_pretrained_weights(args.model), "Failed to load weights"
    pt_model.eval()
    for p in pt_model.parameters():
        p.requires_grad = False

    # ── Phase 1: PyTorch references (with hidden+logit traces) ──
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

    # Find embed_lmhead
    embed_lmhead = os.path.join(args.fp32_dir, "embed_lmhead_combined.mlpackage")
    if not os.path.exists(embed_lmhead):
        embed_lmhead = os.path.join(args.baseline_dir, "embed_lmhead_combined.mlpackage")
    assert os.path.exists(embed_lmhead), f"Cannot find embed_lmhead_combined.mlpackage"
    print(f"\nUsing embed+lmhead: {embed_lmhead}")

    # Determine which ops to test
    ops_to_test = args.ops.split(",") if args.ops else CANDIDATE_OPS
    print(f"\nOps to test: {ops_to_test}")

    # ── Phase 2: Export per-op variants ──
    print("\n" + "=" * 70)
    print("  PHASE 2: Per-Op Export & Inference")
    print("=" * 70)

    # Track all results across: baseline_fp16, each op, full_fp32
    all_results: List[OpResult] = []
    variant_order = ["baseline_fp16"] + ops_to_test + ["full_fp32"]

    # Resume support: load existing results and determine which ops to skip
    completed_ops = set()
    if args.resume:
        existing = _load_existing_results(args.output)
        if existing:
            all_results.extend(existing)
            # An op is complete if it has results for all prompts
            from collections import Counter
            op_counts = Counter(r.op_type for r in existing if r.loadable)
            for op, cnt in op_counts.items():
                if cnt >= len(PROMPTS):
                    completed_ops.add(op)
            print(f"  Resuming: {len(existing)} results loaded, "
                  f"completed ops: {sorted(completed_ops)}")

    for variant_label in variant_order:
        print(f"\n{'─' * 60}")
        print(f"  OP: {variant_label}")
        print(f"{'─' * 60}")

        # Skip if already completed in a previous run
        if variant_label in completed_ops:
            print(f"    [SKIP] Already completed in previous run")
            continue

        chunk_paths = None
        export_time = 0.0
        function_name = None
        mode = "separate"

        if variant_label == "baseline_fp16":
            # Reuse existing FP16 baseline
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
            # Reuse existing FP32
            dedup = os.path.join(args.fp32_dir, "combined_LUT4_dedup")
            if os.path.isdir(dedup):
                chunk_paths = [os.path.join(dedup, f"chunk{ci}.mlpackage") for ci in range(NUM_CHUNKS)]
                function_name = "infer"
                mode = "dedup"
            else:
                chunk_paths = [os.path.join(args.fp32_dir, f"ffn_LUT4_chunk{ci}.mlpackage")
                               for ci in range(NUM_CHUNKS)]
            print(f"    Using existing FP32: {chunk_paths[0]}")

        else:
            # Export variant with this single op in FP32
            print(f"    Exporting 9 chunks with {variant_label} → FP32...")
            try:
                chunk_paths, export_time = export_single_op_variant(
                    pt_model, variant_label, args.output)
                print(f"    Total export: {export_time:.1f}s")
            except Exception as e:
                print(f"    *** Export FAILED for {variant_label}: {e}")
                # Cleanup partial exports
                variant_dir = os.path.join(args.output, f"op_{variant_label}")
                if os.path.isdir(variant_dir):
                    shutil.rmtree(variant_dir)
                for pi in range(len(PROMPTS)):
                    all_results.append(OpResult(op_type=variant_label, prompt_idx=pi))
                _save_incremental(all_results, args.output)
                continue
            mode = "separate"

        # All chunk_paths must exist
        if not chunk_paths or not all(os.path.exists(p) for p in chunk_paths):
            print(f"    *** Missing chunk files, skipping")
            for pi in range(len(PROMPTS)):
                all_results.append(OpResult(op_type=variant_label, prompt_idx=pi))
            continue

        # Load & run inference
        try:
            engine = InferenceEngine(chunk_paths, embed_lmhead, mode=mode,
                                     function_name=function_name)
        except Exception as e:
            print(f"    *** Engine load failed: {e}")
            for pi in range(len(PROMPTS)):
                all_results.append(OpResult(op_type=variant_label, prompt_idx=pi))
            continue

        if not engine.loadable:
            print(f"    *** Not loadable on ANE")
            for pi in range(len(PROMPTS)):
                all_results.append(OpResult(op_type=variant_label, prompt_idx=pi, loadable=False))
            engine.cleanup()
            continue

        for pi, prompt in enumerate(PROMPTS):
            print(f"\n    Prompt {pi}: {prompt[:50]}...")
            ref = pt_refs[pi]

            var_trace = engine.generate_with_traces(
                ref["token_list"], args.max_tokens, stop_ids)

            if var_trace is None:
                all_results.append(OpResult(op_type=variant_label, prompt_idx=pi))
                continue

            # Compare
            metrics = compare_traces(ref, var_trace)
            text = tokenizer.decode(var_trace["gen_tokens"], skip_special_tokens=True)
            rep_onset = detect_repetition_onset(var_trace["gen_tokens"])

            r = OpResult(
                op_type=variant_label,
                prompt_idx=pi,
                hidden_cosine_prefill=metrics["hidden_cos_prefill"],
                hidden_cosine_decode_avg=metrics["hidden_cos_decode_avg"],
                hidden_cosine_decode_min=metrics["hidden_cos_decode_min"],
                logit_cosine_prefill=metrics["logit_cos_prefill"],
                logit_cosine_decode_avg=metrics["logit_cos_decode_avg"],
                kl_divergence_prefill=metrics["kl_prefill"],
                kl_divergence_decode_avg=metrics["kl_decode_avg"],
                top1_match_rate=metrics["top1_match_rate"],
                top5_match_rate=metrics["top5_match_rate"],
                first_token_match=metrics["first_token_match"],
                repetition_onset=rep_onset,
                generated_text=text[:300],
                tokens_per_sec=var_trace["tps"],
                export_time_s=export_time,
                loadable=True,
            )
            all_results.append(r)

            print(f"      h_cos_pf={metrics['hidden_cos_prefill']:.6f} "
                  f"h_cos_dec={metrics['hidden_cos_decode_avg']:.6f} "
                  f"l_cos_pf={metrics['logit_cos_prefill']:.6f}")
            print(f"      KL_pf={metrics['kl_prefill']:.4f} "
                  f"KL_dec={metrics['kl_decode_avg']:.4f} "
                  f"top1={metrics['top1_match_rate']:.0%} top5={metrics['top5_match_rate']:.0%}")
            print(f"      rep={rep_onset} tps={var_trace['tps']:.1f} "
                  f"1st={metrics['first_token_match']}")
            print(f"      Gen: {text[:100]}")

        engine.cleanup()
        gc.collect()

        # Cleanup exported chunks (not baselines)
        if variant_label not in ("baseline_fp16", "full_fp32"):
            variant_dir = os.path.join(args.output, f"op_{variant_label}")
            if os.path.isdir(variant_dir):
                print(f"    Cleaning up {variant_dir}...")
                shutil.rmtree(variant_dir)

        # Save incremental results after each op
        _save_incremental(all_results, args.output)
        print(f"    [Saved {len(all_results)} results so far]")

    # Free PyTorch model
    del pt_model
    gc.collect()

    # ── Phase 3: Results ──
    print("\n" + "=" * 70)
    print("  PHASE 3: Op Sensitivity Ranking")
    print("=" * 70)

    _print_sensitivity_table(all_results, tokenizer)

    # Save
    results_path = os.path.join(args.output, "op_sensitivity_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


def _print_sensitivity_table(results: List[OpResult], tokenizer):
    """Print sensitivity ranking sorted by hidden cosine (lower = more sensitive)."""

    # Group by op
    by_op = {}
    for r in results:
        by_op.setdefault(r.op_type, []).append(r)

    # Compute average metrics per op
    rows = []
    for op, rlist in by_op.items():
        valid = [r for r in rlist if r.loadable and r.hidden_cosine_prefill >= 0]
        if not valid:
            rows.append({
                "op": op, "h_cos_pf": -1, "h_cos_dec": -1, "h_cos_min": -1,
                "l_cos_pf": -1, "l_cos_dec": -1,
                "kl_pf": -1, "kl_dec": -1,
                "top1": -1, "top5": -1,
                "rep_count": 0, "tps": 0, "broken": True,
            })
            continue

        n = len(valid)
        rep_count = sum(1 for r in valid if r.repetition_onset >= 0)
        rows.append({
            "op": op,
            "h_cos_pf": sum(r.hidden_cosine_prefill for r in valid) / n,
            "h_cos_dec": sum(r.hidden_cosine_decode_avg for r in valid if r.hidden_cosine_decode_avg >= 0) / max(1, sum(1 for r in valid if r.hidden_cosine_decode_avg >= 0)),
            "h_cos_min": min(r.hidden_cosine_decode_min for r in valid if r.hidden_cosine_decode_min >= 0) if any(r.hidden_cosine_decode_min >= 0 for r in valid) else -1,
            "l_cos_pf": sum(r.logit_cosine_prefill for r in valid) / n,
            "l_cos_dec": sum(r.logit_cosine_decode_avg for r in valid if r.logit_cosine_decode_avg >= 0) / max(1, sum(1 for r in valid if r.logit_cosine_decode_avg >= 0)),
            "kl_pf": sum(r.kl_divergence_prefill for r in valid) / n,
            "kl_dec": sum(r.kl_divergence_decode_avg for r in valid if r.kl_divergence_decode_avg >= 0) / max(1, sum(1 for r in valid if r.kl_divergence_decode_avg >= 0)),
            "top1": sum(r.top1_match_rate for r in valid) / n,
            "top5": sum(r.top5_match_rate for r in valid) / n,
            "rep_count": rep_count,
            "tps": sum(r.tokens_per_sec for r in valid if r.tokens_per_sec > 0) / max(1, sum(1 for r in valid if r.tokens_per_sec > 0)),
            "broken": False,
        })

    # Sort: baseline_fp16 first, full_fp32 last, ops sorted by hidden cosine (ascending = most divergent first)
    baselines = [r for r in rows if r["op"] in ("baseline_fp16", "full_fp32")]
    ops = [r for r in rows if r["op"] not in ("baseline_fp16", "full_fp32")]
    ops.sort(key=lambda r: r["h_cos_dec"] if r["h_cos_dec"] >= 0 else 2.0)  # broken ops last

    # Print header
    print(f"\n  {'Op':<16} {'h_cos_pf':>9} {'h_cos_dec':>9} {'h_cos_min':>9} "
          f"{'l_cos_pf':>9} {'KL_pf':>8} {'KL_dec':>8} "
          f"{'top1%':>6} {'top5%':>6} {'#Rep':>4} {'tok/s':>6}")
    print("  " + "─" * 105)

    # Print baselines
    for r in baselines:
        if r["op"] == "baseline_fp16":
            _print_row(r)
    # Print ops sorted by ascending hidden cos (most sensitive first)
    for r in ops:
        _print_row(r)
    # Print full FP32
    for r in baselines:
        if r["op"] == "full_fp32":
            _print_row(r)

    # Print sensitivity ranking summary
    valid_ops = [r for r in ops if not r["broken"]]
    if valid_ops:
        print(f"\n  SENSITIVITY RANKING (by avg decode hidden cosine vs PyTorch ref):")
        print(f"  Most sensitive (lowest cosine = most divergent from FP32 reference):")
        for i, r in enumerate(valid_ops):
            marker = "⚠ " if r["rep_count"] > 0 else "  "
            print(f"    {i+1}. {marker}{r['op']:<14} h_cos_dec={r['h_cos_dec']:.6f}  "
                  f"KL={r['kl_dec']:.4f}  top1={r['top1']:.0%}  reps={r['rep_count']}/3")


def _print_row(r):
    if r["broken"]:
        print(f"  {r['op']:<16} {'BROKEN':>9}")
        return
    def fmt(v, w=9, d=6):
        return f"{v:{w}.{d}f}" if v >= 0 else f"{'N/A':>{w}}"
    def fmtp(v, w=6):
        return f"{v:{w}.0%}" if v >= 0 else f"{'N/A':>{w}}"
    print(f"  {r['op']:<16} {fmt(r['h_cos_pf'])} {fmt(r['h_cos_dec'])} {fmt(r['h_cos_min'])} "
          f"{fmt(r['l_cos_pf'])} {fmt(r['kl_pf'], 8, 4)} {fmt(r['kl_dec'], 8, 4)} "
          f"{fmtp(r['top1'])} {fmtp(r['top5'])} {r['rep_count']:>4} {r['tps']:>6.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-op FP16 sensitivity analysis")
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--baseline-dir", required=True, help="FP16 baseline directory")
    parser.add_argument("--fp32-dir", required=True, help="FP32 reference directory")
    parser.add_argument("--output", default="tests/dev/op_sensitivity_results")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--ops", type=str, default=None,
                        help="Comma-separated op types to test (default: all 11)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run, skip ops with saved results")
    args = parser.parse_args()
    print(f"  Model: {args.model}")
    print(f"  Baseline: {args.baseline_dir}")
    print(f"  FP32 ref: {args.fp32_dir}")
    print(f"  Max tokens: {args.max_tokens}")
    run_experiment(args)


if __name__ == "__main__":
    main()
