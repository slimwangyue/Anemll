#!/usr/bin/env python3
"""Per-layer FP16 knockout inside chunk 2 (layers 7-10) of Qwen3.5-4B on ANE.

Start from full FP32 reference model and re-export chunk 2 with per-layer
precision control.  For each variant, specific layers are cast to FP16 while
the rest stay FP32.  The exported chunk is hot-swapped into the FP32 engine
and tested on 3 prompts.

Variants:
  chunk2_all_fp32       — reference (all layers 7-10 in FP32)
  layer7_fp16_only      — only layer 7 (F layer) cast to FP16
  layer8_fp16_only      — only layer 8 (L layer) cast to FP16
  layer9_fp16_only      — only layer 9 (L layer) cast to FP16
  layer10_fp16_only     — only layer 10 (L layer) cast to FP16
  layer7+8_fp16         — layers 7+8 to FP16
  layer8+9_fp16         — layers 8+9 to FP16
  layer9+10_fp16        — layers 9+10 to FP16
  chunk2_all_fp16       — all layers 7-10 to FP16 (= existing FP16 chunk)

Architecture: chunk2 = layers 7-10 = [FLLL]
  layer 7 = F (full-attention)
  layer 8 = L (linear attention / recurrence)
  layer 9 = L
  layer 10 = L

Usage:
    python tests/dev/chunk2_layer_knockout.py \\
        --model models/Qwen__Qwen3.5-4B \\
        --fp16-dir qwen3_5_stable_lut4ffn_lut6em_fp16 \\
        --fp32-dir qwen3_5_stable_lut4ffn_lut6em_fp32 \\
        --output tests/dev/chunk2_knockout_results \\
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
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

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

# Chunk 2 specifics
CHUNK2_IDX = 2
CHUNK2_START, CHUNK2_END = CHUNK_RANGES[CHUNK2_IDX]  # (7, 11)
CHUNK2_LAYERS = list(range(CHUNK2_START, CHUNK2_END))  # [7, 8, 9, 10]
# Layer 7 = F (full-attention), layers 8-10 = L (linear attention)

PROMPTS = [
    "What is a stack in computer science?",
    "教我做红烧鱼",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
]

BNNS_CACHE_DIR = os.path.expanduser(
    "~/Library/Caches/com.apple.python3/com.apple.e5rt.e5bundlecache")

# Regex to extract global layer index from const op names
# In the MIL graph after ct.convert(), const names use dots:
#   model.model.layers.10.self_attn.out_proj.weight
# Earlier MIL dumps showed underscores, but the FP16ComputePrecision callback
# sees the original dot-separated names from PyTorch tracing.
_LAYER_CONST_RE = re.compile(r'model\.model\.layers\.(\d+)\.')


def clear_bnns_cache():
    if os.path.isdir(BNNS_CACHE_DIR):
        for entry in os.listdir(BNNS_CACHE_DIR):
            path = os.path.join(BNNS_CACHE_DIR, entry)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError:
                pass
        print("    [Cleared BNNS cache]")


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer op_selector
# ─────────────────────────────────────────────────────────────────────────────

def _find_layer_from_inputs(op, max_depth=6) -> Optional[int]:
    """BFS walk up an op's input tree to find a const with a layer-specific name.
    Returns the global layer index, or None if not found.
    """
    visited = set()
    queue = [op]
    depth = 0
    while queue and depth < max_depth:
        next_queue = []
        for cur_op in queue:
            if id(cur_op) in visited:
                continue
            visited.add(id(cur_op))
            # Check this op's name for layer pattern
            if cur_op.op_type == 'const':
                m = _LAYER_CONST_RE.search(cur_op.name)
                if m:
                    return int(m.group(1))
            # Walk up through inputs
            for inp_name, inp_var in cur_op.inputs.items():
                if hasattr(inp_var, 'op') and inp_var.op is not None:
                    if id(inp_var.op) not in visited:
                        next_queue.append(inp_var.op)
        queue = next_queue
        depth += 1
    return None


def make_per_layer_selector(fp16_layers: Set[int]):
    """Create an op_selector that casts specific layers to FP16, keeps rest FP32.
    
    For a chunk starting at global layer `start`, this selector:
    - Returns True (cast to FP16) for ops belonging to layers in fp16_layers
    - Returns False (keep FP32) for ops in other layers
    - Const ops: True if their layer is in fp16_layers, else True anyway
      (consts are usually weights — FP16 cast of consts is generally fine, but
       we keep layer-specific consts in FP32 if their layer should be FP32)
    
    NOTE: If we cannot determine which layer an op belongs to (shared ops,
    graph infrastructure), we default to FP32 (conservative).
    """
    stats = {"fp16": 0, "fp32": 0, "unresolved_fp32": 0}

    def op_selector(op):
        # Const ops: check if they belong to a specific layer
        if op.op_type == 'const':
            m = _LAYER_CONST_RE.search(op.name)
            if m:
                layer_idx = int(m.group(1))
                if layer_idx in fp16_layers:
                    stats["fp16"] += 1
                    return True   # Cast const to FP16
                else:
                    stats["fp32"] += 1
                    return False  # Keep const in FP32
            # Non-layer consts (scalar consts, shape consts, etc.) — cast to FP16
            stats["fp16"] += 1
            return True

        # Non-const ops: trace inputs to find layer
        layer_idx = _find_layer_from_inputs(op)
        if layer_idx is not None:
            if layer_idx in fp16_layers:
                stats["fp16"] += 1
                return True   # Cast to FP16
            else:
                stats["fp32"] += 1
                return False  # Keep FP32
        
        # Cannot determine layer — keep FP32 (conservative)
        stats["unresolved_fp32"] += 1
        return False

    return op_selector, stats


def make_inverse_layer_selector(fp32_layers: Set[int]):
    """Create an op_selector that KEEPS specific layers in FP32, casts rest to FP16.
    
    This is the INVERSE approach: start from FP16 baseline and promote specific
    layers to FP32. Unresolved ops default to FP16 (matching FP16 baseline).
    This avoids ANE compilation failures from excessive FP32 islands.
    
    Returns True → cast to FP16, False → keep FP32.
    """
    stats = {"fp16": 0, "fp32": 0, "unresolved_fp16": 0}

    def op_selector(op):
        if op.op_type == 'const':
            m = _LAYER_CONST_RE.search(op.name)
            if m:
                layer_idx = int(m.group(1))
                if layer_idx in fp32_layers:
                    stats["fp32"] += 1
                    return False  # Keep FP32
                else:
                    stats["fp16"] += 1
                    return True   # Cast to FP16
            stats["fp16"] += 1
            return True

        layer_idx = _find_layer_from_inputs(op)
        if layer_idx is not None:
            if layer_idx in fp32_layers:
                stats["fp32"] += 1
                return False  # Keep FP32
            else:
                stats["fp16"] += 1
                return True   # Cast to FP16
        
        # Cannot determine layer — default to FP16 (baseline behavior)
        stats["unresolved_fp16"] += 1
        return True

    return op_selector, stats


# ─────────────────────────────────────────────────────────────────────────────
# Export chunk 2 with per-layer precision
# ─────────────────────────────────────────────────────────────────────────────

def export_chunk2_variant(
    model: Qwen35ForCausalLM,
    variant_name: str,
    fp16_layers: Set[int],
    out_dir: str,
    inverse: bool = False,
) -> Tuple[str, float, dict]:
    """Export chunk 2 with per-layer precision control.
    
    If inverse=False: fp16_layers cast to FP16, rest stays FP32 (from-FP32 approach)
    If inverse=True: fp16_layers is treated as fp32_layers — those layers STAY FP32,
                     rest goes to FP16 (from-FP16 approach, better for ANE)
    
    Returns (path, export_time, stats).
    """
    variant_dir = os.path.join(out_dir, "exported_chunks")
    os.makedirs(variant_dir, exist_ok=True)
    dec_path = os.path.join(variant_dir, f"chunk2_{variant_name}.mlpackage")
    
    if os.path.exists(dec_path):
        print(f"      [skip] {dec_path} exists")
        return dec_path, 0.0, {}

    print(f"      Exporting chunk2 [{variant_name}] — FP16 layers: {sorted(fp16_layers)}...")
    t0 = time.time()

    sl, el = CHUNK_RANGES[CHUNK2_IDX]
    local_num_layers = el - sl
    cfg = model.config

    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=None,
        per_channel=FFN_PER_CHANNEL,
        compute_precision="float16",
    )

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

    # Create per-layer op_selector
    if inverse:
        # inverse mode: fp16_layers is treated as fp32_layers
        op_sel, sel_stats = make_inverse_layer_selector(fp16_layers)
    else:
        op_sel, sel_stats = make_per_layer_selector(fp16_layers)
    compute_prec = FP16ComputePrecision(op_selector=op_sel)

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

    print(f"      Saved ({elapsed:.1f}s) — FP16:{sel_stats.get('fp16',0)} "
          f"FP32:{sel_stats.get('fp32',0)} "
          f"unresolved:{sel_stats.get('unresolved_fp32', sel_stats.get('unresolved_fp16', 0))}")

    del mlmodel, traced, wrapper
    gc.collect()

    return dec_path, elapsed, sel_stats


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


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


def detect_repetition_onset(tokens: List[int], min_repeat_len: int = 3) -> int:
    if len(tokens) < min_repeat_len * 2:
        return -1
    for window in range(min_repeat_len, len(tokens) // 2 + 1):
        for start in range(len(tokens) - window * 2 + 1):
            if tokens[start:start + window] == tokens[start + window:start + window * 2]:
                return start + window
    return -1


def tokens_match_count(ref_tokens: List[int], var_tokens: List[int]) -> int:
    n = min(len(ref_tokens), len(var_tokens))
    for i in range(n):
        if ref_tokens[i] != var_tokens[i]:
            return i
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine with hot-swap (reused from fp32_layer_knockout.py)
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    def __init__(self, chunk_paths: List[str], embed_lmhead_path: str,
                 function_name: str = "infer"):
        self.loadable = True
        self.function_name = function_name
        self.chunk_paths = list(chunk_paths)

        print(f"    Loading embed+lmhead...")
        self.embed = ct.models.MLModel(
            embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
            function_name="embed")
        self.lmhead = ct.models.MLModel(
            embed_lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY,
            function_name="lmhead")

        self.ffns = []
        for ci, path in enumerate(chunk_paths):
            prec = "FP16" if "fp16" in path else "FP32"
            print(f"    Loading chunk {ci} [{prec}] ...")
            try:
                m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                      function_name=function_name)
                self.ffns.append(m)
            except Exception as e:
                print(f"    *** FAILED chunk {ci}: {e}")
                self.loadable = False
                return

        self._build_input_maps()
        self.reset_all()

    def _build_input_maps(self):
        self.inp_maps = []
        for ci in range(len(self.ffns)):
            spec = self.ffns[ci].get_spec()
            imap = {}
            for fn in spec.description.functions:
                if fn.name == self.function_name:
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

    def swap_chunk(self, chunk_idx: int, new_path: str, function_name: str = "_default_"):
        """Hot-swap a single chunk without reloading the rest.
        
        function_name:
          "_default_" — use the engine's default function_name (e.g. 'infer')
          None         — don't pass function_name (loads 'main' single-function model)
          str          — use the specified function name
        """
        if function_name == "_default_":
            fn = self.function_name
        else:
            fn = function_name

        if self.chunk_paths[chunk_idx] == new_path:
            return
        label = os.path.basename(new_path).replace('.mlpackage', '')
        print(f"    Swapping chunk {chunk_idx} → [{label}] ...")
        old_model = self.ffns[chunk_idx]
        try:
            kwargs = dict(compute_units=ct.ComputeUnit.CPU_AND_NE)
            if fn is not None:
                kwargs["function_name"] = fn
            new_model = ct.models.MLModel(new_path, **kwargs)
            self.ffns[chunk_idx] = new_model
            self.chunk_paths[chunk_idx] = new_path
            # Rebuild input map
            spec = new_model.get_spec()
            imap = {}
            if fn is not None:
                for f in spec.description.functions:
                    if f.name == fn:
                        for inp in f.input:
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
            self.inp_maps[chunk_idx] = imap
            del old_model
            gc.collect()
        except Exception as e:
            print(f"    *** Swap FAILED chunk {chunk_idx}: {e}")
            self.loadable = False

    def reset_all(self):
        if not self.loadable:
            return
        try:
            self.states = [m.make_state() for m in self.ffns]
        except Exception as e:
            print(f"    *** make_state failed: {e}")
            self.loadable = False
            return
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

    def generate(self, token_ids: List[int], max_gen: int, stop_ids: set,
                 capture_traces: bool = False) -> dict:
        if not self.loadable:
            return {"gen_tokens": [], "tps": 0}
        self.reset_all()
        t0 = time.time()

        for i, tid in enumerate(token_ids):
            if i >= CTX:
                break
            last_tok, last_hidden, last_logits = self.step(tid, i)
        prefill_ms = (time.time() - t0) * 1000

        result = {
            "gen_tokens": [last_tok],
            "prefill_ms": prefill_ms,
        }
        if capture_traces:
            result["prefill_hidden"] = last_hidden.copy()
            result["prefill_logits"] = last_logits.copy()
            result["decode_hiddens"] = []
            result["decode_logits"] = []

        t_dec = time.time()
        prefill_end = len(token_ids)
        for gi in range(max_gen - 1):
            pos = prefill_end + gi
            if pos >= CTX - 1:
                break
            next_tok, hidden, logits = self.step(result["gen_tokens"][-1], pos)
            result["gen_tokens"].append(next_tok)
            if capture_traces:
                result["decode_hiddens"].append(hidden.copy())
                result["decode_logits"].append(logits.copy())
            if next_tok in stop_ids:
                break

        t_decode = time.time() - t_dec
        n_dec = max(1, len(result["gen_tokens"]) - 1)
        result["tps"] = n_dec / t_decode if t_decode > 0 else 0
        result["decode_ms"] = t_decode * 1000
        return result

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        self.ffns = []
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Compare two runs
# ─────────────────────────────────────────────────────────────────────────────

def compare_runs(ref: dict, var: dict) -> dict:
    ref_toks = ref["gen_tokens"]
    var_toks = var["gen_tokens"]

    match_count = tokens_match_count(ref_toks, var_toks)
    n_common = min(len(ref_toks), len(var_toks))
    total_match = sum(1 for a, b in zip(ref_toks, var_toks) if a == b)

    metrics = {
        "first_token_match": ref_toks[0] == var_toks[0] if ref_toks and var_toks else False,
        "consecutive_match": match_count,
        "total_match": total_match,
        "total_tokens": n_common,
        "match_rate": total_match / max(n_common, 1),
        "exact_match": ref_toks == var_toks,
    }

    # Early agreement (first 10 tokens)
    early_ref = ref_toks[:10]
    early_var = var_toks[:10]
    n_early = min(len(early_ref), len(early_var))
    early_match = sum(1 for a, b in zip(early_ref[:n_early], early_var[:n_early]) if a == b)
    metrics["early_agreement"] = early_match / max(n_early, 1)

    rep_onset = detect_repetition_onset(var_toks)
    metrics["repetition_onset"] = rep_onset

    # Trace-level metrics if available
    if "prefill_hidden" in ref and "prefill_hidden" in var:
        metrics["h_cos_pf"] = cosine_sim(ref["prefill_hidden"], var["prefill_hidden"])
        if ref.get("decode_hiddens") and var.get("decode_hiddens"):
            n = min(len(ref["decode_hiddens"]), len(var["decode_hiddens"]))
            h_coss = [cosine_sim(ref["decode_hiddens"][i], var["decode_hiddens"][i]) for i in range(n)]
            metrics["h_cos_dec"] = float(np.mean(h_coss)) if h_coss else -1.0
        if ref.get("decode_logits") and var.get("decode_logits"):
            n = min(len(ref["decode_logits"]), len(var["decode_logits"]))
            kls = [kl_divergence(ref["decode_logits"][i], var["decode_logits"][i]) for i in range(n)]
            metrics["kl_dec"] = float(np.mean(kls)) if kls else -1.0

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Variant definitions
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1: Single-layer FP32 promotions (inverse: FP16 base + promote one layer to FP32)
# These answer: "which layer benefits most from FP32?"
PHASE1_VARIANTS = OrderedDict([
    ("layer7_fp32_only",  {"fp32_layers": {7},    "desc": "Layer 7 (F) → FP32, rest FP16"}),
    ("layer8_fp32_only",  {"fp32_layers": {8},    "desc": "Layer 8 (L) → FP32, rest FP16"}),
    ("layer9_fp32_only",  {"fp32_layers": {9},    "desc": "Layer 9 (L) → FP32, rest FP16"}),
    ("layer10_fp32_only", {"fp32_layers": {10},   "desc": "Layer 10 (L) → FP32, rest FP16"}),
])

# Phase 2: Two-layer FP32 promotions
PHASE2_VARIANTS = OrderedDict([
    ("layer7+8_fp32",   {"fp32_layers": {7, 8},   "desc": "Layers 7+8 → FP32, rest FP16"}),
    ("layer8+9_fp32",   {"fp32_layers": {8, 9},   "desc": "Layers 8+9 → FP32, rest FP16"}),
    ("layer9+10_fp32",  {"fp32_layers": {9, 10},  "desc": "Layers 9+10 → FP32, rest FP16"}),
])


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    os.makedirs(args.output, exist_ok=True)

    tok_path = args.tokenizer or args.fp32_dir
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    stop_ids = _build_stop_ids(tokenizer)

    # Tokenize prompts
    prompt_tokens = {}
    for pi, prompt in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=True)
        prompt_tokens[pi] = ids[0].tolist() if hasattr(ids, 'tolist') else list(ids[0]) if hasattr(ids[0], 'tolist') else list(ids)
        print(f"  P{pi}: {prompt[:50]}... ({len(prompt_tokens[pi])} tokens)")

    # Load PyTorch model for export
    print(f"\nLoading PyTorch model from {args.model}...")
    cfg = Qwen35Config.from_json(os.path.join(args.model, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    pt_model = Qwen35ForCausalLM(cfg)
    assert pt_model.load_pretrained_weights(args.model), "Failed to load weights"
    pt_model.eval()
    for p in pt_model.parameters():
        p.requires_grad = False
    print("  PyTorch model loaded.")

    # Paths
    embed_lmhead = os.path.join(args.fp32_dir, "embed_lmhead_combined.mlpackage")
    assert os.path.exists(embed_lmhead), f"Missing: {embed_lmhead}"
    fp32_chunk2 = os.path.join(args.fp32_dir, "combined_LUT4_dedup", "chunk2.mlpackage")
    fp16_chunk2 = os.path.join(args.fp16_dir, "combined_LUT4_dedup", "chunk2.mlpackage")

    # Results
    results_path = os.path.join(args.output, "chunk2_results.json")
    all_results = {}

    # ─── Phase 0: Export Phase 1 (single-layer) variants ─────────────────
    print("\n" + "=" * 70)
    print("  PHASE 0: Export Chunk2 Single-Layer FP32 Promotion Variants")
    print("  (Inverse approach: FP16 base + promote individual layers to FP32)")
    print("=" * 70)

    exported_paths = {}  # variant_name -> path

    for vname, vdef in PHASE1_VARIANTS.items():
        fp32_set = vdef["fp32_layers"]
        print(f"\n  {vname}: {vdef['desc']}")
        path, elapsed, stats = export_chunk2_variant(
            pt_model, vname, fp32_set, args.output, inverse=True)
        exported_paths[vname] = path
        if elapsed > 0:
            print(f"      Export time: {elapsed:.1f}s")

    # Keep PyTorch model in case we need Phase 2 combo exports later

    # ─── Phase 1: FP32 Reference + Knockouts ─────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: Full FP32 Reference")
    print("=" * 70)

    clear_bnns_cache()

    # Load full FP32 engine
    fp32_paths = [
        os.path.join(args.fp32_dir, "combined_LUT4_dedup", f"chunk{ci}.mlpackage")
        for ci in range(NUM_CHUNKS)
    ]
    engine = InferenceEngine(fp32_paths, embed_lmhead)
    if not engine.loadable:
        print("  *** Cannot load FP32 engine. Aborting.")
        return

    # Run FP32 reference
    fp32_refs = {}
    all_results["chunk2_all_fp32"] = []
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  P{pi}: {prompt[:50]}...")
        result = engine.generate(prompt_tokens[pi], args.max_tokens, stop_ids,
                                 capture_traces=True)
        fp32_refs[pi] = result
        text = tokenizer.decode(result["gen_tokens"], skip_special_tokens=True)
        rep = detect_repetition_onset(result["gen_tokens"])
        all_results["chunk2_all_fp32"].append({
            "prompt_idx": pi,
            "gen_tokens": result["gen_tokens"],
            "text": text[:300],
            "tps": result["tps"],
            "repetition_onset": rep,
        })
        print(f"    tps={result['tps']:.1f} rep={rep}")
        print(f"    Gen: {text[:120]}")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ─── Phase 2: Single-layer knockouts ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: Single-Layer FP16 Knockouts (hot-swap chunk2)")
    print("=" * 70)

    layer_sensitivity = {}  # vname -> {"avg_match": ..., "per_prompt": [...]}

    def run_variant(vname, vpath, fn_name=None):
        """Swap chunk2, run 3 prompts, swap back. Returns per-prompt results."""
        print(f"\n{'─' * 60}")
        print(f"  {vname}")
        print(f"{'─' * 60}")

        engine.swap_chunk(CHUNK2_IDX, vpath, function_name=fn_name)
        if not engine.loadable:
            print(f"  *** Not loadable after swap")
            engine.loadable = True
            engine.swap_chunk(CHUNK2_IDX, fp32_chunk2)
            return None

        all_results[vname] = []
        prompt_metrics = []

        for pi, prompt in enumerate(PROMPTS):
            print(f"\n    P{pi}: {prompt[:50]}...")
            result = engine.generate(prompt_tokens[pi], args.max_tokens, stop_ids,
                                     capture_traces=True)
            ref = fp32_refs[pi]
            metrics = compare_runs(ref, result)
            text = tokenizer.decode(result["gen_tokens"], skip_special_tokens=True)

            entry = {
                "prompt_idx": pi,
                "gen_tokens": result["gen_tokens"],
                "text": text[:300],
                "tps": result["tps"],
                **metrics,
            }
            all_results[vname].append(entry)
            prompt_metrics.append(metrics)

            match_sym = "✓" if metrics["exact_match"] else "✗"
            print(f"      {match_sym} 1st={metrics['first_token_match']} "
                  f"consec={metrics['consecutive_match']}/{metrics['total_tokens']} "
                  f"match={metrics['match_rate']:.0%} "
                  f"early={metrics['early_agreement']:.0%} "
                  f"rep={metrics['repetition_onset']} "
                  f"tps={result['tps']:.1f}")
            if "h_cos_pf" in metrics:
                print(f"      h_cos_pf={metrics['h_cos_pf']:.6f} "
                      f"h_cos_dec={metrics.get('h_cos_dec', -1):.6f} "
                      f"kl_dec={metrics.get('kl_dec', -1):.4f}")
            if not metrics["exact_match"]:
                ref_text = tokenizer.decode(ref["gen_tokens"], skip_special_tokens=True)
                print(f"      REF: {ref_text[:100]}")
                print(f"      VAR: {text[:100]}")

        # Swap back to FP32
        engine.swap_chunk(CHUNK2_IDX, fp32_chunk2)

        # Save incremental
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        return prompt_metrics

    # Run single-layer knockouts
    for vname, vdef in PHASE1_VARIANTS.items():
        vpath = exported_paths[vname]
        pm = run_variant(vname, vpath, fn_name=None)
        if pm:
            avg_match = sum(m["match_rate"] for m in pm) / len(pm)
            avg_consec = sum(m["consecutive_match"] for m in pm) / len(pm)
            layer_sensitivity[vname] = {
                "desc": vdef["desc"],
                "avg_match_rate": avg_match,
                "avg_consecutive": avg_consec,
                "per_prompt": pm,
            }

    # Also test chunk2_all_fp16 (existing pre-compiled FP16 chunk)
    print(f"\n{'─' * 60}")
    print(f"  chunk2_all_fp16 (existing pre-compiled)")
    print(f"{'─' * 60}")
    pm = run_variant("chunk2_all_fp16", fp16_chunk2, fn_name="infer")
    if pm:
        avg_match = sum(m["match_rate"] for m in pm) / len(pm)
        avg_consec = sum(m["consecutive_match"] for m in pm) / len(pm)
        layer_sensitivity["chunk2_all_fp16"] = {
            "desc": "All layers 7-10 → FP16 (pre-compiled)",
            "avg_match_rate": avg_match,
            "avg_consecutive": avg_consec,
            "per_prompt": pm,
        }

    # ─── Phase 3: Two-layer combinations (if not phase1_only) ────────────
    if not args.phase1_only and PHASE2_VARIANTS:
        print("\n" + "=" * 70)
        print("  PHASE 3: Two-Layer FP32 Promotion Combos (incremental export+test)")
        print("=" * 70)

        for vname, vdef in PHASE2_VARIANTS.items():
            # Export one combo variant at a time to conserve disk space
            fp32_set = vdef["fp32_layers"]
            print(f"\n  Exporting {vname}: {vdef['desc']}...")
            vpath, elapsed, stats = export_chunk2_variant(
                pt_model, vname, fp32_set, args.output, inverse=True)
            exported_paths[vname] = vpath
            if elapsed > 0:
                print(f"      Export time: {elapsed:.1f}s")

            pm = run_variant(vname, vpath, fn_name=None)
            if pm:
                avg_match = sum(m["match_rate"] for m in pm) / len(pm)
                avg_consec = sum(m["consecutive_match"] for m in pm) / len(pm)
                layer_sensitivity[vname] = {
                    "desc": vdef["desc"],
                    "avg_match_rate": avg_match,
                    "avg_consecutive": avg_consec,
                    "per_prompt": pm,
                }

            # Delete this exported combo to free disk space for next one
            if not args.keep_exports and os.path.isdir(vpath):
                print(f"    Cleaning {vpath}...")
                shutil.rmtree(vpath)

    # Free PyTorch model
    del pt_model
    gc.collect()
    print("\n  [PyTorch model freed]")

    engine.cleanup()

    # ─── Final Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY: Per-Layer Sensitivity in Chunk 2")
    print("=" * 70)

    # Sort by average match rate (descending = least sensitive first)
    ranked = sorted(layer_sensitivity.items(),
                    key=lambda x: x[1]["avg_match_rate"], reverse=True)

    print(f"\n  {'Variant':<22} {'P0 con':>7} {'P0 %':>6} {'P1 con':>7} {'P1 %':>6} "
          f"{'P2 con':>7} {'P2 %':>6} {'Avg%':>6} {'h_dec':>7} {'KL_dec':>8}")
    print("  " + "─" * 95)

    # Print FP32 reference first
    print(f"  {'chunk2_all_fp32':<22} {'40/40':>7} {'100%':>6} {'40/40':>7} {'100%':>6} "
          f"{'40/40':>7} {'100%':>6} {'100%':>6} {'1.0000':>7} {'0.0000':>8}")

    for vname, info in ranked:
        pm = info["per_prompt"]
        cols = []
        h_decs = []
        kl_decs = []
        for m in pm:
            c = m["consecutive_match"]
            t = m["total_tokens"]
            r = m["match_rate"]
            cols.append(f"{c}/{t}")
            cols.append(f"{r:.0%}")
            if "h_cos_dec" in m and m["h_cos_dec"] >= 0:
                h_decs.append(m["h_cos_dec"])
            if "kl_dec" in m and m["kl_dec"] >= 0:
                kl_decs.append(m["kl_dec"])
        avg_h = f"{sum(h_decs)/len(h_decs):.4f}" if h_decs else "N/A"
        avg_kl = f"{sum(kl_decs)/len(kl_decs):.4f}" if kl_decs else "N/A"
        print(f"  {vname:<22} {cols[0]:>7} {cols[1]:>6} {cols[2]:>7} {cols[3]:>6} "
              f"{cols[4]:>7} {cols[5]:>6} {info['avg_match_rate']:>5.0%} "
              f"{avg_h:>7} {avg_kl:>8}")

    # Sensitivity ranking
    print(f"\n  SENSITIVITY RANKING (most sensitive first):")
    for i, (vname, info) in enumerate(reversed(ranked)):
        print(f"    {i+1}. {vname:<22} avg_match={info['avg_match_rate']:.0%}")

    # Recommendation
    print(f"\n  RECOMMENDATION (inverse approach: FP16 base + promote layers to FP32):")
    print(f"    Higher match% = more improvement from making that layer FP32")
    
    # Get single-layer results only
    singles = [(vn, info) for vn, info in ranked
               if vn.startswith("layer") and "_only" in vn]
    
    all_fp16_entry = layer_sensitivity.get("chunk2_all_fp16")
    if all_fp16_entry:
        print(f"    All-FP16 baseline: avg_match={all_fp16_entry['avg_match_rate']:.0%}")
    
    if singles:
        best = singles[0]
        worst = singles[-1]
        print(f"    Most beneficial FP32 layer: {best[0]} (avg_match={best[1]['avg_match_rate']:.0%})")
        print(f"    Least beneficial FP32 layer: {worst[0]} (avg_match={worst[1]['avg_match_rate']:.0%})")
        
        # Identify layers that improve significantly over all-FP16
        if all_fp16_entry:
            fp16_baseline = all_fp16_entry['avg_match_rate']
            helpful = [(vn, info) for vn, info in singles
                       if info['avg_match_rate'] > fp16_baseline + 0.10]
            if helpful:
                print(f"    FP32 layers with >10% improvement over all-FP16:")
                for vn, info in helpful:
                    delta = info['avg_match_rate'] - fp16_baseline
                    print(f"      {vn}: +{delta:.0%}")
            else:
                print(f"    No single layer FP32 gives >10% improvement over all-FP16")
                print(f"    → Precision sensitivity is distributed across all chunk2 layers")
                print(f"    → Recommendation: keep ENTIRE chunk2 in FP32")

    # Cleanup exported chunks
    if not args.keep_exports:
        export_dir = os.path.join(args.output, "exported_chunks")
        if os.path.isdir(export_dir):
            print(f"\n  Cleaning up exported chunks...")
            shutil.rmtree(export_dir)

    print(f"\n  Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Chunk2 per-layer FP16 knockout")
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--fp16-dir", required=True, help="FP16 baseline directory")
    parser.add_argument("--fp32-dir", required=True, help="FP32 reference directory")
    parser.add_argument("--output", default="tests/dev/chunk2_knockout_results")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--phase1-only", action="store_true",
                        help="Only run single-layer knockouts (skip combinations)")
    parser.add_argument("--keep-exports", action="store_true",
                        help="Keep exported chunk variants (don't auto-clean)")
    args = parser.parse_args()
    print(f"  Model: {args.model}")
    print(f"  FP16 dir: {args.fp16_dir}")
    print(f"  FP32 dir: {args.fp32_dir}")
    print(f"  Chunk2: layers {CHUNK2_START}-{CHUNK2_END-1} (FLLL)")
    print(f"  Max tokens: {args.max_tokens}")
    run_experiment(args)


if __name__ == "__main__":
    main()
