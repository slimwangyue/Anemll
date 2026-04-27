#!/usr/bin/env python3
"""Qwen3.5 — Step 1: Export all CoreML model components.

Exports (default V4+P2+D2 policy):
  - embeddings (LUT6 gs=8)                  → embeddings.mlpackage
  - lm_head_nosplit (LUT6 gs=8, single out)  → lm_head_nosplit.mlpackage
  - 9 FFN decode chunks (LUT4 gs=4, D2)      → ffn_LUT4_chunk{0..8}.mlpackage
  - 9 FFN prefill chunks (LUT4 gs=4, D2)     → prefill_LUT4_chunk{0..8}.mlpackage

Default features (Milestone 3.3):
  V4: FP32 for kv_cache_state ops in F-layers, FP16 everywhere else
  P2: Per-head attention splitting for ANE L2 cache residency (in model code)
  D2: Keep F-layer attention Q/K/V/O weights in FP16 (skip LUT4 for those)

Usage:
    python scripts_qwen3_5/export.py --model /path/to/Qwen3.5-4B --output /path/to/output
    python scripts_qwen3_5/export.py --nosplit-lmhead --lut-bits 4 --per-channel 4
    python scripts_qwen3_5/export.py --skip-existing
    python scripts_qwen3_5/export.py --no-v4   # disable V4 (full FP16)
    python scripts_qwen3_5/export.py --no-d2   # disable D2 (all weights LUT4)
"""
import gc, time, argparse, os, sys, shutil, glob, re
import numpy as np
import torch
import coremltools as ct
from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as mil_types
from coremltools.converters.mil.mil.passes.helper import block_context_manager

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)  # must be first for config.py

from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, LM_HEAD_LUT,
    PER_CHANNEL, FFN_PER_CHANNEL, FFN_LABEL, DEFAULT_HF_MODEL, DEFAULT_OUTPUT,
    CHUNK_RANGES,
)


# ── V4 F-fp32/L-fp16 precision selector ──

_LAYER_PATTERN = re.compile(r"layers[._](\d+)")


def _is_kv_cache_op(op):
    """Identify kv_cache_state ops by type and graph structure.

    KV cache ops in F (full_attention) layers:
      - slice_update with "cache" in name (cache writes)
      - identity ops (cache read pass-throughs from CoreML state)
      - squeeze feeding cache writes (pre-write reshape)
      - slice_by_index feeding identity (cache read extraction)
    """
    name_lower = op.name.lower()

    # Direct: ops with "cache" in name (slice_update for k/v cache writes)
    if "cache" in name_lower:
        return True

    # Identity ops are used for state reads in coremltools
    if op.op_type == "identity":
        return True

    # Squeeze feeding a cache write (check output consumers)
    if op.op_type == "squeeze":
        try:
            for out_var in op.outputs:
                for child in out_var.child_ops:
                    if "cache" in child.name.lower():
                        return True
        except (AttributeError, TypeError):
            pass

    # Slice_by_index feeding an identity (cache read extraction)
    if op.op_type == "slice_by_index":
        try:
            for out_var in op.outputs:
                for child in out_var.child_ops:
                    if child.op_type == "identity":
                        return True
        except (AttributeError, TypeError):
            pass

    return False


def _make_v4_selector(fp16_layers, fp32_layers):
    """V4 op_selector: FP16 for everything except F-layer kv_cache_state ops.

    Uses "max layer in transitive input set" home-attribution strategy.
    - L-layer ops → FP16
    - F-layer ops → FP16 (except ~8 kv_cache_state ops per F-layer → FP32)
    - Pre-layer ops (no layer attribution) → FP16
    """
    fp16_set = set(fp16_layers)
    fp32_set = set(fp32_layers)
    _cache = {}

    def _get_layers(op, visited=None):
        op_id = id(op)
        if op_id in _cache:
            return _cache[op_id]
        if visited is None:
            visited = set()
        if op_id in visited:
            return set()
        visited.add(op_id)
        layers = set()
        m = _LAYER_PATTERN.search(op.name)
        if m:
            layers.add(int(m.group(1)))
        for inp_val in op.inputs.values():
            if isinstance(inp_val, (list, tuple)):
                for v in inp_val:
                    if hasattr(v, "op") and v.op is not None:
                        layers |= _get_layers(v.op, visited)
            elif hasattr(inp_val, "op") and inp_val.op is not None:
                layers |= _get_layers(inp_val.op, visited)
        _cache[op_id] = layers
        return layers

    def selector(op):
        layers = _get_layers(op)
        if not layers:
            return True  # pre-layer ops: always FP16
        home = max(layers)
        if home in fp16_set:
            return True  # L layer → FP16
        if home in fp32_set:
            # F layer: FP16 unless kv_cache_state op
            return not _is_kv_cache_op(op)
        return True  # unexpected → FP16

    return selector


def get_v4_compute_precision(model, chunk_idx):
    """Build V4 compute precision: FP16 everywhere except F-layer kv_cache ops.

    Returns FP16ComputePrecision with op_selector for chunks containing F-layers,
    or "float16" string for pure-L chunks.
    """
    sl, el = CHUNK_RANGES[chunk_idx]
    fp16_layers = []
    fp32_layers = []
    for li in range(sl, el):
        layer = model.model.layers[li]
        layer_type = getattr(layer, "layer_type", None)
        if layer_type == "full_attention":
            fp32_layers.append(li)
        else:
            fp16_layers.append(li)

    if not fp32_layers:
        # All L-layers: pure fp16 (no kv_cache ops to protect)
        return "float16", fp16_layers, fp32_layers

    # Has F-layers: use V4 selector (kv_cache FP32, everything else FP16)
    selector = _make_v4_selector(fp16_layers, fp32_layers)
    return FP16ComputePrecision(op_selector=selector), fp16_layers, fp32_layers


# ── Post-conversion pass: ensure casts between read_state and slice_update ──

def _ensure_state_slice_update_casts(mlmodel):
    """Inject casts between read_state and slice_update to work around ANE bug.

    The ANE has a runtime defect where slice_update operating directly on
    read_state output ignores dynamic begin/end parameters and uses trace-time
    constants instead.  This causes KV cache writes to go to position 0 instead
    of current_pos during multi-block batch prefill.

    FP16ComputePrecision cannot fix this because read_state and
    coreml_update_state are in its _UNSUPPORTED_FP16_OPS set — the V4
    op_selector is never consulted for state ops.

    This pass finds slice_update ops whose 'x' input comes directly from
    read_state (no intermediate op) and injects a cast(fp16→fp32) + cast back
    to break the chain.  The cast forces the ANE to evaluate dynamic positions.

    Returns the number of casts injected.
    """
    prog = mlmodel._mil_program
    if prog is None:
        return 0

    total_injected = 0

    for fn in prog.functions.values():
        total_injected += _inject_state_casts_in_block(fn)

    return total_injected


@block_context_manager
def _inject_state_casts_in_block(block):
    """Walk a MIL block and inject casts for read_state → slice_update chains."""
    injected = 0

    for op in list(block.operations):
        # Process nested blocks
        for b in op.blocks:
            injected += _inject_state_casts_in_block(b)

        if op.op_type != "slice_update":
            continue

        x_var = op.inputs.get("x")
        if x_var is None or x_var.op is None:
            continue
        if x_var.op.op_type != "read_state":
            continue  # already has intermediate op (cast, etc.)

        # Direct read_state → slice_update — inject cast to break the chain.
        # Cast fp16 → fp32 (forces ANE to re-evaluate dynamic begin/end).
        cast_to_fp32 = mb.cast(
            x=x_var,
            dtype="fp32",
            name=f"{x_var.name}_to_fp32",
            before_op=op,
        )

        # Also promote update to fp32 for type consistency
        update_var = op.inputs.get("update")
        if update_var is not None and update_var.is_tensor_or_scalar_of(dtype="fp16"):
            cast_update = mb.cast(
                x=update_var,
                dtype="fp32",
                name=f"{update_var.name}_to_fp32",
                before_op=op,
            )
        else:
            cast_update = update_var

        # Collect remaining inputs (begin, end, squeeze_mask, stride, etc.)
        new_inputs = {}
        for k, v in op.inputs.items():
            if k == "x":
                new_inputs[k] = cast_to_fp32
            elif k == "update":
                new_inputs[k] = cast_update
            else:
                new_inputs[k] = v

        new_inputs["name"] = f"{op.name}_fp32"
        new_inputs["before_op"] = op

        # Create new slice_update with fp32 inputs
        new_su = mb.slice_update(**new_inputs)

        # Cast output back to fp16 for downstream consumers (coreml_update_state)
        cast_back = mb.cast(
            x=new_su,
            dtype="fp16",
            name=f"{new_su.name}_to_fp16",
            before_op=op,
        )

        # Replace all uses of the old slice_update output
        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=cast_back,
            force_replace=True,
        )

        # Remove old slice_update
        op.enclosing_block.remove_ops([op])
        injected += 1

    return injected


# ── D2: Selective LUT4 — keep F-layer attention Q/K/V/O in FP16 ──

# Attention weight families to keep in FP16 (not quantized to LUT4)
D2_FP16_ATTN_FAMILIES = ["attn_q", "attn_kv", "attn_o"]

# ── E235: D2 + ssm_alpha/beta FP16 + SSM projections LUT6 gs=2 ──
# Best combined quantization policy from chunk-1 experiments.
# +1.9% cos_sim vs baseline, +25 MB/chunk, zero speed impact.
E235_FP16_FAMILIES = ["attn_q", "attn_kv", "attn_o", "ssm_alpha", "ssm_beta"]
E235_LUT6_GS2_FAMILIES = ["ssm_qkv", "ssm_z", "ssm_out"]


def _apply_selective_lut4(mlmodel, fp16_families, lut_bits=4, per_channel=FFN_PER_CHANNEL,
                         lut6_gs2_families=None):
    """Apply LUT palettization while keeping specified weight families in FP16.

    Uses discover_weight_ops from fp16_ablation to identify weight ops
    and skip palettization for ops matching fp16_families.
    Optionally applies LUT6 gs=2 to specified families (E235 policy).
    """
    import warnings
    import coremltools.optimize as cto
    from fp16_ablation import _build_selective_lut_config
    opt_config = _build_selective_lut_config(
        mlmodel, fp16_families, lut_bits=lut_bits, per_channel=per_channel,
        lut6_gs2_families=lut6_gs2_families,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cto.coreml.palettize_weights(mlmodel, opt_config)


# ── Selective FP32: keep only precision-critical ops in fp32 ──

# Op types that are most sensitive to fp16 precision loss.
# These are kept in fp32 (their traced precision) while everything else is fp16.
_FP32_SENSITIVE_OPS = frozenset({
    'softmax',       # attention softmax — exp overflow, precision loss in distribution
    'reduce_sum',    # RMSNorm & attention accumulation
    'reduce_mean',   # normalization ops
    'rsqrt',         # RMSNorm inverse sqrt
    'exp',           # L-layer recurrence (A_log.float().exp())
    'log',           # potential log in recurrence
    'cumsum',        # cumulative sum in recurrence
})


def _make_selective_fp32_selector():
    """Op selector for FP16ComputePrecision: convert everything to fp16
    EXCEPT precision-critical ops (softmax, reduce, rsqrt, exp).

    Returns True = convert to fp16, False = keep original precision.
    """
    def selector(op):
        if op.op_type in _FP32_SENSITIVE_OPS:
            return False  # keep in fp32 (original traced precision)
        return True  # convert to fp16

    return selector


def get_selective_compute_precision():
    """Return FP16ComputePrecision with selective op_selector."""
    return FP16ComputePrecision(op_selector=_make_selective_fp32_selector())


# ── Delta-FP32: keep chunked delta rule accumulations in fp32 (prefill only) ──

# Superset of _FP32_SENSITIVE_OPS plus data-dependent matmuls.
# PyTorch `@` between two data tensors (not fixed weights) traces to MIL `matmul`.
# Fixed-weight projections (nn.Conv2d / nn.Linear) trace to MIL `conv` or `linear`,
# which are NOT in this set → they stay fp16.
_DELTA_FP32_OPS = frozenset({
    'softmax',       # attention softmax
    'reduce_sum',    # l2norm, attention accumulation
    'reduce_mean',   # normalization
    'rsqrt',         # l2norm inverse sqrt
    'exp',           # decay computation (g.exp()), recurrence
    'log',           # potential log in recurrence
    'cumsum',        # cumulative sum (if used directly)
    'matmul',        # data-dependent matmuls: intra-chunk attn, forward substitution,
                     # inter-chunk recurrence, cumsum-via-tril_ones-matmul
})


def _make_delta_fp32_selector():
    """Op selector for prefill: keep delta rule accumulation ops in fp32.

    Also keeps KV cache state ops in fp32 (for F-layer chunks).
    Returns True = convert to fp16, False = keep original precision (fp32).
    """
    def selector(op):
        if op.op_type in _DELTA_FP32_OPS:
            return False  # keep in fp32
        if _is_kv_cache_op(op):
            return False  # keep KV cache ops in fp32
        return True  # convert to fp16
    return selector


def get_delta_fp32_compute_precision():
    """Return FP16ComputePrecision keeping delta rule ops in fp32."""
    return FP16ComputePrecision(op_selector=_make_delta_fp32_selector())


from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


def export_embeddings(model, out_dir, skip_existing, compute_precision="float16"):
    path = os.path.join(out_dir, "embeddings.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] embeddings")
        return
    print(f"  Exporting embeddings (LUT{LUT_BITS} gs={PER_CHANNEL})...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=PER_CHANNEL,
                           compute_precision=compute_precision)
    ml = conv.convert_part_1(model)
    ml.save(path)
    del ml, conv; gc.collect()
    print(f"  Saved embeddings ({time.time()-t0:.1f}s)")

    # Also export fixed-shape variants for ANE-safe multifunction combine
    for seq_len, suffix in [(1, "embed_single"), (BATCH_SIZE, "embed_prefill")]:
        fpath = os.path.join(out_dir, f"{suffix}.mlpackage")
        if skip_existing and os.path.exists(fpath):
            print(f"  [skip] {suffix}")
            continue
        print(f"  Exporting {suffix} (seq_len={seq_len})...")
        t1 = time.time()
        conv2 = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=PER_CHANNEL,
                                compute_precision=compute_precision)
        ml2 = conv2.convert_part_1(model, seq_len=seq_len)
        ml2.save(fpath)
        del ml2, conv2; gc.collect()
        print(f"  Saved {suffix} ({time.time()-t1:.1f}s)")


def export_lm_head(model, out_dir, skip_existing, compute_precision="float16"):
    path = os.path.join(out_dir, "lm_head_logits.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] lm_head_logits")
        return
    print(f"  Exporting lm_head_logits 16-way split (LUT{LM_HEAD_LUT})...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LM_HEAD_LUT, per_channel=PER_CHANNEL,
                           compute_precision=compute_precision)
    ml = conv.convert_part_3(model, argmax_in_model=False)
    ml.save(path)
    del ml, conv; gc.collect()
    print(f"  Saved lm_head_logits ({time.time()-t0:.1f}s)")


def export_lm_head_nosplit(model, out_dir, skip_existing, compute_precision="float16"):
    """Export a NON-SPLIT lm_head (single Conv2d for full vocab)."""
    path = os.path.join(out_dir, "lm_head_nosplit.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] lm_head_nosplit")
        return

    class LMHeadNoSplitWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            parts = [getattr(m, f"lm_head16_{i+1}").weight for i in range(m.lm_head_split)]
            full_weight = torch.cat(parts, dim=0)
            self.lm_head = torch.nn.Conv2d(
                m.config.hidden_size, m.config.vocab_size, 1, bias=False,
                dtype=MODEL_DTYPE,
            ).to(TEST_DEVICE)
            self.lm_head.weight.data.copy_(full_weight)

        def forward(self, hidden_states):
            h = hidden_states.permute(0, 2, 1).unsqueeze(2)
            logits = self.lm_head(h)
            logits = logits.squeeze(2).permute(0, 2, 1)
            return logits

    print(f"  Exporting lm_head_nosplit (LUT{LM_HEAD_LUT} gs={PER_CHANNEL})...")
    t0 = time.time()
    wrapper = LMHeadNoSplitWrapper(model).eval()
    sample_input = torch.zeros((1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_input)

    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LM_HEAD_LUT, per_channel=PER_CHANNEL,
                           compute_precision=compute_precision)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=sample_input.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        compute_precision=conv.compute_precision,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    if LM_HEAD_LUT:
        conv.converted_model = mlmodel
        conv.postprocess(num_workers=1)
        mlmodel = conv.converted_model

    mlmodel.save(path)
    del mlmodel, conv, wrapper; gc.collect()
    print(f"  Saved lm_head_nosplit ({time.time()-t0:.1f}s)")


def export_ffn_chunks(model, out_dir, skip_existing, only_chunk=None, static_prefill=False,
                      lut_bits_override=None, per_channel_override=None, compute_precision="float16",
                      v4_precision=False, selective_fp32=False, fp16_attn=False, e235=False,
                      delta_fp32=False, prefill_only=False):
    lut_bits = lut_bits_override if lut_bits_override is not None else LUT_BITS
    ffn_pc = per_channel_override if per_channel_override is not None else FFN_PER_CHANNEL
    label = f"LUT{lut_bits}"
    chunk_indices = [only_chunk] if only_chunk is not None else list(range(NUM_CHUNKS))
    for ci in chunk_indices:
        sl, el = CHUNK_RANGES[ci]

        # Determine per-chunk compute precision
        # We compute: base_cp (for converter constructor), override_cp (to apply after), cp_label
        override_cp = None  # If set, replaces conv.compute_precision after constructor
        has_f_layers = False
        if selective_fp32:
            cp_label = "SELECTIVE-FP32"
            base_cp = "float32"
            override_cp = get_selective_compute_precision()
        elif v4_precision:
            v4_cp, fp16_ls, fp32_ls = get_v4_compute_precision(model, ci)
            has_f_layers = bool(fp32_ls)
            cp_label = f"V4(F={fp32_ls},L={fp16_ls})"
            base_cp = "float32"
            if not isinstance(v4_cp, str):
                override_cp = v4_cp
            else:
                base_cp = v4_cp  # pure "float16" or "float32"
        else:
            cp_label = compute_precision.upper()
            base_cp = compute_precision

        # D2: selective LUT4 — keep attn Q/K/V/O in FP16 for F-layer chunks
        # E235: D2 + ssm_alpha/beta FP16 + SSM projections LUT6 gs=2 (all chunks)
        use_selective_lut = (fp16_attn and has_f_layers) or e235
        if use_selective_lut:
            if e235:
                cp_label += "+E235"
            else:
                cp_label += "+D2"

        def _make_converter():
            # When using D2 selective LUT, skip built-in palettization
            # (we apply selective LUT4 post-conversion instead)
            effective_lut = None if use_selective_lut else lut_bits
            conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                   num_chunks=NUM_CHUNKS, lut_bits=effective_lut, per_channel=ffn_pc,
                                   compute_precision=base_cp)
            if override_cp is not None:
                conv.compute_precision = override_cp
            return conv

        # For delta-fp32: build separate prefill precision override
        prefill_override_cp = None
        if delta_fp32:
            prefill_override_cp = get_delta_fp32_compute_precision()

        # Decode chunk
        dec_path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        if prefill_only:
            print(f"  [skip] decode chunk {ci} (--prefill-only)")
        elif skip_existing and os.path.exists(dec_path):
            print(f"  [skip] decode chunk {ci} (layers {sl}-{el-1})")
        else:
            print(f"  Exporting decode chunk {ci} layers [{sl}-{el-1}] ({label} gs={ffn_pc} {cp_label})...")
            t0 = time.time()
            conv = _make_converter()
            ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                                     override_start_layer=sl, override_end_layer=el)
            # Inject casts between read_state → slice_update (ANE bug workaround)
            if has_f_layers:
                n_casts = _ensure_state_slice_update_casts(ml)
                if n_casts:
                    print(f"    [state-cast] Injected {n_casts} cast(s) for read_state → slice_update")
            if use_selective_lut:
                if e235:
                    fp16_fams = E235_FP16_FAMILIES if has_f_layers else ["ssm_alpha", "ssm_beta"]
                    print(f"    [E235] FP16: {fp16_fams}, LUT6 gs=2: {E235_LUT6_GS2_FAMILIES}, rest → LUT{lut_bits}")
                    ml = _apply_selective_lut4(ml, fp16_fams, lut_bits=lut_bits, per_channel=ffn_pc,
                                              lut6_gs2_families=E235_LUT6_GS2_FAMILIES)
                else:
                    print(f"    [D2] Applying selective LUT{lut_bits}: attn Q/K/V/O → FP16, rest → LUT{lut_bits}")
                    ml = _apply_selective_lut4(ml, D2_FP16_ATTN_FAMILIES, lut_bits=lut_bits, per_channel=ffn_pc)
            ml.save(dec_path)
            del ml, conv; gc.collect()
            print(f"  Saved decode chunk {ci} ({time.time()-t0:.1f}s)")

        # Prefill chunk
        if static_prefill:
            pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}_bs{BATCH_SIZE}.mlpackage")
            pf_desc = f"prefill chunk {ci} layers [{sl}-{el-1}] static bs{BATCH_SIZE}"
        else:
            pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage")
            pf_desc = f"prefill chunk {ci} layers [{sl}-{el-1}]"
        if skip_existing and os.path.exists(pf_path):
            print(f"  [skip] {pf_desc}")
        else:
            pf_cp_label = cp_label
            if delta_fp32:
                pf_cp_label = f"DELTA-FP32({cp_label})"
            print(f"  Exporting {pf_desc} ({label} gs={ffn_pc} {pf_cp_label})...")
            t0 = time.time()
            conv = _make_converter()
            # Override prefill compute precision for delta-fp32
            if prefill_override_cp is not None:
                conv.compute_precision = prefill_override_cp
            if static_prefill:
                ml = conv.convert_part_2_prefill_exact(
                    model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                    exact_seq_len=BATCH_SIZE,
                    override_start_layer=sl, override_end_layer=el)
            else:
                ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                                                 override_start_layer=sl, override_end_layer=el)
            # Inject casts between read_state → slice_update (ANE bug workaround)
            if has_f_layers:
                n_casts = _ensure_state_slice_update_casts(ml)
                if n_casts:
                    print(f"    [state-cast] Injected {n_casts} cast(s) for read_state → slice_update")
            if use_selective_lut:
                if e235:
                    fp16_fams = E235_FP16_FAMILIES if has_f_layers else ["ssm_alpha", "ssm_beta"]
                    print(f"    [E235] FP16: {fp16_fams}, LUT6 gs=2: {E235_LUT6_GS2_FAMILIES}, rest → LUT{lut_bits}")
                    ml = _apply_selective_lut4(ml, fp16_fams, lut_bits=lut_bits, per_channel=ffn_pc,
                                              lut6_gs2_families=E235_LUT6_GS2_FAMILIES)
                else:
                    print(f"    [D2] Applying selective LUT{lut_bits}: attn Q/K/V/O → FP16, rest → LUT{lut_bits}")
                    ml = _apply_selective_lut4(ml, D2_FP16_ATTN_FAMILIES, lut_bits=lut_bits, per_channel=ffn_pc)
            ml.save(pf_path)
            del ml, conv; gc.collect()
            print(f"  Saved {pf_desc} ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5 for ANE (Milestone 3.3 — V4+P2+D2)")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL,
                        help="Path to HuggingFace Qwen3.5 model directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for exported .mlpackage files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if .mlpackage already exists")
    parser.add_argument("--only-chunk", type=int, default=None,
                        help="Export only the specified chunk index")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Comma-separated chunk indices to export, e.g. '0,1,2,3'")
    parser.add_argument("--ffn-only", action="store_true",
                        help="Only export FFN chunks (skip embeddings and lm_head)")
    parser.add_argument("--lut-bits", type=int, default=None,
                        help="Override FFN LUT bits (e.g. 4 for LUT4). Default: use config.py")
    parser.add_argument("--per-channel", type=int, default=None,
                        help="Override FFN per-channel group size. Default: use config.py")
    parser.add_argument("--static-prefill", action="store_true",
                        help="Use static-shape prefill (convert_part_2_prefill_exact) with valid_len")
    parser.add_argument("--fp32-compute", action="store_true",
                        help="Use FLOAT32 compute precision (default: FLOAT16)")
    # V4 is ON by default; use --no-v4 to disable
    parser.add_argument("--v4-precision", action="store_true", default=True,
                        help="V4 mixed precision: kv_cache_state → FP32, everything else FP16 (default: ON)")
    parser.add_argument("--no-v4", dest="v4_precision", action="store_false",
                        help="Disable V4 precision (use uniform compute precision)")
    # D2 is ON by default; use --no-d2 to disable
    parser.add_argument("--fp16-attn", action="store_true", default=True,
                        help="D2: keep F-layer attention Q/K/V/O in FP16, rest LUT4 (default: ON)")
    parser.add_argument("--no-d2", dest="fp16_attn", action="store_false",
                        help="Disable D2 (all FFN weights quantized to LUT)")
    parser.add_argument("--e235", action="store_true", default=False,
                        help="E235: D2 + ssm_alpha/beta FP16 + SSM proj LUT6 gs=2 (best quality)")
    parser.add_argument("--selective-fp32", action="store_true",
                        help="Selective fp32: keep softmax/exp/rsqrt/reduce in fp32, rest fp16 for ANE")
    parser.add_argument("--delta-fp32", action="store_true",
                        help="Delta-FP32: keep chunked delta rule accumulations (exp/matmul/reduce) in fp32 for prefill only")
    parser.add_argument("--prefill-only", action="store_true",
                        help="Export only prefill chunks (skip decode chunks, embeddings, lm_head)")
    parser.add_argument("--nosplit-lmhead", action="store_true",
                        help="Export lm_head as single Conv2d (no 16-way split). Required for embed_lmhead_combined.")
    parser.add_argument("--ctx", type=int, default=None,
                        help="Override context length (default: from config.py)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch/prefill size (default: from config.py)")
    args = parser.parse_args()

    # CLI overrides for config values
    global CTX, BATCH_SIZE
    if args.ctx is not None:
        CTX = args.ctx
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size

    # --chunks takes precedence over --only-chunk
    if args.chunks is not None:
        args._chunk_list = [int(x.strip()) for x in args.chunks.split(",")]
    elif args.only_chunk is not None:
        args._chunk_list = [args.only_chunk]
    else:
        args._chunk_list = None  # all chunks
    os.makedirs(args.output, exist_ok=True)

    prefill_mode = "static" if args.static_prefill else "dynamic"
    cp = "float32" if args.fp32_compute else "float16"
    v4 = args.v4_precision
    d2 = args.fp16_attn
    e235 = args.e235
    sel_fp32 = args.selective_fp32
    delta_fp32 = args.delta_fp32
    prefill_only = args.prefill_only
    if delta_fp32:
        cp_desc_extra = " + DELTA-FP32(prefill: exp/matmul/reduce→fp32)"
    else:
        cp_desc_extra = ""
    if sel_fp32:
        cp_desc = "SELECTIVE-FP32(softmax/exp/rsqrt/reduce→fp32, rest→fp16)"
    elif v4:
        cp_desc = "V4(kv_cache→FP32, rest→FP16)"
        if e235:
            cp_desc += " + E235(attn+ssm_ab→FP16, ssm_proj→LUT6gs2)"
        elif d2:
            cp_desc += " + D2(attn Q/K/V/O→FP16)"
    else:
        cp_desc = cp.upper()
    cp_desc += cp_desc_extra
    print("=" * 70)
    print(f"  Qwen3.5 ANE Export — Milestone 3.3 (V4+P2+D2)")
    print(f"  Embed: LUT{LUT_BITS} gs={PER_CHANNEL} | LM Head: LUT{LM_HEAD_LUT} gs={PER_CHANNEL} | FFN: {FFN_LABEL} gs={FFN_PER_CHANNEL} × {NUM_CHUNKS} chunks")
    print(f"  Chunk partition ([FLLL] {NUM_CHUNKS}-chunk): {CHUNK_RANGES}")
    print(f"  Batch: {BATCH_SIZE} | CTX: {CTX} | Prefill: {prefill_mode} | Compute: {cp_desc}")
    if args._chunk_list is not None:
        print(f"  Chunks: {args._chunk_list}")
    if args.ffn_only or prefill_only:
        print(f"  Mode: {'prefill-only' if prefill_only else 'FFN-only'} (skipping embeddings & lm_head)")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    print("\nLoading model weights...")
    t_load = time.time()
    cfg = Qwen35Config.from_json(os.path.join(args.model, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(args.model), f"Failed to load weights from {args.model}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t_load:.1f}s")

    t_total = time.time()
    print("\n[1/3] FFN Chunks (decode + prefill)")
    if args._chunk_list is not None:
        for ci in args._chunk_list:
            export_ffn_chunks(model, args.output, args.skip_existing,
                              only_chunk=ci, static_prefill=args.static_prefill,
                              lut_bits_override=args.lut_bits, per_channel_override=args.per_channel,
                              compute_precision=cp, v4_precision=v4, selective_fp32=sel_fp32,
                              fp16_attn=d2, e235=e235,
                              delta_fp32=delta_fp32, prefill_only=prefill_only)
    else:
        export_ffn_chunks(model, args.output, args.skip_existing,
                          static_prefill=args.static_prefill,
                          lut_bits_override=args.lut_bits, per_channel_override=args.per_channel,
                          compute_precision=cp, v4_precision=v4, selective_fp32=sel_fp32,
                          fp16_attn=d2, e235=e235,
                          delta_fp32=delta_fp32, prefill_only=prefill_only)
    if not args.ffn_only and not prefill_only:
        print("\n[2/3] Embeddings")
        export_embeddings(model, args.output, args.skip_existing, compute_precision=cp)
        print("\n[3/3] LM Head")
        if args.nosplit_lmhead:
            export_lm_head_nosplit(model, args.output, args.skip_existing, compute_precision=cp)
        else:
            export_lm_head(model, args.output, args.skip_existing, compute_precision=cp)
    else:
        print("\n  Skipping embeddings and lm_head")
    
    del model; gc.collect()

    total_mb = 0
    print(f"\n{'='*70}")
    print("  EXPORT SUMMARY")
    print(f"{'='*70}")
    for f in sorted(os.listdir(args.output)):
        full = os.path.join(args.output, f)
        if os.path.isdir(full):
            sz = sum(os.path.getsize(os.path.join(dp, fn))
                     for dp, _, fns in os.walk(full) for fn in fns
                     if not os.path.islink(os.path.join(dp, fn))) / (1024 * 1024)
            total_mb += sz
            print(f"  {f:<50s} {sz:>8.1f} MB")
    print(f"  {'TOTAL':<50s} {total_mb:>8.1f} MB")
    # Copy tokenizer files so the output dir is self-contained
    tok_patterns = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "special_tokens_map.json"]
    copied = []
    for pat in tok_patterns:
        for src in glob.glob(os.path.join(args.model, pat)):
            dst = os.path.join(args.output, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(os.path.basename(src))
    if copied:
        print(f"  Copied tokenizer files: {', '.join(copied)}")

    print(f"\n  Elapsed: {time.time()-t_total:.1f}s")
    print(f"\nNext: python scripts_qwen3_5/combine.py --input {args.output}")


if __name__ == "__main__":
    main()
