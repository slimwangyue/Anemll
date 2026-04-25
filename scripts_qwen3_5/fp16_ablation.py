#!/usr/bin/env python3
"""Qwen3.5 4B — FP16 Tensor Family Ablation Experiment.

Evaluates whether keeping specific tensor families in FP16 (vs LUT4)
improves accuracy enough to justify the latency/size cost.

Phases:
  analyze  — Weight-level quantization sensitivity (fast, no CoreML)
  discover — Convert one chunk, discover MIL op names, map families
  ablate   — Per-chunk quality with selective FP16 palettization
  report   — Summarize results and recommend Pareto-optimal policies

Usage:
  python scripts_qwen3_5/fp16_ablation.py analyze  --model models/Qwen__Qwen3.5-4B
  python scripts_qwen3_5/fp16_ablation.py discover --model models/Qwen__Qwen3.5-4B
  python scripts_qwen3_5/fp16_ablation.py ablate   --model models/Qwen__Qwen3.5-4B --output /tmp/ablation
  python scripts_qwen3_5/fp16_ablation.py report   --results-dir /tmp/ablation
"""
import argparse
import gc
import json
import math
import os
import sys
import time
import warnings

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES,
    FFN_PER_CHANNEL, DEFAULT_HF_MODEL,
)

# ── Tensor family definitions ────────────────────────────────────
# Maps: conceptual family name → HF weight name substrings
# HF keys use "self_attn" for F-layers and "linear_attn" for L-layers.
# Our ANEMLL model code maps both to "self_attn" internally.
# MIL op names after tracing use "self_attn" (from our model code).

TENSOR_FAMILIES = {
    # Primary focus
    "attn_q": {
        "hf_patterns": ["self_attn.q_proj.weight"],
        "mil_patterns": ["self_attn_q_proj"],
        "layer_types": {"full_attention"},
        "desc": "Query + gate projection (F-layers; gate is second half of q_proj channels)",
    },
    "ssm_alpha": {
        "hf_patterns": ["linear_attn.in_proj_a.weight", "self_attn.in_proj_a.weight"],
        "mil_patterns": ["self_attn_in_proj_a", "in_proj_a"],
        "layer_types": {"linear_attention"},
        "desc": "SSM alpha projection (L-layers; input to exp(A) state matrix)",
    },
    "ssm_beta": {
        "hf_patterns": ["linear_attn.in_proj_b.weight", "self_attn.in_proj_b.weight"],
        "mil_patterns": ["self_attn_in_proj_b", "in_proj_b"],
        "layer_types": {"linear_attention"},
        "desc": "SSM beta projection (L-layers; forget gate / sigmoid(B))",
    },
    # Secondary
    "attn_kv": {
        "hf_patterns": ["self_attn.k_proj.weight", "self_attn.v_proj.weight"],
        "mil_patterns": ["self_attn_k_proj", "self_attn_v_proj"],
        "layer_types": {"full_attention"},
        "desc": "Key + value projections (F-layers)",
    },
    "attn_o": {
        "hf_patterns": ["self_attn.o_proj.weight"],
        "mil_patterns": ["self_attn_o_proj"],
        "layer_types": {"full_attention"},
        "desc": "Attention output projection (F-layers)",
    },
    "ssm_qkv": {
        "hf_patterns": ["linear_attn.in_proj_qkv.weight", "self_attn.in_proj_qkv.weight"],
        "mil_patterns": ["self_attn_in_proj_qkv", "in_proj_qkv"],
        "layer_types": {"linear_attention"},
        "desc": "SSM combined Q/K/V projection (L-layers)",
    },
    "ssm_z": {
        "hf_patterns": ["linear_attn.in_proj_z.weight", "self_attn.in_proj_z.weight"],
        "mil_patterns": ["self_attn_in_proj_z", "in_proj_z"],
        "layer_types": {"linear_attention"},
        "desc": "SSM gating Z projection (L-layers)",
    },
    "ssm_conv": {
        "hf_patterns": ["linear_attn.conv1d.weight", "linear_attn.conv2d.weight",
                        "self_attn.conv2d.weight"],
        "mil_patterns": ["self_attn_conv2d", "conv2d"],
        "layer_types": {"linear_attention"},
        "desc": "SSM causal convolution (L-layers)",
    },
    "ssm_out": {
        "hf_patterns": ["linear_attn.out_proj.weight", "self_attn.out_proj.weight"],
        "mil_patterns": ["self_attn_out_proj", "out_proj"],
        "layer_types": {"linear_attention"},
        "desc": "SSM output projection (L-layers)",
    },
    "mlp": {
        "hf_patterns": ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"],
        "mil_patterns": ["mlp_gate_proj", "mlp_up_proj", "mlp_down_proj"],
        "layer_types": {"full_attention", "linear_attention"},
        "desc": "MLP gate + up + down projections (all layers)",
    },
    "norms": {
        "hf_patterns": ["input_layernorm.weight", "post_attention_layernorm.weight",
                        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                        "linear_attn.norm.weight", "self_attn.norm.weight",
                        "model.norm.weight"],
        "mil_patterns": ["layernorm", "q_norm", "k_norm"],
        "layer_types": {"full_attention", "linear_attention"},
        "desc": "All normalization weights",
    },
}

# ── Experiment configurations ─────────────────────────────────────
# Each maps config name → list of families to keep in FP16.
# Everything NOT listed gets LUT4.

EXPERIMENT_CONFIGS = {
    # Stage A: Baselines
    "A1_all_lut4": [],
    "A2_all_fp16": list(TENSOR_FAMILIES.keys()),

    # Stage B: Single-family ablations
    "B1_fp16_attn_q": ["attn_q"],
    "B2_fp16_ssm_alpha": ["ssm_alpha"],
    "B3_fp16_ssm_beta": ["ssm_beta"],

    # Stage C: Pairwise combinations
    "C1_fp16_ssm_alpha+beta": ["ssm_alpha", "ssm_beta"],
    "C2_fp16_attn_q+ssm_alpha": ["attn_q", "ssm_alpha"],
    "C3_fp16_attn_q+ssm_beta": ["attn_q", "ssm_beta"],

    # Stage D: Grouped hypotheses
    "D1_fp16_all_primary": ["attn_q", "ssm_alpha", "ssm_beta"],
    "D2_fp16_attn_all": ["attn_q", "attn_kv", "attn_o"],
    "D3_fp16_ssm_all_proj": ["ssm_alpha", "ssm_beta", "ssm_qkv", "ssm_z"],
    "D4_fp16_ssm_small+z": ["ssm_alpha", "ssm_beta", "ssm_z"],

    # Stage E: Targeted (attn_q only in specific layer ranges)
    # These are handled specially in the ablation phase
}


# ═══════════════════════════════════════════════════════════════════
# Phase 1: ANALYZE — weight-level quantization sensitivity
# ═══════════════════════════════════════════════════════════════════

def _load_hf_config(model_path):
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


def _get_layer_types(hf_config):
    """Return per-layer type list from HF config."""
    tc = hf_config.get("text_config", hf_config)
    return tc.get("layer_types", [])


def _classify_weight(key, layer_types):
    """Classify an HF weight key into (layer_idx, layer_type, family_name)."""
    parts = key.split(".")
    # Find layer index
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass
            break

    layer_type = None
    if layer_idx is not None and layer_idx < len(layer_types):
        layer_type = layer_types[layer_idx]

    # Match against families
    for family_name, fdef in TENSOR_FAMILIES.items():
        for pat in fdef["hf_patterns"]:
            if pat in key:
                return layer_idx, layer_type, family_name

    return layer_idx, layer_type, None


def _simulate_lut_error_group(group_flat, n_clusters=16, max_sample=8000):
    """Estimate LUT quantization error for one channel group via mini-batch KMeans."""
    from sklearn.cluster import MiniBatchKMeans

    group_flat = group_flat.astype(np.float32)
    if len(group_flat) <= n_clusters:
        return 0.0, 999.0, 0.0  # trivially quantized

    if len(group_flat) > max_sample:
        idx = np.random.choice(len(group_flat), max_sample, replace=False)
        sample = group_flat[idx]
    else:
        sample = group_flat

    km = MiniBatchKMeans(n_clusters=n_clusters, n_init=1,
                         batch_size=min(512, len(sample)),
                         max_iter=30, random_state=42)
    km.fit(sample.reshape(-1, 1))
    labels = km.predict(group_flat.reshape(-1, 1))
    recon = km.cluster_centers_[labels].flatten()

    mse = np.mean((group_flat - recon) ** 2)
    rmse = math.sqrt(mse)
    rms_sig = math.sqrt(np.mean(group_flat ** 2))
    snr_db = 20 * math.log10(rms_sig / (rmse + 1e-30))
    max_err = float(np.max(np.abs(group_flat - recon)))
    return rmse, snr_db, max_err


def _simulate_family_lut_error(weights_by_family, nbits=4, group_size=4,
                               n_sample_groups=64):
    """Estimate per-family LUT quantization error (sampled)."""
    n_clusters = 2 ** nbits
    results = {}
    for family_name, tensors in weights_by_family.items():
        if not tensors:
            continue
        all_groups = []
        total_elements = 0
        total_bytes_fp16 = 0
        for t in tensors:
            total_elements += t.size
            total_bytes_fp16 += t.size * 2  # FP16 = 2 bytes
            out_ch = t.shape[0]
            gs = min(group_size, out_ch)
            for g_start in range(0, out_ch, gs):
                g_end = min(g_start + gs, out_ch)
                all_groups.append(t[g_start:g_end].flatten())

        # Sample groups for speed
        if len(all_groups) > n_sample_groups:
            sel = np.random.choice(len(all_groups), n_sample_groups, replace=False)
            sampled = [all_groups[i] for i in sel]
        else:
            sampled = all_groups

        # Compute error stats
        rmses, snrs, maxerrs = [], [], []
        for grp in sampled:
            r, s, m = _simulate_lut_error_group(grp, n_clusters=n_clusters)
            rmses.append(r)
            snrs.append(s)
            maxerrs.append(m)

        # LUT4: 4 bits per element + palette overhead (negligible)
        lut_bytes = int(total_elements * nbits / 8)
        # Per-grouped-channel adds a palette per group: n_clusters * 2 bytes each
        n_groups_total = sum(
            max(1, t.shape[0] // group_size) for t in tensors
        )
        palette_bytes = n_groups_total * n_clusters * 2
        lut_total_bytes = lut_bytes + palette_bytes

        results[family_name] = {
            "n_tensors": len(tensors),
            "total_elements": total_elements,
            "total_params_M": total_elements / 1e6,
            "size_fp16_MB": total_bytes_fp16 / 1e6,
            "size_lut4_MB": lut_total_bytes / 1e6,
            "size_delta_MB": (total_bytes_fp16 - lut_total_bytes) / 1e6,
            "avg_rmse": float(np.mean(rmses)),
            "avg_snr_db": float(np.mean(snrs)),
            "max_abs_error": float(np.max(maxerrs)),
            "n_groups_sampled": len(sampled),
        }
    return results


def phase_analyze(args):
    """Phase 1: Weight-level quantization sensitivity analysis."""
    print("=" * 70)
    print("  Phase: ANALYZE — Weight-level quantization sensitivity")
    print("=" * 70)

    hf_config = _load_hf_config(args.model)
    layer_types = _get_layer_types(hf_config)
    n_layers = len(layer_types)
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_linear = sum(1 for t in layer_types if t == "linear_attention")
    print(f"\n  Model: {args.model}")
    print(f"  Layers: {n_layers} total ({n_full} full-attention, {n_linear} linear-attention)")
    print(f"  Layer pattern: {layer_types[:8]}... (repeats)")

    # Load weights
    print("\n  Loading weights from safetensors...")
    t0 = time.time()
    try:
        from safetensors.torch import load_file as _safe_load
        import glob
        import torch as _torch
        shards = sorted(glob.glob(os.path.join(args.model, "*.safetensors")))
        weights = {}
        for shard in shards:
            shard_data = _safe_load(shard, device="cpu")
            for key, t in shard_data.items():
                weights[key] = t.to(_torch.float32).numpy()
            del shard_data
        print(f"  Loaded {len(weights)} tensors from {len(shards)} shards ({time.time()-t0:.1f}s)")
    except ImportError:
        print("  ERROR: safetensors not installed. pip install safetensors")
        return

    # Classify weights into families
    weights_by_family = {name: [] for name in TENSOR_FAMILIES}
    unclassified = []
    for key, tensor in weights.items():
        _, _, family = _classify_weight(key, layer_types)
        if family:
            weights_by_family[family].append(tensor)
        else:
            unclassified.append(key)

    print(f"\n  Weight classification:")
    for fname, tensors in weights_by_family.items():
        if tensors:
            total = sum(t.size for t in tensors)
            shapes = set(t.shape for t in tensors)
            print(f"    {fname:20s}: {len(tensors):3d} tensors, {total/1e6:8.2f}M params, shapes={shapes}")
    if unclassified:
        print(f"    [unclassified]:     {len(unclassified)} tensors")
        for k in unclassified[:5]:
            print(f"      {k}")

    # Compute quantization sensitivity
    print(f"\n  Simulating LUT4 quantization error (gs={FFN_PER_CHANNEL}, sampling 64 groups/family)...")
    t0 = time.time()
    family_stats = _simulate_family_lut_error(
        weights_by_family, nbits=4, group_size=FFN_PER_CHANNEL, n_sample_groups=64
    )
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ── Results table ──
    print("\n" + "=" * 120)
    print(f"  {'Family':<20s} {'Tensors':>7s} {'Params(M)':>10s} {'FP16(MB)':>10s} "
          f"{'LUT4(MB)':>10s} {'Delta(MB)':>10s} {'RMSE':>10s} {'SNR(dB)':>10s} {'MaxErr':>10s}")
    print("-" * 120)

    total_fp16 = 0
    total_lut4 = 0
    sorted_families = sorted(family_stats.items(), key=lambda x: x[1]["avg_snr_db"])
    for fname, stats in sorted_families:
        total_fp16 += stats["size_fp16_MB"]
        total_lut4 += stats["size_lut4_MB"]
        print(f"  {fname:<20s} {stats['n_tensors']:>7d} {stats['total_params_M']:>10.2f} "
              f"{stats['size_fp16_MB']:>10.2f} {stats['size_lut4_MB']:>10.2f} "
              f"{stats['size_delta_MB']:>10.2f} {stats['avg_rmse']:>10.6f} "
              f"{stats['avg_snr_db']:>10.1f} {stats['max_abs_error']:>10.6f}")

    print("-" * 120)
    print(f"  {'TOTAL':<20s} {'':>7s} {'':>10s} {total_fp16:>10.2f} {total_lut4:>10.2f} "
          f"{total_fp16 - total_lut4:>10.2f}")
    print("=" * 120)

    # ── Latency impact analysis ──
    print("\n  LATENCY IMPACT (model size overhead of keeping family in FP16):")
    print(f"  {'Config':<45s} {'Extra MB':>10s} {'% of LUT4':>10s} {'Quality Gain':>15s}")
    print("-" * 90)

    total_lut4_size = total_lut4
    for config_name, fp16_families in sorted(EXPERIMENT_CONFIGS.items()):
        if not fp16_families:
            print(f"  {config_name:<45s} {'0.00':>10s} {'0.0%':>10s} {'baseline':>15s}")
            continue
        extra = sum(family_stats.get(f, {}).get("size_delta_MB", 0) for f in fp16_families)
        pct = 100.0 * extra / total_lut4_size if total_lut4_size > 0 else 0
        # Quality gain proxy: sum of SNR improvements
        worst_snr = min(
            family_stats.get(f, {}).get("avg_snr_db", 999) for f in fp16_families
            if f in family_stats
        ) if any(f in family_stats for f in fp16_families) else 999
        print(f"  {config_name:<45s} {extra:>10.2f} {pct:>9.1f}% {f'fixes SNR {worst_snr:.0f}dB':>15s}")

    print()

    # ── Key findings ──
    print("  KEY FINDINGS:")
    print("  " + "-" * 60)

    # Find tiny families (< 1MB overhead)
    tiny = [(f, s) for f, s in family_stats.items() if s["size_delta_MB"] < 1.0]
    if tiny:
        print(f"  FREE FP16 (< 1MB overhead):")
        for f, s in tiny:
            print(f"    {f}: +{s['size_delta_MB']:.2f}MB, SNR={s['avg_snr_db']:.1f}dB")

    # Find most sensitive (lowest SNR)
    print(f"\n  MOST SENSITIVE TO LUT4 (lowest SNR):")
    for fname, stats in sorted_families[:5]:
        print(f"    {fname}: SNR={stats['avg_snr_db']:.1f}dB, RMSE={stats['avg_rmse']:.6f}")

    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        results_path = os.path.join(args.output, "analyze_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "family_stats": family_stats,
                "model_path": args.model,
                "lut_bits": 4,
                "group_size": FFN_PER_CHANNEL,
            }, f, indent=2)
        print(f"\n  Results saved to {results_path}")

    del weights
    gc.collect()
    return family_stats


# ═══════════════════════════════════════════════════════════════════
# Phase 2: DISCOVER — MIL op name mapping
# ═══════════════════════════════════════════════════════════════════

def _load_model(model_path):
    """Load Qwen3.5 model for CoreML conversion."""
    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
    cfg = Qwen35Config.from_json(os.path.join(model_path, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(model_path), f"Failed to load weights from {model_path}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _convert_chunk_fp16(model, chunk_idx):
    """Convert a single chunk to CoreML FP16 (no LUT) and return the MLModel."""
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    sl, el = CHUNK_RANGES[chunk_idx]
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=FFN_PER_CHANNEL,
        compute_precision="float16",
    )
    mlmodel = conv.convert_part_2(
        model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
        override_start_layer=sl, override_end_layer=el,
    )
    return mlmodel


def discover_weight_ops(mlmodel):
    """Discover weight ops in a CoreML model via get_weights_metadata (preferred)
    or MIL program iteration (fallback).

    Returns dict: op_name → {type, weight_shape, matched_family}
    """
    # Preferred: use get_weights_metadata which gives clean named weight constants
    try:
        import coremltools.optimize.coreml as cto_coreml
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metadata = cto_coreml.get_weights_metadata(mlmodel, weight_threshold=0)
        weight_ops = {}
        for op_name, meta in metadata.items():
            try:
                shape = tuple(meta.val.shape) if hasattr(meta, 'val') else ()
            except (AttributeError, TypeError):
                continue  # Skip ops where val is not a proper array
            matched_family = _match_op_to_family(op_name)
            weight_ops[op_name] = {
                "type": "weight_const",
                "weight_shape": shape,
                "family": matched_family,
            }
        # Check if we got meaningful family matches (not all UNKNOWN)
        n_matched = sum(1 for v in weight_ops.values() if v.get("family"))
        if n_matched > 0:
            return weight_ops
    except Exception as e:
        print(f"  get_weights_metadata failed: {e}")

    # Fallback: iterate MIL program ops and check for weight const inputs
    if not hasattr(mlmodel, '_mil_program') or mlmodel._mil_program is None:
        return {}

    prog = mlmodel._mil_program
    weight_ops = {}
    for fn_name, fn in prog.functions.items():
        for op in fn.operations:
            # Look for const ops that look like weight tensors (large, 2D+)
            if op.op_type == 'const' and hasattr(op, 'outputs'):
                for out in op.outputs:
                    if hasattr(out, 'shape') and len(out.shape) >= 2 and max(out.shape) > 100:
                        matched_family = _match_op_to_family(op.name)
                        weight_ops[op.name] = {
                            "type": "const",
                            "weight_shape": tuple(out.shape),
                            "family": matched_family,
                        }
            elif op.op_type in ('conv', 'linear', 'matmul'):
                out_shapes = []
                for out in op.outputs:
                    if hasattr(out, 'shape'):
                        out_shapes.append(tuple(out.shape))
                matched_family = _match_op_to_family(op.name)
                weight_ops[op.name] = {
                    "type": op.op_type,
                    "weight_shape": out_shapes[0] if out_shapes else (),
                    "family": matched_family,
                }
    return weight_ops


def _match_op_to_family(op_name):
    """Match a MIL op/weight name to a tensor family."""
    op_lower = op_name.lower()
    for fam_name, fdef in TENSOR_FAMILIES.items():
        for pat in fdef.get("mil_patterns", []):
            if pat.lower() in op_lower:
                return fam_name
        for pat in fdef["hf_patterns"]:
            mil_pat = pat.replace(".", "_").lower().replace("_weight", "")
            if mil_pat in op_lower:
                return fam_name
    return None


def phase_discover(args):
    """Phase 2: Discover MIL op names and map to tensor families."""
    print("=" * 70)
    print("  Phase: DISCOVER — MIL op name → tensor family mapping")
    print("=" * 70)

    print("\n  Loading model...")
    t0 = time.time()
    model = _load_model(args.model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Convert chunk 0 (L-layers only: layers 0-2) and chunk 1 (F+L: layers 3-6)
    all_ops = {}
    for ci in [0, 1]:
        sl, el = CHUNK_RANGES[ci]
        layer_types_str = ", ".join(
            _get_layer_types(_load_hf_config(args.model))[sl:el]
        )
        print(f"\n  Converting chunk {ci} (layers {sl}-{el-1}: {layer_types_str})...")
        t0 = time.time()
        mlmodel = _convert_chunk_fp16(model, ci)
        print(f"  Converted in {time.time()-t0:.1f}s")

        # Discover ops
        ops = discover_weight_ops(mlmodel)

        print(f"  Found {len(ops)} weight ops:")
        family_counts = {}
        for op_name, info in sorted(ops.items()):
            fam = info.get("family") or "UNKNOWN"
            family_counts[fam] = family_counts.get(fam, 0) + 1
            shapes_str = str(info.get("weight_shape", ""))
            print(f"    {op_name[:70]:<70s} → {fam:<15s} {shapes_str}")

        print(f"\n  Family summary for chunk {ci}:")
        for fam, count in sorted(family_counts.items()):
            print(f"    {fam:<20s}: {count} ops")

        all_ops[f"chunk{ci}"] = ops
        del mlmodel
        gc.collect()

    # Save mapping
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        map_path = os.path.join(args.output, "op_family_mapping.json")
        serializable = {}
        for chunk_key, ops in all_ops.items():
            serializable[chunk_key] = {
                name: {k: (str(v) if not isinstance(v, (str, list, dict, type(None))) else v)
                       for k, v in info.items()}
                for name, info in ops.items()
            }
        with open(map_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Mapping saved to {map_path}")

    del model
    gc.collect()
    return all_ops


# ═══════════════════════════════════════════════════════════════════
# Phase 3: ABLATE — Per-chunk quality with selective palettization
# ═══════════════════════════════════════════════════════════════════

def _build_selective_lut_config(mlmodel, fp16_families, lut_bits=4, per_channel=4,
                               lut6_families=None, gs2_families=None,
                               lut6_gs2_families=None):
    """Build an OptimizationConfig that applies LUT to all ops EXCEPT those in fp16_families.

    Args:
        mlmodel: CoreML MLModel (freshly converted, _mil_program available)
        fp16_families: list of family names to keep in FP16
        lut_bits: LUT bit width for quantized ops
        per_channel: group size for per-channel palettization
        lut6_families: families to use LUT6 (instead of default lut_bits)
        gs2_families: families to use group_size=2 (instead of default per_channel)
        lut6_gs2_families: families to use LUT6 + group_size=2

    Returns:
        cto.coreml.OptimizationConfig ready for palettize_weights()
    """
    import coremltools.optimize as cto

    # Build global LUT config
    if per_channel > 0:
        global_cfg = cto.coreml.OpPalettizerConfig(
            mode="kmeans", nbits=lut_bits,
            granularity="per_grouped_channel", group_size=per_channel,
            num_kmeans_workers=1,
        )
    else:
        global_cfg = cto.coreml.OpPalettizerConfig(
            mode="kmeans", nbits=lut_bits,
            granularity="per_tensor",
            num_kmeans_workers=1,
        )

    config = cto.coreml.OptimizationConfig(global_config=global_cfg)

    has_overrides = fp16_families or lut6_families or gs2_families or lut6_gs2_families
    if not has_overrides:
        return config  # All default LUT, no overrides

    # Discover weight ops and apply per-family overrides
    fp16_set = set(fp16_families or [])
    lut6_set = set(lut6_families or [])
    gs2_set = set(gs2_families or [])
    lut6_gs2_set = set(lut6_gs2_families or [])

    ops = discover_weight_ops(mlmodel)

    skipped = 0
    for op_name, info in ops.items():
        family = info.get("family")
        if not family:
            continue

        if family in fp16_set:
            config.set_op_name(op_name, None)  # Skip LUT for this op
            skipped += 1
        elif family in lut6_gs2_set:
            config.set_op_name(op_name, cto.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=6,
                granularity="per_grouped_channel", group_size=2,
                num_kmeans_workers=1,
            ))
        elif family in lut6_set:
            config.set_op_name(op_name, cto.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=6,
                granularity="per_grouped_channel", group_size=per_channel,
                num_kmeans_workers=1,
            ))
        elif family in gs2_set:
            config.set_op_name(op_name, cto.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=lut_bits,
                granularity="per_grouped_channel", group_size=2,
                num_kmeans_workers=1,
            ))

    return config


def _evaluate_chunk_quality(mlmodel_fp16, mlmodel_quant, chunk_idx, n_eval_samples=8):
    """Compare FP16 vs quantized chunk outputs on random inputs.

    Returns dict with quality metrics.
    """
    import coremltools as ct

    # Get model input spec
    spec = mlmodel_fp16.get_spec()
    fn_inputs = None
    for fn in spec.description.functions:
        if fn.name == "infer":
            fn_inputs = fn.input
            break
    if fn_inputs is None:
        fn_inputs = spec.description.input

    # Build input dict from spec
    input_shapes = {}
    for inp in fn_inputs:
        try:
            shape = tuple(inp.type.multiArrayType.shape)
            input_shapes[inp.name] = shape
        except Exception:
            pass

    # Run evaluations
    cos_sims = []
    mses = []
    max_errs = []
    np.random.seed(42)

    for sample_i in range(n_eval_samples):
        inputs = {}
        for name, shape in input_shapes.items():
            if name in ("position_ids", "current_pos"):
                inputs[name] = np.array([sample_i % CTX], dtype=np.int32)
            elif name == "causal_mask":
                mask = np.full(shape, -65504.0, dtype=np.float16)
                mask[:, :, :, :sample_i + 1] = 0
                inputs[name] = mask
            else:
                inputs[name] = np.random.randn(*shape).astype(np.float16) * 0.1

        # Get FP16 reference
        try:
            state_fp16 = mlmodel_fp16.make_state()
            out_fp16 = mlmodel_fp16.predict(inputs, state=state_fp16)
        except Exception as e:
            try:
                out_fp16 = mlmodel_fp16.predict(inputs)
            except Exception:
                print(f"    Warning: chunk {chunk_idx} predict failed: {e}")
                continue

        # Get quantized output
        try:
            state_q = mlmodel_quant.make_state()
            out_q = mlmodel_quant.predict(inputs, state=state_q)
        except Exception:
            try:
                out_q = mlmodel_quant.predict(inputs)
            except Exception:
                continue

        # Compare hidden_states output
        key = "output_hidden_states"
        if key not in out_fp16 or key not in out_q:
            continue

        ref = out_fp16[key].flatten().astype(np.float64)
        tst = out_q[key].flatten().astype(np.float64)

        # Cosine similarity
        dot = np.dot(ref, tst)
        norm_r = np.linalg.norm(ref)
        norm_t = np.linalg.norm(tst)
        if norm_r > 0 and norm_t > 0:
            cos_sims.append(dot / (norm_r * norm_t))

        # MSE & max error
        diff = ref - tst
        mses.append(float(np.mean(diff ** 2)))
        max_errs.append(float(np.max(np.abs(diff))))

    if not cos_sims:
        return {"error": "no successful evaluations"}

    return {
        "avg_cosine_sim": float(np.mean(cos_sims)),
        "min_cosine_sim": float(np.min(cos_sims)),
        "avg_mse": float(np.mean(mses)),
        "avg_max_error": float(np.mean(max_errs)),
        "n_samples": len(cos_sims),
    }


def phase_ablate(args):
    """Phase 3: Per-chunk ablation with selective FP16 palettization."""
    import coremltools as ct
    import coremltools.optimize as cto

    print("=" * 70)
    print("  Phase: ABLATE — Per-chunk quality with selective FP16")
    print("=" * 70)

    configs_to_run = args.configs.split(",") if args.configs else list(EXPERIMENT_CONFIGS.keys())
    # Filter to only valid configs
    configs_to_run = [c for c in configs_to_run if c in EXPERIMENT_CONFIGS]
    chunks_to_run = list(range(NUM_CHUNKS))
    if args.chunks:
        chunks_to_run = [int(x) for x in args.chunks.split(",")]

    print(f"\n  Model: {args.model}")
    print(f"  Configs: {len(configs_to_run)} — {', '.join(configs_to_run)}")
    print(f"  Chunks: {chunks_to_run}")
    print(f"  Eval samples per chunk: {args.n_eval}")

    print("\n  Loading model...")
    t0 = time.time()
    model = _load_model(args.model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    results = {}
    os.makedirs(args.output, exist_ok=True)

    for ci in chunks_to_run:
        sl, el = CHUNK_RANGES[ci]
        print(f"\n{'='*60}")
        print(f"  Chunk {ci}: layers [{sl}-{el-1}]")
        print(f"{'='*60}")

        # Convert to FP16 (no LUT) — the reference
        print(f"  Converting FP16 reference...")
        t0 = time.time()
        fp16_model = _convert_chunk_fp16(model, ci)
        t_convert = time.time() - t0
        print(f"  FP16 conversion: {t_convert:.1f}s")

        for config_name in configs_to_run:
            fp16_families = EXPERIMENT_CONFIGS[config_name]
            print(f"\n  Config: {config_name} (FP16: {fp16_families or 'none'})")

            if config_name == "A2_all_fp16":
                # No palettization — use FP16 model directly
                quant_model = fp16_model
                metrics = {"skip": "fp16 baseline, no palettization"}
                # Still evaluate vs itself for sanity
                quality = {"avg_cosine_sim": 1.0, "avg_mse": 0.0,
                           "avg_max_error": 0.0, "n_samples": 0}
            else:
                # Build selective config and palettize
                print(f"    Building selective LUT4 config...")
                t0 = time.time()
                opt_config = _build_selective_lut_config(
                    fp16_model, fp16_families,
                    lut_bits=4, per_channel=FFN_PER_CHANNEL,
                )

                print(f"    Palettizing...")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        quant_model = cto.coreml.palettize_weights(fp16_model, opt_config)
                except Exception as e:
                    print(f"    ERROR: palettization failed: {e}")
                    results[(config_name, ci)] = {"error": str(e)}
                    continue
                t_palettize = time.time() - t0
                print(f"    Palettized in {t_palettize:.1f}s")

                # Evaluate quality
                print(f"    Evaluating quality ({args.n_eval} samples)...")
                t0 = time.time()
                quality = _evaluate_chunk_quality(fp16_model, quant_model, ci, args.n_eval)
                t_eval = time.time() - t0
                print(f"    Evaluated in {t_eval:.1f}s")

                if quant_model is not fp16_model:
                    del quant_model

            cos = quality.get("avg_cosine_sim", 0)
            mse = quality.get("avg_mse", 0)
            print(f"    → cos_sim={cos:.6f}, mse={mse:.8f}, max_err={quality.get('avg_max_error', 0):.6f}")

            results[(config_name, ci)] = quality

        del fp16_model
        gc.collect()

        # Save incremental results
        save_results = {}
        for (cfg, chunk), metrics in results.items():
            key = f"{cfg}__chunk{chunk}"
            save_results[key] = metrics
        with open(os.path.join(args.output, "ablation_results.json"), "w") as f:
            json.dump(save_results, f, indent=2)

    del model
    gc.collect()

    # Print summary
    _print_ablation_summary(results, configs_to_run, chunks_to_run)
    return results


def _print_ablation_summary(results, configs, chunks):
    """Print a summary table of ablation results."""
    print("\n" + "=" * 100)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 100)

    # Per-config average across chunks
    print(f"\n  {'Config':<40s} {'Avg CosSim':>12s} {'Avg MSE':>12s} {'Avg MaxErr':>12s}")
    print("-" * 80)

    config_avgs = {}
    for config_name in configs:
        cos_sims = []
        mses = []
        max_errs = []
        for ci in chunks:
            q = results.get((config_name, ci), {})
            if "avg_cosine_sim" in q:
                cos_sims.append(q["avg_cosine_sim"])
                mses.append(q.get("avg_mse", 0))
                max_errs.append(q.get("avg_max_error", 0))

        if cos_sims:
            avg_cos = float(np.mean(cos_sims))
            avg_mse = float(np.mean(mses))
            avg_maxerr = float(np.mean(max_errs))
            config_avgs[config_name] = {
                "avg_cosine_sim": avg_cos,
                "avg_mse": avg_mse,
                "avg_max_error": avg_maxerr,
            }
            print(f"  {config_name:<40s} {avg_cos:>12.6f} {avg_mse:>12.8f} {avg_maxerr:>12.6f}")

    print("-" * 80)

    # Rank by quality
    ranked = sorted(config_avgs.items(), key=lambda x: -x[1]["avg_cosine_sim"])
    print(f"\n  RANKED BY QUALITY (highest cosine similarity):")
    for i, (name, stats) in enumerate(ranked):
        fp16_fams = EXPERIMENT_CONFIGS.get(name, [])
        print(f"    #{i+1}: {name:<40s} cos={stats['avg_cosine_sim']:.6f} "
              f"({'FP16: ' + ', '.join(fp16_fams) if fp16_fams else 'all LUT4'})")


# ═══════════════════════════════════════════════════════════════════
# Phase 4: REPORT — Compile results and recommend
# ═══════════════════════════════════════════════════════════════════

def phase_report(args):
    """Phase 4: Load results and produce final recommendations."""
    print("=" * 70)
    print("  Phase: REPORT — Pareto analysis and recommendations")
    print("=" * 70)

    results_dir = args.results_dir or args.output
    if not results_dir:
        print("  ERROR: --results-dir or --output required")
        return

    # Load analyze results
    analyze_path = os.path.join(results_dir, "analyze_results.json")
    ablation_path = os.path.join(results_dir, "ablation_results.json")

    analyze_data = None
    if os.path.exists(analyze_path):
        with open(analyze_path) as f:
            analyze_data = json.load(f)
        print(f"  Loaded analyze results from {analyze_path}")

    ablation_data = None
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation_data = json.load(f)
        print(f"  Loaded ablation results from {ablation_path}")

    if not analyze_data and not ablation_data:
        print("  ERROR: No results found. Run 'analyze' and/or 'ablate' first.")
        return

    family_stats = analyze_data.get("family_stats", {}) if analyze_data else {}

    # ── Pareto analysis ──
    print("\n" + "=" * 100)
    print("  PARETO ANALYSIS: Size overhead vs Quality gain")
    print("=" * 100)

    if ablation_data:
        # Parse ablation results
        config_metrics = {}
        for key, metrics in ablation_data.items():
            parts = key.rsplit("__chunk", 1)
            if len(parts) == 2:
                config_name = parts[0]
                if config_name not in config_metrics:
                    config_metrics[config_name] = []
                if "avg_cosine_sim" in metrics:
                    config_metrics[config_name].append(metrics)

        # Compute per-config averages
        print(f"\n  {'Config':<40s} {'Overhead(MB)':>12s} {'Overhead%':>10s} "
              f"{'Avg CosSim':>12s} {'Δ vs LUT4':>10s} {'Pareto?':>8s}")
        print("-" * 100)

        baseline_cos = 0.0
        total_lut4_mb = sum(s.get("size_lut4_MB", 0) for s in family_stats.values())

        pareto_points = []
        for config_name in sorted(EXPERIMENT_CONFIGS.keys()):
            if config_name not in config_metrics:
                continue

            fp16_families = EXPERIMENT_CONFIGS[config_name]
            overhead_mb = sum(family_stats.get(f, {}).get("size_delta_MB", 0) for f in fp16_families)
            overhead_pct = 100.0 * overhead_mb / total_lut4_mb if total_lut4_mb > 0 else 0

            chunk_cos = [m["avg_cosine_sim"] for m in config_metrics[config_name]]
            avg_cos = float(np.mean(chunk_cos)) if chunk_cos else 0

            if config_name == "A1_all_lut4":
                baseline_cos = avg_cos
            delta_cos = avg_cos - baseline_cos

            pareto_points.append((config_name, overhead_mb, avg_cos, delta_cos))

            print(f"  {config_name:<40s} {overhead_mb:>12.2f} {overhead_pct:>9.1f}% "
                  f"{avg_cos:>12.6f} {delta_cos:>+10.6f}")

        # Find Pareto frontier
        print("\n" + "=" * 100)
        print("  PARETO FRONTIER (configs not dominated by any other)")
        print("=" * 100)
        pareto_front = []
        for name, overhead, cos, delta in pareto_points:
            dominated = False
            for name2, overhead2, cos2, delta2 in pareto_points:
                if name2 != name and overhead2 <= overhead and cos2 >= cos and (overhead2 < overhead or cos2 > cos):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append((name, overhead, cos, delta))
                print(f"  ★ {name:<40s} overhead={overhead:.2f}MB  cos_sim={cos:.6f}  Δcos={delta:+.6f}")

    # ── Final recommendations ──
    print("\n" + "=" * 100)
    print("  DEPLOYMENT RECOMMENDATIONS")
    print("=" * 100)

    print("""
  These recommendations should be updated after running the full ablation.

  PRELIMINARY RECOMMENDATIONS (based on weight analysis):

  1. LATENCY-FIRST Policy:
     - Keep ssm_alpha + ssm_beta in FP16 (essentially free: < 1MB overhead)
     - Everything else: LUT4 gs=4
     - Expected impact: ~0% latency increase, potentially measurable quality gain

  2. BALANCED Policy:
     - Keep ssm_alpha + ssm_beta + attn_q in FP16
     - Everything else: LUT4 gs=4
     - Expected impact: small latency increase (q_proj is moderate size)

  3. ACCURACY-FIRST Policy:
     - Keep all SSM small projections (alpha, beta, z) + attn_q in FP16
     - Everything else: LUT4 gs=4
     - Expected impact: moderate latency increase, best quality

  NOTE: Run 'ablate' phase to get measured quality deltas for precise recommendations.
""")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 4B — FP16 Tensor Family Ablation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("phase", choices=["analyze", "discover", "ablate", "report"],
                        help="Experiment phase to run")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--output", default=None,
                        help="Output directory for results")
    parser.add_argument("--results-dir", default=None,
                        help="Results directory for report phase")
    parser.add_argument("--configs", default=None,
                        help="Comma-separated config names to run (default: all)")
    parser.add_argument("--chunks", default=None,
                        help="Comma-separated chunk indices to test (default: all)")
    parser.add_argument("--n-eval", type=int, default=8,
                        help="Number of evaluation samples per chunk (default: 8)")
    args = parser.parse_args()

    if args.phase == "analyze":
        phase_analyze(args)
    elif args.phase == "discover":
        phase_discover(args)
    elif args.phase == "ablate":
        if not args.output:
            args.output = os.path.join(_REPO_ROOT, "ablation_results")
        phase_ablate(args)
    elif args.phase == "report":
        phase_report(args)


if __name__ == "__main__":
    main()
