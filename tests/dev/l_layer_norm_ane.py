#!/usr/bin/env python3
"""Test: Does replacing the doubled-concat RMSNorm with a smaller-dim norm improve ANE%?

Hypothesis: The doubled-concat RMSNorm uses F.layer_norm(2*hidden_size), which
may exceed ANE's native layer_norm dimension limit. By using F.layer_norm(hidden_size)
instead, norms could move from CPU to ANE.

Variants tested:
  A) original — doubled concat: cat([x,-x]) → F.layer_norm(2H) → slice
  B) mean_sub — CLAUDE.md approach: x-mean → F.layer_norm(H) with weight
  C) direct  — manual RMSNorm: x / sqrt(mean(x²)+eps) * weight

All variants are applied to BOTH Qwen35RMSNorm and Qwen35RMSNormGated.
"""
import argparse
import gc
import os
import resource
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

REPO_ROOT = "/Volumes/MySSD/Anemll"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
os.chdir(REPO_ROOT)

import coremltools as ct
from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

F_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
OUTPUT_DIR = os.path.join(REPO_ROOT, "artifacts", "l_layer_norm_ane")
HIDDEN = 2560
WARMUP = 10
RUNS = 30


# ── Replacement norms ────────────────────────────────────────────────

class RMSNorm_MeanSub(nn.Module):
    """RMSNorm via mean subtraction + F.layer_norm(H) — per CLAUDE.md ANE guidance.

    Uses F.layer_norm on original dimension (not doubled), which should
    fit within ANE's native layer_norm limit.

    Math: LayerNorm(x - mean(x)) * (1 + weight)
    This is NOT identical to RMSNorm but is close in practice.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        mean = hidden_states.mean(-1, keepdim=True)
        centered = hidden_states - mean
        normed = F.layer_norm(
            centered, (self.hidden_size,), weight=None, bias=None, eps=self.eps
        )
        return normed * (1.0 + self.weight.to(normed.dtype))


class RMSNormGated_MeanSub(nn.Module):
    """RMSNormGated via mean subtraction + F.layer_norm(H)."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states, gate):
        mean = hidden_states.mean(-1, keepdim=True)
        centered = hidden_states - mean
        normed = F.layer_norm(
            centered, (self.hidden_size,), weight=None, bias=None, eps=self.eps
        )
        out = normed * self.weight.to(hidden_states.dtype)
        out = out * F.silu(gate.to(hidden_states.dtype))
        return out


class RMSNorm_Direct(nn.Module):
    """Direct RMSNorm without F.layer_norm — manual sqrt(mean(x²)).

    Uses only elementwise ops + reduce_mean. No layer_norm call.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps)
        return normed * (1.0 + self.weight.to(normed.dtype))


class RMSNormGated_Direct(nn.Module):
    """Direct RMSNormGated without F.layer_norm."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states, gate):
        variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps)
        out = normed * self.weight.to(hidden_states.dtype)
        out = out * F.silu(gate.to(hidden_states.dtype))
        return out


def load_model():
    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


def patch_norms(model, cfg, start_layer, end_layer, variant):
    """Replace norm modules in-place. Returns undo list."""
    if variant == "original":
        return []

    undo = []
    layer_types = cfg.text_config.layer_types

    for li in range(start_layer, end_layer):
        layer = model.model.layers[li]

        # input_layernorm
        orig_in = layer.input_layernorm
        if variant == "mean_sub":
            new_in = RMSNorm_MeanSub(orig_in.hidden_size, orig_in.eps)
        else:
            new_in = RMSNorm_Direct(orig_in.hidden_size, orig_in.eps)
        new_in.weight.data.copy_(orig_in.weight.data)
        new_in = new_in.to(dtype=orig_in.weight.dtype, device=orig_in.weight.device)
        undo.append((layer, 'input_layernorm', orig_in))
        layer.input_layernorm = new_in

        # post_attention_layernorm
        orig_post = layer.post_attention_layernorm
        if variant == "mean_sub":
            new_post = RMSNorm_MeanSub(orig_post.hidden_size, orig_post.eps)
        else:
            new_post = RMSNorm_Direct(orig_post.hidden_size, orig_post.eps)
        new_post.weight.data.copy_(orig_post.weight.data)
        new_post = new_post.to(dtype=orig_post.weight.dtype, device=orig_post.weight.device)
        undo.append((layer, 'post_attention_layernorm', orig_post))
        layer.post_attention_layernorm = new_post

        # RMSNormGated inside linear attention core_norm_stage
        if layer_types[li] == "linear_attention":
            core_norm = layer.self_attn.core_norm_stage
            orig_gated = core_norm.norm
            if variant == "mean_sub":
                new_gated = RMSNormGated_MeanSub(orig_gated.hidden_size, orig_gated.eps)
            else:
                new_gated = RMSNormGated_Direct(orig_gated.hidden_size, orig_gated.eps)
            new_gated.weight.data.copy_(orig_gated.weight.data)
            new_gated = new_gated.to(dtype=orig_gated.weight.dtype, device=orig_gated.weight.device)
            undo.append((core_norm, 'norm', orig_gated))
            core_norm.norm = new_gated

    return undo


def undo_patches(undo):
    for obj, attr, orig in undo:
        setattr(obj, attr, orig)


def export_chunk(model, chunk_idx, label, skip_existing=False):
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
    out_path = os.path.join(OUTPUT_DIR, f"chunk{chunk_idx}_{label}_decode.mlpackage")
    if skip_existing and os.path.exists(out_path):
        print(f"  [skip] {out_path}")
        return out_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
        compute_precision="float16",
    )
    ml = conv.convert_part_2(
        model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
        override_start_layer=CHUNK_RANGES[chunk_idx][0],
        override_end_layer=CHUNK_RANGES[chunk_idx][1],
    )
    ml.save(out_path)
    print(f"  Exported {label} in {time.time() - t0:.1f}s")
    del ml, conv; gc.collect()
    return out_path


def make_decode_inputs(chunk_idx):
    nl = CHUNK_RANGES[chunk_idx][1] - CHUNK_RANGES[chunk_idx][0]
    return {
        "hidden_states": np.random.randn(1, 1, HIDDEN).astype(np.float16),
        "position_ids": np.array([CTX // 2], dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, CTX), dtype=np.float16),
        "current_pos": np.array([CTX // 2], dtype=np.int32),
        "linear_conv_state": np.zeros((nl, 1024, 32), dtype=np.float16),
        "linear_recurrent_state": np.zeros((nl, 32, 128, 128), dtype=np.float16),
    }


def measure_ane(model_path, pred, label):
    try:
        ml = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = ml.make_state()
        for _ in range(WARMUP):
            ml.predict(pred, state=state)
        times, cpus = [], []
        for _ in range(RUNS):
            r0 = resource.getrusage(resource.RUSAGE_SELF)
            t0 = time.perf_counter()
            ml.predict(pred, state=state)
            t1 = time.perf_counter()
            r1 = resource.getrusage(resource.RUSAGE_SELF)
            times.append(t1 - t0)
            cpus.append((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))
        w = np.median(times) * 1000
        c = np.median(cpus) * 1000
        cp = c / w * 100
        ane = max(0, 100 - cp)
        print(f"  {label}: wall={w:.2f}ms  cpu={c:.2f}ms  CPU%={cp:.1f}%  ANE%={ane:.1f}%")
        return ml, state, {"wall_ms": w, "cpu_ms": c, "cpu_pct": cp, "ane_pct": ane}
    except Exception as e:
        print(f"  {label}: FAILED — {e}")
        return None, None, None


def compare_outputs(ml_a, state_a, ml_b, state_b, pred, label_a, label_b):
    out_a = ml_a.predict(pred, state=state_a)
    out_b = ml_b.predict(pred, state=state_b)
    key = "output_hidden_states"
    if key in out_a and key in out_b:
        a = np.asarray(out_a[key]).flatten().astype(np.float64)
        b = np.asarray(out_b[key]).flatten().astype(np.float64)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        max_abs = float(np.max(np.abs(a - b)))
        print(f"  {label_a} vs {label_b}: cos={cos:.8f}  max_abs={max_abs:.6f}")
        return cos
    return None


def run_experiment(args):
    print("=" * 70)
    print("  Norm Implementation vs ANE Utilization")
    print(f"  CTX={CTX}, chunk={args.chunk}")
    print("=" * 70)

    print("\nLoading model...")
    model, cfg = load_model()
    ci = args.chunk
    start, end = CHUNK_RANGES[ci]
    pattern = "".join("F" if l in F_LAYERS else "L" for l in range(start, end))
    print(f"Chunk {ci}: layers {start}-{end-1}, pattern={pattern}")

    variants = args.variants.split(",")
    results = {}
    mls = {}

    for vname in variants:
        print(f"\n── Variant: {vname} ──")

        undo = patch_norms(model, cfg, start, end, vname)
        path = export_chunk(model, ci, vname, skip_existing=args.skip_existing)
        undo_patches(undo)

        pred = make_decode_inputs(ci)
        ml, state, r = measure_ane(path, pred, vname)
        results[vname] = r
        mls[vname] = (ml, state)
        gc.collect()

    # Compare accuracy
    if "original" in mls and mls["original"][0] is not None:
        print(f"\n── Accuracy vs Original ──")
        pred_det = make_decode_inputs(ci)
        pred_det["hidden_states"] = np.ones((1, 1, HIDDEN), dtype=np.float16) * 0.01
        for vname in variants:
            if vname != "original" and mls.get(vname, (None, None))[0] is not None:
                ml_orig, st_orig = mls["original"]
                ml_v, st_v = mls[vname]
                compare_outputs(ml_orig, st_orig, ml_v, st_v, pred_det, "original", vname)

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS — Chunk {ci} ({pattern})")
    print(f"{'='*70}")
    print(f"{'Variant':>14} {'ANE%':>7} {'CPU%':>7} {'Wall(ms)':>10} {'CPU(ms)':>10} {'Δ vs orig':>10}")
    orig_ane = results.get("original", {}).get("ane_pct", 0) if results.get("original") else 0
    for vname in variants:
        r = results.get(vname)
        if r:
            delta = r["ane_pct"] - orig_ane if orig_ane else 0
            print(f"  {vname:>12}  {r['ane_pct']:>5.1f}%  {r['cpu_pct']:>5.1f}%  {r['wall_ms']:>8.2f}  {r['cpu_ms']:>8.2f}  {delta:>+8.1f}%")
        else:
            print(f"  {vname:>12}  FAILED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--variants", type=str, default="original,mean_sub,direct")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
