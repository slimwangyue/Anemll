#!/usr/bin/env python3
"""ANE-optimized L-layer recurrence v2: minimize MIL ops for better ANE partitioning.

Changes from original:
  1. squeeze(1) instead of transpose(1,2) — eliminates transpose ops
  2. torch.matmul on 4D tensors — no reshape/bmm overhead
  3. No for loop — inlined for seq_len=1 (decode)
  4. No contiguous() — reduce copy ops
  5. Direct output computation — no allocate-then-index pattern

Also tests:
  - torch.einsum variant
  - Original for baseline comparison
"""
import argparse
import gc
import os
import resource
import sys
import time
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
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
OUTPUT_DIR = os.path.join(REPO_ROOT, "artifacts", "l_layer_ane_v2")
HIDDEN = 2560
WARMUP = 10
RUNS = 30


def _l2norm(x, dim=-1, eps=1e-6):
    sq_sum = (x * x).sum(dim=dim, keepdim=True)
    sq_sum = torch.clamp(sq_sum, min=eps)
    return x * torch.rsqrt(sq_sum)


# ── Variant A: Original (baseline) ──────────────────────────────────
def _recurrent_original(query, key, value, g, beta, recurrent_state,
                        output_final_state=True,
                        expected_batch_size=None, expected_num_heads=None,
                        expected_seq_len=None, expected_k_dim=None,
                        expected_v_dim=None, math_dtype=torch.float32):
    """Original implementation — for baseline measurement."""
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    query = _l2norm(query, dim=-1)
    key = _l2norm(key, dim=-1)
    bsz = expected_batch_size if expected_batch_size is not None else key.shape[0]
    n_heads = expected_num_heads if expected_num_heads is not None else key.shape[1]
    seq_len = expected_seq_len if expected_seq_len is not None else key.shape[2]
    k_dim = expected_k_dim if expected_k_dim is not None else key.shape[-1]
    v_dim = expected_v_dim if expected_v_dim is not None else value.shape[-1]
    scale = 1 / (k_dim ** 0.5)
    query = query * scale
    out = torch.zeros(bsz, n_heads, seq_len, v_dim, dtype=value.dtype, device=value.device)
    state = recurrent_state.to(value)
    for i in range(seq_len):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        state = None
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state


# ── Variant B: matmul4d — no reshape, use 4D matmul ────────────────
def _recurrent_matmul4d(query, key, value, g, beta, recurrent_state,
                        output_final_state=True,
                        expected_batch_size=None, expected_num_heads=None,
                        expected_seq_len=None, expected_k_dim=None,
                        expected_v_dim=None, math_dtype=torch.float32):
    """Use torch.matmul on 4D tensors — keeps batch+head dims, no reshape."""
    initial_dtype = query.dtype
    # squeeze(1) for seq_len=1 → (B, H, D) — no transpose!
    q = query.squeeze(1).to(math_dtype)
    k = key.squeeze(1).to(math_dtype)
    v = value.squeeze(1).to(math_dtype)
    b = beta.squeeze(1).to(math_dtype)
    g_val = g.squeeze(1).to(math_dtype)

    q = _l2norm(q, dim=-1)
    k = _l2norm(k, dim=-1)
    k_dim = expected_k_dim if expected_k_dim is not None else k.shape[-1]
    q = q * (1.0 / (k_dim ** 0.5))

    state = recurrent_state.to(math_dtype)  # (B, H, K, V)

    # Decay
    g_exp = g_val.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
    state = state * g_exp

    # kv_mem = state^T @ k: (B,H,V,K) @ (B,H,K,1) → (B,H,V,1) → squeeze
    kv_mem = torch.matmul(state.transpose(-2, -1), k.unsqueeze(-1)).squeeze(-1)

    # Delta
    delta = (v - kv_mem) * b.unsqueeze(-1)

    # Outer product: k @ delta^T: (B,H,K,1) @ (B,H,1,V) → (B,H,K,V)
    state = state + torch.matmul(k.unsqueeze(-1), delta.unsqueeze(-2))

    # Output: state^T @ q: (B,H,V,K) @ (B,H,K,1) → (B,H,V,1) → squeeze
    out = torch.matmul(state.transpose(-2, -1), q.unsqueeze(-1)).squeeze(-1)

    # Reshape to (B, 1, H, V) to match expected output format
    out = out.unsqueeze(1).to(initial_dtype)
    if not output_final_state:
        state = None
    return out, state


# ── Variant C: einsum — let torch optimize contractions ─────────────
def _recurrent_einsum(query, key, value, g, beta, recurrent_state,
                      output_final_state=True,
                      expected_batch_size=None, expected_num_heads=None,
                      expected_seq_len=None, expected_k_dim=None,
                      expected_v_dim=None, math_dtype=torch.float32):
    """Use torch.einsum — potentially better fusion in MIL."""
    initial_dtype = query.dtype
    q = query.squeeze(1).to(math_dtype)
    k = key.squeeze(1).to(math_dtype)
    v = value.squeeze(1).to(math_dtype)
    b = beta.squeeze(1).to(math_dtype)
    g_val = g.squeeze(1).to(math_dtype)

    q = _l2norm(q, dim=-1)
    k = _l2norm(k, dim=-1)
    k_dim = expected_k_dim if expected_k_dim is not None else k.shape[-1]
    q = q * (1.0 / (k_dim ** 0.5))

    state = recurrent_state.to(math_dtype)

    # Decay
    g_exp = g_val.exp().unsqueeze(-1).unsqueeze(-1)
    state = state * g_exp

    # kv_mem = sum_k(state[b,h,k,v] * k[b,h,k]) = state^T @ k
    kv_mem = torch.einsum('bhkv,bhk->bhv', state, k)

    # Delta
    delta = (v - kv_mem) * b.unsqueeze(-1)

    # Outer product
    state = state + torch.einsum('bhk,bhv->bhkv', k, delta)

    # Output
    out = torch.einsum('bhkv,bhk->bhv', state, q)

    out = out.unsqueeze(1).to(initial_dtype)
    if not output_final_state:
        state = None
    return out, state


# ── Variant D: no_norm — skip l2norm to see its ANE impact ─────────
def _recurrent_no_norm(query, key, value, g, beta, recurrent_state,
                       output_final_state=True,
                       expected_batch_size=None, expected_num_heads=None,
                       expected_seq_len=None, expected_k_dim=None,
                       expected_v_dim=None, math_dtype=torch.float32):
    """Skip l2norm — to measure its ANE impact. NOT for production."""
    initial_dtype = query.dtype
    q = query.squeeze(1).to(math_dtype)
    k = key.squeeze(1).to(math_dtype)
    v = value.squeeze(1).to(math_dtype)
    b = beta.squeeze(1).to(math_dtype)
    g_val = g.squeeze(1).to(math_dtype)

    # No l2norm!
    k_dim = expected_k_dim if expected_k_dim is not None else k.shape[-1]
    q = q * (1.0 / (k_dim ** 0.5))

    state = recurrent_state.to(math_dtype)
    g_exp = g_val.exp().unsqueeze(-1).unsqueeze(-1)
    state = state * g_exp
    kv_mem = torch.matmul(state.transpose(-2, -1), k.unsqueeze(-1)).squeeze(-1)
    delta = (v - kv_mem) * b.unsqueeze(-1)
    state = state + torch.matmul(k.unsqueeze(-1), delta.unsqueeze(-2))
    out = torch.matmul(state.transpose(-2, -1), q.unsqueeze(-1)).squeeze(-1)

    out = out.unsqueeze(1).to(initial_dtype)
    if not output_final_state:
        state = None
    return out, state


# ── Variant E: projections only — measure ANE of wrapper sans recurrence ──
def _recurrent_passthrough(query, key, value, g, beta, recurrent_state,
                           output_final_state=True,
                           expected_batch_size=None, expected_num_heads=None,
                           expected_seq_len=None, expected_k_dim=None,
                           expected_v_dim=None, math_dtype=torch.float32):
    """Pass-through: return zeros for output, keep state unchanged.
    Measures ANE% of the projections+MLP wrapper without recurrence overhead.
    NOT for production — accuracy will be wrong."""
    initial_dtype = query.dtype
    v_dim = expected_v_dim if expected_v_dim is not None else value.shape[-1]
    n_heads = expected_num_heads if expected_num_heads is not None else query.shape[2]
    bsz = expected_batch_size if expected_batch_size is not None else query.shape[0]
    out = torch.zeros(bsz, 1, n_heads, v_dim, dtype=initial_dtype, device=query.device)
    return out, recurrent_state


VARIANTS = {
    "original": _recurrent_original,
    "matmul4d": _recurrent_matmul4d,
    "einsum": _recurrent_einsum,
    "no_norm": _recurrent_no_norm,
    "passthrough": _recurrent_passthrough,
}


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


def patch_recurrence(variant_name):
    from anemll.models.qwen3_5_model import Qwen35LinearAttention
    orig = Qwen35LinearAttention._recurrent_gated_delta_rule
    fn = VARIANTS[variant_name]
    Qwen35LinearAttention._recurrent_gated_delta_rule = staticmethod(fn)
    return orig


def restore_recurrence(orig):
    from anemll.models.qwen3_5_model import Qwen35LinearAttention
    Qwen35LinearAttention._recurrent_gated_delta_rule = orig


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
        return {"wall_ms": w, "cpu_ms": c, "cpu_pct": cp, "ane_pct": ane}
    except Exception as e:
        print(f"  {label}: FAILED — {e}")
        return None


def run_experiment(args):
    print("=" * 70)
    print("  L-Layer ANE Optimization v2 — Multi-variant Comparison")
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

    for vname in variants:
        if vname not in VARIANTS:
            print(f"  Unknown variant: {vname}, skipping")
            continue
        print(f"\n── Variant: {vname} ──")

        if vname == "original":
            # No patching needed — use original code
            path = export_chunk(model, ci, vname, skip_existing=args.skip_existing)
        else:
            orig = patch_recurrence(vname)
            path = export_chunk(model, ci, vname, skip_existing=args.skip_existing)
            restore_recurrence(orig)

        pred = make_decode_inputs(ci)
        r = measure_ane(path, pred, vname)
        results[vname] = r
        gc.collect()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  RESULTS — Chunk {ci} ({pattern})")
    print(f"{'='*70}")
    print(f"{'Variant':>14} {'ANE%':>7} {'CPU%':>7} {'Wall(ms)':>10} {'CPU(ms)':>10}")
    for vname in variants:
        r = results.get(vname)
        if r:
            print(f"  {vname:>12}  {r['ane_pct']:>5.1f}%  {r['cpu_pct']:>5.1f}%  {r['wall_ms']:>8.2f}  {r['cpu_ms']:>8.2f}")
        else:
            print(f"  {vname:>12}  FAILED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0,
                        help="Chunk index to test (default: 0)")
    parser.add_argument("--variants", type=str,
                        default="original,matmul4d,einsum,passthrough",
                        help="Comma-separated variants to test")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
