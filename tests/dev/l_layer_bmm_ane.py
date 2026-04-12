#!/usr/bin/env python3
"""ANE-optimized L-layer recurrence: replace reduce_sum with bmm.

The original recurrence uses:
  kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)  → reduce_sum (ANE-hostile)
  outer  = k.unsqueeze(-1) * delta.unsqueeze(-2)  → broadcast multiply
  out    = (state * q.unsqueeze(-1)).sum(dim=-2)   → reduce_sum (ANE-hostile)

This rewrite uses:
  kv_mem = bmm(state^T, k)  → matmul (ANE-friendly)
  outer  = bmm(k, delta^T)  → matmul (ANE-friendly)
  out    = bmm(state^T, q)  → matmul (ANE-friendly)

Steps:
  1. Validate equivalent output in PyTorch
  2. Export chunk 0 (LLL) with patched recurrence
  3. Measure ANE% improvement
  4. Validate accuracy end-to-end
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
OUTPUT_DIR = os.path.join(REPO_ROOT, "artifacts", "l_layer_bmm_ane")
HIDDEN = 2560
WARMUP = 10
RUNS = 30


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    sq_sum = (x * x).sum(dim=dim, keepdim=True)
    sq_sum = torch.clamp(sq_sum, min=eps)
    return x * torch.rsqrt(sq_sum)


def _recurrent_gated_delta_rule_bmm(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    output_final_state: bool = True,
    expected_batch_size: Optional[int] = None,
    expected_num_heads: Optional[int] = None,
    expected_seq_len: Optional[int] = None,
    expected_k_dim: Optional[int] = None,
    expected_v_dim: Optional[int] = None,
    math_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ANE-optimized recurrence using bmm instead of reduce_sum.

    For seq_len=1 (decode), replaces:
      (state * x.unsqueeze(-1)).sum(dim=-2)  →  bmm(state^T, x)
      k.unsqueeze(-1) * delta.unsqueeze(-2)  →  bmm(k, delta^T)

    This eliminates reduce_sum ops (ANE-hostile) and replaces them with
    matmul ops (ANE-friendly).
    """
    initial_dtype = query.dtype
    # Input: (B, S, H, D) → transpose to (B, H, S, D)
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
    BH = bsz * n_heads

    for i in range(seq_len):
        q_t = query[:, :, i]     # (B, H, K)
        k_t = key[:, :, i]       # (B, H, K)
        v_t = value[:, :, i]     # (B, H, V)
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        beta_t = beta[:, :, i].unsqueeze(-1)  # (B, H, 1)

        # Decay
        state = state * g_t  # (B, H, K, V) * (B, H, 1, 1)

        # kv_mem via bmm: state^T @ k → (V, K) @ (K, 1) → (V, 1)
        state_3d = state.reshape(BH, k_dim, v_dim)  # (BH, K, V)
        k_3d = k_t.reshape(BH, k_dim, 1)            # (BH, K, 1)
        kv_mem = torch.bmm(
            state_3d.transpose(1, 2), k_3d  # (BH, V, K) @ (BH, K, 1) → (BH, V, 1)
        ).reshape(bsz, n_heads, v_dim)  # (B, H, V)

        # Delta
        delta = (v_t - kv_mem) * beta_t  # (B, H, V)

        # Outer product via bmm: k @ delta^T → (K, 1) @ (1, V) → (K, V)
        delta_3d = delta.reshape(BH, 1, v_dim)  # (BH, 1, V)
        outer = torch.bmm(k_3d, delta_3d)  # (BH, K, 1) @ (BH, 1, V) → (BH, K, V)
        state = state + outer.reshape(bsz, n_heads, k_dim, v_dim)

        # Output via bmm: state^T @ q → (V, K) @ (K, 1) → (V, 1)
        q_3d = q_t.reshape(BH, k_dim, 1)  # (BH, K, 1)
        state_3d_upd = state.reshape(BH, k_dim, v_dim)
        out[:, :, i] = torch.bmm(
            state_3d_upd.transpose(1, 2), q_3d  # (BH, V, K) @ (BH, K, 1) → (BH, V, 1)
        ).reshape(bsz, n_heads, v_dim)

    if not output_final_state:
        state = None
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state


def load_model():
    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


def patch_recurrence_bmm(enable=True):
    """Monkey-patch the recurrence function on the class to use bmm version."""
    from anemll.models.qwen3_5_model import Qwen35LinearAttention
    orig = Qwen35LinearAttention._recurrent_gated_delta_rule
    if enable:
        Qwen35LinearAttention._recurrent_gated_delta_rule = staticmethod(_recurrent_gated_delta_rule_bmm)
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
    print(f"  Exported {label} decode chunk{chunk_idx} in {time.time() - t0:.1f}s")
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
    for key in sorted(out_a.keys()):
        if key in out_b:
            a = np.asarray(out_a[key]).flatten().astype(np.float64)
            b = np.asarray(out_b[key]).flatten().astype(np.float64)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-10 and norm_b < 1e-10:
                print(f"  {label_a} vs {label_b} [{key}]: both near-zero")
                continue
            cos = float(np.dot(a, b) / (norm_a * norm_b + 1e-30))
            max_abs = float(np.max(np.abs(a - b)))
            print(f"  {label_a} vs {label_b} [{key}]: cos={cos:.8f}  max_abs={max_abs:.6f}")


def run_pytorch_comparison(model, cfg, chunk_idx):
    """Run the recurrence in PyTorch with original vs bmm to verify correctness."""
    from anemll.models.qwen3_5_model import Qwen35LinearAttention, ane_conv_state_shape
    start, end = CHUNK_RANGES[chunk_idx]
    layer_types = cfg.text_config.layer_types

    # Find first L layer
    l_idx = None
    for li in range(start, end):
        if layer_types[li] == "linear_attention":
            l_idx = li
            break
    if l_idx is None:
        print("  No L layer in this chunk!")
        return

    layer = model.model.layers[l_idx]
    attn = layer.self_attn

    # Create test input
    hidden = torch.randn(1, 1, HIDDEN, dtype=torch.float16)
    conv_dim = attn.conv_dim
    conv_kernel = attn.linear_conv_kernel_dim
    conv_state = torch.zeros(1, conv_dim, conv_kernel, dtype=torch.float16)
    rec_state = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=torch.float16)

    # Run with original recurrence
    print("  Running PyTorch original recurrence...")
    out_orig, _, rec_orig = attn._forward_impl(
        hidden, conv_state.clone(), rec_state.clone(),
        has_previous_state=True, expected_batch_size=1, expected_seq_len=1,
        force_recurrent=True,
    )

    # Patch and run with bmm recurrence
    orig_fn = patch_recurrence_bmm(enable=True)
    print("  Running PyTorch bmm recurrence...")
    out_bmm, _, rec_bmm = attn._forward_impl(
        hidden, conv_state.clone(), rec_state.clone(),
        has_previous_state=True, expected_batch_size=1, expected_seq_len=1,
        force_recurrent=True,
    )
    restore_recurrence(orig_fn)

    # Compare
    cos_out = F.cosine_similarity(out_orig.flatten().float(), out_bmm.flatten().float(), dim=0)
    cos_rec = F.cosine_similarity(rec_orig.flatten().float(), rec_bmm.flatten().float(), dim=0)
    max_out = (out_orig - out_bmm).abs().max().item()
    max_rec = (rec_orig - rec_bmm).abs().max().item()
    print(f"  PyTorch orig vs bmm — hidden: cos={cos_out:.8f}  max_abs={max_out:.6f}")
    print(f"  PyTorch orig vs bmm — state:  cos={cos_rec:.8f}  max_abs={max_rec:.6f}")
    return cos_out.item()


def run_experiment(args):
    print("=" * 70)
    print("  L-Layer bmm Recurrence → ANE Optimization")
    print(f"  CTX={CTX}, BATCH_SIZE={BATCH_SIZE}")
    print("=" * 70)

    print("\nLoading model...")
    model, cfg = load_model()

    chunks_to_test = args.chunks
    results = {}

    for ci in chunks_to_test:
        start, end = CHUNK_RANGES[ci]
        pattern = "".join("F" if l in F_LAYERS else "L" for l in range(start, end))
        print(f"\n{'='*70}")
        print(f"  Chunk {ci}: layers {start}-{end-1}, pattern={pattern}")
        print(f"{'='*70}")

        # --- Step 1: PyTorch-level comparison ---
        print("\n[Step 1] PyTorch correctness check...")
        cos_pt = run_pytorch_comparison(model, cfg, ci)
        if cos_pt is not None and cos_pt < 0.999:
            print(f"  WARNING: PyTorch cos={cos_pt:.6f} — possible accuracy issue!")

        # --- Step 2: Export original ---
        print(f"\n[Step 2] Export original (reduce_sum)...")
        path_orig = export_chunk(model, ci, "original", skip_existing=args.skip_existing)

        # --- Step 3: Export bmm version ---
        print(f"\n[Step 3] Export bmm rewrite...")
        orig_fn = patch_recurrence_bmm(enable=True)
        path_bmm = export_chunk(model, ci, "bmm", skip_existing=args.skip_existing)
        restore_recurrence(orig_fn)

        # --- Step 4: Measure ANE% ---
        print(f"\n[Step 4] Measure ANE%...")
        pred = make_decode_inputs(ci)
        ml_orig, st_orig, res_orig = measure_ane(path_orig, pred, f"chunk{ci}_original")
        ml_bmm, st_bmm, res_bmm = measure_ane(path_bmm, pred, f"chunk{ci}_bmm")

        # --- Step 5: Compare CoreML accuracy ---
        if ml_orig and ml_bmm:
            print(f"\n[Step 5] CoreML accuracy comparison...")
            det_pred = make_decode_inputs(ci)
            det_pred["hidden_states"] = np.ones((1, 1, HIDDEN), dtype=np.float16) * 0.01
            compare_outputs(ml_orig, st_orig, ml_bmm, st_bmm, det_pred, "original", "bmm")

        results[ci] = {"original": res_orig, "bmm": res_bmm, "pattern": pattern, "pytorch_cos": cos_pt}
        del ml_orig, ml_bmm, st_orig, st_bmm; gc.collect()

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Chunk':>6} {'Pattern':>8} {'Orig ANE%':>11} {'BMM ANE%':>10} {'Orig Wall':>11} {'BMM Wall':>10} {'Δ ANE':>8} {'PT cos':>10}")
    for ci, r in sorted(results.items()):
        o = r["original"]
        b = r["bmm"]
        cos = r["pytorch_cos"]
        if o and b:
            d_ane = b["ane_pct"] - o["ane_pct"]
            print(f"  {ci:>4}   {r['pattern']:>8}  {o['ane_pct']:>8.1f}%  {b['ane_pct']:>7.1f}%  {o['wall_ms']:>8.2f}ms  {b['wall_ms']:>7.2f}ms  {d_ane:>+6.1f}%  {cos:.6f}" if cos else
                  f"  {ci:>4}   {r['pattern']:>8}  {o['ane_pct']:>8.1f}%  {b['ane_pct']:>7.1f}%  {o['wall_ms']:>8.2f}ms  {b['wall_ms']:>7.2f}ms  {d_ane:>+6.1f}%  N/A")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default="0",
                        help="Comma-separated chunk indices (default: 0)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.chunks = [int(x) for x in args.chunks.split(",")]
    run_experiment(args)


if __name__ == "__main__":
    main()
