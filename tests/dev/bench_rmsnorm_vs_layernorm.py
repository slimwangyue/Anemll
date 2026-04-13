#!/usr/bin/env python3
"""
Benchmark: RMSNorm (reduce_mean + rsqrt) vs LayerNorm (doubled-concat F.layer_norm)
on ANE for Qwen3.5-4B decode (seq_len=1) and prefill (seq_len=512).

Exports 3-layer chunk0 with each variant, measures wall / CPU / ANE timing.
"""
import sys, os, time, gc, resource, warnings, argparse
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Volumes/MySSD/Anemll")
sys.path.insert(0, "/Volumes/MySSD/Anemll/scripts_qwen3_5")
os.chdir("/Volumes/MySSD/Anemll")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

torch.set_grad_enabled(False)

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

HIDDEN = 2560
MODEL_DTYPE = torch.float16
HF_MODEL = "models/Qwen__Qwen3.5-4B"
OUT_DIR = "artifacts/bench_rmsnorm_vs_layernorm"

# ── Original implementations (saved before monkey-patching) ──────────
import anemll.models.qwen3_5_model as qm
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM,
    Qwen35Config,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

orig_rmsnorm_fwd = Qwen35RMSNorm.forward
orig_rmsnormgated_fwd = Qwen35RMSNormGated.forward


# ── LayerNorm (doubled-concat) variant ───────────────────────────────
def layernorm_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Old doubled-concat F.layer_norm approach (generates MIL layer_norm ops)."""
    x = hidden_states
    doubled = torch.cat([x, -x], dim=-1)
    normed = F.layer_norm(
        doubled,
        normalized_shape=(2 * self.hidden_size,),
        weight=None,
        bias=None,
        eps=float(self.eps),
    )
    normed = normed[..., : self.hidden_size]
    scale = 1.0 + self.weight.to(normed.dtype, copy=False).to(
        normed.device, copy=False
    )
    return normed * scale


def layernorm_rmsnormgated_forward(
    self, hidden_states: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Old doubled-concat for gated variant."""
    x = hidden_states
    doubled = torch.cat([x, -x], dim=-1)
    normed = F.layer_norm(
        doubled,
        normalized_shape=(2 * self.hidden_size,),
        weight=None,
        bias=None,
        eps=float(self.eps),
    )
    normed = normed[..., : self.hidden_size]
    out = normed * self.weight.to(hidden_states.dtype)
    out = out * F.silu(gate.to(hidden_states.dtype))
    return out


# ── Export helper ─────────────────────────────────────────────────────
def export_variant(
    name, model, start_layer, end_layer, setup_fn, teardown_fn, skip_existing=False
):
    """Export decode + prefill for a variant."""
    paths = {}
    for mode in ["decode", "prefill"]:
        path = os.path.join(OUT_DIR, f"{name}_{mode}.mlpackage")
        paths[mode] = path
        if skip_existing and os.path.exists(path):
            print(f"  {name} {mode}: cached")
            continue
        setup_fn()
        conv = Qwen35Converter(
            model,
            context_length=CTX,
            batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS,
            lut_bits=4,
            per_channel=4,
            compute_precision="float16",
        )
        t0 = time.time()
        if mode == "decode":
            ml = conv.convert_part_2(
                model,
                chunk_idx=0,
                total_chunks=NUM_CHUNKS,
                override_start_layer=start_layer,
                override_end_layer=end_layer,
            )
        else:
            ml = conv.convert_part_2_prefill(
                model,
                chunk_idx=0,
                total_chunks=NUM_CHUNKS,
                override_start_layer=start_layer,
                override_end_layer=end_layer,
            )
        ml.save(path)
        elapsed = time.time() - t0
        print(f"  {name} {mode}: exported in {elapsed:.1f}s")
        teardown_fn()
        del ml, conv
        gc.collect()
    return paths


# ── MIL op analysis ──────────────────────────────────────────────────
def count_mil_ops(path):
    spec = ct.utils.load_spec(path)
    prog = spec.mlProgram
    for fn in prog.functions:
        func = prog.functions[fn]
        for k in func.block_specializations:
            block = func.block_specializations[k]
            break
        break
    counts = Counter()
    for op in block.operations:
        counts[op.type] += 1
    total = sum(counts.values())
    weight = sum(
        v
        for k, v in counts.items()
        if k in ("const", "constexpr_lut_to_dense")
    )
    return counts, total - weight


# ── Timing helper ────────────────────────────────────────────────────
def measure_timing(path, inputs, n_warmup=5, n_runs=30, compute_unit=None):
    """Measure wall/cpu/ane timing. Returns (wall, cpu, ane, ane_pct, cu_label)."""
    cu_options = (
        [(compute_unit, "custom")]
        if compute_unit
        else [
            (ct.ComputeUnit.CPU_AND_NE, "CPU_AND_NE"),
            (ct.ComputeUnit.ALL, "ALL"),
            (ct.ComputeUnit.CPU_AND_GPU, "CPU_AND_GPU"),
            (ct.ComputeUnit.CPU_ONLY, "CPU_ONLY"),
        ]
    )

    ml = None
    state = None
    cu_label = None
    for cu, label in cu_options:
        try:
            ml = ct.models.MLModel(path, compute_units=cu)
        except Exception as e:
            print(f"[{label} load failed: {str(e)[:50]}] ", end="", flush=True)
            gc.collect()
            continue
        try:
            state = ml.make_state()
        except Exception as e:
            print(f"[{label} make_state failed: {str(e)[:50]}] ", end="", flush=True)
            del ml; ml = None
            gc.collect()
            continue
        try:
            ml.predict(inputs, state=state)
            cu_label = label
            break
        except Exception as e:
            print(f"[{label} predict failed: {str(e)[:50]}] ", end="", flush=True)
            del ml, state; ml = None; state = None
            gc.collect()
            continue

    if ml is None or state is None:
        return None

    # warmup
    for _ in range(n_warmup):
        ml.predict(inputs, state=state)

    times = []
    for _ in range(n_runs):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        ml.predict(inputs, state=state)
        wall = (time.perf_counter() - t0) * 1000
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (
            (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        ) * 1000
        times.append((wall, cpu))

    w = np.median([t[0] for t in times])
    c = np.median([t[1] for t in times])
    a = max(0, w - c)
    pct = a / w * 100 if w > 0 else 0

    del ml
    gc.collect()
    return w, c, a, pct, cu_label


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RMSNorm vs LayerNorm benchmark")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers to export (default 3)")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    start_layer = 0
    end_layer = args.layers

    nl = end_layer - start_layer

    # ── Export ──
    if not args.skip_export:
        print(f"Loading model weights...")
        cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
        cfg.context_length = CTX
        cfg.state_length = CTX
        model = Qwen35ForCausalLM(cfg)
        assert model.load_pretrained_weights(HF_MODEL)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Variant A: current RMSNorm (reduce_mean + rsqrt)
        print("\n[A] RMSNorm (reduce_mean + rsqrt):")
        paths_A = export_variant(
            "A_rmsnorm",
            model,
            start_layer,
            end_layer,
            setup_fn=lambda: None,
            teardown_fn=lambda: None,
            skip_existing=args.skip_existing,
        )

        # Variant B: old LayerNorm (doubled-concat F.layer_norm)
        print("\n[B] LayerNorm (doubled-concat F.layer_norm):")
        paths_B = export_variant(
            "B_layernorm",
            model,
            start_layer,
            end_layer,
            setup_fn=lambda: (
                setattr(Qwen35RMSNorm, "forward", layernorm_rmsnorm_forward),
                setattr(
                    Qwen35RMSNormGated, "forward", layernorm_rmsnormgated_forward
                ),
            ),
            teardown_fn=lambda: (
                setattr(Qwen35RMSNorm, "forward", orig_rmsnorm_fwd),
                setattr(Qwen35RMSNormGated, "forward", orig_rmsnormgated_fwd),
            ),
            skip_existing=args.skip_existing,
        )

        del model
        gc.collect()

    # ── MIL op comparison ──
    print("\n" + "=" * 70)
    print("  MIL OP COMPARISON")
    print("=" * 70)
    for name in ["A_rmsnorm", "B_layernorm"]:
        for mode in ["decode", "prefill"]:
            path = os.path.join(OUT_DIR, f"{name}_{mode}.mlpackage")
            if not os.path.exists(path):
                continue
            counts, compute = count_mil_ops(path)
            layer_norm = counts.get("layer_norm", 0)
            reduce_mean = counts.get("reduce_mean", 0)
            rsqrt = counts.get("rsqrt", 0)
            conv_op = counts.get("conv", 0)
            trans = counts.get("transpose", 0)
            print(
                f"  {name:15s} {mode:8s}: compute={compute:4d}  "
                f"layer_norm={layer_norm:2d}  reduce_mean={reduce_mean:2d}  "
                f"rsqrt={rsqrt:2d}  conv={conv_op:2d}  transpose={trans:2d}"
            )

    # ── Timing ──
    print("\n" + "=" * 70)
    print("  TIMING (CPU_AND_NE)")
    print("=" * 70)

    results = {}

    for name in ["A_rmsnorm", "B_layernorm"]:
        for mode in ["decode"]:  # prefill has different input schema (valid_len etc.)
            path = os.path.join(OUT_DIR, f"{name}_{mode}.mlpackage")
            if not os.path.exists(path):
                continue

            np.random.seed(42)
            inputs = {
                "hidden_states": np.random.randn(1, 1, HIDDEN).astype(
                    np.float16
                )
                * 0.01,
                "position_ids": np.array([0], dtype=np.int32),
                "causal_mask": np.zeros(
                    (1, 1, 1, CTX), dtype=np.float16
                ),
                "current_pos": np.array([0], dtype=np.int32),
                "linear_conv_state": np.zeros(
                    (nl, 1024, 32), dtype=np.float16
                ),
                "linear_recurrent_state": np.zeros(
                    (nl, 32, 128, 128), dtype=np.float16
                ),
            }

            label = f"{name} {mode}"
            print(f"  {label:30s}... ", end="", flush=True)
            result = measure_timing(
                path, inputs, n_warmup=args.n_warmup, n_runs=args.n_runs
            )
            if result is None:
                print("ALL COMPUTE UNITS FAILED")
                continue
            w, c, a, pct, cu_label = result
            results[label] = (w, c, a, pct, cu_label)
            print(f"wall={w:.2f}ms  cpu={c:.2f}ms  ane={a:.2f}ms  ({pct:.0f}%)  [{cu_label}]")

    # ── Accuracy comparison ──
    print("\n" + "=" * 70)
    print("  ACCURACY (cosine similarity)")
    print("=" * 70)
    for mode in ["decode"]:  # skip prefill — different input schema
        pa = os.path.join(OUT_DIR, f"A_rmsnorm_{mode}.mlpackage")
        pb = os.path.join(OUT_DIR, f"B_layernorm_{mode}.mlpackage")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue

        np.random.seed(42)
        inputs = {
            "hidden_states": np.random.randn(1, 1, HIDDEN).astype(np.float16) * 0.01,
            "position_ids": np.array([0], dtype=np.int32),
            "causal_mask": np.zeros((1, 1, 1, CTX), dtype=np.float16),
            "current_pos": np.array([0], dtype=np.int32),
            "linear_conv_state": np.zeros((nl, 1024, 32), dtype=np.float16),
            "linear_recurrent_state": np.zeros((nl, 32, 128, 128), dtype=np.float16),
        }

        try:
            ma = ct.models.MLModel(pa, compute_units=ct.ComputeUnit.CPU_AND_NE)
            mb = ct.models.MLModel(pb, compute_units=ct.ComputeUnit.CPU_AND_NE)
            sa, sb = ma.make_state(), mb.make_state()
            oa = ma.predict(inputs, state=sa)
            ob = mb.predict(inputs, state=sb)

            arr_a = np.asarray(oa["output_hidden_states"]).flatten().astype(np.float64)
            arr_b = np.asarray(ob["output_hidden_states"]).flatten().astype(np.float64)
            cos = float(
                np.dot(arr_a, arr_b)
                / (np.linalg.norm(arr_a) * np.linalg.norm(arr_b) + 1e-30)
            )
            print(f"  {mode:8s}: cos(rmsnorm, layernorm) = {cos:.8f}")
            del ma, mb
            gc.collect()
        except Exception as e:
            print(f"  {mode:8s}: FAILED — {str(e)[:80]}")
            gc.collect()

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Variant':30s} {'Wall':>7s} {'CPU':>7s} {'ANE':>7s} {'ANE%':>5s} {'CU':>12s}")
    print("  " + "-" * 68)
    for label, (w, c, a, pct, cu_label) in sorted(results.items()):
        print(f"  {label:30s} {w:7.2f} {c:7.2f} {a:7.2f} {pct:4.0f}%  {cu_label:>12s}")

    # Delta analysis
    for mode in ["decode", "prefill"]:
        ka = f"A_rmsnorm {mode}"
        kb = f"B_layernorm {mode}"
        if ka in results and kb in results:
            wa = results[ka][0]
            wb = results[kb][0]
            cu_a = results[ka][4]
            cu_b = results[kb][4]
            delta_pct = (wa - wb) / wb * 100
            faster = "RMSNorm" if wa < wb else "LayerNorm"
            print(
                f"\n  {mode}: {faster} is {abs(delta_pct):.1f}% faster  "
                f"(RMSNorm={wa:.2f}ms [{cu_a}], LayerNorm={wb:.2f}ms [{cu_b}], delta={wa-wb:+.2f}ms)"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
