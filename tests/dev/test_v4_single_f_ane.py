#!/usr/bin/env python3
"""
Test: Does keeping KV cache ops in FP32 disable ANE on a single F layer?

Uses the proven single-F-layer wrapper (from bisection phase2) at CTX=512.
Compares:
  A) Full FP16 (baseline, known ~92% ANE)
  B) V4: FP16 + kv_cache ops FP32 (~6 ops)
  C) FP16 + only slice_update cache ops FP32 (2 ops)
  D) Full FP32 (known 0% ANE, compiler fails)
"""
import gc
import os
import resource
import sys
import time
import warnings

import numpy as np
import torch

torch.set_grad_enabled(False)
warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_qwen3_5"))
os.chdir(REPO)

import coremltools as ct
from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

HIDDEN, NUM_KV, HDIM = 2560, 4, 256
CTX = 512
HF = os.path.join(REPO, "models", "Qwen__Qwen3.5-4B")
OUT = os.path.join(REPO, "artifacts", "v4_kvcache_ane_test")
os.makedirs(OUT, exist_ok=True)


class SingleFLayer(torch.nn.Module):
    def __init__(self, model, layer_idx, ctx):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.register_buffer("k_cache", torch.zeros(1, NUM_KV, ctx, HDIM, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(1, NUM_KV, ctx, HDIM, dtype=torch.float16))

    def forward(self, hs, pi, cm, cp):
        la = self.model.model.layers[self.layer_idx]
        x = la.input_layernorm(hs)
        q, k, v, g = la.self_attn.get_new_kv_cache(x, pi)
        p = cp[0]
        self.k_cache[:, :, p : p + 1, :] = k.squeeze(0)
        self.v_cache[:, :, p : p + 1, :] = v.squeeze(0)
        ao = la.self_attn.forward_regular(
            hidden_states=x,
            query_states=q,
            kv_cache_layer=(self.k_cache.squeeze(0), self.v_cache.squeeze(0)),
            causal_mask=cm,
            gate=g,
        )
        hs2 = hs + ao
        return hs2 + la.mlp(la.post_attention_layernorm(hs2))


def load_and_trace():
    print("Loading model...")
    cfg = Qwen35Config.from_json(os.path.join(HF, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    wrapper = SingleFLayer(model, 3, CTX).eval()
    h = torch.randn(1, 1, HIDDEN)
    pi = torch.tensor([CTX // 2], dtype=torch.long)
    m = torch.zeros(1, 1, 1, CTX)
    cp = torch.tensor([CTX // 2], dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (h, pi, m, cp))
    del model, wrapper
    gc.collect()
    return traced, h, pi, m, cp


def get_convert_args(h, pi, m, cp):
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_KV, CTX, HDIM), dtype=np.float16), name="k_cache"),
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_KV, CTX, HDIM), dtype=np.float16), name="v_cache"),
    ]
    inputs = [
        ct.TensorType(name="hidden_states", shape=h.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=pi.shape, dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=m.shape, dtype=np.float16),
        ct.TensorType(name="current_pos", shape=cp.shape, dtype=np.int32),
    ]
    return inputs, states


def export_model(traced, inputs, states, name, prec):
    path = os.path.join(OUT, f"single_F_{name}_ctx{CTX}.mlpackage")
    if os.path.exists(path):
        print(f"  [{name}] exists, skip export")
        return path
    print(f"  [{name}] exporting...")
    t0 = time.time()
    ml = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=states,
        compute_precision=prec,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    ml.save(path)
    del ml
    gc.collect()
    print(f"    saved ({time.time() - t0:.1f}s)")
    return path


def make_v4_selector():
    """FP16 for everything except kv_cache ops → FP32."""
    stats = {"fp16": 0, "fp32": 0}
    fp32_ops = []

    def sel(op):
        name_lower = op.name.lower()
        is_cache = False

        # slice_update with "cache" in name
        if "cache" in name_lower:
            is_cache = True
        # identity ops (state read pass-throughs)
        elif op.op_type == "identity":
            is_cache = True
        # squeeze feeding a cache write
        elif op.op_type == "squeeze":
            try:
                for out_var in op.outputs:
                    for child in out_var.child_ops:
                        if "cache" in child.name.lower():
                            is_cache = True
            except (AttributeError, TypeError):
                pass
        # slice_by_index feeding an identity (cache read)
        elif op.op_type == "slice_by_index":
            try:
                for out_var in op.outputs:
                    for child in out_var.child_ops:
                        if child.op_type == "identity":
                            is_cache = True
            except (AttributeError, TypeError):
                pass

        if is_cache:
            stats["fp32"] += 1
            fp32_ops.append(f"{op.op_type}:{op.name}")
            return False  # keep FP32
        stats["fp16"] += 1
        return True  # cast to FP16

    sel.stats = stats
    sel.fp32_ops = fp32_ops
    return sel


def make_slice_update_only_selector():
    """FP16 for everything except slice_update cache writes (2 ops)."""
    stats = {"fp16": 0, "fp32": 0}

    def sel(op):
        if op.op_type == "slice_update" and "cache" in op.name.lower():
            stats["fp32"] += 1
            return False
        stats["fp16"] += 1
        return True

    sel.stats = stats
    return sel


def measure(path, name, warmup=10, runs=30):
    try:
        ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = ml.make_state()
    except Exception as e:
        print(f"  [{name:35s}] FAILED: {e}")
        return None

    pred = {
        "hidden_states": np.random.randn(1, 1, HIDDEN).astype(np.float16),
        "position_ids": np.array([CTX // 2], dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, CTX), dtype=np.float16),
        "current_pos": np.array([CTX // 2], dtype=np.int32),
    }
    for _ in range(warmup):
        ml.predict(pred, state=state)

    times, cpus = [], []
    for _ in range(runs):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        ml.predict(pred, state=state)
        t1 = time.perf_counter()
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        times.append(t1 - t0)
        cpus.append((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))

    w = np.median(times) * 1000
    c = np.median(cpus) * 1000
    cp = c / w * 100 if w > 0 else 0
    ane = max(0, 100 - cp)
    print(f"  [{name:35s}] wall={w:.2f}ms cpu={c:.2f}ms CPU%={cp:.1f}% ANE%={ane:.1f}%")
    del ml, state
    gc.collect()
    return {"wall": w, "cpu": c, "cpu_pct": cp, "ane": ane}


def main():
    print("=" * 70)
    print("  SINGLE F-LAYER: KV CACHE FP32 ANE IMPACT TEST")
    print(f"  Layer 3, CTX={CTX}")
    print("=" * 70)

    traced, h, pi, m, cp = load_and_trace()
    inputs, states = get_convert_args(h, pi, m, cp)

    # --- Export variants ---
    print("\n[EXPORT]")

    # A: Full FP16
    p_fp16 = export_model(traced, inputs, states, "full_fp16", ct.precision.FLOAT16)

    # B: V4 = FP16 + kv_cache FP32
    sel_v4 = make_v4_selector()
    p_v4 = export_model(traced, inputs, states, "v4_kvcache_fp32", FP16ComputePrecision(op_selector=sel_v4))
    print(f"    V4 stats: {sel_v4.stats['fp16']} fp16, {sel_v4.stats['fp32']} fp32")
    for op_info in sel_v4.fp32_ops:
        print(f"      FP32: {op_info}")

    # C: Only slice_update FP32 (2 ops)
    sel_su = make_slice_update_only_selector()
    p_su = export_model(traced, inputs, states, "only_slice_update_fp32", FP16ComputePrecision(op_selector=sel_su))
    print(f"    slice_update stats: {sel_su.stats['fp16']} fp16, {sel_su.stats['fp32']} fp32")

    # D: Full FP32
    p_fp32 = export_model(traced, inputs, states, "full_fp32", ct.precision.FLOAT32)

    del traced
    gc.collect()

    # --- Measure ---
    print(f"\n[MEASURE ANE] CTX={CTX}")
    results = {}
    for name, path in [
        ("A_full_fp16", p_fp16),
        ("B_v4_kvcache_fp32 (6 ops)", p_v4),
        ("C_only_slice_update_fp32 (2 ops)", p_su),
        ("D_full_fp32", p_fp32),
    ]:
        results[name] = measure(path, name)

    # --- Summary ---
    print()
    print("=" * 70)
    print(f"  SUMMARY — Single F Layer (layer 3), CTX={CTX}")
    print("=" * 70)
    fmt = "  {:<40s} {:>7s} {:>9s} {:>9s}"
    print(fmt.format("Variant", "ANE%", "Wall(ms)", "CPU(ms)"))
    print("  " + "-" * 67)
    for name in results:
        r = results[name]
        if r:
            print(fmt.format(name, f"{r['ane']:.1f}%", f"{r['wall']:.2f}", f"{r['cpu']:.2f}"))
        else:
            print(fmt.format(name, "FAILED", "-", "-"))
    print("=" * 70)

    b = results.get("B_v4_kvcache_fp32 (6 ops)")
    a = results.get("A_full_fp16")
    if b and a:
        delta = b["ane"] - a["ane"]
        print(f"\n  V4 vs FP16:  ANE {b['ane']:.1f}% vs {a['ane']:.1f}% (delta {delta:+.1f}%)")
        print(f"  V4 vs FP16:  wall {b['wall']:.2f} vs {a['wall']:.2f}ms")
        if b["ane"] > 50:
            print(f"\n  >>> KV cache FP32 ops do NOT disable ANE loading <<<")
        else:
            print(f"\n  >>> KV cache FP32 ops DO disable ANE (or severely degrade it) <<<")


if __name__ == "__main__":
    main()
