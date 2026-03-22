#!/usr/bin/env python3
"""Benchmark: fused-argmax lm_head vs logits lm_head + Python argmax.

1. Re-exports lm_head with logits output (argmax_in_model=False)
2. Loads both models
3. Compares output correctness (parity)
4. Benchmarks decode latency
5. Tests logit-space penalties

Usage:
    python scripts_qwen3_5/benchmark_logits_lmhead.py
    python scripts_qwen3_5/benchmark_logits_lmhead.py --skip-export
"""
import gc, time, shutil, argparse, os, sys, json, warnings
import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct
import coremltools.optimize as cto
from config import (
    CTX, BATCH_SIZE, NUM_CHUNKS, LM_HEAD_LUT,
    PER_CHANNEL, DEFAULT_HF_MODEL, DEFAULT_OUTPUT,
)

MODEL_DIR = DEFAULT_OUTPUT
LOGITS_LMHEAD_PATH = os.path.join(MODEL_DIR, "lm_head_logits.mlpackage")
ARGMAX_LMHEAD_PATH = os.path.join(MODEL_DIR, "lm_head.mlpackage")
MODEL_DTYPE = torch.float16


# ── Lightweight export helpers (avoid full model OOM) ────────────────

def _load_lm_head_weight(model_path):
    """Load only lm_head.weight from safetensors (memory-efficient)."""
    import safetensors.torch
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    hidden_size = text_cfg["hidden_size"]
    vocab_size = text_cfg["vocab_size"]

    for fname in sorted(os.listdir(model_path)):
        if not fname.endswith(".safetensors"):
            continue
        with safetensors.torch.safe_open(
                os.path.join(model_path, fname), framework="pt", device="cpu") as f:
            if "lm_head.weight" in f.keys():
                w = f.get_tensor("lm_head.weight")
                return w.view(vocab_size, hidden_size, 1, 1).to(MODEL_DTYPE), hidden_size, vocab_size
    # Fallback: embed_tokens (tied weights)
    for fname in sorted(os.listdir(model_path)):
        if not fname.endswith(".safetensors"):
            continue
        with safetensors.torch.safe_open(
                os.path.join(model_path, fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                if "embed_tokens.weight" in k:
                    w = f.get_tensor(k)
                    return w.view(vocab_size, hidden_size, 1, 1).to(MODEL_DTYPE), hidden_size, vocab_size
    raise RuntimeError("Could not find lm_head.weight or embed_tokens.weight")


class _LMHeadWrapper(nn.Module):
    """Standalone LM head wrapper — logits only (no argmax)."""
    def __init__(self, weight_tensor):
        super().__init__()
        vocab_size, hidden_size = weight_tensor.shape[0], weight_tensor.shape[1]
        self.lm_head = nn.Conv2d.__new__(nn.Conv2d)
        nn.Module.__init__(self.lm_head)
        self.lm_head.in_channels = hidden_size
        self.lm_head.out_channels = vocab_size
        self.lm_head.kernel_size = (1, 1)
        self.lm_head.stride = (1, 1)
        self.lm_head.padding = (0, 0)
        self.lm_head.dilation = (1, 1)
        self.lm_head.groups = 1
        self.lm_head.padding_mode = 'zeros'
        self.lm_head.transposed = False
        self.lm_head.output_padding = (0, 0)
        self.lm_head.weight = nn.Parameter(weight_tensor)
        self.lm_head.bias = None

    def forward(self, hidden_states):
        logits = self.lm_head(hidden_states.permute(0, 2, 1).unsqueeze(2))
        return logits.squeeze(2).permute(0, 2, 1)


def _palettize(mlmodel, lut_bits, per_channel):
    """Apply LUT quantization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig
        cfg = OpPalettizerConfig(
            mode="kmeans", nbits=lut_bits,
            granularity="per_grouped_channel",
            group_size=per_channel, num_kmeans_workers=1)
        return cto.coreml.palettize_weights(
            mlmodel, OptimizationConfig(global_config=cfg))


# ── Step 1: Export logits lm_head ────────────────────────────────────

def export_logits_lmhead(hf_model_path):
    """Export lm_head with logits output (no fused argmax).
    Uses lightweight approach: loads only lm_head weight + EMPTY pipeline.
    """
    if os.path.exists(LOGITS_LMHEAD_PATH):
        print(f"[export] {LOGITS_LMHEAD_PATH} already exists, skipping")
        return

    print(f"[export] Exporting logits lm_head from {hf_model_path}...")
    weight, hidden_size, vocab_size = _load_lm_head_weight(hf_model_path)
    print(f"  Weight: {weight.shape}, hidden={hidden_size}, vocab={vocab_size}")

    wrapper = _LMHeadWrapper(weight).eval()
    del weight; gc.collect()

    sample_input = torch.zeros((1, 1, hidden_size), dtype=MODEL_DTYPE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_input)

    # Use EMPTY pipeline to avoid OOM with 248K vocab
    print("  Converting to CoreML (fp16, EMPTY pipeline)...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states",
                              shape=sample_input.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="output_logits", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        pass_pipeline=ct.PassPipeline.EMPTY,
    )
    del traced, wrapper; gc.collect()
    print(f"  Converted in {time.time()-t0:.1f}s")

    # Save fp16 intermediate, then quantize
    fp16_tmp = os.path.join(MODEL_DIR, "_lm_head_logits_fp16_tmp.mlpackage")
    if os.path.exists(fp16_tmp):
        shutil.rmtree(fp16_tmp)
    mlmodel.save(fp16_tmp)
    del mlmodel; gc.collect()

    print(f"  Applying LUT{LM_HEAD_LUT} quantization (per_channel={PER_CHANNEL})...")
    t0 = time.time()
    mlmodel = ct.models.MLModel(fp16_tmp)
    mlmodel = _palettize(mlmodel, LM_HEAD_LUT, PER_CHANNEL)
    print(f"  Quantized in {time.time()-t0:.1f}s")

    if os.path.exists(LOGITS_LMHEAD_PATH):
        shutil.rmtree(LOGITS_LMHEAD_PATH)
    mlmodel.save(LOGITS_LMHEAD_PATH)
    del mlmodel; gc.collect()

    # Cleanup tmp
    if os.path.exists(fp16_tmp):
        shutil.rmtree(fp16_tmp)

    print(f"  Saved → {LOGITS_LMHEAD_PATH}")


# ── Step 2: Load and verify ─────────────────────────────────────────

def load_models():
    """Load both lm_head variants."""
    cu = ct.ComputeUnit.CPU_AND_NE

    print("[load] Loading argmax lm_head...")
    t0 = time.time()
    m_argmax = ct.models.MLModel(ARGMAX_LMHEAD_PATH, compute_units=cu)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("[load] Loading logits lm_head...")
    t0 = time.time()
    m_logits = ct.models.MLModel(LOGITS_LMHEAD_PATH, compute_units=cu)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Show output signatures
    for name, m in [("argmax", m_argmax), ("logits", m_logits)]:
        spec = m.get_spec()
        outs = [(o.name, list(o.type.multiArrayType.shape)) for o in spec.description.output]
        print(f"  {name} outputs: {outs}")

    return m_argmax, m_logits


# ── Step 3: Correctness parity ──────────────────────────────────────

def test_parity(m_argmax, m_logits, n_trials=20):
    """Verify logits lm_head + np.argmax matches fused argmax."""
    print(f"\n{'='*60}")
    print(f"  PARITY TEST ({n_trials} random inputs)")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    matches = 0

    for i in range(n_trials):
        hidden = rng.standard_normal((1, 1, 2560)).astype(np.float16)

        # Fused argmax
        out_a = m_argmax.predict({"hidden_states": hidden})
        idx_a = int(out_a["argmax_idx"].flatten()[0])

        # Logits + Python argmax
        out_l = m_logits.predict({"hidden_states": hidden})
        logits = out_l["output_logits"].flatten()
        idx_l = int(np.argmax(logits))

        match = idx_a == idx_l
        if match:
            matches += 1
        else:
            print(f"  Trial {i}: MISMATCH argmax={idx_a} logits_argmax={idx_l} "
                  f"(logit_a={logits[idx_a]:.4f} logit_l={logits[idx_l]:.4f} "
                  f"delta={abs(logits[idx_a]-logits[idx_l]):.4f})")

    pct = matches * 100 / n_trials
    print(f"\n  Parity: {matches}/{n_trials} ({pct:.1f}%)")
    if matches == n_trials:
        print(f"  PASS: 100% match")
    elif pct > 95:
        print(f"  WARN: {100-pct:.1f}% mismatch (fp16 tie-breaking)")
    else:
        print(f"  FAIL: significant divergence")
    return pct


# ── Step 4: Latency benchmark ───────────────────────────────────────

def benchmark_latency(m_argmax, m_logits, n_warmup=5, n_trials=50):
    """Measure per-token latency for both lm_head variants."""
    print(f"\n{'='*60}")
    print(f"  LATENCY BENCHMARK ({n_trials} calls after {n_warmup} warmup)")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    hidden = rng.standard_normal((1, 1, 2560)).astype(np.float16)

    results = {}
    for name, model, postproc in [
        ("argmax_fused", m_argmax, lambda o: int(o["argmax_idx"].flatten()[0])),
        ("logits+argmax", m_logits, lambda o: int(np.argmax(o["output_logits"].flatten()))),
    ]:
        # Warmup
        for _ in range(n_warmup):
            out = model.predict({"hidden_states": hidden})
            _ = postproc(out)

        # Timed trials
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            out = model.predict({"hidden_states": hidden})
            _ = postproc(out)
            times.append(time.perf_counter() - t0)

        times_ms = [t * 1000 for t in times]
        median = sorted(times_ms)[n_trials // 2]
        mean = sum(times_ms) / len(times_ms)
        p95 = sorted(times_ms)[int(n_trials * 0.95)]
        results[name] = {"median": median, "mean": mean, "p95": p95}
        print(f"  {name:20s}: median={median:.1f}ms  mean={mean:.1f}ms  p95={p95:.1f}ms")

    overhead = results["logits+argmax"]["median"] - results["argmax_fused"]["median"]
    pct_overhead = overhead / results["argmax_fused"]["median"] * 100
    print(f"\n  Overhead: {overhead:+.1f}ms ({pct_overhead:+.1f}%)")

    return results


# ── Step 5: Memory / size comparison ────────────────────────────────

def compare_sizes():
    """Compare model file sizes."""
    print(f"\n{'='*60}")
    print(f"  MODEL SIZE COMPARISON")
    print(f"{'='*60}")

    import subprocess
    for name, path in [("argmax", ARGMAX_LMHEAD_PATH), ("logits", LOGITS_LMHEAD_PATH)]:
        result = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
        size = result.stdout.split()[0] if result.stdout else "N/A"
        print(f"  {name:20s}: {size}")


# ── Step 6: Logit-space penalty demo ────────────────────────────────

def demo_penalties(m_logits):
    """Demonstrate logit-space penalties on a realistic input."""
    print(f"\n{'='*60}")
    print(f"  LOGIT-SPACE PENALTY DEMO")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    hidden = rng.standard_normal((1, 1, 2560)).astype(np.float16)

    out = m_logits.predict({"hidden_states": hidden})
    logits_raw = out["output_logits"].flatten().astype(np.float32)

    # Baseline
    base_id = int(np.argmax(logits_raw))
    print(f"  Greedy (no penalty): token={base_id} logit={logits_raw[base_id]:.4f}")

    # Simulate prior generation with repetition
    fake_history = [base_id] * 5 + list(range(100, 110))

    # Repetition penalty (standard HF approach)
    for rep_pen in [1.0, 1.05, 1.1, 1.2, 1.5]:
        logits = logits_raw.copy()
        for tok in set(fake_history):
            if logits[tok] > 0:
                logits[tok] /= rep_pen
            else:
                logits[tok] *= rep_pen
        new_id = int(np.argmax(logits))
        changed = "YES" if new_id != base_id else "no"
        print(f"  rep_penalty={rep_pen:.2f}: token={new_id} "
              f"logit={logits[new_id]:.4f} changed={changed}")

    # Presence penalty
    print()
    for pres_pen in [0.0, 0.5, 1.0, 2.0]:
        logits = logits_raw.copy()
        for tok in set(fake_history):
            logits[tok] -= pres_pen
        new_id = int(np.argmax(logits))
        changed = "YES" if new_id != base_id else "no"
        print(f"  presence_pen={pres_pen:.1f}: token={new_id} "
              f"logit={logits[new_id]:.4f} changed={changed}")

    # Frequency penalty
    print()
    from collections import Counter
    freq = Counter(fake_history)
    for freq_pen in [0.0, 0.5, 1.0, 2.0]:
        logits = logits_raw.copy()
        for tok, count in freq.items():
            logits[tok] -= freq_pen * count
        new_id = int(np.argmax(logits))
        changed = "YES" if new_id != base_id else "no"
        print(f"  frequency_pen={freq_pen:.1f}: token={new_id} "
              f"logit={logits[new_id]:.4f} changed={changed}")


# ── Step 7: Logits output size measurement ──────────────────────────

def measure_output_size(m_logits):
    """Measure the output data transfer size."""
    print(f"\n{'='*60}")
    print(f"  OUTPUT TRANSFER SIZE")
    print(f"{'='*60}")

    hidden = np.zeros((1, 1, 2560), dtype=np.float16)
    out = m_logits.predict({"hidden_states": hidden})
    logits = out["output_logits"]
    nbytes = logits.nbytes
    shape = logits.shape
    print(f"  Logits shape: {shape}")
    print(f"  Logits dtype: {logits.dtype}")
    print(f"  Transfer size: {nbytes:,} bytes ({nbytes/1024:.1f} KB)")
    print(f"  Argmax transfer: ~8 bytes (int32 + fp16)")
    print(f"  Ratio: {nbytes/8:.0f}x more data from ANE")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: argmax vs logits lm_head")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip re-export (use existing logits model)")
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL,
                        help="HuggingFace model path")
    args = parser.parse_args()

    print("=" * 60)
    print("  LM Head Benchmark: Fused Argmax vs Logits + Python")
    print("=" * 60)

    # Step 1: Export
    if not args.skip_export:
        export_logits_lmhead(args.hf_model)
    elif not os.path.exists(LOGITS_LMHEAD_PATH):
        print(f"ERROR: {LOGITS_LMHEAD_PATH} not found. Run without --skip-export.")
        return 1

    # Step 2: Load
    m_argmax, m_logits = load_models()

    # Step 3: Parity
    parity = test_parity(m_argmax, m_logits)

    # Step 4: Latency
    latency = benchmark_latency(m_argmax, m_logits)

    # Step 5: Size
    compare_sizes()

    # Step 6: Output transfer
    measure_output_size(m_logits)

    # Step 7: Penalty demo
    demo_penalties(m_logits)

    # ── Summary ──
    overhead_ms = latency["logits+argmax"]["median"] - latency["argmax_fused"]["median"]
    overhead_pct = overhead_ms / latency["argmax_fused"]["median"] * 100

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Q1. Logits export:    YES (convert_part_3 supports both modes)")
    print(f"  Q2. Parity:           {parity:.1f}%")
    print(f"  Q3. Overhead:         {overhead_ms:+.1f}ms ({overhead_pct:+.1f}%)")
    print(f"  Q4. Penalties work:   Demonstrated above")
    print(f"  Q5. vs n-gram guard:  Logit penalties prevent repetition proactively;")
    print(f"       n-gram guard only detects/stops after repetition occurs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
