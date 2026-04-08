#!/usr/bin/env python3
"""Test padding-hidden-state zeroing mitigation on chunk0 prefill.

Exports chunk0 twice (baseline vs masked), running GPU-vs-ANE parity for each.
Deletes each model after measurement to conserve disk space.

Hypothesis: zeroing hidden_states at padding positions between layers
prevents Layer 3 full-attention divergence from propagating through
residual connections.
"""

import os, sys, time, gc, shutil
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import coremltools as ct

# ── config ──────────────────────────────────────────────────────────
BATCH_SIZE   = 512
CTX          = 2048
NUM_CHUNKS   = 6
LUT_BITS     = 6
FFN_PER_CH   = 4

HF_MODEL     = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
OUT_DIR      = "/tmp/diag_padding_mask"

CHUNK0_LAYERS = 6
CHUNK0_LIN    = [True, True, True, False, True, True]
NUM_V_HEADS   = 32
KEY_HEAD_DIM  = 128
VAL_HEAD_DIM  = 128


# ── helpers ─────────────────────────────────────────────────────────
def cmp(a, b):
    a_f = a.astype(np.float32).flatten()
    b_f = b.astype(np.float32).flatten()
    diff = a_f - b_f
    mad  = float(np.max(np.abs(diff)))
    mean = float(np.mean(np.abs(diff)))
    dot  = float(np.dot(a_f, b_f))
    na   = float(np.linalg.norm(a_f))
    nb   = float(np.linalg.norm(b_f))
    cos  = dot / (na * nb + 1e-12)
    return mad, mean, cos


def make_inputs(hidden_np, valid_len):
    """Build prediction inputs matching chunk0 prefill schema."""
    pos = np.zeros(BATCH_SIZE, dtype=np.int32)
    pos[:valid_len] = np.arange(valid_len, dtype=np.int32)

    mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
    for i in range(valid_len):
        mask[0, 0, i, :i+1] = 0.0

    return {
        "hidden_states":          hidden_np,
        "position_ids":           pos,
        "causal_mask":            mask,
        "current_pos":            np.array([0], dtype=np.int32),
        "linear_conv_state":      np.zeros((CHUNK0_LAYERS, 1024, 32), dtype=np.float16),
        "linear_recurrent_state": np.zeros((CHUNK0_LAYERS, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM),
                                           dtype=np.float16),
        "valid_len":              np.array([valid_len], dtype=np.int32),
    }


def run_model(model_path, inputs, compute_unit, label):
    print(f"  {label}...", end="", flush=True)
    t0 = time.time()
    mlmodel = ct.models.MLModel(model_path, compute_units=compute_unit)
    load_t = time.time() - t0
    state = mlmodel.make_state()
    t0 = time.time()
    out = mlmodel.predict(inputs, state=state)
    pred_t = time.time() - t0
    print(f"  load={load_t:.1f}s  predict={pred_t:.1f}s")
    del mlmodel
    gc.collect()
    return out


def measure_parity(model_path, inputs, valid_len, tag):
    """Run GPU vs ANE and report parity."""
    print(f"\n{'='*60}")
    print(f"[{tag}] Measuring GPU-vs-ANE parity")
    print(f"{'='*60}")

    gpu = run_model(model_path, inputs, ct.ComputeUnit.CPU_AND_GPU, "GPU")
    ane = run_model(model_path, inputs, ct.ComputeUnit.CPU_AND_NE,  "ANE")

    results = {}

    # ── Hidden states ──
    gh = gpu["output_hidden_states"]
    ah = ane["output_hidden_states"]
    mad, mean, cos = cmp(gh, ah)
    results["hidden_all"] = {"mad": mad, "mean": mean, "cos": cos}

    # Per-token analysis
    gh3 = gh.reshape(BATCH_SIZE, -1) if gh.ndim > 2 else gh.squeeze(0)
    ah3 = ah.reshape(BATCH_SIZE, -1) if ah.ndim > 2 else ah.squeeze(0)

    valid_mads = []
    pad_mads = []
    for t in range(BATCH_SIZE):
        tm, _, _ = cmp(gh3[t:t+1], ah3[t:t+1])
        if t < valid_len:
            valid_mads.append(tm)
        else:
            pad_mads.append(tm)

    results["valid_worst_mad"] = max(valid_mads) if valid_mads else 0.0
    results["valid_mean_mad"]  = float(np.mean(valid_mads)) if valid_mads else 0.0
    results["pad_worst_mad"]   = max(pad_mads) if pad_mads else 0.0
    results["pad_mean_mad"]    = float(np.mean(pad_mads)) if pad_mads else 0.0

    # ── Recurrent state ──
    gr = gpu["linear_recurrent_state_out"]
    ar = ane["linear_recurrent_state_out"]
    rec_mads = {}
    for li in range(CHUNK0_LAYERS):
        if not CHUNK0_LIN[li]:
            continue
        m, _, c = cmp(gr[li], ar[li])
        rec_mads[li] = m
    results["rec_state"] = rec_mads

    # ── Print summary ──
    print(f"\n  [hidden_states] MAD={mad:.4f}  mean={mean:.6f}  cos={cos:.6f}")
    print(f"  [valid tokens]  worst_MAD={results['valid_worst_mad']:.4f}  mean_MAD={results['valid_mean_mad']:.6f}")
    print(f"  [pad tokens]    worst_MAD={results['pad_worst_mad']:.4f}  mean_MAD={results['pad_mean_mad']:.6f}")
    for li, m in sorted(rec_mads.items()):
        print(f"  [rec_state L{li}] MAD={m:.4f}")

    return results


# ── export ──────────────────────────────────────────────────────────
def load_model():
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    print("\nLoading model...", end="", flush=True)
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length   = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f" {time.time()-t0:.1f}s")
    return model


def embed_prompt(model, prompt="Hello, how are you doing today?"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    ids = tok.encode(prompt, add_special_tokens=True)
    print(f"  Prompt: {prompt!r} → {len(ids)} tokens")
    input_ids = torch.zeros(1, BATCH_SIZE, dtype=torch.long)
    input_ids[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids[0])
        hidden = hidden.unsqueeze(0).to(torch.float16)
    return hidden.cpu().numpy(), len(ids)


def export_chunk0(model, mask_padding, out_path):
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
    tag = "MASKED" if mask_padding else "BASELINE"
    print(f"\n[EXPORT {tag}] Converting chunk0 (LUT6, mask_padding={mask_padding})...")
    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=FFN_PER_CH,
    )
    ml = conv.convert_part_2_prefill(
        model, chunk_idx=0, total_chunks=NUM_CHUNKS,
        mask_padding_hidden_states=mask_padding,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ml.save(out_path)
    elapsed = time.time() - t0
    sz_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_path)
        for f in fns
    ) / 1e6
    print(f"  Saved {out_path} ({sz_mb:.0f}MB, {elapsed:.0f}s)")
    del ml, conv
    gc.collect()


# ── main ────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PADDING-HIDDEN-STATE ZEROING MITIGATION TEST")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    model = load_model()
    hidden_np, valid_len = embed_prompt(model)
    inputs = make_inputs(hidden_np, valid_len)

    baseline_path = os.path.join(OUT_DIR, "baseline_chunk0.mlpackage")
    masked_path   = os.path.join(OUT_DIR, "masked_chunk0.mlpackage")

    # ── Phase 1: Baseline (no masking) ──
    export_chunk0(model, mask_padding=False, out_path=baseline_path)
    baseline = measure_parity(baseline_path, inputs, valid_len, "BASELINE")
    # Delete to free disk
    print(f"  Deleting baseline model to free disk...")
    shutil.rmtree(baseline_path)
    gc.collect()

    # ── Phase 2: Masked ──
    export_chunk0(model, mask_padding=True, out_path=masked_path)
    masked = measure_parity(masked_path, inputs, valid_len, "MASKED")
    # Delete to free disk
    print(f"  Deleting masked model to free disk...")
    shutil.rmtree(masked_path)
    gc.collect()

    # ── Phase 3: Comparison ──
    print("\n" + "=" * 70)
    print("COMPARISON: BASELINE vs MASKED (padding-hidden-state zeroing)")
    print("=" * 70)

    headers = ["Metric", "Baseline", "Masked", "Delta"]
    rows = []

    def row(name, b, m):
        d = m - b
        sign = "+" if d >= 0 else ""
        rows.append((name, f"{b:.4f}", f"{m:.4f}", f"{sign}{d:.4f}"))

    row("hidden_states MAD",  baseline["hidden_all"]["mad"],    masked["hidden_all"]["mad"])
    row("hidden_states mean", baseline["hidden_all"]["mean"],   masked["hidden_all"]["mean"])
    row("hidden_states cos",  baseline["hidden_all"]["cos"],    masked["hidden_all"]["cos"])
    row("valid worst MAD",    baseline["valid_worst_mad"],      masked["valid_worst_mad"])
    row("valid mean MAD",     baseline["valid_mean_mad"],       masked["valid_mean_mad"])
    row("pad worst MAD",      baseline["pad_worst_mad"],        masked["pad_worst_mad"])
    row("pad mean MAD",       baseline["pad_mean_mad"],         masked["pad_mean_mad"])

    # rec_state comparison
    all_layers = sorted(set(baseline["rec_state"].keys()) | set(masked["rec_state"].keys()))
    for li in all_layers:
        b = baseline["rec_state"].get(li, 0.0)
        m = masked["rec_state"].get(li, 0.0)
        row(f"rec_state L{li} MAD", b, m)

    # Print table
    widths = [max(len(r[i]) for r in rows + [headers]) for i in range(4)]
    fmt = "  ".join(f"{{:{w}s}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))

    # Verdict
    bm = baseline["hidden_all"]["mad"]
    mm = masked["hidden_all"]["mad"]
    imp = (bm - mm) / bm * 100 if bm > 0 else 0
    print(f"\nHidden-states MAD improvement: {imp:+.1f}%")
    if imp > 10:
        print("** MITIGATION MATERIALLY HELPS **")
    else:
        print("** MITIGATION DOES NOT MATERIALLY HELP **")

    # Clean up temp dir
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    del model
    gc.collect()
    print("\nDone.")


if __name__ == "__main__":
    main()
