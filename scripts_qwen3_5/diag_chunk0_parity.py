#!/usr/bin/env python3
"""Measure chunk0 GPU-vs-ANE parity improvement from stable-softplus fix.

Compares:
  OLD: existing prefill_LUT6_chunk0.mlpackage (exported before fix)
  NEW: freshly exported chunk0 prefill (with stable softplus)

For each, runs on CPU_AND_GPU and CPU_AND_NE, compares per-layer rec_state MAD.
"""

import os, sys, time, gc
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
MODEL_DIR    = os.path.join(REPO_ROOT, "qwen3_5_stable_models_6chunk")
OLD_PREFILL  = os.path.join(MODEL_DIR, "prefill_LUT6_chunk0.mlpackage")

NEW_DIR      = "/tmp/diag_chunk0_fixed"
NEW_PREFILL  = os.path.join(NEW_DIR, "prefill_LUT6_chunk0.mlpackage")

NUM_V_HEADS   = 32
KEY_HEAD_DIM  = 128
VAL_HEAD_DIM  = 128
CHUNK0_LAYERS = 6
CHUNK0_LIN    = [True, True, True, False, True, True]  # layer 3 = full attn


# ── helpers ─────────────────────────────────────────────────────────
def cmp(name, a, b):
    a_f = a.astype(np.float32).flatten()
    b_f = b.astype(np.float32).flatten()
    diff = a_f - b_f
    mad  = np.max(np.abs(diff))
    mean = np.mean(np.abs(diff))
    dot  = np.dot(a_f, b_f)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
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


def run_prefill(model_path, inputs, compute_unit, label):
    """Load a prefill model and run prediction. Returns outputs dict."""
    print(f"  Loading {label}...", end="", flush=True)
    t0 = time.time()
    mlmodel = ct.models.MLModel(model_path, compute_units=compute_unit)
    print(f" {time.time()-t0:.1f}s", end="", flush=True)

    state = mlmodel.make_state()
    t0 = time.time()
    out = mlmodel.predict(inputs, state=state)
    print(f"  predict {time.time()-t0:.1f}s")
    del mlmodel; gc.collect()
    return out


def compare_gpu_ane(model_path, inputs, tag):
    """Run one model on GPU and ANE, return per-layer comparison."""
    print(f"\n[{tag}] {model_path}")
    gpu_out = run_prefill(model_path, inputs, ct.ComputeUnit.CPU_AND_GPU, f"{tag}/GPU")
    ane_out = run_prefill(model_path, inputs, ct.ComputeUnit.CPU_AND_NE,  f"{tag}/ANE")

    gpu_rec = gpu_out["linear_recurrent_state_out"]
    ane_rec = ane_out["linear_recurrent_state_out"]
    gpu_conv = gpu_out["linear_conv_state_out"]
    ane_conv = ane_out["linear_conv_state_out"]
    gpu_hid = gpu_out["output_hidden_states"]
    ane_hid = ane_out["output_hidden_states"]

    results = {"layers": {}, "hidden": {}, "conv_layers": {}}

    # Per-layer rec_state
    for li in range(CHUNK0_LAYERS):
        if not CHUNK0_LIN[li]:
            continue
        mad, mean, cos = cmp(f"layer{li}", gpu_rec[li], ane_rec[li])
        results["layers"][li] = {"mad": mad, "mean": mean, "cos": cos}

        # Per-head for this layer
        heads = {}
        for h in range(NUM_V_HEADS):
            hm, _, hc = cmp(f"h{h}", gpu_rec[li][h], ane_rec[li][h])
            heads[h] = hm
        results["layers"][li]["heads"] = heads

    # Per-layer conv_state
    for li in range(CHUNK0_LAYERS):
        if not CHUNK0_LIN[li]:
            continue
        mad, mean, cos = cmp(f"conv{li}", gpu_conv[li], ane_conv[li])
        results["conv_layers"][li] = {"mad": mad, "mean": mean, "cos": cos}

    # Hidden states
    hid_mad, hid_mean, hid_cos = cmp("hidden", gpu_hid, ane_hid)
    results["hidden"] = {"mad": hid_mad, "mean": hid_mean, "cos": hid_cos}

    return results


# ── export new chunk0 ───────────────────────────────────────────────
def export_new_chunk0():
    """Export chunk0 prefill using the PATCHED model code."""
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    print("\n[EXPORT] Loading patched model...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length   = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("[EXPORT] Converting chunk0 prefill (LUT6 gs=4)...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=FFN_PER_CH)
    ml = conv.convert_part_2_prefill(model, chunk_idx=0, total_chunks=NUM_CHUNKS)

    os.makedirs(NEW_DIR, exist_ok=True)
    ml.save(NEW_PREFILL)
    print(f"  Saved to {NEW_PREFILL} ({time.time()-t0:.1f}s)")

    # Also return the model for embeddings
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


# ── main ────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("CHUNK0 PARITY MEASUREMENT: stable-softplus fix")
    print("=" * 80)

    # Step 1: Export new chunk0 (also gives us the model for embedding)
    model = export_new_chunk0()
    hidden_np, valid_len = embed_prompt(model)
    del model; gc.collect()

    inputs = make_inputs(hidden_np, valid_len)

    # Step 2: Run OLD model (from disk)
    old_results = compare_gpu_ane(OLD_PREFILL, inputs, "OLD (before fix)")

    # Step 3: Run NEW model (freshly exported)
    new_results = compare_gpu_ane(NEW_PREFILL, inputs, "NEW (stable softplus)")

    # ── Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CHUNK0 GPU-vs-ANE PARITY: BEFORE vs AFTER stable-softplus fix")
    print("=" * 80)

    print(f"\n{'Layer':<8} {'OLD MAD':>10} {'NEW MAD':>10} {'Improv':>10} {'OLD cos':>10} {'NEW cos':>10}")
    print("-" * 60)

    for li in sorted(set(old_results["layers"].keys()) | set(new_results["layers"].keys())):
        old = old_results["layers"].get(li, {})
        new = new_results["layers"].get(li, {})
        old_mad = old.get("mad", float("nan"))
        new_mad = new.get("mad", float("nan"))
        old_cos = old.get("cos", float("nan"))
        new_cos = new.get("cos", float("nan"))
        if old_mad > 0:
            ratio = new_mad / old_mad
            improv = f"{ratio:.4f}x"
        else:
            improv = "n/a"
        print(f"  L{li:<5} {old_mad:>10.4f} {new_mad:>10.4f} {improv:>10} {old_cos:>10.6f} {new_cos:>10.6f}")

    # Conv state comparison
    print(f"\n{'Layer':<8} {'OLD conv':>10} {'NEW conv':>10}")
    print("-" * 30)
    for li in sorted(old_results["conv_layers"].keys()):
        old_c = old_results["conv_layers"][li]["mad"]
        new_c = new_results["conv_layers"][li]["mad"]
        print(f"  L{li:<5} {old_c:>10.4f} {new_c:>10.4f}")

    # Hidden states
    oh = old_results["hidden"]
    nh = new_results["hidden"]
    print(f"\n  hidden_states:  OLD MAD={oh['mad']:.4f} cos={oh['cos']:.6f}"
          f"  →  NEW MAD={nh['mad']:.4f} cos={nh['cos']:.6f}")

    # Per-head detail for layer 0 (the worst offender)
    print(f"\n{'='*80}")
    print("LAYER 0 PER-HEAD: GPU-vs-ANE rec_state MAD")
    print(f"{'='*80}")
    print(f"{'Head':<6} {'OLD MAD':>10} {'NEW MAD':>10} {'Improv':>10}")
    print("-" * 38)
    old_h = old_results["layers"][0]["heads"]
    new_h = new_results["layers"][0]["heads"]
    for h in range(NUM_V_HEADS):
        om = old_h[h]
        nm = new_h[h]
        r = nm / (om + 1e-12)
        flag = " <<<" if om > 0.5 else ""
        print(f"  {h:>3}   {om:>10.4f} {nm:>10.4f} {r:>10.4f}x{flag}")

    # Summary
    all_old = [old_results["layers"][li]["mad"] for li in old_results["layers"]]
    all_new = [new_results["layers"][li]["mad"] for li in new_results["layers"]]
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  rec_state max MAD across layers:  OLD={max(all_old):.4f}  →  NEW={max(all_new):.4f}")
    print(f"  rec_state avg MAD across layers:  OLD={np.mean(all_old):.4f}  →  NEW={np.mean(all_new):.4f}")
    print(f"  hidden output MAD:                OLD={oh['mad']:.4f}  →  NEW={nh['mad']:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
