#!/usr/bin/env python3
"""Minimal ANE-load probe for padding-mask change.

Exports a fp16 chunk0 prefill model (no LUT6 — skips palettization) with
mask_padding_hidden_states=True, then checks:
1. Does ct.convert succeed?
2. Does the model load on CPU_AND_NE?
3. Does predict() return without error?

This is NOT a parity test — just an ANE loadability check.
Uses BATCH=64 (small) to minimize export time while keeping realistic shapes.
"""

import os, sys, time, gc, shutil
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import coremltools as ct

HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
OUT_DIR  = "/tmp/ane_load_probe"
BATCH    = 64
CTX      = 2048
NUM_CHUNKS = 6
VALID_LEN  = 8

CHUNK0_LAYERS = 6
NUM_V_HEADS   = 32
KEY_HEAD_DIM  = 128
VAL_HEAD_DIM  = 128


def main():
    print("=" * 60)
    print("ANE LOAD PROBE: mask_padding_hidden_states")
    print(f"  batch_size={BATCH}, ctx={CTX}, fp16 (no LUT)")
    print("=" * 60)

    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    # Load model
    print("\n[0] Loading model...", end="", flush=True)
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f" {time.time()-t0:.1f}s")
    hidden_size = cfg.hidden_size

    # Export fp16 (no LUT) with mask_padding=True
    print(f"\n[1] Exporting fp16 chunk0 (mask_padding=True)...")
    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH,
        num_chunks=NUM_CHUNKS, lut_bits=0, per_channel=0,
    )
    ml = conv.convert_part_2_prefill(
        model, chunk_idx=0, total_chunks=NUM_CHUNKS,
        mask_padding_hidden_states=True,
    )
    out_path = os.path.join(OUT_DIR, "probe_masked.mlpackage")
    os.makedirs(OUT_DIR, exist_ok=True)
    ml.save(out_path)
    export_t = time.time() - t0
    print(f"    Export done in {export_t:.1f}s")
    del ml, conv; gc.collect()

    # Build inputs
    pos = np.zeros(BATCH, dtype=np.int32)
    pos[:VALID_LEN] = np.arange(VALID_LEN, dtype=np.int32)
    mask = np.full((1, 1, BATCH, CTX), -65504.0, dtype=np.float16)
    for i in range(VALID_LEN):
        mask[0, 0, i, :i+1] = 0.0
    inputs = {
        "hidden_states":          np.random.randn(1, BATCH, hidden_size).astype(np.float16) * 0.01,
        "position_ids":           pos,
        "causal_mask":            mask,
        "current_pos":            np.array([0], dtype=np.int32),
        "linear_conv_state":      np.zeros((CHUNK0_LAYERS, 1024, 32), dtype=np.float16),
        "linear_recurrent_state": np.zeros((CHUNK0_LAYERS, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM),
                                           dtype=np.float16),
        "valid_len":              np.array([VALID_LEN], dtype=np.int32),
    }

    # Test ANE load
    print("\n[2] Loading on CPU_AND_NE (ANE)...", end="", flush=True)
    t0 = time.time()
    try:
        ane_model = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f" OK ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"\n    ** ANE LOAD FAILED: {e}")
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        return False

    # Test predict
    print("[3] ANE predict...", end="", flush=True)
    t0 = time.time()
    try:
        state = ane_model.make_state()
        out = ane_model.predict(inputs, state=state)
        print(f" OK ({time.time()-t0:.1f}s)")
        hid = out["output_hidden_states"]
        print(f"    output: shape={hid.shape}, range=[{hid.min():.4f}, {hid.max():.4f}]")
        pad_max = float(np.abs(hid.reshape(BATCH, -1)[VALID_LEN:]).max())
        print(f"    padding tokens max|val|: {pad_max:.6f}")
        if pad_max < 1e-3:
            print("    PADDING ZEROED: YES")
        else:
            print(f"    PADDING ZEROED: NO (max={pad_max:.4f})")
    except Exception as e:
        print(f"\n    ** ANE PREDICT FAILED: {e}")
        del ane_model; gc.collect()
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        return False
    del ane_model; gc.collect()

    # Cleanup
    shutil.rmtree(OUT_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    print("RESULT: ANE LOAD PROBE PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
