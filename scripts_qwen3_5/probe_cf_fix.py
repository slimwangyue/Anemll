#!/usr/bin/env python3
"""ANE loadability probe + parity test for the channels-first output fix.

Tests:
1. Does the fixed model export to CoreML?
2. Does it load on ANE?
3. Does predict work?
4. Is the output parity (GPU vs ANE) improved?
"""

import os, sys, time, gc, shutil
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import coremltools as ct
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM, MODEL_DTYPE

HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
OUT_DIR  = "/tmp/ane_probe_cf_fix"
BATCH    = 64
CTX      = 2048
NUM_CHUNKS = 6
VALID_LEN  = 8


def main():
    print("=" * 60)
    print("ANE PROBE: channels-first output fix")
    print("=" * 60)

    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

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

    # Export fp16 chunk0 (has Layer 3 full attention)
    print(f"\n[1] Exporting fp16 chunk0 (batch={BATCH})...", end="", flush=True)
    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH,
        num_chunks=NUM_CHUNKS, lut_bits=0, per_channel=0,
    )
    ml = conv.convert_part_2_prefill(
        model, chunk_idx=0, total_chunks=NUM_CHUNKS,
        mask_padding_hidden_states=True,
    )
    out_path = os.path.join(OUT_DIR, "chunk0_fp16.mlpackage")
    os.makedirs(OUT_DIR, exist_ok=True)
    ml.save(out_path)
    print(f" {time.time()-t0:.1f}s")
    del ml, conv; gc.collect()

    # Build inputs
    hidden_size = cfg.hidden_size
    CHUNK0_LAYERS = 6
    NUM_V_HEADS = 32
    KEY_HEAD_DIM = 128
    VAL_HEAD_DIM = 128

    try:
        from transformers import AutoTokenizer
        tokens = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True).encode(
            "Hello, how are you doing today?")
    except ImportError:
        tokens = [9419, 11, 1204, 513, 488, 3604, 3242, 30]
    V = len(tokens)

    # Compute embeddings for input
    input_ids = torch.zeros(1, BATCH, dtype=torch.long)
    input_ids[0, :V] = torch.tensor(tokens, dtype=torch.long)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
    hidden_np = hidden.detach().numpy().astype(np.float16)

    pos = np.zeros(BATCH, dtype=np.int32)
    pos[:V] = np.arange(V, dtype=np.int32)
    mask = np.full((1, 1, BATCH, CTX), -65504.0, dtype=np.float16)
    for i in range(V):
        mask[0, 0, i, :i+1] = 0.0

    inputs = {
        "hidden_states": hidden_np,
        "position_ids": pos,
        "causal_mask": mask,
        "current_pos": np.array([0], dtype=np.int32),
        "linear_conv_state": np.zeros((CHUNK0_LAYERS, 1024, 32), dtype=np.float16),
        "linear_recurrent_state": np.zeros((CHUNK0_LAYERS, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM), dtype=np.float16),
        "valid_len": np.array([V], dtype=np.int32),
    }

    # Test ANE load
    print("\n[2] Loading on CPU_AND_NE...", end="", flush=True)
    t0 = time.time()
    try:
        ane_model = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f" OK ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"\n  ** ANE LOAD FAILED: {e}")
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        return False

    # Test predict
    print("[3] ANE predict...", end="", flush=True)
    t0 = time.time()
    try:
        state = ane_model.make_state()
        ane_out = ane_model.predict(inputs, state=state)
        print(f" OK ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"\n  ** ANE PREDICT FAILED: {e}")
        del ane_model; gc.collect()
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        return False
    del ane_model; gc.collect()

    # GPU predict for parity
    print("[4] GPU predict...", end="", flush=True)
    t0 = time.time()
    gpu_model = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    state = gpu_model.make_state()
    gpu_out = gpu_model.predict(inputs, state=state)
    print(f" OK ({time.time()-t0:.1f}s)")
    del gpu_model; gc.collect()

    # Parity analysis on valid tokens
    gh = gpu_out["output_hidden_states"].astype(np.float32)
    ah = ane_out["output_hidden_states"].astype(np.float32)
    gh_v = gh.reshape(BATCH, -1)[:V]
    ah_v = ah.reshape(BATCH, -1)[:V]

    diff = gh_v - ah_v
    mad = float(np.max(np.abs(diff)))
    mean = float(np.mean(np.abs(diff)))
    dot = float(np.sum(gh_v * ah_v))
    ng = float(np.linalg.norm(gh_v.flatten()))
    na = float(np.linalg.norm(ah_v.flatten()))
    cos = dot / (ng * na + 1e-12)

    print(f"\n[5] PARITY (valid tokens, chunk0 fp16):")
    print(f"    hidden_states MAD={mad:.6f}  mean={mean:.8f}  cos={cos:.6f}")
    print(f"    GPU range: [{gh_v.min():.4f}, {gh_v.max():.4f}]")
    print(f"    ANE range: [{ah_v.min():.4f}, {ah_v.max():.4f}]")

    # Per-token
    for t in range(V):
        td = float(np.max(np.abs(gh.reshape(BATCH, -1)[t] - ah.reshape(BATCH, -1)[t])))
        print(f"      t{t}: MAD={td:.6f}")

    # Padding check
    pad_max = float(np.abs(ah.reshape(BATCH, -1)[V:]).max())
    print(f"    padding max abs: {pad_max:.6f}")

    shutil.rmtree(OUT_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    if cos > 0.99:
        print(f"RESULT: ANE PROBE PASSED — cos={cos:.6f}, MAD={mad:.6f}")
    else:
        print(f"RESULT: PARITY CONCERN — cos={cos:.6f}, MAD={mad:.6f}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
