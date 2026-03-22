#!/usr/bin/env python3
"""Test loading combined dedup models with BOTH infer and prefill function_name.

Goal: determine if we can load 10 models on 16 GB ANE without segfault
when the prefill models share weights with the infer models (same .mlpackage).
"""
import os, sys, time
import numpy as np
import coremltools as ct

MODEL_DIR = "qwen3_5_stable_models"
COMBINED_DIR = os.path.join(MODEL_DIR, "combined_LUT4_dedup")
CU = ct.ComputeUnit.CPU_AND_NE
NUM_CHUNKS = 4


def find_model(base_dir, name):
    for ext in (".mlpackage",):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def main():
    embed_path = find_model(MODEL_DIR, "embeddings")
    lmhead_path = find_model(MODEL_DIR, "lm_head")

    print("=== Loading embed ===", flush=True)
    embed = ct.models.MLModel(embed_path, compute_units=CU)
    print("  OK", flush=True)

    print("=== Loading lm_head ===", flush=True)
    lmhead = ct.models.MLModel(lmhead_path, compute_units=CU)
    print("  OK", flush=True)

    infer_models = []
    print("\n=== Loading 4x infer chunks ===", flush=True)
    for ci in range(NUM_CHUNKS):
        path = find_model(COMBINED_DIR, f"chunk{ci}")
        print(f"  chunk{ci} (infer)...", flush=True)
        m = ct.models.MLModel(path, compute_units=CU, function_name="infer")
        m.make_state()
        infer_models.append(m)
        print(f"  chunk{ci} OK", flush=True)

    prefill_models = []
    print("\n=== Loading 4x prefill chunks ===", flush=True)
    for ci in range(NUM_CHUNKS):
        path = find_model(COMBINED_DIR, f"chunk{ci}")
        print(f"  chunk{ci} (prefill)...", flush=True)
        m = ct.models.MLModel(path, compute_units=CU, function_name="prefill")
        m.make_state()
        prefill_models.append(m)
        print(f"  chunk{ci} OK", flush=True)

    print(f"\n=== ALL 10 MODELS LOADED SUCCESSFULLY ===", flush=True)
    print(f"  embed: {embed.function_name}", flush=True)
    print(f"  lmhead: {lmhead.function_name}", flush=True)
    for ci in range(NUM_CHUNKS):
        print(f"  infer chunk{ci}: fn={infer_models[ci].function_name}", flush=True)
        print(f"  prefill chunk{ci}: fn={prefill_models[ci].function_name}", flush=True)

    # Test: can infer and prefill share a state?
    print("\n=== Testing shared state ===", flush=True)
    state = infer_models[0].make_state()
    # Run infer
    h = np.zeros((1, 1, 2560), dtype=np.float16)
    pos = np.array([0], dtype=np.int32)
    mask = np.full((1, 1, 1, 1024), -65504.0, dtype=np.float16)
    mask[0, 0, 0, 0] = 0
    lc = np.zeros((8, 1024, 32), dtype=np.float16)
    lr = np.zeros((8, 32, 128, 128), dtype=np.float16)
    inp_infer = {
        "hidden_states": h, "position_ids": pos,
        "causal_mask": mask, "current_pos": pos,
        "linear_conv_state": lc, "linear_recurrent_state": lr,
    }
    out = infer_models[0].predict(inp_infer, state=state)
    print(f"  infer predict OK, output shape: {out['output_hidden_states'].shape}", flush=True)

    # Run prefill on SAME state
    h2 = np.zeros((1, 256, 2560), dtype=np.float16)
    pos2 = np.arange(0, 256, dtype=np.int32)
    mask2 = np.full((1, 1, 256, 1024), -65504.0, dtype=np.float16)
    for i in range(256):
        mask2[0, 0, i, :i+1] = 0
    cur2 = np.array([0], dtype=np.int32)
    inp_pf = {
        "hidden_states": h2, "position_ids": pos2,
        "causal_mask": mask2, "current_pos": cur2,
        "linear_conv_state": lc, "linear_recurrent_state": lr,
    }
    out2 = prefill_models[0].predict(inp_pf, state=state)
    print(f"  prefill predict OK, output shape: {out2['output_hidden_states'].shape}", flush=True)

    print("\n=== ALL TESTS PASSED ===", flush=True)


if __name__ == "__main__":
    main()
