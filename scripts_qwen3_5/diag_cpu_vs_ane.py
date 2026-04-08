#!/usr/bin/env python3
"""Minimal diagnostic: CPU vs ANE single-token decode divergence.

Loads the same models twice (CPU_ONLY and CPU_AND_NE), feeds identical
inputs, and compares outputs at each stage:
  1. Embeddings
  2. Each FFN chunk (with state)
  3. LM head

This isolates WHERE the CPU backend diverges from ANE.
"""
import os, sys, time
import numpy as np
import coremltools as ct

# Use the LUT6 decode-only models (same ones both backends run)
MODEL_DIR = "/Users/yw68/Anemll/qwen3_5_blockrecur_full"
CTX = 2048
NUM_CHUNKS = 6

def find_model(base, name):
    for ext in (".mlpackage", ".mlmodelc"):
        p = os.path.join(base, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in {base}")


def load_pair(path, label=""):
    """Load same model for CPU and ANE."""
    print(f"  Loading {label} CPU...", end="", flush=True)
    t0 = time.time()
    m_cpu = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f" {time.time()-t0:.1f}s", end="")
    print(f"  ANE...", end="", flush=True)
    t0 = time.time()
    m_ane = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f" {time.time()-t0:.1f}s")
    return m_cpu, m_ane


def compare_arrays(a, b, label):
    """Compare two arrays, print stats."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    if a.size == 0 or b.size == 0:
        print(f"  {label}: EMPTY")
        return
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos = np.dot(a, b) / (norm_a * norm_b + 1e-30)
    mad = np.abs(a - b).max()
    mean_ad = np.abs(a - b).mean()
    rel_diff = mad / (np.abs(b).max() + 1e-30)
    print(f"  {label}: cos={cos:.8f}  max_abs_diff={mad:.6f}  mean_abs_diff={mean_ad:.8f}  rel_max={rel_diff:.6f}  norm_cpu={norm_a:.2f}  norm_ane={norm_b:.2f}")


def get_chunk_shapes(model):
    """Get linear state shapes from model spec."""
    conv_shape = rec_shape = None
    for inp in model.get_spec().description.input:
        name = inp.name
        shape_obj = inp.type.multiArrayType.shape
        shp = tuple(int(d) for d in shape_obj)
        if name == "linear_conv_state":
            conv_shape = shp
        elif name == "linear_recurrent_state":
            rec_shape = shp
    return conv_shape, rec_shape


def main():
    print("=" * 80)
    print("  CPU vs ANE Single-Token Diagnostic")
    print("=" * 80)

    # --- Load models ---
    print("\n[1] Loading embeddings...")
    embed_cpu, embed_ane = load_pair(find_model(MODEL_DIR, "embeddings"), "embeddings")

    print("\n[2] Loading lm_head...")
    lmhead_cpu, lmhead_ane = load_pair(find_model(MODEL_DIR, "lm_head_logits"), "lm_head")

    print("\n[3] Loading FFN chunks...")
    ffn_cpus, ffn_anes = [], []
    for ci in range(NUM_CHUNKS):
        path = find_model(MODEL_DIR, f"ffn_LUT6_chunk{ci}")
        cpu, ane = load_pair(path, f"chunk{ci}")
        ffn_cpus.append(cpu)
        ffn_anes.append(ane)

    # --- Create states ---
    print("\n[4] Creating states...")
    states_cpu = [m.make_state() for m in ffn_cpus]
    states_ane = [m.make_state() for m in ffn_anes]

    # Get linear state shapes per chunk
    chunk_shapes = [get_chunk_shapes(m) for m in ffn_cpus]
    lin_convs_cpu = [np.zeros(s[0], dtype=np.float16) for s in chunk_shapes]
    lin_recs_cpu  = [np.zeros(s[1], dtype=np.float16) for s in chunk_shapes]
    lin_convs_ane = [np.zeros(s[0], dtype=np.float16) for s in chunk_shapes]
    lin_recs_ane  = [np.zeros(s[1], dtype=np.float16) for s in chunk_shapes]

    # --- Run multiple tokens to let error accumulate ---
    # Use a fixed token sequence (e.g. "Hello world")
    test_tokens = [9906, 1879, 11, 358, 1079]  # "Hello world, I am"
    
    print(f"\n[5] Running {len(test_tokens)} tokens through both backends...")
    print(f"    Tokens: {test_tokens}")

    for step, tok_id in enumerate(test_tokens):
        pos = step
        print(f"\n--- Step {step}: token={tok_id}, pos={pos} ---")

        # Embed
        tok_arr = np.array([[tok_id]], dtype=np.int32)
        hidden_cpu = list(embed_cpu.predict({"input_ids": tok_arr}).values())[0]
        hidden_ane = list(embed_ane.predict({"input_ids": tok_arr}).values())[0]
        compare_arrays(hidden_cpu, hidden_ane, "embed_out")

        # Mask
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        pos_arr = np.array([pos], dtype=np.int32)

        # FFN chunks
        for ci in range(NUM_CHUNKS):
            inp_cpu = {
                "hidden_states": hidden_cpu.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": lin_convs_cpu[ci],
                "linear_recurrent_state": lin_recs_cpu[ci],
            }
            inp_ane = {
                "hidden_states": hidden_ane.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": lin_convs_ane[ci],
                "linear_recurrent_state": lin_recs_ane[ci],
            }

            out_cpu = ffn_cpus[ci].predict(inp_cpu, state=states_cpu[ci])
            out_ane = ffn_anes[ci].predict(inp_ane, state=states_ane[ci])

            hidden_cpu = out_cpu["output_hidden_states"]
            hidden_ane = out_ane["output_hidden_states"]

            compare_arrays(hidden_cpu, hidden_ane, f"  chunk{ci}_hidden")

            if "linear_conv_state_out" in out_cpu:
                compare_arrays(out_cpu["linear_conv_state_out"],
                             out_ane["linear_conv_state_out"], f"  chunk{ci}_conv_st")
                compare_arrays(out_cpu["linear_recurrent_state_out"],
                             out_ane["linear_recurrent_state_out"], f"  chunk{ci}_rec_st")
                lin_convs_cpu[ci] = out_cpu["linear_conv_state_out"]
                lin_recs_cpu[ci]  = out_cpu["linear_recurrent_state_out"]
                lin_convs_ane[ci] = out_ane["linear_conv_state_out"]
                lin_recs_ane[ci]  = out_ane["linear_recurrent_state_out"]

        # LM head
        lm_cpu = lmhead_cpu.predict({"hidden_states": hidden_cpu.astype(np.float16)})
        lm_ane = lmhead_ane.predict({"hidden_states": hidden_ane.astype(np.float16)})

        # Get top-5 from each
        logits_cpu_parts = sorted([k for k in lm_cpu if k.startswith("logits")])
        logits_ane_parts = sorted([k for k in lm_ane if k.startswith("logits")])
        
        if logits_cpu_parts:
            full_cpu = np.concatenate([lm_cpu[k].flatten() for k in logits_cpu_parts])
            full_ane = np.concatenate([lm_ane[k].flatten() for k in logits_ane_parts])
            compare_arrays(full_cpu, full_ane, "  logits")
            
            top5_cpu = np.argsort(full_cpu)[-5:][::-1]
            top5_ane = np.argsort(full_ane)[-5:][::-1]
            argmax_cpu = top5_cpu[0]
            argmax_ane = top5_ane[0]
            match = "✓" if argmax_cpu == argmax_ane else "✗ MISMATCH"
            print(f"  argmax: CPU={argmax_cpu} ANE={argmax_ane} {match}")
            print(f"  top5 CPU: {list(top5_cpu)}")
            print(f"  top5 ANE: {list(top5_ane)}")

    # --- Now test with SHARED inputs to isolate per-chunk divergence ---
    print(f"\n\n{'='*80}")
    print("  ISOLATED CHUNK TEST: Same input to CPU and ANE")
    print("  (reset states, feed identical hidden_states)")
    print(f"{'='*80}")

    # Reset
    states_cpu2 = [m.make_state() for m in ffn_cpus]
    states_ane2 = [m.make_state() for m in ffn_anes]
    
    tok_arr = np.array([[9906]], dtype=np.int32)  # "Hello"
    hidden = list(embed_ane.predict({"input_ids": tok_arr}).values())[0]  # use ANE embed as ground truth
    
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :1] = 0
    pos_arr = np.array([0], dtype=np.int32)
    
    for ci in range(NUM_CHUNKS):
        conv_shape, rec_shape = chunk_shapes[ci]
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_arr,
            "causal_mask": mask,
            "current_pos": pos_arr,
            "linear_conv_state": np.zeros(conv_shape, dtype=np.float16),
            "linear_recurrent_state": np.zeros(rec_shape, dtype=np.float16),
        }
        
        out_cpu = ffn_cpus[ci].predict(inp, state=states_cpu2[ci])
        out_ane = ffn_anes[ci].predict(inp, state=states_ane2[ci])
        
        compare_arrays(out_cpu["output_hidden_states"],
                      out_ane["output_hidden_states"], f"chunk{ci}_hidden (same input)")
        
        if "linear_conv_state_out" in out_cpu:
            compare_arrays(out_cpu["linear_conv_state_out"],
                          out_ane["linear_conv_state_out"], f"chunk{ci}_conv_st (same input)")
            compare_arrays(out_cpu["linear_recurrent_state_out"],
                          out_ane["linear_recurrent_state_out"], f"chunk{ci}_rec_st (same input)")
        
        # Feed ANE output to next chunk (ground truth chain)
        hidden = out_ane["output_hidden_states"]


if __name__ == "__main__":
    main()
