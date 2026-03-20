#!/usr/bin/env python3
"""Prefill parity validation: PyTorch vs CoreML (ANE) for all 4 Qwen3.5-4B chunks.

Phase 1: Generate PyTorch reference outputs (batch_size=64).
Phase 2: Load each CoreML chunk on ANE, compare against PyTorch.

Usage:
    python tests/dev/_prefill_parity_val.py           # Run both phases
    python tests/dev/_prefill_parity_val.py --phase 2  # Only CoreML comparison (refs must exist)
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os
import gc
import argparse
import time
import numpy as np
import torch

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
COREML_DIR = "/tmp/qwen35_ane_test/reshape_test"
REF_DIR = "/tmp/qwen35_prefill_parity_b64"
SEQ_LEN = 64       # Must match the CoreML export batch_size
CONTEXT_LEN = 256   # Must match the CoreML export context_length
NUM_CHUNKS = 4
CHUNKS = [(0, 8), (8, 16), (16, 24), (24, 32)]


def cosine_sim(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


def report_parity(name, ref, cml):
    """Print parity metrics between reference and CoreML outputs."""
    ref_f = ref.astype(np.float32)
    cml_f = cml.astype(np.float32)
    diff = np.abs(ref_f - cml_f)
    flat = diff.flatten()
    cos = cosine_sim(ref, cml)

    print(f"  {name}:")
    print(f"    max_abs={diff.max():.6f}  mean_abs={diff.mean():.8f}  cosine={cos:.10f}")
    print(f"    p99={np.percentile(flat, 99):.6f}  p95={np.percentile(flat, 95):.6f}  "
          f"p50={np.percentile(flat, 50):.8f}")

    # Top-3 worst positions
    worst = np.argsort(flat)[-3:][::-1]
    for w in worst:
        pos = np.unravel_index(w, diff.shape)
        print(f"    worst: pos={pos} ref={ref_f[pos]:.6f} cml={cml_f[pos]:.6f} diff={diff[pos]:.6f}")
    return diff.max(), diff.mean(), cos


# ---------- Phase 1: Generate PyTorch references ----------
def phase1():
    from transformers import AutoTokenizer
    from anemll.models.qwen3_5_model import (
        Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape, MODEL_DTYPE,
    )

    os.makedirs(REF_DIR, exist_ok=True)
    print("=" * 60)
    print("PHASE 1: Generating PyTorch reference outputs")
    print("=" * 60)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    prompt = "Explain stack and heap memory in one paragraph and give one debugging tip."
    text = prompt
    while True:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
        if ids.shape[1] >= SEQ_LEN:
            ids = ids[:, :SEQ_LEN]
            break
        text = text + " " + prompt
    np.save(os.path.join(REF_DIR, "input_ids.npy"), ids.numpy().astype(np.int32))
    print(f"Tokenized: {ids.shape}")

    # Load model
    print("Loading model...")
    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CONTEXT_LEN
    cfg.state_length = CONTEXT_LEN
    model = Qwen35ForCausalLM(cfg)
    ok = model.load_pretrained_weights(MODEL_PATH)
    assert ok, "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Embeddings
    with torch.no_grad():
        embed = model.model.embed_tokens(ids.to(torch.int32)).to(torch.float16)
    np.save(os.path.join(REF_DIR, "embed_out.npy"), embed.numpy())
    print(f"Embeddings: {embed.shape}")

    # Prefill inputs
    position_ids = torch.arange(SEQ_LEN, dtype=torch.int32)
    causal_mask = torch.full((1, 1, SEQ_LEN, CONTEXT_LEN), float("-inf"), dtype=torch.float16)
    for r in range(SEQ_LEN):
        causal_mask[0, 0, r, :r + 1] = 0
    current_pos = torch.tensor([0], dtype=torch.int32)

    # Conv/rec state in ANE-safe shape (matches CoreML export)
    conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)
    print(f"conv_state ANE shape: ({ane_d1}, {ane_d2})")

    hidden = embed.clone()
    with torch.no_grad():
        for ci, (s, e) in enumerate(CHUNKS):
            local_layers = e - s
            k_cache = torch.zeros(
                (local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE)
            v_cache = torch.zeros(
                (local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE)
            conv_st = torch.zeros((local_layers, ane_d1, ane_d2), dtype=MODEL_DTYPE)
            rec_st = torch.zeros(
                (local_layers, cfg.text_config.linear_num_value_heads,
                 cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
                dtype=MODEL_DTYPE)

            # Set export expected sizes on linear-attention layers
            for layer_idx in range(s, e):
                layer = model.model.layers[layer_idx]
                if getattr(layer, "layer_type", None) == "linear_attention":
                    layer.self_attn.export_expected_batch_size = 1
                    layer.self_attn.export_expected_seq_len = SEQ_LEN

            hidden = model.model.process_layers_prefill_export_local_state(
                hidden_states=hidden,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                kv_cache_0=None,
                k_cache=k_cache,
                v_cache=v_cache,
                linear_conv_state=conv_st,
                linear_recurrent_state=rec_st,
                start_layer=s,
                end_layer=e,
                apply_final_norm=False,
                expected_batch_size=1,
                expected_seq_len=SEQ_LEN,
            )
            np.save(os.path.join(REF_DIR, f"torch_chunk{ci+1}.npy"), hidden.numpy())
            print(f"  Chunk {ci+1} (layers {s}-{e-1}): {hidden.shape}")

    # Also save the last-chunk output sliced to first token (matches CoreML)
    np.save(os.path.join(REF_DIR, "torch_chunk4_tok0.npy"), hidden[:, 0:1, :].numpy())

    del model, hidden, embed
    gc.collect()
    print("Phase 1 complete.\n")


# ---------- Phase 2: CoreML comparison ----------
def phase2():
    import coremltools as ct

    print("=" * 60)
    print("PHASE 2: CoreML (ANE) parity comparison")
    print("=" * 60)

    # Verify refs exist
    for ci in range(1, NUM_CHUNKS + 1):
        p = os.path.join(REF_DIR, f"torch_chunk{ci}.npy")
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}. Run with --phase 1 first.")
            sys.exit(1)

    # Load PyTorch references
    embed = np.load(os.path.join(REF_DIR, "embed_out.npy"))
    torch_chunks = {}
    for ci in range(1, NUM_CHUNKS + 1):
        torch_chunks[ci] = np.load(os.path.join(REF_DIR, f"torch_chunk{ci}.npy"))
    print(f"Loaded PyTorch references (embed={embed.shape})")

    # Build inputs
    position_ids = np.arange(SEQ_LEN, dtype=np.int32)
    causal_mask = np.full((1, 1, SEQ_LEN, CONTEXT_LEN), -65504.0, dtype=np.float16)
    for r in range(SEQ_LEN):
        causal_mask[0, 0, r, :r + 1] = 0
    current_pos = np.zeros((1,), dtype=np.int32)

    summary = []
    prev_hidden = embed.copy()  # cascaded input starts from embed

    for ci in range(1, NUM_CHUNKS + 1):
        chunk_path = os.path.join(COREML_DIR, f"prefill_chunk{ci}.mlpackage")
        if not os.path.exists(chunk_path):
            print(f"ERROR: Missing {chunk_path}")
            summary.append((ci, "MISSING", None, None, None))
            continue

        print(f"\n--- Chunk {ci}: {chunk_path} ---")

        # Load CoreML on ANE
        t0 = time.time()
        model = ct.models.MLModel(chunk_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f"  Loaded in {time.time()-t0:.1f}s")

        # Read input spec to confirm shapes
        spec = model.get_spec()
        for inp in spec.description.input:
            if inp.type.HasField("multiArrayType"):
                sh = tuple(inp.type.multiArrayType.shape)
                print(f"  Input {inp.name}: {sh}")

        state = model.make_state()

        # ---- ISOLATED comparison: feed PyTorch output from previous chunk ----
        if ci == 1:
            iso_input = embed.copy()
        else:
            iso_input = torch_chunks[ci - 1].copy()

        inp_dict = {
            "hidden_states": iso_input.astype(np.float16),
            "position_ids": position_ids,
            "causal_mask": causal_mask,
            "current_pos": current_pos,
        }

        try:
            t0 = time.time()
            out = model.predict(inp_dict, state=state)
            elapsed = time.time() - t0
            cml_hidden = list(out.values())[0]
            print(f"  Predicted in {elapsed:.2f}s, output shape: {cml_hidden.shape}")
        except Exception as e:
            print(f"  ❌ ANE PREDICT FAILED: {str(e)[:200]}")
            summary.append((ci, "ANE_FAIL", None, None, None))
            del model
            gc.collect()
            continue

        # Compare
        th = torch_chunks[ci]
        if cml_hidden.shape != th.shape:
            # Last chunk returns (1,1,2560) but torch has (1,64,2560)
            if cml_hidden.shape[1] < th.shape[1]:
                th_cmp = th[:, :cml_hidden.shape[1], :]
                cml_cmp = cml_hidden
                print(f"  Shape mismatch: comparing first {cml_hidden.shape[1]} token(s)")
            else:
                min_seq = min(th.shape[1], cml_hidden.shape[1])
                th_cmp = th[:, :min_seq, :]
                cml_cmp = cml_hidden[:, :min_seq, :]
        else:
            th_cmp = th
            cml_cmp = cml_hidden

        max_abs, mean_abs, cos = report_parity(f"CHUNK {ci} (isolated)", th_cmp, cml_cmp)
        summary.append((ci, "OK", max_abs, mean_abs, cos))

        # Save CoreML output for cascaded comparison
        np.save(os.path.join(REF_DIR, f"cml_iso_chunk{ci}.npy"), cml_hidden)

        # ---- CASCADED comparison: feed CoreML output forward ----
        if ci > 1:
            prev_cml = np.load(os.path.join(REF_DIR, f"cml_iso_chunk{ci-1}.npy"))
            # Only run cascaded if previous chunk output matches input shape
            if prev_cml.shape == iso_input.shape:
                state2 = model.make_state()
                cas_dict = {
                    "hidden_states": prev_cml.astype(np.float16),
                    "position_ids": position_ids,
                    "causal_mask": causal_mask,
                    "current_pos": current_pos,
                }
                try:
                    cas_out = model.predict(cas_dict, state=state2)
                    cas_hidden = list(cas_out.values())[0]
                    if cas_hidden.shape != th_cmp.shape:
                        cas_cmp = cas_hidden[:, :th_cmp.shape[1], :] if cas_hidden.shape[1] > th_cmp.shape[1] else cas_hidden
                        th_cas = th_cmp[:, :cas_cmp.shape[1], :]
                    else:
                        cas_cmp = cas_hidden
                        th_cas = th_cmp
                    report_parity(f"CHUNK {ci} (cascaded)", th_cas, cas_cmp)
                    # Update cascaded output for next chunk
                    np.save(os.path.join(REF_DIR, f"cml_cas_chunk{ci}.npy"), cas_hidden)
                except Exception as e:
                    print(f"  Cascaded predict failed: {str(e)[:100]}")

        del model, state
        gc.collect()

    # Print summary
    print(f"\n{'='*60}")
    print("PARITY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Chunk':<8} {'Status':<10} {'max_abs':<12} {'mean_abs':<14} {'cosine':<15}")
    print("-" * 60)
    for ci, status, mx, mn, cos in summary:
        if status == "OK":
            # Grade
            if cos is not None and cos > 0.999 and mx < 0.5:
                grade = "✅ GOOD"
            elif cos is not None and cos > 0.99 and mx < 2.0:
                grade = "⚠️  ACCEPTABLE"
            else:
                grade = "❌ BAD"
            print(f"  {ci:<6} {grade:<10} {mx:<12.6f} {mn:<14.8f} {cos:<15.10f}")
        else:
            print(f"  {ci:<6} {status}")


def main():
    parser = argparse.ArgumentParser(description="Prefill parity validation")
    parser.add_argument("--phase", type=int, default=0, help="1=PyTorch only, 2=CoreML only, 0=both")
    args = parser.parse_args()

    if args.phase in (0, 1):
        phase1()
    if args.phase in (0, 2):
        phase2()


if __name__ == "__main__":
    main()
