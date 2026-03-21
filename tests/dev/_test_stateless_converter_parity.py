#!/usr/bin/env python3
"""Test the MODIFIED converter's stateless linear attention export.

This script:
1. Exports chunk 0 using the updated converter (stateless linear attn I/O)
2. Runs teacher-forced decode: same tokens to PyTorch and CoreML
3. Measures hidden state cosine vs PyTorch reference
4. Both use the SAME export path (converter with LUT), so comparison is fair
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, tempfile
import numpy as np
import torch
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 256
CHUNK_IDX = 0
TOTAL_CHUNKS = 4
LAYERS_PER_CHUNK = 8
START_LAYER = 0
END_LAYER = 8
NUM_TOKENS = 50


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def main():
    tmpdir = tempfile.mkdtemp(prefix="stateless_conv_test_")
    print("=" * 70)
    print("  Stateless Converter Parity Test")
    print(f"  Chunk 0 (layers {START_LAYER}-{END_LAYER}), CTX={CTX}")
    print(f"  Temp: {tmpdir}")
    print("=" * 70)

    # ── 1. Load model ──
    print("\n--- Loading model ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    prompt = "Explain the difference between a stack and a queue in computer science."
    while True:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        if ids.shape[1] >= NUM_TOKENS:
            ids = ids[:, :NUM_TOKENS]
            break
        prompt = prompt + " " + prompt

    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)

    # ── 2. Export using the MODIFIED converter (stateless linear attn) ──
    print("\n--- Exporting chunk 0 using modified converter ---")
    converter = Qwen35Converter(model, context_length=CTX, batch_size=1,
                                 num_chunks=TOTAL_CHUNKS, lut_bits=None)
    mlmodel = converter.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=TOTAL_CHUNKS)
    export_path = os.path.join(tmpdir, "stateless_chunk0.mlpackage")
    mlmodel.save(export_path)
    print(f"  Saved to {export_path}")

    # Verify outputs have linear state I/O
    spec = mlmodel.get_spec()
    print(f"  Model inputs:  {[inp.name for inp in spec.description.input]}")
    print(f"  Model outputs: {[out.name for out in spec.description.output]}")
    print(f"  Model states:  {[s.name for s in spec.description.state]}")
    del mlmodel, converter; gc.collect()

    # ── 3. Pre-compute embeddings ──
    print(f"\n--- Pre-computing {NUM_TOKENS} embeddings ---")
    embeddings = []
    with torch.no_grad():
        for pos in range(NUM_TOKENS):
            tok = ids[:, pos:pos+1].to(torch.int32)
            emb = model.model.embed_tokens(tok).to(MODEL_DTYPE)
            embeddings.append(emb)

    # ── 4. PyTorch reference (teacher-forced) ──
    print("\n--- PyTorch reference (teacher-forced) ---")
    pt_k = torch.zeros(LAYERS_PER_CHUNK, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    pt_v = torch.zeros(LAYERS_PER_CHUNK, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    pt_conv = torch.zeros(LAYERS_PER_CHUNK, ane_d1, ane_d2, dtype=MODEL_DTYPE)
    pt_rec = torch.zeros(LAYERS_PER_CHUNK, cfg.text_config.linear_num_value_heads,
                         cfg.text_config.linear_key_head_dim,
                         cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)
    pt_hiddens = []
    with torch.no_grad():
        for pos in range(NUM_TOKENS):
            mask = torch.full((1, 1, 1, CTX), float("-inf"), dtype=MODEL_DTYPE)
            mask[:, :, :, :pos+1] = 0
            hidden = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=embeddings[pos], position_ids=torch.tensor([pos], dtype=torch.int32),
                causal_mask=mask, current_pos=torch.tensor([pos], dtype=torch.int32),
                kv_cache_0=None, k_cache=pt_k, v_cache=pt_v,
                linear_conv_state=pt_conv, linear_recurrent_state=pt_rec,
                start_layer=START_LAYER, end_layer=END_LAYER,
                apply_final_norm=False,
            )
            pt_hiddens.append(hidden.numpy().copy())
            if pos % 10 == 0:
                sys.stdout.write(f"\r  PT: {pos+1}/{NUM_TOKENS}")
                sys.stdout.flush()
    print()

    # ── 5. CoreML stateless (teacher-forced) ──
    print("\n--- CoreML stateless (teacher-forced) ---")
    compute_unit = ct.ComputeUnit.CPU_AND_NE
    cml = ct.models.MLModel(export_path, compute_units=compute_unit)
    state = cml.make_state()

    # Initialize linear states as I/O tensors (zeros)
    lin_conv = np.zeros((LAYERS_PER_CHUNK, ane_d1, ane_d2), dtype=np.float16)
    lin_rec = np.zeros((LAYERS_PER_CHUNK, cfg.text_config.linear_num_value_heads,
                        cfg.text_config.linear_key_head_dim,
                        cfg.text_config.linear_value_head_dim), dtype=np.float16)

    cml_hiddens = []
    for pos in range(NUM_TOKENS):
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos+1] = 0
        out = cml.predict({
            "hidden_states": embeddings[pos].numpy().astype(np.float16),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
            "linear_conv_state": lin_conv,
            "linear_recurrent_state": lin_rec,
        }, state=state)
        cml_hiddens.append(out["output_hidden_states"].copy())
        # Update linear states from outputs for next step
        lin_conv = out["linear_conv_state_out"]
        lin_rec = out["linear_recurrent_state_out"]
        if pos % 10 == 0:
            sys.stdout.write(f"\r  CML: {pos+1}/{NUM_TOKENS}")
            sys.stdout.flush()
    print()
    del cml, state; gc.collect()

    # ── 6. Compare ──
    print(f"\n{'='*70}")
    print(f"  Teacher-Forced Parity: Modified Converter (Stateless + LUT)")
    print(f"{'='*70}")
    print(f"\n{'Pos':>4} {'Cosine vs PT':>14}")
    print("-" * 25)

    cos_all = []
    for pos in range(NUM_TOKENS):
        c = cosine(cml_hiddens[pos], pt_hiddens[pos])
        cos_all.append(c)
        if pos < 20 or pos % 5 == 0 or pos == NUM_TOKENS - 1:
            print(f"  {pos:4d} {c:14.8f}")

    print(f"\n{'='*70}")
    print(f"  SUMMARY ({NUM_TOKENS} tokens, stateless I/O, NO LUT)")
    print(f"{'='*70}")
    print(f"  Avg cosine vs PT:       {np.mean(cos_all):.8f}")
    print(f"  Min cosine vs PT:       {np.min(cos_all):.8f}")
    print(f"  Avg cosine tokens 0-9:  {np.mean(cos_all[:10]):.8f}")
    if NUM_TOKENS > 20:
        print(f"  Avg cosine tokens 10-19:{np.mean(cos_all[10:20]):.8f}")
    if NUM_TOKENS > 30:
        print(f"  Avg cosine tokens 20-29:{np.mean(cos_all[20:30]):.8f}")
    print(f"  Avg cosine last 10:     {np.mean(cos_all[-10:]):.8f}")

    # Compare with previous teacher-forced results (stateful export)
    # Previous: avg=0.918 (stateful), avg=0.971 (stateless manual)
    avg = np.mean(cos_all)
    if avg > 0.95:
        print(f"\n✅ Stateless converter parity is GOOD (avg={avg:.4f})")
    elif avg > 0.90:
        print(f"\n⚠️  Stateless converter parity is FAIR (avg={avg:.4f})")
    else:
        print(f"\n❌ Stateless converter parity is POOR (avg={avg:.4f})")

    print(f"\nCleanup: rm -rf {tmpdir}")


if __name__ == "__main__":
    main()
