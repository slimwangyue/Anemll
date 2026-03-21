#!/usr/bin/env python3
"""Text generation quality test: Compare PyTorch vs CoreML decode token-by-token.

Uses PyTorch for prefill (to populate states correctly), then compares
PyTorch vs CoreML FFN chunks for single-token greedy decode.

Usage:
    python tests/dev/_test_textgen_quality.py [--tokens 50] [--prompt "Hello"]
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, argparse
import numpy as np
import torch
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
EXPORT_DIR = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export"
CTX = 256  # must match exported models context_length
NUM_CHUNKS = 4
CHUNKS = [(0, 8), (8, 16), (16, 24), (24, 32)]


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--prompt", type=str, default="Explain the difference between a stack and a queue in computer science.",
                        help="Input prompt")
    parser.add_argument("--cpu-gpu", action="store_true", help="Use CPU+GPU instead of ANE")
    args = parser.parse_args()
    max_gen = args.tokens
    prompt = args.prompt
    compute_unit = ct.ComputeUnit.CPU_AND_GPU if args.cpu_gpu else ct.ComputeUnit.CPU_AND_NE

    print("=" * 70)
    print("  Text Generation Quality Test")
    print("  PyTorch vs CoreML FFN Chunks (greedy decode)")
    print("=" * 70)

    # ── 1. Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = input_ids.shape[1]
    if prompt_len >= CTX:
        input_ids = input_ids[:, :CTX - max_gen]
        prompt_len = input_ids.shape[1]
    print("Prompt (%d tokens): %s" % (prompt_len, prompt[:80]))

    # ── 2. Load PyTorch model ──
    print("\nLoading PyTorch model...")
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
    layers_per_chunk = 8

    # ── 3. PyTorch greedy decode ──
    print("\n--- PyTorch Greedy Decode ---")
    ids_pt = input_ids.clone().to(torch.int32)

    # Initialize states for PyTorch
    pt_k_caches = [torch.zeros(layers_per_chunk, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
                   for _ in range(NUM_CHUNKS)]
    pt_v_caches = [torch.zeros(layers_per_chunk, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
                   for _ in range(NUM_CHUNKS)]
    pt_conv_states = [torch.zeros(layers_per_chunk, ane_d1, ane_d2, dtype=MODEL_DTYPE)
                      for _ in range(NUM_CHUNKS)]
    pt_rec_states = [torch.zeros(layers_per_chunk, cfg.text_config.linear_num_value_heads,
                                 cfg.text_config.linear_key_head_dim,
                                 cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)
                     for _ in range(NUM_CHUNKS)]

    pt_tokens = []
    pt_hiddens = []  # Save hidden states for comparison

    with torch.no_grad():
        # Prefill: process prompt tokens one at a time
        for pos in range(prompt_len):
            tok = ids_pt[:, pos:pos+1]
            hidden = model.model.embed_tokens(tok).to(MODEL_DTYPE)

            mask = torch.full((1, 1, 1, CTX), float("-inf"), dtype=MODEL_DTYPE)
            mask[:, :, :, :pos+1] = 0
            position_ids = torch.tensor([pos], dtype=torch.int32)
            current_pos = torch.tensor([pos], dtype=torch.int32)

            for ci, (s, e) in enumerate(CHUNKS):
                is_last = (ci == NUM_CHUNKS - 1)
                hidden = model.model.process_layers_regular_single_token_export_local_state(
                    hidden_states=hidden, position_ids=position_ids,
                    causal_mask=mask, current_pos=current_pos,
                    kv_cache_0=None, k_cache=pt_k_caches[ci], v_cache=pt_v_caches[ci],
                    linear_conv_state=pt_conv_states[ci],
                    linear_recurrent_state=pt_rec_states[ci],
                    start_layer=s, end_layer=e,
                    apply_final_norm=is_last,
                )
            if pos % 10 == 0:
                sys.stdout.write("\r  Prefill: %d/%d" % (pos + 1, prompt_len))
                sys.stdout.flush()

        # Get first generated token
        logits = model.lm_head(hidden.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
        next_id = torch.argmax(logits, dim=-1).to(torch.int32)
        pt_tokens.append(next_id.item())
        print("\r  Prefill done. First token: %d (%s)" % (next_id.item(), repr(tokenizer.decode([next_id.item()]))))

        # Generate remaining tokens
        for gi in range(max_gen - 1):
            pos = prompt_len + gi
            hidden = model.model.embed_tokens(next_id).to(MODEL_DTYPE)

            mask = torch.full((1, 1, 1, CTX), float("-inf"), dtype=MODEL_DTYPE)
            mask[:, :, :, :pos+1] = 0
            position_ids = torch.tensor([pos], dtype=torch.int32)
            current_pos = torch.tensor([pos], dtype=torch.int32)

            for ci, (s, e) in enumerate(CHUNKS):
                is_last = (ci == NUM_CHUNKS - 1)
                hidden = model.model.process_layers_regular_single_token_export_local_state(
                    hidden_states=hidden, position_ids=position_ids,
                    causal_mask=mask, current_pos=current_pos,
                    kv_cache_0=None, k_cache=pt_k_caches[ci], v_cache=pt_v_caches[ci],
                    linear_conv_state=pt_conv_states[ci],
                    linear_recurrent_state=pt_rec_states[ci],
                    start_layer=s, end_layer=e,
                    apply_final_norm=is_last,
                )

            pt_hiddens.append(hidden.numpy().copy())
            logits = model.lm_head(hidden.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
            next_id = torch.argmax(logits, dim=-1).to(torch.int32)
            pt_tokens.append(next_id.item())

    pt_text = tokenizer.decode(pt_tokens)
    print("  Generated %d tokens" % len(pt_tokens))
    print("  PyTorch output: %s" % repr(pt_text[:200]))

    # Free PyTorch model
    del model, pt_k_caches, pt_v_caches, pt_conv_states, pt_rec_states
    gc.collect()

    # ── 4. CoreML greedy decode ──
    print("\n--- CoreML Greedy Decode (ANE) ---")

    # Load models
    print("  Loading CoreML models...")
    embed_model = ct.models.MLModel(
        os.path.join(EXPORT_DIR, "qwen35_embeddings.mlpackage"),
        compute_units=compute_unit)
    
    ffn_models = []
    ffn_states = []
    for ci in range(NUM_CHUNKS):
        path = os.path.join(EXPORT_DIR, "qwen35_FFN_chunk_%02dof%02d.mlpackage" % (ci + 1, NUM_CHUNKS))
        m = ct.models.MLModel(path, compute_units=compute_unit)
        ffn_models.append(m)
        ffn_states.append(m.make_state())

    lm_head = ct.models.MLModel(
        os.path.join(EXPORT_DIR, "qwen35_lm_head_lut6.mlpackage"),
        compute_units=compute_unit)
    print("  All models loaded.")

    cml_tokens = []
    cml_hiddens = []

    # Check lm_head output format
    lm_spec = lm_head.get_spec()
    lm_output_names = [o.name for o in lm_spec.description.output]
    has_argmax = "argmax_idx" in lm_output_names
    print("  LM head outputs: %s (argmax=%s)" % (lm_output_names, has_argmax))

    # Prefill: process prompt tokens one at a time through CoreML
    for pos in range(prompt_len):
        tok = input_ids[:, pos:pos+1].numpy().astype(np.int32)
        embed_out = embed_model.predict({"input_ids": tok})
        hidden = list(embed_out.values())[0]  # (1, 1, 2560)

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos+1] = 0
        pos_ids = np.array([pos], dtype=np.int32)
        cur_pos = np.array([pos], dtype=np.int32)

        for ci in range(NUM_CHUNKS):
            out = ffn_models[ci].predict({
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": cur_pos,
            }, state=ffn_states[ci])
            hidden = out["output_hidden_states"]

        if pos % 10 == 0:
            sys.stdout.write("\r  Prefill: %d/%d" % (pos + 1, prompt_len))
            sys.stdout.flush()

    # Get first generated token
    lm_out = lm_head.predict({"hidden_states": hidden.astype(np.float16)})
    if has_argmax:
        next_id_cml = int(lm_out["argmax_idx"].flatten()[0])
    else:
        logits_np = lm_out["logits"]
        next_id_cml = int(np.argmax(logits_np.flatten()))
    cml_tokens.append(next_id_cml)
    print("\r  Prefill done. First token: %d (%s)" % (next_id_cml, repr(tokenizer.decode([next_id_cml]))))

    # Generate remaining tokens
    for gi in range(max_gen - 1):
        pos = prompt_len + gi
        tok = np.array([[next_id_cml]], dtype=np.int32)
        embed_out = embed_model.predict({"input_ids": tok})
        hidden = list(embed_out.values())[0]

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos+1] = 0
        pos_ids = np.array([pos], dtype=np.int32)
        cur_pos = np.array([pos], dtype=np.int32)

        for ci in range(NUM_CHUNKS):
            out = ffn_models[ci].predict({
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": cur_pos,
            }, state=ffn_states[ci])
            hidden = out["output_hidden_states"]

        cml_hiddens.append(hidden.copy())

        lm_out = lm_head.predict({"hidden_states": hidden.astype(np.float16)})
        if has_argmax:
            next_id_cml = int(lm_out["argmax_idx"].flatten()[0])
        else:
            logits_np = lm_out["logits"]
            next_id_cml = int(np.argmax(logits_np.flatten()))
        cml_tokens.append(next_id_cml)

        if (gi + 1) % 10 == 0:
            sys.stdout.write("\r  Generating: %d/%d" % (gi + 2, max_gen))
            sys.stdout.flush()

    cml_text = tokenizer.decode(cml_tokens)

    # ── 5. Compare ──
    print("\n\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    # Token match
    matches = sum(1 for a, b in zip(pt_tokens, cml_tokens) if a == b)
    total = min(len(pt_tokens), len(cml_tokens))
    print("\nToken match: %d/%d (%.1f%%)" % (matches, total, 100.0 * matches / total if total > 0 else 0))

    # First divergence
    first_div = None
    for i, (a, b) in enumerate(zip(pt_tokens, cml_tokens)):
        if a != b:
            first_div = i
            break
    if first_div is not None:
        print("First divergence at token %d: PyTorch=%d (%s) vs CoreML=%d (%s)" % (
            first_div,
            pt_tokens[first_div], repr(tokenizer.decode([pt_tokens[first_div]])),
            cml_tokens[first_div], repr(tokenizer.decode([cml_tokens[first_div]]))))
    else:
        print("ALL TOKENS MATCH!")

    # Token-by-token comparison
    print("\nToken-by-token:")
    for i in range(min(total, 30)):  # Show first 30
        pt_tok = pt_tokens[i]
        cml_tok = cml_tokens[i]
        match = "==" if pt_tok == cml_tok else "!="
        print("  [%2d] PT=%6d (%s) %s CML=%6d (%s)" % (
            i, pt_tok, repr(tokenizer.decode([pt_tok]))[:15].ljust(15),
            match,
            cml_tok, repr(tokenizer.decode([cml_tok]))[:15].ljust(15)))

    # Hidden state cosine similarity (generation phase only)
    if pt_hiddens and cml_hiddens:
        cos_vals = []
        for i in range(min(len(pt_hiddens), len(cml_hiddens))):
            c = cosine(pt_hiddens[i], cml_hiddens[i])
            cos_vals.append(c)
        print("\nHidden state cosine similarity (decode phase):")
        print("  min=%.6f  max=%.6f  mean=%.6f" % (min(cos_vals), max(cos_vals),
              sum(cos_vals) / len(cos_vals)))

    print("\n--- PyTorch Output ---")
    print(pt_text[:500])
    print("\n--- CoreML Output ---")
    print(cml_text[:500])
    print("\n" + "=" * 70)

    # Cleanup
    del embed_model, lm_head
    for m in ffn_models:
        del m
    gc.collect()


if __name__ == "__main__":
    main()
