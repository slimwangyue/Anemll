#!/usr/bin/env python3
"""End-to-end single token test for Gemma4 E4B CoreML models.

Loads all components and generates one token from a prompt to verify
the full pipeline works.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import coremltools as ct


MODEL_DIR = "gemma4_E4B_lut4ffn_lut6em"
HF_MODEL = os.path.expanduser("~/local_llm/models/google__gemma-4-E4B-it")
NUM_CHUNKS = 7
CTX = 512
COMPUTE = ct.ComputeUnit.CPU_ONLY  # Decode chunks fail on ANE currently


def main():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(HF_MODEL, "tokenizer.json"))

    prompt = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
    encoded = tokenizer.encode(prompt)
    token_ids = [2] + encoded.ids  # BOS
    print(f"Prompt: {prompt[:50]}...")
    print(f"Token IDs ({len(token_ids)}): {token_ids[:10]}...")

    # Load embeddings (on ANE)
    print("\nLoading embeddings...")
    embed_model = ct.models.MLModel(
        os.path.join(MODEL_DIR, "embeddings.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    # Load LM head (on ANE)
    print("Loading LM head...")
    lm_head_model = ct.models.MLModel(
        os.path.join(MODEL_DIR, "lm_head_lut6.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    # Load decode chunks (CPU_ONLY due to ANE compile error)
    print(f"Loading {NUM_CHUNKS} decode chunks...")
    decode_models = []
    decode_states = []
    for ci in range(NUM_CHUNKS):
        path = os.path.join(MODEL_DIR, f"decode_LUT4_chunk{ci:02d}.mlpackage")
        m = ct.models.MLModel(path, compute_units=COMPUTE)
        decode_models.append(m)
        decode_states.append(m.make_state())

    print("\n" + "=" * 40)
    print("SINGLE TOKEN GENERATION TEST")
    print("=" * 40)

    # Process first token through embeddings
    input_ids = np.array([[token_ids[0]]], dtype=np.int32)
    embed_result = embed_model.predict({"input_ids": input_ids})
    hidden_states = embed_result["hidden_states"]
    per_layer_emb = embed_result["per_layer_emb"]
    print(f"Embeddings: hidden={hidden_states.shape}, ple={per_layer_emb.shape}")

    # Run through all decode chunks
    causal_mask = np.zeros((1, 1, 1, CTX), dtype=np.float16)
    causal_mask[0, 0, 0, 0] = 1.0  # Allow attending to position 0
    position_ids = np.array([0], dtype=np.int32)
    current_pos = np.array([0], dtype=np.int32)

    for ci in range(NUM_CHUNKS):
        result = decode_models[ci].predict(
            {
                "hidden_states": hidden_states.astype(np.float16),
                "position_ids": position_ids,
                "causal_mask": causal_mask,
                "current_pos": current_pos,
                "per_layer_emb": per_layer_emb.astype(np.float16),
            },
            state=decode_states[ci],
        )
        hidden_states = result["output_hidden_states"]
        print(f"  Chunk {ci}: output range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

    # LM head
    logit_chunks = lm_head_model.predict({"hidden_states": hidden_states.astype(np.float16)})
    # Concatenate logit chunks
    logits_list = []
    for i in range(1, 17):
        key = f"logits{i}"
        if key in logit_chunks:
            logits_list.append(logit_chunks[key])
    logits = np.concatenate(logits_list, axis=-1)  # [1, 1, 262144]

    # Apply softcapping
    logits = np.tanh(logits / 30.0) * 30.0

    # Get top token
    token_id = int(np.argmax(logits[0, 0]))
    decoded = tokenizer.decode([token_id])
    print(f"\nLogits shape: {logits.shape}")
    print(f"Top token: {token_id} -> '{decoded}'")

    # Top 5
    top5 = np.argsort(logits[0, 0])[-5:][::-1]
    print("Top 5:")
    for t in top5:
        prob = np.exp(logits[0, 0, t]) / np.exp(logits[0, 0]).sum()
        decoded = tokenizer.decode([int(t)])
        print(f"  {t}: '{decoded}' (logit={logits[0, 0, t]:.3f})")


if __name__ == "__main__":
    main()
