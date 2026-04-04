#!/usr/bin/env python3
"""Correctness test for LUT6 4-chunk bs512 ctx2048 export.

Compares CoreML prefill chunk outputs vs PyTorch reference at different
input lengths. For seq_len < 512, pads to 512 with zeros and uses
correct causal mask for the valid positions.

Usage:
    python3 tests/dev/test_lut6_bs512_correctness.py \
        --model-dir /Users/yw68/Anemll/qwen3_5_4chunk_lut6_bs512_ctx2048 \
        --hf-model /Users/yw68/Anemll/models/Qwen__Qwen3.5-4B
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35Config,
    Qwen35ForCausalLM,
    MODEL_DTYPE,
    ane_conv_state_shape,
)

# ── Constants ──
BUCKET_SIZE = 512
CTX = 2048
NUM_CHUNKS = 4
FFN_LABEL = "LUT6"
TEST_LENGTHS = [64, 128, 256, 384, 512]


def load_pytorch_model(hf_model_path: str):
    cfg = Qwen35Config.from_json(os.path.join(hf_model_path, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(hf_model_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


def build_causal_mask(q_len, cache_len, current_pos):
    """Build causal mask [1, 1, q_len, cache_len]."""
    q_idx = torch.arange(q_len).unsqueeze(-1)
    k_idx = torch.arange(cache_len).unsqueeze(0)
    allowed = k_idx <= (current_pos + q_idx)
    zeros = torch.zeros((q_len, cache_len), dtype=MODEL_DTYPE)
    neg_inf = torch.full((q_len, cache_len), -65504.0, dtype=MODEL_DTYPE)
    mask = torch.where(allowed, zeros, neg_inf)
    return mask.unsqueeze(0).unsqueeze(0)


def get_chunk_layers(total_layers, num_chunks, chunk_idx):
    base, rem = divmod(total_layers, num_chunks)
    start = chunk_idx * base + min(chunk_idx, rem)
    end = start + base + (1 if chunk_idx < rem else 0)
    return start, end


def pytorch_chunk_prefill(model, cfg, hidden_states, chunk_idx, seq_len):
    """Run PyTorch prefill through one chunk's layers, returning hidden state."""
    total_layers = cfg.num_hidden_layers
    start_layer, end_layer = get_chunk_layers(total_layers, NUM_CHUNKS, chunk_idx)
    local_num_layers = end_layer - start_layer
    is_final = (end_layer >= total_layers)

    # Pad to BUCKET_SIZE for consistency with exported model
    padded_len = BUCKET_SIZE
    if seq_len < padded_len:
        pad = torch.zeros(1, padded_len - seq_len, cfg.hidden_size, dtype=MODEL_DTYPE)
        hs_padded = torch.cat([hidden_states[:, :seq_len, :], pad], dim=1)
    else:
        hs_padded = hidden_states[:, :seq_len, :]

    position_ids = torch.zeros(padded_len, dtype=torch.long)
    position_ids[:seq_len] = torch.arange(seq_len, dtype=torch.long)
    causal_mask = build_causal_mask(padded_len, CTX, 0)
    # Mask padding query rows to -inf
    if seq_len < padded_len:
        causal_mask[:, :, seq_len:, :] = -65504.0

    valid_len = torch.tensor([seq_len], dtype=torch.int32)

    # Initialize chunk-local states (ANE-safe shapes for conv state)
    conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)

    k_cache = torch.zeros(local_num_layers, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    v_cache = torch.zeros(local_num_layers, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    lin_conv = torch.zeros(local_num_layers, ane_dim1, ane_dim2, dtype=MODEL_DTYPE)
    lin_rec = torch.zeros(
        local_num_layers,
        cfg.text_config.linear_num_value_heads,
        cfg.text_config.linear_key_head_dim,
        cfg.text_config.linear_value_head_dim,
        dtype=MODEL_DTYPE,
    )

    # Set up linear attention export settings
    for layer_idx in range(start_layer, end_layer):
        layer = model.model.layers[layer_idx]
        if getattr(layer, 'layer_type', None) == 'linear_attention':
            layer.self_attn.export_expected_batch_size = 1
            layer.self_attn.export_expected_seq_len = padded_len

    out = model.model.process_layers_prefill_export_local_state(
        hidden_states=hs_padded,
        position_ids=position_ids,
        causal_mask=causal_mask,
        current_pos=torch.tensor([0], dtype=torch.long),
        kv_cache_0=None,
        k_cache=k_cache,
        v_cache=v_cache,
        linear_conv_state=lin_conv,
        linear_recurrent_state=lin_rec,
        start_layer=start_layer,
        end_layer=end_layer,
        apply_final_norm=False,
        expected_batch_size=1,
        expected_seq_len=padded_len,
        valid_len=valid_len,
    )

    if is_final:
        out = out[:, seq_len - 1 : seq_len, :]
    else:
        out = out[:, :seq_len, :]
    return out


def coreml_chunk_prefill(mlmodel, cfg, hidden_states_np, seq_len, chunk_idx):
    """Run CoreML prefill through one chunk model."""
    total_layers = cfg.num_hidden_layers
    start_layer, end_layer = get_chunk_layers(total_layers, NUM_CHUNKS, chunk_idx)
    local_num_layers = end_layer - start_layer

    padded_len = BUCKET_SIZE

    # Pad hidden states
    if seq_len < padded_len:
        pad = np.zeros((1, padded_len - seq_len, hidden_states_np.shape[2]), dtype=np.float16)
        hs_padded = np.concatenate([hidden_states_np[:, :seq_len, :], pad], axis=1)
    else:
        hs_padded = hidden_states_np[:, :seq_len, :]

    # Position IDs: real for valid, 0 for padding
    pos_ids = np.zeros(padded_len, dtype=np.int32)
    pos_ids[:seq_len] = np.arange(seq_len, dtype=np.int32)

    # Causal mask
    mask = np.full((1, 1, padded_len, CTX), -65504.0, dtype=np.float16)
    for qi in range(seq_len):
        mask[0, 0, qi, :qi + 1] = 0.0

    # Linear states
    conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)

    lin_conv = np.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=np.float16)
    lin_rec = np.zeros((
        local_num_layers,
        cfg.text_config.linear_num_value_heads,
        cfg.text_config.linear_key_head_dim,
        cfg.text_config.linear_value_head_dim,
    ), dtype=np.float16)

    # k_cache and v_cache are CoreML state — use make_state()
    state = mlmodel.make_state()

    pred = mlmodel.predict(
        {
            "hidden_states": hs_padded.astype(np.float16),
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": np.array([0], dtype=np.int32),
            "linear_conv_state": lin_conv,
            "linear_recurrent_state": lin_rec,
            "valid_len": np.array([seq_len], dtype=np.int32),
        },
        state=state,
    )

    return pred["output_hidden_states"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--test-lengths", type=str, default=None)
    parser.add_argument("--chunks-to-test", type=str, default="0,1,2,3",
                        help="Comma-separated chunk indices to test (default: all)")
    args = parser.parse_args()

    test_lengths = TEST_LENGTHS
    if args.test_lengths:
        test_lengths = [int(x) for x in args.test_lengths.split(",")]
    chunks_to_test = [int(x) for x in args.chunks_to_test.split(",")]

    print("=" * 70)
    print("  LUT6 4-chunk bs512 ctx2048 — Correctness Test")
    print(f"  Model dir: {args.model_dir}")
    print(f"  HF model:  {args.hf_model}")
    print(f"  Bucket:    {BUCKET_SIZE}")
    print(f"  CTX:       {CTX}")
    print(f"  Test lengths: {test_lengths}")
    print(f"  Chunks:    {chunks_to_test}")
    print("=" * 70)

    # Load PyTorch model
    print("\nLoading PyTorch model...")
    t0 = time.time()
    pt_model, cfg = load_pytorch_model(args.hf_model)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load CoreML prefill models
    print("\nLoading CoreML prefill models...")
    coreml_models = {}
    for ci in chunks_to_test:
        path = os.path.join(args.model_dir, f"prefill_{FFN_LABEL}_chunk{ci}_bs{BUCKET_SIZE}.mlpackage")
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found")
            sys.exit(1)
        print(f"  Loading chunk {ci}...")
        coreml_models[ci] = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

    results = []

    for ci in chunks_to_test:
        print(f"\n{'=' * 70}")
        print(f"  CHUNK {ci}")
        print(f"{'=' * 70}")

        for seq_len in test_lengths:
            print(f"\n  seq_len={seq_len} (pad={BUCKET_SIZE - seq_len})")

            # Generate random hidden states (simulating embeddings output)
            torch.manual_seed(42 + seq_len)
            hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=MODEL_DTYPE) * 0.1

            # PyTorch reference
            with torch.no_grad():
                pt_out = pytorch_chunk_prefill(pt_model, cfg, hidden, ci, seq_len)
            pt_np = pt_out.float().numpy()

            # CoreML
            hidden_np = hidden.numpy()
            t0 = time.time()
            cml_out = coreml_chunk_prefill(coreml_models[ci], cfg, hidden_np, seq_len, ci)
            cml_time = time.time() - t0
            cml_np = np.array(cml_out, dtype=np.float32)

            # For non-final chunks, CoreML returns full padded output.
            # Compare only the valid positions.
            is_final = (ci == NUM_CHUNKS - 1)
            if not is_final:
                # CoreML: (1, 512, hidden_size) — take first seq_len positions
                cml_np = cml_np[:, :seq_len, :]

            # Compare
            cos = float(F.cosine_similarity(
                torch.from_numpy(cml_np).flatten().float(),
                torch.from_numpy(pt_np).flatten().float(),
                dim=0,
            ).item())
            max_abs = float(np.abs(cml_np - pt_np).max())
            mean_abs = float(np.abs(cml_np - pt_np).mean())

            results.append({
                "chunk": ci,
                "seq_len": seq_len,
                "cosine": cos,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "time_s": cml_time,
            })

            print(f"    cosine:   {cos:.6f}")
            print(f"    max_abs:  {max_abs:.4e}")
            print(f"    mean_abs: {mean_abs:.4e}")
            print(f"    time:     {cml_time:.3f}s")

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'chunk':>5s}  {'seq_len':>8s}  {'cosine':>10s}  {'max_abs':>12s}  {'mean_abs':>12s}  {'time':>8s}")
    print(f"  {'─' * 5}  {'─' * 8}  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 8}")
    for r in results:
        print(f"  {r['chunk']:>5d}  {r['seq_len']:>8d}  {r['cosine']:>10.6f}  {r['max_abs']:>12.4e}  {r['mean_abs']:>12.4e}  {r['time_s']:>8.3f}")

    all_good = all(r["cosine"] > 0.85 for r in results)
    print(f"\n  Overall: {'PASS' if all_good else 'FAIL'} (threshold: cosine > 0.85)")
    print(f"{'=' * 70}")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
