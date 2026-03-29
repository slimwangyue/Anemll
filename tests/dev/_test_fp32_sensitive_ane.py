#!/usr/bin/env python3
"""Test: does direct-RMS (ANE_SAFE_NUMERICS) improve accuracy on ANE?

This test exports a single FFN chunk with two configurations:
  A) FP16 baseline (doubled-LayerNorm for RMSNorm, all FP16) on CPU_AND_NE
  B) FP16 + ANE_SAFE_NUMERICS (direct RMS sum over 3072 vs layer_norm over 6144) on CPU_AND_NE

Usage:
    cd /Users/yw68/Anemll
    python tests/dev/_test_fp32_sensitive_ane.py
"""
import os, sys, time, gc, shutil, tempfile
import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts_qwen3_5"))

from config import BATCH_SIZE, CTX, NUM_CHUNKS, PER_CHANNEL, DEFAULT_HF_MODEL
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
import coremltools as ct

CHUNK_IDX = 0  # first chunk (has linear attention layers)


def load_model():
    print("Loading model weights...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(DEFAULT_HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(DEFAULT_HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model


def get_chunk_bounds(cfg):
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start = CHUNK_IDX * base + min(CHUNK_IDX, rem)
    end = start + base + (1 if CHUNK_IDX < rem else 0)
    return start, end


def run_pytorch_ref(model, hidden_states, position_ids, causal_mask, current_pos):
    """Run chunk through PyTorch in fp32 reference mode."""
    cfg = model.config
    start, end = get_chunk_bounds(cfg)
    local_num_layers = end - start

    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
    lin_conv = torch.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    lin_rec = torch.zeros(
        (local_num_layers, cfg.text_config.linear_num_value_heads,
         cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
        dtype=MODEL_DTYPE, device=TEST_DEVICE,
    )
    model.model.kv_cache_0.zero_()
    with torch.no_grad():
        out = model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden_states,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            kv_cache_0=None,
            k_cache=torch.zeros(
                (local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE,
            ),
            v_cache=torch.zeros(
                (local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE,
            ),
            linear_conv_state=lin_conv,
            linear_recurrent_state=lin_rec,
            start_layer=start,
            end_layer=end,
            apply_final_norm=False,
        )
        if end is None or end == len(model.model.layers):
            out = model.model.norm(out)
    return out


def export_and_predict(model, hidden_states, position_ids, causal_mask, current_pos,
                       ane_safe, label):
    """Export a CoreML chunk, save, reload on ANE, predict, then clean up."""
    cfg = model.config
    start, end = get_chunk_bounds(cfg)
    local_num_layers = end - start

    odir = f"/tmp/direct_rms_{label}"
    pkg_path = os.path.join(odir, "chunk.mlpackage")

    # Export
    print(f"\n[{label}] Exporting (ane_safe_numerics={ane_safe})...")
    t0 = time.time()
    os.makedirs(odir, exist_ok=True)
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=PER_CHANNEL,
        ane_safe_numerics=ane_safe,
    )
    mlmodel_obj = conv.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS)
    mlmodel_obj.save(pkg_path)
    del mlmodel_obj, conv
    gc.collect()
    print(f"  Exported in {time.time() - t0:.1f}s")

    # Load and predict on CPU_AND_NE
    print(f"  Loading on CPU_AND_NE...")
    loaded = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)

    inputs = {
        "hidden_states": hidden_states.numpy().astype(np.float16),
        "position_ids": position_ids.numpy().astype(np.int32),
        "causal_mask": causal_mask.numpy().astype(np.float16),
        "current_pos": current_pos.numpy().astype(np.int32),
        "linear_conv_state": np.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=np.float16),
        "linear_recurrent_state": np.zeros(
            (local_num_layers, cfg.text_config.linear_num_value_heads,
             cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
            dtype=np.float16,
        ),
    }

    state = loaded.make_state()
    result = loaded.predict(inputs, state)

    # Latency measurement
    latencies = []
    for _ in range(5):
        st = loaded.make_state()
        t1 = time.time()
        loaded.predict(inputs, st)
        latencies.append(time.time() - t1)
    lat_ms = min(latencies) * 1000

    out = torch.from_numpy(np.array(result["output_hidden_states"]))
    del loaded, state
    gc.collect()

    # Clean up disk immediately
    shutil.rmtree(odir, ignore_errors=True)
    print(f"  [cleaned] cosine calc next...")
    return out, lat_ms


def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return float(torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))


def max_abs_diff(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    model = load_model()
    cfg = model.config
    start, end = get_chunk_bounds(cfg)
    print(f"Chunk {CHUNK_IDX}: layers {start}..{end - 1}")

    layer_types = cfg.text_config.layer_types
    for i in range(start, end):
        print(f"  layer {i}: {layer_types[i]}")

    # Test input
    torch.manual_seed(42)
    hidden_states = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float16, device=TEST_DEVICE) * 0.1
    position_ids = torch.tensor([5], dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.tensor([5], dtype=torch.int32, device=TEST_DEVICE)

    # PyTorch reference
    print("\nPyTorch reference (fp32 math)...")
    ref_out = run_pytorch_ref(model, hidden_states, position_ids, causal_mask, current_pos)
    print(f"  Ref: range=[{ref_out.min():.4f}, {ref_out.max():.4f}]")

    results = {}

    # A: FP16 baseline on ANE (doubled LayerNorm)
    out_a, lat_a = export_and_predict(
        model, hidden_states, position_ids, causal_mask, current_pos,
        ane_safe=False, label="A",
    )
    cos_a = cosine_sim(ref_out, out_a)
    mad_a = max_abs_diff(ref_out, out_a)
    results["A: FP16 baseline (ANE)"] = (cos_a, mad_a, lat_a)
    print(f"  cosine={cos_a:.6f}, mad={mad_a:.6f}, latency={lat_a:.1f}ms")

    # B: FP16 + direct RMS on ANE (sum over 3072 instead of layer_norm over 6144)
    out_b, lat_b = export_and_predict(
        model, hidden_states, position_ids, causal_mask, current_pos,
        ane_safe=True, label="B",
    )
    cos_b = cosine_sim(ref_out, out_b)
    mad_b = max_abs_diff(ref_out, out_b)
    results["B: FP16+directRMS (ANE)"] = (cos_b, mad_b, lat_b)
    print(f"  cosine={cos_b:.6f}, mad={mad_b:.6f}, latency={lat_b:.1f}ms")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  DIRECT-RMS TEST — chunk {CHUNK_IDX} (layers {start}..{end - 1})")
    print(f"{'=' * 70}")
    print(f"  {'Config':<28s} {'Cosine':>10s} {'MAD':>10s} {'Lat(ms)':>10s}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10}")
    for name, (cos, mad, lat) in results.items():
        print(f"  {name:<28s} {cos:>10.6f} {mad:>10.6f} {lat:>10.1f}")
    print(f"{'=' * 70}")
    delta = cos_b - cos_a
    print(f"  Cosine improvement: {delta:+.6f}")
    if delta > 0.01:
        print(f"  >>> Direct RMS SIGNIFICANTLY HELPS on ANE! <<<")
    elif delta > 0.001:
        print(f"  >>> Direct RMS helps on ANE <<<")
    elif delta < -0.001:
        print(f"  >>> Direct RMS HURTS on ANE! <<<")
    else:
        print(f"  >>> Direct RMS has NEGLIGIBLE effect on ANE <<<")


if __name__ == "__main__":
    main()
