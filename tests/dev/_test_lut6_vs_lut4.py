#!/usr/bin/env python3
"""Compare LUT4 vs LUT6 vs FP16 accuracy on ANE (CPU_AND_NE).

Also tests each with the direct-RMS (ANE_SAFE_NUMERICS) variant.

Usage:
    cd /Users/yw68/Anemll
    python tests/dev/_test_lut6_vs_lut4.py
"""
import os, sys, time, gc, shutil
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

CHUNK_IDX = 0


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
            hidden_states=hidden_states, position_ids=position_ids,
            causal_mask=causal_mask, current_pos=current_pos, kv_cache_0=None,
            k_cache=torch.zeros((local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            v_cache=torch.zeros((local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            linear_conv_state=lin_conv, linear_recurrent_state=lin_rec,
            start_layer=start, end_layer=end, apply_final_norm=False,
        )
        if end is None or end == len(model.model.layers):
            out = model.model.norm(out)
    return out


def export_and_predict(model, inputs_np, lut_bits, ane_safe, label):
    cfg = model.config
    start, end = get_chunk_bounds(cfg)
    local_num_layers = end - start
    odir = f"/tmp/lut_test_{label}"
    pkg_path = os.path.join(odir, "chunk.mlpackage")

    tag = f"LUT{lut_bits}" if lut_bits else "FP16"
    safe_tag = "+directRMS" if ane_safe else ""
    print(f"\n[{label}] {tag}{safe_tag}: Exporting...")
    t0 = time.time()
    os.makedirs(odir, exist_ok=True)
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=lut_bits, per_channel=PER_CHANNEL,
        ane_safe_numerics=ane_safe,
    )
    mlmodel_obj = conv.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS)
    mlmodel_obj.save(pkg_path)
    del mlmodel_obj, conv
    gc.collect()
    print(f"  Exported in {time.time() - t0:.1f}s")

    print(f"  Loading on CPU_AND_NE...")
    loaded = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = loaded.make_state()
    result = loaded.predict(inputs_np, state)

    latencies = []
    for _ in range(5):
        st = loaded.make_state()
        t1 = time.time()
        loaded.predict(inputs_np, st)
        latencies.append(time.time() - t1)
    lat_ms = min(latencies) * 1000

    out = torch.from_numpy(np.array(result["output_hidden_states"]))
    del loaded, state
    gc.collect()
    shutil.rmtree(odir, ignore_errors=True)
    print(f"  [cleaned]")
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
    local_num_layers = end - start
    print(f"Chunk {CHUNK_IDX}: layers {start}..{end - 1}")

    # Build numpy inputs once
    torch.manual_seed(42)
    hidden_states = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float16, device=TEST_DEVICE) * 0.1
    position_ids = torch.tensor([5], dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.tensor([5], dtype=torch.int32, device=TEST_DEVICE)

    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)

    inputs_np = {
        "hidden_states": hidden_states.numpy().astype(np.float16),
        "position_ids": position_ids.numpy().astype(np.int32),
        "causal_mask": causal_mask.numpy().astype(np.float16),
        "current_pos": current_pos.numpy().astype(np.int32),
        "linear_conv_state": np.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=np.float16),
        "linear_recurrent_state": np.zeros(
            (local_num_layers, cfg.text_config.linear_num_value_heads,
             cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
            dtype=np.float16),
    }

    # PyTorch reference
    print("\nPyTorch reference...")
    ref_out = run_pytorch_ref(model, hidden_states, position_ids, causal_mask, current_pos)
    print(f"  Ref: range=[{ref_out.min():.4f}, {ref_out.max():.4f}]")

    configs = [
        # (lut_bits, ane_safe, label)
        (4,    False, "A"),  # LUT4 baseline
        (4,    True,  "B"),  # LUT4 + direct RMS
        (6,    False, "C"),  # LUT6 baseline
        (6,    True,  "D"),  # LUT6 + direct RMS
        (None, False, "E"),  # FP16 baseline (no quant)
        (None, True,  "F"),  # FP16 + direct RMS
    ]

    results = {}
    for lut_bits, ane_safe, label in configs:
        tag = f"LUT{lut_bits}" if lut_bits else "FP16"
        safe_tag = "+dRMS" if ane_safe else ""
        name = f"{label}: {tag}{safe_tag}"
        try:
            out, lat = export_and_predict(model, inputs_np, lut_bits, ane_safe, label)
            cos = cosine_sim(ref_out, out)
            mad = max_abs_diff(ref_out, out)
            results[name] = (cos, mad, lat)
            print(f"  cosine={cos:.6f}, mad={mad:.6f}, latency={lat:.1f}ms")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = None

    # Summary
    print(f"\n{'=' * 76}")
    print(f"  LUT4 vs LUT6 vs FP16 — chunk {CHUNK_IDX} (layers {start}..{end - 1})")
    print(f"{'=' * 76}")
    print(f"  {'Config':<24s} {'Cosine':>10s} {'MAD':>10s} {'Lat(ms)':>10s}")
    print(f"  {'-' * 24} {'-' * 10} {'-' * 10} {'-' * 10}")
    for name, r in results.items():
        if r:
            cos, mad, lat = r
            print(f"  {name:<24s} {cos:>10.6f} {mad:>10.6f} {lat:>10.1f}")
        else:
            print(f"  {name:<24s} {'FAILED':>10s}")
    print(f"{'=' * 76}")

    # Deltas
    base = results.get("A: LUT4")
    if base:
        base_cos = base[0]
        for name, r in results.items():
            if r and name != "A: LUT4":
                print(f"  {name} vs LUT4: cosine delta {r[0] - base_cos:+.6f}")


if __name__ == "__main__":
    main()
