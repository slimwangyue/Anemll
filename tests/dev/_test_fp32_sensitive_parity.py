#!/usr/bin/env python3
"""Quick accuracy test: export one FFN chunk with different quantization configs,
then compare CoreML vs PyTorch hidden-state parity on a real input.

Configs tested:
  A) LUT4 (current baseline — all weights quantized)
  B) FP16 (no quantization — pure fp16 weights)
  C) Selective: LUT4 for MLP/full-attn, FP16 for linear-attn projections
  D) FP32-sensitive ops + selective LUT (best-of-both-worlds)

Usage:
    cd /Users/yw68/Anemll
    python tests/dev/_test_fp32_sensitive_parity.py
"""
import os, sys, time, gc, json, re
import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts_qwen3_5"))

from config import BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, PER_CHANNEL, DEFAULT_HF_MODEL
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

import coremltools as ct

CHUNK_IDX = 0          # test first chunk (has linear attention layers)
OUT_BASE = "/tmp/qwen35_parity"

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
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model

def export_chunk(model, out_dir, lut_bits, fp32_sensitive=False):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"lut{lut_bits}" if lut_bits else "fp16"
    dec_path = os.path.join(out_dir, f"ffn_{tag}_chunk{CHUNK_IDX}.mlpackage")
    if os.path.exists(dec_path):
        print(f"  [cached] {dec_path}")
        return dec_path
    print(f"  Exporting chunk {CHUNK_IDX} (lut_bits={lut_bits}, fp32_sensitive={fp32_sensitive})...")
    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=lut_bits, per_channel=PER_CHANNEL,
        fp32_sensitive_ops=fp32_sensitive,
    )
    ml = conv.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS)
    ml.save(dec_path)
    del ml, conv; gc.collect()
    print(f"  Saved ({time.time()-t0:.1f}s)")
    return dec_path

def run_pytorch_chunk(model, hidden_states, position_ids, causal_mask, current_pos):
    """Run chunk through PyTorch (reference in fp32)."""
    cfg = model.config
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start = CHUNK_IDX * base + min(CHUNK_IDX, rem)
    end = start + base + (1 if CHUNK_IDX < rem else 0)
    local_num_layers = end - start

    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    from anemll.models.qwen3_5_model import ane_conv_state_shape
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
    lin_conv = torch.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    lin_rec = torch.zeros(
        (local_num_layers, cfg.text_config.linear_num_value_heads,
         cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
        dtype=MODEL_DTYPE, device=TEST_DEVICE
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
                dtype=MODEL_DTYPE, device=TEST_DEVICE
            ),
            v_cache=torch.zeros(
                (local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE
            ),
            linear_conv_state=lin_conv,
            linear_recurrent_state=lin_rec,
            start_layer=start,
            end_layer=end,
            apply_final_norm=False,
        )
        if end is None or end == total_layers:
            out = model.model.norm(out)
    return out

def run_coreml_chunk(mlpackage_path, hidden_states, position_ids, causal_mask, current_pos, model_cfg,
                     compute_units=ct.ComputeUnit.CPU_AND_NE):
    """Run chunk through CoreML."""
    cfg = model_cfg
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start = CHUNK_IDX * base + min(CHUNK_IDX, rem)
    end = start + base + (1 if CHUNK_IDX < rem else 0)
    local_num_layers = end - start

    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    from anemll.models.qwen3_5_model import ane_conv_state_shape
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)

    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=compute_units)
    state = mlmodel.make_state()

    lin_conv_np = np.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=np.float16)
    lin_rec_np = np.zeros(
        (local_num_layers, cfg.text_config.linear_num_value_heads,
         cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim),
        dtype=np.float16
    )

    inputs = {
        "hidden_states": hidden_states.numpy().astype(np.float16),
        "position_ids": position_ids.numpy().astype(np.int32),
        "causal_mask": causal_mask.numpy().astype(np.float16),
        "current_pos": current_pos.numpy().astype(np.int32),
        "linear_conv_state": lin_conv_np,
        "linear_recurrent_state": lin_rec_np,
    }
    result = mlmodel.predict(inputs, state)
    out = result["output_hidden_states"]
    return torch.from_numpy(np.array(out))

def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return float(torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))

def max_abs_diff(a, b):
    return float((a.float() - b.float()).abs().max())

def main():
    model = load_model()
    cfg = model.config

    # Print which layers are linear attention
    layer_types = cfg.text_config.layer_types
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start = CHUNK_IDX * base + min(CHUNK_IDX, rem)
    end = start + base + (1 if CHUNK_IDX < rem else 0)
    print(f"\nChunk {CHUNK_IDX}: layers {start}..{end-1}")
    for i in range(start, end):
        print(f"  layer {i}: {layer_types[i]}")

    # Create test input
    torch.manual_seed(42)
    hidden_states = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float16, device=TEST_DEVICE) * 0.1
    position_ids = torch.tensor([5], dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    causal_mask[:, :, :, 6:] = float("-inf")
    current_pos = torch.tensor([5], dtype=torch.int32, device=TEST_DEVICE)

    # PyTorch reference
    print("\n[1] PyTorch reference (fp32 recurrence math)...")
    ref_out = run_pytorch_chunk(model, hidden_states, position_ids, causal_mask, current_pos)
    print(f"  ref_out shape: {ref_out.shape}, range: [{ref_out.min():.4f}, {ref_out.max():.4f}]")

    results = {}

    # Config A: LUT4 (current baseline)
    print("\n[A] LUT4 baseline (current)...")
    path_a = export_chunk(model, f"{OUT_BASE}_lutA", lut_bits=4, fp32_sensitive=False)
    out_a = run_coreml_chunk(path_a, hidden_states, position_ids, causal_mask, current_pos, cfg)
    results['LUT4'] = (cosine_sim(ref_out, out_a), max_abs_diff(ref_out, out_a))
    print(f"  cosine={results['LUT4'][0]:.6f}, max_abs_diff={results['LUT4'][1]:.6f}")

    # Config B: No LUT (pure FP16 weights)
    print("\n[B] FP16 weights (no quantization)...")
    path_b = export_chunk(model, f"{OUT_BASE}_fp16B", lut_bits=None, fp32_sensitive=False)
    out_b = run_coreml_chunk(path_b, hidden_states, position_ids, causal_mask, current_pos, cfg)
    results['FP16'] = (cosine_sim(ref_out, out_b), max_abs_diff(ref_out, out_b))
    print(f"  cosine={results['FP16'][0]:.6f}, max_abs_diff={results['FP16'][1]:.6f}")

    # Config C: FP32-sensitive ops + no LUT
    print("\n[C] FP32-sensitive ops + FP16 weights...")
    path_c = export_chunk(model, f"{OUT_BASE}_fp32C", lut_bits=None, fp32_sensitive=True)
    out_c = run_coreml_chunk(path_c, hidden_states, position_ids, causal_mask, current_pos, cfg)
    results['FP32s+FP16'] = (cosine_sim(ref_out, out_c), max_abs_diff(ref_out, out_c))
    print(f"  cosine={results['FP32s+FP16'][0]:.6f}, max_abs_diff={results['FP32s+FP16'][1]:.6f}")

    # Config D: Selective LUT4 (MLP/full-attn quantized, linear-attn FP16)
    print("\n[D] Selective LUT4 (linear-attn weights stay FP16)...")
    path_d = export_chunk(model, f"{OUT_BASE}_selD", lut_bits=4, fp32_sensitive=True)
    out_d = run_coreml_chunk(path_d, hidden_states, position_ids, causal_mask, current_pos, cfg)
    results['Selective LUT4'] = (cosine_sim(ref_out, out_d), max_abs_diff(ref_out, out_d))
    print(f"  cosine={results['Selective LUT4'][0]:.6f}, max_abs_diff={results['Selective LUT4'][1]:.6f}")

    # Config E: FP16 weights on CPU_AND_GPU (no ANE precision loss)
    print("\n[E] FP16 weights on CPU_AND_GPU...")
    out_e = run_coreml_chunk(path_b, hidden_states, position_ids, causal_mask, current_pos, cfg,
                             compute_units=ct.ComputeUnit.CPU_AND_GPU)
    results['FP16 CPU+GPU'] = (cosine_sim(ref_out, out_e), max_abs_diff(ref_out, out_e))
    print(f"  cosine={results['FP16 CPU+GPU'][0]:.6f}, max_abs_diff={results['FP16 CPU+GPU'][1]:.6f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  PARITY COMPARISON — chunk {CHUNK_IDX} (layers {start}..{end-1})")
    print(f"{'='*70}")
    print(f"  {'Config':<25s} {'Cosine':>12s} {'MaxAbsDiff':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for name, (cos, mad) in results.items():
        print(f"  {name:<25s} {cos:>12.6f} {mad:>12.6f}")
    print(f"{'='*70}")

    # Improvement summary
    baseline_cos = results['LUT4'][0]
    for name, (cos, mad) in results.items():
        if name != 'LUT4':
            delta = cos - baseline_cos
            print(f"  {name} vs LUT4: cosine delta {delta:+.6f}")

if __name__ == "__main__":
    main()
