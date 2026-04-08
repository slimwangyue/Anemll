#!/usr/bin/env python3
"""Ground-truth diagnostic: PyTorch vs CoreML CPU vs CoreML ANE for chunk0.

Uses the EXACT FFNWrapper that CoreML traces, so PyTorch output is the
true ground truth that the CoreML model should reproduce.
"""
import os, sys, time
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct
from config import CTX, NUM_CHUNKS
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter, ane_conv_state_shape

MODEL_DIR = "/Users/yw68/Anemll/qwen3_5_blockrecur_full"
HF_PATH = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"


def compare(a, b, label):
    a64 = np.asarray(a).flatten().astype(np.float64)
    b64 = np.asarray(b).flatten().astype(np.float64)
    cos = np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30)
    mad = np.abs(a64 - b64).max()
    mean_ad = np.abs(a64 - b64).mean()
    print(f"  {label}: cos={cos:.8f}  max_diff={mad:.6f}  mean_diff={mean_ad:.8f}")
    return cos


def main():
    print("=" * 80)
    print("  PyTorch vs CoreML CPU vs CoreML ANE — Chunk0 Ground Truth")
    print("=" * 80)

    # --- 1. Load PyTorch model ---
    print("\n[1] Loading PyTorch model...")
    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval().half()
    for p in model.parameters():
        p.requires_grad = False

    # --- 2. Build FFNWrapper for chunk0 (same as converter) ---
    print("\n[2] Building FFNWrapper for chunk0...")
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start_layer = 0
    end_layer = base + (1 if 0 < rem else 0)
    local_num_layers = end_layer - start_layer
    print(f"  Chunk0: layers {start_layer}-{end_layer-1} ({local_num_layers} layers)")

    # Get linear state shapes
    tcfg = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    conv_dim = tcfg.linear_num_key_heads * tcfg.linear_key_head_dim * 2 + tcfg.linear_num_value_heads * tcfg.linear_value_head_dim
    conv_kernel = max(1, int(tcfg.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
    lin_conv_shape = (local_num_layers, ane_dim1, ane_dim2)
    lin_rec_shape = (local_num_layers, tcfg.linear_num_value_heads, tcfg.linear_key_head_dim, tcfg.linear_value_head_dim)
    print(f"  conv_shape={lin_conv_shape}, rec_shape={lin_rec_shape}")

    # --- 3. Create input tensors ---
    print("\n[3] Creating test inputs...")
    tok_id = 9906  # "Hello"
    with torch.no_grad():
        tok_t = torch.tensor([[tok_id]], dtype=torch.long)
        pt_embed = model.model.embed_tokens(tok_t).half()
    
    hidden_states = pt_embed  # shape: (1, 1, 2560)
    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=torch.float16)
    causal_mask[:, :, :, :1] = 0
    current_pos = torch.zeros((1,), dtype=torch.int32)
    lin_conv = torch.zeros(lin_conv_shape, dtype=torch.float16)
    lin_rec = torch.zeros(lin_rec_shape, dtype=torch.float16)

    print(f"  hidden norm: {hidden_states.norm().item():.4f}")

    # --- 4. Run PyTorch (ground truth) ---
    print("\n[4] PyTorch forward (ground truth)...")
    with torch.no_grad():
        # Use separate tensors so in-place state writes are properly captured
        pt_lin_conv = lin_conv.clone()
        pt_lin_rec = lin_rec.clone()
        pt_out = model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden_states.clone(),
            position_ids=position_ids.clone(),
            causal_mask=causal_mask.clone(),
            current_pos=current_pos.clone(),
            kv_cache_0=None,
            k_cache=torch.zeros(local_num_layers, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16),
            v_cache=torch.zeros(local_num_layers, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16),
            linear_conv_state=pt_lin_conv,
            linear_recurrent_state=pt_lin_rec,
            start_layer=start_layer,
            end_layer=end_layer,
            apply_final_norm=False,
        )
        pt_hidden = pt_out.numpy()
        pt_conv = pt_lin_conv.numpy()
        pt_rec = pt_lin_rec.numpy()
    print(f"  PyTorch hidden out norm: {np.linalg.norm(pt_hidden):.4f}")
    print(f"  PyTorch conv_state norm: {np.linalg.norm(pt_conv):.4f}")
    print(f"  PyTorch rec_state norm: {np.linalg.norm(pt_rec):.4f}")

    # --- 5. CoreML models ---
    print("\n[5] Loading CoreML chunk0...")
    chunk0_lut6_path = os.path.join(MODEL_DIR, "ffn_LUT6_chunk0.mlpackage")
    chunk0_fp16_path = "/Users/yw68/Anemll/qwen3_5_fp16_models/ffn_FP16_chunk0.mlpackage"
    
    print("  LUT6 CPU...", end="", flush=True)
    t0 = time.time()
    chunk0_lut6_cpu = ct.models.MLModel(chunk0_lut6_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f" {time.time()-t0:.1f}s")
    
    print("  LUT6 ANE...", end="", flush=True)
    t0 = time.time()
    chunk0_lut6_ane = ct.models.MLModel(chunk0_lut6_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f" {time.time()-t0:.1f}s")

    print("  FP16 CPU...", end="", flush=True)
    t0 = time.time()
    chunk0_fp16_cpu = ct.models.MLModel(chunk0_fp16_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f" {time.time()-t0:.1f}s")

    print("  FP16 ANE...", end="", flush=True)
    t0 = time.time()
    chunk0_fp16_ane = ct.models.MLModel(chunk0_fp16_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f" {time.time()-t0:.1f}s")

    # --- 6. Run CoreML ---
    print("\n[6] CoreML forward...")
    np_hidden = hidden_states.numpy().astype(np.float16)
    np_pos = position_ids.numpy().astype(np.int32)
    np_mask = causal_mask.numpy().astype(np.float16)
    np_curpos = current_pos.numpy().astype(np.int32)
    np_conv = np.zeros(lin_conv_shape, dtype=np.float16)
    np_rec = np.zeros(lin_rec_shape, dtype=np.float16)

    def run_coreml(model, label):
        feed = {
            "hidden_states": np_hidden.copy(),
            "position_ids": np_pos.copy(),
            "causal_mask": np_mask.copy(),
            "current_pos": np_curpos.copy(),
            "linear_conv_state": np_conv.copy(),
            "linear_recurrent_state": np_rec.copy(),
        }
        state = model.make_state()
        out = model.predict(feed, state=state)
        print(f"  {label} hidden norm: {np.linalg.norm(out['output_hidden_states']):.4f}")
        return out

    out_lut6_cpu = run_coreml(chunk0_lut6_cpu, "LUT6-CPU")
    out_lut6_ane = run_coreml(chunk0_lut6_ane, "LUT6-ANE")
    out_fp16_cpu = run_coreml(chunk0_fp16_cpu, "FP16-CPU")
    out_fp16_ane = run_coreml(chunk0_fp16_ane, "FP16-ANE")

    # --- 7. Compare ALL ---
    print("\n" + "=" * 80)
    print("  RESULTS: Who matches PyTorch?")
    print("=" * 80)
    
    results = {
        "PyTorch": {"hidden": pt_hidden, "conv": pt_conv, "rec": pt_rec},
        "LUT6-CPU": {"hidden": out_lut6_cpu["output_hidden_states"],
                     "conv": out_lut6_cpu.get("linear_conv_state_out"),
                     "rec": out_lut6_cpu.get("linear_recurrent_state_out")},
        "LUT6-ANE": {"hidden": out_lut6_ane["output_hidden_states"],
                     "conv": out_lut6_ane.get("linear_conv_state_out"),
                     "rec": out_lut6_ane.get("linear_recurrent_state_out")},
        "FP16-CPU": {"hidden": out_fp16_cpu["output_hidden_states"],
                     "conv": out_fp16_cpu.get("linear_conv_state_out"),
                     "rec": out_fp16_cpu.get("linear_recurrent_state_out")},
        "FP16-ANE": {"hidden": out_fp16_ane["output_hidden_states"],
                     "conv": out_fp16_ane.get("linear_conv_state_out"),
                     "rec": out_fp16_ane.get("linear_recurrent_state_out")},
    }
    
    names = list(results.keys())
    for key in ["hidden", "conv", "rec"]:
        print(f"\n  {key.upper()} states:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = results[names[i]][key]
                b = results[names[j]][key]
                if a is not None and b is not None:
                    compare(a, b, f"{names[i]} vs {names[j]}")

    print("\n  VERDICT (hidden states cosine to PyTorch):")
    for name in names[1:]:
        a64 = pt_hidden.flatten().astype(np.float64)
        b64 = np.asarray(results[name]["hidden"]).flatten().astype(np.float64)
        cos = np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30)
        print(f"    {name:12s}: {cos:.8f}")


if __name__ == "__main__":
    main()
