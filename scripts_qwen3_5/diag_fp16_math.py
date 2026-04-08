#!/usr/bin/env python3
"""Test: Does forcing FP16 math in PyTorch match CoreML better?

If yes, the root cause is FP32→FP16 truncation during ct.convert.
"""
import os, sys
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct
from config import CTX, NUM_CHUNKS
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape


def compare(a, b, label):
    a64 = np.asarray(a).flatten().astype(np.float64)
    b64 = np.asarray(b).flatten().astype(np.float64)
    cos = np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30)
    print(f"  {label}: cos={cos:.8f}")
    return cos


HF_PATH = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"
MODEL_DIR = "/Users/yw68/Anemll/qwen3_5_blockrecur_full"


def main():
    print("=" * 60)
    print("  FP32 vs FP16 math in recurrence")
    print("=" * 60)

    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval().half()
    for p in model.parameters():
        p.requires_grad = False

    # Chunk0 params
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    start_layer = 0
    end_layer = base + (1 if 0 < rem else 0)
    local_num_layers = end_layer - start_layer

    tcfg = cfg.text_config
    conv_dim = tcfg.linear_num_key_heads * tcfg.linear_key_head_dim * 2 + tcfg.linear_num_value_heads * tcfg.linear_value_head_dim
    conv_kernel = max(1, int(tcfg.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
    lin_conv_shape = (local_num_layers, ane_dim1, ane_dim2)
    lin_rec_shape = (local_num_layers, tcfg.linear_num_value_heads, tcfg.linear_key_head_dim, tcfg.linear_value_head_dim)

    # Input
    with torch.no_grad():
        tok = torch.tensor([[9906]], dtype=torch.long)
        hidden = model.model.embed_tokens(tok).half()

    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=torch.float16)
    causal_mask[:, :, :, :1] = 0
    current_pos = torch.zeros((1,), dtype=torch.int32)

    def run_pytorch(force_fp16_math):
        """Run chunk0 through PyTorch, optionally forcing FP16 math."""
        lin_conv = torch.zeros(lin_conv_shape, dtype=torch.float16)
        lin_rec = torch.zeros(lin_rec_shape, dtype=torch.float16)
        k_cache = torch.zeros(local_num_layers, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16)
        v_cache = torch.zeros(local_num_layers, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16)

        # Monkey-patch: set force_fp16_math on all linear attention layers in chunk0
        layer_types = tcfg.layer_types
        patched_fns = []
        for li in range(start_layer, end_layer):
            if layer_types[li] == "linear_attention":
                layer = model.model.layers[li]
                orig_fwd = layer.self_attn.forward_regular
                # Create a patched version that forces fp16 math
                def make_patched(orig):
                    def patched(*args, **kwargs):
                        kwargs['force_fp16_math'] = force_fp16_math
                        return orig(*args, **kwargs)
                    return patched
                layer.self_attn.forward_regular = make_patched(orig_fwd)
                patched_fns.append((layer.self_attn, 'forward_regular', orig_fwd))

        with torch.no_grad():
            out = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden.clone(),
                position_ids=position_ids.clone(),
                causal_mask=causal_mask.clone(),
                current_pos=current_pos.clone(),
                kv_cache_0=None,
                k_cache=k_cache,
                v_cache=v_cache,
                linear_conv_state=lin_conv,
                linear_recurrent_state=lin_rec,
                start_layer=start_layer,
                end_layer=end_layer,
                apply_final_norm=False,
            )

        # Restore originals
        for obj, name, orig in patched_fns:
            setattr(obj, name, orig)

        return out.numpy(), lin_conv.numpy(), lin_rec.numpy()

    print("\n[1] PyTorch with FP32 math (default)...")
    pt_fp32_h, pt_fp32_c, pt_fp32_r = run_pytorch(force_fp16_math=False)
    print(f"  hidden norm: {np.linalg.norm(pt_fp32_h):.4f}")

    print("\n[2] PyTorch with FP16 math (simulating CoreML)...")
    pt_fp16_h, pt_fp16_c, pt_fp16_r = run_pytorch(force_fp16_math=True)
    print(f"  hidden norm: {np.linalg.norm(pt_fp16_h):.4f}")

    print("\n[3] CoreML FP16 CPU...")
    chunk0 = ct.models.MLModel(
        os.path.join(MODEL_DIR, "ffn_LUT6_chunk0.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    feed = {
        "hidden_states": hidden.numpy().astype(np.float16),
        "position_ids": position_ids.numpy().astype(np.int32),
        "causal_mask": causal_mask.numpy().astype(np.float16),
        "current_pos": current_pos.numpy().astype(np.int32),
        "linear_conv_state": np.zeros(lin_conv_shape, dtype=np.float16),
        "linear_recurrent_state": np.zeros(lin_rec_shape, dtype=np.float16),
    }
    state = chunk0.make_state()
    cm_out = chunk0.predict(feed, state=state)
    cm_h = cm_out["output_hidden_states"]
    cm_c = cm_out["linear_conv_state_out"]
    cm_r = cm_out["linear_recurrent_state_out"]
    print(f"  hidden norm: {np.linalg.norm(cm_h):.4f}")

    # Comparisons
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print("\n  PyTorch FP32-math vs FP16-math (how much does precision matter?):")
    compare(pt_fp32_h, pt_fp16_h, "hidden")
    compare(pt_fp32_c, pt_fp16_c, "conv_state")
    compare(pt_fp32_r, pt_fp16_r, "rec_state")

    print("\n  PyTorch FP32-math vs CoreML CPU:")
    cos_fp32_cm = compare(pt_fp32_h, cm_h, "hidden")
    compare(pt_fp32_c, cm_c, "conv_state")
    compare(pt_fp32_r, cm_r, "rec_state")

    print("\n  PyTorch FP16-math vs CoreML CPU (should be much closer!):")
    cos_fp16_cm = compare(pt_fp16_h, cm_h, "hidden")
    compare(pt_fp16_c, cm_c, "conv_state")
    compare(pt_fp16_r, cm_r, "rec_state")

    print(f"\n  VERDICT:")
    print(f"    FP32-math→CoreML cos: {cos_fp32_cm:.6f}")
    print(f"    FP16-math→CoreML cos: {cos_fp16_cm:.6f}")
    if cos_fp16_cm > cos_fp32_cm + 0.01:
        print(f"    ✓ FP16 math is {cos_fp16_cm - cos_fp32_cm:.4f} closer → FP32→FP16 truncation is the root cause")
    else:
        print(f"    ✗ FP16 math NOT significantly closer → something else causes the divergence")


if __name__ == "__main__":
    main()
