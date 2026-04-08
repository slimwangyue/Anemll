#!/usr/bin/env python3
"""Quick test: Convert chunk0 with FLOAT32 precision and compare to FLOAT16.

If FLOAT32 conversion dramatically improves accuracy on CPU, the error
is from FP16 intermediate truncation during ct.convert.
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
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape, MODEL_DTYPE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

HF_PATH = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"

def compare(a, b, label):
    a64 = np.asarray(a).flatten().astype(np.float64)
    b64 = np.asarray(b).flatten().astype(np.float64)
    cos = np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30)
    print(f"  {label}: cos={cos:.8f}")
    return cos


def main():
    print("=" * 60)
    print("  FLOAT32 vs FLOAT16 ct.convert precision test")
    print("=" * 60)

    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval().half()
    for p in model.parameters():
        p.requires_grad = False

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

    class FFNWrapper(torch.nn.Module):
        def __init__(self, mdl, s_layer, e_layer):
            super().__init__()
            self.model = mdl
            self.start_layer = s_layer
            self.end_layer = e_layer
            n_local = e_layer - s_layer
            self.register_buffer("k_cache", torch.zeros(n_local, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE))
            self.register_buffer("v_cache", torch.zeros(n_local, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE))
            self.states = Qwen35Converter.GetChunkLocalTransformerStates(mdl, n_local, prefix="", split_full_attention_kv=True)

        def forward(self, hidden_states, position_ids, causal_mask, current_pos, linear_conv_state, linear_recurrent_state):
            out = self.model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden_states, position_ids=position_ids, causal_mask=causal_mask,
                current_pos=current_pos, kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
                linear_conv_state=linear_conv_state, linear_recurrent_state=linear_recurrent_state,
                start_layer=self.start_layer, end_layer=self.end_layer, apply_final_norm=False,
            )
            return out, linear_conv_state, linear_recurrent_state

    print("\n[1] Building and tracing FFNWrapper...")
    wrapper = FFNWrapper(model, start_layer, end_layer).eval()

    with torch.no_grad():
        tok = torch.tensor([[9906]], dtype=torch.long)
        hidden = model.model.embed_tokens(tok).half()

    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=torch.float16)
    causal_mask[:, :, :, :1] = 0
    current_pos = torch.zeros((1,), dtype=torch.int32)
    lin_conv = torch.zeros(lin_conv_shape, dtype=torch.float16)
    lin_rec = torch.zeros(lin_rec_shape, dtype=torch.float16)

    # Get PyTorch ground truth
    print("\n[2] PyTorch eager ground truth...")
    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    lc = lin_conv.clone(); lr = lin_rec.clone()
    with torch.no_grad():
        pt_h, _, _ = wrapper(hidden.clone(), position_ids.clone(), causal_mask.clone(), current_pos.clone(), lc, lr)
    pt_h = pt_h.numpy()

    # Trace for conversion
    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    traced = torch.jit.trace(wrapper, (hidden.clone(), position_ids.clone(), causal_mask.clone(), current_pos.clone(), lin_conv.clone(), lin_rec.clone()), check_trace=False)
    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    for name, buf in traced.named_buffers():
        if 'k_cache' in name or 'v_cache' in name:
            buf.zero_()

    # --- Convert with FLOAT16 ---
    print("\n[3] Converting with FLOAT16 precision...")
    t0 = time.time()
    ml_fp16 = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
        ],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Reset traced state buffers for second conversion
    for name, buf in traced.named_buffers():
        if 'k_cache' in name or 'v_cache' in name:
            buf.zero_()

    # --- Convert with FLOAT32 ---
    print("\n[4] Converting with FLOAT32 precision...")
    t0 = time.time()
    ml_fp32 = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
        ],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # --- Run both ---
    print("\n[5] Running inference...")
    feed = {
        "hidden_states": hidden.numpy().astype(np.float16),
        "position_ids": position_ids.numpy().astype(np.int32),
        "causal_mask": causal_mask.numpy().astype(np.float16),
        "current_pos": current_pos.numpy().astype(np.int32),
        "linear_conv_state": np.zeros(lin_conv_shape, dtype=np.float16),
        "linear_recurrent_state": np.zeros(lin_rec_shape, dtype=np.float16),
    }

    state16 = ml_fp16.make_state()
    out16 = ml_fp16.predict(dict(feed), state=state16)
    h16 = out16["output_hidden_states"]

    state32 = ml_fp32.make_state()
    out32 = ml_fp32.predict(dict(feed), state=state32)
    h32 = out32["output_hidden_states"]

    print(f"  FP16 model hidden norm: {np.linalg.norm(h16):.4f}")
    print(f"  FP32 model hidden norm: {np.linalg.norm(h32):.4f}")
    print(f"  PyTorch hidden norm:    {np.linalg.norm(pt_h):.4f}")

    # --- Compare ---
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print("\n  Hidden states:")
    cos_pt_16 = compare(pt_h, h16, "PyTorch vs CoreML-FP16")
    cos_pt_32 = compare(pt_h, h32, "PyTorch vs CoreML-FP32")
    compare(h16, h32, "CoreML-FP16 vs CoreML-FP32")

    print(f"\n  VERDICT:")
    print(f"    FP16 conversion cos: {cos_pt_16:.6f}")
    print(f"    FP32 conversion cos: {cos_pt_32:.6f}")
    if cos_pt_32 > 0.999:
        print(f"    ✓ FP32 conversion solves the precision problem!")
    elif cos_pt_32 > cos_pt_16 + 0.01:
        print(f"    ~ FP32 is better by {cos_pt_32 - cos_pt_16:.4f} but still imperfect")
    else:
        print(f"    ✗ FP32 doesn't help — error is somewhere else")


if __name__ == "__main__":
    main()
