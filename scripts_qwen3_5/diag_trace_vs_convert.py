#!/usr/bin/env python3
"""Test: Is the error from JIT tracing or from ct.convert?

Compares: PyTorch eager → JIT traced → CoreML CPU
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
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape, MODEL_DTYPE
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

HF_PATH = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"
MODEL_DIR = "/Users/yw68/Anemll/qwen3_5_blockrecur_full"
TEST_DEVICE = "cpu"


def compare(a, b, label):
    a64 = np.asarray(a).flatten().astype(np.float64)
    b64 = np.asarray(b).flatten().astype(np.float64)
    cos = np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30)
    mad = np.abs(a64 - b64).max()
    print(f"  {label}: cos={cos:.8f}  max_diff={mad:.6f}")
    return cos


def main():
    print("=" * 60)
    print("  Tracing vs ct.convert isolation test")
    print("=" * 60)

    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval().half()
    for p in model.parameters():
        p.requires_grad = False

    # Chunk0
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

    # Build FFNWrapper exactly like the converter does
    print("\n[1] Building FFNWrapper for chunk0...")

    class FFNWrapper(torch.nn.Module):
        def __init__(self, mdl, s_layer, e_layer):
            super().__init__()
            self.model = mdl
            self.start_layer = s_layer
            self.end_layer = e_layer
            n_local = e_layer - s_layer
            self.register_buffer(
                "k_cache",
                torch.zeros(n_local, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE),
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(n_local, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE),
            )
            self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                mdl, n_local, prefix="", split_full_attention_kv=True,
            )

        def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                    linear_conv_state, linear_recurrent_state):
            out = self.model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                kv_cache_0=None,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                linear_conv_state=linear_conv_state,
                linear_recurrent_state=linear_recurrent_state,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                apply_final_norm=False,
            )
            return out, linear_conv_state, linear_recurrent_state

    wrapper = FFNWrapper(model, start_layer, end_layer).eval()

    # Input tensors
    with torch.no_grad():
        tok = torch.tensor([[9906]], dtype=torch.long)
        hidden = model.model.embed_tokens(tok).half()

    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.full((1, 1, 1, CTX), -65504.0, dtype=torch.float16)
    causal_mask[:, :, :, :1] = 0
    current_pos = torch.zeros((1,), dtype=torch.int32)
    lin_conv = torch.zeros(lin_conv_shape, dtype=torch.float16)
    lin_rec = torch.zeros(lin_rec_shape, dtype=torch.float16)

    # --- Eager PyTorch ---
    print("\n[2] PyTorch eager forward...")
    def reset_wrapper():
        wrapper.k_cache.zero_()
        wrapper.v_cache.zero_()

    reset_wrapper()
    lc = lin_conv.clone()
    lr = lin_rec.clone()
    with torch.no_grad():
        eager_h, eager_c, eager_r = wrapper(
            hidden.clone(), position_ids.clone(), causal_mask.clone(),
            current_pos.clone(), lc, lr,
        )
    eager_h = eager_h.numpy()
    eager_c = lc.numpy()
    eager_r = lr.numpy()
    print(f"  hidden norm: {np.linalg.norm(eager_h):.4f}")

    # --- JIT traced ---
    print("\n[3] JIT tracing...")
    reset_wrapper()
    traced = torch.jit.trace(
        wrapper,
        (hidden.clone(), position_ids.clone(), causal_mask.clone(),
         current_pos.clone(), lin_conv.clone(), lin_rec.clone()),
        check_trace=False,
    )

    print("  Running traced forward...")
    reset_wrapper()
    # Also reset traced wrapper's state buffers
    for name, buf in traced.named_buffers():
        if 'k_cache' in name or 'v_cache' in name:
            buf.zero_()

    lc_t = lin_conv.clone()
    lr_t = lin_rec.clone()
    with torch.no_grad():
        traced_h, traced_c, traced_r = traced(
            hidden.clone(), position_ids.clone(), causal_mask.clone(),
            current_pos.clone(), lc_t, lr_t,
        )
    traced_h = traced_h.numpy()
    traced_c = lc_t.numpy()
    traced_r = lr_t.numpy()
    print(f"  hidden norm: {np.linalg.norm(traced_h):.4f}")

    # --- CoreML CPU ---
    print("\n[4] CoreML CPU...")
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

    # --- Compare ---
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print("\n  Hidden states:")
    compare(eager_h, traced_h, "Eager vs Traced")
    compare(eager_h, cm_h,     "Eager vs CoreML")
    compare(traced_h, cm_h,    "Traced vs CoreML")

    print("\n  Conv state:")
    compare(eager_c, traced_c, "Eager vs Traced")
    compare(eager_c, cm_c,     "Eager vs CoreML")
    compare(traced_c, cm_c,    "Traced vs CoreML")

    print("\n  Recurrent state:")
    compare(eager_r, traced_r, "Eager vs Traced")
    compare(eager_r, cm_r,     "Eager vs CoreML")
    compare(traced_r, cm_r,    "Traced vs CoreML")


if __name__ == "__main__":
    main()
