#!/usr/bin/env python3
"""Ablation: which layer type causes FP16 compute precision error?

Splits chunk0 (layers 0-5) into 3 contiguous groups:
  Group A: layers 0-2  (linear, linear, linear)
  Group B: layer  3    (full_attention)
  Group C: layers 4-5  (linear, linear)

Converts each group at BOTH FP16 and FP32 precision, then composes:
  Exp1 "linear=FP32": A@FP32 → B@FP16 → C@FP32   (linear layers stay FP32)
  Exp2 "full=FP32":   A@FP16 → B@FP32 → C@FP16   (full-attn layer stays FP32)
Controls:
  all-FP16:           A@FP16 → B@FP16 → C@FP16
  all-FP32:           A@FP32 → B@FP32 → C@FP32
"""
import os, sys, time, warnings
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

# ---------- helpers ----------
def cos_sim(a, b):
    a64 = np.asarray(a).flatten().astype(np.float64)
    b64 = np.asarray(b).flatten().astype(np.float64)
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))


# ---------- per-group wrapper ----------
class GroupWrapper(torch.nn.Module):
    """Wraps a contiguous range of layers using the same code path as the
    full-chunk FFNWrapper used during export."""

    def __init__(self, model, start_layer, end_layer, cfg):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        n = end_layer - start_layer
        self.register_buffer(
            "k_cache",
            torch.zeros(n, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(n, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE),
        )
        self.states = Qwen35Converter.GetChunkLocalTransformerStates(
            model, n, prefix="", split_full_attention_kv=True,
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


def build_and_convert(model, cfg, start_layer, end_layer, precision, tcfg):
    """Build wrapper, trace, convert at given precision. Returns CoreML model."""
    n = end_layer - start_layer
    wrapper = GroupWrapper(model, start_layer, end_layer, cfg).eval()

    # Linear state shapes for this group
    conv_dim = (tcfg.linear_num_key_heads * tcfg.linear_key_head_dim * 2
                + tcfg.linear_num_value_heads * tcfg.linear_value_head_dim)
    conv_kernel = max(1, int(tcfg.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
    lin_conv_shape = (n, ane_dim1, ane_dim2)
    lin_rec_shape = (n, tcfg.linear_num_value_heads, tcfg.linear_key_head_dim, tcfg.linear_value_head_dim)

    hidden = torch.zeros(1, 1, cfg.hidden_size, dtype=torch.float16)
    pos_ids = torch.zeros(1, dtype=torch.int32)
    mask = torch.zeros(1, 1, 1, CTX, dtype=torch.float16)
    cur_pos = torch.zeros(1, dtype=torch.int32)
    lc = torch.zeros(lin_conv_shape, dtype=torch.float16)
    lr = torch.zeros(lin_rec_shape, dtype=torch.float16)

    # Reset state buffers then trace
    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    traced = torch.jit.trace(wrapper, (hidden, pos_ids, mask, cur_pos, lc, lr), check_trace=False)
    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    for _, buf in traced.named_buffers():
        buf.zero_()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=pos_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=cur_pos.shape, dtype=np.int32),
                ct.TensorType(name="linear_conv_state", shape=lc.shape, dtype=np.float16),
                ct.TensorType(name="linear_recurrent_state", shape=lr.shape, dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="output_hidden_states", dtype=np.float16),
                ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
                ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
            ],
            states=wrapper.states,
            compute_precision=precision,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
    return mlmodel, lin_conv_shape, lin_rec_shape


def run_group(mlmodel, hidden_np, mask_np, pos_np, conv_shape, rec_shape):
    """Run a per-group CoreML model with fresh states."""
    feed = {
        "hidden_states": hidden_np.astype(np.float16),
        "position_ids": pos_np.copy(),
        "causal_mask": mask_np.copy(),
        "current_pos": pos_np.copy(),
        "linear_conv_state": np.zeros(conv_shape, dtype=np.float16),
        "linear_recurrent_state": np.zeros(rec_shape, dtype=np.float16),
    }
    state = mlmodel.make_state()
    out = mlmodel.predict(feed, state=state)
    return out["output_hidden_states"], out["linear_conv_state_out"], out["linear_recurrent_state_out"]


def main():
    print("=" * 70)
    print("  ABLATION: Linear-attention vs Full-attention FP16 error")
    print("=" * 70)

    # Load model
    print("\n[1] Loading PyTorch model...")
    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX; cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval().half()
    for p in model.parameters():
        p.requires_grad = False
    tcfg = cfg.text_config

    layer_types = tcfg.layer_types[:6]
    print(f"  Chunk0 layers: {layer_types}")

    # Groups: contiguous by type
    groups = [
        (0, 3, "A: layers 0-2 (L,L,L)"),
        (3, 4, "B: layer 3    (F)"),
        (4, 6, "C: layers 4-5 (L,L)"),
    ]

    # PyTorch ground truth (full chunk0)
    print("\n[2] PyTorch ground truth for full chunk0...")
    with torch.no_grad():
        tok = torch.tensor([[9906]], dtype=torch.long)  # "Hello"
        hidden = model.model.embed_tokens(tok).half()
    hidden_np = hidden.numpy().astype(np.float16)
    pos_np = np.array([0], dtype=np.int32)
    mask_np = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask_np[:, :, :, :1] = 0

    # Run PyTorch through all 6 layers
    total_layers = cfg.num_hidden_layers
    base, rem = divmod(total_layers, NUM_CHUNKS)
    local_num = base + (1 if 0 < rem else 0)  # 6 for chunk0
    conv_dim = (tcfg.linear_num_key_heads * tcfg.linear_key_head_dim * 2
                + tcfg.linear_num_value_heads * tcfg.linear_value_head_dim)
    conv_kernel = max(1, int(tcfg.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
    full_conv_shape = (local_num, ane_dim1, ane_dim2)
    full_rec_shape = (local_num, tcfg.linear_num_value_heads, tcfg.linear_key_head_dim, tcfg.linear_value_head_dim)

    with torch.no_grad():
        pt_lc = torch.zeros(full_conv_shape, dtype=torch.float16)
        pt_lr = torch.zeros(full_rec_shape, dtype=torch.float16)
        pt_out = model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden.clone(),
            position_ids=torch.zeros(1, dtype=torch.int32),
            causal_mask=torch.from_numpy(mask_np.copy()),
            current_pos=torch.zeros(1, dtype=torch.int32),
            kv_cache_0=None,
            k_cache=torch.zeros(local_num, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16),
            v_cache=torch.zeros(local_num, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16),
            linear_conv_state=pt_lc,
            linear_recurrent_state=pt_lr,
            start_layer=0, end_layer=6, apply_final_norm=False,
        )
    pt_hidden = pt_out.numpy()
    print(f"  PyTorch output norm: {np.linalg.norm(pt_hidden):.4f}")

    # Convert each group at FP16 and FP32
    print("\n[3] Converting 6 per-group models (3 groups × 2 precisions)...")
    models = {}  # key: (group_idx, precision_str) → (mlmodel, conv_shape, rec_shape)
    for gi, (start, end, label) in enumerate(groups):
        for prec_str, prec in [("fp16", ct.precision.FLOAT16), ("fp32", ct.precision.FLOAT32)]:
            tag = f"{label} @ {prec_str.upper()}"
            t0 = time.time()
            ml, cs, rs = build_and_convert(model, cfg, start, end, prec, tcfg)
            dt = time.time() - t0
            models[(gi, prec_str)] = (ml, cs, rs)
            print(f"  {tag}  ({dt:.1f}s)")

    # Define experiments
    experiments = {
        "all-FP16":      ["fp16", "fp16", "fp16"],
        "all-FP32":      ["fp32", "fp32", "fp32"],
        "linear=FP32":   ["fp32", "fp16", "fp32"],  # linear groups FP32, full-attn FP16
        "full-attn=FP32":["fp16", "fp32", "fp16"],  # linear groups FP16, full-attn FP32
    }

    # Run each experiment
    print("\n[4] Running ablation experiments...")
    results = {}
    for exp_name, prec_list in experiments.items():
        h = hidden_np.copy()
        for gi, prec_str in enumerate(prec_list):
            ml, cs, rs = models[(gi, prec_str)]
            h, _, _ = run_group(ml, h, mask_np, pos_np, cs, rs)
        cos = cos_sim(pt_hidden, h)
        norm = np.linalg.norm(h)
        results[exp_name] = cos
        print(f"  {exp_name:18s}: cos={cos:.8f}  norm={norm:.4f}")

    # Verdict
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Experiment':<20s} {'Cosine to PyTorch':>20s}")
    print(f"  {'-'*20} {'-'*20}")
    for name, cos in results.items():
        print(f"  {name:<20s} {cos:>20.8f}")

    fp16_err = 1.0 - results["all-FP16"]
    fp32_err = 1.0 - results["all-FP32"]
    lin_fp32_err = 1.0 - results["linear=FP32"]
    full_fp32_err = 1.0 - results["full-attn=FP32"]

    # How much error does each fix remove?
    linear_contribution = (fp16_err - lin_fp32_err) / (fp16_err - fp32_err + 1e-30) * 100
    full_contribution = (fp16_err - full_fp32_err) / (fp16_err - fp32_err + 1e-30) * 100

    print(f"\n  Error breakdown (1 - cos, lower = better):")
    print(f"    all-FP16 error:      {fp16_err:.8f}")
    print(f"    all-FP32 error:      {fp32_err:.8f}")
    print(f"    linear=FP32 error:   {lin_fp32_err:.8f}  → fixing linear removes {linear_contribution:.1f}% of error")
    print(f"    full-attn=FP32 error:{full_fp32_err:.8f}  → fixing full-attn removes {full_contribution:.1f}% of error")

    # Per-group standalone error
    print(f"\n  Per-group standalone error (FP16 vs FP32):")
    for gi, (start, end, label) in enumerate(groups):
        ml16, cs, rs = models[(gi, "fp16")]
        ml32, _, _ = models[(gi, "fp32")]
        h16, _, _ = run_group(ml16, hidden_np.copy(), mask_np, pos_np, cs, rs)
        h32, _, _ = run_group(ml32, hidden_np.copy(), mask_np, pos_np, cs, rs)
        cos_16_32 = cos_sim(h16, h32)
        print(f"    {label}: cos(FP16, FP32) = {cos_16_32:.8f}")


if __name__ == "__main__":
    main()
