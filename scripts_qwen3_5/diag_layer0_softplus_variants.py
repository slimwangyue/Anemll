#!/usr/bin/env python3
"""Phase 5: Test replacing F.softplus with decomposed ops.

The ANE corrupts F.softplus when a depthwise conv is in the same graph.
Try: log(1 + exp(x)) decomposed, or relu approximation.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
OUT_DIR = "/tmp/diag_g_fix2"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 512
NUM_V_HEADS = 32
KEY_HEAD_DIM = 128
VAL_HEAD_DIM = 128
NUM_K_HEADS = 16
KEY_DIM = NUM_K_HEADS * KEY_HEAD_DIM
VALUE_DIM = NUM_V_HEADS * VAL_HEAD_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
CONV_KERNEL = 4


def cmp(name, a, b):
    a_np = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else a.astype(np.float32)
    b_np = b.detach().float().cpu().numpy() if isinstance(b, torch.Tensor) else b.astype(np.float32)
    diff = a_np - b_np
    mad = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    a_f, b_f = a_np.flatten(), b_np.flatten()
    cos = np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12)
    print(f"  {name:55s}  MAD={mad:.6f}  mean={mean_abs:.6f}  cos={cos:.6f}  "
          f"|a|={np.max(np.abs(a_np)):.4f}  |b|={np.max(np.abs(b_np)):.4f}")
    return mad


def load_model():
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    print("Loading model...")
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = 2048
    cfg.state_length = 2048
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_inputs(model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    ids = tok.encode("Hello, how are you doing today?", add_special_tokens=True)
    input_ids = torch.zeros(1, BATCH_SIZE, dtype=torch.long)
    input_ids[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids[0]).unsqueeze(0).to(torch.float16)
    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    with torch.no_grad():
        x = layer0.input_layernorm(hidden)
        mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(x)
    return mixed_qkv_pre, z_cf, b_cf, a_cf


def _manual_softplus(x):
    """softplus(x) = log(1 + exp(x)), decomposed into individual ops."""
    return torch.log(1.0 + torch.exp(x))


def _stable_softplus(x):
    """Numerically stable softplus: max(x,0) + log(1 + exp(-|x|))"""
    return F.relu(x) + torch.log(1.0 + torch.exp(-torch.abs(x)))


def _threshold_softplus(x, threshold=20.0):
    """softplus with threshold (same as F.softplus default)."""
    return torch.where(x > threshold, x, torch.log(1.0 + torch.exp(x)))


class ConvLayoutDecomposedSoftplus(torch.nn.Module):
    """Conv + layout with softplus decomposed as log(1 + exp(x))."""
    def __init__(self, conv_stage, layout_stage, A_log, dt_bias, seq_len, softplus_fn):
        super().__init__()
        self.conv_stage = conv_stage
        self.layout_stage = layout_stage
        self.register_buffer("A_log", A_log)
        self.register_buffer("dt_bias", dt_bias)
        self.seq_len = seq_len
        self.softplus_fn = softplus_fn

    def forward(self, mixed_qkv_pre, z_cf, b_cf, a_cf, conv_state):
        # Compute g with custom softplus
        a = a_cf.squeeze(2).transpose(1, 2)
        g = -self.A_log.float().exp() * self.softplus_fn(a.float() + self.dt_bias)

        # Run conv + layout (ignore layout's g)
        conv_out_cf, next_conv_state = self.conv_stage(
            mixed_qkv_pre, conv_state, expected_seq_len=self.seq_len
        )
        query, key, value, _g, beta, z = self.layout_stage(
            conv_out_cf, z_cf, b_cf,
            torch.zeros_like(b_cf),
            bsz=1, seq_len=self.seq_len,
            force_fp16_math=False,
        )
        return query, key, value, g, beta, z, next_conv_state


def test_variant(model, mixed_qkv_pre, z_cf, b_cf, a_cf, name, softplus_fn):
    """Test a specific softplus variant."""
    attn = model.model.layers[0].self_attn
    conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL, dtype=torch.float16)

    wrapper = ConvLayoutDecomposedSoftplus(
        attn.conv_stage, attn.layout_stage,
        attn.A_log.data.clone(), attn.dt_bias.data.clone(),
        BATCH_SIZE, softplus_fn
    )
    wrapper.eval()

    inputs = (mixed_qkv_pre, z_cf, b_cf, a_cf, conv_state)
    with torch.no_grad():
        pt_out = wrapper(*inputs)
    pt_g = pt_out[3]

    print(f"  Tracing {name}...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, inputs, check_trace=False)

    print(f"  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mixed_qkv_pre", shape=mixed_qkv_pre.shape, dtype=np.float16),
            ct.TensorType(name="z_cf", shape=z_cf.shape, dtype=np.float16),
            ct.TensorType(name="b_cf", shape=b_cf.shape, dtype=np.float16),
            ct.TensorType(name="a_cf", shape=a_cf.shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="query", dtype=np.float16),
            ct.TensorType(name="key", dtype=np.float16),
            ct.TensorType(name="value", dtype=np.float16),
            ct.TensorType(name="g", dtype=np.float16),
            ct.TensorType(name="beta", dtype=np.float16),
            ct.TensorType(name="z", dtype=np.float16),
            ct.TensorType(name="next_conv_state", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    save_path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    mlmodel.save(save_path)

    inp_np = {
        "mixed_qkv_pre": mixed_qkv_pre.cpu().numpy(),
        "z_cf": z_cf.cpu().numpy(),
        "b_cf": b_cf.cpu().numpy(),
        "a_cf": a_cf.cpu().numpy(),
        "conv_state": conv_state.cpu().numpy(),
    }

    results = {}
    for label, cu in [("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE)]:
        m = ct.models.MLModel(save_path, compute_units=cu)
        out = m.predict(inp_np)
        results[label] = out

    g_gpu = results["GPU"]["g"].astype(np.float32)
    g_ane = results["ANE"]["g"].astype(np.float32)
    g_mad = cmp(f"g GPU vs ANE ({name})", results["GPU"]["g"], results["ANE"]["g"])

    for h in [4, 12, 13, 20, 21]:
        d = np.abs(g_gpu[0, :, h] - g_ane[0, :, h]).max()
        print(f"    Head {h}: MAD={d:.6f}  GPU=[{g_gpu[0,:8,h].min():.4f},{g_gpu[0,:8,h].max():.4f}]  "
              f"ANE=[{g_ane[0,:8,h].min():.4f},{g_ane[0,:8,h].max():.4f}]")

    return g_mad


def main():
    print("=" * 80)
    print("PHASE 5: Test decomposed softplus variants")
    print("=" * 80)

    model = load_model()
    mixed_qkv_pre, z_cf, b_cf, a_cf = get_inputs(model)

    variants = [
        ("manual_log1pexp", _manual_softplus),
        ("stable_relu_log", _stable_softplus),
        ("threshold_20", _threshold_softplus),
    ]

    mads = {}
    for name, fn in variants:
        print(f"\n{'='*80}")
        print(f"Variant: {name}")
        print(f"{'='*80}")
        mads[name] = test_variant(model, mixed_qkv_pre, z_cf, b_cf, a_cf, name, fn)

    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"  Original F.softplus:     MAD=6.45 (from Phase 2)")
    for name, mad in mads.items():
        status = "FIXED!" if mad < 0.01 else "STILL BROKEN"
        print(f"  {name:25s}: MAD={mad:.6f}  {status}")
    print("\n[Done]")


if __name__ == "__main__":
    main()
