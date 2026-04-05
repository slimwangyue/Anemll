#!/usr/bin/env python3
"""Verify the stable-softplus fix: re-export conv+layout with patched model and compare GPU vs ANE."""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
OUT_DIR = "/tmp/diag_verify_fix"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 512
NUM_V_HEADS = 32
NUM_K_HEADS = 16
KEY_HEAD_DIM = 128
VAL_HEAD_DIM = 128
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
    print(f"  {name:55s}  MAD={mad:.6f}  mean={mean_abs:.6f}  cos={cos:.6f}")
    return mad


class ConvLayoutWrapper(torch.nn.Module):
    def __init__(self, conv_stage, layout_stage, seq_len):
        super().__init__()
        self.conv_stage = conv_stage
        self.layout_stage = layout_stage
        self.seq_len = seq_len

    def forward(self, mixed_qkv_pre, z_cf, b_cf, a_cf, conv_state):
        conv_out_cf, next_conv_state = self.conv_stage(
            mixed_qkv_pre, conv_state, expected_seq_len=self.seq_len
        )
        query, key, value, g, beta, z = self.layout_stage(
            conv_out_cf, z_cf, b_cf, a_cf,
            bsz=1, seq_len=self.seq_len, force_fp16_math=False,
        )
        return query, key, value, g, beta, z, next_conv_state


def main():
    print("=" * 80)
    print("VERIFY: stable-softplus fix in patched model")
    print("=" * 80)

    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    from transformers import AutoTokenizer

    print("Loading patched model...")
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = 2048
    cfg.state_length = 2048
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

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

    wrapper = ConvLayoutWrapper(attn.conv_stage, attn.layout_stage, BATCH_SIZE)
    wrapper.eval()
    conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL, dtype=torch.float16)

    with torch.no_grad():
        pt_out = wrapper(mixed_qkv_pre, z_cf, b_cf, a_cf, conv_state)
    pt_g = pt_out[3]
    print(f"  PyTorch g range: [{pt_g.min():.4f}, {pt_g.max():.4f}]")

    print("  Tracing patched conv+layout...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (mixed_qkv_pre, z_cf, b_cf, a_cf, conv_state), check_trace=False)

    print("  Converting to CoreML...")
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
    save_path = os.path.join(OUT_DIR, "conv_layout_fixed.mlpackage")
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
        print(f"  Loading on {label}...")
        m = ct.models.MLModel(save_path, compute_units=cu)
        out = m.predict(inp_np)
        results[label] = out
        cmp(f"g (PyTorch vs {label})", pt_g, out["g"])

    print(f"\n  GPU vs ANE comparison (all outputs):")
    for key in ["query", "key", "value", "g", "beta"]:
        cmp(f"{key} (GPU vs ANE)", results["GPU"][key], results["ANE"][key])

    g_gpu = results["GPU"]["g"].astype(np.float32)
    g_ane = results["ANE"]["g"].astype(np.float32)
    print(f"\n  Per-head g GPU vs ANE (problem heads):")
    for h in [4, 12, 13, 20, 21]:
        d = np.abs(g_gpu[0, :, h] - g_ane[0, :, h]).max()
        print(f"    Head {h}: MAD={d:.6f}  GPU=[{g_gpu[0,:8,h].min():.4f},{g_gpu[0,:8,h].max():.4f}]  "
              f"ANE=[{g_ane[0,:8,h].min():.4f},{g_ane[0,:8,h].max():.4f}]  "
              f"{'OK' if d < 0.01 else 'BROKEN'}")

    all_ok = np.max(np.abs(g_gpu - g_ane)) < 0.01
    print(f"\n  RESULT: {'PASS — stable-softplus fix works!' if all_ok else 'FAIL — still diverging'}")
    print("\n[Done]")


if __name__ == "__main__":
    main()
