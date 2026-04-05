#!/usr/bin/env python3
"""Phase 4: Test mitigation — isolate g computation from conv in the same model.

Root cause: CoreML ANE graph optimizer corrupts the g branch when
it's fused with the depthwise conv. The fix must prevent this fusion.

Test approaches:
  A. Compute g separately (before conv) and pass as input to layout
  B. Add a .contiguous() barrier between conv and g computation
  C. Use a separate nn.Module for g that creates a subgraph boundary
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
OUT_DIR = "/tmp/diag_g_fix"
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


# ══════════════════════════════════════════════════════════════════════════
# FIX A: Precompute g from a_cf, pass as input to model
# ══════════════════════════════════════════════════════════════════════════

class ConvLayoutWithGInput(torch.nn.Module):
    """Conv + layout, but receives g as an INPUT instead of computing it."""
    def __init__(self, conv_stage, layout_stage, seq_len):
        super().__init__()
        self.conv_stage = conv_stage
        self.layout_stage = layout_stage
        self.seq_len = seq_len

    def forward(self, mixed_qkv_pre, z_cf, b_cf, g_precomputed, conv_state):
        conv_out_cf, next_conv_state = self.conv_stage(
            mixed_qkv_pre, conv_state, expected_seq_len=self.seq_len
        )
        # Run layout but REPLACE g with precomputed value
        query, key, value, _g_ignored, beta, z = self.layout_stage(
            conv_out_cf, z_cf, b_cf,
            # Pass a_cf=zeros so the computed g is zero (we'll replace it)
            torch.zeros_like(b_cf),
            bsz=1, seq_len=self.seq_len,
            force_fp16_math=False,
        )
        # Return precomputed g instead of the layout-computed one
        return query, key, value, g_precomputed, beta, z, next_conv_state


# ══════════════════════════════════════════════════════════════════════════
# FIX B: Separate g computation into its own nn.Module subgraph
# ══════════════════════════════════════════════════════════════════════════

class GCompModule(torch.nn.Module):
    """Compute g = -exp(A_log) * softplus(a + dt_bias) as a separate module."""
    def __init__(self, A_log, dt_bias):
        super().__init__()
        self.register_buffer("A_log", A_log)
        self.register_buffer("dt_bias", dt_bias)

    def forward(self, a_cf):
        a = a_cf.squeeze(2).transpose(1, 2)  # (1,32,1,512) → (1,512,32)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        return g


class ConvLayoutWithSeparateG(torch.nn.Module):
    """Conv + layout, with g computed in a separate module."""
    def __init__(self, conv_stage, layout_stage, g_module, seq_len):
        super().__init__()
        self.conv_stage = conv_stage
        self.layout_stage = layout_stage
        self.g_module = g_module
        self.seq_len = seq_len

    def forward(self, mixed_qkv_pre, z_cf, b_cf, a_cf, conv_state):
        # Compute g in its own subgraph
        g = self.g_module(a_cf)

        # Run conv + layout (layout will compute its own g but we ignore it)
        conv_out_cf, next_conv_state = self.conv_stage(
            mixed_qkv_pre, conv_state, expected_seq_len=self.seq_len
        )
        query, key, value, _g_layout, beta, z = self.layout_stage(
            conv_out_cf, z_cf, b_cf,
            torch.zeros_like(b_cf),  # dummy a_cf
            bsz=1, seq_len=self.seq_len,
            force_fp16_math=False,
        )
        return query, key, value, g, beta, z, next_conv_state


# ══════════════════════════════════════════════════════════════════════════
# FIX C: Compute g FIRST, then conv, with contiguous() barrier
# ══════════════════════════════════════════════════════════════════════════

class ConvLayoutGFirst(torch.nn.Module):
    """Compute g BEFORE conv to prevent them sharing graph resources on ANE."""
    def __init__(self, conv_stage, layout_stage, A_log, dt_bias, seq_len):
        super().__init__()
        self.conv_stage = conv_stage
        self.layout_stage = layout_stage
        self.register_buffer("A_log", A_log)
        self.register_buffer("dt_bias", dt_bias)
        self.seq_len = seq_len

    def forward(self, mixed_qkv_pre, z_cf, b_cf, a_cf, conv_state):
        # COMPUTE G FIRST (before conv)
        a = a_cf.squeeze(2).transpose(1, 2)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        g = g.contiguous()  # Force materialization before conv

        # THEN run conv
        conv_out_cf, next_conv_state = self.conv_stage(
            mixed_qkv_pre, conv_state, expected_seq_len=self.seq_len
        )

        # Layout WITHOUT g (get q,k,v,beta,z from conv output)
        query, key, value, _g_ignored, beta, z = self.layout_stage(
            conv_out_cf, z_cf, b_cf,
            torch.zeros_like(b_cf),
            bsz=1, seq_len=self.seq_len,
            force_fp16_math=False,
        )
        return query, key, value, g, beta, z, next_conv_state


def test_fix(model, mixed_qkv_pre, z_cf, b_cf, a_cf, fix_name, wrapper, extra_inputs=None):
    """Export a model and compare GPU vs ANE."""
    attn = model.model.layers[0].self_attn
    conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL, dtype=torch.float16)

    wrapper.eval()

    # Build inputs
    if extra_inputs is None:
        inputs_tuple = (mixed_qkv_pre, z_cf, b_cf, a_cf, conv_state)
        input_specs = [
            ct.TensorType(name="mixed_qkv_pre", shape=mixed_qkv_pre.shape, dtype=np.float16),
            ct.TensorType(name="z_cf", shape=z_cf.shape, dtype=np.float16),
            ct.TensorType(name="b_cf", shape=b_cf.shape, dtype=np.float16),
            ct.TensorType(name="a_cf", shape=a_cf.shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
        ]
    else:
        inputs_tuple = extra_inputs
        input_specs = [
            ct.TensorType(name="mixed_qkv_pre", shape=mixed_qkv_pre.shape, dtype=np.float16),
            ct.TensorType(name="z_cf", shape=z_cf.shape, dtype=np.float16),
            ct.TensorType(name="b_cf", shape=b_cf.shape, dtype=np.float16),
            ct.TensorType(name="g_precomputed", shape=inputs_tuple[3].shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
        ]

    # PyTorch reference
    with torch.no_grad():
        pt_out = wrapper(*inputs_tuple)
    pt_g = pt_out[3]
    print(f"  PyTorch g range: [{pt_g.min():.4f}, {pt_g.max():.4f}]")

    # Trace + export
    print(f"  Tracing {fix_name}...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, inputs_tuple, check_trace=False)

    print(f"  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
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
    save_path = os.path.join(OUT_DIR, f"{fix_name}.mlpackage")
    mlmodel.save(save_path)

    # Build numpy inputs
    if extra_inputs is None:
        inp_np = {
            "mixed_qkv_pre": mixed_qkv_pre.cpu().numpy(),
            "z_cf": z_cf.cpu().numpy(),
            "b_cf": b_cf.cpu().numpy(),
            "a_cf": a_cf.cpu().numpy(),
            "conv_state": conv_state.cpu().numpy(),
        }
    else:
        inp_np = {
            "mixed_qkv_pre": mixed_qkv_pre.cpu().numpy(),
            "z_cf": z_cf.cpu().numpy(),
            "b_cf": b_cf.cpu().numpy(),
            "g_precomputed": inputs_tuple[3].cpu().numpy(),
            "conv_state": conv_state.cpu().numpy(),
        }

    results = {}
    for label, cu in [("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE)]:
        print(f"  Loading on {label}...")
        m = ct.models.MLModel(save_path, compute_units=cu)
        out = m.predict(inp_np)
        results[label] = out
        cmp(f"g (PyTorch vs {label})", pt_g, out["g"])

    if "GPU" in results and "ANE" in results:
        g_mad = cmp(f"g (GPU vs ANE)", results["GPU"]["g"], results["ANE"]["g"])
        
        # Per-head check for problem heads
        g_gpu = results["GPU"]["g"].astype(np.float32)
        g_ane = results["ANE"]["g"].astype(np.float32)
        for h in [4, 12, 13, 20, 21]:
            d = np.abs(g_gpu[0, :, h] - g_ane[0, :, h]).max()
            print(f"    Head {h}: MAD={d:.6f}  GPU=[{g_gpu[0,:8,h].min():.4f},{g_gpu[0,:8,h].max():.4f}]  "
                  f"ANE=[{g_ane[0,:8,h].min():.4f},{g_ane[0,:8,h].max():.4f}]")
        
        return g_mad
    return None


def main():
    print("=" * 80)
    print("PHASE 4: Test mitigations for g corruption on ANE")
    print("=" * 80)

    model = load_model()
    mixed_qkv_pre, z_cf, b_cf, a_cf = get_inputs(model)
    attn = model.model.layers[0].self_attn
    conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL, dtype=torch.float16)

    # ── FIX A: g as input ──
    print(f"\n{'='*80}")
    print("FIX A: Precompute g and pass as model input")
    print(f"{'='*80}")
    
    # Precompute g in PyTorch
    with torch.no_grad():
        a = a_cf.squeeze(2).transpose(1, 2)
        g_precomputed = (-attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)).to(torch.float16)
    
    wrapper_a = ConvLayoutWithGInput(attn.conv_stage, attn.layout_stage, BATCH_SIZE)
    # Pass g_precomputed instead of a_cf
    extra = (mixed_qkv_pre, z_cf, b_cf, g_precomputed, conv_state)
    mad_a = test_fix(model, mixed_qkv_pre, z_cf, b_cf, a_cf, "fix_a_g_input", wrapper_a, extra_inputs=extra)

    # ── FIX B: Separate g module ──
    print(f"\n{'='*80}")
    print("FIX B: Separate g computation in its own nn.Module")
    print(f"{'='*80}")

    g_module = GCompModule(attn.A_log.data.clone(), attn.dt_bias.data.clone())
    wrapper_b = ConvLayoutWithSeparateG(attn.conv_stage, attn.layout_stage, g_module, BATCH_SIZE)
    mad_b = test_fix(model, mixed_qkv_pre, z_cf, b_cf, a_cf, "fix_b_separate_module", wrapper_b)

    # ── FIX C: g first, then conv ──
    print(f"\n{'='*80}")
    print("FIX C: Compute g FIRST, then conv (with contiguous barrier)")
    print(f"{'='*80}")

    wrapper_c = ConvLayoutGFirst(
        attn.conv_stage, attn.layout_stage,
        attn.A_log.data.clone(), attn.dt_bias.data.clone(),
        BATCH_SIZE
    )
    mad_c = test_fix(model, mixed_qkv_pre, z_cf, b_cf, a_cf, "fix_c_g_first", wrapper_c)

    # ── Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"  Original conv+layout g GPU-vs-ANE MAD: 6.45 (from Phase 2)")
    if mad_a is not None:
        print(f"  Fix A (g as input):        MAD={mad_a:.6f}  {'FIXED!' if mad_a < 0.01 else 'STILL BROKEN'}")
    if mad_b is not None:
        print(f"  Fix B (separate module):   MAD={mad_b:.6f}  {'FIXED!' if mad_b < 0.01 else 'STILL BROKEN'}")
    if mad_c is not None:
        print(f"  Fix C (g first + barrier): MAD={mad_c:.6f}  {'FIXED!' if mad_c < 0.01 else 'STILL BROKEN'}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
