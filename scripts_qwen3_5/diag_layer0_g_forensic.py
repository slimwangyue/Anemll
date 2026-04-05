#!/usr/bin/env python3
"""Phase 3: Isolate the exact operation causing g=0 on ANE.

Previous findings:
  - g = -exp(A_log) * softplus(a + dt_bias) 
  - GPU computes correctly, ANE zeroes out specific heads
  - This script tests: is it softplus? exp? the multiply? or the full chain?
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
OUT_DIR = "/tmp/diag_g_computation"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_V_HEADS = 32
BATCH_SIZE = 512


def cmp(name, a, b):
    a_np = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else a.astype(np.float32)
    b_np = b.detach().float().cpu().numpy() if isinstance(b, torch.Tensor) else b.astype(np.float32)
    diff = a_np - b_np
    mad = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    a_f, b_f = a_np.flatten(), b_np.flatten()
    dot = np.dot(a_f, b_f)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    cos = dot / (na * nb + 1e-12)
    print(f"  {name:55s}  MAD={mad:.6f}  mean={mean_abs:.6f}  cos={cos:.6f}  "
          f"|a|={np.max(np.abs(a_np)):.4f}  |b|={np.max(np.abs(b_np)):.4f}")
    return mad


def load_layer0_params():
    """Load A_log and dt_bias for layer 0."""
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    print("Loading model for parameters...")
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = 2048
    cfg.state_length = 2048
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()

    attn = model.model.layers[0].self_attn
    A_log = attn.A_log.data.clone()
    dt_bias = attn.dt_bias.data.clone()
    return model, A_log, dt_bias


def print_head_params(A_log, dt_bias):
    """Print A_log and dt_bias for each head."""
    print("\n  Layer 0 parameters per head:")
    print(f"  {'Head':>4s}  {'A_log':>10s}  {'exp(A_log)':>12s}  {'dt_bias':>10s}  {'a+dt':>10s}")
    for h in range(NUM_V_HEADS):
        a = A_log[h].item()
        ea = np.exp(a)
        dt = dt_bias[h].item()
        print(f"  {h:4d}  {a:10.5f}  {ea:12.8f}  {dt:10.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Test A: Just softplus on a range of values
# ══════════════════════════════════════════════════════════════════════════

class SoftplusModel(torch.nn.Module):
    def forward(self, x):
        return F.softplus(x)


class GComputeModel(torch.nn.Module):
    """g = -exp(A_log) * softplus(a + dt_bias), matching layout_stage exactly."""
    def __init__(self, A_log, dt_bias):
        super().__init__()
        self.register_buffer("A_log", A_log)
        self.register_buffer("dt_bias", dt_bias)

    def forward(self, a):
        # Match the non-force_fp16 path in layout_stage:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        return g


class GComputeFP16Model(torch.nn.Module):
    """Same as above but explicitly fp16 math (what ANE probably does)."""
    def __init__(self, A_log, dt_bias):
        super().__init__()
        self.register_buffer("A_log", A_log.to(torch.float16))
        self.register_buffer("dt_bias", dt_bias.to(torch.float16))

    def forward(self, a):
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        return g


def test_softplus_on_ane():
    """Test softplus accuracy on ANE for various input ranges."""
    print("\n" + "=" * 80)
    print("TEST A: Softplus accuracy on ANE")
    print("=" * 80)

    wrapper = SoftplusModel()
    wrapper.eval()
    
    # Test with a range of values
    x = torch.linspace(-20, 20, 1024).reshape(1, 1024).to(torch.float16)
    
    # PyTorch reference
    with torch.no_grad():
        pt_ref = wrapper(x)
    
    print("  Tracing softplus...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (x,), check_trace=False)
    
    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    save_path = os.path.join(OUT_DIR, "softplus.mlpackage")
    mlmodel.save(save_path)
    
    x_np = x.cpu().numpy()
    pt_np = pt_ref.cpu().numpy()
    
    for label, cu in [("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE)]:
        m = ct.models.MLModel(save_path, compute_units=cu)
        out = m.predict({"x": x_np})["y"]
        
        diff = np.abs(pt_np.astype(np.float32) - out.astype(np.float32))
        print(f"\n  softplus PyTorch vs {label}:")
        
        # Show by input range
        x_f = x_np.flatten().astype(np.float32)
        out_f = out.flatten().astype(np.float32)
        pt_f = pt_np.flatten().astype(np.float32)
        diff_f = diff.flatten()
        
        for lo, hi in [(-20, -15), (-15, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20)]:
            mask = (x_f >= lo) & (x_f < hi)
            if mask.sum() > 0:
                d_max = diff_f[mask].max()
                d_mean = diff_f[mask].mean()
                pt_range = f"[{pt_f[mask].min():.6f}, {pt_f[mask].max():.6f}]"
                out_range = f"[{out_f[mask].min():.6f}, {out_f[mask].max():.6f}]"
                print(f"    x in [{lo:3d},{hi:3d}):  PyTorch={pt_range:30s}  {label}={out_range:30s}  MAD={d_max:.8f}")


def test_g_computation(model, A_log, dt_bias):
    """Test the full g computation on GPU vs ANE."""
    print("\n" + "=" * 80)
    print("TEST B: Full g computation (-exp(A_log) * softplus(a + dt_bias))")
    print("=" * 80)

    # Get real 'a' values from the model
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    ids = tok.encode("Hello, how are you doing today?", add_special_tokens=True)
    input_ids = torch.zeros(1, BATCH_SIZE, dtype=torch.long)
    input_ids[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids[0]).unsqueeze(0).to(torch.float16)
        # Run input_layernorm + in_proj_a
        layer0 = model.model.layers[0]
        attn = layer0.self_attn
        x = layer0.input_layernorm(hidden)
        a_cf = attn._conv2d_proj_cf(attn.in_proj_a, attn._to_channels_first_4d(x))
        # channels-first to channels-last: (1, 32, 1, 512) → (1, 512, 32)
        a = a_cf.squeeze(2).transpose(1, 2)

    print(f"\n  Real 'a' values: shape={list(a.shape)}, range=[{a.min():.4f}, {a.max():.4f}]")

    # Print a + dt_bias per head
    print(f"\n  a + dt_bias per head (first 8 tokens, all 32 heads):")
    a_np = a[0].float().cpu().numpy()
    dt_np = dt_bias.float().cpu().numpy()
    for h in range(NUM_V_HEADS):
        apdt = a_np[:8, h] + dt_np[h]
        sp = np.log1p(np.exp(apdt))
        g_ref = -np.exp(float(A_log[h])) * sp
        print(f"    head {h:2d}: a+dt_bias={apdt[:4]}...  softplus={sp[:4]}...  g_ref={g_ref[:4]}...")

    # ── Export the g computation model ──
    g_model = GComputeModel(A_log, dt_bias)
    g_model.eval()

    with torch.no_grad():
        pt_g = g_model(a)
    print(f"\n  PyTorch g: range=[{pt_g.min():.4f}, {pt_g.max():.4f}]")

    # Trace and export
    print("  Tracing g computation...")
    with torch.no_grad():
        traced = torch.jit.trace(g_model, (a,), check_trace=False)

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="a", shape=a.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="g", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    save_path = os.path.join(OUT_DIR, "g_compute.mlpackage")
    mlmodel.save(save_path)

    a_np_input = a.cpu().numpy()
    results = {}
    for label, cu in [("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE)]:
        m = ct.models.MLModel(save_path, compute_units=cu)
        out = m.predict({"a": a_np_input})["g"]
        results[label] = out
        print(f"\n  PyTorch vs {label}:")
        cmp(f"g (PyTorch vs {label})", pt_g, out)
        # Per head
        g_out = out.astype(np.float32)
        g_pt = pt_g[0].float().cpu().numpy()
        for h in range(NUM_V_HEADS):
            d = np.abs(g_pt[:, h] - g_out[0, :, h])
            flag = " <<<" if d.max() > 0.1 else ""
            print(f"    head {h:2d}: MAD={d.max():.6f}  PyTorch=[{g_pt[:8,h].min():.4f},{g_pt[:8,h].max():.4f}]  "
                  f"{label}=[{g_out[0,:8,h].min():.4f},{g_out[0,:8,h].max():.4f}]{flag}")

    # GPU vs ANE
    if "GPU" in results and "ANE" in results:
        print(f"\n  GPU vs ANE:")
        cmp("g (GPU vs ANE)", results["GPU"], results["ANE"])

    # ── Also test the fp16-only version ──
    print("\n" + "=" * 80)
    print("TEST C: g computation with fp16 math (no .float() cast)")
    print("=" * 80)

    g_fp16_model = GComputeFP16Model(A_log, dt_bias)
    g_fp16_model.eval()
    
    a_fp16 = a.to(torch.float16)
    with torch.no_grad():
        pt_g_fp16 = g_fp16_model(a_fp16)
    print(f"  PyTorch fp16 g: range=[{pt_g_fp16.min():.4f}, {pt_g_fp16.max():.4f}]")
    
    # Compare fp32 vs fp16 PyTorch
    cmp("g fp32 vs fp16 (PyTorch)", pt_g, pt_g_fp16)

    # Per head fp32 vs fp16
    g_pt32 = pt_g[0].float().cpu().numpy()
    g_pt16 = pt_g_fp16[0].float().cpu().numpy()
    print(f"\n  Per-head fp32 vs fp16 (PyTorch):")
    for h in range(NUM_V_HEADS):
        d = np.abs(g_pt32[:, h] - g_pt16[:, h])
        flag = " <<<" if d.max() > 0.1 else ""
        print(f"    head {h:2d}: MAD={d.max():.6f}  fp32=[{g_pt32[:8,h].min():.4f},{g_pt32[:8,h].max():.4f}]  "
              f"fp16=[{g_pt16[:8,h].min():.4f},{g_pt16[:8,h].max():.4f}]{flag}")

    # Export the fp16 version
    print("\n  Tracing fp16 g computation...")
    with torch.no_grad():
        traced_fp16 = torch.jit.trace(g_fp16_model, (a_fp16,), check_trace=False)

    print("  Converting to CoreML...")
    mlmodel_fp16 = ct.convert(
        traced_fp16,
        inputs=[ct.TensorType(name="a", shape=a_fp16.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="g", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    save_path_fp16 = os.path.join(OUT_DIR, "g_compute_fp16.mlpackage")
    mlmodel_fp16.save(save_path_fp16)

    for label, cu in [("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE)]:
        m = ct.models.MLModel(save_path_fp16, compute_units=cu)
        out = m.predict({"a": a_fp16.cpu().numpy()})["g"]
        print(f"\n  fp16 model: PyTorch-fp16 vs {label}:")
        cmp(f"g fp16 (PyTorch vs {label})", pt_g_fp16, out)
        # Key heads
        g_out = out.astype(np.float32)
        for h in [4, 12, 13, 20, 21]:
            d = np.abs(g_pt16[:, h] - g_out[0, :, h])
            print(f"    head {h:2d}: MAD={d.max():.6f}  PyTorch-fp16=[{g_pt16[:8,h].min():.4f},{g_pt16[:8,h].max():.4f}]  "
                  f"{label}=[{g_out[0,:8,h].min():.4f},{g_out[0,:8,h].max():.4f}]")


def main():
    print("=" * 80)
    print("PHASE 3: g computation forensics")
    print("=" * 80)

    model, A_log, dt_bias = load_layer0_params()
    print_head_params(A_log, dt_bias)

    test_softplus_on_ane()
    test_g_computation(model, A_log, dt_bias)

    print("\n[Done]")


if __name__ == "__main__":
    main()
