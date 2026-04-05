#!/usr/bin/env python3
"""Phase 2 diagnostic: isolate which pre-core stage causes GPU-vs-ANE divergence.

Phase 1 (diag_layer0_forensic.py) proved the delta rule itself is accurate on ANE.
The 1.74 MAD divergence comes from different inputs (q,k,v,g,beta) to the delta rule.

This script exports layer 0's proj_stage and layout_stage as separate CoreML models
to measure exactly where the ANE divergence enters.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Config constants
BATCH_SIZE = 512
CTX = 2048
HIDDEN_SIZE = 2560
NUM_K_HEADS = 16
KEY_HEAD_DIM = 128
NUM_V_HEADS = 32
VAL_HEAD_DIM = 128
KEY_DIM = NUM_K_HEADS * KEY_HEAD_DIM   # 2048
VALUE_DIM = NUM_V_HEADS * VAL_HEAD_DIM # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM     # 8192
CONV_KERNEL = 4

HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
MODEL_DIR = os.path.join(REPO_ROOT, "qwen3_5_stable_models_6chunk")
OUT_DIR = "/tmp/diag_prestages"


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
    return mad, cos


def cmp_per_head(name, a, b, n_heads, head_dim_k, head_dim_v=None):
    """Per-head comparison for q/k (B,S,n_heads,head_dim_k) or similar."""
    a_np = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else a.astype(np.float32)
    b_np = b.detach().float().cpu().numpy() if isinstance(b, torch.Tensor) else b.astype(np.float32)
    print(f"  {name} per-head:")
    # Assume shape (B, S, n_heads, dim) or (n_heads, ...)
    for h in range(min(n_heads, 32)):
        if a_np.ndim == 4:
            ah, bh = a_np[0, :, h, :], b_np[0, :, h, :]
        elif a_np.ndim == 3:
            ah, bh = a_np[0, :, h], b_np[0, :, h]
        else:
            ah, bh = a_np[h], b_np[h]
        diff = ah - bh
        mad = np.max(np.abs(diff))
        mean_abs = np.mean(np.abs(diff))
        flag = " <<<" if mad > 0.1 else ""
        print(f"    head {h:2d}: MAD={mad:.6f}  mean={mean_abs:.6f}  "
              f"|a|={np.max(np.abs(ah)):.4f}  |b|={np.max(np.abs(bh)):.4f}{flag}")


def load_model():
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    print("Loading HF model...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model


def get_hidden_states(model):
    """Get real embedded hidden states for a prompt."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    prompt = "Hello, how are you doing today?"
    ids = tok.encode(prompt, add_special_tokens=True)
    print(f"  Prompt: {prompt!r} → {len(ids)} tokens")
    input_ids = torch.zeros(1, BATCH_SIZE, dtype=torch.long)
    input_ids[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids[0]).unsqueeze(0).to(torch.float16)
    return hidden, len(ids)


# ══════════════════════════════════════════════════════════════════════════
# Test 1: Export PROJECTION STAGE (4 LUT6 Conv2d) as standalone model
# ══════════════════════════════════════════════════════════════════════════

class ProjStageWrapper(torch.nn.Module):
    """Wraps layer 0's proj_stage with input_layernorm."""
    def __init__(self, layernorm, proj_stage):
        super().__init__()
        self.layernorm = layernorm
        self.proj_stage = proj_stage

    def forward(self, hidden_states):
        x = self.layernorm(hidden_states)
        mixed_qkv_pre, z_cf, b_cf, a_cf = self.proj_stage(x)
        return mixed_qkv_pre, z_cf, b_cf, a_cf


class ConvLayoutWrapper(torch.nn.Module):
    """Wraps conv_stage + layout_stage (takes proj outputs, returns q,k,v,g,beta,z)."""
    def __init__(self, conv_stage, layout_stage, seq_len, num_k_heads, num_v_heads):
        super().__init__()
        self.conv_stage = conv_stage
        self.layout_stage = layout_stage
        self.seq_len = seq_len
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads

    def forward(self, mixed_qkv_pre, z_cf, b_cf, a_cf, conv_state):
        conv_out_cf, next_conv_state = self.conv_stage(
            mixed_qkv_pre, conv_state, expected_seq_len=self.seq_len
        )
        query, key, value, g, beta, z = self.layout_stage(
            conv_out_cf, z_cf, b_cf, a_cf,
            bsz=1, seq_len=self.seq_len,
            force_fp16_math=False,
        )
        return query, key, value, g, beta, z, next_conv_state


def export_and_test_proj_stage(model, hidden_states):
    """Export proj_stage to CoreML and compare GPU vs ANE."""
    from anemll.models.qwen3_5_model import MODEL_DTYPE
    layer0 = model.model.layers[0]
    
    wrapper = ProjStageWrapper(layer0.input_layernorm, layer0.self_attn.proj_stage)
    wrapper.eval()

    # PyTorch reference
    print("\n  Running PyTorch proj_stage reference...")
    with torch.no_grad():
        pt_mixed, pt_z, pt_b, pt_a = wrapper(hidden_states)
    print(f"    mixed_qkv_pre: {list(pt_mixed.shape)} range=[{pt_mixed.min():.3f}, {pt_mixed.max():.3f}]")
    print(f"    z_cf: {list(pt_z.shape)} range=[{pt_z.min():.3f}, {pt_z.max():.3f}]")
    print(f"    b_cf: {list(pt_b.shape)} range=[{pt_b.min():.3f}, {pt_b.max():.3f}]")
    print(f"    a_cf: {list(pt_a.shape)} range=[{pt_a.min():.3f}, {pt_a.max():.3f}]")

    # Trace and export
    print("  Tracing proj_stage...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (hidden_states,), check_trace=False)

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="mixed_qkv_pre", dtype=np.float16),
            ct.TensorType(name="z_cf", dtype=np.float16),
            ct.TensorType(name="b_cf", dtype=np.float16),
            ct.TensorType(name="a_cf", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, "proj_stage.mlpackage")
    mlmodel.save(save_path)
    print(f"  Saved to {save_path}")

    # Run on GPU and ANE
    h_np = hidden_states.cpu().numpy()

    for label, cu in [("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE)]:
        print(f"\n  Loading proj_stage on {label}...")
        t0 = time.time()
        m = ct.models.MLModel(save_path, compute_units=cu)
        print(f"  Loaded in {time.time()-t0:.1f}s")
        out = m.predict({"hidden_states": h_np})
        
        cml_mixed = out["mixed_qkv_pre"]
        cml_z = out["z_cf"]
        cml_b = out["b_cf"]
        cml_a = out["a_cf"]

        print(f"\n  PyTorch vs {label}:")
        cmp(f"mixed_qkv_pre (PyTorch vs {label})", pt_mixed, cml_mixed)
        cmp(f"z_cf (PyTorch vs {label})", pt_z, cml_z)
        cmp(f"b_cf (PyTorch vs {label})", pt_b, cml_b)
        cmp(f"a_cf (PyTorch vs {label})", pt_a, cml_a)

    return pt_mixed, pt_z, pt_b, pt_a


# ══════════════════════════════════════════════════════════════════════════
# Test 2: Export CONV + LAYOUT STAGE as standalone model
# ══════════════════════════════════════════════════════════════════════════

def export_and_test_conv_layout(model, pt_mixed, pt_z, pt_b, pt_a):
    """Export conv+layout stage to CoreML and compare GPU vs ANE."""
    layer0 = model.model.layers[0]
    attn = layer0.self_attn

    wrapper = ConvLayoutWrapper(
        attn.conv_stage, attn.layout_stage,
        seq_len=BATCH_SIZE, num_k_heads=NUM_K_HEADS, num_v_heads=NUM_V_HEADS
    )
    wrapper.eval()

    conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL, dtype=torch.float16)

    # PyTorch reference
    print("\n  Running PyTorch conv+layout reference...")
    with torch.no_grad():
        pt_q, pt_k, pt_v, pt_g, pt_beta, pt_z_out, pt_next_conv = wrapper(
            pt_mixed, pt_z, pt_b, pt_a, conv_state
        )
    print(f"    query: {list(pt_q.shape)} range=[{pt_q.min():.4f}, {pt_q.max():.4f}]")
    print(f"    key:   {list(pt_k.shape)} range=[{pt_k.min():.4f}, {pt_k.max():.4f}]")
    print(f"    value: {list(pt_v.shape)} range=[{pt_v.min():.4f}, {pt_v.max():.4f}]")
    print(f"    g:     {list(pt_g.shape)} range=[{pt_g.min():.4f}, {pt_g.max():.4f}]")
    print(f"    beta:  {list(pt_beta.shape)} range=[{pt_beta.min():.4f}, {pt_beta.max():.4f}]")

    # Trace and export
    print("  Tracing conv+layout stage...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (pt_mixed, pt_z, pt_b, pt_a, conv_state), check_trace=False)

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mixed_qkv_pre", shape=pt_mixed.shape, dtype=np.float16),
            ct.TensorType(name="z_cf", shape=pt_z.shape, dtype=np.float16),
            ct.TensorType(name="b_cf", shape=pt_b.shape, dtype=np.float16),
            ct.TensorType(name="a_cf", shape=pt_a.shape, dtype=np.float16),
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
    save_path = os.path.join(OUT_DIR, "conv_layout.mlpackage")
    mlmodel.save(save_path)
    print(f"  Saved to {save_path}")

    # Run on GPU and ANE with PyTorch-computed proj outputs as input
    inp = {
        "mixed_qkv_pre": pt_mixed.cpu().numpy(),
        "z_cf": pt_z.cpu().numpy(),
        "b_cf": pt_b.cpu().numpy(),
        "a_cf": pt_a.cpu().numpy(),
        "conv_state": conv_state.cpu().numpy(),
    }

    results = {}
    for label, cu in [("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE)]:
        print(f"\n  Loading conv+layout on {label}...")
        t0 = time.time()
        m = ct.models.MLModel(save_path, compute_units=cu)
        print(f"  Loaded in {time.time()-t0:.1f}s")
        out = m.predict(inp)
        results[label] = out

        print(f"\n  PyTorch vs {label}:")
        cmp(f"query (PyTorch vs {label})", pt_q, out["query"])
        cmp(f"key (PyTorch vs {label})", pt_k, out["key"])
        cmp(f"value (PyTorch vs {label})", pt_v, out["value"])
        cmp(f"g (PyTorch vs {label})", pt_g, out["g"])
        cmp(f"beta (PyTorch vs {label})", pt_beta, out["beta"])

    # GPU vs ANE direct comparison
    if "GPU" in results and "ANE" in results:
        print(f"\n  GPU vs ANE (conv+layout outputs):")
        for key in ["query", "key", "value", "g", "beta"]:
            cmp(f"{key} (GPU vs ANE)", results["GPU"][key], results["ANE"][key])

        # Per-head breakdown for g (the decay factor)
        print(f"\n  g per-head GPU vs ANE:")
        g_gpu = results["GPU"]["g"].astype(np.float32)
        g_ane = results["ANE"]["g"].astype(np.float32)
        for h in range(NUM_V_HEADS):
            g_diff = np.abs(g_gpu[0, :, h] - g_ane[0, :, h])
            print(f"    head {h:2d}: MAD={g_diff.max():.6f}  mean={g_diff.mean():.6f}  "
                  f"GPU_range=[{g_gpu[0,:,h].min():.4f},{g_gpu[0,:,h].max():.4f}]  "
                  f"ANE_range=[{g_ane[0,:,h].min():.4f},{g_ane[0,:,h].max():.4f}]")

    return pt_q, pt_k, pt_v, pt_g, pt_beta


# ══════════════════════════════════════════════════════════════════════════
# Test 3: Full layer 0 with EXACT PyTorch proj outputs (bypass LUT6)
# ══════════════════════════════════════════════════════════════════════════

def test_delta_with_exact_vs_perturbed_inputs(model, pt_q, pt_k, pt_v, pt_g, pt_beta, valid_len):
    """Show how small input perturbations affect rec_state through the delta rule."""
    from anemll.models.qwen3_5_model import Qwen35LinearAttention

    print("\n  Testing delta rule sensitivity to input perturbations...")

    # Apply valid_len mask
    positions = torch.arange(BATCH_SIZE, dtype=torch.int32)
    valid_mask = (positions < valid_len).to(pt_k.dtype)
    mask_bsh1 = valid_mask.reshape(1, BATCH_SIZE, 1, 1)
    mask_bsh = valid_mask.reshape(1, BATCH_SIZE, 1)
    k_masked = pt_k * mask_bsh1
    v_masked = pt_v * mask_bsh1
    beta_masked = pt_beta * mask_bsh
    g_masked = pt_g * mask_bsh

    # Reference: exact inputs
    with torch.no_grad():
        _, ref_rec = Qwen35LinearAttention._chunk_gated_delta_rule(
            pt_q, k_masked, v_masked, g=g_masked, beta=beta_masked,
            initial_state=None, output_final_state=True,
            expected_batch_size=1, expected_num_heads=NUM_V_HEADS,
            expected_seq_len=BATCH_SIZE, expected_k_dim=KEY_HEAD_DIM,
            expected_v_dim=VAL_HEAD_DIM, math_dtype=torch.float32,
        )

    # Perturbed: add small noise to g (simulating ANE projection error)
    for noise_level in [0.01, 0.05, 0.1, 0.5]:
        g_noisy = g_masked + torch.randn_like(g_masked) * noise_level
        with torch.no_grad():
            _, noisy_rec = Qwen35LinearAttention._chunk_gated_delta_rule(
                pt_q, k_masked, v_masked, g=g_noisy, beta=beta_masked,
                initial_state=None, output_final_state=True,
                expected_batch_size=1, expected_num_heads=NUM_V_HEADS,
                expected_seq_len=BATCH_SIZE, expected_k_dim=KEY_HEAD_DIM,
                expected_v_dim=VAL_HEAD_DIM, math_dtype=torch.float32,
            )
        cmp(f"rec_state with g noise={noise_level:.2f}", ref_rec, noisy_rec)

    # Perturbed: add noise to key
    for noise_level in [0.01, 0.05, 0.1]:
        k_noisy = k_masked + torch.randn_like(k_masked) * noise_level
        with torch.no_grad():
            _, noisy_rec = Qwen35LinearAttention._chunk_gated_delta_rule(
                pt_q, k_noisy, v_masked, g=g_masked, beta=beta_masked,
                initial_state=None, output_final_state=True,
                expected_batch_size=1, expected_num_heads=NUM_V_HEADS,
                expected_seq_len=BATCH_SIZE, expected_k_dim=KEY_HEAD_DIM,
                expected_v_dim=VAL_HEAD_DIM, math_dtype=torch.float32,
            )
        cmp(f"rec_state with key noise={noise_level:.2f}", ref_rec, noisy_rec)


def main():
    print("=" * 80)
    print("PHASE 2 DIAGNOSTIC: Isolate pre-core stage divergence")
    print("=" * 80)

    model = load_model()
    hidden, valid_len = get_hidden_states(model)

    # Test 1: Projection stage
    print(f"\n{'='*80}")
    print("TEST 1: Projection Stage (input_layernorm + 4× LUT6 Conv2d)")
    print(f"{'='*80}")
    pt_mixed, pt_z, pt_b, pt_a = export_and_test_proj_stage(model, hidden)

    # Test 2: Conv + Layout stage
    print(f"\n{'='*80}")
    print("TEST 2: Conv + Layout Stage (depthwise conv + sigmoid/softplus/exp)")
    print(f"{'='*80}")
    pt_q, pt_k, pt_v, pt_g, pt_beta = export_and_test_conv_layout(
        model, pt_mixed, pt_z, pt_b, pt_a
    )

    # Test 3: Delta rule sensitivity analysis
    print(f"\n{'='*80}")
    print("TEST 3: Delta Rule Sensitivity (how much input noise → rec_state error)")
    print(f"{'='*80}")
    test_delta_with_exact_vs_perturbed_inputs(model, pt_q, pt_k, pt_v, pt_g, pt_beta, valid_len)

    print(f"\n[Done]")


if __name__ == "__main__":
    main()
