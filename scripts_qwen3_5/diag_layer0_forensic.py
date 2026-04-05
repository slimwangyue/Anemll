#!/usr/bin/env python3
"""Forensic diagnostic: isolate layer 0 linear attention rec_state GPU-vs-ANE divergence.

Strategy:
  1. Load HF model in PyTorch, embed a real prompt, extract layer 0 sub-stage intermediates
  2. Run chunk0 prefill model on CPU_AND_GPU and CPU_AND_NE with same inputs
  3. Compare rec_state[0] from: PyTorch-fp32, PyTorch-fp16, CoreML-GPU, CoreML-ANE
  4. Narrow divergence to specific sub-stage(s) inside layer 0
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

# ── project paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Inline config constants (from scripts_qwen3_5/config.py) to avoid shadow import
BATCH_SIZE = 512
CTX = 2048
NUM_CHUNKS = 6
FFN_LABEL = "LUT6"

HF_MODEL   = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
MODEL_DIR   = os.path.join(REPO_ROOT, "qwen3_5_stable_models_6chunk")

# ── model config ───────────────────────────────────────────────────────
NUM_K_HEADS   = 16
KEY_HEAD_DIM  = 128
NUM_V_HEADS   = 32
VAL_HEAD_DIM  = 128
HIDDEN_SIZE   = 2560
KEY_DIM       = NUM_K_HEADS * KEY_HEAD_DIM   # 2048
VALUE_DIM     = NUM_V_HEADS * VAL_HEAD_DIM   # 4096
CONV_DIM      = KEY_DIM * 2 + VALUE_DIM      # 8192
CONV_KERNEL   = 4
CHUNK_SIZE    = 16  # delta-rule chunk size

# ANE state reshape
ANE_STATE_MAX = 1024
ANE_CONV_GROUP = (CONV_DIM + ANE_STATE_MAX - 1) // ANE_STATE_MAX  # 8
ANE_CONV_DIM1 = CONV_DIM // ANE_CONV_GROUP                         # 1024
ANE_CONV_DIM2 = CONV_KERNEL * ANE_CONV_GROUP                       # 32

# Chunk0: layers 0-5 = [lin, lin, lin, full, lin, lin]
# 6 state slots total (one per layer in chunk)
CHUNK0_LAYERS = 6
CHUNK0_LIN_PATTERN = [True, True, True, False, True, True]  # which are linear

# ──────────────────────────────────────────────────────────────────────
def cmp(name, a, b, head_dim=None):
    """Compare two tensors and print statistics."""
    a_np = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else a.astype(np.float32)
    b_np = b.detach().float().cpu().numpy() if isinstance(b, torch.Tensor) else b.astype(np.float32)
    diff = a_np - b_np
    mad = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    # Cosine similarity (flattened)
    a_f, b_f = a_np.flatten(), b_np.flatten()
    dot = np.dot(a_f, b_f)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    cos = dot / (na * nb + 1e-12)
    print(f"  {name:45s}  MAD={mad:.6f}  mean_abs={mean_abs:.6f}  cos={cos:.6f}  "
          f"|a|_max={np.max(np.abs(a_np)):.4f}  |b|_max={np.max(np.abs(b_np)):.4f}")
    return mad, cos


def cmp_per_head(name, a, b, n_heads):
    """Per-head comparison for rec_state shaped (1, n_heads, K, V) or (n_heads, K, V)."""
    a_np = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else a.astype(np.float32)
    b_np = b.detach().float().cpu().numpy() if isinstance(b, torch.Tensor) else b.astype(np.float32)
    if a_np.ndim == 4:
        a_np, b_np = a_np[0], b_np[0]
    print(f"  {name} per-head:")
    for h in range(n_heads):
        ah, bh = a_np[h], b_np[h]
        diff = ah - bh
        mad = np.max(np.abs(diff))
        mean_abs = np.mean(np.abs(diff))
        dot = np.dot(ah.flatten(), bh.flatten())
        na, nb = np.linalg.norm(ah.flatten()), np.linalg.norm(bh.flatten())
        cos = dot / (na * nb + 1e-12)
        flag = " <<<" if mad > 0.5 else ""
        print(f"    head {h:2d}: MAD={mad:.4f}  mean={mean_abs:.6f}  cos={cos:.6f}  "
              f"|a|={np.max(np.abs(ah)):.3f}  |b|={np.max(np.abs(bh)):.3f}{flag}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: PyTorch reference — run layer 0 sub-stages with fp32 & fp16
# ══════════════════════════════════════════════════════════════════════

def load_pytorch_model():
    """Load the HF Qwen3.5-4B model into PyTorch."""
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
    print("[Phase 1] Loading HF model...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model


def embed_prompt(model, prompt="Hello, how are you doing today?"):
    """Tokenize and embed a prompt, returning (token_ids, hidden_states_fp16)."""
    # Use the tokenizer to get token IDs
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    ids = tok.encode(prompt, add_special_tokens=True)
    print(f"  Prompt: {prompt!r} → {len(ids)} tokens")

    # Embed
    input_ids = torch.zeros(1, BATCH_SIZE, dtype=torch.long)
    input_ids[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids[0])  # (BATCH_SIZE, HIDDEN_SIZE)
        hidden = hidden.unsqueeze(0).to(torch.float16)    # (1, BATCH_SIZE, HIDDEN_SIZE)
    return ids, hidden, len(ids)


def run_layer0_substages(model, hidden_states, valid_len, math_dtype_label="fp32"):
    """Run layer 0's linear attention sub-stages manually, return intermediates."""
    layer0 = model.model.layers[0]
    attn = layer0.self_attn

    # input_layernorm
    x = layer0.input_layernorm(hidden_states)

    # Set expected dimensions for tracing
    attn.export_expected_batch_size = 1
    attn.export_expected_seq_len = BATCH_SIZE
    bsz, seq_len = 1, BATCH_SIZE

    force_fp16 = (math_dtype_label == "fp16")

    with torch.no_grad():
        # Stage 1: proj
        mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(x)

        # Stage 2: conv
        conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL, dtype=torch.float16)
        conv_out_cf, next_conv_state = attn.conv_stage(mixed_qkv_pre, conv_state, expected_seq_len=seq_len)

        # Stage 3: layout
        query, key, value, g, beta, z = attn.layout_stage(
            conv_out_cf, z_cf, b_cf, a_cf, bsz, seq_len, force_fp16_math=force_fp16
        )

        # Apply valid_len mask (same as _forward_prefill_export_impl)
        positions = torch.arange(seq_len, dtype=torch.int32)
        valid_mask = (positions < valid_len).to(key.dtype)
        mask_bsh1 = valid_mask.reshape(1, seq_len, 1, 1)
        mask_bsh = valid_mask.reshape(1, seq_len, 1)
        key_masked = key * mask_bsh1
        value_masked = value * mask_bsh1
        beta_masked = beta * mask_bsh
        g_masked = g * mask_bsh

        # Stage 4a: core delta rule
        rec_state_init = torch.zeros(1, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM, dtype=torch.float16)
        math_dtype = torch.float16 if force_fp16 else torch.float32

        from anemll.models.qwen3_5_model import Qwen35LinearAttention
        core, rec_state = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key_masked, value_masked,
            g=g_masked,
            beta=beta_masked,
            initial_state=None,  # fresh prompt, no prior state
            output_final_state=True,
            expected_batch_size=bsz,
            expected_num_heads=NUM_V_HEADS,
            expected_seq_len=seq_len,
            expected_k_dim=KEY_HEAD_DIM,
            expected_v_dim=VAL_HEAD_DIM,
            math_dtype=math_dtype,
        )

    return {
        "x_normed": x,
        "mixed_qkv_pre": mixed_qkv_pre,
        "z_cf": z_cf,
        "b_cf": b_cf,
        "a_cf": a_cf,
        "conv_out_cf": conv_out_cf,
        "next_conv_state": next_conv_state,
        "query": query,
        "key": key,
        "value": value,
        "g": g,
        "beta": beta,
        "z": z,
        "key_masked": key_masked,
        "value_masked": value_masked,
        "g_masked": g_masked,
        "beta_masked": beta_masked,
        "core": core,
        "rec_state": rec_state,
    }


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: CoreML chunk0 — run on GPU and ANE
# ══════════════════════════════════════════════════════════════════════

def run_coreml_chunk0(hidden_states_np, valid_len_int, compute_unit, label="GPU"):
    """Run chunk0 prefill on CoreML, return rec_state for layer 0."""
    import coremltools as ct

    model_path = os.path.join(MODEL_DIR, "prefill_LUT6_chunk0.mlpackage")
    print(f"  Loading chunk0 prefill on {label} ({compute_unit})...")
    t0 = time.time()
    mlmodel = ct.models.MLModel(model_path, compute_units=compute_unit)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Build inputs matching chat_server conventions
    seq_len = BATCH_SIZE
    pos_ids = np.zeros(seq_len, dtype=np.int32)
    pos_ids[:valid_len_int] = np.arange(valid_len_int, dtype=np.int32)

    # Causal mask: (1, 1, seq_len, CTX)
    mask = np.full((1, 1, seq_len, CTX), -65504.0, dtype=np.float16)
    for i in range(valid_len_int):
        mask[0, 0, i, :i+1] = 0.0

    cur_pos = np.array([0], dtype=np.int32)
    lin_conv = np.zeros((CHUNK0_LAYERS, ANE_CONV_DIM1, ANE_CONV_DIM2), dtype=np.float16)
    lin_rec = np.zeros((CHUNK0_LAYERS, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM), dtype=np.float16)
    valid_len_arr = np.array([valid_len_int], dtype=np.int32)

    inp = {
        "hidden_states": hidden_states_np,
        "position_ids": pos_ids,
        "causal_mask": mask,
        "current_pos": cur_pos,
        "linear_conv_state": lin_conv,
        "linear_recurrent_state": lin_rec,
        "valid_len": valid_len_arr,
    }

    # Create state for KV cache
    state = mlmodel.make_state()

    print(f"  Running prediction ({label})...")
    t0 = time.time()
    out = mlmodel.predict(inp, state=state)
    print(f"  Prediction done in {time.time()-t0:.1f}s")

    rec_out = out["linear_recurrent_state_out"]  # (6, 32, 128, 128)
    conv_out = out["linear_conv_state_out"]       # (6, 1024, 32)
    hidden_out = out["output_hidden_states"]      # (1, 512, 2560)

    return {
        "rec_state_all": rec_out,
        "rec_state_layer0": rec_out[0],  # (32, 128, 128)
        "conv_state_all": conv_out,
        "conv_state_layer0": conv_out[0],
        "hidden_out": hidden_out,
    }


# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Export & test CORE-ONLY sub-model (delta rule, no weights)
# ══════════════════════════════════════════════════════════════════════

def export_core_delta_model(q, k, v, g, beta, output_dir="/tmp/diag_core_delta"):
    """Export the chunk_gated_delta_rule as a standalone CoreML model.

    This isolates the pure-math recurrence from LUT6 projections.
    Inputs: q, k, v, g, beta (from PyTorch layout_stage)
    Outputs: core_attn, next_rec_state
    """
    import coremltools as ct
    from anemll.models.qwen3_5_model import Qwen35LinearAttention

    class CoreDeltaWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, query, key, value, g, beta):
            core, rec_state = Qwen35LinearAttention._chunk_gated_delta_rule(
                query, key, value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                expected_batch_size=1,
                expected_num_heads=NUM_V_HEADS,
                expected_seq_len=BATCH_SIZE,
                expected_k_dim=KEY_HEAD_DIM,
                expected_v_dim=VAL_HEAD_DIM,
                math_dtype=torch.float32,
            )
            return core, rec_state

    wrapper = CoreDeltaWrapper()
    wrapper.eval()

    # trace with real tensors
    print("  Tracing core delta rule...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (q, k, v, g, beta), check_trace=False)

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="query",  shape=q.shape, dtype=np.float16),
            ct.TensorType(name="key",    shape=k.shape, dtype=np.float16),
            ct.TensorType(name="value",  shape=v.shape, dtype=np.float16),
            ct.TensorType(name="g",      shape=g.shape, dtype=np.float16),
            ct.TensorType(name="beta",   shape=beta.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="core_attn", dtype=np.float16),
            ct.TensorType(name="rec_state", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "core_delta.mlpackage")
    mlmodel.save(save_path)
    print(f"  Saved to {save_path}")
    return save_path


def run_core_delta_coreml(model_path, q_np, k_np, v_np, g_np, beta_np, compute_unit, label="GPU"):
    """Run the standalone core delta model on specified compute unit."""
    import coremltools as ct
    print(f"  Loading core delta on {label}...")
    mlmodel = ct.models.MLModel(model_path, compute_units=compute_unit)
    inp = {
        "query": q_np, "key": k_np, "value": v_np,
        "g": g_np, "beta": beta_np,
    }
    print(f"  Running core delta ({label})...")
    out = mlmodel.predict(inp)
    return out["rec_state"]


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    import coremltools as ct

    print("=" * 80)
    print("FORENSIC DIAGNOSTIC: Layer 0 Linear Attention Rec-State GPU vs ANE")
    print("=" * 80)

    prompt = "Hello, how are you doing today?"

    # ── Phase 1: PyTorch reference ──
    model = load_pytorch_model()
    token_ids, hidden, valid_len = embed_prompt(model, prompt)

    print(f"\n[Phase 1a] Running layer 0 sub-stages with fp32 math...")
    inter_fp32 = run_layer0_substages(model, hidden, valid_len, "fp32")

    print(f"\n[Phase 1b] Running layer 0 sub-stages with fp16 math...")
    inter_fp16 = run_layer0_substages(model, hidden, valid_len, "fp16")

    print(f"\n--- PyTorch fp32 vs fp16 intermediates ---")
    cmp("proj mixed_qkv_pre", inter_fp32["mixed_qkv_pre"], inter_fp16["mixed_qkv_pre"])
    cmp("layout query", inter_fp32["query"], inter_fp16["query"])
    cmp("layout key", inter_fp32["key"], inter_fp16["key"])
    cmp("layout value", inter_fp32["value"], inter_fp16["value"])
    cmp("layout g", inter_fp32["g"], inter_fp16["g"])
    cmp("layout beta", inter_fp32["beta"], inter_fp16["beta"])
    cmp("core output", inter_fp32["core"], inter_fp16["core"])
    cmp("rec_state (fp32 vs fp16 math)", inter_fp32["rec_state"], inter_fp16["rec_state"])
    cmp_per_head("rec_state fp32-vs-fp16", inter_fp32["rec_state"], inter_fp16["rec_state"], NUM_V_HEADS)

    # Print PyTorch rec_state magnitudes
    rs32 = inter_fp32["rec_state"].float().cpu().numpy()
    rs16 = inter_fp16["rec_state"].float().cpu().numpy()
    print(f"\n  PyTorch fp32 rec_state: min={rs32.min():.4f}  max={rs32.max():.4f}  mean_abs={np.mean(np.abs(rs32)):.6f}")
    print(f"  PyTorch fp16 rec_state: min={rs16.min():.4f}  max={rs16.max():.4f}  mean_abs={np.mean(np.abs(rs16)):.6f}")

    # ── Phase 2: CoreML chunk0 ──
    hidden_np = hidden.cpu().numpy()

    print(f"\n[Phase 2a] CoreML chunk0 on GPU...")
    try:
        gpu_results = run_coreml_chunk0(hidden_np, valid_len, ct.ComputeUnit.CPU_AND_GPU, "GPU")
    except Exception as e:
        print(f"  GPU failed: {e}")
        gpu_results = None

    print(f"\n[Phase 2b] CoreML chunk0 on ANE...")
    try:
        ane_results = run_coreml_chunk0(hidden_np, valid_len, ct.ComputeUnit.CPU_AND_NE, "ANE")
    except Exception as e:
        print(f"  ANE failed: {e}")
        ane_results = None

    # ── Phase 3: Full comparison matrix ──
    print(f"\n{'='*80}")
    print("COMPARISON MATRIX: rec_state layer 0 (shape: 32 × 128 × 128)")
    print(f"{'='*80}")

    pt_fp32_rec = inter_fp32["rec_state"][0].float().cpu().numpy()  # (32, 128, 128)
    pt_fp16_rec = inter_fp16["rec_state"][0].float().cpu().numpy()

    comparisons = []
    comparisons.append(("PyTorch fp32 vs fp16", pt_fp32_rec, pt_fp16_rec))

    if gpu_results is not None:
        gpu_rec = gpu_results["rec_state_layer0"].astype(np.float32)
        comparisons.append(("PyTorch fp32 vs GPU", pt_fp32_rec, gpu_rec))
        comparisons.append(("PyTorch fp16 vs GPU", pt_fp16_rec, gpu_rec))

    if ane_results is not None:
        ane_rec = ane_results["rec_state_layer0"].astype(np.float32)
        comparisons.append(("PyTorch fp32 vs ANE", pt_fp32_rec, ane_rec))
        comparisons.append(("PyTorch fp16 vs ANE", pt_fp16_rec, ane_rec))

    if gpu_results is not None and ane_results is not None:
        comparisons.append(("GPU vs ANE", gpu_rec, ane_rec))

    print()
    for name, a, b in comparisons:
        cmp(name, a, b)

    # Per-head breakdown for GPU vs ANE
    if gpu_results is not None and ane_results is not None:
        print(f"\n{'='*80}")
        print("PER-HEAD BREAKDOWN: GPU vs ANE rec_state layer 0")
        print(f"{'='*80}")
        cmp_per_head("GPU vs ANE", gpu_rec, ane_rec, NUM_V_HEADS)

        # Also compare all chunk0 layers
        print(f"\n--- All chunk0 layers rec_state: GPU vs ANE ---")
        for layer_idx in range(CHUNK0_LAYERS):
            if not CHUNK0_LIN_PATTERN[layer_idx]:
                print(f"  Layer {layer_idx}: FULL ATTENTION (skip)")
                continue
            gpu_l = gpu_results["rec_state_all"][layer_idx].astype(np.float32)
            ane_l = ane_results["rec_state_all"][layer_idx].astype(np.float32)
            cmp(f"Layer {layer_idx} rec_state (GPU vs ANE)", gpu_l, ane_l)

        # Conv state too
        print(f"\n--- All chunk0 layers conv_state: GPU vs ANE ---")
        for layer_idx in range(CHUNK0_LAYERS):
            if not CHUNK0_LIN_PATTERN[layer_idx]:
                print(f"  Layer {layer_idx}: FULL ATTENTION (skip)")
                continue
            gpu_c = gpu_results["conv_state_all"][layer_idx].astype(np.float32)
            ane_c = ane_results["conv_state_all"][layer_idx].astype(np.float32)
            cmp(f"Layer {layer_idx} conv_state (GPU vs ANE)", gpu_c, ane_c)

    # ── Phase 4: Export and test core delta rule in isolation ──
    print(f"\n{'='*80}")
    print("PHASE 4: Isolated core delta rule (no LUT6 weights)")
    print(f"{'='*80}")

    # Use the fp32-computed q,k,v,g,beta from PyTorch as inputs
    q_pt = inter_fp32["key_masked"]   # Wait - we need query, not key
    q_pt = inter_fp32["query"]
    k_pt = inter_fp32["key_masked"]
    v_pt = inter_fp32["value_masked"]
    g_pt = inter_fp32["g_masked"]
    beta_pt = inter_fp32["beta_masked"]

    print(f"  Input shapes: q={list(q_pt.shape)} k={list(k_pt.shape)} v={list(v_pt.shape)} "
          f"g={list(g_pt.shape)} beta={list(beta_pt.shape)}")

    try:
        core_model_path = export_core_delta_model(q_pt, k_pt, v_pt, g_pt, beta_pt)

        q_np = q_pt.cpu().numpy().astype(np.float16)
        k_np = k_pt.cpu().numpy().astype(np.float16)
        v_np = v_pt.cpu().numpy().astype(np.float16)
        g_np = g_pt.cpu().numpy().astype(np.float16)
        beta_np = beta_pt.cpu().numpy().astype(np.float16)

        core_gpu_rec = run_core_delta_coreml(
            core_model_path, q_np, k_np, v_np, g_np, beta_np,
            ct.ComputeUnit.CPU_AND_GPU, "GPU"
        )
        core_ane_rec = run_core_delta_coreml(
            core_model_path, q_np, k_np, v_np, g_np, beta_np,
            ct.ComputeUnit.CPU_AND_NE, "ANE"
        )

        print(f"\n--- Isolated core delta rule: GPU vs ANE ---")
        cmp("Core-only rec_state (GPU vs ANE)", core_gpu_rec, core_ane_rec)
        cmp_per_head("Core-only GPU vs ANE", core_gpu_rec, core_ane_rec, NUM_V_HEADS)

        # Also compare to PyTorch fp32 and fp16
        core_gpu_np = core_gpu_rec.astype(np.float32)
        core_ane_np = core_ane_rec.astype(np.float32)
        print(f"\n--- Isolated core vs PyTorch ---")
        cmp("PyTorch fp32 vs Core-GPU", pt_fp32_rec, core_gpu_np)
        cmp("PyTorch fp32 vs Core-ANE", pt_fp32_rec, core_ane_np)
        cmp("PyTorch fp16 vs Core-GPU", pt_fp16_rec, core_gpu_np)
        cmp("PyTorch fp16 vs Core-ANE", pt_fp16_rec, core_ane_np)

        # KEY DIAGNOSTIC: if Core-GPU ≈ Core-ANE, divergence is in proj/conv/layout (inputs differ)
        # if Core-GPU ≠ Core-ANE, divergence is in the delta rule operators themselves
        print(f"\n{'='*80}")
        print("INTERPRETATION:")
        core_delta_mad, core_delta_cos = cmp("Core-only GPU vs ANE (repeated)",
                                              core_gpu_rec, core_ane_rec)
        if gpu_results is not None and ane_results is not None:
            full_mad, full_cos = cmp("Full chunk0 GPU vs ANE (repeated)", gpu_rec, ane_rec)
            ratio = core_delta_mad / (full_mad + 1e-12)
            print(f"\n  Core-only/Full ratio: {ratio:.3f}")
            if ratio > 0.5:
                print("  → DIVERGENCE IS IN THE DELTA RULE ITSELF (core operators: l2norm, exp, matmul)")
            elif ratio < 0.1:
                print("  → DIVERGENCE IS IN THE INPUT STAGES (LUT6 projections/conv/layout)")
            else:
                print("  → DIVERGENCE IS MIXED (both input errors and core operator errors)")
        print(f"{'='*80}")

    except Exception as e:
        print(f"  Core delta export/test failed: {e}")
        import traceback; traceback.print_exc()

    # ── Phase 5: g statistics (decay factors) ──
    print(f"\n{'='*80}")
    print("PHASE 5: Decay factor (g) statistics at layer 0")
    print(f"{'='*80}")
    g_full = inter_fp32["g_masked"][0, :valid_len].float().cpu().numpy()  # (valid_len, 32)
    print(f"  g shape: {g_full.shape}")
    print(f"  g range: [{g_full.min():.4f}, {g_full.max():.4f}]")
    print(f"  g mean abs: {np.mean(np.abs(g_full)):.4f}")
    # exp(g) is the decay per step — if exp(g) ≈ 1.0, state persists; if ≈ 0, state forgets
    exp_g = np.exp(g_full.astype(np.float64))
    print(f"  exp(g) range: [{exp_g.min():.6f}, {exp_g.max():.6f}]")
    print(f"  exp(g) mean: {exp_g.mean():.6f}")
    # Per-head
    for h in range(NUM_V_HEADS):
        eg = exp_g[:, h]
        print(f"    head {h:2d}: exp(g) min={eg.min():.6f}  max={eg.max():.6f}  mean={eg.mean():.6f}")

    print(f"\n[Done]")


if __name__ == "__main__":
    main()
