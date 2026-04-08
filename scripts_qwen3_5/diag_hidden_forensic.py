#!/usr/bin/env python3
"""Hidden-state divergence forensic: find where hidden_states error originates.

Strategy (3 phases):
  Phase 1: fp16 chunk0 (no LUT6) GPU vs ANE → isolates ANE compute vs quantization
  Phase 2: Per-layer fp16 single-layer models → identifies which layer dominates
  Phase 3: Sub-stage export within worst layer → pinpoints exact operator

All exports are fp16 (NO LUT6) for speed. Then we compare to the existing
LUT6 model on disk to quantify quantization contribution.
"""

import os, sys, time, gc, json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import coremltools as ct

# ── Config ────────────────────────────────────────────────────────
BATCH_SIZE    = 512
CTX           = 2048
NUM_CHUNKS    = 6
HF_MODEL      = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
MODEL_DIR     = os.path.join(REPO_ROOT, "qwen3_5_stable_models_6chunk")
LUT6_PREFILL  = "/tmp/diag_chunk0_fixed/prefill_LUT6_chunk0.mlpackage"  # Post-softplus-fix

OUT_DIR       = "/tmp/diag_hidden_forensic"

# Model config
HIDDEN_SIZE   = 2560
NUM_K_HEADS   = 16
KEY_HEAD_DIM  = 128
NUM_V_HEADS   = 32
VAL_HEAD_DIM  = 128
KEY_DIM       = NUM_K_HEADS * KEY_HEAD_DIM   # 2048
VALUE_DIM     = NUM_V_HEADS * VAL_HEAD_DIM   # 4096
CONV_DIM      = KEY_DIM * 2 + VALUE_DIM      # 8192
CONV_KERNEL   = 4

# Chunk0: layers 0-5 = [lin, lin, lin, full, lin, lin]
CHUNK0_LAYERS = 6
CHUNK0_LIN    = [True, True, True, False, True, True]
CHUNK0_START  = 0
CHUNK0_END    = 6

ANE_STATE_MAX  = 1024
ANE_CONV_GROUP = (CONV_DIM + ANE_STATE_MAX - 1) // ANE_STATE_MAX  # 8
ANE_CONV_DIM1  = CONV_DIM // ANE_CONV_GROUP                       # 1024
ANE_CONV_DIM2  = CONV_KERNEL * ANE_CONV_GROUP                     # 32


# ── helpers ───────────────────────────────────────────────────────
def cmp(name, a, b):
    a_f = a.astype(np.float32).flatten()
    b_f = b.astype(np.float32).flatten()
    diff = a_f - b_f
    mad  = np.max(np.abs(diff))
    mean = np.mean(np.abs(diff))
    dot  = np.dot(a_f, b_f)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    cos  = dot / (na * nb + 1e-12)
    return mad, mean, cos


def make_inputs(hidden_np, valid_len):
    pos = np.zeros(BATCH_SIZE, dtype=np.int32)
    pos[:valid_len] = np.arange(valid_len, dtype=np.int32)
    mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
    for i in range(valid_len):
        mask[0, 0, i, :i+1] = 0.0
    return {
        "hidden_states":          hidden_np,
        "position_ids":           pos,
        "causal_mask":            mask,
        "current_pos":            np.array([0], dtype=np.int32),
        "linear_conv_state":      np.zeros((CHUNK0_LAYERS, ANE_CONV_DIM1, ANE_CONV_DIM2), dtype=np.float16),
        "linear_recurrent_state": np.zeros((CHUNK0_LAYERS, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM), dtype=np.float16),
        "valid_len":              np.array([valid_len], dtype=np.int32),
    }


def predict(path, inputs, cu, label):
    t0 = time.time()
    try:
        m = ct.models.MLModel(path, compute_units=cu)
    except Exception as e:
        print(f"    {label}: LOAD FAILED — {e}")
        return None
    load_t = time.time() - t0
    try:
        state = m.make_state()
        t0 = time.time()
        out = m.predict(inputs, state=state)
    except Exception:
        try:
            # Model has no state OR failed to compile on this compute unit
            t0 = time.time()
            out = m.predict(inputs)
        except Exception as e:
            print(f"    {label}: PREDICT FAILED — {type(e).__name__}")
            del m; gc.collect()
            return None
    pred_t = time.time() - t0
    print(f"    {label}: load {load_t:.1f}s  predict {pred_t:.1f}s")
    del m; gc.collect()
    return out


# ══════════════════════════════════════════════════════════════════
# PHASE 1: LUT6 chunk0 GPU vs ANE decomposition
# (fp16 chunk0 is too large for ANE and disk; use LUT6 only)
# ══════════════════════════════════════════════════════════════════
def phase1(model, hidden_np, valid_len):
    print("\n" + "=" * 80)
    print("PHASE 1: LUT6 chunk0 GPU vs ANE — baseline decomposition")
    print("=" * 80)

    inputs = make_inputs(hidden_np, valid_len)

    # LUT6 GPU vs ANE
    print("\n  Running LUT6 chunk0 (post-softplus-fix):")
    lut6_gpu = predict(LUT6_PREFILL, inputs, ct.ComputeUnit.CPU_AND_GPU, "LUT6/GPU")
    lut6_ane = predict(LUT6_PREFILL, inputs, ct.ComputeUnit.CPU_AND_NE,  "LUT6/ANE")

    # Comparisons
    print("\n  ── HIDDEN STATES ─────────────────────────────────────────")
    h_lut6_gpu_ane = cmp("h", lut6_gpu["output_hidden_states"], lut6_ane["output_hidden_states"])
    print(f"    LUT6 GPU vs ANE:    MAD={h_lut6_gpu_ane[0]:.4f}  mean={h_lut6_gpu_ane[1]:.4f}  cos={h_lut6_gpu_ane[2]:.6f}")

    # Token-level analysis: which token positions have the most divergence?
    gpu_h = lut6_gpu["output_hidden_states"].astype(np.float32)
    ane_h = lut6_ane["output_hidden_states"].astype(np.float32)
    per_token_mad = np.max(np.abs(gpu_h - ane_h), axis=-1)[0]  # (BATCH_SIZE,)
    per_token_mean = np.mean(np.abs(gpu_h - ane_h), axis=-1)[0]

    print(f"\n  ── PER-TOKEN hidden divergence (MAD across 2560 channels) ──")
    print(f"    Tokens 0-{valid_len-1} (valid):")
    for t in range(valid_len):
        print(f"      tok{t:3d}: MAD={per_token_mad[t]:.4f}  mean={per_token_mean[t]:.4f}")
    print(f"    Padding token max MAD: {per_token_mad[valid_len:].max():.4f}")
    print(f"    Worst token: {np.argmax(per_token_mad)} (MAD={per_token_mad.max():.4f})")

    # REC STATE per-layer
    print("\n  ── REC STATE (per-layer MAD: LUT6 GPU vs ANE) ───────────")
    for li in range(CHUNK0_LAYERS):
        if not CHUNK0_LIN[li]:
            continue
        r_lut6 = cmp("r", lut6_gpu["linear_recurrent_state_out"][li],
                          lut6_ane["linear_recurrent_state_out"][li])
        print(f"    L{li}: MAD={r_lut6[0]:.4f}  mean={r_lut6[1]:.4f}  cos={r_lut6[2]:.6f}")

    # CONV STATE per-layer
    print("\n  ── CONV STATE (per-layer MAD: LUT6 GPU vs ANE) ──────────")
    for li in range(CHUNK0_LAYERS):
        if not CHUNK0_LIN[li]:
            continue
        c_lut6 = cmp("c", lut6_gpu["linear_conv_state_out"][li],
                          lut6_ane["linear_conv_state_out"][li])
        print(f"    L{li}: MAD={c_lut6[0]:.4f}  mean={c_lut6[1]:.4f}  cos={c_lut6[2]:.6f}")

    return {
        "lut6_gpu_ane_mad": h_lut6_gpu_ane[0],
        "lut6_gpu": lut6_gpu,
        "lut6_ane": lut6_ane,
    }


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Per-layer fp16 models — which layer dominates?
# ══════════════════════════════════════════════════════════════════
def phase2(model, hidden_np, valid_len):
    """Export single-layer models and chain forward, measuring GPU vs ANE at each step."""
    from anemll.models.qwen3_5_model import (
        Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
        ane_conv_state_shape,
    )
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    print("\n" + "=" * 80)
    print("PHASE 2: Per-layer fp16 models — which layer dominates?")
    print("=" * 80)

    # Strategy: export 6 single-layer chunk models (one layer each) using
    # total_chunks=32 → layer k = chunk k
    # Actually, total_chunks must divide 32 layers evenly... let me just use
    # the converter with start_layer/end_layer overrides.

    # We'll manually create PrefillWrapper for single layers.
    # Simpler: use convert_part_2_prefill but override chunk range.

    # Actually let me just export chunk0 but override it to be 1 layer at a time.
    # The converter computes: start = chunk_idx * (32 // total_chunks)
    # For total_chunks=32: start=chunk_idx, end=chunk_idx+1
    # This works! 32/32 = 1 layer per chunk.

    layer_paths = []
    for li in range(CHUNK0_LAYERS):
        global_li = CHUNK0_START + li
        path = os.path.join(OUT_DIR, f"layer{global_li}_fp16.mlpackage")
        if os.path.exists(path):
            print(f"  [skip] layer {global_li} already exported")
            layer_paths.append(path)
            continue
        print(f"  Exporting layer {global_li} fp16...")
        t0 = time.time()
        conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                               num_chunks=32, lut_bits=0, per_channel=0)
        ml = conv.convert_part_2_prefill(model, chunk_idx=global_li, total_chunks=32)
        ml.save(path)
        del ml, conv; gc.collect()
        print(f"  Saved layer {global_li} in {time.time()-t0:.1f}s")
        layer_paths.append(path)

    # Chain forward: start with prompt embeddings → layer 0 → layer 1 → ... → layer 5
    # Run each on both GPU and ANE.
    # Feed GPU→GPU and ANE→ANE (each backend sees its own intermediate results).
    print("\n  Chaining layers (GPU path and ANE path)...")

    gpu_hidden = hidden_np.copy()
    ane_hidden = hidden_np.copy()

    layer_results = []

    for li in range(CHUNK0_LAYERS):
        global_li = CHUNK0_START + li
        is_linear = CHUNK0_LIN[li]

        # Build inputs for this single-layer model
        # Single-layer model has local_num_layers=1
        pos = np.zeros(BATCH_SIZE, dtype=np.int32)
        pos[:valid_len] = np.arange(valid_len, dtype=np.int32)
        mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
        for i in range(valid_len):
            mask[0, 0, i, :i+1] = 0.0

        def mk_inp(h):
            return {
                "hidden_states":          h,
                "position_ids":           pos,
                "causal_mask":            mask,
                "current_pos":            np.array([0], dtype=np.int32),
                "linear_conv_state":      np.zeros((1, ANE_CONV_DIM1, ANE_CONV_DIM2), dtype=np.float16),
                "linear_recurrent_state": np.zeros((1, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM), dtype=np.float16),
                "valid_len":              np.array([valid_len], dtype=np.int32),
            }

        path = layer_paths[li]
        gpu_inp = mk_inp(gpu_hidden)
        ane_inp = mk_inp(ane_hidden)

        gpu_out = predict(path, gpu_inp, ct.ComputeUnit.CPU_AND_GPU, f"L{global_li}/GPU")
        ane_out = predict(path, ane_inp, ct.ComputeUnit.CPU_AND_NE,  f"L{global_li}/ANE")

        if gpu_out is None or ane_out is None:
            print(f"    L{global_li}: SKIPPED (model failed to load/predict)")
            layer_results.append({
                "layer": global_li, "type": lt,
                "same_mad": float('nan'), "same_cos": float('nan'),
                "accum_mad": float('nan'), "accum_cos": float('nan'),
            })
            # Still update hidden for chain (use GPU if available)
            if gpu_out is not None:
                gpu_hidden = gpu_out["output_hidden_states"]
            if ane_out is not None:
                ane_hidden = ane_out["output_hidden_states"]
            continue

        # Same-input comparison: feed SAME hidden to both GPU and ANE
        same_inp = mk_inp(gpu_hidden)  # use GPU hidden for both
        same_out_ane = predict(path, same_inp, ct.ComputeUnit.CPU_AND_NE, f"L{global_li}/ANE-same")

        if same_out_ane is None:
            h_same = (float('nan'), 0, float('nan'))
        else:
            h_same = cmp(f"L{global_li}", gpu_out["output_hidden_states"],
                                           same_out_ane["output_hidden_states"])

        # Accumulated comparison: GPU chain vs ANE chain
        h_accum = cmp(f"L{global_li}", gpu_out["output_hidden_states"],
                                        ane_out["output_hidden_states"])

        lt = "lin" if is_linear else "full"
        print(f"    L{global_li} ({lt}): same-input MAD={h_same[0]:.4f} cos={h_same[2]:.6f}"
              f"  | accumulated MAD={h_accum[0]:.4f} cos={h_accum[2]:.6f}")

        layer_results.append({
            "layer": global_li, "type": lt,
            "same_mad": h_same[0], "same_cos": h_same[2],
            "accum_mad": h_accum[0], "accum_cos": h_accum[2],
        })

        # Update for next layer
        gpu_hidden = gpu_out["output_hidden_states"]
        ane_hidden = ane_out["output_hidden_states"]

    print("\n  ── PER-LAYER SUMMARY ─────────────────────────────────────")
    print(f"  {'Layer':>6} {'Type':>5} {'Same-in MAD':>12} {'Same-in cos':>12} {'Accum MAD':>12} {'Accum cos':>12}")
    for r in layer_results:
        print(f"  L{r['layer']:>4} {r['type']:>5} {r['same_mad']:>12.4f} {r['same_cos']:>12.6f}"
              f" {r['accum_mad']:>12.4f} {r['accum_cos']:>12.6f}")

    # Find worst same-input layer
    worst = max(layer_results, key=lambda r: r["same_mad"])
    print(f"\n  Worst same-input layer: L{worst['layer']} ({worst['type']}) MAD={worst['same_mad']:.4f}")

    return layer_results, worst["layer"]


# ══════════════════════════════════════════════════════════════════
# PHASE 3: Sub-stage forensic within worst layer
# ══════════════════════════════════════════════════════════════════
def phase3(model, hidden_np, valid_len, target_layer):
    """Export sub-stage models for the target layer; compare GPU vs ANE per stage."""
    from anemll.models.qwen3_5_model import (
        Qwen35LinearAttention, Qwen35FullAttention,
        Qwen35LinearProjStage, Qwen35LinearConvStage,
        Qwen35LinearLayoutStage, Qwen35LinearCoreNormStage,
        Qwen35RMSNorm, Qwen35RMSNormGated, Qwen35MLP,
        MODEL_DTYPE, TEST_DEVICE,
    )

    print("\n" + "=" * 80)
    print(f"PHASE 3: Sub-stage forensic for layer {target_layer}")
    print("=" * 80)

    layer = model.model.layers[target_layer]

    # Prepare hidden input: run preceding layers in PyTorch
    input_ids = torch.zeros(1, BATCH_SIZE, dtype=torch.long)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    ids = tok.encode("Hello, how are you doing today?", add_special_tokens=True)
    input_ids[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids[0])
        hidden = hidden.unsqueeze(0).to(torch.float16)

        for li in range(target_layer):
            prev_layer = model.model.layers[li]
            if prev_layer.layer_type == "linear_attention":
                prev_attn = prev_layer.self_attn
                prev_attn.export_expected_batch_size = 1
                prev_attn.export_expected_seq_len = BATCH_SIZE
                x = prev_layer.input_layernorm(hidden)
                conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL, dtype=torch.float16)
                rec_state = torch.zeros(1, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM, dtype=torch.float16)
                attn_out, _, _ = prev_attn._forward_prefill_export_impl(
                    x, conv_state, rec_state,
                    has_previous_state=True,
                    force_fp16_math=True,
                    valid_len=torch.tensor([valid_len], dtype=torch.int32),
                )
                hidden = hidden + attn_out
                post = prev_layer.post_attention_layernorm(hidden)
                hidden = hidden + prev_layer.mlp(post)
            else:
                hidden = prev_layer(hidden, None, torch.arange(BATCH_SIZE))
        print(f"  Ran {target_layer} preceding layers in PyTorch")

    target_input = hidden.clone()
    target_input_np = target_input.cpu().numpy()

    if layer.layer_type == "full_attention":
        phase3_full_attention(model, layer, target_layer, target_input, target_input_np, valid_len)
    else:
        phase3_linear_attention(model, layer, target_layer, target_input, target_input_np, valid_len)


def phase3_full_attention(model, layer, target_layer, target_input, target_input_np, valid_len):
    """Investigate full_attention layer sub-stages."""
    from anemll.models.qwen3_5_model import MODEL_DTYPE

    print(f"\n  Layer {target_layer} is FULL ATTENTION — investigating sub-stages")

    attn = layer.self_attn

    # ── Export sub-stage models ──

    # Stage A: LayerNorm + QKV projection + rotary + gate
    class QKVProjWrapper(torch.nn.Module):
        def __init__(self, layernorm, self_attn):
            super().__init__()
            self.layernorm = layernorm
            self.self_attn = self_attn
        def forward(self, hidden, position_ids):
            x = self.layernorm(hidden)
            query, key, value, gate = self.self_attn.get_new_kv_cache_prefill(x, position_ids)
            return query, key, value, gate

    qkv_wrap = QKVProjWrapper(layer.input_layernorm, attn).eval()
    qkv_path = os.path.join(OUT_DIR, f"layer{target_layer}_qkv.mlpackage")
    pos_ids = torch.arange(BATCH_SIZE, dtype=torch.long)
    if not os.path.exists(qkv_path):
        print("  Exporting QKV projection sub-model...")
        traced = torch.jit.trace(qkv_wrap, (target_input, pos_ids), check_trace=False)
        ml = ct.convert(traced,
                        inputs=[ct.TensorType(name="hidden", shape=target_input.shape, dtype=np.float16),
                                ct.TensorType(name="position_ids", shape=pos_ids.shape, dtype=np.int32)],
                        outputs=[ct.TensorType(name="query", dtype=np.float16),
                                 ct.TensorType(name="key", dtype=np.float16),
                                 ct.TensorType(name="value", dtype=np.float16),
                                 ct.TensorType(name="gate", dtype=np.float16)],
                        compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.iOS18,
                        convert_to="mlprogram")
        ml.save(qkv_path)
        print(f"    Saved {qkv_path}")
        del ml; gc.collect()

    # Stage B: Attention compute (softmax matmul) + gate + output proj
    # We need the Q/K/V from PyTorch to feed into this stage
    with torch.no_grad():
        x_normed = layer.input_layernorm(target_input)
        query, key, value, gate = attn.get_new_kv_cache_prefill(x_normed, pos_ids)

    class AttnComputeWrapper(torch.nn.Module):
        """Wraps: scaled_dot_product_attention + gate + out_proj."""
        def __init__(self, self_attn):
            super().__init__()
            self.o_proj = self_attn.o_proj
            self.num_heads = self_attn.num_heads
            self.head_dim = self_attn.head_dim
        def forward(self, query, key, value, gate):
            # query/key/value: (1, num_heads, seq_len, head_dim)
            # Compute attention
            import torch.nn.functional as F
            attn_out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            # Reshape: (1, num_heads, seq_len, head_dim) → (1, seq_len, hidden_size)
            bsz, nh, sl, hd = attn_out.shape
            attn_out = attn_out.transpose(1, 2).reshape(bsz, sl, nh * hd)
            # Apply gate and output projection
            attn_out = attn_out * gate
            # out_proj is a Conv2d — need channels-first 4D
            cf = attn_out.transpose(1, 2).unsqueeze(2)
            out_cf = self.o_proj(cf.to(torch.float16))
            out = out_cf.squeeze(2).transpose(1, 2)
            return out

    attn_compute_wrap = AttnComputeWrapper(attn).eval()
    attn_path = os.path.join(OUT_DIR, f"layer{target_layer}_attn_compute.mlpackage")
    if not os.path.exists(attn_path):
        print("  Exporting attention compute sub-model...")
        traced = torch.jit.trace(attn_compute_wrap, (query, key, value, gate), check_trace=False)
        ml = ct.convert(traced,
                        inputs=[ct.TensorType(name="query", shape=query.shape, dtype=np.float16),
                                ct.TensorType(name="key", shape=key.shape, dtype=np.float16),
                                ct.TensorType(name="value", shape=value.shape, dtype=np.float16),
                                ct.TensorType(name="gate", shape=gate.shape, dtype=np.float16)],
                        outputs=[ct.TensorType(name="attn_out", dtype=np.float16)],
                        compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.iOS18,
                        convert_to="mlprogram")
        ml.save(attn_path)
        print(f"    Saved {attn_path}")
        del ml; gc.collect()

    # Stage C: MLP
    class MLPWrapper(torch.nn.Module):
        def __init__(self, post_ln, mlp):
            super().__init__()
            self.post_ln = post_ln
            self.mlp = mlp
        def forward(self, hidden):
            return self.mlp(self.post_ln(hidden))

    mlp_wrap = MLPWrapper(layer.post_attention_layernorm, layer.mlp).eval()
    mlp_path = os.path.join(OUT_DIR, f"layer{target_layer}_mlp.mlpackage")
    if not os.path.exists(mlp_path):
        print("  Exporting MLP sub-model...")
        with torch.no_grad():
            attn_output = attn_compute_wrap(query, key, value, gate)
            post_attn = target_input + attn_output
        traced = torch.jit.trace(mlp_wrap, (post_attn,), check_trace=False)
        ml = ct.convert(traced,
                        inputs=[ct.TensorType(name="hidden", shape=post_attn.shape, dtype=np.float16)],
                        outputs=[ct.TensorType(name="mlp_out", dtype=np.float16)],
                        compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.iOS18,
                        convert_to="mlprogram")
        ml.save(mlp_path)
        print(f"    Saved {mlp_path}")
        del ml; gc.collect()

    # ── Run sub-stages ──
    print(f"\n  Running sub-stages on GPU and ANE:")

    # Stage A: QKV projection
    pos_np = np.arange(BATCH_SIZE, dtype=np.int32)
    qkv_inputs = {"hidden": target_input_np, "position_ids": pos_np}
    qkv_gpu = predict(qkv_path, qkv_inputs, ct.ComputeUnit.CPU_AND_GPU, "qkv/GPU")
    qkv_ane = predict(qkv_path, qkv_inputs, ct.ComputeUnit.CPU_AND_NE,  "qkv/ANE")
    if qkv_gpu and qkv_ane:
        for oname in ["query", "key", "value", "gate"]:
            m, mn, c = cmp(oname, qkv_gpu[oname], qkv_ane[oname])
            flag = " <<<" if m > 0.1 else ""
            print(f"    qkv  {oname:8s}: MAD={m:.6f}  mean={mn:.6f}  cos={c:.6f}{flag}")

    # Stage B: Attention compute + gate + out_proj
    if qkv_gpu:
        attn_inputs = {
            "query": qkv_gpu["query"],
            "key": qkv_gpu["key"],
            "value": qkv_gpu["value"],
            "gate": qkv_gpu["gate"],
        }
        attn_gpu = predict(attn_path, attn_inputs, ct.ComputeUnit.CPU_AND_GPU, "attn/GPU")
        attn_ane = predict(attn_path, attn_inputs, ct.ComputeUnit.CPU_AND_NE,  "attn/ANE")
        if attn_gpu and attn_ane:
            m, mn, c = cmp("attn_out", attn_gpu["attn_out"], attn_ane["attn_out"])
            print(f"    attn {'out':8s}: MAD={m:.6f}  mean={mn:.6f}  cos={c:.6f}")

            # Per-token breakdown
            gpu_a = attn_gpu["attn_out"].astype(np.float32)
            ane_a = attn_ane["attn_out"].astype(np.float32)
            per_tok = np.max(np.abs(gpu_a - ane_a), axis=-1)
            if per_tok.ndim > 1:
                per_tok = per_tok[0]
            print(f"\n    Per-token attn_out MAD (first {valid_len} valid + some padding):")
            for t in range(min(valid_len + 4, BATCH_SIZE)):
                tag = "VALID" if t < valid_len else "pad"
                flag = " <<<" if per_tok[t] > 0.5 else ""
                print(f"      tok{t:3d} ({tag}): MAD={per_tok[t]:.4f}{flag}")
            print(f"      Max padding: tok{np.argmax(per_tok[valid_len:]) + valid_len}"
                  f" MAD={per_tok[valid_len:].max():.4f}")

            # Post-residual hidden for MLP input
            post_attn_np = target_input_np + attn_gpu["attn_out"]
        else:
            post_attn_np = None
    else:
        post_attn_np = None

    # Stage C: MLP
    if post_attn_np is not None:
        mlp_gpu = predict(mlp_path, {"hidden": post_attn_np}, ct.ComputeUnit.CPU_AND_GPU, "mlp/GPU")
        mlp_ane = predict(mlp_path, {"hidden": post_attn_np}, ct.ComputeUnit.CPU_AND_NE,  "mlp/ANE")
        if mlp_gpu and mlp_ane:
            m, mn, c = cmp("mlp_out", mlp_gpu["mlp_out"], mlp_ane["mlp_out"])
            print(f"    mlp  {'out':8s}: MAD={m:.6f}  mean={mn:.6f}  cos={c:.6f}")

    # ── Summary ──
    print(f"\n  ── LAYER {target_layer} (full_attention) DIVERGENCE SUMMARY ──")
    print(f"  Check: is divergence from QKV projection, attention compute, or MLP?")
    print(f"  Check: is it concentrated in valid tokens or padding?")


def phase3_linear_attention(model, layer, target_layer, target_input, target_input_np, valid_len):
    """Investigate linear_attention layer sub-stages."""
    from anemll.models.qwen3_5_model import (
        Qwen35LinearAttention, MODEL_DTYPE,
    )

    print(f"\n  Layer {target_layer} is LINEAR ATTENTION — investigating sub-stages")

    attn = layer.self_attn
    attn.export_expected_batch_size = 1
    attn.export_expected_seq_len = BATCH_SIZE
    bsz, seq_len = 1, BATCH_SIZE
    vl = torch.tensor([valid_len], dtype=torch.int32)

    with torch.no_grad():
        x_normed = layer.input_layernorm(target_input)
        mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(x_normed)
        conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL, dtype=torch.float16)
        conv_out_cf, _ = attn.conv_stage(mixed_qkv_pre, conv_state,
                                          expected_seq_len=seq_len, valid_len=vl)
        query, key, value, g, beta, z = attn.layout_stage(
            conv_out_cf, z_cf, b_cf, a_cf, bsz, seq_len, force_fp16_math=True)

    # Export core_norm stage only (most likely culprit based on amplification note in code)
    class CoreNormWrapper(torch.nn.Module):
        def __init__(self, core_norm_stage):
            super().__init__()
            self.core_norm_stage = core_norm_stage
        def forward(self, query, key, value, g, beta, z, rec_state):
            out, next_rec = self.core_norm_stage(
                query=query, key=key, value=value, g=g, beta=beta, z=z,
                recurrent_state=rec_state, has_previous_state=True,
                bsz=1, seq_len=BATCH_SIZE, force_recurrent=False, force_fp16_math=True)
            return out, next_rec

    core_path = os.path.join(OUT_DIR, f"layer{target_layer}_core_norm.mlpackage")
    rec_in = torch.zeros(1, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM, dtype=torch.float16)

    positions = torch.arange(seq_len, dtype=torch.int32)
    valid_mask = (positions < vl).to(key.dtype)
    mask_bsh1 = valid_mask.reshape(1, seq_len, 1, 1)
    mask_bsh = valid_mask.reshape(1, seq_len, 1)
    key_m = key * mask_bsh1
    value_m = value * mask_bsh1
    beta_m = beta * mask_bsh
    g_m = g * mask_bsh

    if not os.path.exists(core_path):
        corewrap = CoreNormWrapper(attn.core_norm_stage).eval()
        traced = torch.jit.trace(corewrap, (query, key_m, value_m, g_m, beta_m, z, rec_in),
                                 check_trace=False)
        ml = ct.convert(traced,
                        inputs=[ct.TensorType(name="query", shape=query.shape, dtype=np.float16),
                                ct.TensorType(name="key", shape=key_m.shape, dtype=np.float16),
                                ct.TensorType(name="value", shape=value_m.shape, dtype=np.float16),
                                ct.TensorType(name="g", shape=g_m.shape, dtype=np.float16),
                                ct.TensorType(name="beta", shape=beta_m.shape, dtype=np.float16),
                                ct.TensorType(name="z", shape=z.shape, dtype=np.float16),
                                ct.TensorType(name="rec_state", shape=rec_in.shape, dtype=np.float16)],
                        outputs=[ct.TensorType(name="attn_out", dtype=np.float16),
                                 ct.TensorType(name="next_rec", dtype=np.float16)],
                        compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.iOS18,
                        convert_to="mlprogram")
        ml.save(core_path)
        del ml; gc.collect()

    core_inputs = {
        "query": query.cpu().numpy(),
        "key": key_m.cpu().numpy(),
        "value": value_m.cpu().numpy(),
        "g": g_m.cpu().numpy(),
        "beta": beta_m.cpu().numpy(),
        "z": z.cpu().numpy(),
        "rec_state": np.zeros((1, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM), dtype=np.float16),
    }
    core_gpu = predict(core_path, core_inputs, ct.ComputeUnit.CPU_AND_GPU, "core/GPU")
    core_ane = predict(core_path, core_inputs, ct.ComputeUnit.CPU_AND_NE,  "core/ANE")
    if core_gpu and core_ane:
        for oname in ["attn_out", "next_rec"]:
            m, mn, c = cmp(oname, core_gpu[oname], core_ane[oname])
            print(f"    core  {oname:12s}: MAD={m:.6f}  cos={c:.6f}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

    print("=" * 80)
    print("HIDDEN-STATE DIVERGENCE FORENSIC")
    print("=" * 80)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load model
    print("\nLoading model...")
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

    # Embed prompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    prompt = "Hello, how are you doing today?"
    ids = tok.encode(prompt, add_special_tokens=True)
    print(f"  Prompt: {prompt!r} → {len(ids)} tokens")

    input_ids = torch.zeros(1, BATCH_SIZE, dtype=torch.long)
    input_ids[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids[0])
        hidden = hidden.unsqueeze(0).to(torch.float16)
    hidden_np = hidden.cpu().numpy()
    valid_len = len(ids)

    # Phase 1
    p1 = phase1(model, hidden_np, valid_len)

    # Phase 2
    layer_results, worst_layer = phase2(model, hidden_np, valid_len)

    # Phase 3: drill into worst layer (if it's a linear attention layer)
    phase3(model, hidden_np, valid_len, worst_layer)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
