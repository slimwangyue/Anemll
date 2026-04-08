#!/usr/bin/env python3
"""Phase 3 ONLY: Full-attention layer 3 sub-stage forensic.

From Phase 2 (prior run), we know:
  L0 (lin): same-input MAD=0.2930
  L1 (lin): same-input MAD=0.1270
  L2 (lin): same-input MAD=0.1494
  L3 (full): same-input MAD=1.5210  ← DOMINANT
  L4 (lin): same-input MAD=0.2266

This script isolates the sub-stages of Layer 3 (full_attention) to find
where the GPU-vs-ANE divergence originates.

Sub-stages:
  A) LayerNorm + QKV proj + rotary   → query, key, value, gate
  B) Softmax attention + gate + out_proj → attn_out
  C) MLP (post_layernorm + mlp)     → mlp_out

Each exported as a small CoreML model, run GPU vs ANE.
"""

import os, sys, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import coremltools as ct

BATCH_SIZE    = 512
CTX           = 2048
NUM_CHUNKS    = 6
HF_MODEL      = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
OUT_DIR       = "/tmp/diag_L3_forensic"

HIDDEN_SIZE   = 2560
NUM_K_HEADS   = 16
KEY_HEAD_DIM  = 128
NUM_V_HEADS   = 32
VAL_HEAD_DIM  = 128
CONV_DIM      = 8192
CONV_KERNEL   = 4

TARGET_LAYER  = 3  # full_attention


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


def export_and_predict(wrapper, trace_args, input_specs, output_specs, inputs_np, path, label):
    """Export a wrapper, run GPU and ANE, return (gpu_out, ane_out). Deletes model after."""
    if not os.path.exists(path):
        print(f"  Exporting {label}...")
        traced = torch.jit.trace(wrapper, trace_args, check_trace=False)
        ml = ct.convert(traced,
                        inputs=input_specs,
                        outputs=output_specs,
                        compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.iOS18,
                        convert_to="mlprogram")
        ml.save(path)
        del ml, traced; gc.collect()
        print(f"    Saved {path}")

    gpu = predict(path, inputs_np, ct.ComputeUnit.CPU_AND_GPU, f"{label}/GPU")
    ane = predict(path, inputs_np, ct.ComputeUnit.CPU_AND_NE,  f"{label}/ANE")

    # Clean up to save disk
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)

    return gpu, ane


def main():
    from anemll.models.qwen3_5_model import (
        Qwen35Config, Qwen35ForCausalLM,
        Qwen35LinearAttention, MODEL_DTYPE,
    )

    print("=" * 80)
    print("PHASE 3: Layer 3 (full_attention) Sub-Stage Forensic")
    print("=" * 80)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load model
    print("\nLoading model...")
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

    # Tokenize
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    ids = tok.encode("Hello, how are you doing today?", add_special_tokens=True)
    valid_len = len(ids)
    print(f"  Prompt: {valid_len} tokens")

    input_ids = torch.zeros(1, BATCH_SIZE, dtype=torch.long)
    input_ids[0, :valid_len] = torch.tensor(ids, dtype=torch.long)

    # Run PyTorch forward through layers 0-2 to get Layer 3 input
    print(f"\n  Running layers 0-{TARGET_LAYER-1} in PyTorch...")
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids[0])
        hidden = hidden.unsqueeze(0).to(torch.float16)

        for li in range(TARGET_LAYER):
            layer = model.model.layers[li]
            if layer.layer_type == "linear_attention":
                attn = layer.self_attn
                attn.export_expected_batch_size = 1
                attn.export_expected_seq_len = BATCH_SIZE
                x = layer.input_layernorm(hidden)
                cs = torch.zeros(1, CONV_DIM, CONV_KERNEL, dtype=torch.float16)
                rs = torch.zeros(1, NUM_V_HEADS, KEY_HEAD_DIM, VAL_HEAD_DIM, dtype=torch.float16)
                ao, _, _ = attn._forward_prefill_export_impl(
                    x, cs, rs, has_previous_state=True, force_fp16_math=True,
                    valid_len=torch.tensor([valid_len], dtype=torch.int32))
                hidden = hidden + ao
                post = layer.post_attention_layernorm(hidden)
                hidden = hidden + layer.mlp(post)
            else:
                hidden = layer(hidden, None, torch.arange(BATCH_SIZE))

    target_input = hidden.clone()
    target_input_np = target_input.cpu().numpy()
    print(f"  Layer {TARGET_LAYER} input: shape={list(target_input.shape)}"
          f" |max|={target_input.abs().max():.4f} |mean|={target_input.abs().mean():.4f}")

    layer3 = model.model.layers[TARGET_LAYER]
    attn3 = layer3.self_attn

    # ══════════════════════════════════════════════════════════════
    # Stage A: LayerNorm + QKV Projection + RoPE + QK Norm
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Stage A: LayerNorm + QKV + RoPE (full_attention L3)")
    print(f"{'='*60}")

    class QKVWrapper(torch.nn.Module):
        def __init__(self, ln, attn):
            super().__init__()
            self.ln = ln
            self.attn = attn
        def forward(self, hidden, position_ids):
            x = self.ln(hidden)
            q, k, v, gate = self.attn.get_new_kv_cache_prefill(x, position_ids)
            return q, k, v, gate

    qkv_wrap = QKVWrapper(layer3.input_layernorm, attn3).eval()
    pos_ids = torch.arange(BATCH_SIZE, dtype=torch.long)

    with torch.no_grad():
        pt_q, pt_k, pt_v, pt_gate = qkv_wrap(target_input, pos_ids)
    print(f"  Q shape: {list(pt_q.shape)}, K: {list(pt_k.shape)}, V: {list(pt_v.shape)}, gate: {list(pt_gate.shape)}")

    qkv_gpu, qkv_ane = export_and_predict(
        qkv_wrap,
        (target_input, pos_ids),
        [ct.TensorType(name="hidden", shape=target_input.shape, dtype=np.float16),
         ct.TensorType(name="position_ids", shape=pos_ids.shape, dtype=np.int32)],
        [ct.TensorType(name="query", dtype=np.float16),
         ct.TensorType(name="key", dtype=np.float16),
         ct.TensorType(name="value", dtype=np.float16),
         ct.TensorType(name="gate", dtype=np.float16)],
        {"hidden": target_input_np, "position_ids": np.arange(BATCH_SIZE, dtype=np.int32)},
        os.path.join(OUT_DIR, "L3_qkv.mlpackage"),
        "QKV"
    )

    if qkv_gpu and qkv_ane:
        for oname in ["query", "key", "value", "gate"]:
            m, mn, c = cmp(oname, qkv_gpu[oname], qkv_ane[oname])
            flag = " <<<" if m > 0.1 else ""
            print(f"  {oname:8s}: MAD={m:.6f}  mean={mn:.6f}  cos={c:.6f}{flag}")

    # ══════════════════════════════════════════════════════════════
    # Stage B: Softmax Attention + Gate + OutProj
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Stage B: Softmax Attention + Gate + OutProj (L3)")
    print(f"{'='*60}")

    class AttnWrapper(torch.nn.Module):
        def __init__(self, o_proj, num_heads, num_kv_heads, head_dim):
            super().__init__()
            self.o_proj = o_proj
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.n_rep = num_heads // num_kv_heads
            self.head_dim = head_dim
        def forward(self, query, key, value, gate):
            # GQA: repeat K/V to match Q head count
            if self.n_rep > 1:
                key = key.repeat_interleave(self.n_rep, dim=1)
                value = value.repeat_interleave(self.n_rep, dim=1)
            attn_out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            bsz, nh, sl, hd = attn_out.shape
            attn_out = attn_out.transpose(1, 2).reshape(bsz, sl, nh * hd)
            # Gate (sigmoid)
            attn_out = attn_out * torch.sigmoid(gate.to(attn_out.dtype))
            # out_proj (Conv2d)
            cf = attn_out.transpose(1, 2).unsqueeze(2)
            out_cf = self.o_proj(cf.to(torch.float16))
            out = out_cf.squeeze(2).transpose(1, 2)
            return out

    attn_wrap = AttnWrapper(attn3.o_proj, attn3.num_heads, attn3.num_kv_heads, attn3.head_dim).eval()

    # Use GPU QKV outputs as inputs (ensures same data)
    if qkv_gpu:
        q_np, k_np, v_np, g_np = qkv_gpu["query"], qkv_gpu["key"], qkv_gpu["value"], qkv_gpu["gate"]
        q_t = torch.from_numpy(q_np)
        k_t = torch.from_numpy(k_np)
        v_t = torch.from_numpy(v_np)
        g_t = torch.from_numpy(g_np)

        attn_gpu, attn_ane = export_and_predict(
            attn_wrap,
            (q_t, k_t, v_t, g_t),
            [ct.TensorType(name="query", shape=q_t.shape, dtype=np.float16),
             ct.TensorType(name="key", shape=k_t.shape, dtype=np.float16),
             ct.TensorType(name="value", shape=v_t.shape, dtype=np.float16),
             ct.TensorType(name="gate", shape=g_t.shape, dtype=np.float16)],
            [ct.TensorType(name="attn_out", dtype=np.float16)],
            {"query": q_np, "key": k_np, "value": v_np, "gate": g_np},
            os.path.join(OUT_DIR, "L3_attn.mlpackage"),
            "Attn"
        )

        if attn_gpu and attn_ane:
            m, mn, c = cmp("attn_out", attn_gpu["attn_out"], attn_ane["attn_out"])
            print(f"  attn_out: MAD={m:.6f}  mean={mn:.6f}  cos={c:.6f}")

            # Per-token analysis
            a_gpu = attn_gpu["attn_out"].astype(np.float32)
            a_ane = attn_ane["attn_out"].astype(np.float32)
            ptmad = np.max(np.abs(a_gpu - a_ane), axis=-1)
            if ptmad.ndim > 1:
                ptmad = ptmad[0]
            print(f"\n  Per-token attn_out MAD:")
            for t in range(min(valid_len + 5, 20)):
                tag = "VAL" if t < valid_len else "pad"
                flag = " <<<" if ptmad[t] > 0.5 else ""
                print(f"    tok{t:3d} ({tag}): MAD={ptmad[t]:.4f}{flag}")
            print(f"    Max valid:   {ptmad[:valid_len].max():.4f} (tok {np.argmax(ptmad[:valid_len])})")
            print(f"    Max padding: {ptmad[valid_len:].max():.4f} (tok {np.argmax(ptmad[valid_len:]) + valid_len})")

            # Per-channel analysis for worst token
            worst_tok = np.argmax(ptmad)
            ch_diff = np.abs(a_gpu[0, worst_tok] - a_ane[0, worst_tok])
            top_ch = np.argsort(-ch_diff)[:10]
            print(f"\n  Worst token {worst_tok}: top-10 divergent channels:")
            for ci in top_ch:
                print(f"    ch{ci:4d}: diff={ch_diff[ci]:.4f}  gpu={a_gpu[0,worst_tok,ci]:.4f}  ane={a_ane[0,worst_tok,ci]:.4f}")
    else:
        attn_gpu = attn_ane = None
        print("  SKIPPED (QKV stage failed)")

    # ══════════════════════════════════════════════════════════════
    # Stage B2: Attention ONLY (no gate, no out_proj)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Stage B2: Softmax Attention ONLY (no gate, no out_proj)")
    print(f"{'='*60}")

    class SoftmaxAttnOnly(torch.nn.Module):
        def __init__(self, n_rep):
            super().__init__()
            self.n_rep = n_rep
        def forward(self, query, key, value):
            if self.n_rep > 1:
                key = key.repeat_interleave(self.n_rep, dim=1)
                value = value.repeat_interleave(self.n_rep, dim=1)
            return F.scaled_dot_product_attention(query, key, value, is_causal=True)

    sdpa_wrap = SoftmaxAttnOnly(attn3.num_heads // attn3.num_kv_heads).eval()

    if qkv_gpu:
        sdpa_gpu, sdpa_ane = export_and_predict(
            sdpa_wrap,
            (q_t, k_t, v_t),
            [ct.TensorType(name="query", shape=q_t.shape, dtype=np.float16),
             ct.TensorType(name="key", shape=k_t.shape, dtype=np.float16),
             ct.TensorType(name="value", shape=v_t.shape, dtype=np.float16)],
            [ct.TensorType(name="sdpa_out", dtype=np.float16)],
            {"query": q_np, "key": k_np, "value": v_np},
            os.path.join(OUT_DIR, "L3_sdpa.mlpackage"),
            "SDPA"
        )

        if sdpa_gpu and sdpa_ane:
            m, mn, c = cmp("sdpa_out", sdpa_gpu["sdpa_out"], sdpa_ane["sdpa_out"])
            print(f"  sdpa_out: MAD={m:.6f}  mean={mn:.6f}  cos={c:.6f}")

            # Per-token
            s_gpu = sdpa_gpu["sdpa_out"].astype(np.float32)
            s_ane = sdpa_ane["sdpa_out"].astype(np.float32)
            # shape is (1, num_heads, seq_len, head_dim)
            if s_gpu.ndim == 4:
                # Per-head analysis
                print(f"\n  Per-head SDPA MAD:")
                n_heads = s_gpu.shape[1]
                for h in range(n_heads):
                    hm = np.max(np.abs(s_gpu[0, h] - s_ane[0, h]))
                    hmn = np.mean(np.abs(s_gpu[0, h] - s_ane[0, h]))
                    flag = " <<<" if hm > 0.1 else ""
                    print(f"    head {h:2d}: MAD={hm:.6f}  mean={hmn:.6f}{flag}")

                # Per-token (max across heads and head_dim)
                ptmad = np.max(np.abs(s_gpu - s_ane), axis=(1, 3))[0]  # (seq_len,)
                print(f"\n  Per-token SDPA MAD (max over heads,dim):")
                for t in range(min(valid_len + 5, 20)):
                    tag = "VAL" if t < valid_len else "pad"
                    flag = " <<<" if ptmad[t] > 0.5 else ""
                    print(f"    tok{t:3d} ({tag}): MAD={ptmad[t]:.4f}{flag}")
                print(f"    Max valid:   {ptmad[:valid_len].max():.4f}")
                print(f"    Max padding: {ptmad[valid_len:].max():.4f}")

    # ══════════════════════════════════════════════════════════════
    # Stage C: MLP
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Stage C: MLP (post_attention_layernorm + MLP)")
    print(f"{'='*60}")

    class MLPWrapper(torch.nn.Module):
        def __init__(self, post_ln, mlp):
            super().__init__()
            self.post_ln = post_ln
            self.mlp = mlp
        def forward(self, hidden):
            return self.mlp(self.post_ln(hidden))

    mlp_wrap = MLPWrapper(layer3.post_attention_layernorm, layer3.mlp).eval()

    # Compute post-attention hidden from PyTorch (for representative input)
    with torch.no_grad():
        x_normed = layer3.input_layernorm(target_input)
        q, k, v, gate = attn3.get_new_kv_cache_prefill(x_normed, pos_ids)
        # GQA: repeat K/V
        n_rep = attn3.num_heads // attn3.num_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        pt_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        bsz, nh, sl, hd = pt_attn.shape
        pt_attn = pt_attn.transpose(1, 2).reshape(bsz, sl, nh * hd)
        pt_attn = pt_attn * torch.sigmoid(gate.to(pt_attn.dtype))
        cf = pt_attn.transpose(1, 2).unsqueeze(2)
        pt_out = attn3.o_proj(cf.to(torch.float16)).squeeze(2).transpose(1, 2)
        post_attn = target_input + pt_out

    mlp_gpu, mlp_ane = export_and_predict(
        mlp_wrap,
        (post_attn,),
        [ct.TensorType(name="hidden", shape=post_attn.shape, dtype=np.float16)],
        [ct.TensorType(name="mlp_out", dtype=np.float16)],
        {"hidden": post_attn.cpu().numpy()},
        os.path.join(OUT_DIR, "L3_mlp.mlpackage"),
        "MLP"
    )

    if mlp_gpu and mlp_ane:
        m, mn, c = cmp("mlp_out", mlp_gpu["mlp_out"], mlp_ane["mlp_out"])
        print(f"  mlp_out: MAD={m:.6f}  mean={mn:.6f}  cos={c:.6f}")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("LAYER 3 (full_attention) SUB-STAGE SUMMARY")
    print(f"{'='*80}")
    print(f"  From Phase 2: full layer L3 same-input MAD=1.5210")
    print(f"  Investigate which sub-stage dominates:")
    print(f"    A) QKV + RoPE   → see MADs above")
    print(f"    B) Softmax SDPA → see MADs above (this is the prime suspect)")
    print(f"    C) MLP          → see MADs above")
    print(f"\n  If SDPA dominates: fp16 softmax has poor precision for long padding")
    print(f"  sequences on ANE. The 512-token sequence has 504 padding positions")
    print(f"  where Q/K come from zero embeddings, creating degenerate softmax inputs.")

    # Cleanup
    import shutil
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
