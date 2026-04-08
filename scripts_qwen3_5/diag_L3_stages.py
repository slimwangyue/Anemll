#!/usr/bin/env python3
"""Stage-by-stage diagnostic for Layer 3 full-attention GPU vs ANE divergence.

Exports Layer 3 as a standalone CoreML model with 10 intermediate outputs,
runs on both GPU and ANE, compares at each sub-stage for valid tokens only.

Isolates WHERE inside Layer 3 the divergence first appears materially.
"""

import os, sys, time, gc, shutil, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35Config, Qwen35ForCausalLM,
    MODEL_DTYPE, _repeat_kv, apply_rotary_pos_emb_prefill,
)

HF_MODEL  = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
OUT_DIR   = "/tmp/diag_L3_stages"
BATCH     = 64       # small for fast export
CTX       = 2048
VALID_LEN = 8
PROMPT    = "Hello, how are you doing today?"


# ═══════════════════════════════════════════════════════════════════
# Diagnostic wrapper — mirrors Layer 3 computation exactly but
# returns ALL intermediate tensors as separate outputs.
# ═══════════════════════════════════════════════════════════════════
class Layer3DiagModel(nn.Module):
    def __init__(self, layer3, batch_size, ctx_len):
        super().__init__()
        self.ln     = layer3.input_layernorm
        self.attn   = layer3.self_attn
        self.S      = batch_size
        self.CTX    = ctx_len

    def forward(self, hidden_states, position_ids, causal_mask):
        # Stage 0: Input LayerNorm
        x = self.ln(hidden_states)

        # Stage 1–3: Q / K / V projection + Q/K-Norm + RoPE
        q, k, v, gate = self.attn._project_qkvg(x)
        q = self.attn.q_norm(q)
        k = self.attn.k_norm(k)
        cos, sin = self.attn.rotary.get(x, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb_prefill(
            q, k, cos, sin, self.attn.rotary.rotary_dim
        )
        q_rot = q_rot.to(MODEL_DTYPE)
        k_rot = k_rot.to(MODEL_DTYPE)
        v     = v.to(MODEL_DTYPE)
        gate  = gate.to(MODEL_DTYPE)

        # Stage 4: Pad KV to CTX length + GQA repeat
        # (simulates zero-initialized KV cache with write at pos 0)
        k_padded = F.pad(k_rot, (0, 0, 0, self.CTX - self.S))
        v_padded = F.pad(v,     (0, 0, 0, self.CTX - self.S))
        n_rep   = self.attn.num_heads // self.attn.num_kv_heads
        key_rep = _repeat_kv(k_padded, n_rep)
        val_rep = _repeat_kv(v_padded, n_rep)

        # Stage 5: Attention scores  Q @ K^T * scale
        attn_scores = torch.matmul(
            q_rot, key_rep.transpose(-2, -1)
        ) * self.attn.scale

        # Stage 6: Causal mask
        attn_masked = attn_scores + causal_mask.to(MODEL_DTYPE)

        # Stage 7: Softmax
        attn_weights = torch.softmax(attn_masked, dim=-1)

        # Stage 8: V aggregation  weights @ V
        context = torch.matmul(attn_weights, val_rep)

        # Stage 9: Gate(sigmoid) * flatten + O_proj
        attn_flat = context.transpose(1, 2).contiguous().flatten(2, 3)
        gated     = attn_flat * torch.sigmoid(gate)
        projected = self.attn.o_proj(
            gated.permute(0, 2, 1).unsqueeze(2)
        ).squeeze(2).permute(0, 2, 1)

        # Stage 10: Residual
        after_attn = hidden_states + projected

        return (
            x,              # 0  layernorm output        (1, S, hidden)
            q_rot,          # 1  Q (proj+norm+RoPE)      (1, H, S, d)
            k_rot,          # 2  K (proj+norm+RoPE)      (1, Hkv, S, d)
            v,              # 3  V (proj)                 (1, Hkv, S, d)
            attn_scores,    # 4  Q@K^T*scale             (1, H, S, CTX)
            attn_weights,   # 5  softmax                 (1, H, S, CTX)
            context,        # 6  weights@V               (1, H, S, d)
            gated,          # 7  sigmoid(gate)*flat      (1, S, H*d)
            projected,      # 8  o_proj                  (1, S, hidden)
            after_attn,     # 9  residual                (1, S, hidden)
        )


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def cmp(a, b):
    af = a.astype(np.float32).flatten()
    bf = b.astype(np.float32).flatten()
    d = af - bf
    mad  = float(np.max(np.abs(d)))
    mean = float(np.mean(np.abs(d)))
    dot  = float(np.dot(af, bf))
    na   = float(np.linalg.norm(af))
    nb   = float(np.linalg.norm(bf))
    cos  = dot / (na * nb + 1e-12)
    return mad, mean, cos


def analyze_stage(name, g_all, a_all, token_dim, heads_dim=None):
    """Detailed GPU-vs-ANE comparison on valid tokens for one stage."""
    g = g_all.astype(np.float32)
    a = a_all.astype(np.float32)

    # ── valid-token slice ──
    vs = [slice(None)] * g.ndim
    vs[token_dim] = slice(0, VALID_LEN)
    gv = g[tuple(vs)]
    av = a[tuple(vs)]

    diff = gv - av
    mad  = float(np.max(np.abs(diff)))
    mean = float(np.mean(np.abs(diff)))
    dot  = float(np.sum(gv * av))
    ng   = float(np.linalg.norm(gv.flatten()))
    na_  = float(np.linalg.norm(av.flatten()))
    cos  = dot / (ng * na_ + 1e-12)

    print(f"\n  [{name}]  shape={g_all.shape}")
    print(f"    MAD={mad:.6f}  meanAD={mean:.8f}  cos={cos:.6f}")
    print(f"    GPU valid range: [{gv.min():.5f}, {gv.max():.5f}]")
    print(f"    ANE valid range: [{av.min():.5f}, {av.max():.5f}]")

    # ── per-token ──
    tok_mads = []
    for t in range(VALID_LEN):
        s = [slice(None)] * g.ndim
        s[token_dim] = t
        td = float(np.max(np.abs(g[tuple(s)] - a[tuple(s)])))
        tok_mads.append(td)
    wt = int(np.argmax(tok_mads))
    print(f"    per-token MAD (worst t{wt}={tok_mads[wt]:.6f}):")
    for t, m in enumerate(tok_mads):
        mk = " <<<" if t == wt else ""
        print(f"      t{t}: {m:.6f}{mk}")

    # ── per-head ──
    if heads_dim is not None:
        nh = gv.shape[heads_dim]
        h_mads = []
        for h in range(nh):
            s = [slice(None)] * gv.ndim
            s[heads_dim] = h
            hd = float(np.max(np.abs(gv[tuple(s)] - av[tuple(s)])))
            h_mads.append(hd)
        wh = int(np.argmax(h_mads))
        print(f"    per-head MAD (worst h{wh}={h_mads[wh]:.6f}):")
        for h, m in enumerate(h_mads):
            mk = " <<<" if h == wh else ""
            print(f"      h{h:2d}: {m:.6f}{mk}")

    return mad, mean, cos


def analyze_softmax(g_scores, a_scores, g_weights, a_weights):
    """Extra softmax analysis: entropy, max-weight, score range."""
    gs = g_scores.astype(np.float32)
    a_s = a_scores.astype(np.float32)
    gw = g_weights.astype(np.float32)
    aw = a_weights.astype(np.float32)

    print(f"\n  [SOFTMAX DEEP DIVE]")
    # Score range (pre-softmax, pre-mask)
    for t in range(VALID_LEN):
        gs_t = gs[0, :, t, :t+1]   # valid keys only (0..t)
        as_t = a_s[0, :, t, :t+1]
        print(f"    t{t} scores(valid keys 0..{t}): "
              f"GPU [{gs_t.min():.3f},{gs_t.max():.3f}]  "
              f"ANE [{as_t.min():.3f},{as_t.max():.3f}]  "
              f"diff_range [{(gs_t-as_t).min():.5f},{(gs_t-as_t).max():.5f}]")

    # Max attention weight per (head, token)
    print(f"\n    Max softmax weight per valid token (across heads):")
    for t in range(VALID_LEN):
        gw_t = gw[0, :, t, :]           # (H, CTX)
        aw_t = aw[0, :, t, :]
        gmax = gw_t.max(axis=-1)        # (H,)
        amax = aw_t.max(axis=-1)
        gpos = gw_t.argmax(axis=-1)
        apos = aw_t.argmax(axis=-1)
        for h in range(gw_t.shape[0]):
            if abs(gmax[h] - amax[h]) > 0.01 or gpos[h] != apos[h]:
                print(f"      t{t} h{h:2d}: GPU max={gmax[h]:.5f}@pos{gpos[h]}  "
                      f"ANE max={amax[h]:.5f}@pos{apos[h]}  diff={gmax[h]-amax[h]:.5f}")

    # Attention entropy per token
    print(f"\n    Softmax entropy per valid token (mean over heads):")
    for t in range(VALID_LEN):
        gw_t = gw[0, :, t, :t+1]     # only valid key positions
        aw_t = aw[0, :, t, :t+1]
        # clip to avoid log(0)
        gw_t = np.clip(gw_t, 1e-10, 1.0)
        aw_t = np.clip(aw_t, 1e-10, 1.0)
        g_ent = -np.sum(gw_t * np.log(gw_t), axis=-1).mean()  # mean over heads
        a_ent = -np.sum(aw_t * np.log(aw_t), axis=-1).mean()
        print(f"      t{t}: GPU_H={g_ent:.5f}  ANE_H={a_ent:.5f}  diff={g_ent-a_ent:.6f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("LAYER 3 FULL-ATTENTION: STAGE-BY-STAGE GPU vs ANE DIAGNOSTIC")
    print(f"  batch={BATCH}, ctx={CTX}, valid_len={VALID_LEN}, fp16 (no LUT)")
    print("=" * 70)

    # ── Load model ──
    print("\n[0] Loading model...", end="", flush=True)
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length   = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f" {time.time()-t0:.1f}s")

    # ── Tokenize ──
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
        tokens = tok.encode(PROMPT)
    except ImportError:
        # Fallback: pre-computed tokens for "Hello, how are you doing today?"
        tokens = [9707, 11, 1246, 525, 498, 3727, 3351, 30]
    print(f"  Prompt: '{PROMPT}' → {len(tokens)} tokens: {tokens}")
    actual_valid = len(tokens)
    assert actual_valid <= VALID_LEN, f"Prompt has {actual_valid} tokens, expected ≤ {VALID_LEN}"

    # ── Compute Layer 3 input via PyTorch layers 0-2 ──
    print("[1] Computing Layer 3 input (PyTorch layers 0–2)...", end="", flush=True)
    t0 = time.time()
    input_ids   = torch.zeros(1, BATCH, dtype=torch.long)
    input_ids[0, :actual_valid] = torch.tensor(tokens, dtype=torch.long)
    position_ids = torch.zeros(BATCH, dtype=torch.long)
    position_ids[:actual_valid] = torch.arange(actual_valid)

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        for li in range(3):
            hidden = model.model.layers[li](hidden, None, position_ids)
        # Apply padding mask (matches export-path mitigation)
        pos_mask = (torch.arange(BATCH) < actual_valid).to(MODEL_DTYPE).reshape(1, BATCH, 1)
        hidden = hidden * pos_mask

    layer3_input_np = hidden.detach().numpy().astype(np.float16)
    print(f" {time.time()-t0:.1f}s")
    valid_range = hidden[0, :actual_valid]
    print(f"  Layer 3 input: shape={hidden.shape}, "
          f"valid range=[{valid_range.min():.4f}, {valid_range.max():.4f}]")

    # ── Build causal mask ──
    causal_mask_np = np.full((1, 1, BATCH, CTX), -65504.0, dtype=np.float16)
    for i in range(actual_valid):
        causal_mask_np[0, 0, i, :i+1] = 0.0
    # padding rows (i >= actual_valid): all -65504

    # ── Export diagnostic model ──
    print("\n[2] Exporting Layer 3 diagnostic model (fp16, 10 outputs)...")
    t0 = time.time()

    diag = Layer3DiagModel(model.model.layers[3], BATCH, CTX)
    diag.eval()

    trace_h  = torch.from_numpy(layer3_input_np)
    trace_p  = position_ids.int()
    trace_m  = torch.from_numpy(causal_mask_np)

    traced = torch.jit.trace(diag, (trace_h, trace_p, trace_m))

    hidden_size = cfg.hidden_size
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, BATCH, hidden_size)),
            ct.TensorType(name="position_ids",  shape=(BATCH,), dtype=np.int32),
            ct.TensorType(name="causal_mask",   shape=(1, 1, BATCH, CTX)),
        ],
        outputs=[
            ct.TensorType(name="out_layernorm"),
            ct.TensorType(name="out_q_rot"),
            ct.TensorType(name="out_k_rot"),
            ct.TensorType(name="out_v"),
            ct.TensorType(name="out_attn_scores"),
            ct.TensorType(name="out_attn_weights"),
            ct.TensorType(name="out_context"),
            ct.TensorType(name="out_gated"),
            ct.TensorType(name="out_projected"),
            ct.TensorType(name="out_after_attn"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    out_path = os.path.join(OUT_DIR, "layer3_diag.mlpackage")
    os.makedirs(OUT_DIR, exist_ok=True)
    ml.save(out_path)
    export_t = time.time() - t0
    print(f"  Exported in {export_t:.1f}s")
    del ml, traced, diag; gc.collect()

    # ── Predict: GPU then ANE ──
    inputs = {
        "hidden_states": layer3_input_np,
        "position_ids":  position_ids.numpy().astype(np.int32),
        "causal_mask":   causal_mask_np,
    }

    print("\n[3] GPU predict...", end="", flush=True)
    t0 = time.time()
    gpu_m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    gpu = gpu_m.predict(inputs)
    print(f" {time.time()-t0:.1f}s")
    del gpu_m; gc.collect()

    print("[4] ANE predict...", end="", flush=True)
    t0 = time.time()
    ane_m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    ane = ane_m.predict(inputs)
    print(f" {time.time()-t0:.1f}s")
    del ane_m; gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # Stage-by-stage comparison (valid tokens only)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STAGE-BY-STAGE COMPARISON  (valid tokens 0..%d)" % (actual_valid - 1))
    print("=" * 70)

    # (name, output_key, token_dim, heads_dim_or_None)
    STAGES = [
        ("layernorm",            "out_layernorm",     1, None),
        ("Q (proj+norm+RoPE)",   "out_q_rot",         2, 1),
        ("K (proj+norm+RoPE)",   "out_k_rot",         2, 1),
        ("V (proj)",             "out_v",             2, 1),
        ("attn_scores Q@K^T*s",  "out_attn_scores",   2, 1),
        ("softmax",              "out_attn_weights",  2, 1),
        ("context (w@V)",        "out_context",       2, 1),
        ("gate*sigmoid+flat",    "out_gated",         1, None),
        ("o_proj",               "out_projected",     1, None),
        ("after_residual",       "out_after_attn",    1, None),
    ]

    results = {}
    for name, key, td, hd in STAGES:
        mad, mean, cos = analyze_stage(name, gpu[key], ane[key], td, hd)
        results[name] = (mad, mean, cos)

    # ── Softmax deep dive ──
    print("\n" + "=" * 70)
    print("SOFTMAX DEEP DIVE")
    print("=" * 70)
    analyze_softmax(
        gpu["out_attn_scores"], ane["out_attn_scores"],
        gpu["out_attn_weights"], ane["out_attn_weights"],
    )

    # ═══════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY TABLE  (valid tokens only)")
    print("=" * 70)
    print(f"{'Stage':<25s} {'MAD':>10s} {'mean_AD':>12s} {'cosine':>10s}")
    print("-" * 57)
    prev_mad = 0.0
    for name, (m, mn, c) in results.items():
        jump = ""
        if prev_mad > 0 and m > prev_mad * 2:
            jump = f"  ↑{m/prev_mad:.1f}×"
        print(f"{name:<25s} {m:10.6f} {mn:12.8f} {c:10.6f}{jump}")
        prev_mad = m

    # ── Find first material jump ──
    mads = [(name, m) for name, (m, _, _) in results.items()]
    first_big = None
    for i in range(1, len(mads)):
        prev_name, prev_m = mads[i-1]
        curr_name, curr_m = mads[i]
        if prev_m > 0 and curr_m > prev_m * 3:
            first_big = (curr_name, prev_name, curr_m, prev_m)
            break

    if first_big:
        cn, pn, cm, pm = first_big
        print(f"\n>>> FIRST MATERIAL JUMP: '{cn}' (MAD {cm:.6f}) "
              f"vs prior '{pn}' (MAD {pm:.6f})  —  {cm/pm:.1f}× increase")
    else:
        # Just report the stage with the worst MAD
        worst = max(results.items(), key=lambda x: x[1][0])
        print(f"\n>>> WORST STAGE: '{worst[0]}' MAD={worst[1][0]:.6f}")

    # ── Cleanup ──
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
