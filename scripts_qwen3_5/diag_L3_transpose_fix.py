#!/usr/bin/env python3
"""Test channels-first mitigation for Layer 3 transpose corruption.

Root cause: context.transpose(1,2).flatten(2,3) on matmul output
scrambles data on ANE (cos=0.135).

Proposed fix: context.permute(0,1,3,2).reshape(1,-1,S) — channels-first
path that avoids the problematic H↔S transpose.

This script exports BOTH paths and compares GPU vs ANE for each.
"""

import os, sys, time, gc, shutil
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
OUT_DIR   = "/tmp/diag_L3_fix"
BATCH     = 64
CTX       = 2048
VALID_LEN = 8


class TransposeTestModel(nn.Module):
    """Tests both original and fixed transpose paths side by side."""

    def __init__(self, layer3, batch_size, ctx_len):
        super().__init__()
        self.ln   = layer3.input_layernorm
        self.attn = layer3.self_attn
        self.S    = batch_size
        self.CTX  = ctx_len

    def forward(self, hidden_states, position_ids, causal_mask):
        x = self.ln(hidden_states)
        q, k, v, gate = self.attn._project_qkvg(x)
        q = self.attn.q_norm(q)
        k = self.attn.k_norm(k)
        cos, sin = self.attn.rotary.get(x, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb_prefill(q, k, cos, sin, self.attn.rotary.rotary_dim)
        q_rot = q_rot.to(MODEL_DTYPE)
        k_rot = k_rot.to(MODEL_DTYPE)
        v     = v.to(MODEL_DTYPE)
        gate  = gate.to(MODEL_DTYPE)

        # Attention
        k_padded = F.pad(k_rot, (0, 0, 0, self.CTX - self.S))
        v_padded = F.pad(v,     (0, 0, 0, self.CTX - self.S))
        n_rep   = self.attn.num_heads // self.attn.num_kv_heads
        key_rep = _repeat_kv(k_padded, n_rep)
        val_rep = _repeat_kv(v_padded, n_rep)
        attn_s = torch.matmul(q_rot, key_rep.transpose(-2, -1)) * self.attn.scale
        attn_m = attn_s + causal_mask.to(MODEL_DTYPE)
        attn_w = torch.softmax(attn_m, dim=-1)
        context = torch.matmul(attn_w, val_rep)  # (1, H, S, D)

        # ════════ PATH A: Original (broken on ANE) ════════
        # (1, H, S, D) → transpose(1,2) → (1, S, H, D) → flatten → (1, S, H*D)
        attn_flat_A = context.transpose(1, 2).contiguous().flatten(2, 3)
        gated_A     = attn_flat_A * torch.sigmoid(gate)
        proj_A      = self.attn.o_proj(
            gated_A.permute(0, 2, 1).unsqueeze(2)
        ).squeeze(2).permute(0, 2, 1)
        out_A       = hidden_states + proj_A

        # ════════ PATH B: Channels-first fix ════════
        # (1, H, S, D) → permute(0,1,3,2) → (1, H, D, S) → reshape → (1, H*D, S)
        context_cf = context.permute(0, 1, 3, 2).reshape(1, -1, self.S)  # (1, H*D, S)
        gate_cf    = gate.permute(0, 2, 1)  # (1, H*D, S)
        gated_cf   = context_cf * torch.sigmoid(gate_cf)
        proj_cf    = self.attn.o_proj(
            gated_cf.unsqueeze(2)             # (1, H*D, 1, S) → Conv2d
        ).squeeze(2).permute(0, 2, 1)        # → (1, S, hidden)
        out_B      = hidden_states + proj_cf

        return (
            context,       # 0: raw context (1, H, S, D) — reference
            attn_flat_A,   # 1: path A transpose+flatten (1, S, H*D) — BROKEN?
            context_cf,    # 2: path B channels-first (1, H*D, S) — hopefully CLEAN
            gated_A,       # 3: path A gated
            gated_cf,      # 4: path B gated (channels-first)
            out_A,         # 5: path A final output
            out_B,         # 6: path B final output
        )


def cmp(a, b):
    af = a.astype(np.float32).flatten()
    bf = b.astype(np.float32).flatten()
    d = af - bf
    return float(np.max(np.abs(d))), float(np.mean(np.abs(d))), \
           float(np.dot(af, bf)) / (float(np.linalg.norm(af)) * float(np.linalg.norm(bf)) + 1e-12)


def main():
    print("=" * 70)
    print("TRANSPOSE FIX TEST: channels-first vs original")
    print("=" * 70)

    # Load model
    print("\n[0] Loading model...", end="", flush=True)
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX; cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f" {time.time()-t0:.1f}s")

    # Tokenize + Layer 3 input
    try:
        from transformers import AutoTokenizer
        tokens = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True).encode(
            "Hello, how are you doing today?")
    except ImportError:
        tokens = [9419, 11, 1204, 513, 488, 3604, 3242, 30]
    V = len(tokens)

    input_ids = torch.zeros(1, BATCH, dtype=torch.long)
    input_ids[0, :V] = torch.tensor(tokens, dtype=torch.long)
    pos_ids = torch.zeros(BATCH, dtype=torch.long)
    pos_ids[:V] = torch.arange(V)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        for li in range(3):
            hidden = model.model.layers[li](hidden, None, pos_ids)
        hidden = hidden * (torch.arange(BATCH) < V).to(MODEL_DTYPE).reshape(1, BATCH, 1)
    layer3_in = hidden.detach().numpy().astype(np.float16)

    mask = np.full((1, 1, BATCH, CTX), -65504.0, dtype=np.float16)
    for i in range(V):
        mask[0, 0, i, :i+1] = 0.0

    # Verify on CPU first: both paths give same result
    print("[1] CPU verification...", end="", flush=True)
    diag = TransposeTestModel(model.model.layers[3], BATCH, CTX)
    diag.eval()
    with torch.no_grad():
        cpu_out = diag(torch.from_numpy(layer3_in), pos_ids.int(), torch.from_numpy(mask))
        # attn_flat_A (1,S,H*D) vs context_cf (1,H*D,S) — should be transposes of each other
        a_flat = cpu_out[1].numpy()  # (1, S, H*D)
        c_cf   = cpu_out[2].numpy()  # (1, H*D, S)
        # Compare: a_flat[0, s, :] should equal c_cf[0, :, s]
        diff = float(np.max(np.abs(a_flat[0].T - c_cf[0])))
        print(f" paths agree within {diff:.8f}")

    # Export
    print("[2] Exporting...", end="", flush=True)
    t0 = time.time()
    traced = torch.jit.trace(diag, (
        torch.from_numpy(layer3_in), pos_ids.int(), torch.from_numpy(mask)))
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, BATCH, cfg.hidden_size)),
            ct.TensorType(name="position_ids",  shape=(BATCH,), dtype=np.int32),
            ct.TensorType(name="causal_mask",   shape=(1, 1, BATCH, CTX)),
        ],
        outputs=[
            ct.TensorType(name="context"),
            ct.TensorType(name="attn_flat_A"),
            ct.TensorType(name="context_cf_B"),
            ct.TensorType(name="gated_A"),
            ct.TensorType(name="gated_cf_B"),
            ct.TensorType(name="out_A"),
            ct.TensorType(name="out_B"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    out_path = os.path.join(OUT_DIR, "fix_test.mlpackage")
    os.makedirs(OUT_DIR, exist_ok=True)
    ml.save(out_path)
    print(f" {time.time()-t0:.1f}s")
    del ml, traced, diag; gc.collect()

    # Predict
    inputs = {
        "hidden_states": layer3_in,
        "position_ids":  pos_ids.numpy().astype(np.int32),
        "causal_mask":   mask,
    }
    print("[3] GPU predict...", end="", flush=True)
    gpu_m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    gpu = gpu_m.predict(inputs)
    print(f" OK"); del gpu_m; gc.collect()

    print("[4] ANE predict...", end="", flush=True)
    ane_m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    ane = ane_m.predict(inputs)
    print(f" OK"); del ane_m; gc.collect()

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON (valid tokens only, 0..%d)" % (V-1))
    print("=" * 70)

    def report(name, g_arr, a_arr, token_dim):
        g = g_arr.astype(np.float32)
        a = a_arr.astype(np.float32)
        vs = [slice(None)] * g.ndim
        vs[token_dim] = slice(0, V)
        gv, av = g[tuple(vs)], a[tuple(vs)]
        mad, mean, cos = cmp(gv, av)
        print(f"  {name:<30s}  MAD={mad:10.6f}  mean={mean:12.8f}  cos={cos:.6f}")
        print(f"    GPU range: [{gv.min():.5f}, {gv.max():.5f}]")
        print(f"    ANE range: [{av.min():.5f}, {av.max():.5f}]")
        return mad, cos

    context_mad, context_cos = report("context (w@V)", gpu["context"], ane["context"], 2)

    print("\n  ── PATH A: Original (transpose+flatten) ──")
    a_flat_mad, a_flat_cos = report("attn_flat_A (broken?)", gpu["attn_flat_A"], ane["attn_flat_A"], 1)
    a_gated_mad, a_gated_cos = report("gated_A", gpu["gated_A"], ane["gated_A"], 1)
    a_out_mad, a_out_cos = report("out_A (residual)", gpu["out_A"], ane["out_A"], 1)

    print("\n  ── PATH B: Channels-first fix ──")
    b_cf_mad, b_cf_cos = report("context_cf_B (fix?)", gpu["context_cf_B"], ane["context_cf_B"], 2)
    b_gated_mad, b_gated_cos = report("gated_cf_B", gpu["gated_cf_B"], ane["gated_cf_B"], 2)
    b_out_mad, b_out_cos = report("out_B (residual)", gpu["out_B"], ane["out_B"], 1)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Path A (original):  attn_flat cos={a_flat_cos:.6f}  output MAD={a_out_mad:.6f}")
    print(f"  Path B (CF fix):    context_cf cos={b_cf_cos:.6f}  output MAD={b_out_mad:.6f}")

    if b_cf_cos > 0.99 and a_flat_cos < 0.5:
        print(f"\n  >>> CHANNELS-FIRST FIX WORKS!")
        print(f"      Path A cos={a_flat_cos:.4f} (broken) → Path B cos={b_cf_cos:.4f} (fixed)")
        print(f"      Output MAD: {a_out_mad:.4f} → {b_out_mad:.4f}")
    elif b_cf_cos > a_flat_cos:
        print(f"\n  >>> PARTIAL IMPROVEMENT: cos {a_flat_cos:.4f} → {b_cf_cos:.4f}")
    else:
        print(f"\n  >>> FIX DID NOT HELP")

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print("\nDone.")

if __name__ == "__main__":
    main()
