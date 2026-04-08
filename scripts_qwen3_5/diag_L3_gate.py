#!/usr/bin/env python3
"""Focused diagnostic: isolate the gate corruption in Layer 3 full attention.

Previous diagnostic showed the gate application (attn_flat * sigmoid(gate))
has GPU vs ANE MAD=1.50 and cos=0.22, while context (w@V) is clean at MAD=0.02.

This script decomposes the gate path to find exactly which sub-op is broken:
  1. gate_raw   — raw gate values from q_proj upper half
  2. gate_sig   — sigmoid(gate_raw)
  3. attn_flat  — context after transpose+flatten (should match context parity)
  4. gated      — attn_flat * gate_sig (the broken step)
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
OUT_DIR   = "/tmp/diag_L3_gate"
BATCH     = 64
CTX       = 2048
VALID_LEN = 8


class GateDiagModel(nn.Module):
    """Decomposes the gate path with all sub-operations visible."""

    def __init__(self, layer3, batch_size, ctx_len):
        super().__init__()
        self.ln    = layer3.input_layernorm
        self.attn  = layer3.self_attn
        self.S     = batch_size
        self.CTX   = ctx_len

    def forward(self, hidden_states, position_ids, causal_mask):
        x = self.ln(hidden_states)

        # === Q/K/V/gate projection ===
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

        # === Attention ===
        k_padded = F.pad(k_rot, (0, 0, 0, self.CTX - self.S))
        v_padded = F.pad(v,     (0, 0, 0, self.CTX - self.S))
        n_rep    = self.attn.num_heads // self.attn.num_kv_heads
        key_rep  = _repeat_kv(k_padded, n_rep)
        val_rep  = _repeat_kv(v_padded, n_rep)
        attn_s   = torch.matmul(q_rot, key_rep.transpose(-2, -1)) * self.attn.scale
        attn_m   = attn_s + causal_mask.to(MODEL_DTYPE)
        attn_w   = torch.softmax(attn_m, dim=-1)
        context  = torch.matmul(attn_w, val_rep)

        # === Gate decomposition ===
        # attn_flat: (1, H, S, D) → (1, S, H, D) → (1, S, H*D)
        attn_flat = context.transpose(1, 2).contiguous().flatten(2, 3)
        # gate: already (1, S, H*D) from _project_qkvg
        gate_sig  = torch.sigmoid(gate)
        gated     = attn_flat * gate_sig

        # === Also output gate in per-head format for analysis ===
        # gate reshaped: (1, S, 4096) → (1, S, 16, 256) → (1, 16, S, 256)
        gate_per_head = gate.reshape(1, self.S, self.attn.num_heads, self.attn.head_dim)
        gate_per_head = gate_per_head.permute(0, 2, 1, 3)

        return (
            gate,          # 0: raw gate (1, S, H*D=4096)
            gate_sig,      # 1: sigmoid(gate) (1, S, 4096)
            attn_flat,     # 2: context reshaped (1, S, 4096)
            gated,         # 3: attn_flat * gate_sig (1, S, 4096)
            context,       # 4: weights@V (1, H, S, D) for reference
            gate_per_head, # 5: gate in head format (1, H, S, D) for per-head
        )


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


def main():
    print("=" * 70)
    print("GATE DECOMPOSITION: Layer 3 full-attention")
    print(f"  batch={BATCH}, ctx={CTX}, valid_len={VALID_LEN}")
    print("=" * 70)

    # Load model
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

    # Tokenize
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
        tokens = tok.encode("Hello, how are you doing today?")
    except ImportError:
        tokens = [9419, 11, 1204, 513, 488, 3604, 3242, 30]
    actual_valid = len(tokens)
    print(f"  {actual_valid} tokens")

    # Compute Layer 3 input
    print("[1] Layer 3 input...", end="", flush=True)
    input_ids = torch.zeros(1, BATCH, dtype=torch.long)
    input_ids[0, :actual_valid] = torch.tensor(tokens, dtype=torch.long)
    position_ids = torch.zeros(BATCH, dtype=torch.long)
    position_ids[:actual_valid] = torch.arange(actual_valid)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        for li in range(3):
            hidden = model.model.layers[li](hidden, None, position_ids)
        pos_mask = (torch.arange(BATCH) < actual_valid).to(MODEL_DTYPE).reshape(1, BATCH, 1)
        hidden = hidden * pos_mask
    layer3_in = hidden.detach().numpy().astype(np.float16)
    print(f" done")

    # Also compute PyTorch reference for gate
    print("[2] PyTorch reference...", end="", flush=True)
    with torch.no_grad():
        layer3 = model.model.layers[3]
        x_ref = layer3.input_layernorm(hidden)
        q, k, v, gate_ref = layer3.self_attn._project_qkvg(x_ref)
        gate_ref_np = gate_ref.to(MODEL_DTYPE).detach().numpy()
    print(f" gate shape={gate_ref.shape}, range=[{gate_ref.min():.4f}, {gate_ref.max():.4f}]")

    # Build causal mask
    mask = np.full((1, 1, BATCH, CTX), -65504.0, dtype=np.float16)
    for i in range(actual_valid):
        mask[0, 0, i, :i+1] = 0.0

    # Export diagnostic model
    print("\n[3] Exporting gate diagnostic model...", end="", flush=True)
    t0 = time.time()
    diag = GateDiagModel(model.model.layers[3], BATCH, CTX)
    diag.eval()
    hidden_size = cfg.hidden_size
    traced = torch.jit.trace(diag, (
        torch.from_numpy(layer3_in),
        position_ids.int(),
        torch.from_numpy(mask),
    ))
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, BATCH, hidden_size)),
            ct.TensorType(name="position_ids",  shape=(BATCH,), dtype=np.int32),
            ct.TensorType(name="causal_mask",   shape=(1, 1, BATCH, CTX)),
        ],
        outputs=[
            ct.TensorType(name="gate_raw"),
            ct.TensorType(name="gate_sig"),
            ct.TensorType(name="attn_flat"),
            ct.TensorType(name="gated"),
            ct.TensorType(name="context"),
            ct.TensorType(name="gate_per_head"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    out_path = os.path.join(OUT_DIR, "gate_diag.mlpackage")
    os.makedirs(OUT_DIR, exist_ok=True)
    ml.save(out_path)
    print(f" {time.time()-t0:.1f}s")
    del ml, traced, diag; gc.collect()

    # Predict
    inputs = {
        "hidden_states": layer3_in,
        "position_ids":  position_ids.numpy().astype(np.int32),
        "causal_mask":   mask,
    }
    print("[4] GPU predict...", end="", flush=True)
    t0 = time.time()
    gpu_m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    gpu = gpu_m.predict(inputs)
    print(f" {time.time()-t0:.1f}s")
    del gpu_m; gc.collect()

    print("[5] ANE predict...", end="", flush=True)
    t0 = time.time()
    ane_m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    ane = ane_m.predict(inputs)
    print(f" {time.time()-t0:.1f}s")
    del ane_m; gc.collect()

    V = actual_valid

    # ====================================================================
    print("\n" + "=" * 70)
    print("GATE DECOMPOSITION RESULTS  (valid tokens 0..%d)" % (V-1))
    print("=" * 70)

    # --- gate_raw ---
    g_gate = gpu["gate_raw"].astype(np.float32)
    a_gate = ane["gate_raw"].astype(np.float32)
    gv = g_gate[:, :V]; av = a_gate[:, :V]
    mad, mean, cos = cmp(gv, av)
    print(f"\n  [gate_raw]  shape={gpu['gate_raw'].shape}")
    print(f"    GPU-vs-ANE:  MAD={mad:.6f}  mean={mean:.8f}  cos={cos:.6f}")
    print(f"    GPU range: [{gv.min():.5f}, {gv.max():.5f}]")
    print(f"    ANE range: [{av.min():.5f}, {av.max():.5f}]")
    # Compare against PyTorch reference
    ref = gate_ref_np.astype(np.float32)[:, :V]
    mad_gpu_ref, _, cos_gpu_ref = cmp(gv, ref)
    mad_ane_ref, _, cos_ane_ref = cmp(av, ref)
    print(f"    GPU-vs-PyTorch: MAD={mad_gpu_ref:.6f}  cos={cos_gpu_ref:.6f}")
    print(f"    ANE-vs-PyTorch: MAD={mad_ane_ref:.6f}  cos={cos_ane_ref:.6f}")
    # Per-token
    for t in range(V):
        td = float(np.max(np.abs(g_gate[:, t] - a_gate[:, t])))
        tr = float(np.max(np.abs(a_gate[:, t].flatten() - ref[:, t].flatten())))
        print(f"      t{t}: GPU-ANE MAD={td:.6f}  ANE-ref MAD={tr:.6f}")

    # Per-head analysis
    g_ph = gpu["gate_per_head"].astype(np.float32)  # (1, H, S, D)
    a_ph = ane["gate_per_head"].astype(np.float32)
    print(f"\n    Per-head gate_raw analysis (valid tokens):")
    num_heads = g_ph.shape[1]
    head_dim = g_ph.shape[3]
    for h in range(num_heads):
        gv_h = g_ph[0, h, :V]
        av_h = a_ph[0, h, :V]
        hm = float(np.max(np.abs(gv_h - av_h)))
        hd = float(np.dot(gv_h.flatten(), av_h.flatten()))
        hn = float(np.linalg.norm(gv_h.flatten()))
        ha = float(np.linalg.norm(av_h.flatten()))
        hc = hd / (hn * ha + 1e-12)
        print(f"      h{h:2d}: MAD={hm:.6f}  cos={hc:.6f}  "
              f"GPU [{gv_h.min():.4f},{gv_h.max():.4f}]  "
              f"ANE [{av_h.min():.4f},{av_h.max():.4f}]")

    # --- gate_sig ---
    print(f"\n  [gate_sigmoid]  shape={gpu['gate_sig'].shape}")
    g_sig = gpu["gate_sig"].astype(np.float32)[:, :V]
    a_sig = ane["gate_sig"].astype(np.float32)[:, :V]
    mad, mean, cos = cmp(g_sig, a_sig)
    print(f"    GPU-vs-ANE:  MAD={mad:.6f}  mean={mean:.8f}  cos={cos:.6f}")
    print(f"    GPU range: [{g_sig.min():.5f}, {g_sig.max():.5f}]")
    print(f"    ANE range: [{a_sig.min():.5f}, {a_sig.max():.5f}]")

    # --- attn_flat ---
    print(f"\n  [attn_flat]  shape={gpu['attn_flat'].shape}")
    g_af = gpu["attn_flat"].astype(np.float32)[:, :V]
    a_af = ane["attn_flat"].astype(np.float32)[:, :V]
    mad, mean, cos = cmp(g_af, a_af)
    print(f"    GPU-vs-ANE:  MAD={mad:.6f}  mean={mean:.8f}  cos={cos:.6f}")
    print(f"    GPU range: [{g_af.min():.5f}, {g_af.max():.5f}]")
    print(f"    ANE range: [{a_af.min():.5f}, {a_af.max():.5f}]")

    # --- context (reference) ---
    print(f"\n  [context (w@V)]  shape={gpu['context'].shape}")
    g_ctx = gpu["context"].astype(np.float32)[:, :, :V]
    a_ctx = ane["context"].astype(np.float32)[:, :, :V]
    mad, mean, cos = cmp(g_ctx, a_ctx)
    print(f"    GPU-vs-ANE:  MAD={mad:.6f}  mean={mean:.8f}  cos={cos:.6f}")

    # --- gated ---
    print(f"\n  [gated = attn_flat * sigmoid(gate)]  shape={gpu['gated'].shape}")
    g_gated = gpu["gated"].astype(np.float32)[:, :V]
    a_gated = ane["gated"].astype(np.float32)[:, :V]
    mad, mean, cos = cmp(g_gated, a_gated)
    print(f"    GPU-vs-ANE:  MAD={mad:.6f}  mean={mean:.8f}  cos={cos:.6f}")
    print(f"    GPU range: [{g_gated.min():.5f}, {g_gated.max():.5f}]")
    print(f"    ANE range: [{a_gated.min():.5f}, {a_gated.max():.5f}]")

    # --- Identify the corrupted component ---
    print("\n" + "=" * 70)
    print("CORRUPTION LOCALIZATION")
    print("=" * 70)
    gate_ok  = cmp(gpu["gate_raw"][:, :V].astype(np.float32),
                   ane["gate_raw"][:, :V].astype(np.float32))
    sig_ok   = cmp(gpu["gate_sig"][:, :V].astype(np.float32),
                   ane["gate_sig"][:, :V].astype(np.float32))
    flat_ok  = cmp(gpu["attn_flat"][:, :V].astype(np.float32),
                   ane["attn_flat"][:, :V].astype(np.float32))
    gated_ok = cmp(gpu["gated"][:, :V].astype(np.float32),
                   ane["gated"][:, :V].astype(np.float32))

    print(f"  gate_raw:    MAD={gate_ok[0]:.6f}  cos={gate_ok[2]:.6f}")
    print(f"  sigmoid:     MAD={sig_ok[0]:.6f}  cos={sig_ok[2]:.6f}")
    print(f"  attn_flat:   MAD={flat_ok[0]:.6f}  cos={flat_ok[2]:.6f}")
    print(f"  gated:       MAD={gated_ok[0]:.6f}  cos={gated_ok[2]:.6f}")

    if gate_ok[2] < 0.9:
        print("\n  >>> ROOT CAUSE: gate_raw is corrupted on ANE")
        print("      The q_proj upper-half extraction diverges on ANE.")
    elif sig_ok[2] < 0.9:
        print("\n  >>> ROOT CAUSE: sigmoid(gate) is corrupted on ANE")
        print("      The gate values are fine but sigmoid diverges.")
    elif flat_ok[2] < 0.9:
        print("\n  >>> ROOT CAUSE: attn_flat (transpose+flatten) is corrupted on ANE")
        print("      Context is fine but the reshape introduces corruption.")
    elif gated_ok[2] < 0.9 and gate_ok[2] > 0.99 and flat_ok[2] > 0.99:
        print("\n  >>> ROOT CAUSE: the multiply attn_flat*sigmoid(gate) diverges on ANE")
        print("      Both inputs are fine individually but multiplication is corrupted.")
    else:
        print("\n  >>> INCONCLUSIVE — all components have similar divergence")

    # Cleanup
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
