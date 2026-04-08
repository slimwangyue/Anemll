#!/usr/bin/env python3
"""Full-model precision sensitivity map (disk-efficient).

Converts one chunk at a time, runs prediction, caches numpy output,
then discards the CoreML model to stay within disk limits.

Phase A:  isolated per-chunk error + cascaded end-to-end
Phase A3: one-chunk-FP32 marginal gains (ranking)
Phase B:  combination tests
Phase B2: per-layer drill-down of top-2 sensitive chunks
"""
import os, sys, time, warnings, gc, shutil, glob, tempfile
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct
from config import CTX, NUM_CHUNKS
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape, MODEL_DTYPE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

HF_PATH = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"
TEMP_CACHE = "/var/folders/r5/dn4v9jhx1cvg33xnvt8nmjxr0000gn/T"

def cos_sim(a, b):
    a64 = np.asarray(a).flatten().astype(np.float64)
    b64 = np.asarray(b).flatten().astype(np.float64)
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))

def max_diff(a, b):
    return float(np.abs(np.asarray(a).astype(np.float64) - np.asarray(b).astype(np.float64)).max())

def chunk_bounds(total, n):
    base, rem = divmod(total, n)
    return [(c * base + min(c, rem), c * base + min(c, rem) + base + (1 if c < rem else 0)) for c in range(n)]

def cleanup_temp():
    for pat in ["*.mlmodelc", "tmp*.mlpackage"]:
        for p in glob.glob(os.path.join(TEMP_CACHE, pat)):
            shutil.rmtree(p, ignore_errors=True)

def get_state_shapes(tcfg, n):
    cd = tcfg.linear_num_key_heads * tcfg.linear_key_head_dim * 2 + tcfg.linear_num_value_heads * tcfg.linear_value_head_dim
    ck = max(1, int(tcfg.linear_conv_kernel_dim))
    a1, a2 = ane_conv_state_shape(cd, ck)
    return (n, a1, a2), (n, tcfg.linear_num_value_heads, tcfg.linear_key_head_dim, tcfg.linear_value_head_dim)


class ChunkWrapper(torch.nn.Module):
    def __init__(self, model, cfg, s, e):
        super().__init__()
        self.model, self.s, self.e = model, s, e
        n = e - s
        self.register_buffer("k_cache", torch.zeros(n, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE))
        self.register_buffer("v_cache", torch.zeros(n, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE))
        self.states = Qwen35Converter.GetChunkLocalTransformerStates(model, n, prefix="", split_full_attention_kv=True)

    def forward(self, hs, pid, cm, cp, lc, lr):
        out = self.model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hs, position_ids=pid, causal_mask=cm, current_pos=cp,
            kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
            linear_conv_state=lc, linear_recurrent_state=lr,
            start_layer=self.s, end_layer=self.e, apply_final_norm=False)
        return out, lc, lr


def convert_and_run(model, cfg, tcfg, start, end, precision, hidden_np, mask_np, pos_np):
    """Convert chunk → save to disk → reload → predict → cleanup → return hidden_states numpy."""
    n = end - start
    w = ChunkWrapper(model, cfg, start, end).eval()
    cs, rs = get_state_shapes(tcfg, n)

    h = torch.zeros(1, 1, cfg.hidden_size, dtype=torch.float16)
    pid = torch.zeros(1, dtype=torch.int32)
    mk = torch.zeros(1, 1, 1, CTX, dtype=torch.float16)
    cp = torch.zeros(1, dtype=torch.int32)
    lc = torch.zeros(cs, dtype=torch.float16)
    lr = torch.zeros(rs, dtype=torch.float16)

    w.k_cache.zero_(); w.v_cache.zero_()
    tr = torch.jit.trace(w, (h, pid, mk, cp, lc, lr), check_trace=False)
    w.k_cache.zero_(); w.v_cache.zero_()
    for _, buf in tr.named_buffers(): buf.zero_()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml = ct.convert(tr,
            inputs=[ct.TensorType(name="hidden_states", shape=h.shape, dtype=np.float16),
                    ct.TensorType(name="position_ids", shape=pid.shape, dtype=np.int32),
                    ct.TensorType(name="causal_mask", shape=mk.shape, dtype=np.float16),
                    ct.TensorType(name="current_pos", shape=cp.shape, dtype=np.int32),
                    ct.TensorType(name="linear_conv_state", shape=lc.shape, dtype=np.float16),
                    ct.TensorType(name="linear_recurrent_state", shape=lr.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16),
                     ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
                     ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16)],
            states=w.states,
            compute_precision=precision,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram")
    del w, tr; gc.collect()

    # Save to disk and reload — make_state() is only reliable on loaded models
    tmpdir = tempfile.mkdtemp(prefix="diag_chunk_")
    tmppath = os.path.join(tmpdir, "model.mlpackage")
    ml.save(tmppath)
    del ml; gc.collect()
    ml2 = ct.models.MLModel(tmppath, compute_units=ct.ComputeUnit.CPU_ONLY)

    feed = {"hidden_states": hidden_np.astype(np.float16),
            "position_ids": pos_np.copy(), "causal_mask": mask_np.copy(),
            "current_pos": pos_np.copy(),
            "linear_conv_state": np.zeros(cs, dtype=np.float16),
            "linear_recurrent_state": np.zeros(rs, dtype=np.float16)}
    state = ml2.make_state()
    out = ml2.predict(feed, state=state)
    result = out["output_hidden_states"].copy()
    del ml2, feed, state, out; gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)
    cleanup_temp()
    return result


def main():
    print("=" * 72)
    print("  FULL-MODEL PRECISION SENSITIVITY MAP")
    print("=" * 72)
    cleanup_temp()

    print("\n[1] Loading PyTorch model...")
    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX; cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval().half()
    for p in model.parameters(): p.requires_grad = False
    tcfg = cfg.text_config

    bounds = chunk_bounds(cfg.num_hidden_layers, NUM_CHUNKS)
    for ci, (s, e) in enumerate(bounds):
        types = ''.join('L' if tcfg.layer_types[i] == 'linear_attention' else 'F' for i in range(s, e))
        print(f"  chunk{ci}: layers {s}-{e-1}  [{types}]")

    # PyTorch ground truth
    print("\n[2] PyTorch ground truth cascade...")
    with torch.no_grad():
        embed = model.model.embed_tokens(torch.tensor([[9906]], dtype=torch.long)).half()
    pos_np = np.array([0], dtype=np.int32)
    mask_np = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16); mask_np[:, :, :, :1] = 0
    pid_t = torch.zeros(1, dtype=torch.int32)
    mask_t = torch.from_numpy(mask_np)
    cp_t = torch.zeros(1, dtype=torch.int32)

    pt_ckpts = [embed.numpy().copy()]
    h = embed.clone()
    with torch.no_grad():
        for ci, (s, e) in enumerate(bounds):
            n = e - s; cs, rs = get_state_shapes(tcfg, n)
            lc = torch.zeros(cs, dtype=torch.float16); lr = torch.zeros(rs, dtype=torch.float16)
            kc = torch.zeros(n, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16)
            vc = torch.zeros(n, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16)
            h = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=h.clone(), position_ids=pid_t.clone(), causal_mask=mask_t.clone(),
                current_pos=cp_t.clone(), kv_cache_0=None, k_cache=kc, v_cache=vc,
                linear_conv_state=lc, linear_recurrent_state=lr,
                start_layer=s, end_layer=e, apply_final_norm=False)
            pt_ckpts.append(h.numpy().copy())
            print(f"  chunk{ci} [{s:2d}-{e-1:2d}]: norm={np.linalg.norm(h.numpy()):.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  PHASE A — Isolated per-chunk error (HARDCODED from prior run)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PHASE A: ISOLATED PER-CHUNK ERROR (from prior run)")
    print("=" * 72)

    iso = {
        (0, "fp16"): 0.9636, (0, "fp32"): 0.99999,
        (1, "fp16"): 0.9749, (1, "fp32"): 0.99999,
        (2, "fp16"): 0.9798, (2, "fp32"): 0.99997,
        (3, "fp16"): 0.3748, (3, "fp32"): 0.9896,
        (4, "fp16"): 0.9821, (4, "fp32"): 0.99999,
        (5, "fp16"): 0.8656, (5, "fp32"): 0.99999,
    }
    for ci in range(NUM_CHUNKS):
        s, e = bounds[ci]
        types = ''.join('L' if tcfg.layer_types[i] == 'linear_attention' else 'F' for i in range(s, e))
        r = (1 - iso[(ci,"fp16")]) / (1 - iso[(ci,"fp32")] + 1e-30)
        print(f"  chunk{ci} [{s:2d}-{e-1:2d}] {types} FP16={iso[(ci,'fp16')]:.4f}  FP32={iso[(ci,'fp32')]:.5f}  ratio={r:.0f}×")

    # ═══════════════════════════════════════════════════════════════
    #  PHASE A2+A3+B UNIFIED — Cascaded tests with mixed precision
    #  Ordered by priority. Each cascade = 6 convert_and_run calls.
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  CASCADED TESTS (mixed-precision)")
    print("=" * 72)

    # Define test configs: (label, set_of_chunks_at_FP32)
    cascade_configs = [
        ("ALL FP16",            set()),
        ("ALL FP32",            {0, 1, 2, 3, 4, 5}),
        ("ch3 → FP32",         {3}),
        ("ch5 → FP32",         {5}),
        ("ch3+5 → FP32",       {3, 5}),
        ("ch3+5+0 → FP32",     {0, 3, 5}),
        ("ch0+1+3+5 → FP32",   {0, 1, 3, 5}),
    ]

    cascade_results = {}
    for label, fp32_set in cascade_configs:
        t0 = time.time()
        h_np = pt_ckpts[0].copy()
        per_chunk_cos = []
        for ci, (s, e) in enumerate(bounds):
            prec = ct.precision.FLOAT32 if ci in fp32_set else ct.precision.FLOAT16
            h_np = convert_and_run(model, cfg, tcfg, s, e, prec, h_np, mask_np, pos_np)
            c = cos_sim(pt_ckpts[ci+1], h_np)
            per_chunk_cos.append(c)
        final_cos = cos_sim(pt_ckpts[-1], h_np)
        dt = time.time() - t0
        fp32_count = len(fp32_set)
        cascade_results[label] = {"final_cos": final_cos, "per_chunk": per_chunk_cos, "fp32_count": fp32_count}
        print(f"\n  {label} ({fp32_count}/6 FP32):")
        for ci2, cc in enumerate(per_chunk_cos):
            print(f"    after chunk{ci2}: cos={cc:.10f}")
        print(f"    FINAL: cos={final_cos:.10f}  err={1-final_cos:.2e}  ({dt:.0f}s)")

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)

    baseline_err = 1 - cascade_results["ALL FP16"]["final_cos"]
    print(f"\n  {'Config':28s} {'FP32':>4s} {'Final cos':>14s} {'Error':>12s} {'Recovery':>10s}")
    print(f"  {'─'*28} {'─'*4} {'─'*14} {'─'*12} {'─'*10}")
    for label, fp32_set in cascade_configs:
        r = cascade_results[label]
        err = 1 - r["final_cos"]
        recovery = (baseline_err - err) / baseline_err * 100 if baseline_err > 0 else 0
        print(f"  {label:28s} {r['fp32_count']:>4d} {r['final_cos']:14.10f} {err:12.2e} {recovery:+9.1f}%")

    # ═══════════════════════════════════════════════════════════════
    #  PHASE B2 — Per-layer drill-down of chunk3 (most sensitive)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PHASE B2: PER-LAYER DRILL-DOWN (chunk3)")
    print("=" * 72)

    ci_t = 3  # chunk3 is the most catastrophic
    cs_l, ce_l = bounds[ci_t]
    print(f"\n  chunk{ci_t} [{cs_l}-{ce_l-1}]: isolated single-layer FP16 vs FP32")

    # Per-layer PyTorch checkpoints within chunk3
    lck = {}
    with torch.no_grad():
        h_pt = torch.from_numpy(pt_ckpts[ci_t]).half()
        lck[cs_l] = h_pt.numpy().copy()
        for li in range(cs_l, ce_l):
            lcs, lrs = get_state_shapes(tcfg, 1)
            lc2 = torch.zeros(lcs, dtype=torch.float16); lr2 = torch.zeros(lrs, dtype=torch.float16)
            kc2 = torch.zeros(1, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16)
            vc2 = torch.zeros(1, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16)
            h_pt = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=h_pt.clone(), position_ids=pid_t.clone(), causal_mask=mask_t.clone(),
                current_pos=cp_t.clone(), kv_cache_0=None, k_cache=kc2, v_cache=vc2,
                linear_conv_state=lc2, linear_recurrent_state=lr2,
                start_layer=li, end_layer=li+1, apply_final_norm=False)
            lck[li+1] = h_pt.numpy().copy()

    for li in range(cs_l, ce_l):
        lt = tcfg.layer_types[li]
        tag = "Lin" if lt == "linear_attention" else "Ful"
        for ps, prec in [("FP16", ct.precision.FLOAT16), ("FP32", ct.precision.FLOAT32)]:
            out_l = convert_and_run(model, cfg, tcfg, li, li+1, prec, lck[li], mask_np, pos_np)
            c = cos_sim(lck[li+1], out_l)
            md = max_diff(lck[li+1], out_l)
            print(f"    layer {li:2d} ({tag}) {ps}: cos={c:.10f}  max_diff={md:.6f}")

    # ═══════════════════════════════════════════════════════════════
    #  FINAL SENSITIVITY MAP
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  FINAL SENSITIVITY MAP")
    print("=" * 72)
    print(f"\n  {'Chunk':>6} {'Layers':>7} {'Types':>8} {'Iso FP16':>10} {'Iso FP32':>10} {'Ratio':>7} {'Rating':>8}")
    print(f"  {'─'*6} {'─'*7} {'─'*8} {'─'*10} {'─'*10} {'─'*7} {'─'*8}")
    for ci in range(NUM_CHUNKS):
        s, e = bounds[ci]
        types = ''.join('L' if tcfg.layer_types[i] == 'linear_attention' else 'F' for i in range(s, e))
        fc = iso[(ci,"fp16")]; gc = iso[(ci,"fp32")]
        r = (1-fc) / (1-gc+1e-30)
        sens = "CRITICAL" if fc < 0.5 else "HIGH" if fc < 0.9 else "MODERATE" if fc < 0.97 else "LOW"
        print(f"  ch{ci:1d}   {s:2d}-{e-1:2d}   {types:<8s} {fc:10.4f} {gc:10.5f} {r:5.0f}×  {sens:>8s}")

    print("\n  RECOMMENDED STRATEGY:")
    print("  • chunk3 [17-21] MUST be FP32 — cos=0.375 at FP16 is catastrophic")
    print("  • chunk5 [27-31] SHOULD be FP32 — cos=0.866 at FP16 is very poor")
    print("  • chunks 0,1,2,4 are acceptable at FP16 (cos 0.96-0.98)")
    print("  • Alternative: split sensitive chunks into 1-2 layer sub-chunks")
    print("    (prior ablation showed single-layer FP16 gives cos≈0.99998)")


if __name__ == "__main__":
    main()
