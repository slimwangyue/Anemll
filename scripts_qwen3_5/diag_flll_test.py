#!/usr/bin/env python3
"""Quick follow-up: test the [FLLL] partition that was missing from the main scan.

Boundaries BEFORE each F layer (after layers 2,6,10,14,18,22,26):
  [0-2] LLL, [3-6] FLLL, [7-10] FLLL, [11-14] FLLL,
  [15-18] FLLL, [19-22] FLLL, [23-26] FLLL, [27-31] FLLLF

Also test: [0-2],[3-6],...,[27-30],[31] = 9 chunks (isolate F31)
Also test: [0-2],[3-6],...,[19],[20-22],[23-26],[27-30],[31] = 10 chunks (isolate F19 + F31)
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
from config import CTX
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape, MODEL_DTYPE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

HF_PATH = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"
TEMP_CACHE = "/var/folders/r5/dn4v9jhx1cvg33xnvt8nmjxr0000gn/T"
PRECISION = ct.precision.FLOAT16

def cos_sim(a, b):
    a64 = np.asarray(a).flatten().astype(np.float64)
    b64 = np.asarray(b).flatten().astype(np.float64)
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))

def cleanup_temp():
    for pat in ["*.mlmodelc", "tmp*.mlpackage", "diag_chunk_*"]:
        for p in glob.glob(os.path.join(TEMP_CACHE, pat)):
            shutil.rmtree(p, ignore_errors=True)
    bnns_cache = os.path.expanduser("~/Library/Caches/com.apple.python3/com.apple.e5rt.e5bundlecache")
    if os.path.isdir(bnns_cache):
        shutil.rmtree(bnns_cache, ignore_errors=True)

def get_state_shapes(tcfg, n):
    cd = (tcfg.linear_num_key_heads * tcfg.linear_key_head_dim * 2
          + tcfg.linear_num_value_heads * tcfg.linear_value_head_dim)
    ck = max(1, int(tcfg.linear_conv_kernel_dim))
    a1, a2 = ane_conv_state_shape(cd, ck)
    return ((n, a1, a2),
            (n, tcfg.linear_num_value_heads, tcfg.linear_key_head_dim, tcfg.linear_value_head_dim))

def layer_tag(layer_types, i):
    return "F" if layer_types[i] != "linear_attention" else "L"


class ChunkWrapper(torch.nn.Module):
    def __init__(self, model, cfg, s, e):
        super().__init__()
        self.model, self.s, self.e = model, s, e
        n = e - s
        self.register_buffer("k_cache",
            torch.zeros(n, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE))
        self.register_buffer("v_cache",
            torch.zeros(n, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE))
        self.states = Qwen35Converter.GetChunkLocalTransformerStates(
            model, n, prefix="", split_full_attention_kv=True)

    def forward(self, hs, pid, cm, cp, lc, lr):
        out = self.model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hs, position_ids=pid, causal_mask=cm, current_pos=cp,
            kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
            linear_conv_state=lc, linear_recurrent_state=lr,
            start_layer=self.s, end_layer=self.e, apply_final_norm=False)
        return out, lc, lr


def convert_and_run(model, cfg, tcfg, start, end, hidden_np, mask_np, pos_np):
    n = end - start
    w = ChunkWrapper(model, cfg, start, end).eval()
    cs, rs = get_state_shapes(tcfg, n)
    h  = torch.zeros(1, 1, cfg.hidden_size, dtype=torch.float16)
    pid = torch.zeros(1, dtype=torch.int32)
    mk = torch.zeros(1, 1, 1, CTX, dtype=torch.float16)
    cp = torch.zeros(1, dtype=torch.int32)
    lc = torch.zeros(cs, dtype=torch.float16)
    lr = torch.zeros(rs, dtype=torch.float16)
    w.k_cache.zero_(); w.v_cache.zero_()
    tr = torch.jit.trace(w, (h, pid, mk, cp, lc, lr), check_trace=False)
    w.k_cache.zero_(); w.v_cache.zero_()
    for _, buf in tr.named_buffers():
        buf.zero_()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml = ct.convert(tr,
            inputs=[
                ct.TensorType(name="hidden_states",          shape=h.shape,   dtype=np.float16),
                ct.TensorType(name="position_ids",           shape=pid.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask",            shape=mk.shape,  dtype=np.float16),
                ct.TensorType(name="current_pos",            shape=cp.shape,  dtype=np.int32),
                ct.TensorType(name="linear_conv_state",      shape=lc.shape,  dtype=np.float16),
                ct.TensorType(name="linear_recurrent_state", shape=lr.shape,  dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="output_hidden_states",       dtype=np.float16),
                ct.TensorType(name="linear_conv_state_out",      dtype=np.float16),
                ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
            ],
            states=w.states,
            compute_precision=PRECISION,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram")
    del w, tr; gc.collect()
    tmpdir = tempfile.mkdtemp(prefix="diag_chunk_", dir=TEMP_CACHE)
    tmppath = os.path.join(tmpdir, "model.mlpackage")
    ml.save(tmppath)
    del ml; gc.collect()
    ml2 = ct.models.MLModel(tmppath, compute_units=ct.ComputeUnit.CPU_ONLY)
    feed = {
        "hidden_states":          hidden_np.astype(np.float16),
        "position_ids":           pos_np.copy(),
        "causal_mask":            mask_np.copy(),
        "current_pos":            pos_np.copy(),
        "linear_conv_state":      np.zeros(cs, dtype=np.float16),
        "linear_recurrent_state": np.zeros(rs, dtype=np.float16),
    }
    state = ml2.make_state()
    out = ml2.predict(feed, state=state)
    result = out["output_hidden_states"].copy()
    del ml2, feed, state, out; gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)
    cleanup_temp()
    return result


def cascade_partition(model, cfg, tcfg, boundaries, embed_np, mask_np, pos_np, pt_ckpts):
    N = len(pt_ckpts) - 1
    edges = [-1] + sorted(boundaries) + [N - 1]
    chunks = [(edges[i] + 1, edges[i + 1] + 1) for i in range(len(edges) - 1)]
    h_np = embed_np.copy()
    per_chunk = []
    for (s, e) in chunks:
        h_np = convert_and_run(model, cfg, tcfg, s, e, h_np, mask_np, pos_np)
        c = cos_sim(pt_ckpts[e], h_np)
        per_chunk.append((s, e, c))
    final_cos = cos_sim(pt_ckpts[N], h_np)
    return final_cos, per_chunk


def main():
    warnings.filterwarnings("ignore")
    cleanup_temp()

    print("=" * 72)
    print("  [FLLL] PARTITION TEST — Boundaries BEFORE each Full-attention layer")
    print("=" * 72)

    # Load model
    print("\n[1] Loading model...")
    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX; cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval().half()
    for p in model.parameters():
        p.requires_grad = False
    tcfg = cfg.text_config
    N = cfg.num_hidden_layers
    lt = tcfg.layer_types
    print(f"  {N} layers")

    # PyTorch ground truth
    print("\n[2] Computing PyTorch ground truth...")
    with torch.no_grad():
        embed = model.model.embed_tokens(torch.tensor([[9906]], dtype=torch.long)).half()
    pos_np  = np.array([0], dtype=np.int32)
    mask_np = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask_np[:, :, :, :1] = 0
    pid_t  = torch.zeros(1, dtype=torch.int32)
    mask_t = torch.from_numpy(mask_np)
    cp_t   = torch.zeros(1, dtype=torch.int32)

    pt_ckpts = [embed.numpy().copy()]
    h = embed.clone()
    with torch.no_grad():
        for li in range(N):
            cs, rs = get_state_shapes(tcfg, 1)
            lc = torch.zeros(cs, dtype=torch.float16)
            lr = torch.zeros(rs, dtype=torch.float16)
            kc = torch.zeros(1, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16)
            vc = torch.zeros(1, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=torch.float16)
            h = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=h.clone(), position_ids=pid_t.clone(),
                causal_mask=mask_t.clone(), current_pos=cp_t.clone(),
                kv_cache_0=None, k_cache=kc, v_cache=vc,
                linear_conv_state=lc, linear_recurrent_state=lr,
                start_layer=li, end_layer=li + 1, apply_final_norm=False)
            pt_ckpts.append(h.numpy().copy())
    print(f"  {len(pt_ckpts)} checkpoints")

    # ── Test partitions ────────────────────────────────────────────
    partitions = [
        # KEY TEST: boundaries BEFORE each F layer
        ("[FLLL] 8-chunk: [0-2|3-6|7-10|11-14|15-18|19-22|23-26|27-31]",
         [2, 6, 10, 14, 18, 22, 26]),

        # Isolate F31 too
        ("[FLLL] 9-chunk: [0-2|3-6|7-10|11-14|15-18|19-22|23-26|27-30|31]",
         [2, 6, 10, 14, 18, 22, 26, 30]),

        # Isolate F19 as single layer
        ("[FLLL] 9-chunk+F19iso: [0-2|3-6|7-10|11-14|15-18|19|20-22|23-26|27-31]",
         [2, 6, 10, 14, 18, 19, 22, 26]),

        # Isolate both F19 and F31
        ("[FLLL] 10-chunk: [0-2|3-6|7-10|11-14|15-18|19|20-22|23-26|27-30|31]",
         [2, 6, 10, 14, 18, 19, 22, 26, 30]),

        # For reference: all-individual cascade
        # (skipped — already known: cos=0.994)

        # Compare: [LLLF] 8-chunk (known bad)
        ("[LLLF] 8-chunk for comparison: [0-3|4-7|8-11|12-15|16-19|20-23|24-27|28-31]",
         [3, 7, 11, 15, 19, 23, 27]),
    ]

    for name, bounds in partitions:
        t0 = time.time()
        try:
            final_cos, per_chunk = cascade_partition(
                model, cfg, tcfg, bounds, pt_ckpts[0], mask_np, pos_np, pt_ckpts)
        except Exception as exc:
            print(f"\n  {name}:  FAILED — {exc}")
            continue
        dt = time.time() - t0
        n_ch = len(bounds) + 1
        print(f"\n  {name}:")
        for (s, e, c) in per_chunk:
            types = ''.join(layer_tag(lt, i) for i in range(s, e))
            print(f"    [{s:2d}-{e-1:2d}] ({e-s:2d} layers) {types}  cos={c:.10f}")
        print(f"    FINAL: cos={final_cos:.10f}  ({n_ch} chunks, {dt:.0f}s)")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
