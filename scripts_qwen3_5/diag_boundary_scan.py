#!/usr/bin/env python3
"""Inter-layer BOUNDARY sensitivity scan for chunk partitioning.

Constraint: All 32 layers remain at FP16 compute precision (simulating LUT6).
We study ONLY where to place chunk boundaries to minimize activation error.

When layers are compiled together in one CoreML model, the MIL pipeline applies
FP16 truncation to intermediate activations, and this error compounds across
layers within the same model. By splitting at a boundary, we break the graph
so that inter-layer compounding is interrupted.

Phase 0: References
  - PyTorch FP16 ground truth (checkpoint after every layer)
  - All-individual cascade: 32 single-layer models → upper bound on quality

Phase 1: Isolated group error scan (sliding window)
  For group sizes G=2,4,6,8 and each starting position k:
  Convert [k..k+G-1] as one FP16 CoreML model, compare to PyTorch.
  → Heat map showing where FP16 error compounds most.

Phase 2: Full cascade partition tests
  Test practical chunk partitions end-to-end (current, equal-N, optimized).

Phase 3: Chunking recommendation
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
PRECISION = ct.precision.FLOAT16          # deployment precision
RESULTS_FILE = "/tmp/boundary_scan_results.txt"


# ── Utilities ─────────────────────────────────────────────────────

def cos_sim(a, b):
    a64 = np.asarray(a).flatten().astype(np.float64)
    b64 = np.asarray(b).flatten().astype(np.float64)
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))

def max_diff(a, b):
    return float(np.abs(np.asarray(a).astype(np.float64) - np.asarray(b).astype(np.float64)).max())

def mean_diff(a, b):
    return float(np.abs(np.asarray(a).astype(np.float64) - np.asarray(b).astype(np.float64)).mean())

def cleanup_temp():
    for pat in ["*.mlmodelc", "tmp*.mlpackage", "diag_chunk_*"]:
        for p in glob.glob(os.path.join(TEMP_CACHE, pat)):
            shutil.rmtree(p, ignore_errors=True)
    # Also clean CoreML BNNS compilation cache (can grow to 100+ GB)
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

def log(msg):
    """Print and append to results file."""
    print(msg)
    with open(RESULTS_FILE, "a") as f:
        f.write(msg + "\n")


# ── ChunkWrapper ──────────────────────────────────────────────────

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


# ── CoreML convert → save → reload → predict → cleanup ───────────

def convert_and_run(model, cfg, tcfg, start, end, hidden_np, mask_np, pos_np):
    """Convert layers [start..end) as one FP16 CoreML model, predict, return output."""
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


# ── Cascade helper: run a partition through CoreML ────────────────

def cascade_partition(model, cfg, tcfg, boundaries, embed_np, mask_np, pos_np, pt_ckpts):
    """Run a full cascade with given chunk boundaries.

    boundaries: sorted list of split-after positions, e.g. [5, 11, 16, 21, 26]
                means chunks [0..5], [6..11], [12..16], [17..21], [22..26], [27..31]

    Returns (final_cos, per_chunk_cos_list).
    """
    N = len(pt_ckpts) - 1  # number of layers
    # Build chunk ranges from boundaries
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


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    # Clear results file
    with open(RESULTS_FILE, "w") as f:
        f.write("")

    log("=" * 72)
    log("  INTER-LAYER BOUNDARY SENSITIVITY SCAN")
    log("  All layers FP16 compute precision (simulating LUT6 deployment)")
    log("  Studying: where to place chunk boundaries")
    log("=" * 72)
    cleanup_temp()

    # ── Load model ────────────────────────────────────────────────
    log("\n[1] Loading PyTorch model...")
    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX; cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval().half()
    for p in model.parameters():
        p.requires_grad = False
    tcfg = cfg.text_config
    N = cfg.num_hidden_layers  # 32
    layer_types = tcfg.layer_types

    log(f"  {N} layers, pattern: {''.join(layer_tag(layer_types, i) for i in range(N))}")
    log(f"  Full-attention layers at: {[i for i in range(N) if layer_types[i] != 'linear_attention']}")

    # ── PyTorch ground truth ──────────────────────────────────────
    log("\n[2] Computing PyTorch ground truth (checkpoint after every layer)...")
    with torch.no_grad():
        embed = model.model.embed_tokens(torch.tensor([[9906]], dtype=torch.long)).half()
    pos_np  = np.array([0], dtype=np.int32)
    mask_np = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask_np[:, :, :, :1] = 0
    pid_t  = torch.zeros(1, dtype=torch.int32)
    mask_t = torch.from_numpy(mask_np)
    cp_t   = torch.zeros(1, dtype=torch.int32)

    pt_ckpts = [embed.numpy().copy()]   # pt_ckpts[0] = embed, [k] = after layer k-1
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
    log(f"  {len(pt_ckpts)} checkpoints (embed + {N} layers)")
    log(f"  Final hidden norm: {np.linalg.norm(pt_ckpts[-1]):.4f}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 0: Reference baselines
    # ══════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  PHASE 0: REFERENCE BASELINES")
    log("=" * 72)

    # ── All-individual: 32 single-layer models cascaded ───────────
    log("\n  [0a] All-individual cascade (32 single-layer models)...")
    t0_all = time.time()
    h_ind = pt_ckpts[0].copy()
    ind_cos = []
    for li in range(N):
        h_ind = convert_and_run(model, cfg, tcfg, li, li + 1, h_ind, mask_np, pos_np)
        c = cos_sim(pt_ckpts[li + 1], h_ind)
        ind_cos.append(c)
        if li % 8 == 7 or li == N - 1:
            log(f"    after layer {li:2d} ({layer_tag(layer_types, li)}): cos={c:.10f}")
    ref_individual = cos_sim(pt_ckpts[N], h_ind)
    log(f"  All-individual FINAL:  cos={ref_individual:.10f}  ({time.time()-t0_all:.0f}s)")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 1: ISOLATED GROUP ERROR SCAN (sliding window)
    #  For each group size G and starting position k,
    #  convert [k..k+G-1] as one FP16 CoreML model.
    #  Feed PyTorch ground truth for layer k as input.
    #  Compare output to PyTorch ground truth after layer k+G-1.
    # ══════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  PHASE 1: ISOLATED GROUP ERROR SCAN (sliding window)")
    log("  Each group converted independently. Input = PyTorch ground truth.")
    log("  This reveals WHERE in the network FP16 error compounds most.")
    log("=" * 72)

    group_results = {}  # (G, k) → cos_sim

    for G in [2, 4, 6, 8]:
        log(f"\n  ─── Group size {G} ({N - G + 1} windows) ───")
        log(f"  {'Start':>5s}  {'Range':>8s}  {'Types':>{G}s}  {'cos_sim':>14s}  {'1-cos':>12s}")
        t0_g = time.time()
        for k in range(N - G + 1):
            t0 = time.time()
            out_np = convert_and_run(
                model, cfg, tcfg, k, k + G, pt_ckpts[k], mask_np, pos_np)
            c = cos_sim(pt_ckpts[k + G], out_np)
            group_results[(G, k)] = c
            dt = time.time() - t0
            types = ''.join(layer_tag(layer_types, i) for i in range(k, k + G))
            err = 1 - c
            log(f"  {k:5d}  [{k:2d}-{k+G-1:2d}]  {types:>{G}s}  {c:14.10f}  {err:12.2e}  ({dt:.0f}s)")
        dt_g = time.time() - t0_g
        # Summary for this group size
        items = [(kk, group_results[(G, kk)]) for kk in range(N - G + 1)]
        worst = min(items, key=lambda x: x[1])
        best = max(items, key=lambda x: x[1])
        log(f"  Size {G} done ({dt_g:.0f}s). Best: [{best[0]:2d}-{best[0]+G-1:2d}] cos={best[1]:.10f}  "
            f"Worst: [{worst[0]:2d}-{worst[0]+G-1:2d}] cos={worst[1]:.10f}")

    # ── Phase 1 summary ──────────────────────────────────────────
    log("\n" + "─" * 72)
    log("  PHASE 1 HEAT MAP SUMMARY")
    log("─" * 72)
    for G in [4, 6, 8]:
        items = sorted([(kk, group_results[(G, kk)]) for kk in range(N - G + 1)],
                        key=lambda x: x[1])
        log(f"\n  Group size {G} — bottom 5 (most error):")
        for rank, (kk, c) in enumerate(items[:5]):
            types = ''.join(layer_tag(layer_types, i) for i in range(kk, kk + G))
            log(f"    #{rank+1}  [{kk:2d}-{kk+G-1:2d}] {types}  cos={c:.10f}  1-cos={1-c:.2e}")
        log(f"  Group size {G} — top 3 (least error):")
        for rank, (kk, c) in enumerate(items[-3:]):
            types = ''.join(layer_tag(layer_types, i) for i in range(kk, kk + G))
            log(f"    [{kk:2d}-{kk+G-1:2d}] {types}  cos={c:.10f}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 2: FULL CASCADE PARTITION TESTS
    #  Test practical chunk partitions end-to-end.
    # ══════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  PHASE 2: FULL CASCADE PARTITION TESTS")
    log("=" * 72)

    partitions = [
        # Current partition
        ("Current 6-chunk [0-5|6-11|12-16|17-21|22-26|27-31]",
         [5, 11, 16, 21, 26]),
        # Equal-size partitions
        ("Equal-4 (8+8+8+8) bdys=7,15,23",
         [7, 15, 23]),
        ("Split-at-F (8×4) bdys=3,7,11,15,19,23,27",
         [3, 7, 11, 15, 19, 23, 27]),
        # Denser in problematic middle region
        ("Dense-middle-8 [0-3|4-7|8-11|12-15|16-19|20-21|22-27|28-31]",
         [3, 7, 11, 15, 19, 21, 27]),
        # Even denser middle
        ("Dense-middle-10 [0-3|4-7|8-11|12-13|14-15|16-17|18-19|20-23|24-27|28-31]",
         [3, 7, 11, 13, 15, 17, 19, 23, 27]),
    ]
    # Equal-N partitions
    for n_chunks in [6, 8, 10, 16]:
        base, rem = divmod(N, n_chunks)
        bdys = []
        pos = 0
        for c_idx in range(n_chunks - 1):
            pos += base + (1 if c_idx < rem else 0)
            bdys.append(pos - 1)
        partitions.append((f"Equal-{n_chunks} ({n_chunks} chunks)", bdys))

    cascade_results = []
    for name, bounds in partitions:
        t0 = time.time()
        try:
            final_cos, per_chunk = cascade_partition(
                model, cfg, tcfg, bounds, pt_ckpts[0], mask_np, pos_np, pt_ckpts)
        except Exception as exc:
            log(f"\n  {name}:  FAILED — {exc}")
            continue
        dt = time.time() - t0
        n_ch = len(bounds) + 1
        cascade_results.append((name, bounds, final_cos, per_chunk, dt))
        log(f"\n  {name}:")
        for (s, e, c) in per_chunk:
            types = ''.join(layer_tag(layer_types, i) for i in range(s, e))
            log(f"    [{s:2d}-{e-1:2d}] ({e-s:2d} layers) {types}  cos={c:.10f}")
        log(f"    FINAL: cos={final_cos:.10f}  ({n_ch} chunks, {dt:.0f}s)")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 2b: OPTIMIZED PARTITION SEARCH
    #  Use Phase 1 heat map to propose better partitions.
    # ══════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  PHASE 2b: OPTIMIZED PARTITION SEARCH")
    log("=" * 72)

    # Greedy chunk builder: place boundaries where isolated error is highest.
    # For each gap between consecutive boundaries, compute the group error.
    # Split the worst group at the midpoint. Repeat until target num chunks.
    def build_greedy_partition(target_chunks, max_group=10):
        """Greedily add boundaries at midpoints of worst-error groups."""
        bdys = set()
        for _ in range(target_chunks - 1):
            # Build current chunks
            edges = sorted([-1] + [b for b in bdys] + [N - 1])
            chunks = [(edges[i] + 1, edges[i + 1] + 1) for i in range(len(edges) - 1)]
            # Find the chunk with worst isolated error
            worst_err = -1
            worst_chunk = None
            for (s, e) in chunks:
                g = e - s
                if g <= 1:
                    continue
                # Use Phase 1 data if available, else estimate
                if (g, s) in group_results:
                    err = 1 - group_results[(g, s)]
                else:
                    # Estimate from closest available data
                    err = 1 - 0.999  # default small
                    for gg in [g, g - 1, g + 1, g - 2, g + 2]:
                        if (gg, s) in group_results:
                            err = 1 - group_results[(gg, s)]
                            break
                if err > worst_err:
                    worst_err = err
                    worst_chunk = (s, e)
            if worst_chunk is None:
                break
            s, e = worst_chunk
            mid = (s + e) // 2
            if mid > s:
                bdys.add(mid - 1)
        return sorted(bdys)

    for target in [4, 6, 8]:
        bdys = build_greedy_partition(target)
        name = f"Greedy-opt-{target}"
        t0 = time.time()
        try:
            final_cos, per_chunk = cascade_partition(
                model, cfg, tcfg, bdys, pt_ckpts[0], mask_np, pos_np, pt_ckpts)
        except Exception as exc:
            log(f"\n  {name}: FAILED — {exc}")
            continue
        dt = time.time() - t0
        n_ch = len(bdys) + 1
        cascade_results.append((name, bdys, final_cos, per_chunk, dt))
        bdys_str = ",".join(str(b) for b in bdys)
        log(f"\n  {name} (bdys=[{bdys_str}]):")
        for (s, e, c) in per_chunk:
            types = ''.join(layer_tag(layer_types, i) for i in range(s, e))
            log(f"    [{s:2d}-{e-1:2d}] ({e-s:2d} layers) {types}  cos={c:.10f}")
        log(f"    FINAL: cos={final_cos:.10f}  ({n_ch} chunks, {dt:.0f}s)")

    # ══════════════════════════════════════════════════════════════
    #  FINAL SUMMARY AND RECOMMENDATION
    # ══════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("  FINAL SUMMARY")
    log("=" * 72)

    log(f"\n  Reference: all-individual (32 chunks) = {ref_individual:.10f}")

    log(f"\n  ISOLATED GROUP ERROR (size 6, sliding window):")
    for k in range(N - 6 + 1):
        c = group_results.get((6, k), 0)
        err = 1 - c
        bar_len = min(60, int(err * 5000))
        bar = "█" * bar_len
        types = ''.join(layer_tag(layer_types, i) for i in range(k, k + 6))
        log(f"    [{k:2d}-{k+5:2d}] {types} cos={c:.10f} err={err:.2e} {bar}")

    log(f"\n  CASCADE PARTITION RANKING (sorted by final cos):")
    cascade_results.sort(key=lambda x: x[2], reverse=True)
    for rank, (name, bounds, fc, per_chunk, dt) in enumerate(cascade_results):
        n_ch = len(bounds) + 1
        bdys_str = ",".join(str(b) for b in bounds)
        log(f"    #{rank+1:2d}  cos={fc:.10f}  ({n_ch:2d} chunks)  {name}")

    # Pick best ≤8 chunks config
    best_le8 = None
    for name, bounds, fc, per_chunk, dt in cascade_results:
        if len(bounds) + 1 <= 8:
            best_le8 = (name, bounds, fc, per_chunk)
            break

    log(f"\n  RECOMMENDED PARTITION (≤8 chunks):")
    if best_le8:
        name, bounds, fc, per_chunk = best_le8
        n_ch = len(bounds) + 1
        log(f"  {name}")
        log(f"  {n_ch} chunks, boundaries after layers: {bounds}")
        log(f"  Expected quality: cos={fc:.10f}")
        for s, e, c in per_chunk:
            types = ''.join(layer_tag(layer_types, i) for i in range(s, e))
            log(f"    chunk [{s:2d}-{e-1:2d}] ({e-s:2d} layers) {types}  cos={c:.10f}")

    log(f"\n  NEXT ENGINEERING STEP:")
    log(f"  Export with recommended partition and measure end-to-end")
    log(f"  text generation quality (perplexity, benchmark scores).")
    log(f"\n  Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
