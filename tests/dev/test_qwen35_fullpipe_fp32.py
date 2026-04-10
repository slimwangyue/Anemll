#!/usr/bin/env python3
"""Qwen3.5-4B — Full-Pipeline Selective FP32 Experiment.

Chains ALL 9 decode chunks sequentially and measures accumulated cosine
divergence at each chunk boundary.  Compares against full-FP32 pipeline.

Skips V3 (exp+mul+add) and V5 (recurrence full) — known broken from
single-chunk test (cos ~0.004 due to mixed-precision cast corruption).

Usage:
    cd /Users/yw68/Anemll && source .venv/bin/activate
    PYTHONPATH=. python tests/dev/test_qwen35_fullpipe_fp32.py 2>&1 \
        | tee tests/dev/fullpipe_fp32_results.txt
"""
import os, sys, gc, time, importlib.util, warnings, shutil, glob, tempfile
import numpy as np
import torch

# ── Load config ──
_spec = importlib.util.spec_from_file_location(
    "qwen_config", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "../../scripts_qwen3_5/config.py"))
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
BATCH_SIZE, CTX, NUM_CHUNKS = _cfg.BATCH_SIZE, _cfg.CTX, _cfg.NUM_CHUNKS
CHUNK_RANGES = _cfg.CHUNK_RANGES
DEFAULT_HF_MODEL, PER_CHANNEL = _cfg.DEFAULT_HF_MODEL, _cfg.PER_CHANNEL

import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

# ── Config ──
TMPDIR = os.path.join(tempfile.gettempdir(), "qwen35_fullpipe")
N_LATENCY = 3

ALL_VARIANTS = [
    ("V0_fp16",           "baseline FP16",              set()),
    ("V1_fp32",           "full FP32 [ref]",            None),
    ("V2_exp",            "exp → FP32",                 {"exp"}),
    ("V4_exp_mul_add_mm", "exp+mul+add+matmul → FP32",  {"exp", "mul", "add", "matmul"}),
    ("V6_attn_input",     "attn-input → FP32",          {"layer_norm", "softmax", "matmul"}),
    ("V7_V5_plus_V6",     "recurrence+attn → FP32",     {"exp", "reduce_sum", "log", "relu", "abs",
                                                          "sub", "clip", "rsqrt", "split", "mul",
                                                          "layer_norm", "softmax", "matmul"}),
    ("V8_core_recurrence","core recurrence → FP32",     {"exp", "reduce_sum", "log", "rsqrt"}),
    ("V9_conv",           "conv (projections) → FP32",  {"conv"}),
    ("V10_conv_ln",       "conv+layer_norm → FP32",     {"conv", "layer_norm"}),
]
# Use --all flag to run all variants (requires ~30GB free)
# Default: run only V0+V1 to measure accumulated FP16 vs FP32 gap
VARIANTS = ALL_VARIANTS if "--all" in sys.argv else ALL_VARIANTS[:2]


# ──────────────────────── helpers ────────────────────────

def make_compute_precision(fp32_ops):
    """Return a compute_precision argument for ct.convert()."""
    if fp32_ops is None:
        return ct.precision.FLOAT32
    if len(fp32_ops) == 0:
        return ct.precision.FLOAT16
    def op_selector(op):
        return op.op_type not in fp32_ops   # True → FP16
    return ct.transform.FP16ComputePrecision(op_selector=op_selector)


def linear_state_shapes(cfg, local_n):
    """Return (conv_shape, rec_shape) for a chunk with `local_n` layers."""
    if cfg.has_linear_attention():
        tc = cfg.text_config
        conv_dim = (tc.linear_num_key_heads * tc.linear_key_head_dim * 2
                    + tc.linear_num_value_heads * tc.linear_value_head_dim)
        conv_kernel = max(1, int(tc.linear_conv_kernel_dim))
        d1, d2 = ane_conv_state_shape(conv_dim, conv_kernel)
        return ((local_n, d1, d2),
                (local_n, tc.linear_num_value_heads,
                 tc.linear_key_head_dim, tc.linear_value_head_dim))
    return ((local_n, 1, 1), (local_n, 1, 1, 1))


def _cleanup_stale_mlmodelc():
    """Remove stale .mlmodelc and tmp*.mlpackage from system temp (left by coremltools)."""
    tmp = tempfile.gettempdir()
    for pattern in ("*.mlmodelc", "tmp*.mlpackage"):
        for p in glob.glob(os.path.join(tmp, pattern)):
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass


def export_chunk(model, chunk_idx, fp32_ops, out_path):
    """Export one decode chunk. Move (not copy) temp mlpackage to out_path."""
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=PER_CHANNEL,
        compute_precision="float16")
    conv.compute_precision = make_compute_precision(fp32_ops)

    sl, el = CHUNK_RANGES[chunk_idx]
    ml = conv.convert_part_2(
        model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
        override_start_layer=sl, override_end_layer=el)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    # Move (rename) the temp mlpackage — avoids 850MB copy
    temp_pkg = getattr(ml, 'package_path', None)
    if temp_pkg and os.path.exists(temp_pkg):
        shutil.move(temp_pkg, out_path)
    else:
        ml.save(out_path)
    del ml, conv; gc.collect()


def run_pipeline(cfg, model, variant_name, fp32_ops, initial_hidden):
    """Export + chain all 9 chunks (save to disk for CoreML state), return per-chunk hidden states."""
    hidden = initial_hidden.copy()
    pos_ids = np.array([10], dtype=np.int32)
    cmask = np.zeros((1, 1, 1, CTX), dtype=np.float16)
    cmask[:, :, :, 11:] = -10000.0
    cpos = np.array([10], dtype=np.int32)

    chunk_hiddens = []          # hidden after each chunk
    chunk_lats = []             # latency per chunk (ms)
    total_export_s = 0.0

    for ci in range(NUM_CHUNKS):
        sl, el = CHUNK_RANGES[ci]
        local_n = el - sl
        lc_shape, lr_shape = linear_state_shapes(cfg, local_n)

        # ── disk space check ──
        free_gb = shutil.disk_usage("/").free / (1024**3)
        if free_gb < 1.5:
            sys.stderr.write(f"\n  !!! Only {free_gb:.1f}GB free at chunk{ci}, aborting variant.\n")
            return None, None, total_export_s

        # ── export (move temp → single reused path, no copy) ──
        pkg_path = os.path.join(TMPDIR, "chunk.mlpackage")
        t0 = time.time()
        export_chunk(model, ci, fp32_ops, pkg_path)
        export_t = time.time() - t0
        total_export_s += export_t

        # ── load ──
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ml = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

        # ── predict ──
        feed = {
            "hidden_states": hidden,
            "position_ids": pos_ids,
            "causal_mask": cmask,
            "current_pos": cpos,
            "linear_conv_state": np.zeros(lc_shape, dtype=np.float16),
            "linear_recurrent_state": np.zeros(lr_shape, dtype=np.float16),
        }
        state = ml.make_state()
        out = ml.predict(feed, state=state)
        hidden = out["output_hidden_states"]

        # ── latency (median of N_LATENCY runs) ──
        lats = []
        for _ in range(N_LATENCY):
            s = ml.make_state()
            t0 = time.time()
            ml.predict(feed, state=s)
            lats.append((time.time() - t0) * 1000)
        chunk_lats.append(float(np.median(lats)))

        chunk_hiddens.append(hidden.astype(np.float32).copy())
        # Aggressively release CoreML proxy to unmap compiled model files
        try:
            ml._MLModel__proxy = None
        except Exception:
            pass
        del ml, state
        gc.collect(); gc.collect()

        # ── cleanup to save disk ──
        shutil.rmtree(pkg_path, ignore_errors=True)
        _cleanup_stale_mlmodelc()
        # Clean e5rt compilation cache (grows ~20MB per model)
        e5rt_cache = os.path.expanduser("~/Library/Caches/com.apple.python3/com.apple.e5rt.e5bundlecache")
        if os.path.isdir(e5rt_cache):
            shutil.rmtree(e5rt_cache, ignore_errors=True)
        os.sync()

        sys.stdout.write(f"    chunk{ci} (L{sl}-{el-1}) export={export_t:.1f}s "
                         f"lat={chunk_lats[-1]:.1f}ms "
                         f"disk={shutil.disk_usage('/').free/(1024**3):.1f}GB\n")
        sys.stdout.flush()

    return chunk_hiddens, chunk_lats, total_export_s


def cosine(a, b):
    a, b = a.flatten(), b.flatten()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (d + 1e-30))


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fullpipe_results")


def _save_variant_result(vname, hiddens, lats, export_s, wall):
    """Save one variant's per-chunk hidden states + timing to npz."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, f"{vname}.npz"),
             **{f"h{ci}": h for ci, h in enumerate(hiddens)},
             lats=np.array(lats), export_s=np.float64(export_s), wall=np.float64(wall))


def _load_variant_result(vname):
    """Load previously saved variant result."""
    path = os.path.join(RESULTS_DIR, f"{vname}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    hiddens = [d[f"h{ci}"] for ci in range(NUM_CHUNKS)]
    return dict(hiddens=hiddens, lats=d["lats"].tolist(),
                export_s=float(d["export_s"]), wall=float(d["wall"]),
                total_lat=float(d["lats"].sum()))


# ──────────────────────── main modes ────────────────────────

def run_single_variant(vname):
    """Run a single variant and save results to disk."""
    vdict = {v[0]: v for v in ALL_VARIANTS}
    if vname not in vdict:
        print(f"Unknown variant: {vname}. Available: {list(vdict.keys())}")
        return
    _, vdesc, fp32_ops = vdict[vname]

    os.makedirs(TMPDIR, exist_ok=True)
    _cleanup_stale_mlmodelc()

    print(f"Running {vname}: {vdesc}")
    print(f"  Free disk: {shutil.disk_usage('/').free / (1024**3):.1f}GB")

    cfg = Qwen35Config.from_json(os.path.join(DEFAULT_HF_MODEL, "config.json"))
    cfg.context_length = CTX; cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(DEFAULT_HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    torch.manual_seed(42)
    initial_hidden = (torch.randn(1, 1, cfg.hidden_size, dtype=torch.float16) * 0.1).numpy()

    t0 = time.time()
    hiddens, lats, export_s = run_pipeline(cfg, model, vname, fp32_ops, initial_hidden)
    wall = time.time() - t0
    del model; gc.collect()

    if hiddens is None:
        print(f"  ABORTED (disk space)")
        return
    _save_variant_result(vname, hiddens, lats, export_s, wall)
    print(f"  total: export={export_s:.0f}s  lat={sum(lats):.1f}ms  wall={wall:.0f}s")
    print(f"  Saved to {RESULTS_DIR}/{vname}.npz")


def compare_results():
    """Load all saved variant results and print comparison tables."""
    print("=" * 120)
    print("  Qwen3.5-4B — Full-Pipeline Selective FP32: COMPARISON")
    print("=" * 120)

    all_results = {}
    for vname, vdesc, _ in ALL_VARIANTS:
        r = _load_variant_result(vname)
        if r is not None:
            all_results[vname] = r
            print(f"  Loaded {vname}: lat={r['total_lat']:.1f}ms, wall={r['wall']:.0f}s")

    if not all_results:
        print("\n  No results found. Run variants first.")
        return

    ref = all_results.get("V1_fp32")
    if ref is None:
        print("\n  V1_fp32 not found. Cannot compute cosines.")
        return

    # ── per-chunk cosine table ──
    print("\n" + "=" * 120)
    print("  PER-CHUNK COSINE vs FP32 REFERENCE  (accumulated through pipeline)")
    print("=" * 120)
    chunk_hdr = "".join(f"  chunk{ci:d}" for ci in range(NUM_CHUNKS))
    print(f"  {'Variant':<22s}{chunk_hdr}  {'total_lat':>9s}")
    print("-" * 120)

    for vname, vdesc, _ in ALL_VARIANTS:
        r = all_results.get(vname)
        if r is None:
            continue
        cos_vals = [cosine(r["hiddens"][ci], ref["hiddens"][ci]) for ci in range(NUM_CHUNKS)]
        cos_str = "".join(f"  {c:.5f}" for c in cos_vals)
        print(f"  {vname:<22s}{cos_str}  {r['total_lat']:>8.1f}ms")

    # ── final-chunk summary ──
    print("\n" + "=" * 100)
    print("  FINAL OUTPUT COSINE (after all 9 chunks)  vs V1_fp32")
    print("=" * 100)
    print(f"  {'Variant':<22s} {'Description':<28s} {'FinalCos':>10s} {'L2':>10s} "
          f"{'MaxDiff':>10s} {'Latency':>9s}")
    print("-" * 100)

    final_results = {}
    for vname, vdesc, _ in ALL_VARIANTS:
        r = all_results.get(vname)
        if r is None:
            continue
        fh = r["hiddens"][-1]
        rh = ref["hiddens"][-1]
        c = cosine(fh, rh)
        l2 = float(np.linalg.norm(fh.flatten() - rh.flatten()))
        md = float(np.max(np.abs(fh.flatten() - rh.flatten())))
        final_results[vname] = dict(cos=c, l2=l2, max_diff=md, lat=r["total_lat"])
        print(f"  {vname:<22s} {vdesc:<28s} {c:>10.6f} {l2:>10.4f} "
              f"{md:>10.4f} {r['total_lat']:>8.1f}ms")

    # ── analysis ──
    print("\n" + "=" * 100)
    print("  ANALYSIS")
    print("=" * 100)
    v0 = final_results.get("V0_fp16", {})
    v1 = final_results.get("V1_fp32", {})
    fp16_cos = v0.get("cos", 0)
    fp32_cos = v1.get("cos", 1.0)
    fp16_lat = v0.get("lat", 999)
    fp32_lat = v1.get("lat", 999)
    gap = fp32_cos - fp16_cos

    print(f"\n  FP16 baseline: final_cos={fp16_cos:.6f}, lat={fp16_lat:.1f}ms")
    print(f"  FP32 full:     final_cos={fp32_cos:.6f}, lat={fp32_lat:.1f}ms")
    print(f"  Quality gap:   {gap:.6f}")

    best_name, best_score = None, -1e9
    for vname, vdesc, fp32_ops in ALL_VARIANTS:
        if vname in ("V0_fp16", "V1_fp32"):
            continue
        fr = final_results.get(vname)
        if fr is None:
            continue
        recovery = (fr["cos"] - fp16_cos) / (gap + 1e-30)
        slowdown = fr["lat"] / (fp16_lat + 1e-10)
        score = recovery / slowdown if slowdown > 0 else 0
        print(f"  {vname}: {recovery:.1%} quality recovered, "
              f"{slowdown:.2f}x slowdown, score={score:.3f}")
        if score > best_score:
            best_score, best_name = score, vname

    if best_name:
        print(f"\n  >>> BEST TRADEOFF: {best_name} (score={best_score:.3f})")
    print(f"\n  Done!")


def main():
    if "--compare" in sys.argv:
        compare_results()
        return
    if "--variant" in sys.argv:
        idx = sys.argv.index("--variant")
        if idx + 1 < len(sys.argv):
            run_single_variant(sys.argv[idx + 1])
        else:
            print("Usage: --variant <NAME>")
        return

    # Default: run selected variants in-process
    os.makedirs(TMPDIR, exist_ok=True)
    _cleanup_stale_mlmodelc()

    print("=" * 100)
    print("  Qwen3.5-4B — Full-Pipeline Selective FP32 Experiment")
    print(f"  {NUM_CHUNKS} chunks, {len(VARIANTS)} variants")
    print(f"  CHUNK_RANGES = {CHUNK_RANGES}")
    print("=" * 100)

    # ── load model ──
    print("\n[1] Loading model...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(DEFAULT_HF_MODEL, "config.json"))
    cfg.context_length = CTX; cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(DEFAULT_HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"    Loaded in {time.time()-t0:.1f}s")

    # ── fixed initial hidden ──
    torch.manual_seed(42)
    initial_hidden = (torch.randn(1, 1, cfg.hidden_size, dtype=torch.float16) * 0.1).numpy()
    print(f"    initial_hidden norm = {np.linalg.norm(initial_hidden):.6f}")

    # ── run each variant ──
    all_results = {}
    for vi, (vname, vdesc, fp32_ops) in enumerate(VARIANTS):
        print(f"\n{'='*80}")
        print(f"  [{vi+1}/{len(VARIANTS)}] {vname}: {vdesc}")
        print(f"{'='*80}")
        t0 = time.time()
        hiddens, lats, export_s = run_pipeline(
            cfg, model, vname, fp32_ops, initial_hidden)
        wall = time.time() - t0
        if hiddens is None:
            print(f"    ABORTED (disk space)")
            continue
        all_results[vname] = dict(
            hiddens=hiddens, lats=lats,
            export_s=export_s,
            total_lat=sum(lats), wall=wall,
        )
        _save_variant_result(vname, hiddens, lats, export_s, wall)
        print(f"    total: export={export_s:.0f}s  lat={sum(lats):.1f}ms  wall={wall:.0f}s")

    del model; gc.collect()

    # ── compute cosines vs FP32 reference ──
    ref = all_results.get("V1_fp32")
    if ref is None:
        print("\nERROR: V1_fp32 failed, cannot compute cosines.")
        return

    print("\n" + "=" * 120)
    print("  PER-CHUNK COSINE vs FP32 REFERENCE  (accumulated through pipeline)")
    print("=" * 120)

    # Header
    chunk_hdr = "".join(f"  chunk{ci:d}" for ci in range(NUM_CHUNKS))
    print(f"  {'Variant':<22s}{chunk_hdr}  {'total_lat':>9s}")
    print("-" * 120)

    cosine_table = {}
    for vname, vdesc, _ in VARIANTS:
        r = all_results.get(vname)
        if r is None:
            print(f"  {vname:<22s}  FAILED")
            continue
        cos_vals = []
        for ci in range(NUM_CHUNKS):
            c = cosine(r["hiddens"][ci], ref["hiddens"][ci])
            cos_vals.append(c)
        cosine_table[vname] = cos_vals
        cos_str = "".join(f"  {c:.5f}" for c in cos_vals)
        print(f"  {vname:<22s}{cos_str}  {r['total_lat']:>8.1f}ms")

    # ── final-chunk summary ──
    print("\n" + "=" * 100)
    print("  FINAL OUTPUT COSINE (after all 9 chunks)  vs V1_fp32")
    print("=" * 100)
    print(f"  {'Variant':<22s} {'Description':<28s} {'FinalCos':>10s} {'L2':>10s} "
          f"{'MaxDiff':>10s} {'Latency':>9s}")
    print("-" * 100)

    final_results = {}
    for vname, vdesc, _ in VARIANTS:
        r = all_results.get(vname)
        if r is None:
            continue
        fh = r["hiddens"][-1]
        rh = ref["hiddens"][-1]
        c = cosine(fh, rh)
        l2 = float(np.linalg.norm(fh.flatten() - rh.flatten()))
        md = float(np.max(np.abs(fh.flatten() - rh.flatten())))
        final_results[vname] = dict(cos=c, l2=l2, max_diff=md, lat=r["total_lat"])
        print(f"  {vname:<22s} {vdesc:<28s} {c:>10.6f} {l2:>10.4f} "
              f"{md:>10.4f} {r['total_lat']:>8.1f}ms")

    # ── analysis ──
    print("\n" + "=" * 100)
    print("  ANALYSIS")
    print("=" * 100)
    v0 = final_results.get("V0_fp16", {})
    v1 = final_results.get("V1_fp32", {})
    fp16_cos = v0.get("cos", 0)
    fp32_cos = v1.get("cos", 1.0)
    fp16_lat = v0.get("lat", 999)
    fp32_lat = v1.get("lat", 999)
    gap = fp32_cos - fp16_cos

    print(f"\n  FP16 baseline: final_cos={fp16_cos:.6f}, lat={fp16_lat:.1f}ms")
    print(f"  FP32 full:     final_cos={fp32_cos:.6f}, lat={fp32_lat:.1f}ms")
    print(f"  Quality gap:   {gap:.6f}")
    if gap < 1e-8:
        print("  (gap too small — FP16 and FP32 are equivalent at this pipeline depth)")

    best_name, best_score = None, -1e9
    for vname, vdesc, fp32_ops in VARIANTS:
        if vname in ("V0_fp16", "V1_fp32"):
            continue
        fr = final_results.get(vname)
        if fr is None:
            continue
        recovery = (fr["cos"] - fp16_cos) / (gap + 1e-30)
        slowdown = fr["lat"] / (fp16_lat + 1e-10)
        score = recovery / slowdown if slowdown > 0 else 0
        print(f"  {vname}: {recovery:.1%} quality recovered, "
              f"{slowdown:.2f}x slowdown, score={score:.3f}")
        if score > best_score:
            best_score, best_name = score, vname

    if best_name:
        print(f"\n  >>> BEST TRADEOFF: {best_name} (score={best_score:.3f})")
    else:
        print("\n  >>> No valid selective variant.")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
