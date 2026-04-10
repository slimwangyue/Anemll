#!/usr/bin/env python3
"""Subprocess-based full-pipeline selective FP32 experiment.

Each chunk conversion + predict runs as a SEPARATE subprocess so that
APFS can reclaim disk space between chunks (CoreML memory-maps compiled
models, preventing reclamation within a process).

Usage:
    cd /Users/yw68/Anemll && source .venv/bin/activate
    PYTHONPATH=. python tests/dev/run_fullpipe_fp32.py 2>&1 \
        | tee tests/dev/fullpipe_fp32_results.txt
"""
import os, sys, json, time, subprocess, shutil, tempfile, glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "_fullpipe_results")
ANEMLL_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

ALL_VARIANTS = [
    ("V0_fp16",           "baseline FP16",              "set()"),
    ("V1_fp32",           "full FP32 [ref]",            "None"),
    ("V2_exp",            "exp → FP32",                 "{'exp'}"),
    ("V4_exp_mul_add_mm", "exp+mul+add+matmul → FP32",  "{'exp','mul','add','matmul'}"),
    ("V6_attn_input",     "attn-input → FP32",          "{'layer_norm','softmax','matmul'}"),
    ("V7_V5_plus_V6",     "recurrence+attn → FP32",
     "{'exp','reduce_sum','log','relu','abs','sub','clip','rsqrt','split','mul','layer_norm','softmax','matmul'}"),
    ("V8_core_recurrence","core recurrence → FP32",     "{'exp','reduce_sum','log','rsqrt'}"),
    ("V9_conv",           "conv (projections) → FP32",  "{'conv'}"),
    ("V10_conv_ln",       "conv+layer_norm → FP32",     "{'conv','layer_norm'}"),
]

# Inline script for subprocess: export one chunk, predict, save output
CHUNK_SCRIPT = r'''
import os, sys, gc, time, importlib.util, warnings, shutil, glob, tempfile, json
import numpy as np
import torch

args = json.loads(sys.argv[1])

_spec = importlib.util.spec_from_file_location(
    "qwen_config", os.path.join(args["anemll_root"], "scripts_qwen3_5/config.py"))
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)

import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

CTX = _cfg.CTX
BATCH_SIZE = _cfg.BATCH_SIZE
NUM_CHUNKS = _cfg.NUM_CHUNKS
CHUNK_RANGES = _cfg.CHUNK_RANGES
DEFAULT_HF_MODEL = _cfg.DEFAULT_HF_MODEL
PER_CHANNEL = _cfg.PER_CHANNEL

chunk_idx = args["chunk_idx"]
fp32_ops_str = args["fp32_ops"]
fp32_ops = eval(fp32_ops_str)
hidden_in_path = args["hidden_in"]
hidden_out_path = args["hidden_out"]
meta_out_path = args["meta_out"]

# Compute precision
if fp32_ops is None:
    cp = ct.precision.FLOAT32
elif len(fp32_ops) == 0:
    cp = ct.precision.FLOAT16
else:
    def op_selector(op):
        return op.op_type not in fp32_ops
    cp = ct.transform.FP16ComputePrecision(op_selector=op_selector)

# Linear state shapes
def linear_state_shapes(cfg, local_n):
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

# Load model
cfg = Qwen35Config.from_json(os.path.join(DEFAULT_HF_MODEL, "config.json"))
cfg.context_length = CTX; cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(DEFAULT_HF_MODEL)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Export chunk
conv = Qwen35Converter(
    model, context_length=CTX, batch_size=BATCH_SIZE,
    num_chunks=NUM_CHUNKS, lut_bits=None, per_channel=PER_CHANNEL,
    compute_precision="float16")
conv.compute_precision = cp
sl, el = CHUNK_RANGES[chunk_idx]
ml = conv.convert_part_2(model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
                         override_start_layer=sl, override_end_layer=el)
# Move temp to output path (avoids 850MB copy)
tmp_dir = os.path.join(tempfile.gettempdir(), "qwen35_chunk_tmp")
os.makedirs(tmp_dir, exist_ok=True)
pkg_path = os.path.join(tmp_dir, "chunk.mlpackage")
if os.path.exists(pkg_path):
    shutil.rmtree(pkg_path)
temp_pkg = getattr(ml, 'package_path', None)
if temp_pkg and os.path.exists(temp_pkg):
    shutil.move(temp_pkg, pkg_path)
else:
    ml.save(pkg_path)
del ml, conv, model; gc.collect()

# Load and predict
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    loaded = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

hidden = np.load(hidden_in_path)
local_n = el - sl
lc_shape, lr_shape = linear_state_shapes(cfg, local_n)

feed = {
    "hidden_states": hidden,
    "position_ids": np.array([10], dtype=np.int32),
    "causal_mask": np.concatenate([np.zeros((1,1,1,11), dtype=np.float16),
                                    np.full((1,1,1,CTX-11), -10000.0, dtype=np.float16)], axis=-1),
    "current_pos": np.array([10], dtype=np.int32),
    "linear_conv_state": np.zeros(lc_shape, dtype=np.float16),
    "linear_recurrent_state": np.zeros(lr_shape, dtype=np.float16),
}
state = loaded.make_state()
t_pred = time.time()
out = loaded.predict(feed, state=state)
lat = (time.time() - t_pred) * 1000

# Save result
out_hidden = out["output_hidden_states"]
np.save(hidden_out_path, out_hidden)

# Save metadata
with open(meta_out_path, 'w') as f:
    json.dump({"lat_ms": lat, "chunk_idx": chunk_idx}, f)

# Cleanup
loaded._MLModel__proxy = None
del loaded, state; gc.collect(); gc.collect()
shutil.rmtree(pkg_path, ignore_errors=True)
for p in glob.glob(os.path.join(tempfile.gettempdir(), "*.mlmodelc")):
    shutil.rmtree(p, ignore_errors=True)
e5rt = os.path.expanduser("~/Library/Caches/com.apple.python3/com.apple.e5rt.e5bundlecache")
if os.path.isdir(e5rt):
    shutil.rmtree(e5rt, ignore_errors=True)
os.sync()
'''


def cosine(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (d + 1e-30))


def cleanup_temp():
    tmp = tempfile.gettempdir()
    for pattern in ("*.mlmodelc", "tmp*.mlpackage", "qwen35_chunk_tmp"):
        for p in glob.glob(os.path.join(tmp, pattern)):
            shutil.rmtree(p, ignore_errors=True)
    e5rt = os.path.expanduser("~/Library/Caches/com.apple.python3/com.apple.e5rt.e5bundlecache")
    if os.path.isdir(e5rt):
        shutil.rmtree(e5rt, ignore_errors=True)


def run_variant(vname, fp32_ops_str, num_chunks=9):
    """Run all chunks for one variant via subprocesses."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initial hidden (deterministic)
    import torch
    torch.manual_seed(42)
    initial_hidden = (torch.randn(1, 1, 2560, dtype=torch.float16) * 0.1).numpy()
    hidden_path = os.path.join(RESULTS_DIR, f"{vname}_h_in.npy")
    np.save(hidden_path, initial_hidden)

    chunk_hiddens = []
    chunk_lats = []
    total_export_s = 0.0

    for ci in range(num_chunks):
        cleanup_temp()
        free_gb = shutil.disk_usage("/").free / (1024**3)
        sys.stdout.write(f"    chunk{ci} disk={free_gb:.1f}GB ... ")
        sys.stdout.flush()

        h_in = os.path.join(RESULTS_DIR, f"{vname}_h_in.npy")
        h_out = os.path.join(RESULTS_DIR, f"{vname}_h_out.npy")
        meta = os.path.join(RESULTS_DIR, f"{vname}_meta.json")

        args_json = json.dumps({
            "anemll_root": ANEMLL_ROOT,
            "chunk_idx": ci,
            "fp32_ops": fp32_ops_str,
            "hidden_in": h_in,
            "hidden_out": h_out,
            "meta_out": meta,
        })

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-c", CHUNK_SCRIPT, args_json],
            env={**os.environ, "PYTHONPATH": ANEMLL_ROOT},
            capture_output=True, text=True, timeout=300,
        )
        elapsed = time.time() - t0
        total_export_s += elapsed

        if result.returncode != 0:
            sys.stdout.write(f"FAILED ({elapsed:.0f}s)\n")
            sys.stderr.write(f"  stderr: {result.stderr[-500:]}\n")
            return None

        # Read outputs
        hidden_out = np.load(h_out)
        with open(meta) as f:
            m = json.load(f)
        lat = m["lat_ms"]

        chunk_hiddens.append(hidden_out.astype(np.float32).copy())
        chunk_lats.append(lat)

        # Set up for next chunk
        np.save(h_in, hidden_out)

        sys.stdout.write(f"lat={lat:.1f}ms wall={elapsed:.0f}s\n")
        sys.stdout.flush()

    # Save final results
    np.savez(os.path.join(RESULTS_DIR, f"{vname}.npz"),
             **{f"h{ci}": h for ci, h in enumerate(chunk_hiddens)},
             lats=np.array(chunk_lats),
             export_s=np.float64(total_export_s),
             wall=np.float64(total_export_s))

    return chunk_hiddens, chunk_lats, total_export_s


def load_result(vname):
    path = os.path.join(RESULTS_DIR, f"{vname}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    # Find number of chunks
    ci = 0
    hiddens = []
    while f"h{ci}" in d:
        hiddens.append(d[f"h{ci}"])
        ci += 1
    return dict(hiddens=hiddens, lats=d["lats"].tolist(),
                export_s=float(d["export_s"]), wall=float(d["wall"]),
                total_lat=float(d["lats"].sum()))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Select variants
    if "--variant" in sys.argv:
        idx = sys.argv.index("--variant")
        vname = sys.argv[idx + 1]
        vdict = {v[0]: v for v in ALL_VARIANTS}
        if vname not in vdict:
            print(f"Unknown: {vname}")
            return
        _, vdesc, fp32_ops_str = vdict[vname]
        variants = [(vname, vdesc, fp32_ops_str)]
    elif "--compare" in sys.argv:
        variants = []
    elif "--all" in sys.argv:
        variants = ALL_VARIANTS
    else:
        variants = ALL_VARIANTS[:2]  # V0 + V1 by default

    # Run variants
    for vname, vdesc, fp32_ops_str in variants:
        print(f"\n{'='*80}")
        print(f"  {vname}: {vdesc}")
        print(f"{'='*80}")
        result = run_variant(vname, fp32_ops_str)
        if result is None:
            print(f"  FAILED")
        else:
            _, lats, export_s = result
            print(f"  total: lat={sum(lats):.1f}ms  wall={export_s:.0f}s")

    # Compare
    if not variants and "--compare" not in sys.argv:
        return

    print("\n" + "=" * 120)
    print("  COMPARISON: PER-CHUNK COSINE vs FP32 REFERENCE")
    print("=" * 120)

    ref = load_result("V1_fp32")
    if ref is None:
        print("  V1_fp32 not found. Run it first.")
        return

    num_chunks = len(ref["hiddens"])
    chunk_hdr = "".join(f"  c{ci}" for ci in range(num_chunks))
    print(f"  {'Variant':<22s}{chunk_hdr}  {'lat':>8s}")
    print("-" * 120)

    for vname, vdesc, _ in ALL_VARIANTS:
        r = load_result(vname)
        if r is None:
            continue
        cos_vals = [cosine(r["hiddens"][ci], ref["hiddens"][ci])
                    for ci in range(min(num_chunks, len(r["hiddens"])))]
        cos_str = "".join(f" {c:.4f}" for c in cos_vals)
        print(f"  {vname:<22s}{cos_str}  {r['total_lat']:>7.1f}ms")

    # Final summary
    print(f"\n  {'Variant':<22s} {'FinalCos':>10s} {'L2':>10s} {'MaxDiff':>10s}")
    print("-" * 60)
    for vname, vdesc, _ in ALL_VARIANTS:
        r = load_result(vname)
        if r is None:
            continue
        fh, rh = r["hiddens"][-1], ref["hiddens"][-1]
        c = cosine(fh, rh)
        l2 = float(np.linalg.norm(fh.flatten() - rh.flatten()))
        md = float(np.max(np.abs(fh.flatten() - rh.flatten())))
        print(f"  {vname:<22s} {c:>10.6f} {l2:>10.4f} {md:>10.4f}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
