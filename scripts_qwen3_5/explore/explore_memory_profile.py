#!/usr/bin/env python3
"""Qwen3.5-4B — Memory profiling for stable + explore configs.

Measures process RSS (resident set size) at each stage of model loading
and after inference, plus on-disk model sizes. Each config runs in an
isolated subprocess to get clean measurements on 16GB Mac.

Usage:
    QWEN35_HF_MODEL=/path/to/Qwen3.5-4B python scripts_qwen3_5/explore/explore_memory_profile.py
    python scripts_qwen3_5/explore/explore_memory_profile.py --config stable
    python scripts_qwen3_5/explore/explore_memory_profile.py --config batch256_ctx2048
    python scripts_qwen3_5/explore/explore_memory_profile.py --config all
"""
import sys, os, json, argparse, subprocess, time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts_qwen3_5.explore.explore_config import CONFIGS, get_config, STABLE


# ── Subprocess child: measure memory for one config ──────────────────

CHILD_SCRIPT = r'''
import sys, os, gc, time, json
import numpy as np

def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def main():
    cfg = json.loads(sys.argv[1])
    model_dir = cfg["output_dir"]
    ctx = cfg["CTX"]
    batch_size = cfg["BATCH_SIZE"]
    num_chunks = cfg["NUM_CHUNKS"]
    name = cfg["name"]

    # ── Architecture constants (Qwen3.5-4B text config) ──
    NUM_LAYERS = 32
    HIDDEN_SIZE = 2560
    NUM_KV_HEADS = 4          # full attention
    HEAD_DIM = 256             # full attention
    FULL_ATTN_INTERVAL = 4    # layers 0,4,8,12,16,20,24,28 = 8 full attn layers
    LINEAR_CONV_KERNEL = 4
    LINEAR_KEY_HEADS = 16
    LINEAR_KEY_DIM = 128
    LINEAR_VALUE_HEADS = 32
    LINEAR_VALUE_DIM = 128
    VOCAB_SIZE = 248320

    num_full_attn = NUM_LAYERS // FULL_ATTN_INTERVAL  # 8
    num_linear_attn = NUM_LAYERS - num_full_attn       # 24
    layers_per_chunk = NUM_LAYERS // num_chunks        # 8

    report = {"config": name, "model_dir": model_dir, "ctx": ctx,
              "batch_size": batch_size, "num_chunks": num_chunks}

    # ── On-disk sizes ──
    disk = {}
    def _find(base, n):
        for ext in (".mlmodelc", ".mlpackage"):
            p = os.path.join(base, n + ext)
            if os.path.exists(p):
                return p
        return None

    embed_path = _find(model_dir, "embeddings")
    if embed_path:
        disk["embeddings"] = round(dir_size_mb(embed_path), 1)

    lm_path = _find(model_dir, "lm_head_logits") or _find(model_dir, "lm_head")
    if lm_path:
        disk["lm_head"] = round(dir_size_mb(lm_path), 1)

    combined_dir = os.path.join(model_dir, "combined_LUT4_dedup")
    use_combined = os.path.isdir(combined_dir)
    has_separate = os.path.exists(os.path.join(model_dir, "ffn_LUT4_chunk0.mlpackage"))
    for ci in range(num_chunks):
        if use_combined:
            p = _find(combined_dir, f"chunk{ci}")
        elif has_separate:
            p = _find(model_dir, f"ffn_LUT4_chunk{ci}")
        else:
            p = None
        if p:
            disk[f"chunk{ci}"] = round(dir_size_mb(p), 1)
    disk["total"] = round(sum(disk.values()), 1)
    report["disk_mb"] = disk

    # ── KV cache memory (full attention layers) ──
    # Each full attention layer: K + V = 2 * num_kv_heads * head_dim * ctx * fp16
    bytes_per_full_kv = 2 * NUM_KV_HEADS * HEAD_DIM * ctx * 2  # fp16 = 2 bytes
    full_attn_total_bytes = bytes_per_full_kv * num_full_attn
    full_attn_mb = full_attn_total_bytes / (1024 * 1024)

    # ── Linear attention state memory ──
    # Conv state: per layer = kernel_dim * hidden_size (but model uses [8, ctx, 32] per chunk)
    # From stable model: linear_conv_state shape = [8, 1024, 32] per chunk
    # This is [num_linear_layers_in_chunk, ctx(?), conv_kernel_dim*something]
    # Actually from the spec: conv = [8, 1024, 32], rec = [8, 32, 128, 128]
    # These shapes come from the model — the 1024 in conv_state is NOT ctx-dependent
    # Let me read them from the spec if possible
    
    # Use known shapes from prior runs for estimation
    # conv_state shape: [linear_layers_per_chunk, state_dim, conv_kernel_related]
    # For Qwen3.5-4B: [8, 1024, 32] per chunk — does NOT scale with CTX
    # rec_state: [8, 32, 128, 128] per chunk — does NOT scale with CTX
    conv_per_chunk_mb = 2 * 8 * 1024 * 32 / (1024*1024)      # 0.5 MB
    rec_per_chunk_mb = 2 * 8 * 32 * 128 * 128 / (1024*1024)   # 8.0 MB
    linear_state_per_chunk = conv_per_chunk_mb + rec_per_chunk_mb  # 8.5 MB
    linear_state_total = linear_state_per_chunk * num_chunks

    # ── Causal mask buffer ──
    mask_mb = 2 * 1 * 1 * 1 * ctx / (1024 * 1024)  # [1,1,1,CTX] fp16

    # ── Prefill mask buffer ──
    prefill_mask_mb = 2 * 1 * 1 * batch_size * ctx / (1024 * 1024)

    # ── Weight memory (approx = disk size for LUT4 quantized) ──
    weight_mb = disk["total"]

    # ── Activation memory (transient, per inference step) ──
    # hidden_states: [1, 1, 1, hidden_size] fp16
    activation_mb = 2 * HIDDEN_SIZE / (1024 * 1024)  # tiny

    # ── Summary ──
    kv_state = {
        "full_attention_layers": num_full_attn,
        "linear_attention_layers": num_linear_attn,
        "layers_per_chunk": layers_per_chunk,
        "full_attn_kv_per_layer_mb": round(bytes_per_full_kv / (1024*1024), 2),
        "full_attn_kv_total_mb": round(full_attn_mb, 2),
        "linear_conv_per_chunk_mb": round(conv_per_chunk_mb, 2),
        "linear_rec_per_chunk_mb": round(rec_per_chunk_mb, 2),
        "linear_state_per_chunk_mb": round(linear_state_per_chunk, 2),
        "linear_state_total_mb": round(linear_state_total, 2),
    }
    report["kv_state"] = kv_state

    total_state_mb = full_attn_mb + linear_state_total
    total_runtime = weight_mb + total_state_mb + mask_mb
    
    report["memory_breakdown_mb"] = {
        "weights_on_disk": round(weight_mb, 1),
        "full_attn_kv_cache": round(full_attn_mb, 1),
        "linear_state_buffers": round(linear_state_total, 1),
        "causal_mask_decode": round(mask_mb, 4),
        "causal_mask_prefill": round(prefill_mask_mb, 4),
        "total_state": round(total_state_mb, 1),
        "estimated_runtime": round(total_runtime, 1),
    }

    report["summary"] = {
        "disk_total_mb": disk["total"],
        "full_attn_kv_mb": round(full_attn_mb, 1),
        "linear_state_mb": round(linear_state_total, 1),
        "total_state_mb": round(total_state_mb, 1),
        "mask_decode_mb": round(mask_mb, 4),
        "mask_prefill_mb": round(prefill_mask_mb, 4),
        "estimated_runtime_mb": round(total_runtime, 1),
    }

    print(json.dumps(report))

if __name__ == "__main__":
    main()
'''


def profile_config(cfg, python_bin):
    """Run memory profiling for one config in a subprocess."""
    cfg_json = json.dumps(cfg)
    env = os.environ.copy()
    result = subprocess.run(
        [python_bin, "-c", CHILD_SCRIPT, cfg_json],
        capture_output=True, text=True, env=env,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  ERROR: subprocess failed (rc={result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    # Parse JSON from last line of stdout
    lines = result.stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    print(f"  ERROR: no JSON in output")
    print(f"  stdout: {result.stdout[-500:]}")
    return None


def print_report(report):
    """Pretty-print a single config's memory report."""
    name = report["config"]
    print(f"\n  Config: {name}")
    print(f"  Model dir: {report['model_dir']}")
    print(f"  Context length: {report['ctx']}, Batch size: {report['batch_size']}")

    print(f"\n  On-disk sizes:")
    disk = report["disk_mb"]
    for k, v in disk.items():
        if k != "total":
            print(f"    {k:<20} {v:>8.1f} MB")
    print(f"    {'─'*30}")
    print(f"    {'TOTAL':<20} {disk['total']:>8.1f} MB")

    kv = report["kv_state"]
    print(f"\n  KV-cache / state memory:")
    print(f"    Full attention: {kv['full_attention_layers']} layers × "
          f"{kv['full_attn_kv_per_layer_mb']:.2f} MB = {kv['full_attn_kv_total_mb']:.1f} MB "
          f"(scales with CTX={report['ctx']})")
    print(f"    Linear conv:    {kv['linear_conv_per_chunk_mb']:.2f} MB/chunk × {report['num_chunks']} = "
          f"{kv['linear_conv_per_chunk_mb'] * report['num_chunks']:.1f} MB (fixed)")
    print(f"    Linear recur:   {kv['linear_rec_per_chunk_mb']:.2f} MB/chunk × {report['num_chunks']} = "
          f"{kv['linear_rec_per_chunk_mb'] * report['num_chunks']:.1f} MB (fixed)")
    print(f"    Total state:    {kv['linear_state_total_mb'] + kv['full_attn_kv_total_mb']:.1f} MB")

    mb = report["memory_breakdown_mb"]
    print(f"\n  ── Memory Breakdown ──")
    print(f"    Weights (disk):      {mb['weights_on_disk']:>8.1f} MB")
    print(f"    Full attn KV cache:  {mb['full_attn_kv_cache']:>8.1f} MB")
    print(f"    Linear state bufs:   {mb['linear_state_buffers']:>8.1f} MB")
    print(f"    Causal mask (dec):   {mb['causal_mask_decode']:>8.4f} MB")
    print(f"    Causal mask (pf):    {mb['causal_mask_prefill']:>8.4f} MB")
    print(f"    ─────────────────────────────")
    print(f"    Total state:         {mb['total_state']:>8.1f} MB")
    print(f"    Est. runtime:        {mb['estimated_runtime']:>8.1f} MB")


def print_comparison(reports):
    """Print side-by-side comparison table."""
    print(f"\n{'='*85}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*85}")

    headers = [r["config"] for r in reports]
    col_w = max(16, max(len(h) for h in headers) + 2)

    def row(label, vals, fmt=".1f"):
        cells = "".join(f"{v:>{col_w}{fmt}}" for v in vals)
        print(f"  {label:<28}{cells}")

    # Header
    header_line = "".join(f"{h:>{col_w}}" for h in headers)
    print(f"  {'Metric':<28}{header_line}")
    print(f"  {'─'*(28 + col_w * len(headers))}")

    row("Disk total (MB)",
        [r["summary"]["disk_total_mb"] for r in reports])
    row("Full attn KV (MB)",
        [r["summary"]["full_attn_kv_mb"] for r in reports])
    row("Linear state (MB)",
        [r["summary"]["linear_state_mb"] for r in reports])
    row("Total state (MB)",
        [r["summary"]["total_state_mb"] for r in reports])
    row("Mask decode (MB)",
        [r["summary"]["mask_decode_mb"] for r in reports], ".4f")
    row("Mask prefill (MB)",
        [r["summary"]["mask_prefill_mb"] for r in reports], ".4f")
    row("Est. runtime (MB)",
        [r["summary"]["estimated_runtime_mb"] for r in reports])

    # Context length
    row("Context length",
        [r["ctx"] for r in reports], ".0f")


def main():
    parser = argparse.ArgumentParser(
        description="Memory profiling for Qwen3.5-4B configs")
    parser.add_argument("--config", type=str, default="all",
                        help="Config name or 'all' (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to file")
    args = parser.parse_args()

    python_bin = sys.executable

    # Build config list
    if args.config == "all":
        config_names = ["stable"] + [k for k in CONFIGS.keys()
                                     if os.path.isdir(CONFIGS[k]["output_dir"])]
    else:
        config_names = [args.config]

    print("=" * 85)
    print("  MEMORY PROFILING — Qwen3.5-4B")
    print(f"  Configs: {', '.join(config_names)}")
    print(f"  Python: {python_bin}")
    print("=" * 85)

    reports = []
    for cname in config_names:
        cfg = get_config(cname)
        if not os.path.isdir(cfg["output_dir"]):
            print(f"\n  SKIP {cname}: {cfg['output_dir']} not found")
            continue
        print(f"\n{'─'*85}")
        print(f"  Profiling: {cname}")
        print(f"{'─'*85}")
        t0 = time.time()
        report = profile_config(cfg, python_bin)
        elapsed = time.time() - t0
        if report:
            reports.append(report)
            print_report(report)
            print(f"\n  Profiled in {elapsed:.0f}s")
        else:
            print(f"  FAILED after {elapsed:.0f}s")

    if len(reports) > 1:
        print_comparison(reports)

    if args.output and reports:
        with open(args.output, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
