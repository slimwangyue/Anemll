#!/usr/bin/env python3
"""Runner: sweep LUT4 lm_head group_size using subprocess-per-config.

Runs each config (LUT6 baseline + LUT4 gs=1,2,4,8) as a SEPARATE Python
process to avoid memory pressure / segfaults from loading all models at once.

Usage:
    python tests/dev/test_lmhead_lut4_sweep_runner.py
    python tests/dev/test_lmhead_lut4_sweep_runner.py --tokens 40 --group-sizes 1 2
"""
import sys, os, json, subprocess, argparse

STABLE_DIR = "/Users/yw68/Anemll/qwen3_5_stable_models"
EXPORT_DIR = "/tmp/lmhead_groupsize_sweep"
RESULTS_DIR = os.path.join(EXPORT_DIR, "results")
SCRIPT = os.path.join(os.path.dirname(__file__), "test_lmhead_lut4_single.py")
PYTHON = sys.executable


def dir_size_mb(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def find_model(base_dir, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def run_config(label, lmhead_path, tokens):
    """Run a single config in a subprocess. Returns result dict or None."""
    out_file = os.path.join(RESULTS_DIR, f"{label}.json")
    cmd = [
        PYTHON, SCRIPT,
        "--lmhead-path", lmhead_path,
        "--label", label,
        "--tokens", str(tokens),
        "--output", out_file,
    ]
    print(f"\n{'='*70}")
    print(f"  Running: {label}")
    print(f"  LM Head: {lmhead_path}")
    print(f"{'='*70}", flush=True)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  ❌ {label} CRASHED (exit code {result.returncode})")
        return None

    if os.path.exists(out_file):
        with open(out_file) as f:
            return json.load(f)
    print(f"  ❌ {label} no output file")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build config list: baseline + LUT4 variants
    configs = []

    # Baseline: LUT6 gs=8
    prod_lmhead = find_model(STABLE_DIR, "lm_head_logits")
    if prod_lmhead is None:
        prod_lmhead = find_model(STABLE_DIR, "lm_head")
    if prod_lmhead:
        configs.append(("LUT6_gs8", prod_lmhead))

    # LUT4 variants
    for gs in args.group_sizes:
        label = f"LUT4_gs{gs}"
        path = os.path.join(EXPORT_DIR, f"lm_head_LUT4_gs{gs}.mlpackage")
        if os.path.exists(path):
            configs.append((label, path))
        else:
            print(f"  ⚠️  {label} not found at {path}")

    # Size comparison
    print(f"\n{'='*70}")
    print("  SIZE COMPARISON")
    print(f"{'='*70}")
    prod_sz = dir_size_mb(prod_lmhead) if prod_lmhead else 0
    print(f"\n  {'Config':<25} {'Size (MB)':>10} {'vs LUT6':>10}")
    print(f"  {'-'*47}")
    for label, path in configs:
        sz = dir_size_mb(path)
        saving = prod_sz - sz
        print(f"  {label:<25} {sz:>9.1f}M {saving:>+9.1f}M")

    # Run each config as a subprocess
    all_results = {}
    for label, path in configs:
        data = run_config(label, path, args.tokens)
        if data:
            all_results[label] = data

    # Compare results
    if "LUT6_gs8" not in all_results:
        print("\n  ❌ Baseline (LUT6_gs8) failed — cannot compare.")
        return

    ref = all_results["LUT6_gs8"]

    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")

    num_turns = len(ref['results'])
    for ti in range(num_turns):
        ref_toks = ref['results'][ti]['tokens']
        print(f"\n  Turn {ti+1} (prompt={ref['results'][ti]['prompt_len']} tok):")
        print(f"    {'Config':<25} {'Match':>16} {'1st Diff':>10}")
        print(f"    {'-'*53}")

        for label, data in all_results.items():
            toks = data['results'][ti]['tokens']
            if label == "LUT6_gs8":
                print(f"    {label:<25} {'baseline':>16} {'---':>10}")
            else:
                matches = sum(1 for a, b in zip(ref_toks, toks) if a == b)
                total = min(len(ref_toks), len(toks))
                pct = 100 * matches / total if total > 0 else 0
                first_diff = "none"
                for pos, (a, b) in enumerate(zip(ref_toks, toks)):
                    if a != b:
                        first_diff = f"pos {pos}"
                        break
                print(f"    {label:<25} {matches:>3}/{total} ({pct:>5.1f}%) {first_diff:>10}")

    # Verdict
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    print(f"\n  {'Config':<25} {'Accuracy':>10} {'Size':>8} {'Saving':>8}  Status")
    print(f"  {'-'*65}")

    for label, data in all_results.items():
        path = None
        for l, p in configs:
            if l == label:
                path = p
                break
        sz = dir_size_mb(path) if path else 0
        saving = prod_sz - sz

        if label == "LUT6_gs8":
            print(f"  {label:<25} {'baseline':>10} {sz:>7.1f}M {'---':>8}  PRODUCTION")
            continue

        total_match = 0
        total_tok = 0
        for ti in range(num_turns):
            ref_toks = ref['results'][ti]['tokens']
            toks = data['results'][ti]['tokens']
            matches = sum(1 for a, b in zip(ref_toks, toks) if a == b)
            total = min(len(ref_toks), len(toks))
            total_match += matches
            total_tok += total

        pct = 100 * total_match / total_tok if total_tok > 0 else 0
        status = "✅ PASS" if pct >= 99.0 else ("⚠️  MARGINAL" if pct >= 90.0 else "❌ FAIL")
        print(f"  {label:<25} {pct:>9.1f}% {sz:>7.1f}M {saving:>+7.1f}M  {status}")

    # Generated text comparison
    print(f"\n{'='*70}")
    print("  GENERATED TEXT (turn 1)")
    print(f"{'='*70}")
    for label, data in all_results.items():
        text = data['results'][0]['text'][:200].replace('\n', ' ')
        print(f"  [{label}] {text}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
