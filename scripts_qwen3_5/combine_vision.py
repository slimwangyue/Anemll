#!/usr/bin/env python3
"""Combine multi-resolution vision encoder mlpackages into one multi-function CoreML model.

Takes the individual per-resolution vision_encoder_HxW.mlpackage files produced by
export_vision.py and combines them into a single multi-function CoreML package using
coremltools save_multifunction + weight deduplication.

Since the vision encoder weights (ViT attention, MLP, layer norms) are byte-identical
across all resolutions, CoreML's built-in dedup in save_multifunction will share them
automatically, resulting in a combined package roughly the same size as one single model.

Function naming convention: f_{H}x{W}  (e.g. f_448x448, f_896x448)
Default function: f_448x448

Usage:
    python scripts_qwen3_5/combine_vision.py
    python scripts_qwen3_5/combine_vision.py --output /path/to/output_dir
    python scripts_qwen3_5/combine_vision.py --name vision_encoder_multi --skip-existing
"""
import argparse, os, sys, re, json, time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct

from config import DEFAULT_OUTPUT


def _pkg_size_mb(path):
    """Compute total size of a .mlpackage directory in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / 1e6


def combine_vision_encoders(output_dir, combined_name="vision_encoder_multi",
                             skip_existing=False, suffix=""):
    """Combine individual vision encoder mlpackages into one multi-function model.

    Args:
        output_dir: Directory containing the individual vision_encoder_HxW.mlpackage files.
        combined_name: Base name for the output (without .mlpackage extension).
        skip_existing: If True, skip if the combined model already exists.
        suffix: Filename suffix before .mlpackage (e.g. '_lut6' to match
                vision_encoder_HxW_lut6.mlpackage and vision_encoder_lut6.mlpackage).

    Returns:
        Path to the combined .mlpackage file.
    """
    output_path = os.path.join(output_dir, f"{combined_name}.mlpackage")
    if skip_existing and os.path.exists(output_path):
        print(f"[combine] Already exists: {output_path}")
        return output_path

    # ── Discover individual per-resolution packages ──
    packages = []  # list of (H, W, path)

    # Build regex based on suffix (e.g. '' or '_lut6')
    pattern = re.compile(r'vision_encoder_(\d+)x(\d+)' + re.escape(suffix) + r'\.mlpackage$')
    for fname in sorted(os.listdir(output_dir)):
        m = pattern.match(fname)
        if m:
            h, w = int(m.group(1)), int(m.group(2))
            packages.append((h, w, os.path.join(output_dir, fname)))

    # Also pick up legacy square-only vision_encoder{suffix}.mlpackage as 448×448
    default_pkg = os.path.join(output_dir, f"vision_encoder{suffix}.mlpackage")
    if os.path.exists(default_pkg):
        has_448 = any(h == 448 and w == 448 for h, w, _ in packages)
        if not has_448:
            packages.insert(0, (448, 448, default_pkg))

    if len(packages) < 2:
        raise RuntimeError(
            f"Need at least 2 vision encoder packages in {output_dir}, found {len(packages)}. "
            "Run export_vision.py --resolutions all first."
        )

    # Sort: 448×448 first (anchor for dedup), then by H, W
    packages.sort(key=lambda x: (x[0] != 448 or x[1] != 448, x[0], x[1]))

    print(f"[combine] Found {len(packages)} vision encoder packages:")
    total_input_mb = 0.0
    for h, w, p in packages:
        sz = _pkg_size_mb(p)
        total_input_mb += sz
        print(f"  f_{h}x{w}: {os.path.basename(p)}  ({sz:.1f} MB)")
    print(f"  Total input size: {total_input_mb:.1f} MB")
    print()

    # ── Build MultiFunctionDescriptor ──
    # Function names: f_448x448, f_448x672, etc.
    desc = ct.utils.MultiFunctionDescriptor()
    default_fn = None
    for h, w, path in packages:
        fn_name = f"f_{h}x{w}"
        desc.add_function(path, "main", fn_name)
        if default_fn is None:
            default_fn = fn_name  # first = anchor = default

    desc.default_function_name = default_fn
    print(f"[combine] Default function: {default_fn}")
    print(f"[combine] Saving combined model → {output_path}")

    t0 = time.time()
    ct.utils.save_multifunction(desc, output_path)
    elapsed = time.time() - t0

    sz = _pkg_size_mb(output_path)
    print(f"\n[combine] Done in {elapsed:.1f}s")
    print(f"[combine] Combined model: {sz:.1f} MB  (vs {total_input_mb:.1f} MB input, "
          f"ratio {sz/total_input_mb:.2f})")
    print(f"[combine] Saved: {output_path}")

    # Save a combined metadata file listing all resolutions
    _save_combined_meta(output_dir, combined_name, packages)

    return output_path


def _save_combined_meta(output_dir, combined_name, packages):
    """Save a JSON metadata file matching the Swift MultiResVisionMeta format.

    Swift expects per-function: {image_height, image_width, num_tokens}
    and top-level: {hidden_size, patch_size, spatial_merge_size, temporal_patch_size}.
    """
    functions = {}
    # Read shared config from the first per-resolution meta file found
    hidden_size = None
    patch_size = 16
    merge_size = 2

    for h, w, path in packages:
        fn_name = f"f_{h}x{w}"
        num_tokens = (h // (patch_size * merge_size)) * (w // (patch_size * merge_size))

        # Try to load per-resolution meta file for out_hidden_size
        for meta_fname in (f"vision_meta_{h}x{w}.json", "vision_meta.json"):
            mp = os.path.join(output_dir, meta_fname)
            if os.path.exists(mp):
                with open(mp) as f:
                    per_res = json.load(f)
                if hidden_size is None:
                    hidden_size = per_res.get("out_hidden_size", 2560)
                    patch_size = per_res.get("patch_size", 16)
                    merge_size = per_res.get("spatial_merge_size", 2)
                    # Recompute num_tokens with actual config
                    num_tokens = (h // (patch_size * merge_size)) * (w // (patch_size * merge_size))
                break

        # Swift ResolutionFunctionMeta: {image_height, image_width, num_tokens}
        functions[fn_name] = {
            "image_height": h,
            "image_width": w,
            "num_tokens": num_tokens,
        }

    combined_meta = {
        "model_name": combined_name,
        "functions": functions,
        "hidden_size": hidden_size or 2560,
        "patch_size": patch_size,
        "spatial_merge_size": merge_size,
        "temporal_patch_size": 2,
        "quantization": "LUT6" if "lut6" in combined_name else None,
    }
    meta_path = os.path.join(output_dir, f"{combined_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(combined_meta, f, indent=2)
    print(f"[combine] Saved combined metadata: {os.path.basename(meta_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multi-resolution vision encoder mlpackages into one "
                    "multi-function CoreML model."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Directory containing vision_encoder_HxW.mlpackage files")
    parser.add_argument("--name", default="vision_encoder_multi",
                        help="Base name for combined output (default: vision_encoder_multi)")
    parser.add_argument("--suffix", default="",
                        help="Filename suffix to match (e.g. '_lut6' for vision_encoder_HxW_lut6.mlpackage)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if combined model already exists")
    args = parser.parse_args()

    print("=== Vision Encoder Multi-Function Combine ===")
    print(f"  Output dir: {args.output}")
    print(f"  Combined name: {args.name}")
    if args.suffix:
        print(f"  Suffix: {args.suffix}")
    print()

    combine_vision_encoders(args.output, args.name, args.skip_existing, args.suffix)


if __name__ == "__main__":
    main()
