#!/usr/bin/env python3
"""Quantize vision encoder with LUT4/LUT6 and test accuracy vs FP16/HF."""
import sys, os, time, json, glob
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct
import coremltools.optimize.coreml as cto_coreml


def quantize_model(input_path, nbits, group_size=8):
    """Apply LUT quantization to a CoreML model."""
    model = ct.models.MLModel(input_path)
    config = cto_coreml.OptimizationConfig(
        global_config=cto_coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=nbits,
            granularity="per_grouped_channel",
            group_size=group_size,
            num_kmeans_workers=1,
        )
    )
    t0 = time.time()
    quantized = cto_coreml.palettize_weights(model, config)
    elapsed = time.time() - t0
    return quantized, elapsed


def dir_size(path):
    return sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fns in os.walk(path) for f in fns)


def test_accuracy(fp16_path, lut_path, label, image_size=448):
    """Compare LUT model vs FP16 model on a test image."""
    from PIL import Image
    from chat_server_vision import preprocess_image

    # Create test image
    np.random.seed(42)
    px = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
    img = Image.fromarray(px)
    pv = preprocess_image(img, target_size=image_size)

    # Run FP16
    fp16_model = ct.models.MLModel(fp16_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    fp16_out = list(fp16_model.predict({"pixel_values": pv}).values())[0].squeeze(0)

    # Run LUT
    lut_model = ct.models.MLModel(lut_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    t0 = time.time()
    lut_out = list(lut_model.predict({"pixel_values": pv}).values())[0].squeeze(0)
    lut_time = (time.time() - t0) * 1000

    # Compare
    from numpy.linalg import norm
    fp16_f = fp16_out.astype(np.float32)
    lut_f = lut_out.astype(np.float32)
    diff = np.abs(fp16_f - lut_f)
    cos = np.array([np.dot(fp16_f[i], lut_f[i]) / (norm(fp16_f[i]) * norm(lut_f[i]) + 1e-8)
                    for i in range(len(fp16_f))])
    corr = np.corrcoef(fp16_f.flatten(), lut_f.flatten())[0, 1]

    print(f"\n  {label} vs FP16:")
    print(f"    Cosine: mean={cos.mean():.6f} min={cos.min():.6f}")
    print(f"    Abs diff: mean={diff.mean():.6f} max={diff.max():.6f}")
    print(f"    Pearson r: {corr:.6f}")
    print(f"    Inference: {lut_time:.0f}ms")
    return cos.mean()


def discover_vision_packages(model_dir):
    """Find all per-resolution FP16 vision encoder packages in a directory.

    Returns list of (name_stem, path) tuples, e.g.:
      [("vision_encoder", ".../vision_encoder.mlpackage"),
       ("vision_encoder_448x896", ".../vision_encoder_448x896.mlpackage"), ...]
    """
    import re
    packages = []
    for fname in sorted(os.listdir(model_dir)):
        # Match vision_encoder.mlpackage or vision_encoder_HxW.mlpackage
        # but NOT _lut4 / _lut6 variants
        if not fname.endswith(".mlpackage"):
            continue
        stem = fname[:-len(".mlpackage")]
        if stem == "vision_encoder":
            packages.append((stem, os.path.join(model_dir, fname)))
        elif re.match(r'^vision_encoder_\d+x\d+$', stem):
            packages.append((stem, os.path.join(model_dir, fname)))
    return packages


def quantize_all_resolutions(model_dir, nbits=6, group_size=8, skip_existing=False):
    """Quantize all per-resolution FP16 vision encoders in model_dir.

    Produces <stem>_lut<nbits>.mlpackage for each FP16 package found.
    """
    packages = discover_vision_packages(model_dir)
    if not packages:
        print(f"ERROR: No vision_encoder*.mlpackage files found in {model_dir}")
        sys.exit(1)

    suffix = f"lut{nbits}"
    results = []
    for stem, fp16_path in packages:
        out_name = f"{stem}_{suffix}.mlpackage"
        out_path = os.path.join(model_dir, out_name)

        if skip_existing and os.path.exists(out_path):
            print(f"  [skip] {out_name} (already exists)")
            results.append((stem, out_path))
            continue

        fp16_sz = dir_size(fp16_path)
        print(f"\n--- {suffix.upper()} quantization: {os.path.basename(fp16_path)} ---")
        q_model, elapsed = quantize_model(fp16_path, nbits=nbits, group_size=group_size)
        q_model.save(out_path)
        q_sz = dir_size(out_path)
        print(f"  Done in {elapsed:.1f}s → {q_sz / 1e6:.1f} MB "
              f"({100 * q_sz // fp16_sz}% of {fp16_sz / 1e6:.1f} MB)")
        results.append((stem, out_path))

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Quantize vision encoder(s) with LUT palettization.")
    parser.add_argument("--model-dir", required=True,
                        help="Dir containing vision_encoder*.mlpackage files")
    parser.add_argument("--nbits", type=int, default=6, choices=[4, 6],
                        help="Quantization bits (default: 6)")
    parser.add_argument("--group-size", type=int, default=8,
                        help="Per-grouped-channel group size (default: 8)")
    parser.add_argument("--all-resolutions", action="store_true",
                        help="Quantize all per-resolution packages (not just 448×448)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if quantized model already exists")
    parser.add_argument("--test", action="store_true",
                        help="Test accuracy after quantization (single-resolution only)")
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()

    if args.all_resolutions:
        # Multi-resolution mode: quantize all found packages
        results = quantize_all_resolutions(
            args.model_dir, nbits=args.nbits, group_size=args.group_size,
            skip_existing=args.skip_existing)
        print(f"\nQuantized {len(results)} vision encoder(s) to LUT{args.nbits}")
        return

    # Legacy single-resolution mode (backward compatible)
    fp16_path = os.path.join(args.model_dir, "vision_encoder.mlpackage")
    if not os.path.exists(fp16_path):
        print(f"ERROR: {fp16_path} not found")
        sys.exit(1)

    fp16_sz = dir_size(fp16_path)
    print(f"FP16 model: {fp16_sz / 1e6:.1f} MB")

    # LUT4
    lut4_path = os.path.join(args.model_dir, "vision_encoder_lut4.mlpackage")
    print(f"\n--- LUT4 quantization ---")
    lut4_model, t4 = quantize_model(fp16_path, nbits=4)
    lut4_model.save(lut4_path)
    lut4_sz = dir_size(lut4_path)
    print(f"  Done in {t4:.1f}s → {lut4_sz / 1e6:.1f} MB ({100 * lut4_sz // fp16_sz}%)")

    # LUT6
    lut6_path = os.path.join(args.model_dir, "vision_encoder_lut6.mlpackage")
    print(f"\n--- LUT6 quantization ---")
    lut6_model, t6 = quantize_model(fp16_path, nbits=6)
    lut6_model.save(lut6_path)
    lut6_sz = dir_size(lut6_path)
    print(f"  Done in {t6:.1f}s → {lut6_sz / 1e6:.1f} MB ({100 * lut6_sz // fp16_sz}%)")

    if args.test:
        print(f"\n=== Accuracy Test (image_size={args.image_size}) ===")
        cos4 = test_accuracy(fp16_path, lut4_path, "LUT4", args.image_size)
        cos6 = test_accuracy(fp16_path, lut6_path, "LUT6", args.image_size)

        print(f"\n=== Summary ===")
        print(f"  FP16: {fp16_sz / 1e6:.1f} MB (reference)")
        print(f"  LUT4: {lut4_sz / 1e6:.1f} MB, cosine={cos4:.6f}")
        print(f"  LUT6: {lut6_sz / 1e6:.1f} MB, cosine={cos6:.6f}")

        if cos4 > 0.995:
            print(f"\n  → LUT4 is accurate enough (cosine > 0.995)")
            best = "lut4"
        elif cos6 > 0.995:
            print(f"\n  → LUT4 too lossy, use LUT6 (cosine > 0.995)")
            best = "lut6"
        else:
            print(f"\n  → Both LUT4 and LUT6 have significant loss, keep FP16")
            best = "fp16"

        # Create the final vision_encoder with the best quantization
        final_path = os.path.join(args.model_dir, f"vision_encoder_{best}_final.mlpackage")
        if best == "lut4":
            import shutil
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.copytree(lut4_path, final_path)
        elif best == "lut6":
            import shutil
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.copytree(lut6_path, final_path)


if __name__ == "__main__":
    main()
