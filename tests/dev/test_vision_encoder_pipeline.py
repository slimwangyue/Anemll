#!/usr/bin/env python3
"""
Validate the Qwen3.5-VL vision encoder preprocessing pipeline.

Tests that:
1. Preprocessing produces values in expected range [-1, 1]
2. Vision encoder produces non-trivial, image-dependent embeddings
3. Results match the expected normalization (mean=std=0.5)

Usage:
    python tests/dev/test_vision_encoder_pipeline.py [path/to/image.jpg]

If an image path is provided, also runs the encoder on that image and prints
token-level statistics useful for diagnosing hallucination issues.
"""

import sys
import os
import numpy as np
from PIL import Image

MODEL_PATH = "/Volumes/MySSD/Edge-AI-agent/local_llm/Resources/Models.bundle/Qwen3.5_4B/vision_encoder.mlpackage"
IMAGE_SIZE = 448  # from vision_meta.json: image_size


def preprocess_image(img_rgb_uint8: np.ndarray, size: int) -> np.ndarray:
    """
    Replicate VisionEncoder.swift preprocessImage() exactly.

    Normalization: mean=std=0.5 → pixel in [0,1] → (pixel - 0.5)/0.5 = 2*pixel - 1 ∈ [-1,1]
    Output shape: [1, 3, 2, H, W] float16 (temporal_patch_size=2, duplicate frames)
    """
    assert img_rgb_uint8.shape == (size, size, 3), \
        f"Expected {size}x{size}x3, got {img_rgb_uint8.shape}"
    assert img_rgb_uint8.dtype == np.uint8

    # Normalize: mean=std=0.5
    normalized = (img_rgb_uint8.astype(np.float32) / 255.0 - 0.5) / 0.5  # [-1, 1]

    # Pack into [1, 3, 2, H, W] float16
    pixel_values = np.zeros((1, 3, 2, size, size), dtype=np.float16)
    for c in range(3):
        pixel_values[0, c, 0] = normalized[:, :, c].astype(np.float16)
        pixel_values[0, c, 1] = normalized[:, :, c].astype(np.float16)

    return pixel_values


def run_encoder(model, pixel_values: np.ndarray) -> np.ndarray:
    output = model.predict({'pixel_values': pixel_values})
    return output['visual_embeddings']  # [1, 196, 2560]


def print_stats(label: str, arr: np.ndarray):
    a = arr.astype(np.float32)
    print(f"  {label}: shape={arr.shape} mean={a.mean():.4f} std={a.std():.4f} "
          f"min={a.min():.4f} max={a.max():.4f}")


def main():
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed. Run: pip install coremltools")
        sys.exit(1)

    print("=" * 60)
    print("Vision Encoder Pipeline Validation")
    print("=" * 60)

    # ── 1. Load model ──
    print(f"\n[1] Loading vision encoder...")
    print(f"    {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("    ERROR: model not found!")
        sys.exit(1)
    model = ct.models.MLModel(MODEL_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print("    OK")

    # ── 2. Preprocessing range test ──
    print(f"\n[2] Preprocessing range test (all-black, all-white):")
    black = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    white = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)

    pv_black = preprocess_image(black, IMAGE_SIZE)
    pv_white = preprocess_image(white, IMAGE_SIZE)

    print(f"  all-black R[0,0]={pv_black[0,0,0,0,0]:.4f}  expected -1.0")
    print(f"  all-white R[0,0]={pv_white[0,0,0,0,0]:.4f}  expected +1.0")

    ok = (abs(float(pv_black[0, 0, 0, 0, 0]) - (-1.0)) < 0.01 and
          abs(float(pv_white[0, 0, 0, 0, 0]) -  1.0)  < 0.01)
    print(f"  Normalization: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  !! Normalization is wrong — values are not in [-1, 1]")
        print("  !! Check VisionEncoder.swift mean/std constants")

    # ── 3. Encoder output: black vs white ──
    print(f"\n[3] Encoder output for all-black vs all-white image:")
    emb_black = run_encoder(model, pv_black)
    emb_white = run_encoder(model, pv_white)
    print_stats("black", emb_black)
    print_stats("white", emb_white)

    diff = np.abs(emb_black.astype(np.float32) - emb_white.astype(np.float32)).mean()
    print(f"  Mean abs diff (black vs white): {diff:.4f}")
    if diff < 0.01:
        print("  !! WARNING: black and white produce nearly identical embeddings!")
        print("  !! Encoder may be broken or producing constant output")
    else:
        print("  Encoder is image-sensitive: OK")

    # ── 4. Random image test ──
    print(f"\n[4] Random image (seed=42):")
    np.random.seed(42)
    rand_img = (np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 255).astype(np.uint8)
    pv_rand = preprocess_image(rand_img, IMAGE_SIZE)
    emb_rand = run_encoder(model, pv_rand)
    print_stats("random", emb_rand)
    print(f"  token[0] first 8 dims: {emb_rand[0, 0, :8]}")

    # ── 5. Check for NaN/Inf ──
    nan_count = np.isnan(emb_rand.astype(np.float32)).sum()
    inf_count = np.isinf(emb_rand.astype(np.float32)).sum()
    print(f"\n[5] NaN/Inf check: NaN={nan_count} Inf={inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("  !! WARNING: embeddings contain NaN or Inf — encoder may be failing on ANE")
    else:
        print("  Clean output: OK")

    # ── 6. Real image (optional) ──
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"\n[6] Real image: {img_path}")
        img = Image.open(img_path).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        img_array = np.array(img)
        print(f"    Source size: {img.size}, array shape: {img_array.shape}")
        pv_real = preprocess_image(img_array, IMAGE_SIZE)
        emb_real = run_encoder(model, pv_real)
        print_stats("real image", emb_real)
        print(f"  token[0] first 8 dims: {emb_real[0, 0, :8]}")
        print(f"  token[97] (center) first 8 dims: {emb_real[0, 97, :8]}")

        # Check if different spatial regions have different embeddings
        diff_corner_center = np.abs(
            emb_real[0, 0].astype(np.float32) - emb_real[0, 97].astype(np.float32)
        ).mean()
        print(f"  token[0] vs token[97] mean diff: {diff_corner_center:.4f}")
        if diff_corner_center < 0.01:
            print("  !! WARNING: spatial tokens are too similar — may indicate wrong input")
        else:
            print("  Spatial variation: OK")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == '__main__':
    main()
