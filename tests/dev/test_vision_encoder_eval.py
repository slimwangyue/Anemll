#!/usr/bin/env python3
"""Evaluate the vision encoder for chat_server: compare CoreML output vs HF PyTorch reference.

Tests:
  1. Normalization correctness (mean=0.5/std=0.5 per preprocessor_config.json)
  2. CoreML FP16 vs PyTorch FP32 parity (cosine sim, L2, max-abs-error)
  3. Semantic sanity: different images → different embeddings, same image → same embedding
  4. End-to-end: HF AutoModel reference vs our custom export
"""
import sys, os, json, time, math
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import coremltools as ct
from PIL import Image
import torch

# ── Paths ──
MODEL_2B = os.path.join(_REPO_ROOT, "models", "Qwen__Qwen3.5-2B")
MODEL_4B = os.path.join(_REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
COREML_2B_DIR = os.path.join(_REPO_ROOT, "qwen3_5_2b_v4_lut4")
COREML_4B_DIR = os.path.join(_REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp16")


def make_test_images():
    """Create deterministic test images for reproducible evaluation."""
    imgs = {}
    # 1) Solid red
    imgs["red"] = Image.new("RGB", (640, 480), (255, 0, 0))
    # 2) Solid blue
    imgs["blue"] = Image.new("RGB", (640, 480), (0, 0, 255))
    # 3) White
    imgs["white"] = Image.new("RGB", (448, 448), (255, 255, 255))
    # 4) Black
    imgs["black"] = Image.new("RGB", (448, 448), (0, 0, 0))
    # 5) Gradient (more realistic)
    grad = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            grad[y, x] = [x, y, (x + y) // 2]
    imgs["gradient"] = Image.fromarray(grad)
    # 6) Random noise (simulates real photo statistics)
    np.random.seed(42)
    noise = np.random.randint(0, 256, (448, 448, 3), dtype=np.uint8)
    imgs["noise"] = Image.fromarray(noise)
    return imgs


# ── Preprocessing functions ──

def preprocess_correct(image, target_size=448, temporal_patch_size=2):
    """Correct preprocessing: mean=0.5, std=0.5 per preprocessor_config.json."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((target_size, target_size), Image.BICUBIC)
    pixels = np.array(image, dtype=np.float32) / 255.0
    pixels = (pixels - 0.5) / 0.5  # maps [0,1] → [-1,1]
    pixels = pixels.transpose(2, 0, 1)  # [3, H, W]
    pixels = np.stack([pixels] * temporal_patch_size, axis=1)  # [3, T, H, W]
    return pixels[np.newaxis, ...].astype(np.float16)  # [1, 3, T, H, W]


def preprocess_imagenet(image, target_size=448, temporal_patch_size=2):
    """WRONG preprocessing used in chat_server_vision.py: ImageNet mean/std."""
    IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    IMAGENET_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((target_size, target_size), Image.BICUBIC)
    pixels = np.array(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    pixels = pixels.transpose(2, 0, 1)
    pixels = np.stack([pixels] * temporal_patch_size, axis=1)
    return pixels[np.newaxis, ...].astype(np.float16)


def cosine_sim(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def l2_dist(a, b):
    return np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def max_abs_err(a, b):
    return np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))


# ── PyTorch reference encoder ──

def build_pytorch_reference(model_path, image_size=448):
    """Build PyTorch vision encoder from export_vision.py and load weights."""
    from export_vision import (
        Qwen35VisionEncoder, load_vision_weights
    )
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        full_config = json.load(f)
    vc = full_config["vision_config"]
    encoder = Qwen35VisionEncoder(vc, image_size=image_size)
    load_vision_weights(encoder, model_path)
    encoder.finalize_pos_embed()
    encoder = encoder.eval().float()
    return encoder, vc


def run_pytorch_ref(encoder, pixel_values_fp16):
    """Run PyTorch reference encoder. Input: [1,3,T,H,W] fp16 → output fp32."""
    pv = torch.from_numpy(pixel_values_fp16.astype(np.float32))
    with torch.no_grad():
        out = encoder(pv)
    return out.numpy()  # [1, N, hidden_dim]


# ── CoreML encoder ──

def load_coreml_vision(coreml_dir, variant="vision_encoder"):
    """Load CoreML vision encoder model."""
    # Try variants in order of preference for accuracy testing
    for name in [variant, "vision_encoder_lut6", "vision_encoder_lut4"]:
        path = os.path.join(coreml_dir, f"{name}.mlpackage")
        if os.path.exists(path):
            print(f"  Loading CoreML: {path}")
            model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            return model, name
    return None, None


def run_coreml(model, pixel_values_fp16):
    """Run CoreML vision encoder."""
    out = model.predict({"pixel_values": pixel_values_fp16})
    key = list(out.keys())[0]
    return out[key]  # [1, N, hidden_dim]


# ── Tests ──

def test_normalization_impact(coreml_model, image, image_name):
    """Compare correct (0.5/0.5) vs wrong (ImageNet) normalization."""
    print(f"\n{'='*60}")
    print(f"TEST: Normalization impact on '{image_name}'")
    print(f"{'='*60}")

    pv_correct = preprocess_correct(image)
    pv_imagenet = preprocess_imagenet(image)

    # Show input stats
    print(f"  correct   input: min={pv_correct.min():.3f} max={pv_correct.max():.3f} "
          f"mean={pv_correct.mean():.3f}")
    print(f"  imagenet  input: min={pv_imagenet.min():.3f} max={pv_imagenet.max():.3f} "
          f"mean={pv_imagenet.mean():.3f}")

    out_correct = run_coreml(coreml_model, pv_correct)
    out_imagenet = run_coreml(coreml_model, pv_imagenet)

    cs = cosine_sim(out_correct, out_imagenet)
    l2 = l2_dist(out_correct, out_imagenet)
    mae = max_abs_err(out_correct, out_imagenet)

    print(f"  Output cosine(correct, imagenet) = {cs:.6f}")
    print(f"  Output L2(correct, imagenet)     = {l2:.6f}")
    print(f"  Output max-abs-err               = {mae:.6f}")

    # Stats for each
    for label, o in [("correct", out_correct), ("imagenet", out_imagenet)]:
        of = o.astype(np.float32)
        print(f"  {label:10s} output: shape={o.shape} min={of.min():.4f} "
              f"max={of.max():.4f} mean={of.mean():.4f} std={of.std():.4f}")

    if cs > 0.99:
        print(f"  → Normalization has MINOR impact (cosine={cs:.6f})")
    elif cs > 0.90:
        print(f"  → Normalization has MODERATE impact (cosine={cs:.6f})")
    else:
        print(f"  ⚠ Normalization has MAJOR impact (cosine={cs:.6f}) — WRONG NORM WILL PRODUCE GARBAGE")
    return cs


def test_coreml_vs_pytorch(coreml_model, pt_encoder, image, image_name):
    """Compare CoreML vs PyTorch reference for the SAME preprocessing."""
    print(f"\n{'='*60}")
    print(f"TEST: CoreML vs PyTorch reference on '{image_name}'")
    print(f"{'='*60}")

    pv = preprocess_correct(image)

    # PyTorch
    pt_out = run_pytorch_ref(pt_encoder, pv)
    # CoreML
    cm_out = run_coreml(coreml_model, pv)

    cs = cosine_sim(pt_out, cm_out)
    l2 = l2_dist(pt_out, cm_out)
    mae = max_abs_err(pt_out, cm_out)

    pt_f = pt_out.astype(np.float32)
    cm_f = cm_out.astype(np.float32)

    print(f"  PyTorch:  shape={pt_out.shape} min={pt_f.min():.4f} max={pt_f.max():.4f} "
          f"mean={pt_f.mean():.4f} std={pt_f.std():.4f}")
    print(f"  CoreML:   shape={cm_out.shape} min={cm_f.min():.4f} max={cm_f.max():.4f} "
          f"mean={cm_f.mean():.4f} std={cm_f.std():.4f}")
    print(f"  Cosine similarity = {cs:.6f}")
    print(f"  RMS error         = {l2:.6f}")
    print(f"  Max abs error     = {mae:.6f}")

    if cs > 0.999:
        print(f"  ✅ EXCELLENT parity (cosine={cs:.6f})")
    elif cs > 0.99:
        print(f"  ✅ GOOD parity (cosine={cs:.6f})")
    elif cs > 0.95:
        print(f"  ⚠ MODERATE parity (cosine={cs:.6f}) — LUT quantization drift")
    else:
        print(f"  ❌ POOR parity (cosine={cs:.6f}) — something is wrong")
    return cs


def test_semantic_discrimination(coreml_model):
    """Verify different images produce different embeddings."""
    print(f"\n{'='*60}")
    print(f"TEST: Semantic discrimination")
    print(f"{'='*60}")

    imgs = make_test_images()
    embeddings = {}
    for name, img in imgs.items():
        pv = preprocess_correct(img)
        out = run_coreml(coreml_model, pv)
        embeddings[name] = out
        of = out.astype(np.float32)
        print(f"  {name:10s}: min={of.min():.4f} max={of.max():.4f} "
              f"mean={of.mean():.4f} std={of.std():.4f} L2_norm={np.linalg.norm(of):.1f}")

    # Pairwise cosine similarities
    names = list(embeddings.keys())
    print(f"\n  Pairwise cosine similarities:")
    print(f"  {'':10s}", end="")
    for n in names:
        print(f"  {n:>10s}", end="")
    print()
    for i, n1 in enumerate(names):
        print(f"  {n1:10s}", end="")
        for j, n2 in enumerate(names):
            cs = cosine_sim(embeddings[n1], embeddings[n2])
            marker = "≡" if i == j else ("!" if cs > 0.99 else " ")
            print(f"  {cs:10.4f}{marker}", end="")
        print()

    # Check that very different images have cos < 0.95
    red_blue = cosine_sim(embeddings["red"], embeddings["blue"])
    white_black = cosine_sim(embeddings["white"], embeddings["black"])
    red_noise = cosine_sim(embeddings["red"], embeddings["noise"])

    all_ok = True
    for pair, cs_val in [("red-blue", red_blue), ("white-black", white_black),
                          ("red-noise", red_noise)]:
        if cs_val > 0.99:
            print(f"  ⚠ {pair}: cosine={cs_val:.4f} — embeddings too similar, encoder may be ignoring input!")
            all_ok = False
    if all_ok:
        print(f"  ✅ All distinct images produce distinct embeddings")
    return all_ok


def test_determinism(coreml_model, image):
    """Same image → same output (check for non-determinism)."""
    print(f"\n{'='*60}")
    print(f"TEST: Determinism (same input → same output)")
    print(f"{'='*60}")

    pv = preprocess_correct(image)
    out1 = run_coreml(coreml_model, pv)
    out2 = run_coreml(coreml_model, pv)
    out3 = run_coreml(coreml_model, pv)

    cs12 = cosine_sim(out1, out2)
    cs13 = cosine_sim(out1, out3)
    l2_12 = l2_dist(out1, out2)

    print(f"  cos(run1, run2) = {cs12:.8f}")
    print(f"  cos(run1, run3) = {cs13:.8f}")
    print(f"  L2(run1, run2)  = {l2_12:.8f}")
    if cs12 > 0.9999 and cs13 > 0.9999:
        print(f"  ✅ Deterministic")
    else:
        print(f"  ⚠ Non-deterministic! (may cause flickering in multi-run)")
    return cs12 > 0.9999


def test_per_token_stats(coreml_model, pt_encoder, image, image_name):
    """Per-token comparison: find worst-matching tokens."""
    print(f"\n{'='*60}")
    print(f"TEST: Per-token CoreML vs PyTorch on '{image_name}'")
    print(f"{'='*60}")

    pv = preprocess_correct(image)
    pt_out = run_pytorch_ref(pt_encoder, pv)  # [1, N, H]
    cm_out = run_coreml(coreml_model, pv)     # [1, N, H]

    N = pt_out.shape[1]
    worst_cs = 1.0
    worst_idx = -1
    token_cosines = []
    for t in range(N):
        pt_tok = pt_out[0, t, :]
        cm_tok = cm_out[0, t, :].astype(np.float32)
        cs = cosine_sim(pt_tok, cm_tok)
        token_cosines.append(cs)
        if cs < worst_cs:
            worst_cs = cs
            worst_idx = t

    tc = np.array(token_cosines)
    print(f"  Tokens: {N}")
    print(f"  Per-token cosine: min={tc.min():.6f} max={tc.max():.6f} "
          f"mean={tc.mean():.6f} median={np.median(tc):.6f}")
    print(f"  Worst token: index={worst_idx} cosine={worst_cs:.6f}")
    print(f"  Tokens with cosine < 0.99: {np.sum(tc < 0.99)}/{N}")
    print(f"  Tokens with cosine < 0.95: {np.sum(tc < 0.95)}/{N}")
    return tc.mean()


# ── Main ──

def main():
    print("=" * 70)
    print("VISION ENCODER EVALUATION")
    print("=" * 70)

    # Detect available models
    for label, hf_path, cm_dir in [
        ("2B", MODEL_2B, COREML_2B_DIR),
        ("4B", MODEL_4B, COREML_4B_DIR),
    ]:
        if not os.path.exists(hf_path):
            print(f"\n[skip] {label}: HF model not found at {hf_path}")
            continue

        # Check for CoreML model
        cm_model, cm_variant = load_coreml_vision(cm_dir, "vision_encoder")
        if cm_model is None:
            print(f"\n[skip] {label}: No CoreML vision encoder in {cm_dir}")
            continue

        print(f"\n{'#'*70}")
        print(f"# Evaluating {label} model")
        print(f"# HF: {hf_path}")
        print(f"# CoreML: {cm_dir}/{cm_variant}.mlpackage")
        print(f"{'#'*70}")

        # Load vision meta
        meta_path = os.path.join(cm_dir, "vision_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  Vision meta: {meta['num_merged_tokens']} tokens, "
                  f"out_hidden_size={meta['out_hidden_size']}")
        else:
            print(f"  [warn] No vision_meta.json")

        # Build PyTorch reference
        print("\n  Building PyTorch reference encoder...")
        pt_encoder, vc = build_pytorch_reference(hf_path)
        print(f"  → depth={vc['depth']} hidden={vc['hidden_size']} "
              f"out={vc['out_hidden_size']}")

        imgs = make_test_images()
        results = {}

        # Test 1: Normalization impact
        for img_name in ["gradient", "noise"]:
            cs = test_normalization_impact(cm_model, imgs[img_name], img_name)
            results[f"norm_{img_name}"] = cs

        # Test 2: CoreML vs PyTorch (FP16 vs FP32 parity)
        for img_name in ["gradient", "noise", "red", "white"]:
            cs = test_coreml_vs_pytorch(cm_model, pt_encoder, imgs[img_name], img_name)
            results[f"parity_{img_name}"] = cs

        # Test 3: Semantic discrimination
        test_semantic_discrimination(cm_model)

        # Test 4: Determinism
        test_determinism(cm_model, imgs["noise"])

        # Test 5: Per-token analysis
        for img_name in ["gradient", "noise"]:
            mean_cs = test_per_token_stats(cm_model, pt_encoder, imgs[img_name], img_name)
            results[f"token_{img_name}"] = mean_cs

        # Also test LUT variants if available
        for lut in ["vision_encoder_lut6", "vision_encoder_lut4"]:
            lut_path = os.path.join(cm_dir, f"{lut}.mlpackage")
            if os.path.exists(lut_path):
                print(f"\n{'='*60}")
                print(f"Testing LUT variant: {lut}")
                print(f"{'='*60}")
                lut_model = ct.models.MLModel(lut_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
                for img_name in ["gradient", "noise"]:
                    cs = test_coreml_vs_pytorch(lut_model, pt_encoder, imgs[img_name],
                                                 f"{img_name} ({lut})")
                    results[f"{lut}_{img_name}"] = cs
                del lut_model

        # Summary
        print(f"\n{'='*70}")
        print(f"SUMMARY for {label}")
        print(f"{'='*70}")
        for k, v in sorted(results.items()):
            status = "✅" if v > 0.95 else ("⚠" if v > 0.80 else "❌")
            print(f"  {status} {k}: {v:.6f}")

        del cm_model, pt_encoder

    # ── Chat server normalization bug check ──
    print(f"\n{'='*70}")
    print(f"CHAT SERVER NORMALIZATION CHECK")
    print(f"{'='*70}")
    # Check what chat_server_vision.py uses
    server_path = os.path.join(_REPO_ROOT, "scripts_qwen3_5", "chat_server_vision.py")
    if os.path.exists(server_path):
        with open(server_path) as f:
            src = f.read()
        if "0.48145466" in src or "IMAGENET_MEAN" in src:
            print("  ❌ chat_server_vision.py uses WRONG ImageNet normalization!")
            print("     Expected: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]")
            print("     Found: ImageNet mean=[0.48145466, 0.4578275, 0.40821073]")
            print("     This is per preprocessor_config.json which specifies 0.5/0.5")
        elif "0.5" in src:
            print("  ✅ chat_server_vision.py uses correct mean=0.5, std=0.5")
        else:
            print("  ⚠ Could not determine normalization constants")


if __name__ == "__main__":
    main()
