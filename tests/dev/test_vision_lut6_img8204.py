#!/usr/bin/env python3
"""Test LUT6 2B vision encoder on IMG_8204.PNG end-to-end.

Loads VisionChatEngine with LUT6 vision encoder + full 2B chat pipeline,
feeds the image, and checks if the model describes it correctly.
"""
import sys, os, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import coremltools as ct
from PIL import Image

# ── Paths ──
MODEL_DIR = os.path.join(_REPO_ROOT, "qwen3_5_2b_v4_lut4")
HF_PATH   = os.path.join(_REPO_ROOT, "models", "Qwen__Qwen3.5-2B")
IMAGE_PATH = os.path.join(_REPO_ROOT, "tests", "IMG_8204.PNG")
VISION_LUT6 = os.path.join(MODEL_DIR, "vision_encoder_lut6.mlpackage")
VISION_FP16 = os.path.join(MODEL_DIR, "vision_encoder.mlpackage")

CU = ct.ComputeUnit.CPU_AND_NE
NUM_CHUNKS = 7
CTX = 4096


def cosine_sim(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)


def run_pipeline(vision_path, label):
    """Run full vision+chat pipeline and return the generated text."""
    from chat_server_vision import VisionChatEngine, preprocess_image

    print(f"\n{'='*60}")
    print(f"Pipeline: {label} ({os.path.basename(vision_path)})")
    print(f"{'='*60}")

    engine = VisionChatEngine(
        model_dir=MODEL_DIR,
        hf_path=HF_PATH,
        ctx=CTX,
        num_chunks=NUM_CHUNKS,
        vision_model_path=vision_path,
        compute_unit=CU,
    )
    print("Loading models...")
    t0 = time.time()
    engine.load()
    print(f"Models loaded in {time.time()-t0:.1f}s")

    # Load & preprocess image
    image = Image.open(IMAGE_PATH)
    print(f"Image: {image.size} mode={image.mode}")

    # Run chat with image
    user_msg = "Describe what you see in this image."
    print(f"User: {user_msg}")

    generated = []
    t0 = time.time()
    for event in engine.chat_stream_with_image(
            user_msg, image=image, max_tokens=300,
            enable_thinking=False, repetition_guard=False,
            temperature=0.0, top_p=1.0, top_k=0,
            repetition_penalty=1.0, presence_penalty=0.0,
            frequency_penalty=0.0):
        if event.get("type") == "token":
            generated.append(event["text"])
        elif event.get("type") == "error":
            print(f"  ERROR: {event['message']}")
            break
    elapsed = time.time() - t0
    text = "".join(generated)
    tok_count = len(generated)
    tok_per_s = tok_count / elapsed if elapsed > 0 else 0

    print(f"\nGeneration: {tok_count} tokens in {elapsed:.1f}s ({tok_per_s:.1f} tok/s)")
    print(f"\n--- {label} OUTPUT ---")
    print(text)
    print(f"--- end ---")

    return text


def main():
    print("=" * 60)
    print("Testing LUT6 (2B) vision pipeline on IMG_8204.PNG")
    print("=" * 60)

    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        return
    if not os.path.exists(HF_PATH):
        print(f"ERROR: HF model not found: {HF_PATH}")
        print("       Need tokenizer from HF model dir")
        return

    # Run with LUT6
    text_lut6 = run_pipeline(VISION_LUT6, "LUT6")

    # Run with FP16 for comparison
    text_fp16 = run_pipeline(VISION_FP16, "FP16")

    # Semantic check
    print(f"\n{'='*60}")
    print("SEMANTIC VALIDATION")
    print(f"{'='*60}")
    print("Ground truth: Amazon product page showing ASUS TUF Gaming")
    print("  GeForce RTX 5090 32GB GPU, $1,999.99, out of stock")

    keywords = ["gpu", "graphics", "asus", "tuf", "rtx", "5090", "gaming",
                "1999", "price", "stock", "card", "nvidia", "geforce",
                "amazon", "product"]

    for label, text in [("LUT6", text_lut6), ("FP16", text_fp16)]:
        if not text:
            print(f"\n  {label}: NO OUTPUT")
            continue
        text_lower = text.lower()
        hits = [k for k in keywords if k in text_lower]
        print(f"\n  {label}: {len(hits)}/{len(keywords)} keywords found: {hits}")
        if len(hits) >= 3:
            print(f"  ✅ {label} output is semantically CORRECT")
        elif len(hits) >= 1:
            print(f"  ⚠️  {label} output is PARTIALLY correct")
        else:
            print(f"  ❌ {label} output is WRONG (no relevant keywords)")


if __name__ == "__main__":
    main()
