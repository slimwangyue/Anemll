#!/usr/bin/env python3
"""Test Qwen3.5-2B vision with MRoPE-enabled models on IMG_8204.PNG.

Uses the re-exported LUT6 models with 3D position_ids (MRoPE).
"""
import sys, os, time
import numpy as np
from PIL import Image

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts_qwen3_5"))

# Use 2B config
os.environ["QWEN35_MODEL_SIZE"] = "2B"

MODEL_DIR = os.path.join(_REPO, "qwen3_5_2b_v4_lut6")
HF_PATH = os.path.join(_REPO, "models/Qwen__Qwen3.5-2B")
IMAGE_PATH = os.path.join(_REPO, "tests/IMG_8204.PNG")
VISION_MODEL = os.path.join(MODEL_DIR, "vision_encoder.mlpackage")

# If vision encoder not in lut6 dir, try lut4 dir (vision encoder doesn't change)
if not os.path.exists(VISION_MODEL):
    VISION_MODEL = os.path.join(_REPO, "qwen3_5_2b_v4_lut4", "vision_encoder.mlpackage")

MAX_TOKENS = 300

def main():
    from chat_server_vision import VisionChatEngine
    from config import CTX, NUM_CHUNKS

    print("=" * 60)
    print("Vision MRoPE Test — Qwen3.5-2B LUT6")
    print("=" * 60)
    print(f"Model dir: {MODEL_DIR}")
    print(f"Vision:    {VISION_MODEL}")
    print(f"Image:     {IMAGE_PATH}")
    print()

    # Check models exist
    combined_dir = os.path.join(MODEL_DIR, "combined_LUT6_dedup")
    if not os.path.exists(combined_dir):
        print(f"ERROR: Combined model dir not found: {combined_dir}")
        print("Run combine.py and compile.py first.")
        sys.exit(1)

    server = VisionChatEngine(
        model_dir=MODEL_DIR,
        hf_path=HF_PATH,
        vision_model_path=VISION_MODEL,
        ctx=CTX,
        num_chunks=NUM_CHUNKS,
    )
    server.load()

    # Load image
    image = Image.open(IMAGE_PATH)
    print(f"\nImage loaded: {image.size}, mode={image.mode}")

    # Test with thinking OFF
    print("\n" + "#" * 60)
    print("# TEST: MRoPE + think=off")
    print("#" * 60)

    output_text = ""
    t0 = time.time()
    for event in server.chat_stream_with_image(
        "What do you see in this image? Describe the product, price, and details.",
        image=image,
        max_tokens=MAX_TOKENS,
        enable_thinking=False,
        temperature=0.0,  # greedy for reproducibility
    ):
        if event.get("type") == "token":
            tok = event.get("text", "")
            output_text += tok
            print(tok, end="", flush=True)
        elif event.get("type") == "done":
            break
    elapsed = time.time() - t0
    print(f"\n\n--- Generated {len(output_text.split())} words in {elapsed:.1f}s ---")

    # Check for expected keywords
    text_lower = output_text.lower()
    keywords = {
        "asus": "ASUS" in output_text or "asus" in text_lower,
        "tuf": "TUF" in output_text or "tuf" in text_lower,
        "gaming": "gaming" in text_lower,
        "rtx": "RTX" in output_text or "rtx" in text_lower,
        "5090": "5090" in output_text,
        "gpu": "gpu" in text_lower or "graphics" in text_lower,
        "nvidia": "nvidia" in text_lower or "geforce" in text_lower,
        "price": "$" in output_text or "price" in text_lower,
        "1999": "1999" in output_text or "1,999" in output_text,
    }

    print("\n" + "=" * 60)
    print("KEYWORD CHECK:")
    for kw, found in keywords.items():
        status = "✅" if found else "❌"
        print(f"  {status} {kw}")

    found_count = sum(keywords.values())
    total = len(keywords)
    print(f"\nScore: {found_count}/{total} keywords found")

    if found_count >= 5:
        print("\n🎉 MRoPE FIX WORKING — model correctly reads image text!")
    elif found_count >= 3:
        print("\n⚠️  Partial improvement — some keywords found")
    else:
        print("\n❌ Still garbled — MRoPE may not be the only issue")

    # Compare with old garbled output
    garbled_markers = ["ASUF", "Gng Gce", "R090", "TamieForTX", "532GIDR"]
    garbled_count = sum(1 for m in garbled_markers if m in output_text)
    if garbled_count > 0:
        print(f"\n⚠️  Found {garbled_count}/{len(garbled_markers)} garbled markers still present")
    else:
        print(f"\n✅ No garbled markers found (was: ASUF, Gng Gce, R090, TamieForTX, 532GIDR)")

if __name__ == "__main__":
    main()
