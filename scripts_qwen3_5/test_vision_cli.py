#!/usr/bin/env python3
"""Quick CLI test for vision encoder on a single image+question."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import coremltools as ct
from PIL import Image

# Import the vision chat engine
from chat_server_vision import VisionChatEngine
from chat_server import _find_combined_dir, BATCH_SIZE, BLOCK_SIZE

MODEL_DIR = "/Volumes/MySSD/Anemll/qwen3_5_4b_mrope"
IMAGE_PATH = "/Volumes/MySSD/Anemll/tests/IMG_8981.jpg"
QUESTION = "有几个物体？是什么物体？"
CTX = 4096

# Auto-detect num_chunks
ffn_base = _find_combined_dir(MODEL_DIR)
num_chunks = 4
if ffn_base and os.path.isdir(ffn_base):
    num_chunks = sum(
        1 for f in os.listdir(ffn_base)
        if f.startswith("chunk") and (f.endswith(".mlpackage") or f.endswith(".mlmodelc"))
    )
    if num_chunks > 6:
        num_chunks //= 2
print(f"Auto-detected {num_chunks} FFN chunks")

# Auto-detect batch size
for _ep_name in ("embed_prefill.mlpackage", "embed_prefill.mlmodelc"):
    _ep_path = os.path.join(MODEL_DIR, _ep_name)
    if os.path.exists(_ep_path):
        try:
            _ep_spec = ct.utils.load_spec(_ep_path)
            for inp in _ep_spec.description.input:
                if inp.name == "input_ids":
                    import chat_server
                    chat_server.BATCH_SIZE = inp.type.multiArrayType.shape[1]
                    chat_server.BLOCK_SIZE = chat_server.BATCH_SIZE
                    print(f"Batch size (auto): {chat_server.BATCH_SIZE}")
                    break
        except Exception:
            pass
        break

# Create engine
engine = VisionChatEngine(
    MODEL_DIR, MODEL_DIR,
    ctx=CTX, num_chunks=num_chunks,
    compute_unit=ct.ComputeUnit.CPU_AND_NE,
    vision_model_path=None,  # auto-detect
    image_size=448,
)

print(f"\nModel dir: {MODEL_DIR}")
print(f"Image: {IMAGE_PATH}")
print(f"Question: {QUESTION}")
print(f"CTX: {CTX}\n")

engine.load()

# Load image
print(f"\nLoading image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH)
print(f"Image size: {image.size}, mode: {image.mode}")

# Run chat with image
print(f"\nAsking: {QUESTION}")
print("=" * 60)
print("Response: ", end="", flush=True)

full_response = ""
for event in engine.chat_stream_with_image(
    QUESTION, image=image, max_tokens=512,
    enable_thinking=False, repetition_guard=False,
    temperature=0.7, top_p=0.9, top_k=20,
    repetition_penalty=1.1,
):
    if event.get("type") == "token":
        tok = event.get("text", "") or event.get("token", "")
        print(tok, end="", flush=True)
        full_response += tok
    elif event.get("type") == "error":
        print(f"\nERROR: {event.get('message')}")
    elif event.get("type") == "done":
        break

print("\n" + "=" * 60)
print(f"Full response: {full_response}")
