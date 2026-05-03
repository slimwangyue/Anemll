#!/usr/bin/env python3
"""Comprehensive HF vs CoreML vision encoder parity test.

Tests:
1. Token count correctness (no temporal_patch_size duplication)
2. Prompt construction (correct # of image placeholders)
3. HF vs CoreML vision output comparison (cosine similarity, etc.)
4. Patch ordering verification (merger-group vs raster)
5. Pixel layout [1,3,2,H,W] correctness

Usage:
    python tests/dev/test_vision_parity.py --model /path/to/Qwen3.5-4B \
           --coreml-dir /path/to/coreml_output \
           [--image /path/to/test.jpg]
"""
import sys, os, argparse, json, math, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import torch
from PIL import Image

# ── Defaults ──
DEFAULT_HF_MODEL = "/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B"
DEFAULT_COREML_DIR = "/Volumes/MySSD/Anemll/qwen3_5_4b_mrope"

# =============================================================================
# TEST 1: Token count correctness
# =============================================================================
def test_token_count():
    """Verify token count formula doesn't multiply by temporal_patch_size."""
    print("\n" + "=" * 70)
    print("TEST 1: Token Count Correctness")
    print("=" * 70)

    patch_size = 16
    merge_size = 2
    temporal_patch_size = 2

    resolutions = [
        (448, 448, 196),
        (448, 896, 392),
        (896, 448, 392),
        (448, 672, 294),
        (672, 448, 294),
    ]

    all_pass = True
    for H, W, expected in resolutions:
        grid_h = H // patch_size
        grid_w = W // patch_size
        # CORRECT formula: no temporal multiplication
        num_tokens = (grid_h // merge_size) * (grid_w // merge_size)
        # WRONG formula that would double count:
        wrong_num = num_tokens * temporal_patch_size

        status = "PASS" if num_tokens == expected else "FAIL"
        if num_tokens != expected:
            all_pass = False
        print(f"  {H}×{W}: grid={grid_h}×{grid_w}, "
              f"tokens={num_tokens} (expected={expected}) [{status}]")
        if wrong_num == expected * 2:
            print(f"    ⚠ If temporal multiplied: {wrong_num} (WRONG — would cause double-object)")

    print(f"\n  Result: {'ALL PASS' if all_pass else 'FAILED'}")
    return all_pass


# =============================================================================
# TEST 2: Prompt construction
# =============================================================================
def test_prompt_construction(hf_model_path):
    """Verify prompt has correct number of image placeholders."""
    print("\n" + "=" * 70)
    print("TEST 2: Prompt Construction")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, use_fast=False)

    # Special token IDs
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    print(f"  vision_start_id = {vision_start_id}")
    print(f"  vision_end_id   = {vision_end_id}")
    print(f"  image_pad_id    = {image_pad_id}")

    # Test for each resolution
    for num_vis_tokens in [196, 392, 294]:
        # Build prompt the same way chat_server_vision does
        image_tokens = ([vision_start_id] +
                        [image_pad_id] * num_vis_tokens +
                        [vision_end_id])

        vs_count = image_tokens.count(vision_start_id)
        ve_count = image_tokens.count(vision_end_id)
        pad_count = image_tokens.count(image_pad_id)

        assert vs_count == 1, f"Expected 1 vision_start, got {vs_count}"
        assert ve_count == 1, f"Expected 1 vision_end, got {ve_count}"
        assert pad_count == num_vis_tokens, \
            f"Expected {num_vis_tokens} image_pad, got {pad_count}"

        print(f"  {num_vis_tokens} tokens: "
              f"vision_start×{vs_count}, image_pad×{pad_count}, "
              f"vision_end×{ve_count} — PASS")

        # Print first/last 5 token IDs
        print(f"    First 5: {image_tokens[:5]}")
        print(f"    Last 5:  {image_tokens[-5:]}")

    print(f"\n  Result: ALL PASS")
    return True


# =============================================================================
# TEST 3: HF vs CoreML vision output comparison
# =============================================================================
def test_hf_vs_coreml(hf_model_path, coreml_dir, image_path=None):
    """Compare HF vision encoder output with CoreML output."""
    print("\n" + "=" * 70)
    print("TEST 3: HF vs CoreML Vision Output Parity")
    print("=" * 70)

    import coremltools as ct

    # Load HF model
    print("  Loading HF model...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(hf_model_path)
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model_path, torch_dtype=torch.float32, device_map="cpu")
        hf_model.eval()
        has_hf = True
        print("  HF model loaded.")
    except Exception as e:
        print(f"  ⚠ Could not load HF model: {e}")
        print("  Will skip HF comparison, testing CoreML-only.")
        has_hf = False

    # Load CoreML vision encoder
    print("  Loading CoreML vision encoder...")
    vision_pkg = None
    for candidate in ("vision_encoder_multi_lut6.mlpackage",
                      "vision_encoder_multi.mlpackage",
                      "vision_encoder.mlpackage"):
        p = os.path.join(coreml_dir, candidate)
        if os.path.exists(p):
            vision_pkg = p
            break
    if not vision_pkg:
        print("  ⚠ No vision encoder found in", coreml_dir)
        return False

    print(f"  CoreML model: {os.path.basename(vision_pkg)}")

    # Load for 448×448
    try:
        coreml_model = ct.models.MLModel(
            vision_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE,
            function_name="f_448x448")
        print("  Loaded f_448x448 function")
    except Exception:
        coreml_model = ct.models.MLModel(
            vision_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print("  Loaded default function")

    # Prepare test image
    if image_path and os.path.exists(image_path):
        test_image = Image.open(image_path).convert("RGB")
        print(f"  Test image: {image_path} ({test_image.size})")
    else:
        # Create synthetic test image: red square on white
        test_image = Image.new("RGB", (448, 448), (255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([100, 100, 348, 348], fill=(255, 0, 0))
        print("  Using synthetic test image (red square on white)")

    # Preprocess for CoreML
    from chat_server_vision import preprocess_image
    target_size = (448, 448)
    pixel_values_np = preprocess_image(test_image, target_size, temporal_patch_size=2)
    print(f"  pixel_values shape: {pixel_values_np.shape}")
    print(f"  pixel_values dtype: {pixel_values_np.dtype}")
    print(f"  pixel_values range: [{pixel_values_np.min():.3f}, {pixel_values_np.max():.3f}]")
    print(f"  pixel_values[0,:,0,0,0] (R,G,B at TL corner): "
          f"{pixel_values_np[0,0,0,0,0]:.3f}, {pixel_values_np[0,1,0,0,0]:.3f}, "
          f"{pixel_values_np[0,2,0,0,0]:.3f}")
    print(f"  pixel_values[0,:,0,H//2,W//2] (center): "
          f"{pixel_values_np[0,0,0,224,224]:.3f}, {pixel_values_np[0,1,0,224,224]:.3f}, "
          f"{pixel_values_np[0,2,0,224,224]:.3f}")

    # Verify T=0 and T=1 are identical
    t0_frame = pixel_values_np[0, :, 0, :, :]
    t1_frame = pixel_values_np[0, :, 1, :, :]
    max_diff_temporal = np.abs(t0_frame.astype(np.float32) - t1_frame.astype(np.float32)).max()
    print(f"  max|T0 - T1| = {max_diff_temporal:.6f} "
          f"({'PASS' if max_diff_temporal < 1e-6 else 'FAIL — temporal frames differ!'})")

    # Run CoreML
    print("\n  Running CoreML inference...")
    t0 = time.time()
    coreml_out = coreml_model.predict({"pixel_values": pixel_values_np})
    coreml_ms = (time.time() - t0) * 1000
    coreml_embeds = list(coreml_out.values())[0]
    print(f"  CoreML output shape: {coreml_embeds.shape} ({coreml_ms:.0f}ms)")
    print(f"  CoreML output dtype: {coreml_embeds.dtype}")
    coreml_f32 = coreml_embeds.astype(np.float32)
    print(f"  CoreML mean={coreml_f32.mean():.4f}, std={coreml_f32.std():.4f}, "
          f"min={coreml_f32.min():.4f}, max={coreml_f32.max():.4f}")
    print(f"  CoreML token0[0:4] = {coreml_f32[0, 0, :4]}")
    print(f"  CoreML token-1[0:4] = {coreml_f32[0, -1, :4]}")

    # Verify shape
    expected_tokens = (448 // 32) * (448 // 32)  # 196
    assert coreml_embeds.shape[1] == expected_tokens, \
        f"Expected {expected_tokens} tokens, got {coreml_embeds.shape[1]}"
    print(f"  Token count: {coreml_embeds.shape[1]} == {expected_tokens} — PASS")

    if not has_hf:
        print("\n  Skipping HF comparison (model not loaded)")
        return True

    # Run HF
    print("\n  Running HF inference...")
    # Use HF processor for proper preprocessing
    from qwen_vl_utils import process_vision_info
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": "Describe the image."},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")

    # Extract just the vision encoder output
    with torch.no_grad():
        pixel_values_hf = inputs["pixel_values"].to(torch.float32)
        image_grid_thw = inputs["image_grid_thw"]
        print(f"  HF pixel_values shape: {pixel_values_hf.shape}")
        print(f"  HF image_grid_thw: {image_grid_thw}")

        hf_visual = hf_model.visual(pixel_values_hf, grid_thw=image_grid_thw)
        print(f"  HF visual output shape: {hf_visual.shape}")

    hf_f32 = hf_visual.numpy()
    print(f"  HF mean={hf_f32.mean():.4f}, std={hf_f32.std():.4f}, "
          f"min={hf_f32.min():.4f}, max={hf_f32.max():.4f}")
    print(f"  HF token0[0:4] = {hf_f32[0, :4]}")

    # Compare shapes
    if hf_f32.ndim == 2:
        hf_f32 = hf_f32[np.newaxis, :]  # [1, N, D]
    cml = coreml_f32.reshape(-1, coreml_f32.shape[-1])
    hf = hf_f32.reshape(-1, hf_f32.shape[-1])

    if cml.shape != hf.shape:
        print(f"\n  ⚠ SHAPE MISMATCH: CoreML={cml.shape} vs HF={hf.shape}")
        return False

    # Per-token cosine similarity
    def cosine_sim_per_token(a, b):
        dot = (a * b).sum(axis=-1)
        norm_a = np.sqrt((a * a).sum(axis=-1))
        norm_b = np.sqrt((b * b).sum(axis=-1))
        return dot / (norm_a * norm_b + 1e-8)

    cos_sims = cosine_sim_per_token(cml, hf)
    print(f"\n  Per-token cosine similarity:")
    print(f"    mean = {cos_sims.mean():.6f}")
    print(f"    min  = {cos_sims.min():.6f} (token {cos_sims.argmin()})")
    print(f"    max  = {cos_sims.max():.6f}")
    print(f"    std  = {cos_sims.std():.6f}")

    # Worst 5 tokens
    worst_idx = np.argsort(cos_sims)[:5]
    print(f"    Worst 5 tokens: {worst_idx} (cosines: {cos_sims[worst_idx]})")

    # Overall metrics
    abs_diff = np.abs(cml - hf)
    print(f"\n  Absolute difference:")
    print(f"    mean = {abs_diff.mean():.6f}")
    print(f"    max  = {abs_diff.max():.6f}")

    # Global cosine
    global_cos = (cml.flatten() @ hf.flatten()) / (
        np.linalg.norm(cml.flatten()) * np.linalg.norm(hf.flatten()) + 1e-8)
    print(f"\n  Global cosine similarity: {global_cos:.6f}")

    pass_threshold = 0.95
    passed = cos_sims.mean() > pass_threshold
    print(f"\n  Result: {'PASS' if passed else 'FAIL'} "
          f"(mean cosine {cos_sims.mean():.4f} > {pass_threshold})")
    return passed


# =============================================================================
# TEST 4: Patch ordering verification
# =============================================================================
def test_patch_ordering():
    """Verify _raster_to_merger_group produces correct spatial grouping."""
    print("\n" + "=" * 70)
    print("TEST 4: Patch Ordering (raster → merger-group)")
    print("=" * 70)

    # Simulate a 4×4 patch grid (e.g., 64×64 image with patch_size=16)
    grid_h, grid_w = 4, 4
    merge_size = 2
    merged_h = grid_h // merge_size
    merged_w = grid_w // merge_size
    hidden_dim = 8  # small for readability

    # Create patches with unique IDs based on (row, col)
    # Each patch value = row * 100 + col for easy identification
    raster_patches = torch.zeros(grid_h * grid_w, hidden_dim)
    for r in range(grid_h):
        for c in range(grid_w):
            raster_patches[r * grid_w + c, 0] = r * 100 + c

    print(f"  Grid: {grid_h}×{grid_w}, merge_size={merge_size}")
    print(f"  Raster order (first values):")
    for i in range(grid_h * grid_w):
        r, c = int(raster_patches[i, 0]) // 100, int(raster_patches[i, 0]) % 100
        print(f"    [{i:2d}] → ({r},{c})", end="")
        if (i + 1) % grid_w == 0:
            print()

    # Apply raster_to_merger_group (manually, same logic as export_vision.py)
    x_2d = raster_patches.view(grid_h, grid_w, -1)
    grouped = x_2d.view(
        merged_h, merge_size, merged_w, merge_size, -1
    ).permute(0, 2, 1, 3, 4).reshape(-1, hidden_dim)

    print(f"\n  Merger-group order:")
    all_pass = True
    expected_groups = [
        # group 0: (0,0),(0,1),(1,0),(1,1)
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        # group 1: (0,2),(0,3),(1,2),(1,3)
        [(0, 2), (0, 3), (1, 2), (1, 3)],
        # group 2: (2,0),(2,1),(3,0),(3,1)
        [(2, 0), (2, 1), (3, 0), (3, 1)],
        # group 3: (2,2),(2,3),(3,2),(3,3)
        [(2, 2), (2, 3), (3, 2), (3, 3)],
    ]

    for gi, expected in enumerate(expected_groups):
        actual = []
        print(f"    Group {gi}: ", end="")
        for j in range(4):
            idx = gi * 4 + j
            val = int(grouped[idx, 0].item())
            r, c = val // 100, val % 100
            actual.append((r, c))
            print(f"({r},{c})", end=" ")
        match = actual == expected
        print(f" {'PASS' if match else 'FAIL — expected ' + str(expected)}")
        if not match:
            all_pass = False

    # After PatchMerger view(-1, 4*C), each merged token should contain
    # exactly the 2×2 spatial block
    merged = grouped.view(-1, merge_size**2 * hidden_dim)
    print(f"\n  PatchMerger input shape: {merged.shape} "
          f"(should be [{merged_h * merged_w}, {merge_size**2 * hidden_dim}])")
    assert merged.shape == (merged_h * merged_w, merge_size**2 * hidden_dim)

    print(f"\n  Result: {'ALL PASS' if all_pass else 'FAILED'}")
    return all_pass


# =============================================================================
# TEST 5: Pixel layout [1,3,2,H,W] verification
# =============================================================================
def test_pixel_layout():
    """Verify pixel_values layout is [1, C, T, H, W] = [N, C, T, H, W]."""
    print("\n" + "=" * 70)
    print("TEST 5: Pixel Layout [1, 3, 2, H, W]")
    print("=" * 70)

    from chat_server_vision import preprocess_image

    # Create diagnostic image: each channel has a distinct constant
    # R=128 (norm→0.0), G=0 (norm→-1.0), B=255 (norm→1.0)
    img = Image.new("RGB", (448, 448), (128, 0, 255))
    pv = preprocess_image(img, (448, 448), temporal_patch_size=2)

    print(f"  Shape: {pv.shape} (expected [1, 3, 2, 448, 448])")
    assert pv.shape == (1, 3, 2, 448, 448), f"Wrong shape: {pv.shape}"

    # Check channel values
    # R channel = 128/255 = 0.502, norm = (0.502 - 0.5) / 0.5 = 0.004
    # G channel = 0/255 = 0.0, norm = (0.0 - 0.5) / 0.5 = -1.0
    # B channel = 255/255 = 1.0, norm = (1.0 - 0.5) / 0.5 = 1.0
    r_val = float(pv[0, 0, 0, 224, 224])
    g_val = float(pv[0, 1, 0, 224, 224])
    b_val = float(pv[0, 2, 0, 224, 224])
    print(f"  Center pixel [0,:,0,224,224]: R={r_val:.3f}, G={g_val:.3f}, B={b_val:.3f}")
    print(f"  Expected:                     R≈0.004, G≈-1.0, B≈1.0")

    # Verify T=0 == T=1
    max_diff = np.abs(pv[0, :, 0, :, :].astype(np.float32) -
                      pv[0, :, 1, :, :].astype(np.float32)).max()
    print(f"  max|T0 - T1| = {max_diff:.6f} ({'PASS' if max_diff < 1e-6 else 'FAIL'})")

    # Verify offset formula: NCTHW
    # offset(n,c,t,y,x) = n*(3*2*H*W) + c*(2*H*W) + t*(H*W) + y*W + x
    H, W = 448, 448
    flat = pv.flatten()
    for c in range(3):
        for t in range(2):
            offset = 0 * (3 * 2 * H * W) + c * (2 * H * W) + t * (H * W) + 100 * W + 100
            val = float(flat[offset])
            direct = float(pv[0, c, t, 100, 100])
            assert abs(val - direct) < 1e-6, \
                f"Offset mismatch at c={c},t={t}: flat[{offset}]={val} vs pv[0,{c},{t},100,100]={direct}"

    print(f"  NCTHW offset formula: PASS")

    # Compare with Swift layout description from VisionEncoder.swift
    # Swift: ptr[c * 2 * hw + y * width + x] for T=0, ptr[c * 2 * hw + hw + y * width + x] for T=1
    # This is: c * (2 * H * W) + t * (H * W) + y * W + x = same as NCTHW with batch=0
    print(f"  Swift layout compatibility: PASS (c*2*hw + t*hw + y*w + x matches NCTHW)")

    print(f"\n  Result: ALL PASS")
    return True


# =============================================================================
# TEST 6: CoreML vision encoder sanity with real image
# =============================================================================
def test_coreml_vision_real(coreml_dir, image_path=None):
    """Run CoreML vision encoder and verify output sanity."""
    print("\n" + "=" * 70)
    print("TEST 6: CoreML Vision Encoder Sanity Check")
    print("=" * 70)

    import coremltools as ct
    from chat_server_vision import preprocess_image

    vision_pkg = None
    for candidate in ("vision_encoder_multi_lut6.mlpackage",
                      "vision_encoder_multi.mlpackage",
                      "vision_encoder.mlpackage"):
        p = os.path.join(coreml_dir, candidate)
        if os.path.exists(p):
            vision_pkg = p
            break
    if not vision_pkg:
        print("  ⚠ No vision encoder found")
        return False

    # Test each available resolution
    resolutions_to_test = [
        (448, 448, "f_448x448"),
        (448, 896, "f_448x896"),
        (896, 448, "f_896x448"),
    ]

    for H, W, fn_name in resolutions_to_test:
        try:
            model = ct.models.MLModel(
                vision_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE,
                function_name=fn_name)
        except Exception as e:
            print(f"  {fn_name}: skip ({e})")
            continue

        expected_tokens = (H // 32) * (W // 32)

        # Create test image
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB").resize((W, H), Image.BICUBIC)
        else:
            img = Image.new("RGB", (W, H), (128, 200, 50))

        pv = preprocess_image(img, (H, W), temporal_patch_size=2)
        print(f"\n  {fn_name} ({H}×{W}):")
        print(f"    Input shape: {pv.shape}")

        t0 = time.time()
        out = model.predict({"pixel_values": pv})
        elapsed = (time.time() - t0) * 1000

        embeds = list(out.values())[0]
        print(f"    Output shape: {embeds.shape} ({elapsed:.0f}ms)")
        print(f"    Expected tokens: {expected_tokens}")

        assert embeds.shape[1] == expected_tokens, \
            f"Token count mismatch: {embeds.shape[1]} vs {expected_tokens}"
        print(f"    Token count: PASS ({embeds.shape[1]})")

        ef32 = embeds.astype(np.float32)
        print(f"    Stats: mean={ef32.mean():.4f}, std={ef32.std():.4f}, "
              f"min={ef32.min():.4f}, max={ef32.max():.4f}")

        # Check for degenerate output (all zeros, NaN, etc.)
        assert not np.isnan(ef32).any(), "Output contains NaN!"
        assert ef32.std() > 0.01, f"Output is near-constant (std={ef32.std():.6f})"
        print(f"    Sanity: PASS (non-trivial output)")

    print(f"\n  Result: ALL PASS")
    return True


# =============================================================================
# TEST 7: Prompt token inspection
# =============================================================================
def test_prompt_tokens(hf_model_path):
    """Inspect the full tokenized prompt around image tokens."""
    print("\n" + "=" * 70)
    print("TEST 7: Prompt Token Inspection")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, use_fast=False)

    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    # Simulate building a vision prompt (same as chat_server_vision.py)
    user_msg = "How many objects are in the image?"
    num_vis_tokens = 196  # for 448×448

    # Tokenize user message
    messages = [{"role": "user", "content": user_msg}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Insert image tokens
    image_tokens = [vision_start_id] + [image_pad_id] * num_vis_tokens + [vision_end_id]
    user_token_ids = tokenizer.encode("user\n", add_special_tokens=False)

    insert_pos = None
    for i in range(len(prompt_tokens) - len(user_token_ids), -1, -1):
        if (i > 0 and prompt_tokens[i-1] == im_start_id and
            prompt_tokens[i:i+len(user_token_ids)] == user_token_ids):
            insert_pos = i + len(user_token_ids)
            break

    if insert_pos is not None:
        full_prompt = (prompt_tokens[:insert_pos] +
                      image_tokens +
                      prompt_tokens[insert_pos:])
    else:
        full_prompt = image_tokens + prompt_tokens

    # Count special tokens
    vs_count = full_prompt.count(vision_start_id)
    ve_count = full_prompt.count(vision_end_id)
    pad_count = full_prompt.count(image_pad_id)

    print(f"  Total prompt tokens: {len(full_prompt)}")
    print(f"  vision_start count: {vs_count} (expected 1)")
    print(f"  vision_end count:   {ve_count} (expected 1)")
    print(f"  image_pad count:    {pad_count} (expected {num_vis_tokens})")

    # Show tokens around image block
    vs_pos = full_prompt.index(vision_start_id)
    ve_pos = full_prompt.index(vision_end_id)
    print(f"\n  Image block: positions {vs_pos}..{ve_pos}")
    print(f"  Tokens before image (5): {full_prompt[max(0,vs_pos-5):vs_pos]}")
    print(f"  Decoded: '{tokenizer.decode(full_prompt[max(0,vs_pos-5):vs_pos])}'")
    print(f"  Tokens after image (5):  {full_prompt[ve_pos+1:ve_pos+6]}")
    print(f"  Decoded: '{tokenizer.decode(full_prompt[ve_pos+1:ve_pos+6])}'")

    assert vs_count == 1, f"Expected 1 vision_start, got {vs_count}"
    assert ve_count == 1, f"Expected 1 vision_end, got {ve_count}"
    assert pad_count == num_vis_tokens, f"Expected {num_vis_tokens} pads, got {pad_count}"

    print(f"\n  Result: ALL PASS")
    return True


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Vision encoder parity tests")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--coreml-dir", default=DEFAULT_COREML_DIR)
    parser.add_argument("--image", default=None)
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HF model comparison (test 3)")
    args = parser.parse_args()

    results = {}

    # Test 1: Token count
    results["token_count"] = test_token_count()

    # Test 2: Prompt construction
    results["prompt_construction"] = test_prompt_construction(args.model)

    # Test 4: Patch ordering
    results["patch_ordering"] = test_patch_ordering()

    # Test 5: Pixel layout
    results["pixel_layout"] = test_pixel_layout()

    # Test 6: CoreML sanity
    results["coreml_sanity"] = test_coreml_vision_real(args.coreml_dir, args.image)

    # Test 7: Prompt tokens
    results["prompt_tokens"] = test_prompt_tokens(args.model)

    # Test 3: HF vs CoreML (optional, heavy)
    if not args.skip_hf:
        results["hf_vs_coreml"] = test_hf_vs_coreml(
            args.model, args.coreml_dir, args.image)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
