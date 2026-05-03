#!/usr/bin/env python3
"""Diagnose prompt building and test position strategies for vision tokens.

Checks:
1. Exact prompt token sequence matches expected format
2. Tests different RoPE position strategies for image tokens
3. Compares results to isolate MRoPE vs prompt-building issues
"""
import sys, os, time, json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import coremltools as ct
from PIL import Image

MODEL_DIR = os.path.join(_REPO_ROOT, "qwen3_5_2b_v4_lut4")
HF_PATH   = os.path.join(_REPO_ROOT, "models", "Qwen__Qwen3.5-2B")
IMAGE_PATH = os.path.join(_REPO_ROOT, "tests", "IMG_8204.PNG")
VISION_FP16 = os.path.join(MODEL_DIR, "vision_encoder.mlpackage")

CU = ct.ComputeUnit.CPU_AND_NE
NUM_CHUNKS = 7
CTX = 4096


def step_with_vision_custom_rope(engine, prompt_tokens, visual_embeds,
                                  rope_strategy="sequential"):
    """Process prompt with custom RoPE position strategy for image tokens.

    rope_strategy:
      "sequential" - default: positions 0,1,2,...,N (current behavior)
      "compressed" - image tokens get compressed positions (sqrt(N) range)
      "flat"       - all image tokens get same RoPE position
      "grid"       - image tokens get 2D grid positions flattened row-major
    """
    image_token_id = engine.vision_meta["image_token_id"]
    n_total = len(prompt_tokens)
    num_vis = engine.vision_meta["num_merged_tokens"]

    image_positions = [i for i, t in enumerate(prompt_tokens) if t == image_token_id]
    if not image_positions:
        return engine._process_prompt(prompt_tokens)

    grid_h = int(np.sqrt(num_vis))  # 14 for 196 tokens
    grid_w = num_vis // grid_h      # 14

    print(f"[diag] Strategy={rope_strategy}, {len(image_positions)} image tokens, "
          f"grid={grid_h}x{grid_w}")

    vis_idx = 0
    last_next = None
    image_rope_start = None  # RoPE position of first image token

    for ti, tok_id in enumerate(prompt_tokens):
        if engine.pos >= engine.ctx:
            return None
        is_last = (ti == n_total - 1)

        if tok_id == image_token_id and vis_idx < visual_embeds.shape[1]:
            hidden = visual_embeds[:, vis_idx:vis_idx+1, :]

            # KV cache position always sequential
            mask = engine._mask_buf
            mask[:, :, :, :] = -65504.0
            mask[:, :, :, :engine.pos + 1] = 0

            pos_arr = engine._pos_buf
            pos_arr[0] = engine.pos

            # RoPE position depends on strategy
            rope_arr = engine._rope_buf
            base_rope = engine.pos + engine.rope_offset

            if image_rope_start is None:
                image_rope_start = base_rope

            if rope_strategy == "sequential":
                rope_arr[0] = base_rope
            elif rope_strategy == "compressed":
                # Map 196 image tokens to ~14 positions (grid_h range)
                row = vis_idx // grid_w
                col = vis_idx % grid_w
                # Use max(row, col) to compress to grid_h range
                rope_arr[0] = image_rope_start + max(row, col)
            elif rope_strategy == "flat":
                # All image tokens get the same RoPE position
                rope_arr[0] = image_rope_start
            elif rope_strategy == "grid":
                # Use row index only (compressed to grid_h positions)
                row = vis_idx // grid_w
                rope_arr[0] = image_rope_start + row

            if vis_idx < 3 or vis_idx >= num_vis - 2:
                print(f"  vis[{vis_idx:3d}] pos={engine.pos} rope={rope_arr[0]}")

            vis_idx += 1

            for ci in range(engine.num_chunks):
                inp = {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": rope_arr,
                    "causal_mask": mask,
                    "current_pos": pos_arr,
                    "linear_conv_state": engine.lin_convs[ci],
                    "linear_recurrent_state": engine.lin_recs[ci],
                }
                out = engine.ffns[ci].predict(inp, state=engine.states[ci])
                hidden = out["output_hidden_states"]
                if 'linear_conv_state_out' in out:
                    engine.lin_convs[ci] = out['linear_conv_state_out']
                    engine.lin_recs[ci] = out['linear_recurrent_state_out']

            if is_last:
                lm_out = engine.lmhead.predict(
                    {"hidden_states": hidden.astype(np.float16)})
                if engine.lmhead_mode == "logits":
                    logits = engine._extract_logits(lm_out)
                    last_next = int(np.argmax(logits))
                else:
                    last_next = int(lm_out["argmax_idx"].flatten()[0])

            engine.pos += 1
        else:
            # Text token — adjust rope_offset after image section
            if rope_strategy != "sequential" and image_rope_start is not None and vis_idx > 0:
                # After image section, adjust rope_offset so text continues
                # from a position closer to what MRoPE would produce
                if vis_idx == num_vis:
                    if rope_strategy == "compressed":
                        # Text continues from image_rope_start + grid_h
                        desired_rope = image_rope_start + grid_h
                    elif rope_strategy == "flat":
                        # Text continues from image_rope_start + 1
                        desired_rope = image_rope_start + 1
                    elif rope_strategy == "grid":
                        # Text continues from image_rope_start + grid_h
                        desired_rope = image_rope_start + grid_h
                    else:
                        desired_rope = engine.pos + engine.rope_offset

                    actual_rope = engine.pos + engine.rope_offset
                    if actual_rope != desired_rope:
                        delta = desired_rope - actual_rope
                        engine.rope_offset += delta
                        print(f"  [rope adjust] pos={engine.pos}, "
                              f"rope_offset adjusted by {delta} to {engine.rope_offset}, "
                              f"effective_rope={engine.pos + engine.rope_offset}")
                    vis_idx += 1  # prevent re-adjustment

            if is_last:
                last_next, _ = engine._step(tok_id, engine.pos)
            else:
                engine._step_kv_only(tok_id, engine.pos)
            engine.pos += 1

    print(f"[diag] Prefill complete: pos={engine.pos}, rope_offset={engine.rope_offset}")
    return last_next


def run_with_strategy(vision_path, rope_strategy, max_tokens=200):
    """Run full pipeline with a specific RoPE strategy."""
    from chat_server_vision import VisionChatEngine, preprocess_image

    engine = VisionChatEngine(
        model_dir=MODEL_DIR,
        hf_path=HF_PATH,
        ctx=CTX,
        num_chunks=NUM_CHUNKS,
        vision_model_path=vision_path,
        compute_unit=CU,
    )
    engine.load()

    image = Image.open(IMAGE_PATH)
    visual_embeds, active_meta = engine._encode_image(image)

    user_msg = "Describe what you see in this image."
    engine.messages.append({"role": "user", "content": user_msg})
    prompt_tokens = engine._build_vision_prompt_tokens(user_msg)

    # DIAGNOSTIC: Print exact prompt token sequence
    if rope_strategy == "sequential":
        print(f"\n{'='*60}")
        print("PROMPT TOKEN DIAGNOSTIC")
        print(f"{'='*60}")
        print(f"Total tokens: {len(prompt_tokens)}")
        image_token_id = engine.vision_meta["image_token_id"]
        img_count = sum(1 for t in prompt_tokens if t == image_token_id)
        print(f"Image tokens: {img_count}")
        print(f"\nFirst 10 tokens:")
        for i in range(min(10, len(prompt_tokens))):
            tok_str = engine.tokenizer.decode([prompt_tokens[i]])
            print(f"  [{i:3d}] {prompt_tokens[i]:6d} = {repr(tok_str)}")
        # Show around image boundaries
        img_positions = [i for i, t in enumerate(prompt_tokens) if t == image_token_id]
        if img_positions:
            start = max(0, img_positions[0] - 2)
            end = min(len(prompt_tokens), img_positions[-1] + 3)
            print(f"\nAround image region ({img_positions[0]}..{img_positions[-1]}):")
            for i in range(start, min(start + 5, len(prompt_tokens))):
                tok_str = engine.tokenizer.decode([prompt_tokens[i]])
                print(f"  [{i:3d}] {prompt_tokens[i]:6d} = {repr(tok_str)}")
            print(f"  ... {img_count} image tokens ...")
            for i in range(max(start, end - 5), end):
                tok_str = engine.tokenizer.decode([prompt_tokens[i]])
                print(f"  [{i:3d}] {prompt_tokens[i]:6d} = {repr(tok_str)}")
        # Show last tokens
        print(f"\nLast 10 tokens:")
        for i in range(max(0, len(prompt_tokens) - 10), len(prompt_tokens)):
            tok_str = engine.tokenizer.decode([prompt_tokens[i]])
            print(f"  [{i:3d}] {prompt_tokens[i]:6d} = {repr(tok_str)}")

    # Run with custom strategy
    print(f"\n{'='*60}")
    print(f"Running with rope_strategy={rope_strategy}")
    print(f"{'='*60}")

    t0 = time.time()
    last_next = step_with_vision_custom_rope(
        engine, prompt_tokens, visual_embeds, rope_strategy)
    prefill_ms = (time.time() - t0) * 1000
    print(f"Prefill: {prefill_ms:.0f}ms, first_token={last_next}")

    if last_next is None:
        return "(prefill failed)"

    # Decode
    generated_ids = [last_next]
    for gi in range(max_tokens - 1):
        if engine.pos >= engine.ctx:
            break
        fed_tok = generated_ids[-1]
        next_id, logits = engine._step(fed_tok, engine.pos)
        engine.pos += 1

        if next_id in engine.stop_ids:
            break
        generated_ids.append(next_id)

    text = engine.tokenizer.decode(generated_ids, skip_special_tokens=True)
    tok_count = len(generated_ids)
    elapsed = time.time() - t0
    print(f"Generated {tok_count} tokens in {elapsed:.1f}s")
    return text


def main():
    print("=" * 60)
    print("Vision Prompt Diagnostic & RoPE Strategy Test")
    print("=" * 60)

    strategies = ["sequential", "grid", "flat"]
    results = {}

    for strategy in strategies:
        print(f"\n\n{'#'*60}")
        print(f"# STRATEGY: {strategy}")
        print(f"{'#'*60}")
        text = run_with_strategy(VISION_FP16, strategy)
        results[strategy] = text
        print(f"\n--- {strategy} OUTPUT (first 500 chars) ---")
        print(text[:500])
        print("--- end ---")

    # Summary comparison
    print(f"\n\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    keywords = ["gpu", "graphics", "asus", "tuf", "rtx", "5090", "gaming",
                "1999", "price", "stock", "card", "nvidia", "geforce",
                "amazon", "product"]

    for strategy, text in results.items():
        text_lower = text.lower()
        hits = [k for k in keywords if k in text_lower]
        print(f"\n  {strategy}: {len(hits)} keywords: {hits}")
        # Show first 100 chars
        print(f"  Preview: {text[:100]}...")


if __name__ == "__main__":
    main()
