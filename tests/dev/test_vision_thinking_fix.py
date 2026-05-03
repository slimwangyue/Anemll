#!/usr/bin/env python3
"""Test vision pipeline with enable_thinking fix and rope_delta adjustment.

Tests:
1. enable_thinking=True (old behavior)
2. enable_thinking=False (bug fix — no <think> in prompt)
3. enable_thinking=False + rope_delta (HF-compatible position offset)
"""
import sys, os, time, json, math
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


def step_with_vision_rope_delta(engine, prompt_tokens, visual_embeds):
    """Like _step_with_vision but adjusts rope_offset after image tokens.

    After processing N image tokens in a grid_h x grid_w grid,
    adjusts rope_offset so text tokens get positions as if only
    grid_h positions were consumed (matching HF's rope_deltas).
    """
    image_token_id = engine.vision_meta["image_token_id"]
    num_vis = engine.vision_meta["num_merged_tokens"]
    n_total = len(prompt_tokens)
    grid_h = int(math.sqrt(num_vis))

    image_positions = [i for i, t in enumerate(prompt_tokens) if t == image_token_id]
    if not image_positions:
        return engine._process_prompt(prompt_tokens)

    print(f"[rope_delta] {len(image_positions)} image tokens, grid={grid_h}x{grid_h}")
    print(f"[rope_delta] HF rope_delta would be: {grid_h - num_vis} = "
          f"{grid_h} (effective grid) - {num_vis} (num tokens)")

    vis_idx = 0
    last_next = None
    image_section_done = False

    for ti, tok_id in enumerate(prompt_tokens):
        if engine.pos >= engine.ctx:
            return None
        is_last = (ti == n_total - 1)

        if tok_id == image_token_id and vis_idx < visual_embeds.shape[1]:
            hidden = visual_embeds[:, vis_idx:vis_idx+1, :]
            vis_idx += 1

            mask = engine._mask_buf
            mask[:, :, :, :] = -65504.0
            mask[:, :, :, :engine.pos + 1] = 0

            pos_arr = engine._pos_buf
            pos_arr[0] = engine.pos

            rope_arr = engine._rope_buf
            rope_arr[0] = engine.pos + engine.rope_offset

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
            # After all image tokens, adjust rope_offset
            if not image_section_done and vis_idx == num_vis:
                image_section_done = True
                # HF rope_delta = max_pos_in_grid - num_image_tokens
                # = (text_before + grid_h - 1) - (text_before + num_vis - 1)
                # = grid_h - num_vis
                rope_delta = grid_h - num_vis  # e.g., 14 - 196 = -182
                engine.rope_offset += rope_delta
                print(f"[rope_delta] Applied delta={rope_delta}, "
                      f"new rope_offset={engine.rope_offset}, "
                      f"next text rope={engine.pos + engine.rope_offset}")

            if is_last:
                last_next, _ = engine._step(tok_id, engine.pos)
            else:
                engine._step_kv_only(tok_id, engine.pos)
            engine.pos += 1

    print(f"[rope_delta] Prefill complete: pos={engine.pos}, "
          f"rope_offset={engine.rope_offset}")
    return last_next


def run_test(vision_path, enable_thinking, use_rope_delta, max_tokens=200):
    """Run full pipeline with specific settings."""
    from chat_server_vision import VisionChatEngine, preprocess_image

    label = f"think={'on' if enable_thinking else 'off'}"
    if use_rope_delta:
        label += "+rope_delta"

    print(f"\n{'#'*60}")
    print(f"# TEST: {label}")
    print(f"{'#'*60}")

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
    prompt_tokens = engine._build_vision_prompt_tokens(
        user_msg, enable_thinking=enable_thinking)

    # Show prompt summary
    image_token_id = engine.vision_meta["image_token_id"]
    img_count = sum(1 for t in prompt_tokens if t == image_token_id)
    print(f"Prompt: {len(prompt_tokens)} tokens ({img_count} image)")

    # Show last few tokens to verify thinking presence
    print("Last 5 tokens:")
    for i in range(max(0, len(prompt_tokens) - 5), len(prompt_tokens)):
        tok_str = engine.tokenizer.decode([prompt_tokens[i]])
        print(f"  [{i}] {prompt_tokens[i]} = {repr(tok_str)}")

    # Prefill
    t0 = time.time()
    if use_rope_delta:
        last_next = step_with_vision_rope_delta(
            engine, prompt_tokens, visual_embeds)
    else:
        last_next = engine._step_with_vision(prompt_tokens, visual_embeds)
    prefill_s = time.time() - t0

    if last_next is None:
        return label, "(prefill failed)"

    first_tok = engine.tokenizer.decode([last_next])
    print(f"Prefill: {prefill_s:.1f}s, first_token={last_next} ({repr(first_tok)})")

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
    elapsed = time.time() - t0
    print(f"Generated {len(generated_ids)} tokens in {elapsed:.1f}s")

    print(f"\n--- {label} OUTPUT ---")
    print(text[:600])
    print("--- end ---")

    return label, text


def main():
    print("=" * 60)
    print("Vision Pipeline: Thinking Bug Fix + RoPE Delta Test")
    print("=" * 60)

    results = []

    # Test 1: Original behavior (thinking ON)
    results.append(run_test(VISION_FP16, enable_thinking=True,
                            use_rope_delta=False))

    # Test 2: Thinking OFF (bug fix)
    results.append(run_test(VISION_FP16, enable_thinking=False,
                            use_rope_delta=False))

    # Test 3: Thinking OFF + rope_delta
    results.append(run_test(VISION_FP16, enable_thinking=False,
                            use_rope_delta=True))

    # Summary
    print(f"\n\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    keywords = ["gpu", "graphics", "asus", "tuf", "rtx", "5090", "gaming",
                "1999", "price", "stock", "card", "nvidia", "geforce",
                "amazon", "product"]

    for label, text in results:
        text_lower = text.lower()
        hits = [k for k in keywords if k in text_lower]
        print(f"\n  {label}:")
        print(f"    Keywords: {len(hits)}/{len(keywords)} = {hits}")
        print(f"    First 120: {text[:120]}...")
        if len(hits) >= 5:
            print(f"    ✅ GOOD")
        elif len(hits) >= 3:
            print(f"    ⚠️  PARTIAL")
        else:
            print(f"    ❌ POOR")


if __name__ == "__main__":
    main()
