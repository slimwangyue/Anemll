#!/usr/bin/env python3
"""
Test batch prefill with mRoPE vision positions.

Goal: Reproduce the garbled output from Swift's batch prefill by doing
batch prefill with visual embedding injection in Python, then compare
to the known-good sequential result.
"""
import sys, os, time
import numpy as np
from PIL import Image

_SCRIPT_DIR = "/Volumes/MySSD/Anemll/scripts_qwen3_5"
_REPO_ROOT = "/Volumes/MySSD/Anemll"
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

import coremltools as ct

MODEL_DIR = "/Volumes/MySSD/Anemll/qwen3_5_4b_mrope"
HF_PATH = "/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B"
IMG_PATH = "/Volumes/MySSD/Anemll/tests/IMG_8204.PNG"


def batch_prefill_with_vision(engine, prompt_tokens, visual_embeds,
                               grid_h, grid_w, image_token_id=248056):
    """Batch prefill that injects visual embeddings and uses mRoPE positions.

    This replicates what Swift's batchPrefill + injectVisualEmbeddings does.
    """
    bs = engine._prefill_bs
    n_total = len(prompt_tokens)
    num_chunks = engine.num_chunks

    # Find image span
    img_positions = [i for i, t in enumerate(prompt_tokens) if t == image_token_id]
    n_image = len(img_positions)
    span_start = img_positions[0] if img_positions else -1
    span_end = span_start + n_image if img_positions else -1
    rope_delta = max(grid_h, grid_w) - n_image

    print(f"[batch_vision] {n_total} tokens, bs={bs}, "
          f"img_span={span_start}..{span_end-1}, rope_delta={rope_delta}")

    last_next = None
    pos = 0

    # Process in blocks of bs
    block_idx = 0
    offset = 0
    while offset < n_total:
        block_end = min(offset + bs, n_total)
        block_tokens = prompt_tokens[offset:block_end]
        valid_len = len(block_tokens)
        is_last_block = (block_end >= n_total)

        # ── 1. Embed ──
        input_ids = engine._batch_tok_buf
        input_ids[0, :] = 0
        input_ids[0, :valid_len] = block_tokens
        hidden = list(engine.embed_prefill.predict({"input_ids": input_ids}).values())[0]

        # ── 2. Zero-fill padding ──
        if valid_len < bs:
            hidden[:, valid_len:, :] = 0.0

        # ── 3. Inject visual embeddings ──
        replaced = 0
        for i in range(valid_len):
            gp = offset + i
            if gp >= span_start and gp < span_end:
                vis_row = gp - span_start
                hidden[0, i, :] = visual_embeds[0, vis_row, :]
                replaced += 1

        # ── 4. Causal mask ──
        mask = engine._batch_mask_buf
        mask[:, :, :, :] = -65504.0
        for i in range(valid_len):
            mask[0, 0, i, :offset + i + 1] = 0
        for i in range(valid_len, bs):
            mask[0, 0, i, 0] = 0.0

        # ── 5. mRoPE position_ids [3, bs] ──
        pos_ids = engine._batch_pos_buf.copy()
        pos_ids[:, :] = 0
        for i in range(valid_len):
            gp = offset + i
            if gp < span_start:
                # Text before image
                pos_ids[:, i] = gp + engine.rope_offset
            elif gp < span_end:
                vis_idx = gp - span_start
                row = vis_idx // grid_w
                col = vis_idx % grid_w
                pos_ids[0, i] = span_start + engine.rope_offset  # temporal
                pos_ids[1, i] = span_start + engine.rope_offset + row  # height
                pos_ids[2, i] = span_start + engine.rope_offset + col  # width
            else:
                # Text after image
                pos_ids[:, i] = gp + engine.rope_offset + rope_delta

        # ── 6. current_pos and valid_len ──
        cur_pos = engine._batch_cur_buf.copy()
        cur_pos[0] = offset

        valid_len_arr = engine._valid_len_buf.copy()
        valid_len_arr[0] = valid_len

        # ── 7. Run FFN chunks ──
        t0 = time.time()
        for ci in range(num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": cur_pos,
                "linear_conv_state": engine.lin_convs[ci],
                "linear_recurrent_state": engine.lin_recs[ci],
                "valid_len": valid_len_arr,
            }
            out = engine.prefills[ci].predict(inp, state=engine.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                engine.lin_convs[ci] = out['linear_conv_state_out']
                engine.lin_recs[ci] = out['linear_recurrent_state_out']
            if valid_len < bs:
                hidden[:, valid_len:, :] = 0.0
        elapsed = time.time() - t0

        print(f"  block {block_idx}: offset={offset}, valid={valid_len}, "
              f"replaced={replaced}, {elapsed*1000:.0f}ms")

        # ── 8. LM head on last block ──
        if is_last_block:
            if hidden.ndim >= 3 and hidden.shape[1] > 1:
                hidden = hidden[:, valid_len-1:valid_len, :]
            lm_out = engine.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
            if engine.lmhead_mode == "logits":
                logits = engine._extract_logits(lm_out)
                last_next = int(np.argmax(logits))
            else:
                last_next = int(lm_out["argmax_idx"].flatten()[0])

        engine.pos = offset + valid_len
        offset += valid_len
        block_idx += 1

    engine.rope_offset += rope_delta
    print(f"[batch_vision] Done: pos={engine.pos}, rope_offset={engine.rope_offset}")
    return last_next


def decode_tokens(engine, first_token, max_tokens=150):
    """Decode tokens greedily."""
    tokenizer = engine.tokenizer
    generated = []
    stop_ids = {248044, 248046}  # <|im_end|>, <|endoftext|>
    next_id = first_token

    for gen_idx in range(max_tokens):
        if next_id in stop_ids:
            break
        generated.append(next_id)
        next_id, _ = engine._step(next_id, engine.pos)
        engine.pos += 1

    return tokenizer.decode(generated)


def main():
    from chat_server_vision import VisionChatEngine

    print("=" * 60)
    print("BATCH PREFILL VISION TEST")
    print("=" * 60)

    # ── Load engine ──
    print("\n[1] Loading engine...")
    engine = VisionChatEngine(
        MODEL_DIR, HF_PATH,
        ctx=4096, num_chunks=8,
        vision_model_path=os.path.join(MODEL_DIR, "vision_encoder_multi_lut6.mlpackage"),
        image_size=448,
        compute_unit=ct.ComputeUnit.ALL,
    )
    engine.load()
    bs = engine._prefill_bs
    print(f"    Engine ready: bs={bs}, ctx={engine.ctx}, chunks={engine.num_chunks}")
    print(f"    has_prefill={engine.has_prefill}, prefill models={len(engine.prefills) if engine.has_prefill else 0}")

    # ── Load & encode image ──
    print("\n[2] Processing image...")
    image = Image.open(IMG_PATH)
    visual_embeds, active_meta = engine._encode_image(image)
    num_vis = visual_embeds.shape[1]
    img_size = active_meta.get("image_size", [448, 448])
    res_h, res_w = img_size if isinstance(img_size, (list, tuple)) else (img_size, img_size)
    grid_h = res_h // (16 * 2)
    grid_w = res_w // (16 * 2)
    print(f"    Visual: {visual_embeds.shape}, grid={grid_h}x{grid_w}, res={res_h}x{res_w}")

    # ── Build prompt tokens ──
    print("\n[3] Building prompt...")
    engine.messages = [{"role": "user", "content": "What is shown in this image? Describe the product and price."}]
    prompt_tokens = engine._build_vision_prompt_tokens(
        "What is shown in this image? Describe the product and price.",
        enable_thinking=False,
        num_vis_tokens=num_vis,
        active_meta=active_meta,
    )
    n_total = len(prompt_tokens)
    img_positions = [i for i, t in enumerate(prompt_tokens) if t == 248056]
    print(f"    Prompt: {n_total} tokens, {len(img_positions)} image tokens at {img_positions[0]}..{img_positions[-1]}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST A: Batch prefill with vision (should reproduce garbled output)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"TEST A: BATCH PREFILL WITH VISION")
    print(f"{'='*60}")
    engine._reset_states()

    t0 = time.time()
    first_token = batch_prefill_with_vision(
        engine, prompt_tokens, visual_embeds, grid_h, grid_w)
    batch_time = time.time() - t0

    result_batch = decode_tokens(engine, first_token, max_tokens=150)
    print(f"\n[BATCH RESULT] ({batch_time*1000:.0f}ms prefill):")
    print(result_batch[:500])

    # ═══════════════════════════════════════════════════════════════════
    # TEST B: Sequential prefill with vision (known-good reference)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"TEST B: SEQUENTIAL PREFILL WITH VISION (reference)")
    print(f"{'='*60}")
    engine._reset_states()
    engine.messages = [{"role": "user", "content": "What is shown in this image? Describe the product and price."}]

    t0 = time.time()
    first_token_seq = engine._step_with_vision(prompt_tokens, visual_embeds, active_meta)
    seq_time = time.time() - t0

    result_seq = decode_tokens(engine, first_token_seq, max_tokens=150)
    print(f"\n[SEQUENTIAL RESULT] ({seq_time*1000:.0f}ms prefill):")
    print(result_seq[:500])

    # ── Compare ──
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"Batch first token:      {first_token}")
    print(f"Sequential first token: {first_token_seq}")
    print(f"Match: {first_token == first_token_seq}")
    print(f"\nBatch output[:100]:      {result_batch[:100]}")
    print(f"Sequential output[:100]: {result_seq[:100]}")


if __name__ == "__main__":
    main()
