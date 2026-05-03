#!/usr/bin/env python3
"""Vision-capable chat server for Qwen3.5-4B on ANE.

Extends the text-only chat_server with image understanding.
Loads an additional vision_encoder CoreML model that processes images
into visual embeddings injected at <image> token positions.

Usage:
    python scripts_qwen3_5/chat_server_vision.py --model-dir /path/to/models
    python scripts_qwen3_5/chat_server_vision.py --image-size 448

Then open http://localhost:8080 in your browser and paste/upload images.
"""
import sys, os, json, time, io, base64, re, argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct
from PIL import Image

# ── Multi-resolution support ──
# Import shared resolution helpers from export_vision
from export_vision import (
    smart_resize, find_best_resolution, SUPPORTED_RESOLUTIONS,
    DEFAULT_IMAGE_SIZE,
)

# ── Image Preprocessing ──

# Qwen3.5-VL normalization: mean=0.5, std=0.5 (per preprocessor_config.json)
# NOT ImageNet normalization — this was verified by comparing outputs:
# wrong norm (ImageNet) vs correct norm (0.5/0.5) gives cosine ~0.57-0.72
QWEN_VL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
QWEN_VL_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess_image(image, target_size=448, temporal_patch_size=2):
    """Preprocess a PIL Image for the Qwen3.5 vision encoder.

    Args:
        image: PIL Image (any size, any mode)
        target_size: int for square, or (H, W) tuple for rectangular
        temporal_patch_size: number of temporal frames (2 for Qwen3.5)

    Returns:
        pixel_values: np.ndarray [1, 3, T, H, W] float16
    """
    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Parse target size
    if isinstance(target_size, (tuple, list)):
        target_h, target_w = target_size
    else:
        target_h = target_w = target_size

    # Resize (bicubic, like HF processor)
    image = image.resize((target_w, target_h), Image.BICUBIC)

    # To numpy [H, W, 3] float32 in [0, 1]
    pixels = np.array(image, dtype=np.float32) / 255.0

    # Normalize
    pixels = (pixels - QWEN_VL_MEAN) / QWEN_VL_STD

    # [H, W, 3] → [3, H, W]
    pixels = pixels.transpose(2, 0, 1)

    # Duplicate for temporal dimension: [3, H, W] → [3, T, H, W]
    pixels = np.stack([pixels] * temporal_patch_size, axis=1)

    # Add batch dim: [1, 3, T, H, W]
    pixels = pixels[np.newaxis, ...]

    return pixels.astype(np.float16)


def decode_image_from_data_url(data_url):
    """Decode a base64 data URL to PIL Image."""
    # data:image/png;base64,iVBOR...
    match = re.match(r'data:image/[^;]+;base64,(.+)', data_url)
    if match:
        img_data = base64.b64decode(match.group(1))
        return Image.open(io.BytesIO(img_data))
    return None


def decode_image_from_path(path):
    """Load image from file path."""
    return Image.open(path)


# ── Vision-Capable Chat Engine ──

# Import the base chat server components
from chat_server import (
    ChatEngine, ChatHandler,
    _load_model, _find_model, _find_combined_dir, _chunk_tokens,
    RepetitionDetector, _cleanup_ane_temp,
    BATCH_SIZE, BLOCK_SIZE, MIN_GEN_RESERVE, PREFILL_CROSSOVER,
    SYSTEM_PROMPT
)


class VisionChatEngine(ChatEngine):
    """Chat engine with vision/image understanding capabilities."""

    def __init__(self, *args, vision_model_path=None, image_size=448, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_model_path = vision_model_path
        self.image_size = image_size
        self.vision_model = None
        self.vision_meta = None
        self.pending_image = None  # PIL Image waiting to be processed
        # Multi-resolution support: maps (H, W) → loaded CoreML model
        self.vision_models = {}
        # Maps (H, W) → vision_meta dict
        self.vision_metas = {}
        # List of available (H, W) resolutions (sorted)
        self.available_resolutions = []
        # Combined multi-function model (vision_encoder_multi.mlpackage), if present.
        # Populated by load(); one MLModel instance per resolution, each locked to
        # the corresponding function name (f_HxW).
        self._multi_pkg_path = None  # path to the combined package (for logging)

    def _load_multi_vision_meta(self, model_dir, h, w):
        """Load or compute metadata for one resolution from the combined meta file."""
        # Try combined meta first
        for meta_fname in (f"vision_encoder_multi_meta.json",):
            mp = os.path.join(model_dir, meta_fname)
            if os.path.exists(mp):
                with open(mp) as f:
                    combined = json.load(f)
                fn = f"f_{h}x{w}"
                if fn in combined.get("functions", {}):
                    return combined["functions"][fn]
        # Per-resolution meta
        for meta_fname in (f"vision_meta_{h}x{w}.json", "vision_meta.json"):
            mp = os.path.join(model_dir, meta_fname)
            if os.path.exists(mp):
                with open(mp) as f:
                    return json.load(f)
        # Fallback: compute defaults
        patch_size, merge_size = 16, 2
        return {
            "image_size": [h, w],
            "num_merged_tokens": (h // (patch_size * merge_size))
                                 * (w // (patch_size * merge_size)),
            "image_token_id": 248056,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "temporal_patch_size": 2,
            "patch_size": patch_size,
            "spatial_merge_size": merge_size,
        }

    def load(self):
        """Load all models including vision encoder(s)."""
        super().load()

        model_dir = os.path.dirname(self.vision_model_path) if self.vision_model_path else self.model_dir

        # ── Priority 1: combined multi-function vision encoder ──
        if model_dir and os.path.isdir(model_dir):
            import re as _re
            # Accept any variant: vision_encoder_multi*.mlpackage
            multi_pkg = None
            for candidate in ("vision_encoder_multi_lut6.mlpackage",
                              "vision_encoder_multi.mlpackage"):
                p = os.path.join(model_dir, candidate)
                if os.path.exists(p):
                    multi_pkg = p
                    break
            if multi_pkg and os.path.exists(multi_pkg):
                print(f"[engine] Found combined vision encoder: {multi_pkg}")
                # Load combined meta to discover available resolutions
                combined_meta_path = os.path.join(model_dir, "vision_encoder_multi_meta.json")
                resolutions = []
                if os.path.exists(combined_meta_path):
                    with open(combined_meta_path) as f:
                        combined_meta = json.load(f)
                    for h, w in combined_meta.get("resolutions", []):
                        resolutions.append((h, w))
                # Fallback: discover resolutions from resolutions.json or SUPPORTED_RESOLUTIONS
                if not resolutions:
                    res_json = os.path.join(model_dir, "vision_resolutions.json")
                    if os.path.exists(res_json):
                        with open(res_json) as f:
                            idx = json.load(f)
                        resolutions = [tuple(r) for r in idx.get("supported_resolutions", [])]
                if not resolutions:
                    resolutions = list(SUPPORTED_RESOLUTIONS)

                # Load one MLModel instance per resolution, each locked to its function
                self._multi_pkg_path = multi_pkg
                loaded = 0
                for h, w in resolutions:
                    fn_name = f"f_{h}x{w}"
                    try:
                        model = _load_model(multi_pkg, self.compute_unit,
                                            function_name=fn_name)
                        self.vision_models[(h, w)] = model
                        self.vision_metas[(h, w)] = self._load_multi_vision_meta(
                            model_dir, h, w)
                        tokens = self.vision_metas[(h, w)]["num_merged_tokens"]
                        print(f"  {h}×{w} → fn={fn_name}: {tokens} tokens")
                        loaded += 1
                    except Exception as e:
                        print(f"  Warning: could not load {fn_name} from combined model: {e}")
                print(f"[engine] Combined multi-function vision: {loaded} functions loaded")
                self.available_resolutions = sorted(self.vision_models.keys())
                # Set vision_meta to the default resolution's meta for fallback use
                default_res = (448, 448)
                if default_res in self.vision_metas:
                    self.vision_meta = self.vision_metas[default_res]
                    self.vision_model = self.vision_models[default_res]
                elif self.available_resolutions:
                    first = self.available_resolutions[0]
                    self.vision_meta = self.vision_metas[first]
                    self.vision_model = self.vision_models[first]

        # ── Priority 2: individual per-resolution mlpackages ──
        if not self.vision_models and model_dir and os.path.isdir(model_dir):
            import re as _re
            for fname in sorted(os.listdir(model_dir)):
                # Match vision_encoder_HxW.mlpackage or .mlmodelc
                m = _re.match(r'vision_encoder_(\d+)x(\d+)\.(mlpackage|mlmodelc)$', fname)
                if m:
                    res_h, res_w = int(m.group(1)), int(m.group(2))
                    # Skip if already loaded (e.g. both .mlmodelc and .mlpackage present)
                    if (res_h, res_w) in self.vision_models:
                        continue
                    res_path = os.path.join(model_dir, fname)
                    print(f"[engine] Loading vision encoder {res_h}×{res_w}...")
                    self.vision_models[(res_h, res_w)] = _load_model(
                        res_path, self.compute_unit)
                    # Load per-resolution metadata
                    for ext in (".json",):
                        meta_name = f"vision_meta_{res_h}x{res_w}{ext}"
                        meta_path = os.path.join(model_dir, meta_name)
                        if os.path.exists(meta_path):
                            with open(meta_path) as f:
                                self.vision_metas[(res_h, res_w)] = json.load(f)
                            break
                    if (res_h, res_w) not in self.vision_metas:
                        # Compute default meta
                        patch_size = 16
                        merge_size = 2
                        self.vision_metas[(res_h, res_w)] = {
                            "image_size": [res_h, res_w],
                            "num_merged_tokens": (res_h // (patch_size * merge_size))
                                                 * (res_w // (patch_size * merge_size)),
                            "image_token_id": 248056,
                            "vision_start_token_id": 248053,
                            "vision_end_token_id": 248054,
                            "temporal_patch_size": 2,
                            "patch_size": patch_size,
                            "spatial_merge_size": merge_size,
                        }
                    print(f"  {res_h}×{res_w}: {self.vision_metas[(res_h, res_w)]['num_merged_tokens']} tokens")

            self.available_resolutions = sorted(self.vision_models.keys())
            if self.available_resolutions:
                print(f"[engine] Multi-resolution vision: {len(self.available_resolutions)} models loaded")
                # Set default vision_meta from first available resolution
                if self.vision_meta is None:
                    default_res = (self.image_size, self.image_size)
                    if default_res in self.vision_metas:
                        self.vision_meta = self.vision_metas[default_res]
                    else:
                        first = self.available_resolutions[0]
                        self.vision_meta = self.vision_metas[first]

        # ── Load default/fallback vision encoder ──
        if self.vision_model_path and os.path.exists(self.vision_model_path):
            # Only load the default model if it wasn't already loaded as a resolution variant
            default_res = (self.image_size, self.image_size)
            if default_res not in self.vision_models:
                print(f"[engine] Loading vision encoder (default {self.image_size}×{self.image_size})...")
                self.vision_model = _load_model(self.vision_model_path, self.compute_unit)
                print(f"  Vision encoder loaded ({self.image_size}×{self.image_size})")
            else:
                self.vision_model = self.vision_models[default_res]

            # Load vision metadata
            meta_path = os.path.join(
                os.path.dirname(self.vision_model_path), "vision_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self.vision_meta = json.load(f)
                print(f"  Vision meta: {self.vision_meta['num_merged_tokens']} "
                      f"visual tokens, image_token_id={self.vision_meta['image_token_id']}")
            else:
                # Default metadata
                self.vision_meta = {
                    "image_size": [self.image_size, self.image_size],
                    "num_merged_tokens": (self.image_size // 16) ** 2 // 4,
                    "image_token_id": 248056,
                    "vision_start_token_id": 248053,
                    "vision_end_token_id": 248054,
                    "temporal_patch_size": 2,
                }
                print(f"  Using default vision meta")
        else:
            if not self.available_resolutions:
                print(f"[engine] No vision encoder found, image input disabled")

    def set_pending_image(self, image):
        """Set a PIL Image to be processed on the next chat turn."""
        self.pending_image = image

    def _select_vision_model(self, image):
        """Select the best vision encoder for the given image.

        Returns (model, meta, (target_h, target_w)) tuple.
        Uses aspect-ratio matching when multi-resolution models are available,
        falls back to default square model otherwise.
        """
        if self.available_resolutions:
            img_w, img_h = image.size  # PIL size is (W, H)
            best_res = find_best_resolution(img_h, img_w, self.available_resolutions)
            model = self.vision_models[best_res]
            meta = self.vision_metas[best_res]
            print(f"[vision] Image {img_w}×{img_h} → best resolution {best_res[0]}×{best_res[1]} "
                  f"({meta['num_merged_tokens']} tokens)")
            return model, meta, best_res

        # Fallback: default square model
        if self.vision_model is not None:
            return self.vision_model, self.vision_meta, (self.image_size, self.image_size)
        return None, None, None

    def _encode_image(self, image):
        """Run image through the best-matching vision encoder, return visual embeddings."""
        model, meta, target_size = self._select_vision_model(image)
        if model is None:
            return None, None

        t0 = time.time()
        T = meta.get("temporal_patch_size", 2)
        pixel_values = preprocess_image(image, target_size, T)
        out = model.predict({"pixel_values": pixel_values})
        visual_embeds = list(out.values())[0]  # [1, num_tokens, hidden_dim]
        elapsed = time.time() - t0
        print(f"[vision] Encoded image: {visual_embeds.shape} in {elapsed*1000:.0f}ms")
        return visual_embeds.astype(np.float16), meta

    def _build_vision_prompt_tokens(self, user_msg, enable_thinking=True,
                                     num_vis_tokens=None, active_meta=None):
        """Build prompt tokens with <image> placeholder tokens.

        Returns token IDs where image_token_id appears num_vis_tokens times.
        If num_vis_tokens is None, uses active_meta or self.vision_meta default.
        """
        meta = active_meta or self.vision_meta
        if num_vis_tokens is None:
            num_vis_tokens = meta["num_merged_tokens"]
        image_token_id = meta["image_token_id"]

        # Build message with image placeholder
        # Qwen VL format: <|vision_start|><|image_pad|>×N<|vision_end|>text
        vision_start_id = meta.get("vision_start_token_id", 248053)
        vision_end_id = meta.get("vision_end_token_id", 248054)

        # Tokenize the text part normally
        messages_with_image = list(self.messages)
        # The last message should be the user message with image reference
        # We'll construct tokens manually with vision placeholders

        # First tokenize without image
        prompt_tokens = self._tokenize_messages(self.messages, enable_thinking=enable_thinking)

        # Find where to insert image tokens (after the user content start)
        # Insert: <vision_start> + image_token×N + <vision_end> at the beginning
        # of the user's message in the token stream
        image_tokens = [vision_start_id] + [image_token_id] * num_vis_tokens + [vision_end_id]

        # Insert image tokens right after the last <|im_start|>user\n
        # Find the position by looking for the user role tokens
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        user_token_ids = self.tokenizer.encode("user\n", add_special_tokens=False)

        insert_pos = None
        for i in range(len(prompt_tokens) - len(user_token_ids), -1, -1):
            if (i > 0 and prompt_tokens[i-1] == im_start_id and
                prompt_tokens[i:i+len(user_token_ids)] == user_token_ids):
                insert_pos = i + len(user_token_ids)
                break

        if insert_pos is not None:
            prompt_tokens = (prompt_tokens[:insert_pos] +
                           image_tokens +
                           prompt_tokens[insert_pos:])
        else:
            # Fallback: insert at the start
            prompt_tokens = image_tokens + prompt_tokens

        return prompt_tokens

    def _step_with_vision(self, prompt_tokens, visual_embeds, active_meta=None):
        """Process prompt tokens, replacing image_token positions with visual embeddings.

        This overrides the normal prefill to inject visual embeddings.
        Uses MRoPE (3D position_ids) for proper spatial positioning of image tokens.

        MRoPE position mapping:
        - Text tokens: t=h=w = sequential position (collapses to 1D RoPE)
        - Image tokens: t=constant, h=row_in_grid, w=col_in_grid
        - After image section, text positions resume from img_start + max(grid_h, grid_w)
          instead of img_start + n_image_tokens, resulting in a rope_delta of
          max(grid_h, grid_w) - n_image_tokens.
        """
        if active_meta is None:
            active_meta = self.vision_meta
        image_token_id = active_meta.get("image_token_id",
                                          self.vision_meta["image_token_id"])
        n_total = len(prompt_tokens)

        # Find image token positions
        image_positions = [i for i, t in enumerate(prompt_tokens) if t == image_token_id]

        if not image_positions:
            # No image tokens, normal prefill
            return self._process_prompt(prompt_tokens)

        print(f"[vision] Found {len(image_positions)} image tokens in prompt "
              f"(positions {image_positions[0]}..{image_positions[-1]})")

        # Compute image grid dimensions from resolution metadata
        n_image = len(image_positions)
        img_size = active_meta.get("image_size", [self.image_size, self.image_size])
        if isinstance(img_size, (list, tuple)):
            res_h, res_w = img_size
        else:
            res_h = res_w = img_size
        patch_size = active_meta.get("patch_size", 16)
        merge_size = active_meta.get("spatial_merge_size", 2)
        grid_h = res_h // (patch_size * merge_size)
        grid_w = res_w // (patch_size * merge_size)
        expected_tokens = grid_h * grid_w
        if expected_tokens != n_image:
            print(f"[vision] WARNING: grid {grid_h}×{grid_w}={expected_tokens} "
                  f"doesn't match {n_image} image tokens, falling back to sqrt")
            grid_size = int(np.sqrt(n_image))
            if grid_size * grid_size != n_image:
                grid_h = grid_size
                grid_w = (n_image + grid_h - 1) // grid_h
            else:
                grid_h = grid_w = grid_size

        # Pre-compute 3D MRoPE positions for ALL tokens in the prompt
        # Shape: (3, n_total) — [temporal, height, width]
        mrope_positions = np.zeros((3, n_total), dtype=np.int32)
        logical_pos = 0  # tracks the logical RoPE position
        vis_idx = 0
        img_start_logical = None

        for ti, tok_id in enumerate(prompt_tokens):
            if tok_id == image_token_id and vis_idx < n_image:
                if vis_idx == 0:
                    img_start_logical = logical_pos
                row = vis_idx // grid_w
                col = vis_idx % grid_w
                mrope_positions[0, ti] = img_start_logical + self.rope_offset  # temporal: constant
                mrope_positions[1, ti] = img_start_logical + self.rope_offset + row  # height
                mrope_positions[2, ti] = img_start_logical + self.rope_offset + col  # width
                vis_idx += 1
                if vis_idx == n_image:
                    # After image section, text resumes from img_start + max(grid)
                    logical_pos = img_start_logical + max(grid_h, grid_w)
                else:
                    # Don't advance logical_pos during image section
                    pass
                # Always increment for physical position tracking
            else:
                # Text token: all 3 dims = same sequential position
                mrope_positions[:, ti] = logical_pos + self.rope_offset
                logical_pos += 1

        # Compute final rope_delta for post-prefill generation
        rope_delta = max(grid_h, grid_w) - n_image  # typically -182 for 14×14

        # Now process tokens sequentially
        vis_idx = 0
        last_next = None

        for ti, tok_id in enumerate(prompt_tokens):
            if self.pos >= self.ctx:
                print(f"[prefill] OVERFLOW at pos={self.pos}")
                return None

            is_last = (ti == n_total - 1)

            if tok_id == image_token_id and vis_idx < visual_embeds.shape[1]:
                # Image token: use visual embedding
                hidden = visual_embeds[:, vis_idx:vis_idx+1, :]
                vis_idx += 1

                mask = self._mask_buf
                mask[:, :, :, :] = -65504.0
                mask[:, :, :, :self.pos + 1] = 0

                pos_arr = self._pos_buf
                pos_arr[0] = self.pos

                rope_arr = self._rope_buf
                rope_arr[0] = mrope_positions[0, ti]
                rope_arr[1] = mrope_positions[1, ti]
                rope_arr[2] = mrope_positions[2, ti]

                for ci in range(self.num_chunks):
                    inp = {
                        "hidden_states": hidden.astype(np.float16),
                        "position_ids": rope_arr,
                        "causal_mask": mask,
                        "current_pos": pos_arr,
                        "linear_conv_state": self.lin_convs[ci],
                        "linear_recurrent_state": self.lin_recs[ci],
                    }
                    out = self.ffns[ci].predict(inp, state=self.states[ci])
                    hidden = out["output_hidden_states"]
                    if 'linear_conv_state_out' in out:
                        self.lin_convs[ci] = out['linear_conv_state_out']
                        self.lin_recs[ci] = out['linear_recurrent_state_out']

                if is_last:
                    lm_out = self.lmhead.predict(
                        {"hidden_states": hidden.astype(np.float16)})
                    if self.lmhead_mode == "logits":
                        logits = self._extract_logits(lm_out)
                        last_next = int(np.argmax(logits))
                    else:
                        last_next = int(lm_out["argmax_idx"].flatten()[0])

                self.pos += 1
            else:
                # Text token — temporarily adjust rope_offset so _step/_step_kv_only
                # uses the pre-computed MRoPE position
                saved_offset = self.rope_offset
                # The base methods compute: rope_val = pos + rope_offset
                # We want: rope_val = mrope_positions[:, ti]
                # Since all 3 dims are equal for text, we can use any dim
                target_rope = int(mrope_positions[0, ti])
                self.rope_offset = target_rope - self.pos

                if is_last:
                    last_next, _ = self._step(tok_id, self.pos)
                else:
                    self._step_kv_only(tok_id, self.pos)

                self.rope_offset = saved_offset
                self.pos += 1

        # Update rope_offset for generation tokens that follow the prefill.
        # Generation tokens use: rope_val = self.pos + self.rope_offset
        # The next logical position should be: logical_pos + self.rope_offset
        # So: rope_offset_new = logical_pos + original_rope_offset - self.pos
        self.rope_offset = self.rope_offset + rope_delta

        print(f"[vision] Prefill complete: {n_total} tokens "
              f"({len(image_positions)} visual), pos={self.pos}/{self.ctx}, "
              f"rope_delta={rope_delta}, rope_offset={self.rope_offset}")
        return last_next

    def chat_stream_with_image(self, user_msg, image=None, max_tokens=4096,
                                enable_thinking=True, repetition_guard=False,
                                temperature=0.7, top_p=0.9, top_k=20,
                                repetition_penalty=1.1, presence_penalty=0.0,
                                frequency_penalty=0.0):
        """Generator yielding SSE events for a streaming response with optional image.

        If image is provided, encodes it and injects visual embeddings.
        Otherwise falls back to normal text-only chat.
        """
        if image is None and self.pending_image is not None:
            image = self.pending_image
            self.pending_image = None

        if image is None or (self.vision_model is None and not self.available_resolutions):
            # No image or no vision model — use normal chat
            yield from self.chat_stream(
                user_msg, max_tokens, enable_thinking, repetition_guard,
                temperature, top_p, top_k,
                repetition_penalty, presence_penalty, frequency_penalty)
            return

        with self.lock:
            # Encode image (returns embeddings + per-resolution meta)
            result = self._encode_image(image)
            if result[0] is None:
                yield {"type": "error", "message": "Vision encoding failed"}
                return
            visual_embeds, active_meta = result

            is_first_turn = not any(
                m["role"] == "user" for m in self.messages)

            self.messages.append({"role": "user", "content": user_msg})

            # Build prompt with image tokens using the active resolution's token count
            num_vis_tokens = active_meta["num_merged_tokens"]
            prompt_tokens = self._build_vision_prompt_tokens(
                user_msg, enable_thinking=enable_thinking,
                num_vis_tokens=num_vis_tokens, active_meta=active_meta)

            turn_num = (len(self.messages) + 1) // 2
            print(f"\n{'='*60}")
            print(f"[chat+vision] Turn {turn_num}: image + "
                  f"\"{user_msg[:60]}{'...' if len(user_msg)>60 else ''}\"")
            print(f"[chat+vision] prompt={len(prompt_tokens)} tok "
                  f"(incl {visual_embeds.shape[1]} visual), "
                  f"cache={self.pos}/{self.ctx}")

            # Check overflow
            prompt_tokens, overflowed = self._handle_overflow(
                prompt_tokens, enable_thinking)
            if overflowed:
                print(f"[cache] After rebuild: prompt={len(prompt_tokens)} tok")

            remaining = self.ctx - self.pos - len(prompt_tokens)
            if remaining < 10:
                yield {"type": "error",
                       "message": f"Context full ({self.pos}/{self.ctx})."}
                return
            max_tokens = min(max_tokens, remaining)

            # PREFILL with vision
            t0 = time.time()
            last_next = self._step_with_vision(
                prompt_tokens, visual_embeds, active_meta=active_meta)
            if last_next is None:
                yield {"type": "error",
                       "message": "Context overflow during prefill."}
                return
            self.token_history.extend(prompt_tokens)

            # DECODE (same as normal chat)
            from inference_config import get_sampling_config
            rep_detector = RepetitionDetector() if repetition_guard else None
            generated_ids = [last_next]
            if rep_detector:
                rep_detector.add_token(last_next)
            text_so_far = self.tokenizer.decode(
                [last_next], skip_special_tokens=True).rstrip('\ufffd')
            if text_so_far:
                yield {"type": "token", "text": text_so_far, "id": last_next}

            has_penalties = (repetition_penalty > 1.0
                            or presence_penalty != 0.0
                            or frequency_penalty != 0.0
                            or (temperature > 0 and temperature != 1.0)
                            or (top_p > 0 and top_p < 1.0)
                            or (top_k is not None and top_k > 0))

            PENALTY_HISTORY_WINDOW = 256
            history_prefix = list(self.token_history)[-PENALTY_HISTORY_WINDOW:]

            stopped_by_rep = False
            for gi in range(max_tokens - 1):
                if self.pos >= self.ctx:
                    if not self.compact_cache():
                        break
                fed_tok = generated_ids[-1]
                next_id, logits = self._step(fed_tok, self.pos)
                self.pos += 1
                self.token_history.append(fed_tok)
                if logits is not None and has_penalties:
                    if not enable_thinking:
                        if self.think_token_id is not None:
                            logits[self.think_token_id] = -1e9
                        if self.endthink_token_id is not None:
                            logits[self.endthink_token_id] = -1e9
                    penalty_ids = history_prefix + generated_ids
                    next_id = self._apply_penalties(
                        logits, penalty_ids,
                        repetition_penalty, presence_penalty,
                        frequency_penalty, temperature, top_p, top_k)
                generated_ids.append(next_id)

                if rep_detector and rep_detector.add_token(next_id):
                    stopped_by_rep = True
                    break

                if next_id in self.stop_ids:
                    break

                new_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True)
                stable_text = new_text.rstrip('\ufffd')
                if len(stable_text) > len(text_so_far):
                    delta = stable_text[len(text_so_far):]
                    text_so_far = stable_text
                    yield {"type": "token", "text": delta, "id": next_id}

            elapsed = time.time() - t0
            decode_count = len(generated_ids)
            full_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True)
            self.messages.append({"role": "assistant", "content": full_text})

            stop_reason = ("repetition" if stopped_by_rep
                          else "eos" if (generated_ids and generated_ids[-1] in self.stop_ids)
                          else "length")
            print(f"[decode] {decode_count} tok in {elapsed:.1f}s, "
                  f"pos={self.pos}/{self.ctx}, stop={stop_reason}")

            yield {
                "type": "done",
                "decode_tokens": decode_count,
                "end_pos": self.pos,
                "elapsed": round(elapsed, 1),
                "stop_reason": stop_reason,
            }


# ── HTML page with image support ──

VISION_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen3.5 Vision Chat (ANE)</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e; color: #e0e0e0; height: 100vh;
    display: flex; flex-direction: column;
  }
  .header {
    background: #16213e; padding: 12px 20px; display: flex;
    align-items: center; justify-content: space-between;
    border-bottom: 1px solid #0f3460;
  }
  .header h1 { font-size: 16px; color: #e94560; }
  .header .info { font-size: 12px; color: #888; }
  .header button {
    background: #0f3460; color: #e0e0e0; border: 1px solid #e94560;
    padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px;
  }
  .header button:hover { background: #e94560; color: #fff; }
  #chat {
    flex: 1; overflow-y: auto; padding: 16px 20px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; }
  .msg.user {
    align-self: flex-end; background: #0f3460;
    border-bottom-right-radius: 4px;
  }
  .msg.assistant {
    align-self: flex-start; background: #16213e;
    border-bottom-left-radius: 4px; white-space: pre-wrap;
  }
  .msg.error { align-self: center; background: #8b0000; font-size: 13px; }
  .msg img { max-width: 300px; max-height: 200px; border-radius: 8px; margin-bottom: 8px; display: block; }
  .input-area {
    display: flex; gap: 8px; padding: 12px 20px;
    background: #16213e; border-top: 1px solid #0f3460;
    align-items: flex-end;
  }
  #image-preview {
    display: none; position: relative; margin-bottom: 4px;
  }
  #image-preview img {
    max-height: 80px; border-radius: 6px; border: 1px solid #0f3460;
  }
  #image-preview .remove {
    position: absolute; top: -6px; right: -6px;
    background: #e94560; color: #fff; border: none; border-radius: 50%;
    width: 20px; height: 20px; cursor: pointer; font-size: 12px; line-height: 20px;
    text-align: center;
  }
  #msg {
    flex: 1; padding: 10px 14px; border-radius: 10px;
    border: 1px solid #0f3460; background: #0d1b2a; color: #e0e0e0;
    font-size: 14px; resize: none; min-height: 42px; max-height: 120px;
  }
  #msg:focus { outline: none; border-color: #e94560; }
  .btn-group { display: flex; gap: 4px; }
  .input-area button {
    padding: 10px 16px; border-radius: 10px; border: none;
    cursor: pointer; font-size: 14px;
  }
  #send-btn { background: #e94560; color: #fff; }
  #send-btn:disabled { opacity: 0.5; }
  #img-btn { background: #0f3460; color: #e0e0e0; font-size: 18px; padding: 8px 12px; }
  #img-btn:hover { background: #1a4a8a; }
</style>
</head>
<body>
<div class="header">
  <h1>🔮 Qwen3.5 Vision Chat (ANE)</h1>
  <div class="info" id="status">Loading...</div>
  <button onclick="resetChat()">Reset</button>
</div>
<div id="chat"></div>
<div style="padding: 0 20px;">
  <div id="image-preview">
    <img id="preview-img" src="" />
    <button class="remove" onclick="clearImage()">×</button>
  </div>
</div>
<div class="input-area">
  <button id="img-btn" onclick="document.getElementById('file-input').click()" title="Attach image">📷</button>
  <input type="file" id="file-input" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
  <textarea id="msg" rows="1" placeholder="Type a message... (paste or attach an image)"
    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
  <button id="send-btn" onclick="send()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const msgInput = document.getElementById('msg');
const sendBtn = document.getElementById('send-btn');
const statusEl = document.getElementById('status');
let pendingImage = null;  // base64 data URL

function addMsg(role, content, imageDataUrl) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  if (imageDataUrl) {
    const img = document.createElement('img');
    img.src = imageDataUrl;
    d.appendChild(img);
  }
  const textNode = document.createElement('span');
  textNode.textContent = content;
  d.appendChild(textNode);
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(ev) {
    pendingImage = ev.target.result;
    document.getElementById('preview-img').src = pendingImage;
    document.getElementById('image-preview').style.display = 'block';
  };
  reader.readAsDataURL(file);
  e.target.value = '';
}

// Handle paste
document.addEventListener('paste', function(e) {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      e.preventDefault();
      const blob = item.getAsFile();
      const reader = new FileReader();
      reader.onload = function(ev) {
        pendingImage = ev.target.result;
        document.getElementById('preview-img').src = pendingImage;
        document.getElementById('image-preview').style.display = 'block';
      };
      reader.readAsDataURL(blob);
      break;
    }
  }
});

function clearImage() {
  pendingImage = null;
  document.getElementById('image-preview').style.display = 'none';
  document.getElementById('preview-img').src = '';
}

async function send() {
  const text = msgInput.value.trim();
  if (!text && !pendingImage) return;
  const msg = text || "What's in this image?";
  const imageToSend = pendingImage;

  addMsg('user', msg, imageToSend);
  msgInput.value = '';
  clearImage();
  sendBtn.disabled = true;

  const d = addMsg('assistant', '');
  const textSpan = d.querySelector('span');
  let full = '';

  try {
    const body = { message: msg };
    if (imageToSend) body.image = imageToSend;

    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') continue;
        try {
          const ev = JSON.parse(data);
          if (ev.type === 'token') {
            full += ev.text;
            textSpan.textContent = full;
            chat.scrollTop = chat.scrollHeight;
          } else if (ev.type === 'done') {
            statusEl.textContent = `${ev.decode_tokens} tok, ${ev.elapsed}s, pos=${ev.end_pos}`;
          } else if (ev.type === 'error') {
            textSpan.textContent = '⚠ ' + ev.message;
            d.className = 'msg error';
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    textSpan.textContent = '⚠ Connection error';
    d.className = 'msg error';
  }
  sendBtn.disabled = false;
  msgInput.focus();
}

async function resetChat() {
  await fetch('/api/reset', {method:'POST'});
  chat.innerHTML = '';
  statusEl.textContent = 'Reset OK';
}

statusEl.textContent = 'Ready (vision enabled)';
msgInput.focus();
</script>
</body>
</html>"""


# ── HTTP Handler ──

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

engine = None
_model_size = "4B"


class VisionChatHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access logs

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        body = VISION_HTML.encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/reset":
            engine.reset()
            self._send_json({"ok": True})

        elif path == "/api/chat/stream":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, 400)
                return

            message = data.get("message", "").strip()
            image_data = data.get("image")  # base64 data URL or None

            if not message and not image_data:
                self._send_json({"error": "Empty message"}, 400)
                return

            if not message:
                message = "What's in this image?"

            # Decode image if provided
            image = None
            if image_data:
                image = decode_image_from_data_url(image_data)
                if image:
                    print(f"[http] Received image: {image.size} {image.mode}")

            max_tokens = min(int(data.get("max_tokens", 4096)), 4096)
            enable_thinking = data.get("enable_thinking", True)
            repetition_guard = data.get("repetition_guard", False)

            from inference_config import get_sampling_config
            _sc = get_sampling_config(_model_size, enable_thinking)
            temperature = float(data.get("temperature", _sc["temperature"]))
            top_p = float(data.get("top_p", _sc["top_p"]))
            top_k_val = data.get("top_k", _sc["top_k"])
            top_k = int(top_k_val) if top_k_val is not None else _sc["top_k"]
            repetition_penalty = float(data.get("repetition_penalty", _sc["repetition_penalty"]))
            presence_penalty = float(data.get("presence_penalty", _sc.get("presence_penalty", 0.0)))
            frequency_penalty = float(data.get("frequency_penalty", _sc.get("frequency_penalty", 0.0)))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                for event in engine.chat_stream_with_image(
                        message, image=image,
                        max_tokens=max_tokens,
                        enable_thinking=enable_thinking,
                        repetition_guard=repetition_guard,
                        temperature=temperature, top_p=top_p, top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty):
                    line = f"data: {json.dumps(event)}\n\n"
                    self.wfile.write(line.encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


# ── Main ──

def main():
    global engine, _model_size

    from config import DEFAULT_OUTPUT, DEFAULT_HF_MODEL

    parser = argparse.ArgumentParser(description="Qwen3.5 Vision Chat Server (ANE)")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-size", default="4B", choices=["4B", "2B"])
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--embed-lmhead", default=None)
    parser.add_argument("--ffn-dir", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--vision-model", default=None,
                       help="Path to vision_encoder.mlpackage (auto-detected in model-dir)")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--compute-unit", default="all",
                       choices=["all", "cpu", "cpu_and_gpu", "cpu_and_ne"])
    args = parser.parse_args()

    _model_size = args.model_size

    if args.tokenizer is None:
        args.tokenizer = args.model_dir

    # Auto-detect vision model
    vision_path = args.vision_model
    if vision_path is None:
        for ext in (".mlmodelc", ".mlpackage"):
            candidate = os.path.join(args.model_dir, f"vision_encoder{ext}")
            if os.path.exists(candidate):
                vision_path = candidate
                break

    # Auto-detect num_chunks
    num_chunks = args.num_chunks
    if num_chunks is None:
        ffn_base = args.ffn_dir or _find_combined_dir(args.model_dir)
        if ffn_base and os.path.isdir(ffn_base):
            num_chunks = sum(
                1 for f in os.listdir(ffn_base)
                if f.startswith("chunk") and (f.endswith(".mlpackage") or f.endswith(".mlmodelc"))
            )
            if num_chunks > 6:
                num_chunks //= 2
        if not num_chunks:
            num_chunks = 4
        print(f"  Auto-detected {num_chunks} FFN chunks")

    _cu_map = {
        "all": ct.ComputeUnit.CPU_AND_NE,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }
    compute_unit = _cu_map[args.compute_unit]

    # Auto-detect batch size
    global BATCH_SIZE, BLOCK_SIZE
    for _ep_name in ("embed_prefill.mlpackage", "embed_prefill.mlmodelc"):
        _ep_path = os.path.join(args.model_dir, _ep_name)
        if os.path.exists(_ep_path):
            try:
                _ep_spec = ct.utils.load_spec(_ep_path)
                for inp in _ep_spec.description.input:
                    if inp.name == "input_ids":
                        BATCH_SIZE = inp.type.multiArrayType.shape[1]
                        BLOCK_SIZE = BATCH_SIZE
                        print(f"  Batch size (auto): {BATCH_SIZE}")
                        break
            except Exception:
                pass
            break

    engine = VisionChatEngine(
        args.model_dir, args.tokenizer,
        ctx=args.ctx, num_chunks=num_chunks,
        embed_lmhead_path=args.embed_lmhead,
        ffn_dir=args.ffn_dir,
        system_prompt=args.system_prompt,
        compute_unit=compute_unit,
        vision_model_path=vision_path,
        image_size=args.image_size,
    )

    print(f"\n  Model dir: {args.model_dir}")
    print(f"  Vision model: {vision_path or 'none'}")
    print(f"  CTX: {args.ctx}\n")
    engine.load()

    server = HTTPServer(("0.0.0.0", args.port), VisionChatHandler)
    print(f"\n  Vision Chat server ready on http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
