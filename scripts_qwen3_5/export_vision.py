#!/usr/bin/env python3
"""Qwen3.5 — Export vision encoder to CoreML for ANE.

Exports the Qwen3.5 vision encoder (ViT + PatchMerger) as a single CoreML
model that takes preprocessed pixel patches and outputs visual embeddings
compatible with the text decoder's hidden_states.

The vision encoder processes a fixed-size image (default 448×448) through:
  1. 3D Conv patch embedding  (3×2×16×16 → 1024-d tokens)
  2. Learned position embeddings + 2D rotary pos encoding
  3. 24 vision transformer blocks (LN + SDPA + LN + MLP)
  4. Spatial merge (2×2 → concat → MLP → 2560-d)

Output shape: [1, num_visual_tokens, 2560] where
  num_visual_tokens = (H/16) * (W/16) / (merge_size^2) = 28*28/4 = 196
  for a 448×448 image.

Usage:
    python scripts_qwen3_5/export_vision.py --model /path/to/Qwen3.5-4B --output /path/to/output
    python scripts_qwen3_5/export_vision.py --image-size 448
"""
import gc, time, argparse, os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from config import DEFAULT_HF_MODEL, DEFAULT_OUTPUT

# ── Vision config defaults (from Qwen3.5-4B config.json) ──
VISION_HIDDEN_SIZE = 1024
VISION_NUM_HEADS = 16
VISION_DEPTH = 24
VISION_INTERMEDIATE_SIZE = 4096
VISION_PATCH_SIZE = 16
VISION_TEMPORAL_PATCH_SIZE = 2
VISION_IN_CHANNELS = 3
VISION_SPATIAL_MERGE_SIZE = 2
VISION_NUM_POS_EMBEDDINGS = 2304
VISION_OUT_HIDDEN_SIZE = 2560  # must match text_config.hidden_size

# Default image size (must be divisible by patch_size)
DEFAULT_IMAGE_SIZE = 448

# ── Supported multi-resolution sizes (H×W) ──
# All dims must be divisible by patch_size * merge_size = 16 * 2 = 32.
# Each resolution produces (H/32)*(W/32) merged visual tokens.
SUPPORTED_RESOLUTIONS = [
    (448, 448),   # 196 tokens — square (default)
    (448, 672),   # 294 tokens — 2:3 landscape
    (672, 448),   # 294 tokens — 3:2 portrait
    (448, 896),   # 392 tokens — 1:2 wide landscape
    (896, 448),   # 392 tokens — 2:1 tall portrait (phone screenshots)
]


def smart_resize(height, width, factor=32, min_pixels=448*448, max_pixels=896*448):
    """Resize dimensions preserving aspect ratio, snapped to multiples of factor.

    Matches HuggingFace Qwen2VLImageProcessor.smart_resize logic:
    1. Round each dim to nearest multiple of factor.
    2. If total pixels > max_pixels, scale down.
    3. If total pixels < min_pixels, scale up.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Aspect ratio too extreme: {height}x{width}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def find_best_resolution(img_h, img_w, supported_resolutions=None):
    """Pick the supported resolution with the closest aspect ratio."""
    if supported_resolutions is None:
        supported_resolutions = SUPPORTED_RESOLUTIONS
    img_aspect = img_w / img_h
    best = supported_resolutions[0]
    best_diff = float('inf')
    for h, w in supported_resolutions:
        res_aspect = w / h
        diff = abs(img_aspect - res_aspect)
        # Tie-break: prefer more pixels
        if diff < best_diff or (diff == best_diff and h * w > best[0] * best[1]):
            best_diff = diff
            best = (h, w)
    return best


MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"


# ── Vision Encoder Components ──


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Apply 2D rotary embeddings to query and key."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.float().unsqueeze(-2)  # [seq, 1, head_dim]
    sin = sin.float().unsqueeze(-2)  # [seq, 1, head_dim]

    def rotate_half(x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out.to(orig_q_dtype), k_out.to(orig_k_dtype)


class VisionAttention(nn.Module):
    """Multi-head self-attention for vision blocks."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.scaling = self.head_dim ** -0.5

    def forward(self, hidden_states, cos, sin):
        seq_len = hidden_states.shape[0]
        # Fused QKV
        qkv = self.qkv(hidden_states)  # [seq, 3*hidden]
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)  # each [seq, heads, head_dim]

        # Apply rotary embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Standard attention: [1, heads, seq, head_dim]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(seq_len, self.hidden_size)
        return self.proj(attn_out)


class VisionMLP(nn.Module):
    """MLP with GELU activation for vision blocks."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act = nn.GELU(approximate='tanh')
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    """Single vision transformer block."""
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PatchMerger(nn.Module):
    """Merge spatial_merge_size×spatial_merge_size patches into one token."""
    def __init__(self, hidden_size, out_hidden_size, spatial_merge_size):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        merged_dim = hidden_size * (spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(merged_dim, merged_dim, bias=True)
        self.act = nn.GELU(approximate='tanh')
        self.linear_fc2 = nn.Linear(merged_dim, out_hidden_size, bias=True)

    def forward(self, x):
        # x: [num_patches, hidden_size] already spatially reordered
        x = self.norm(x)
        x = x.view(-1, self.spatial_merge_size ** 2 * x.shape[-1])
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen35VisionEncoder(nn.Module):
    """Complete Qwen3.5 vision encoder for ANE export.

    Takes pixel_values [1, seq_patches, C*T*Hp*Wp] and produces
    visual embeddings [1, num_merged_tokens, out_hidden_size].

    For ANE tracing, we fix the image size and pre-compute all
    position-dependent values statically.
    """
    def __init__(self, config, image_size=448):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.depth = config['depth']
        self.patch_size = config['patch_size']
        self.temporal_patch_size = config['temporal_patch_size']
        self.in_channels = config['in_channels']
        self.spatial_merge_size = config['spatial_merge_size']
        self.out_hidden_size = config['out_hidden_size']
        self.num_position_embeddings = config['num_position_embeddings']
        # Accept int (square) or (H, W) tuple
        if isinstance(image_size, (tuple, list)):
            self.image_h, self.image_w = image_size
        else:
            self.image_h = self.image_w = image_size

        # Patch embedding: Conv3d(3, 1024, kernel=(2,16,16), stride=(2,16,16))
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.patch_embed_proj = nn.Conv3d(
            self.in_channels, self.hidden_size,
            kernel_size=kernel, stride=kernel, bias=True
        )

        # Position embeddings (learned) - will be interpolated to fixed pos_emb buffer
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        # Rotary embeddings - pre-computed for fixed image size (no dynamic ops)
        head_dim = self.hidden_size // self.num_heads

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionBlock(self.hidden_size, self.num_heads,
                       config['intermediate_size'])
            for _ in range(self.depth)
        ])

        # Patch merger
        self.merger = PatchMerger(
            self.hidden_size, self.out_hidden_size, self.spatial_merge_size
        )

        # Pre-compute grid dimensions for fixed image size
        self.grid_h = self.image_h // self.patch_size
        self.grid_w = self.image_w // self.patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.merged_h = self.grid_h // self.spatial_merge_size
        self.merged_w = self.grid_w // self.spatial_merge_size
        self.num_merged = self.merged_h * self.merged_w

        # Pre-compute rotary cos/sin as fixed buffers (avoids torch.outer at trace time)
        rope_cos, rope_sin = self._precompute_rope(head_dim)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # Pre-compute position embedding interpolation indices and weights
        self._precompute_pos_interp()

    def _precompute_rope(self, head_dim):
        """Pre-compute rotary cos/sin for fixed grid size."""
        merge = self.spatial_merge_size
        merged_h = self.grid_h // merge
        merged_w = self.grid_w // merge

        # Build rotary frequency table
        rotary_dim = head_dim // 2  # 32
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        max_grid = max(self.grid_h, self.grid_w)
        seq = torch.arange(max_grid, dtype=torch.float32)
        freq_table = torch.outer(seq, inv_freq)  # [max_grid, 16]

        # Build position IDs in merger spatial grouping order
        block_rows = torch.arange(merged_h)
        block_cols = torch.arange(merged_w)
        intra_row = torch.arange(merge)
        intra_col = torch.arange(merge)

        row_idx = block_rows[:, None, None, None] * merge + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge + intra_col[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, merge, merge).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge, merge).reshape(-1)

        rope_pos_ids = torch.stack([row_idx, col_idx], dim=-1)  # [num_patches, 2]
        rope_emb = freq_table[rope_pos_ids]  # [num_patches, 2, 16]
        rope_emb = rope_emb.reshape(rope_emb.shape[0], -1)  # [num_patches, 32]
        full_emb = torch.cat([rope_emb, rope_emb], dim=-1)  # [num_patches, 64=head_dim]

        return full_emb.cos(), full_emb.sin()  # each [num_patches, head_dim]

    def _precompute_pos_interp(self):
        """Pre-compute bilinear interpolation weights for position embeddings."""
        h, w = self.grid_h, self.grid_w
        merge = self.spatial_merge_size
        merged_h, merged_w = h // merge, w // merge
        num_grid_per_side = int(self.num_position_embeddings ** 0.5)  # 48

        h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

        h_floor = h_idxs.long()
        w_floor = w_idxs.long()
        h_ceil = (h_floor + 1).clamp(max=num_grid_per_side - 1)
        w_ceil = (w_floor + 1).clamp(max=num_grid_per_side - 1)
        dh = h_idxs - h_floor.float()
        dw = w_idxs - w_floor.float()

        h_floor_2d = h_floor.unsqueeze(1).expand(h, w)
        w_floor_2d = w_floor.unsqueeze(0).expand(h, w)
        h_ceil_2d = h_ceil.unsqueeze(1).expand(h, w)
        w_ceil_2d = w_ceil.unsqueeze(0).expand(h, w)
        dh_2d = dh.unsqueeze(1).expand(h, w)
        dw_2d = dw.unsqueeze(0).expand(h, w)

        self.register_buffer("pos_idx_tl",
            (h_floor_2d * num_grid_per_side + w_floor_2d).reshape(-1), persistent=False)
        self.register_buffer("pos_idx_tr",
            (h_floor_2d * num_grid_per_side + w_ceil_2d).reshape(-1), persistent=False)
        self.register_buffer("pos_idx_bl",
            (h_ceil_2d * num_grid_per_side + w_floor_2d).reshape(-1), persistent=False)
        self.register_buffer("pos_idx_br",
            (h_ceil_2d * num_grid_per_side + w_ceil_2d).reshape(-1), persistent=False)
        self.register_buffer("pos_w_tl",
            ((1 - dh_2d) * (1 - dw_2d)).reshape(-1, 1), persistent=False)
        self.register_buffer("pos_w_tr",
            ((1 - dh_2d) * dw_2d).reshape(-1, 1), persistent=False)
        self.register_buffer("pos_w_bl",
            (dh_2d * (1 - dw_2d)).reshape(-1, 1), persistent=False)
        self.register_buffer("pos_w_br",
            (dh_2d * dw_2d).reshape(-1, 1), persistent=False)

    def finalize_pos_embed(self):
        """Compute the grouped position embedding buffer from loaded weights.
        Must be called AFTER load_state_dict populates pos_embed.weight.
        """
        pe = self.pos_embed.weight.detach()  # [2304, 1024]
        pos_emb = (pe[self.pos_idx_tl] * self.pos_w_tl +
                   pe[self.pos_idx_tr] * self.pos_w_tr +
                   pe[self.pos_idx_bl] * self.pos_w_bl +
                   pe[self.pos_idx_br] * self.pos_w_br)  # [784, 1024] raster

        # Reorder to merger-grouped order (matching HF fast_pos_embed_interpolate)
        merge = self.spatial_merge_size
        pos_2d = pos_emb.view(self.grid_h, self.grid_w, self.hidden_size)
        pos_grouped = pos_2d.view(
            self.merged_h, merge, self.merged_w, merge, self.hidden_size
        ).permute(0, 2, 1, 3, 4).reshape(-1, self.hidden_size)

        self.register_buffer("pos_embed_grouped", pos_grouped, persistent=False)

    def _raster_to_merger_group(self, x):
        """Reorder patches from raster order to merger-group order.

        Raster order:  (0,0),(0,1),(0,2),(0,3),...,(1,0),(1,1),...
        Merger-group:  (0,0),(0,1),(1,0),(1,1),(0,2),(0,3),(1,2),(1,3),...

        Each group of spatial_merge_size^2 consecutive patches in the output
        forms one 2×2 spatial block — matching the PatchMerger's view(-1, 4*C).

        This matches HF's image processor which delivers patches in merger-group
        order, and HF's window_index which maintains this grouping.
        """
        merge = self.spatial_merge_size
        # Reshape to 2D grid, group by merge blocks, then flatten
        x_2d = x.view(self.grid_h, self.grid_w, -1)
        x_grouped = x_2d.view(
            self.merged_h, merge, self.merged_w, merge, -1
        ).permute(0, 2, 1, 3, 4).reshape(-1, x.shape[-1])
        return x_grouped

    def forward(self, pixel_values):
        """Forward pass with fixed image size.

        Args:
            pixel_values: [1, C, T, H, W] = [1, 3, 2, 448, 448]
                         (2 identical frames for temporal_patch_size=2)

        Returns:
            visual_embeddings: [1, num_merged_tokens, out_hidden_size]
                             = [1, 196, 2560] for 448×448 image
        """
        # ── 1. Patch embedding ──
        # Conv3d: [1, 3, 2, 448, 448] → [1, 1024, 1, 28, 28]
        hidden = self.patch_embed_proj(pixel_values)
        # Reshape: [1, 1024, 1, 28, 28] → [784, 1024] (num_patches = 28*28)
        hidden = hidden.squeeze(0).squeeze(1)  # [1024, 28, 28]
        hidden = hidden.permute(1, 2, 0).reshape(-1, self.hidden_size)  # [784, 1024]

        # ── 1b. Reorder from raster to merger-group order ──
        # HF delivers patches in merger-group order from the image processor.
        # Our Conv3d produces raster order. We must reorder so that:
        #   - pos_embed_grouped (merger-group order) aligns with content
        #   - rope_cos/sin (merger-group order) aligns with content
        #   - PatchMerger's view(-1, 4*C) groups correct 2×2 spatial blocks
        hidden = self._raster_to_merger_group(hidden)  # [784, 1024] merger-group

        # ── 2. Position embeddings (pre-computed, in merger-grouped order) ──
        hidden = hidden + self.pos_embed_grouped  # [784, 1024]

        # ── 3. Vision transformer blocks (cos/sin are pre-computed buffers) ──
        cos = self.rope_cos  # [num_patches, head_dim]
        sin = self.rope_sin

        for block in self.blocks:
            hidden = block(hidden, cos, sin)

        # ── 4. Spatial merge ──
        # After merger, tokens are in merged-grid raster order (row-major of
        # the merged_h × merged_w grid), matching mRoPE expectations.
        visual_embeds = self.merger(hidden)  # [num_merged, out_hidden_size]

        return visual_embeds.unsqueeze(0)  # [1, num_merged, out_hidden_size]


def load_vision_weights(encoder, model_path):
    """Load vision encoder weights from HF safetensors checkpoint."""
    import glob
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    state_dict = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for key in st.keys():
                if key.startswith("model.visual."):
                    # Strip "model.visual." prefix
                    local_key = key[len("model.visual."):]
                    state_dict[local_key] = st.get_tensor(key)

    # Map HF keys to our module keys
    mapped = {}
    for k, v in state_dict.items():
        if k == "patch_embed.proj.weight":
            mapped["patch_embed_proj.weight"] = v
        elif k == "patch_embed.proj.bias":
            mapped["patch_embed_proj.bias"] = v
        elif k == "pos_embed.weight":
            mapped["pos_embed.weight"] = v
        elif k.startswith("blocks."):
            mapped[k] = v
        elif k.startswith("merger."):
            mapped[k] = v
        else:
            print(f"  [warn] Skipping unknown vision key: {k}")

    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    if missing:
        print(f"  [warn] Missing keys: {missing}")
    if unexpected:
        print(f"  [warn] Unexpected keys: {unexpected}")
    print(f"  Loaded {len(mapped)} vision weights")


def export_vision_encoder(model_path, output_dir, image_size=448, skip_existing=False):
    """Export the Qwen3.5 vision encoder to CoreML.

    Args:
        image_size: int for square, or (H, W) tuple for rectangular.
    """
    import json

    # Parse image_size: int → (H, W)
    if isinstance(image_size, (tuple, list)):
        img_h, img_w = image_size
    else:
        img_h = img_w = image_size

    # Output filename: vision_encoder.mlpackage for square 448,
    # vision_encoder_HxW.mlpackage otherwise
    if img_h == img_w == DEFAULT_IMAGE_SIZE:
        out_name = "vision_encoder.mlpackage"
    else:
        out_name = f"vision_encoder_{img_h}x{img_w}.mlpackage"
    out_path = os.path.join(output_dir, out_name)
    if skip_existing and os.path.exists(out_path):
        print(f"  [skip] {out_name}")
        return

    # Load vision config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        full_config = json.load(f)
    vision_config = full_config["vision_config"]

    print(f"  Vision config: depth={vision_config['depth']}, "
          f"hidden={vision_config['hidden_size']}, "
          f"heads={vision_config['num_heads']}, "
          f"out={vision_config['out_hidden_size']}")
    print(f"  Image size: {img_h}×{img_w}")

    grid_h = img_h // vision_config['patch_size']
    grid_w = img_w // vision_config['patch_size']
    num_merged = (grid_h // vision_config['spatial_merge_size']) * \
                 (grid_w // vision_config['spatial_merge_size'])
    print(f"  Grid: {grid_h}×{grid_w} patches → {num_merged} merged tokens")

    # Build model
    print("  Building vision encoder...")
    encoder = Qwen35VisionEncoder(vision_config, image_size=(img_h, img_w))

    # Load weights
    print("  Loading weights...")
    load_vision_weights(encoder, model_path)
    # Finalize position embeddings now that weights are loaded
    encoder.finalize_pos_embed()
    # Trace in float32 — CoreML compute_precision handles fp16 conversion.
    # Conv3d has known issues with mixed fp16/fp32 in JIT trace.
    encoder = encoder.eval().float()

    # Trace
    print("  Tracing...")
    T = vision_config['temporal_patch_size']
    sample = torch.zeros(1, 3, T, img_h, img_w, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(encoder, sample)

    # Verify output
    with torch.no_grad():
        test_out = traced(sample)
    print(f"  Traced output shape: {test_out.shape}")
    assert test_out.shape == (1, num_merged, vision_config['out_hidden_size']), \
        f"Expected (1, {num_merged}, {vision_config['out_hidden_size']}), got {test_out.shape}"

    # Convert to CoreML
    print("  Converting to CoreML...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=(1, 3, T, img_h, img_w),
                dtype=np.float16,
            ),
        ],
        outputs=[
            ct.TensorType(name="visual_embeddings", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    mlmodel.save(out_path)
    print(f"  Saved {out_name} ({time.time()-t0:.1f}s) → {out_path}")

    # Also save vision metadata (per-resolution)
    meta = {
        "image_size": [img_h, img_w],
        "patch_size": vision_config['patch_size'],
        "temporal_patch_size": T,
        "spatial_merge_size": vision_config['spatial_merge_size'],
        "num_merged_tokens": num_merged,
        "out_hidden_size": vision_config['out_hidden_size'],
        "image_token_id": full_config.get('image_token_id', 248056),
        "vision_start_token_id": full_config.get('vision_start_token_id', 248053),
        "vision_end_token_id": full_config.get('vision_end_token_id', 248054),
    }
    # Save per-resolution meta file
    if img_h == img_w == DEFAULT_IMAGE_SIZE:
        meta_name = "vision_meta.json"
    else:
        meta_name = f"vision_meta_{img_h}x{img_w}.json"
    meta_path = os.path.join(output_dir, meta_name)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_name}")

    del mlmodel, traced, encoder
    gc.collect()
    return meta


def export_vision_encoder_multi(model_path, output_dir, resolutions=None,
                                 skip_existing=False):
    """Export vision encoder for multiple resolutions.

    Creates one vision_encoder_HxW.mlpackage per resolution plus a
    vision_resolutions.json index that the chat server uses at runtime.
    """
    import json

    if resolutions is None:
        resolutions = SUPPORTED_RESOLUTIONS

    all_meta = {}
    for h, w in resolutions:
        print(f"\n--- Exporting vision encoder {h}×{w} ---")
        meta = export_vision_encoder(
            model_path, output_dir,
            image_size=(h, w),
            skip_existing=skip_existing,
        )
        if meta is not None:
            all_meta[f"{h}x{w}"] = meta

    # Save combined resolution index
    index = {
        "supported_resolutions": [[h, w] for h, w in resolutions],
        "default_resolution": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
        "resolutions": all_meta,
    }
    index_path = os.path.join(output_dir, "vision_resolutions.json")
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved vision_resolutions.json ({len(resolutions)} resolutions)")
    return index


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5 vision encoder to CoreML")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL,
                       help="Path to HF model directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                       help="Output directory")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                       help="Image size for single square export (default: 448)")
    parser.add_argument("--resolutions", default=None,
                       help="Comma-separated HxW resolutions for multi-res export, "
                            "e.g. '448x448,448x672,672x448,448x896,896x448'. "
                            "Use 'all' for all supported resolutions.")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip if already exported")
    args = parser.parse_args()

    print(f"=== Qwen3.5 Vision Encoder Export ===")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print()

    if args.resolutions:
        if args.resolutions.lower() == "all":
            resolutions = SUPPORTED_RESOLUTIONS
        else:
            resolutions = []
            for s in args.resolutions.split(","):
                h, w = s.strip().split("x")
                resolutions.append((int(h), int(w)))
        index = export_vision_encoder_multi(
            args.model, args.output,
            resolutions=resolutions,
            skip_existing=args.skip_existing,
        )
        print(f"\nDone! Exported {len(resolutions)} vision encoder variants.")
    else:
        meta = export_vision_encoder(
            args.model, args.output,
            image_size=args.image_size,
            skip_existing=args.skip_existing,
        )
        if meta:
            print(f"\nDone! Vision encoder exported with "
                  f"{meta['num_merged_tokens']} visual tokens.")
            print(f"Use image_token_id={meta['image_token_id']} "
                  f"in the prompt template.")


if __name__ == "__main__":
    main()
