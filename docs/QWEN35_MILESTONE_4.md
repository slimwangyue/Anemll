# Qwen3.5-4B Milestone 4 — Vision Encoder & Multimodal Inference

**Date**: 2026-05-02  
**Model**: Qwen3.5-4B VLM (32 layers: hybrid Full/Linear attention, Mamba-2 style)  
**Previous**: Milestone 3.4 (KV Cache Write-Position Fix, 2026-04-27)

## Summary

Added a complete vision pipeline to Qwen3.5-4B on ANE: a multi-resolution ViT
vision encoder exported as a single CoreML multi-function model, integrated
with the text decoder via visual embedding injection and 3D MRoPE position
encoding. The vision encoder processes images through Conv3d patch embedding,
24 transformer blocks with 2D rotary embeddings, and a 2×2 PatchMerger to
produce visual tokens compatible with the text decoder's hidden states. End-to-end
multimodal inference is verified on ANE — the model correctly identifies objects
in photographs with Chinese-language responses.

## Architecture

### Vision Encoder Pipeline

```
Image [H, W, 3]
  → Preprocess: resize bicubic, normalize (mean=0.5, std=0.5), duplicate T=2
  → pixel_values [1, 3, 2, H, W] float16
  → Conv3d(3→1024, kernel=(2,16,16), stride=(2,16,16))
  → [1, 1024, 1, H/16, W/16]
  → Reshape + raster-to-merger-group reorder
  → [num_patches, 1024] in merger-grouped order
  → + interpolated position embeddings (2304→grid bilinear)
  → 24× VisionBlock (LayerNorm + SDPA + 2D RoPE + LayerNorm + MLP)
  → PatchMerger (2×2 spatial merge: norm → concat 4×1024 → MLP → 2560)
  → visual_embeddings [1, num_merged_tokens, 2560]
```

### Key Design Decisions

1. **Raster-to-merger-group reorder** (`_raster_to_merger_group()`): Conv3d
   outputs patches in raster order (row-major scan), but HuggingFace's
   processor delivers them in merger-group order (2×2 spatial blocks grouped
   consecutively). This reorder ensures position embeddings, rotary embeddings,
   and the PatchMerger's `view(-1, 4*C)` all align correctly.

2. **Pre-computed rotary embeddings**: 2D RoPE cos/sin tables are computed
   at init time in merger-group order and registered as buffers, avoiding
   `torch.outer` at trace time for ANE compatibility.

3. **Bilinear position interpolation**: The 48×48=2304 learned position
   embeddings are interpolated to the actual grid size using pre-computed
   indices and weights, then reordered to merger-group order.

4. **Multi-resolution support**: Three resolutions exported as CoreML
   multi-function model — one function per resolution, shared weights.

### Multi-Resolution Design

| Resolution | Grid | Merged Tokens | Aspect | Use Case |
|-----------|------|--------------|--------|----------|
| 448×448 | 28×28 → 14×14 | 196 | 1:1 square | Default, general photos |
| 448×896 | 28×56 → 14×28 | 392 | 1:2 landscape | Wide panoramas |
| 896×448 | 56×28 → 28×14 | 392 | 2:1 portrait | Phone screenshots |

Resolution selection uses log-scale aspect ratio matching to pick the closest
supported resolution for each input image.

Token count formula: `num_merged_tokens = (H / (patch_size × merge_size)) × (W / (patch_size × merge_size))`

Where `patch_size=16`, `merge_size=2`, so effective stride = 32 pixels per merged token.

### Multimodal Inference Flow

```
1. Vision encoder: image → visual_embeddings [1, N, 2560]

2. Prompt construction (token-level):
   <|im_start|>user\n
   <|vision_start|>                     ← vision_start_token_id (248053)
   <|image_pad|> × N                    ← image_pad_token_id (248056) × num_merged_tokens
   <|vision_end|>                       ← vision_end_token_id (248054)
   {user_text}<|im_end|>\n
   <|im_start|>assistant\n
   <think>\n\n</think>\n\n              ← or <think>\n if think mode enabled

3. Prefill with visual embedding injection:
   For each token position:
     - Run through embed model → hidden_states [1, 1, 2560]
     - If position ∈ [spanStart, spanStart+N):
         overwrite hidden_states with visual_embeddings[pos - spanStart]
     - Set MRoPE position_ids:
         text before image: [p, p, p] (sequential)
         image token i:     [spanStart, spanStart+row, spanStart+col]
         text after image:  [p+Δ, p+Δ, p+Δ] where Δ = max(gridH,gridW) - N
     - Run through FFN chunks → update KV cache

4. Decode: standard autoregressive with ropeDelta offset
```

### MRoPE (Multi-dimensional Rotary Position Encoding)

Qwen3.5-VL uses 3D position IDs `[temporal, height, width]` for rotary
embeddings with `mrope_section=[11, 11, 10]` (32 total rotary dims):

| Token Type | Temporal | Height | Width |
|-----------|----------|--------|-------|
| Text (before image) | pos | pos | pos |
| Image token (row r, col c) | spanStart | spanStart + r | spanStart + c |
| Text (after image) | pos + Δ | pos + Δ | pos + Δ |
| Decode tokens | pos + Δ | pos + Δ | pos + Δ |

Where `Δ = ropeDelta = max(gridH, gridW) - numImageTokens`.

For 448×448: gridH=14, gridW=14, N=196, Δ = 14 - 196 = **-182**.

This compresses 196 image tokens into a 14×14 spatial grid in the position
space, then resumes text positions from `spanStart + 14` instead of
`spanStart + 196`, preserving the spatial locality learned during training.

## Model Files

### Vision Encoder

```
qwen3_5_4b_mrope/
  vision_encoder_multi_lut6.mlpackage     ← Multi-function CoreML model (LUT6)
  vision_encoder_multi_lut6_meta.json     ← Per-function metadata
  vision_encoder_multi_meta.json          ← Combined function index
  vision_resolutions.json                 ← Resolution selection index
  vision_meta.json                        ← 448×448 metadata
  vision_meta_448x896.json                ← 448×896 metadata
  vision_meta_896x448.json                ← 896×448 metadata
```

The multi-function model contains 3 functions with shared weights:
- `f_448x448`: input `[1, 3, 2, 448, 448]` → output `[1, 196, 2560]`
- `f_448x896`: input `[1, 3, 2, 448, 896]` → output `[1, 392, 2560]`
- `f_896x448`: input `[1, 3, 2, 896, 448]` → output `[1, 392, 2560]`

### Text Decoder (unchanged from Milestone 3.4)

```
qwen3_5_4b_mrope/
  embed_single.mlpackage / .mlmodelc      ← Token embeddings (single token)
  embed_prefill.mlpackage / .mlmodelc     ← Token embeddings (batch prefill)
  lm_head_nosplit.mlpackage / .mlmodelc   ← LM head (argmax)
  combined_LUT4_dedup/
    chunk{0..8}.mlpackage / .mlmodelc     ← 9 FFN chunks (LUT4, combined infer+prefill)
```

Configuration: 9-chunk FLLL partition, LUT4 FFN, LUT6 embeddings/LM head,
batch_size=256, context_length=4096, V4 KV cache precision policy.

## Vision Pipeline Scripts

### Export & Quantization

| Script | Purpose |
|--------|---------|
| `scripts_qwen3_5/export_vision.py` | Export Qwen3.5 ViT to CoreML (single or multi-resolution) |
| `scripts_qwen3_5/quantize_vision.py` | LUT6 quantization of vision encoder |
| `scripts_qwen3_5/combine_vision.py` | Combine per-resolution models into multi-function package |

### Inference & Testing

| Script | Purpose |
|--------|---------|
| `scripts_qwen3_5/chat_server_vision.py` | Vision-capable chat server (extends ChatEngine) |
| `scripts_qwen3_5/test_vision_cli.py` | End-to-end vision test (image + question → answer) |
| `scripts_qwen3_5/test_vision_multi.py` | Multi-resolution vision test |

### Development & Diagnostics

All in `tests/dev/`:

| Script | Purpose |
|--------|---------|
| `test_vision_parity.py` | 7-test parity suite: token count, prompt construction, patch ordering, pixel layout, CoreML sanity, prompt tokens |
| `test_vision_encoder_eval.py` | Vision encoder evaluation against HF reference |
| `test_vision_encoder_pipeline.py` | Pipeline integration test |
| `test_vision_lut6_img8204.py` | LUT6 quality test with specific image |
| `test_vision_mrope.py` | MRoPE position computation test |
| `test_vision_rope_diag.py` | Detailed rotary embedding diagnostics |
| `test_vision_thinking_fix.py` | Think mode + vision interaction test |
| `test_batch_vision_prefill.py` | Batch prefill with visual embedding injection test |
| `requantize_vision_dedup.py` | Re-quantize vision encoder with dedup |

## Export Pipeline

```bash
# 1. Export vision encoder for all resolutions (fp32 trace → CoreML fp16)
python scripts_qwen3_5/export_vision.py \
    --model models/Qwen__Qwen3.5-4B \
    --output qwen3_5_4b_mrope \
    --resolutions 448x448,448x896,896x448

# 2. Quantize to LUT6
python scripts_qwen3_5/quantize_vision.py \
    --input qwen3_5_4b_mrope \
    --output qwen3_5_4b_mrope

# 3. Combine into multi-function model
python scripts_qwen3_5/combine_vision.py \
    --input qwen3_5_4b_mrope \
    --output qwen3_5_4b_mrope

# 4. Test end-to-end
python scripts_qwen3_5/test_vision_cli.py
```

## Preprocessing Specification

Image preprocessing must match exactly between Python server and Swift client:

| Parameter | Value |
|-----------|-------|
| Color space | RGB (convert from any mode) |
| Resize method | Bicubic (Python PIL) / Lanczos (Swift UIKit) |
| Normalization mean | 0.5 (per channel) |
| Normalization std | 0.5 (per channel) |
| Output range | [-1.0, +1.0] |
| Temporal duplication | T=2 (identical frames) |
| Output layout | NCTHW: `[1, 3, 2, H, W]` float16 |
| EXIF orientation | Applied before processing |

Normalization formula: `pixel = (raw / 255.0 - 0.5) / 0.5 = raw / 127.5 - 1.0`

**Note**: This uses Qwen3.5-VL specific normalization (mean=std=0.5), NOT
ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

## Verification Results

### End-to-End Test

**Test image**: `tests/IMG_8981.jpg` (airplane in blue sky)  
**Question**: "有几个物体？是什么物体？" (How many objects? What objects?)

| Platform | Response | Status |
|----------|----------|--------|
| Python (`test_vision_cli.py`) | "图中有一个物体，是飞机。它正处于空中飞行状态，背景为蓝色天空。" | **PASS** |
| Swift (iOS app) | Under testing | — |

Translation: "There is one object in the image, an airplane. It is in flight, with a blue sky background."

### Parity Test Suite (`test_vision_parity.py`)

| Test | Description | Status |
|------|-------------|--------|
| Token count | Grid math produces correct merged token count | **PASS** |
| Prompt construction | vision_start + pad×N + vision_end in correct positions | **PASS** |
| Patch ordering | Raster-to-merger-group reorder matches HF | **PASS** |
| Pixel layout | NCTHW memory layout matches Python preprocessing | **PASS** |
| CoreML sanity | Vision encoder loads and produces non-trivial output | **PASS** |
| Prompt tokens | Full prompt tokenization matches expected structure | **PASS** |

### Swift Pipeline Audit

Full line-by-line audit of the iOS/macOS vision pipeline
(`VisionEncoder.swift` + `AnemllInferenceProvider.swift`) confirmed:

| Component | Verified |
|-----------|----------|
| Pixel preprocessing (NCTHW, mean/std=0.5, EXIF rotation) | ✓ |
| Stride-aware encoder output reading (fast + slow path) | ✓ |
| fp16→fp32→fp16 visual embedding conversion | ✓ |
| Token-level prompt construction (buildMultimodalTurnIDs) | ✓ |
| Visual span position computation (cold + warm path) | ✓ |
| Batch embedding injection (injectVisualEmbeddingsIntoBatch) | ✓ |
| Single-step embedding injection (injectVisualEmbeddingSingle) | ✓ |
| MRoPE computation in batch prefill (3 branches: pre-image, image, post-image) | ✓ |
| MRoPE computation in sequential prefill (stepKVOnly) | ✓ |
| MRoPE computation in decode (step, ropeDelta offset) | ✓ |
| ropeDelta = max(gridH, gridW) - numImageTokens | ✓ |
| Causal mask construction | ✓ |

## Key Technical Details

### Raster-to-Merger-Group Reorder

The critical insight in the vision encoder: Conv3d outputs patches in **raster
order** (left-to-right, top-to-bottom), but the HuggingFace model expects
**merger-group order** (2×2 spatial blocks grouped consecutively).

For a 4×4 grid with merge_size=2:

```
Raster order:     0  1  2  3        Merger-group order:  0  1  4  5
                  4  5  6  7                             2  3  6  7
                  8  9  10 11                            8  9  12 13
                  12 13 14 15                            10 11 14 15
```

Implementation:
```python
x_2d = x.view(grid_h, grid_w, -1)
x_grouped = x_2d.view(merged_h, merge, merged_w, merge, -1)
                 .permute(0, 2, 1, 3, 4)
                 .reshape(-1, hidden_size)
```

Without this reorder, position embeddings and rotary embeddings would be
misaligned with the patch content, and the PatchMerger would concatenate
patches from wrong spatial locations.

### Visual Embedding Injection

Visual embeddings replace the text embedding output **after** the embed model
and **before** the FFN chunks, so visual information flows through the entire
transformer stack — the same path text tokens take.

```
Token at position p:
  1. embed(token_id) → hidden [1, 1, 2560]
  2. if p ∈ image_span: hidden = visual_embeddings[p - spanStart]
  3. FFN_chunk_0(hidden, ...) → hidden
  4. FFN_chunk_1(hidden, ...) → hidden
  ...
  9. FFN_chunk_8(hidden, ...) → hidden
  10. if last_token: lm_head(hidden) → next_token
```

For **batch prefill** (256 tokens at once), injection overwrites rows in the
batch hidden state array. For **sequential prefill** (one token at a time),
injection overwrites the single hidden state.

## Comparison with Milestone 3.4

| Metric | Milestone 3.4 | Milestone 4 |
|--------|--------------|-------------|
| Modality | Text only | **Text + Vision** |
| Vision encoder | — | **ViT + PatchMerger (24 blocks, LUT6)** |
| Multi-resolution | — | **3 resolutions (448², 448×896, 896×448)** |
| Position encoding | 1D RoPE | **3D MRoPE (temporal, height, width)** |
| Visual tokens | — | **196 (square) / 392 (rectangular)** |
| Prompt construction | String-based | **Token-level (multimodal spans)** |
| Text decoder | 9-chunk FLLL, LUT4, CTX=4096 | **Unchanged** |
| Decode speed | ~5.9 tok/s (M4 Pro) | **~5.9 tok/s** (vision adds only prefill cost) |

## Known Limitations

1. **No compiled .mlmodelc**: The vision encoder exists only as `.mlpackage`.
   CoreML loads and compiles it at runtime (first-load latency ~5–10s).

2. **Three resolutions only**: Images with extreme aspect ratios (>2:1) are
   resized to the nearest supported resolution, which may crop or distort.

3. **Single image per turn**: The current implementation supports one image
   per user message. Multi-image support would require extending the prompt
   construction and visual span tracking.

4. **Vision encoder latency**: The 24-block ViT adds ~200–500ms to the first
   token latency (prefill phase only, does not affect decode speed).

## iOS On-Device Warm Continuation — Context-Dependent Validation

**Date**: 2026-05-02  
**Device**: iPhone 14 Pro Max (A16 Bionic), iOS 26.5  
**Model**: Qwen3.5-2B (7 chunks, hidden=2048, batch_size=256, CPU+ANE)

### Problem

Initial warm continuation tests used 5 independent questions (e.g., "What is 2+2?",
"What color is the sky?") that could be answered without any prior context. This meant
the test validated TTFT speedup but **not** whether the KV cache actually preserved
conversation context across rounds.

### Fix: Dependent Round Design

The `testWarmContinuation()` test in `Anemll2BMultiRoundPrefillTests.swift` was
redesigned with context-dependent questions where each round's correct answer
requires information established in Round 1:

| Round | Question | Expected Answer | Validates |
|-------|----------|----------------|-----------|
| R1 | "My name is Alice. I have a dog named Max and a cat named Luna. Remember these facts." | Acknowledgment | Fact seeding |
| R2 | "What is the name of my dog?" | Must contain "Max" | KV cache preserves R1 context |
| R3 | "And what is the name of my cat?" | Must contain "Luna" | Continued context retention |
| R4 | "How many pets do I have in total?" | Must reference "2" | Cross-fact reasoning |
| R5 | "Summarize: what is my name and what are my pets?" | Must contain "Alice" | Full context recall |

### Assertions

Three `XCTAssert` validations enforce context dependency:

```swift
XCTAssertTrue(r2Lower.contains("max"),
    "R2 should mention dog 'Max' from R1 context")
XCTAssertTrue(r3Lower.contains("luna"),
    "R3 should mention cat 'Luna' from R1 context")
XCTAssertTrue(r5Lower.contains("alice"),
    "R5 summary should mention 'Alice' from R1 context")
```

### Results (iPhone 14 Pro Max, A16)

**All assertions passed** — `TEST EXECUTE SUCCEEDED`, 0 failures, 262.876 seconds.

| Round | Output | TTFT (ms) | Speedup vs R1 | Decode |
|-------|--------|-----------|---------------|--------|
| R1 (cold) | *(fact acknowledgment)* | 224,749 | — | — |
| R2 | "Your dog's name is **Max**." | 2,273 | **98.9×** | 3.8 tok/s |
| R3 | *(mentions Luna)* | 2,321 | **96.8×** | — |
| R4 | *(mentions 2 pets)* | 2,287 | **98.3×** | — |
| R5 | "You are **Alice**, and you have two pets: a dog named **Max** and a cat named **Luna**." | 2,753 | **81.6×** | 4.6 tok/s |

### Key Findings

1. **Context fully preserved**: The model correctly recalled all facts (Alice, Max,
   Luna, 2 pets) across 5 warm rounds, proving the KV cache retains prior conversation
   state during warm continuation.

2. **Consistent TTFT speedup**: Warm rounds achieve ~80–99× faster TTFT compared to
   the cold first round (~2.3s vs ~225s), since only the new user message needs
   prefilling — the prior KV cache state is reused.

3. **Decode throughput stable**: 3.8–4.6 tok/s across warm rounds on A16, consistent
   with the 2B model's ANE decode performance.

4. **R1 cold start dominates**: The 225s cold TTFT for R1 includes full model loading
   and initial prefill. Subsequent warm rounds skip this entirely.

### Test File

`/Volumes/MySSD/Edge-AI-agent/local_llmTests/Anemll2BMultiRoundPrefillTests.swift`
— method `testWarmContinuation()`.

## Next Steps

- [ ] Compile vision encoder to `.mlmodelc` for faster first-load
- [ ] Add 672×448 and 448×672 resolutions (3:2 / 2:3 aspect ratios)
- [ ] Multi-image support (multiple visual spans per prompt)
- [ ] Vision encoder latency optimization (layer pruning, token merging)
- [ ] Upload vision-capable model bundle to HuggingFace
- [ ] Integrate into `anemll-swift-cli` for command-line vision testing
