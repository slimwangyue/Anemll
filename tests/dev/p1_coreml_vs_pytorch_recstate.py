#!/usr/bin/env python3
"""P1-extended: Measure per-token CoreML-vs-PyTorch recurrent state divergence.

Runs chunk 0 (layers 0-3) simultaneously through:
  A) CoreML  (FP16 compute, ANE)
  B) PyTorch (FP16 compute, CPU — matching what CoreML does)
  C) PyTorch (FP32 compute, CPU — the reference)

Tracks divergence of recurrent state and hidden output at each token.
This reveals whether:
  - CoreML FP16 diverges from PyTorch FP16 (CoreML-specific issue)
  - PyTorch FP16 diverges from PyTorch FP32 (fundamental FP16 issue)
  - Or both

Usage:
  cd /Users/yw68/Anemll
  source .venv/bin/activate
  python tests/dev/p1_coreml_vs_pytorch_recstate.py
"""
import gc, os, sys, time
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
sys.path.insert(0, REPO_ROOT)

import coremltools as ct
from config import (
    CTX, NUM_CHUNKS, LUT_BITS, CHUNK_RANGES,
    DEFAULT_HF_MODEL,
)
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
    ane_conv_state_shape,
)

HF_PATH = DEFAULT_HF_MODEL
CHUNK_IDX = 0
SL, EL = CHUNK_RANGES[CHUNK_IDX]  # (0, 3)
# Use the already-exported FP16 chunk
COREML_PATH = os.path.join(REPO_ROOT, "tests", "dev", "_p1_fp32_experiment", "chunk0_fp16.mlpackage")
EMBED_PATH = os.path.join(REPO_ROOT, "qwen3_5_flll_9chunk", "embeddings.mlpackage")
LMHEAD_PATH = os.path.join(REPO_ROOT, "qwen3_5_flll_9chunk", "lm_head_logits.mlpackage")

N_TOKENS = 80  # Total steps to run


def softmax(x):
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, trust_remote_code=True)

    prompt = "What is a stack in computer science? Explain in detail."
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    # ── Load PyTorch model ──
    print("\nLoading PyTorch model...")
    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    pt_model = Qwen35ForCausalLM(cfg)
    assert pt_model.load_pretrained_weights(HF_PATH)
    pt_model.eval()
    for p in pt_model.parameters():
        p.requires_grad = False

    # ── Load CoreML models ──
    print("Loading CoreML chunk0...")
    cm_chunk = ct.models.MLModel(COREML_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE)
    cm_state = cm_chunk.make_state()

    print("Loading CoreML embeddings + lm_head...")
    cm_embed = ct.models.MLModel(EMBED_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)
    cm_lmhead = ct.models.MLModel(LMHEAD_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)

    # Get input shapes from spec
    spec = cm_chunk.get_spec()
    inp_shapes = {}
    for inp in spec.description.input:
        try:
            inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
        except:
            pass

    conv_shape = inp_shapes['linear_conv_state']
    rec_shape = inp_shapes['linear_recurrent_state']
    local_num_layers = EL - SL

    # ── Initialize states ──
    # CoreML state
    cm_conv = np.zeros(conv_shape, dtype=np.float16)
    cm_rec = np.zeros(rec_shape, dtype=np.float16)

    # PyTorch FP16 state
    pt16_conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    pt16_conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(pt16_conv_dim, pt16_conv_kernel)

    # Initialize PyTorch KV caches (same as CoreML)
    pt16_k_cache = torch.zeros(
        (local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
        dtype=torch.float16)
    pt16_v_cache = torch.zeros_like(pt16_k_cache)
    pt32_k_cache = torch.zeros_like(pt16_k_cache)
    pt32_v_cache = torch.zeros_like(pt16_k_cache)

    # Conv and recurrent states for PyTorch
    pt16_conv = torch.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=torch.float16)
    pt16_rec = torch.zeros(rec_shape, dtype=torch.float16)
    pt32_conv = torch.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=torch.float16)
    pt32_rec = torch.zeros(rec_shape, dtype=torch.float32)

    # ── Generate tokens through full pipeline ──
    # For CoreML: embed → chunk0 → (remaining chunks not available, so we use chunk0 output for lm_head)
    # For PyTorch: run same layers 0-2

    # Actually, since we only have chunk 0, we can only compare chunk 0 outputs.
    # For a full generation comparison, we'd need all chunks.
    # Instead, let's compare the recurrent state and hidden output of chunk 0 ONLY.

    print(f"\n{'='*80}")
    print("CoreML vs PyTorch: Chunk 0 recurrent state per-token comparison")
    print(f"{'='*80}")
    print(f"\n{'Step':>4} {'CM_rec_L2':>11} {'PT16_rec_L2':>12} {'PT32_rec_L2':>12} "
          f"{'CM-PT16_cos':>12} {'CM-PT32_cos':>12} {'PT16-PT32_cos':>14}")
    print("-" * 95)

    all_tokens = list(prompt_tokens)

    for step in range(min(len(all_tokens), N_TOKENS)):
        tok_id = all_tokens[step]

        # ── CoreML path ──
        tok_np = np.array([[tok_id]], dtype=np.int32)
        hidden_cm = list(cm_embed.predict({"input_ids": tok_np}).values())[0]

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :step + 1] = 0

        cm_inp = {
            "hidden_states": hidden_cm.astype(np.float16),
            "position_ids": np.array([step], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([step], dtype=np.int32),
            "linear_conv_state": cm_conv,
            "linear_recurrent_state": cm_rec,
        }
        cm_out = cm_chunk.predict(cm_inp, state=cm_state)
        cm_hidden = cm_out['output_hidden_states']
        cm_conv = cm_out['linear_conv_state_out']
        cm_rec = cm_out['linear_recurrent_state_out']

        # ── PyTorch FP16 path ──
        hidden_pt = torch.from_numpy(hidden_cm.astype(np.float16))
        mask_pt = torch.from_numpy(mask)
        pos_pt = torch.tensor([step], dtype=torch.int32)
        cur_pos_pt = torch.tensor([step], dtype=torch.int32)

        with torch.no_grad():
            pt16_out = pt_model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden_pt.half(),
                position_ids=pos_pt,
                causal_mask=mask_pt,
                current_pos=cur_pos_pt,
                kv_cache_0=None,
                k_cache=pt16_k_cache,
                v_cache=pt16_v_cache,
                linear_conv_state=pt16_conv,
                linear_recurrent_state=pt16_rec,
                start_layer=SL,
                end_layer=EL,
                apply_final_norm=False,
            )
        # The function updates states in-place if using the same tensors
        # But actually the model function returns hidden_states only (not state)
        # Need to check how states are updated...

        # Actually, looking at the model code, linear_conv_state and
        # linear_recurrent_state are modified in-place via index assignment.
        # So pt16_conv and pt16_rec ARE updated after the call.
        pt16_hidden = pt16_out

        # ── PyTorch FP32 path (reference) ──
        with torch.no_grad():
            pt32_out = pt_model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden_pt.half(),
                position_ids=pos_pt,
                causal_mask=mask_pt,
                current_pos=cur_pos_pt,
                kv_cache_0=None,
                k_cache=pt32_k_cache,
                v_cache=pt32_v_cache,
                linear_conv_state=pt32_conv,
                linear_recurrent_state=pt32_rec,
                start_layer=SL,
                end_layer=EL,
                apply_final_norm=False,
            )
        pt32_hidden = pt32_out

        # ── Compare recurrent states ──
        cm_rec_flat = cm_rec.flatten().astype(np.float64)
        pt16_rec_flat = pt16_rec.detach().numpy().flatten().astype(np.float64)
        pt32_rec_flat = pt32_rec.detach().numpy().flatten().astype(np.float64)

        cm_l2 = np.sqrt(np.sum(cm_rec_flat ** 2))
        pt16_l2 = np.sqrt(np.sum(pt16_rec_flat ** 2))
        pt32_l2 = np.sqrt(np.sum(pt32_rec_flat ** 2))

        def cosine(a, b):
            d = np.dot(a, b)
            n = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            return d / n

        cm_pt16_cos = cosine(cm_rec_flat, pt16_rec_flat)
        cm_pt32_cos = cosine(cm_rec_flat, pt32_rec_flat)
        pt16_pt32_cos = cosine(pt16_rec_flat, pt32_rec_flat)

        if step % 3 == 0 or step < 5:
            print(f"{step:4d} {cm_l2:11.4f} {pt16_l2:12.4f} {pt32_l2:12.4f} "
                  f"{cm_pt16_cos:12.8f} {cm_pt32_cos:12.8f} {pt16_pt32_cos:14.8f}")

    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL STATE COMPARISON")
    print(f"{'='*80}")

    cm_rec_f = cm_rec.flatten().astype(np.float64)
    pt16_rec_f = pt16_rec.detach().numpy().flatten().astype(np.float64)
    pt32_rec_f = pt32_rec.detach().numpy().flatten().astype(np.float64)

    print(f"\nRecurrent state L2 norms:")
    print(f"  CoreML:          {np.sqrt(np.sum(cm_rec_f**2)):.6f}")
    print(f"  PyTorch FP16:    {np.sqrt(np.sum(pt16_rec_f**2)):.6f}")
    print(f"  PyTorch FP32:    {np.sqrt(np.sum(pt32_rec_f**2)):.6f}")

    print(f"\nRecurrent state cosine similarities:")
    print(f"  CoreML vs PT-FP16: {cosine(cm_rec_f, pt16_rec_f):.8f}")
    print(f"  CoreML vs PT-FP32: {cosine(cm_rec_f, pt32_rec_f):.8f}")
    print(f"  PT-FP16 vs PT-FP32: {cosine(pt16_rec_f, pt32_rec_f):.8f}")

    diff_cm_pt16 = np.sqrt(np.sum((cm_rec_f - pt16_rec_f) ** 2))
    diff_cm_pt32 = np.sqrt(np.sum((cm_rec_f - pt32_rec_f) ** 2))
    diff_pt16_pt32 = np.sqrt(np.sum((pt16_rec_f - pt32_rec_f) ** 2))
    print(f"\nRecurrent state L2 diffs:")
    print(f"  CoreML vs PT-FP16: {diff_cm_pt16:.6f}")
    print(f"  CoreML vs PT-FP32: {diff_cm_pt32:.6f}")
    print(f"  PT-FP16 vs PT-FP32: {diff_pt16_pt32:.6f}")

    # Hidden state comparison (last step)
    cm_h = cm_hidden.flatten().astype(np.float64)
    pt16_h = pt16_hidden.detach().numpy().flatten().astype(np.float64)
    pt32_h = pt32_hidden.detach().numpy().flatten().astype(np.float64)
    print(f"\nHidden state cosine (last step):")
    print(f"  CoreML vs PT-FP16: {cosine(cm_h, pt16_h):.8f}")
    print(f"  CoreML vs PT-FP32: {cosine(cm_h, pt32_h):.8f}")
    print(f"  PT-FP16 vs PT-FP32: {cosine(pt16_h, pt32_h):.8f}")


if __name__ == "__main__":
    main()
