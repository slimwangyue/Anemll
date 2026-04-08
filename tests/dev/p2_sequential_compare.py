#!/usr/bin/env python3
"""Minimal sequential comparison: FP16 vs FP32 compute precision on chunk 0.

Loads only ONE CoreML model at a time + PyTorch reference.
Saves results to avoid reloading.
"""
import gc, os, sys, time
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
sys.path.insert(0, REPO_ROOT)

import coremltools as ct
from config import CTX, CHUNK_RANGES, DEFAULT_HF_MODEL
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
)

HF_PATH = DEFAULT_HF_MODEL
SL, EL = CHUNK_RANGES[0]  # (0, 3)
EMBED_PATH = os.path.join(REPO_ROOT, "qwen3_5_flll_9chunk", "embeddings.mlpackage")
FP16_CHUNK = os.path.join(REPO_ROOT, "tests", "dev", "_p2_fp32_compute", "chunk0_fp16compute_lut6.mlpackage")
FP32_CHUNK = os.path.join(REPO_ROOT, "tests", "dev", "_p2_fp32_compute", "chunk0_fp32compute_lut6.mlpackage")


def run_coreml_chunk(chunk_path, label, tokens, embed):
    """Run a CoreML chunk on tokens, return final recurrent state + hidden per step."""
    print(f"\nRunning {label}...")
    m = ct.models.MLModel(chunk_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = m.make_state()

    spec = m.get_spec()
    inp_shapes = {}
    for inp in spec.description.input:
        try:
            inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
        except:
            pass

    conv = np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16)
    rec = np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16)

    rec_per_step = []
    hid_per_step = []

    for step, tok_id in enumerate(tokens):
        tok_np = np.array([[tok_id]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok_np}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :step + 1] = 0

        out = m.predict({
            "hidden_states": hidden.astype(np.float16),
            "position_ids": np.array([step], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([step], dtype=np.int32),
            "linear_conv_state": conv,
            "linear_recurrent_state": rec,
        }, state=state)

        conv = out['linear_conv_state_out']
        rec = out['linear_recurrent_state_out']
        rec_per_step.append(rec.flatten().astype(np.float64).copy())
        hid_per_step.append(out['output_hidden_states'].flatten().astype(np.float64).copy())

    print(f"  Done ({len(tokens)} steps)")
    del m, state
    gc.collect()
    return rec_per_step, hid_per_step


def run_pytorch_chunk(model, tokens, embed):
    """Run PyTorch chunk 0 on tokens, return final recurrent state per step."""
    print(f"\nRunning PyTorch FP32 reference...")
    cfg = model.config
    local_num_layers = EL - SL

    k_cache = torch.zeros(
        (local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
        dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)

    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)

    rec_shape = (local_num_layers, cfg.text_config.linear_num_value_heads,
                 cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim)
    pt_conv = torch.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=torch.float16)
    pt_rec = torch.zeros(rec_shape, dtype=torch.float32)

    rec_per_step = []
    hid_per_step = []

    for step, tok_id in enumerate(tokens):
        tok_np = np.array([[tok_id]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok_np}).values())[0]
        hidden_pt = torch.from_numpy(hidden.astype(np.float16))

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :step + 1] = 0

        with torch.no_grad():
            pt_out = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden_pt.half(),
                position_ids=torch.tensor([step], dtype=torch.int32),
                causal_mask=torch.from_numpy(mask),
                current_pos=torch.tensor([step], dtype=torch.int32),
                kv_cache_0=None,
                k_cache=k_cache, v_cache=v_cache,
                linear_conv_state=pt_conv,
                linear_recurrent_state=pt_rec,
                start_layer=SL, end_layer=EL,
                apply_final_norm=False,
            )

        rec_per_step.append(pt_rec.detach().numpy().flatten().astype(np.float64).copy())
        hid_per_step.append(pt_out.detach().numpy().flatten().astype(np.float64).copy())

    print(f"  Done ({len(tokens)} steps)")
    return rec_per_step, hid_per_step


def cosine(a, b):
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return d / n


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, trust_remote_code=True)

    prompt = "What is a stack in computer science? Explain in detail."
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(tokens)}")

    # Load embeddings (CPU_ONLY, stays loaded throughout)
    print("Loading embeddings...")
    embed = ct.models.MLModel(EMBED_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)

    # Run FP16-compute CoreML
    rec_fp16, hid_fp16 = run_coreml_chunk(FP16_CHUNK, "CoreML FP16-compute", tokens, embed)

    # Run FP32-compute CoreML
    rec_fp32, hid_fp32 = run_coreml_chunk(FP32_CHUNK, "CoreML FP32-compute", tokens, embed)

    # Run PyTorch reference
    print("\nLoading PyTorch model...")
    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    rec_pt, hid_pt = run_pytorch_chunk(model, tokens, embed)

    del model, embed
    gc.collect()

    # Comparison table
    print(f"\n{'='*90}")
    print(f"RECURRENT STATE COSINE SIMILARITY vs PyTorch-FP32 REFERENCE")
    print(f"{'='*90}")
    print(f"{'Step':>4} {'CM-FP16 vs PT':>14} {'CM-FP32 vs PT':>14} {'CM-FP16 vs CM-FP32':>19} {'PT rec L2':>10}")
    print("-" * 70)

    for step in range(len(tokens)):
        c16_pt = cosine(rec_fp16[step], rec_pt[step])
        c32_pt = cosine(rec_fp32[step], rec_pt[step])
        c16_32 = cosine(rec_fp16[step], rec_fp32[step])
        pt_l2 = np.linalg.norm(rec_pt[step])

        if step % 3 == 0 or step < 5 or step == len(tokens) - 1:
            print(f"{step:4d} {c16_pt:14.8f} {c32_pt:14.8f} {c16_32:19.8f} {pt_l2:10.4f}")

    # Final summary
    n = len(tokens) - 1
    print(f"\n{'='*60}")
    print(f"FINAL (step {n}):")
    print(f"  CoreML FP16 vs PyTorch: {cosine(rec_fp16[n], rec_pt[n]):.8f}")
    print(f"  CoreML FP32 vs PyTorch: {cosine(rec_fp32[n], rec_pt[n]):.8f}")
    print(f"  CoreML FP16 vs FP32:    {cosine(rec_fp16[n], rec_fp32[n]):.8f}")

    # Hidden state comparison
    print(f"\n  Hidden cosine (step {n}):")
    print(f"    CM-FP16 vs PT: {cosine(hid_fp16[n], hid_pt[n]):.8f}")
    print(f"    CM-FP32 vs PT: {cosine(hid_fp32[n], hid_pt[n]):.8f}")
    print(f"    CM-FP16 vs CM-FP32: {cosine(hid_fp16[n], hid_fp32[n]):.8f}")

    # Diagnosis
    improvement = cosine(rec_fp32[n], rec_pt[n]) - cosine(rec_fp16[n], rec_pt[n])
    print(f"\n  FP32 compute improvement: {improvement:+.8f}")
    if improvement > 0.001:
        print("  → FP32 compute HELPS. CoreML FP16 compute causes divergence.")
        print("  → Recommendation: use compute_precision=FLOAT32 for all chunks")
    elif improvement > 0.0001:
        print("  → FP32 compute provides marginal improvement.")
    else:
        print("  → FP32 compute does NOT help. Divergence is from MIL lowering, not compute precision.")
        print("  → The issue may be in op decomposition, quantization effects, or ANE-specific numerics.")


if __name__ == "__main__":
    main()
