#!/usr/bin/env python3
"""Verify: is the parity gap caused by fp32 vs fp16 math in linear attention?

Compares:
1. PyTorch fp32 math (default) — current reference 
2. PyTorch fp16 math (force_fp16_math=True) — simulates CoreML/ANE precision
3. CoreML ANE output

If fp16-math PyTorch matches CoreML better, the root cause is fp32→fp16 precision loss.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, shutil
import numpy as np
import torch
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
SEQ_LEN = 256
CTX = 1024
LAYER_IDX = 0


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def channel_report(ref, cml, label, top_n=5):
    rf = ref.astype(np.float32)
    cf = cml.astype(np.float32)
    diff = np.abs(rf - cf)
    cos = cosine(ref, cml)
    print(f"  [{label}]")
    print(f"    cosine={cos:.10f}  max_abs={diff.max():.6f}  mean_abs={diff.mean():.8f}")
    if len(diff.shape) == 3:
        ch_err = diff[0].mean(axis=0)
        worst = np.argsort(ch_err)[-top_n:][::-1]
        for c in worst:
            bias = (cf[0,:,c] - rf[0,:,c]).mean()
            print(f"      ch={c:4d} mean_abs={ch_err[c]:.6f} bias={bias:+.6f}")
    return cos


# ── 1. Load model ──
print("=" * 70)
print(f"  fp32 vs fp16 Math Diagnosis  (seq={SEQ_LEN}, ctx={CTX}, layer={LAYER_IDX})")
print("=" * 70)
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Realistic input from embed
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
prompt = "Explain stack and heap memory in one paragraph and give one debugging tip."
text = prompt
while True:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    if ids.shape[1] >= SEQ_LEN:
        ids = ids[:, :SEQ_LEN]
        break
    text = text + " " + prompt
ids = ids.to(torch.int32)
with torch.no_grad():
    embed = model.model.embed_tokens(ids).to(torch.float16)

layer = model.model.layers[LAYER_IDX]
attn = layer.self_attn
conv_dim = attn.conv_dim
conv_kernel = attn.linear_conv_kernel_dim

# ── 2. PyTorch reference with fp32 math (default) ──
print("\n── PyTorch fp32 math (default) ──")
attn.export_expected_batch_size = 1
attn.export_expected_seq_len = SEQ_LEN
with torch.no_grad():
    x_norm = layer.input_layernorm(embed)
    conv_st = torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE)
    rec_st = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=MODEL_DTYPE)
    attn_out_fp32, _, _ = attn.forward_prefill_export(
        hidden_states=x_norm, conv_state=conv_st, recurrent_state=rec_st,
        has_previous_state=True, force_fp16_math=False)
    torch_fp32 = (embed + attn_out_fp32).numpy()
print(f"  attn_out mean={attn_out_fp32.float().mean():.6f} std={attn_out_fp32.float().std():.6f}")

# ── 3. PyTorch reference with fp16 math ──
print("\n── PyTorch fp16 math (simulates ANE precision) ──")
with torch.no_grad():
    conv_st2 = torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE)
    rec_st2 = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=MODEL_DTYPE)
    attn_out_fp16, _, _ = attn.forward_prefill_export(
        hidden_states=x_norm, conv_state=conv_st2, recurrent_state=rec_st2,
        has_previous_state=True, force_fp16_math=True)
    torch_fp16 = (embed + attn_out_fp16).numpy()
print(f"  attn_out mean={attn_out_fp16.float().mean():.6f} std={attn_out_fp16.float().std():.6f}")

# ── 4. Compare fp32 vs fp16 PyTorch (shows precision loss in math) ──
print("\n── Comparison: PyTorch fp32 vs fp16 math ──")
cos_fp32_fp16 = channel_report(torch_fp32, torch_fp16, "PyTorch fp32 vs fp16 math")

# Also compare just the attention outputs (before residual)
print("\n── Attention output only: fp32 vs fp16 ──")
cos_attn_fp32_fp16 = channel_report(
    attn_out_fp32.numpy(), attn_out_fp16.numpy(), "attn_out fp32 vs fp16")

# ── 5. Load existing CoreML model and run on ANE ──
print("\n── CoreML ANE output ──")
cml_pkg = "/tmp/qwen35_single_layer_debug/layer0_linear.mlpackage"
if not os.path.exists(cml_pkg):
    print(f"  {cml_pkg} not found, skipping CoreML comparison")
    cml_layer0 = None
else:
    cml = ct.models.MLModel(cml_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = cml.make_state()
    position_ids = np.arange(SEQ_LEN, dtype=np.int32)
    causal_mask = np.full((1, 1, SEQ_LEN, CTX), -65504.0, dtype=np.float16)
    for r in range(SEQ_LEN):
        causal_mask[0, 0, r, :r + 1] = 0
    out = cml.predict({
        "hidden_states": embed.numpy().astype(np.float16),
        "position_ids": position_ids,
        "causal_mask": causal_mask,
        "current_pos": np.zeros((1,), dtype=np.int32),
    }, state=state)
    cml_layer0 = list(out.values())[0]
    print(f"  shape={cml_layer0.shape}")
    del cml, state; gc.collect()

# ── 6. Summary comparisons ──
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")

cos_fp32_cml = None
cos_fp16_cml = None
if cml_layer0 is not None:
    cos_fp32_cml = channel_report(torch_fp32, cml_layer0, "PyTorch fp32 vs CoreML ANE")
    print()
    cos_fp16_cml = channel_report(torch_fp16, cml_layer0, "PyTorch fp16 vs CoreML ANE")

print(f"\n  {'Comparison':<45} {'cosine':<15}")
print(f"  {'-'*60}")
print(f"  {'PyTorch fp32 vs fp16 math':<45} {cos_fp32_fp16:<15.10f}")
if cos_fp32_cml is not None:
    print(f"  {'PyTorch fp32 vs CoreML ANE':<45} {cos_fp32_cml:<15.10f}")
    print(f"  {'PyTorch fp16 vs CoreML ANE':<45} {cos_fp16_cml:<15.10f}")
    if cos_fp16_cml > cos_fp32_cml:
        print(f"\n  CONFIRMED: fp16 PyTorch is closer to CoreML ({cos_fp16_cml:.6f} > {cos_fp32_cml:.6f})")
        print(f"  Root cause: fp32→fp16 precision loss in recurrence math")
    else:
        print(f"\n  UNEXPECTED: fp16 PyTorch is NOT closer to CoreML")
        print(f"  Root cause is NOT purely fp32→fp16 — investigate ANE-specific op behavior")
print(f"{'='*70}")
