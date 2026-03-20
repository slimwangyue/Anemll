#!/usr/bin/env python3
"""Chunk 1 prefill parity: export (batch=1, ctx=256), PyTorch ref, ANE compare.

Usage:
    python tests/dev/_chunk1_parity.py
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, shutil, time
import numpy as np
import torch
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_chunk1_parity"
BATCH = 1        # seq_len per prefill call
CTX = 256        # context / state length
START_LAYER = 0
END_LAYER = 8    # chunk 1: layers 0-7
NUM_LOCAL = END_LAYER - START_LAYER

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load model ──────────────────────────────────────────────
print("Loading model …")
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
model.eval()
for p in model.parameters():
    p.requires_grad = False

conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)
print(f"  hidden={cfg.hidden_size}  layers=0-7  batch={BATCH}  ctx={CTX}")
print(f"  conv_state ANE shape=({ane_d1},{ane_d2})  [orig ({conv_dim},{conv_kernel})]")

# ── 2. Tokenize ────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
prompt = "Explain stack and heap memory in one paragraph."
ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
# Pad/truncate to BATCH tokens
if ids.shape[1] < BATCH:
    ids = torch.cat([ids, ids[:, :BATCH - ids.shape[1]]], dim=1)
ids = ids[:, :BATCH].to(torch.int32)
print(f"  input_ids: {ids.shape}  tokens: {ids[0,:6].tolist()} …")

# ── 3. PyTorch reference ───────────────────────────────────────
print("\n── PyTorch reference (chunk 1, layers 0-7) ──")
with torch.no_grad():
    embed = model.model.embed_tokens(ids).to(torch.float16)
print(f"  embed: {embed.shape}")

position_ids = torch.arange(BATCH, dtype=torch.int32)
causal_mask = torch.full((1, 1, BATCH, CTX), float("-inf"), dtype=torch.float16)
for r in range(BATCH):
    causal_mask[0, 0, r, :r + 1] = 0
current_pos = torch.tensor([0], dtype=torch.int32)

k_cache = torch.zeros((NUM_LOCAL, cfg.num_key_value_heads, CTX, cfg.head_dim), dtype=MODEL_DTYPE)
v_cache = torch.zeros((NUM_LOCAL, cfg.num_key_value_heads, CTX, cfg.head_dim), dtype=MODEL_DTYPE)
conv_st = torch.zeros((NUM_LOCAL, ane_d1, ane_d2), dtype=MODEL_DTYPE)
rec_st = torch.zeros((NUM_LOCAL, cfg.text_config.linear_num_value_heads,
                       cfg.text_config.linear_key_head_dim,
                       cfg.text_config.linear_value_head_dim), dtype=MODEL_DTYPE)

for li in range(START_LAYER, END_LAYER):
    layer = model.model.layers[li]
    if getattr(layer, "layer_type", None) == "linear_attention":
        layer.self_attn.export_expected_batch_size = 1
        layer.self_attn.export_expected_seq_len = BATCH

with torch.no_grad():
    torch_out = model.model.process_layers_prefill_export_local_state(
        hidden_states=embed.clone(),
        position_ids=position_ids,
        causal_mask=causal_mask,
        current_pos=current_pos,
        kv_cache_0=None,
        k_cache=k_cache,
        v_cache=v_cache,
        linear_conv_state=conv_st,
        linear_recurrent_state=rec_st,
        start_layer=START_LAYER,
        end_layer=END_LAYER,
        apply_final_norm=False,
        expected_batch_size=1,
        expected_seq_len=BATCH,
    )

print(f"  torch_out: {torch_out.shape}")
np.save(os.path.join(OUT_DIR, "torch_chunk1.npy"), torch_out.numpy())
np.save(os.path.join(OUT_DIR, "embed.npy"), embed.numpy())

# ── 4. Export CoreML ────────────────────────────────────────────
print("\n── Exporting CoreML (chunk 1, batch=1) ──")

class PrefillWrapper(torch.nn.Module):
    def __init__(self, mdl, s, e, seq):
        super().__init__()
        self.model = mdl; self.start_layer = s; self.end_layer = e
        self.export_seq_len = seq; c = mdl.config; nl = e - s
        self.register_buffer("k_cache", torch.zeros(
            (nl, c.num_key_value_heads, c.state_length, c.head_dim), dtype=MODEL_DTYPE))
        self.register_buffer("v_cache", torch.zeros(
            (nl, c.num_key_value_heads, c.state_length, c.head_dim), dtype=MODEL_DTYPE))
        cd = (c.text_config.linear_num_key_heads * c.text_config.linear_key_head_dim * 2
              + c.text_config.linear_num_value_heads * c.text_config.linear_value_head_dim)
        ck = max(1, int(c.text_config.linear_conv_kernel_dim))
        a1, a2 = ane_conv_state_shape(cd, ck)
        self.register_buffer("linear_conv_state",
            torch.zeros((nl, a1, a2), dtype=MODEL_DTYPE))
        self.register_buffer("linear_recurrent_state",
            torch.zeros((nl, c.text_config.linear_num_value_heads,
                         c.text_config.linear_key_head_dim,
                         c.text_config.linear_value_head_dim), dtype=MODEL_DTYPE))
        for li in range(s, e):
            la = mdl.model.layers[li]
            if getattr(la, "layer_type", None) == "linear_attention":
                la.self_attn.export_expected_batch_size = 1
                la.self_attn.export_expected_seq_len = seq
        self.states = Qwen35Converter.GetChunkLocalTransformerStates(
            mdl, nl, prefix="", split_full_attention_kv=True)

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        return self.model.model.process_layers_prefill_export_local_state(
            hidden_states=hidden_states, position_ids=position_ids,
            causal_mask=causal_mask, current_pos=current_pos,
            kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
            linear_conv_state=self.linear_conv_state,
            linear_recurrent_state=self.linear_recurrent_state,
            start_layer=self.start_layer, end_layer=self.end_layer,
            apply_final_norm=False, expected_batch_size=1,
            expected_seq_len=self.export_seq_len)

wrapper = PrefillWrapper(model, START_LAYER, END_LAYER, BATCH).eval()

h_in = torch.zeros((1, BATCH, cfg.hidden_size), dtype=torch.float16)
p_in = torch.zeros((BATCH,), dtype=torch.int32)
m_in = torch.zeros((1, 1, BATCH, CTX), dtype=torch.float16)
c_in = torch.zeros((1,), dtype=torch.int32)

wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
wrapper.linear_conv_state.zero_(); wrapper.linear_recurrent_state.zero_()

print("  Tracing …")
traced = torch.jit.trace(wrapper, (h_in, p_in, m_in, c_in))
states = Qwen35Converter.GetChunkLocalTransformerStates(
    model, NUM_LOCAL, prefix="", split_full_attention_kv=True)

pkg = os.path.join(OUT_DIR, "chunk1.mlpackage")
if os.path.exists(pkg):
    shutil.rmtree(pkg)

print("  Converting …")
mlm = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="hidden_states", shape=h_in.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=p_in.shape, dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=m_in.shape, dtype=np.float16),
        ct.TensorType(name="current_pos", shape=c_in.shape, dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    states=states,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm.save(pkg)
print(f"  Saved → {pkg}")
del mlm, traced, wrapper
gc.collect()

# Free PyTorch model before loading CoreML
del model
gc.collect()

# ── 5. ANE parity ──────────────────────────────────────────────
print("\n── ANE parity comparison ──")
torch_ref = np.load(os.path.join(OUT_DIR, "torch_chunk1.npy"))
embed_np = np.load(os.path.join(OUT_DIR, "embed.npy"))

print(f"  Loading CoreML on ANE …")
t0 = time.time()
cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
print(f"  Loaded in {time.time()-t0:.1f}s")
state = cml.make_state()

inp = {
    "hidden_states": embed_np.astype(np.float16),
    "position_ids": np.arange(BATCH, dtype=np.int32),
    "causal_mask": causal_mask.numpy(),
    "current_pos": np.zeros((1,), dtype=np.int32),
}

t0 = time.time()
out = cml.predict(inp, state=state)
cml_out = list(out.values())[0]
print(f"  Predicted in {time.time()-t0:.3f}s  shape={cml_out.shape}")

# ── Metrics ─────────────────────────────────────────────────────
ref = torch_ref.astype(np.float32)
cml_f = cml_out.astype(np.float32)
diff = np.abs(ref - cml_f)
flat = diff.flatten()

rf64 = ref.flatten().astype(np.float64)
cf64 = cml_f.flatten().astype(np.float64)
cos = np.dot(rf64, cf64) / (np.linalg.norm(rf64) * np.linalg.norm(cf64) + 1e-12)

print(f"\n  ╔═══════════════════════════════════════════════════╗")
print(f"  ║  CHUNK 1 PARITY  (batch={BATCH}, ctx={CTX})          ║")
print(f"  ╠═══════════════════════════════════════════════════╣")
print(f"  ║  cosine similarity : {cos:.10f}               ║")
print(f"  ║  max_abs_diff      : {diff.max():.6f}                   ║")
print(f"  ║  mean_abs_diff     : {diff.mean():.8f}                 ║")
print(f"  ║  p99               : {np.percentile(flat,99):.6f}                   ║")
print(f"  ║  p95               : {np.percentile(flat,95):.6f}                   ║")
print(f"  ║  p50               : {np.percentile(flat,50):.8f}                 ║")
print(f"  ╚═══════════════════════════════════════════════════╝")

# Top-5 worst
worst5 = np.argsort(flat)[-5:][::-1]
print("\n  Top-5 worst positions:")
for w in worst5:
    pos = np.unravel_index(w, diff.shape)
    print(f"    pos={pos}  ref={ref[pos]:.6f}  cml={cml_f[pos]:.6f}  diff={diff[pos]:.6f}")

# Per-token cosine (if multiple tokens)
if ref.shape[1] > 1:
    print("\n  Per-token cosine:")
    for t in range(min(ref.shape[1], 10)):
        r = ref[0, t, :].astype(np.float64)
        c = cml_f[0, t, :].astype(np.float64)
        tc = np.dot(r, c) / (np.linalg.norm(r) * np.linalg.norm(c) + 1e-12)
        td = np.abs(r - c)
        print(f"    token {t}: cos={tc:.8f}  max_diff={td.max():.6f}  mean_diff={td.mean():.6f}")

# Grade
if cos > 0.999 and diff.max() < 0.5:
    print("\n  ✅ GOOD parity")
elif cos > 0.99 and diff.max() < 2.0:
    print("\n  ⚠️  ACCEPTABLE parity")
else:
    print("\n  ❌ POOR parity — investigate fp32 vs fp16 math or op lowering")

np.save(os.path.join(OUT_DIR, "cml_chunk1.npy"), cml_out)
print(f"\nOutputs saved to {OUT_DIR}/")
