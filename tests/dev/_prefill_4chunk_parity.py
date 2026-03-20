#!/usr/bin/env python3
"""4-chunk prefill parity: export all chunks (batch=1, ctx=256), PyTorch ref, ANE compare.

Usage:
    python tests/dev/_prefill_4chunk_parity.py
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
OUT_DIR = "/tmp/qwen35_4chunk_parity"
BATCH = 1        # seq_len per prefill call
CTX = 256        # context / state length
NUM_CHUNKS = 4
CHUNKS = [(0, 8), (8, 16), (16, 24), (24, 32)]

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load model ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Load model, generate PyTorch refs, export CoreML")
print("=" * 60)
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
print(f"  hidden={cfg.hidden_size}  layers=32  batch={BATCH}  ctx={CTX}")
print(f"  conv_state ANE shape=({ane_d1},{ane_d2})  [orig ({conv_dim},{conv_kernel})]")

# ── 2. Tokenize ────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
prompt = "Explain stack and heap memory in one paragraph."
ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
if ids.shape[1] < BATCH:
    ids = torch.cat([ids, ids[:, :BATCH - ids.shape[1]]], dim=1)
ids = ids[:, :BATCH].to(torch.int32)
print(f"  input_ids: {ids.shape}  tokens: {ids[0,:6].tolist()} …")

# ── 3. PyTorch references (cascaded through all 4 chunks) ─────
print("\n── PyTorch references (all 4 chunks) ──")
with torch.no_grad():
    embed = model.model.embed_tokens(ids).to(torch.float16)
print(f"  embed: {embed.shape}")
np.save(os.path.join(OUT_DIR, "embed.npy"), embed.numpy())

position_ids = torch.arange(BATCH, dtype=torch.int32)
causal_mask = torch.full((1, 1, BATCH, CTX), float("-inf"), dtype=torch.float16)
for r in range(BATCH):
    causal_mask[0, 0, r, :r + 1] = 0
current_pos = torch.tensor([0], dtype=torch.int32)

hidden = embed.clone()
with torch.no_grad():
    for ci, (s, e) in enumerate(CHUNKS):
        nl = e - s
        k_cache = torch.zeros((nl, cfg.num_key_value_heads, CTX, cfg.head_dim), dtype=MODEL_DTYPE)
        v_cache = torch.zeros((nl, cfg.num_key_value_heads, CTX, cfg.head_dim), dtype=MODEL_DTYPE)
        conv_st = torch.zeros((nl, ane_d1, ane_d2), dtype=MODEL_DTYPE)
        rec_st = torch.zeros((nl, cfg.text_config.linear_num_value_heads,
                               cfg.text_config.linear_key_head_dim,
                               cfg.text_config.linear_value_head_dim), dtype=MODEL_DTYPE)
        for li in range(s, e):
            layer = model.model.layers[li]
            if getattr(layer, "layer_type", None) == "linear_attention":
                layer.self_attn.export_expected_batch_size = 1
                layer.self_attn.export_expected_seq_len = BATCH

        hidden = model.model.process_layers_prefill_export_local_state(
            hidden_states=hidden,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            kv_cache_0=None,
            k_cache=k_cache,
            v_cache=v_cache,
            linear_conv_state=conv_st,
            linear_recurrent_state=rec_st,
            start_layer=s,
            end_layer=e,
            apply_final_norm=False,
            expected_batch_size=1,
            expected_seq_len=BATCH,
        )
        np.save(os.path.join(OUT_DIR, f"torch_chunk{ci+1}.npy"), hidden.numpy())
        print(f"  Chunk {ci+1} (layers {s}-{e-1}): {hidden.shape}  "
              f"mean={hidden.float().mean():.6f}  std={hidden.float().std():.6f}")

# ── 4. Export all 4 CoreML chunks ──────────────────────────────
print("\n── Exporting 4 CoreML chunks ──")

class PrefillWrapper(torch.nn.Module):
    def __init__(self, mdl, s, e, seq):
        super().__init__()
        self.model = mdl
        self.start_layer = s
        self.end_layer = e
        self.export_seq_len = seq
        c = mdl.config
        nl = e - s
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

h_in = torch.zeros((1, BATCH, cfg.hidden_size), dtype=torch.float16)
p_in = torch.zeros((BATCH,), dtype=torch.int32)
m_in = torch.zeros((1, 1, BATCH, CTX), dtype=torch.float16)
c_in = torch.zeros((1,), dtype=torch.int32)

for ci, (s, e) in enumerate(CHUNKS):
    nl = e - s
    pkg = os.path.join(OUT_DIR, f"chunk{ci+1}.mlpackage")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)

    print(f"\n  Chunk {ci+1} (layers {s}-{e-1}): tracing …")
    wrapper = PrefillWrapper(model, s, e, BATCH).eval()
    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    wrapper.linear_conv_state.zero_(); wrapper.linear_recurrent_state.zero_()

    traced = torch.jit.trace(wrapper, (h_in, p_in, m_in, c_in))
    for name, buf in traced.named_buffers():
        if any(k in name for k in ("k_cache", "v_cache", "linear_conv", "linear_recurrent")):
            buf.zero_()

    print(f"  Chunk {ci+1}: converting …")
    states = Qwen35Converter.GetChunkLocalTransformerStates(
        model, nl, prefix="", split_full_attention_kv=True)
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
    print(f"  Chunk {ci+1}: saved → {pkg}")
    del mlm, traced, wrapper
    gc.collect()

# Free PyTorch model before loading CoreML
del model, hidden, embed
gc.collect()

# ── 5. ANE parity comparison ──────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: ANE parity comparison")
print("=" * 60)

embed_np = np.load(os.path.join(OUT_DIR, "embed.npy"))
summary = []

for ci in range(NUM_CHUNKS):
    s, e = CHUNKS[ci]
    pkg = os.path.join(OUT_DIR, f"chunk{ci+1}.mlpackage")
    torch_ref = np.load(os.path.join(OUT_DIR, f"torch_chunk{ci+1}.npy"))

    # Input: embed for chunk 1, else PyTorch output from previous chunk
    if ci == 0:
        inp_hidden = embed_np.copy()
    else:
        inp_hidden = np.load(os.path.join(OUT_DIR, f"torch_chunk{ci}.npy"))

    print(f"\n--- Chunk {ci+1} (layers {s}-{e-1}) ---")
    t0 = time.time()
    cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    state = cml.make_state()

    inp = {
        "hidden_states": inp_hidden.astype(np.float16),
        "position_ids": np.arange(BATCH, dtype=np.int32),
        "causal_mask": causal_mask.numpy(),
        "current_pos": np.zeros((1,), dtype=np.int32),
    }

    try:
        t0 = time.time()
        out = cml.predict(inp, state=state)
        cml_out = list(out.values())[0]
        print(f"  Predicted in {time.time()-t0:.3f}s  shape={cml_out.shape}")
    except Exception as exc:
        print(f"  ❌ ANE PREDICT FAILED: {str(exc)[:200]}")
        summary.append((ci + 1, "ANE_FAIL", None, None, None))
        del cml
        gc.collect()
        continue

    # Compare shapes — handle last-chunk slice if needed
    ref = torch_ref
    cml_cmp = cml_out
    if cml_out.shape != ref.shape:
        min_seq = min(cml_out.shape[1], ref.shape[1])
        ref = ref[:, :min_seq, :]
        cml_cmp = cml_out[:, :min_seq, :]
        print(f"  Shape mismatch — comparing first {min_seq} token(s)")

    ref_f = ref.astype(np.float32)
    cml_f = cml_cmp.astype(np.float32)
    diff = np.abs(ref_f - cml_f)
    flat = diff.flatten()
    rf64 = ref_f.flatten().astype(np.float64)
    cf64 = cml_f.flatten().astype(np.float64)
    cos = float(np.dot(rf64, cf64) / (np.linalg.norm(rf64) * np.linalg.norm(cf64) + 1e-12))
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    print(f"  cosine={cos:.10f}  max_abs={max_abs:.6f}  mean_abs={mean_abs:.8f}")
    print(f"  p99={np.percentile(flat, 99):.6f}  p95={np.percentile(flat, 95):.6f}  "
          f"p50={np.percentile(flat, 50):.8f}")

    # Top-3 worst
    worst3 = np.argsort(flat)[-3:][::-1]
    for w in worst3:
        pos = np.unravel_index(w, diff.shape)
        print(f"    worst: pos={pos} ref={ref_f[pos]:.6f} cml={cml_f[pos]:.6f} diff={diff[pos]:.6f}")

    np.save(os.path.join(OUT_DIR, f"cml_chunk{ci+1}.npy"), cml_out)
    summary.append((ci + 1, "OK", max_abs, mean_abs, cos))

    del cml, state, out
    gc.collect()

# ── 6. Summary ─────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  PREFILL PARITY SUMMARY  (batch={BATCH}, ctx={CTX})")
print(f"{'='*64}")
print(f"  {'Chunk':<8} {'Grade':<14} {'max_abs':<12} {'mean_abs':<14} {'cosine':<15}")
print(f"  {'-'*60}")
for ci, status, mx, mn, cos in summary:
    if status == "OK":
        if cos > 0.999 and mx < 0.5:
            grade = "✅ GOOD"
        elif cos > 0.99 and mx < 2.0:
            grade = "⚠️  ACCEPT"
        else:
            grade = "❌ BAD"
        print(f"  {ci:<8} {grade:<14} {mx:<12.6f} {mn:<14.8f} {cos:<15.10f}")
    else:
        print(f"  {ci:<8} {status}")
print(f"{'='*64}")
print(f"Outputs saved to {OUT_DIR}/")
