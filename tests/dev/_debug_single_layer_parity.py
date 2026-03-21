#!/usr/bin/env python3
"""Single-layer parity: test each layer type (linear_attn / full_attn) individually.

Exports layer 0 (linear_attention) and layer 3 (full_attention) separately
with batch=256, ctx=1024. Includes sub-component analysis (attn-only vs full layer).
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, shutil, time
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_single_layer_debug"
SEQ_LEN = 256
CTX = 1024

os.makedirs(OUT_DIR, exist_ok=True)


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12)) if d > 1e-12 else 0.0


def analyze(ref, cml, label):
    rf = ref.astype(np.float32)
    cf = cml.astype(np.float32)
    diff = np.abs(rf - cf)
    cos = cosine(ref, cml)
    print(f"  [{label}] cosine={cos:.10f}  max_abs={diff.max():.6f}  mean_abs={diff.mean():.8f}")
    # channel analysis (last dim)
    if len(diff.shape) == 3:
        ch_err = diff[0].mean(axis=0)
        worst5 = np.argsort(ch_err)[-5:][::-1]
        for c in worst5:
            bias = (cf[0,:,c] - rf[0,:,c]).mean()
            print(f"    ch={c:4d} mean_abs={ch_err[c]:.6f} bias={bias:+.6f}")
    return cos


# ── 1. Load model ──────────────────────────────────────────────
print("=" * 70)
print(f"  Single-Layer Parity Debug  (seq={SEQ_LEN}, ctx={CTX})")
print("=" * 70)
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
model.eval()
for p in model.parameters():
    p.requires_grad = False

# ── 2. Get embed output as realistic input ─────────────────────
from transformers import AutoTokenizer
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
print(f"  embed: {embed.shape}  mean={embed.float().mean():.6f}  std={embed.float().std():.6f}")

position_ids = torch.arange(SEQ_LEN, dtype=torch.int32)
causal_mask = torch.full((1, 1, SEQ_LEN, CTX), float("-inf"), dtype=torch.float16)
for r in range(SEQ_LEN):
    causal_mask[0, 0, r, :r + 1] = 0
current_pos = torch.tensor([0], dtype=torch.int32)

conv_dim_val = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
conv_kernel_val = max(1, int(cfg.text_config.linear_conv_kernel_dim))
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim_val, conv_kernel_val)


# ── 3. Test single LINEAR ATTENTION layer (layer 0) ──────────
print(f"\n{'='*70}")
print("  TEST A: Single Linear Attention Layer (layer 0)")
print(f"{'='*70}")

layer0 = model.model.layers[0]
assert layer0.layer_type == "linear_attention"

# PyTorch reference
with torch.no_grad():
    layer0.self_attn.export_expected_batch_size = 1
    layer0.self_attn.export_expected_seq_len = SEQ_LEN

    # Full layer: RMSNorm + attention + residual + RMSNorm + MLP + residual
    x_norm = layer0.input_layernorm(embed)
    conv_state_0 = torch.zeros(1, conv_dim_val, conv_kernel_val, dtype=MODEL_DTYPE)
    rec_state_0 = torch.zeros(1, cfg.text_config.linear_num_value_heads,
                               cfg.text_config.linear_key_head_dim,
                               cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)

    attn_out_0, next_conv_0, next_rec_0 = layer0.self_attn.forward_prefill_export(
        hidden_states=x_norm, conv_state=conv_state_0, recurrent_state=rec_state_0,
        has_previous_state=True,
    )
    hidden_after_attn = embed + attn_out_0
    post_norm = layer0.post_attention_layernorm(hidden_after_attn)
    mlp_out = layer0.mlp(post_norm)
    torch_layer0_out = hidden_after_attn + mlp_out

print(f"  torch attn_out: mean={attn_out_0.float().mean():.6f} std={attn_out_0.float().std():.6f}")
print(f"  torch mlp_out:  mean={mlp_out.float().mean():.6f} std={mlp_out.float().std():.6f}")
print(f"  torch layer_out: mean={torch_layer0_out.float().mean():.6f} std={torch_layer0_out.float().std():.6f}")

np.save(os.path.join(OUT_DIR, "torch_attn_out_0.npy"), attn_out_0.numpy())
np.save(os.path.join(OUT_DIR, "torch_layer0_out.npy"), torch_layer0_out.numpy())

# Export as CoreML (full single-layer wrapper)
class SingleLayerWrapper(nn.Module):
    def __init__(self, mdl, layer_idx, layer_type):
        super().__init__()
        self.model = mdl
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        c = mdl.config
        if layer_type == "full_attention":
            self.register_buffer("k_cache", torch.zeros(1, c.num_key_value_heads, c.state_length, c.head_dim, dtype=MODEL_DTYPE))
            self.register_buffer("v_cache", torch.zeros(1, c.num_key_value_heads, c.state_length, c.head_dim, dtype=MODEL_DTYPE))
        if layer_type == "linear_attention":
            cd = conv_dim_val; ck = conv_kernel_val
            a1, a2 = ane_conv_state_shape(cd, ck)
            self.register_buffer("linear_conv_state", torch.zeros(1, a1, a2, dtype=MODEL_DTYPE))
            self.register_buffer("linear_recurrent_state", torch.zeros(
                1, c.text_config.linear_num_value_heads, c.text_config.linear_key_head_dim,
                c.text_config.linear_value_head_dim, dtype=MODEL_DTYPE))
            la = mdl.model.layers[layer_idx]
            la.self_attn.export_expected_batch_size = 1
            la.self_attn.export_expected_seq_len = SEQ_LEN

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        k_c = getattr(self, "k_cache", None)
        v_c = getattr(self, "v_cache", None)
        lcs = getattr(self, "linear_conv_state", None)
        lrs = getattr(self, "linear_recurrent_state", None)
        return self.model.model._process_layer_prefill_export_local_state(
            layer_idx=self.layer_idx, local_layer_idx=0,
            hidden_states=hidden_states, position_ids=position_ids,
            causal_mask=causal_mask, current_pos=current_pos,
            kv_cache_0=None, k_cache=k_c, v_cache=v_c,
            linear_conv_state=lcs,
            linear_recurrent_state=lrs,
            local_num_layers=1, expected_batch_size=1, expected_seq_len=SEQ_LEN)


def get_states_for_layer(model, layer_type):
    """Return ct.StateType list appropriate for the layer type."""
    cfg = model.config
    states = []
    if layer_type == "full_attention":
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=np.float16),
            name="k_cache"))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=np.float16),
            name="v_cache"))
    if layer_type == "linear_attention":
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, ane_d1, ane_d2), dtype=np.float16),
            name="linear_conv_state"))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, cfg.text_config.linear_num_value_heads,
                       cfg.text_config.linear_key_head_dim,
                       cfg.text_config.linear_value_head_dim), dtype=np.float16),
            name="linear_recurrent_state"))
    return states

print("\n  Exporting layer 0 CoreML …")
wrapper = SingleLayerWrapper(model, 0, "linear_attention").eval()
h_in = torch.zeros((1, SEQ_LEN, cfg.hidden_size), dtype=torch.float16)
p_in = torch.zeros((SEQ_LEN,), dtype=torch.int32)
m_in = torch.zeros((1, 1, SEQ_LEN, CTX), dtype=torch.float16)
c_in = torch.zeros((1,), dtype=torch.int32)

traced = torch.jit.trace(wrapper, (h_in, p_in, m_in, c_in))
for name, buf in traced.named_buffers():
    if any(k in name for k in ("k_cache", "v_cache", "linear_conv", "linear_recurrent")):
        buf.zero_()

states = get_states_for_layer(model, "linear_attention")
pkg = os.path.join(OUT_DIR, "layer0_linear.mlpackage")
if os.path.exists(pkg): shutil.rmtree(pkg)
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
print(f"  Saved {pkg}")
del mlm, traced, wrapper; gc.collect()

# Run on ANE
print("  Running on ANE …")
cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
state = cml.make_state()
inp = {
    "hidden_states": embed.numpy().astype(np.float16),
    "position_ids": position_ids.numpy(),
    "causal_mask": causal_mask.numpy(),
    "current_pos": np.zeros((1,), dtype=np.int32),
}
out = cml.predict(inp, state=state)
cml_layer0 = list(out.values())[0]
print(f"  CoreML output: {cml_layer0.shape}")

cos_layer0 = analyze(torch_layer0_out.numpy(), cml_layer0, "Layer 0 (linear_attn full layer)")
del cml, state; gc.collect()


# ── 4. Test single FULL ATTENTION layer (layer 3) ──────────────
print(f"\n{'='*70}")
print("  TEST B: Single Full Attention Layer (layer 3)")
print(f"{'='*70}")

layer3 = model.model.layers[3]
assert layer3.layer_type == "full_attention", f"Layer 3 type is {layer3.layer_type}"

# PyTorch reference (using embed as input, ignoring layers 0-2 for isolation)
with torch.no_grad():
    x_norm3 = layer3.input_layernorm(embed)
    q3, k3, v3, g3 = layer3.self_attn.get_new_kv_cache_prefill(x_norm3, position_ids)

    k_cache3 = torch.zeros(cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    v_cache3 = torch.zeros(cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    k_cache3[:, 0:SEQ_LEN, :] = k3.squeeze(0)
    v_cache3[:, 0:SEQ_LEN, :] = v3.squeeze(0)

    attn_out_3 = layer3.self_attn.forward_prefill(
        hidden_states=x_norm3, query_states=q3,
        kv_cache_layer=(k_cache3, v_cache3), causal_mask=causal_mask, gate=g3)
    hidden_after_attn3 = embed + attn_out_3
    post_norm3 = layer3.post_attention_layernorm(hidden_after_attn3)
    mlp_out3 = layer3.mlp(post_norm3)
    torch_layer3_out = hidden_after_attn3 + mlp_out3

print(f"  torch attn_out: mean={attn_out_3.float().mean():.6f} std={attn_out_3.float().std():.6f}")
print(f"  torch mlp_out:  mean={mlp_out3.float().mean():.6f} std={mlp_out3.float().std():.6f}")
print(f"  torch layer_out: mean={torch_layer3_out.float().mean():.6f} std={torch_layer3_out.float().std():.6f}")
np.save(os.path.join(OUT_DIR, "torch_layer3_out.npy"), torch_layer3_out.numpy())

# Export layer 3
print("\n  Exporting layer 3 CoreML …")
wrapper3 = SingleLayerWrapper(model, 3, "full_attention").eval()
traced3 = torch.jit.trace(wrapper3, (h_in, p_in, m_in, c_in))
for name, buf in traced3.named_buffers():
    if any(k in name for k in ("k_cache", "v_cache", "linear_conv", "linear_recurrent")):
        buf.zero_()

states3 = get_states_for_layer(model, "full_attention")
pkg3 = os.path.join(OUT_DIR, "layer3_full.mlpackage")
if os.path.exists(pkg3): shutil.rmtree(pkg3)
mlm3 = ct.convert(
    traced3,
    inputs=[
        ct.TensorType(name="hidden_states", shape=h_in.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=p_in.shape, dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=m_in.shape, dtype=np.float16),
        ct.TensorType(name="current_pos", shape=c_in.shape, dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    states=states3,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm3.save(pkg3)
print(f"  Saved {pkg3}")
del mlm3, traced3, wrapper3; gc.collect()

# Run on ANE
print("  Running on ANE …")
cml3 = ct.models.MLModel(pkg3, compute_units=ct.ComputeUnit.CPU_AND_NE)
state3 = cml3.make_state()
out3 = cml3.predict(inp, state=state3)
cml_layer3 = list(out3.values())[0]
print(f"  CoreML output: {cml_layer3.shape}")

cos_layer3 = analyze(torch_layer3_out.numpy(), cml_layer3, "Layer 3 (full_attn full layer)")
del cml3, state3; gc.collect()


# ── 5. Test just RMSNorm + MLP (no attention) ──────────────────
print(f"\n{'='*70}")
print("  TEST C: RMSNorm + MLP only (layer 0, no attention)")
print(f"{'='*70}")

class MLPOnlyWrapper(nn.Module):
    def __init__(self, mdl, layer_idx):
        super().__init__()
        self.norm = mdl.model.layers[layer_idx].post_attention_layernorm
        self.mlp = mdl.model.layers[layer_idx].mlp

    def forward(self, hidden_states):
        x = self.norm(hidden_states)
        return self.mlp(x)

with torch.no_grad():
    torch_mlp0_out = layer0.mlp(layer0.post_attention_layernorm(embed))
np.save(os.path.join(OUT_DIR, "torch_mlp0_out.npy"), torch_mlp0_out.numpy())

print("  Exporting MLP-only CoreML …")
mlp_wrapper = MLPOnlyWrapper(model, 0).eval()
h_mlp = torch.zeros((1, SEQ_LEN, cfg.hidden_size), dtype=torch.float16)
traced_mlp = torch.jit.trace(mlp_wrapper, (h_mlp,))

pkg_mlp = os.path.join(OUT_DIR, "layer0_mlp_only.mlpackage")
if os.path.exists(pkg_mlp): shutil.rmtree(pkg_mlp)
mlm_mlp = ct.convert(
    traced_mlp,
    inputs=[ct.TensorType(name="hidden_states", shape=h_mlp.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_mlp.save(pkg_mlp)
del mlm_mlp, traced_mlp, mlp_wrapper; gc.collect()

print("  Running on ANE …")
cml_mlp = ct.models.MLModel(pkg_mlp, compute_units=ct.ComputeUnit.CPU_AND_NE)
out_mlp = cml_mlp.predict({"hidden_states": embed.numpy().astype(np.float16)})
cml_mlp_out = list(out_mlp.values())[0]
cos_mlp = analyze(torch_mlp0_out.numpy(), cml_mlp_out, "Layer 0 MLP only (RMSNorm + FFN)")
del cml_mlp; gc.collect()


# ── 6. Test just linear attention (no MLP, no residual) ────────
print(f"\n{'='*70}")
print("  TEST D: Linear Attention only (layer 0, no residual, no MLP)")
print(f"{'='*70}")

class LinearAttnOnlyWrapper(nn.Module):
    def __init__(self, mdl, layer_idx, seq_len):
        super().__init__()
        self.norm = mdl.model.layers[layer_idx].input_layernorm
        self.attn = mdl.model.layers[layer_idx].self_attn
        cd = conv_dim_val; ck = conv_kernel_val
        a1, a2 = ane_conv_state_shape(cd, ck)
        self.register_buffer("conv_state", torch.zeros(1, a1, a2, dtype=MODEL_DTYPE))
        self.register_buffer("rec_state", torch.zeros(
            1, cfg.text_config.linear_num_value_heads, cfg.text_config.linear_key_head_dim,
            cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE))
        self.attn.export_expected_batch_size = 1
        self.attn.export_expected_seq_len = seq_len

    def forward(self, hidden_states):
        x = self.norm(hidden_states)
        conv_dim = self.attn.conv_dim
        conv_kernel = self.attn.linear_conv_kernel_dim
        cs = self.conv_state.reshape(1, conv_dim, conv_kernel)
        out, nc, nr = self.attn.forward_prefill_export(
            hidden_states=x, conv_state=cs, recurrent_state=self.rec_state,
            has_previous_state=True)
        return out  # just the attention output, no residual

with torch.no_grad():
    torch_attn_only = attn_out_0  # already computed above

print("  Exporting linear-attn-only CoreML …")
attn_wrapper = LinearAttnOnlyWrapper(model, 0, SEQ_LEN).eval()
traced_attn = torch.jit.trace(attn_wrapper, (h_mlp,))
for name, buf in traced_attn.named_buffers():
    if any(k in name for k in ("conv_state", "rec_state")):
        buf.zero_()

# Need states for conv_state and rec_state
attn_states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, ane_d1, ane_d2), dtype=np.float16),
        name="conv_state"),
    ct.StateType(
        wrapped_type=ct.TensorType(
            shape=(1, cfg.text_config.linear_num_value_heads,
                   cfg.text_config.linear_key_head_dim,
                   cfg.text_config.linear_value_head_dim), dtype=np.float16),
        name="rec_state"),
]

pkg_attn = os.path.join(OUT_DIR, "layer0_attn_only.mlpackage")
if os.path.exists(pkg_attn): shutil.rmtree(pkg_attn)
mlm_attn = ct.convert(
    traced_attn,
    inputs=[ct.TensorType(name="hidden_states", shape=h_mlp.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    states=attn_states,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm_attn.save(pkg_attn)
del mlm_attn, traced_attn, attn_wrapper; gc.collect()

print("  Running on ANE …")
cml_attn = ct.models.MLModel(pkg_attn, compute_units=ct.ComputeUnit.CPU_AND_NE)
state_attn = cml_attn.make_state()
out_attn = cml_attn.predict({"hidden_states": embed.numpy().astype(np.float16)}, state=state_attn)
cml_attn_out = list(out_attn.values())[0]
cos_attn = analyze(torch_attn_only.numpy(), cml_attn_out, "Layer 0 Linear Attn only (RMSNorm + attn, no residual)")
del cml_attn, state_attn; gc.collect()


# ── 7. Summary ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SUMMARY  (seq={SEQ_LEN}, ctx={CTX})")
print(f"{'='*70}")
print(f"  {'Test':<45} {'cosine':<15}")
print(f"  {'-'*60}")
print(f"  {'A: Layer 0 full layer (linear_attn)':<45} {cos_layer0:<15.10f}")
print(f"  {'B: Layer 3 full layer (full_attn)':<45} {cos_layer3:<15.10f}")
print(f"  {'C: Layer 0 MLP only (RMSNorm+FFN)':<45} {cos_mlp:<15.10f}")
print(f"  {'D: Layer 0 Linear Attn only (no resid)':<45} {cos_attn:<15.10f}")
print(f"{'='*70}")
