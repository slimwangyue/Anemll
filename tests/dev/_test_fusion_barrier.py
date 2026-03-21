#!/usr/bin/env python3
"""Quick test: does .clone() fusion barrier improve linear attention parity?"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, shutil
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE,
)
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
SEQ_LEN, CTX = 256, 1024

cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = CTX; cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
model.load_pretrained_weights(MODEL_PATH)
model.eval()
for p in model.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
prompt = "Explain stack and heap memory in one paragraph and give one debugging tip."
text = prompt
while True:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    if ids.shape[1] >= SEQ_LEN:
        ids = ids[:, :SEQ_LEN]; break
    text = text + " " + prompt
ids = ids.to(torch.int32)
with torch.no_grad():
    embed = model.model.embed_tokens(ids).to(torch.float16)

conv_dim_val = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
conv_kernel_val = max(1, int(cfg.text_config.linear_conv_kernel_dim))
ane_d1, ane_d2 = ane_conv_state_shape(conv_dim_val, conv_kernel_val)

# PyTorch reference
model.model.layers[0].self_attn.export_expected_batch_size = 1
model.model.layers[0].self_attn.export_expected_seq_len = SEQ_LEN
with torch.no_grad():
    x = model.model.layers[0].input_layernorm(embed)
    cs = torch.zeros(1, conv_dim_val, conv_kernel_val, dtype=MODEL_DTYPE)
    rs = torch.zeros(1, cfg.text_config.linear_num_value_heads,
                      cfg.text_config.linear_key_head_dim,
                      cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)
    ao, _, _ = model.model.layers[0].self_attn.forward_prefill_export(
        hidden_states=x, conv_state=cs, recurrent_state=rs, has_previous_state=True)
    torch_ref = ao.numpy()


class AttnOnlyW(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.norm = mdl.model.layers[0].input_layernorm
        self.attn = mdl.model.layers[0].self_attn
        a1, a2 = ane_conv_state_shape(conv_dim_val, conv_kernel_val)
        self.register_buffer("conv_state", torch.zeros(1, a1, a2, dtype=MODEL_DTYPE))
        self.register_buffer("rec_state", torch.zeros(
            1, cfg.text_config.linear_num_value_heads,
            cfg.text_config.linear_key_head_dim,
            cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE))
        self.attn.export_expected_batch_size = 1
        self.attn.export_expected_seq_len = SEQ_LEN

    def forward(self, hidden_states):
        x = self.norm(hidden_states)
        cs = self.conv_state.reshape(1, conv_dim_val, conv_kernel_val)
        out, nc, nr = self.attn.forward_prefill_export(
            hidden_states=x, conv_state=cs, recurrent_state=self.rec_state,
            has_previous_state=True)
        return out


w = AttnOnlyW(model).eval()
h = torch.zeros((1, SEQ_LEN, cfg.hidden_size), dtype=torch.float16)
tr = torch.jit.trace(w, (h,))
for n, b in tr.named_buffers():
    if any(k in n for k in ("conv_state", "rec_state")):
        b.zero_()

attn_states = [
    ct.StateType(wrapped_type=ct.TensorType(shape=(1, ane_d1, ane_d2), dtype=np.float16),
                 name="conv_state"),
    ct.StateType(wrapped_type=ct.TensorType(
        shape=(1, cfg.text_config.linear_num_value_heads,
               cfg.text_config.linear_key_head_dim,
               cfg.text_config.linear_value_head_dim), dtype=np.float16),
                 name="rec_state"),
]
pkg = "/tmp/qwen35_fusion_barrier_test.mlpackage"
if os.path.exists(pkg):
    shutil.rmtree(pkg)
print("Converting...")
mlm = ct.convert(
    tr,
    inputs=[ct.TensorType(name="hidden_states", shape=h.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
    states=attn_states,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
mlm.save(pkg)
del mlm, tr, w
gc.collect()

print("Running on ANE...")
cml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
state = cml.make_state()
out = cml.predict({"hidden_states": embed.numpy().astype(np.float16)}, state=state)
cml_out = list(out.values())[0]

af = torch_ref.flatten().astype(np.float64)
bf = cml_out.flatten().astype(np.float64)
cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
diff = np.abs(torch_ref.astype(np.float32) - cml_out.astype(np.float32))
ch_err = diff[0].mean(axis=0)

print(f"clone() barrier: cos={cos:.10f}  max_abs={diff.max():.6f}  mean_abs={diff.mean():.8f}")
print(f"  ch=0 mean_abs={ch_err[0]:.6f}")
print(f"  Comparison: without barrier cos=0.9940, ch0=0.136")
