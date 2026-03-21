#!/usr/bin/env python3
"""Diagnose why split E2 norm+proj gives same cos as combined baseline."""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import numpy as np, torch, coremltools as ct, gc
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, Qwen35LinearAttention, MODEL_DTYPE,
)
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT_DIR = "/tmp/qwen35_split_corenorm"
SEQ_LEN = 256; CTX = 1024

def cosine(a, b):
    af = a.flatten().astype(np.float64); bf = b.flatten().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))

cfg = Qwen35Config.from_json(MODEL_PATH + "/config.json")
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

attn = model.model.layers[0].self_attn
attn.export_expected_batch_size = 1
attn.export_expected_seq_len = SEQ_LEN

with torch.no_grad():
    x_norm = model.model.layers[0].input_layernorm(embed)
    conv_s = torch.zeros(1, attn.conv_dim, attn.linear_conv_kernel_dim, dtype=MODEL_DTYPE)
    rec_s = torch.zeros(1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, dtype=MODEL_DTYPE)
    mixed_qkv_pre, z_cf, b_cf, a_cf = attn.proj_stage(x_norm)
    conv_out, _ = attn.conv_stage(mixed_qkv_pre, conv_s, expected_seq_len=SEQ_LEN)
    query, key, value, g, beta, z = attn.layout_stage(conv_out, z_cf, b_cf, a_cf, 1, SEQ_LEN)
    core_raw, _ = Qwen35LinearAttention._chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=rec_s, output_final_state=True,
        expected_batch_size=1, expected_num_heads=attn.num_v_heads,
        expected_seq_len=SEQ_LEN, expected_k_dim=attn.head_k_dim, expected_v_dim=attn.head_v_dim)

print("PyTorch core_raw shape:", core_raw.shape)

# Load ANE recurrence model
cml_r = ct.models.MLModel(OUT_DIR + "/E_recurrence.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
state_r = cml_r.make_state()
out_r = cml_r.predict({
    "query": query.numpy().astype(np.float16),
    "key": key.numpy().astype(np.float16),
    "value": value.numpy().astype(np.float16),
    "g": g.numpy().astype(np.float16),
    "beta": beta.numpy().astype(np.float16),
}, state=state_r)
cml_core_raw = list(out_r.values())[0]
print("CoreML core_raw shape:", cml_core_raw.shape)
print("Recurrence cos: %.10f" % cosine(core_raw.numpy(), cml_core_raw))

# Check data layout
print("\nData layout check:")
print("  PyTorch first 5:", core_raw.numpy().flatten()[:5])
print("  CoreML  first 5:", cml_core_raw.flatten()[:5])
print("  Element max diff: %.6f" % np.abs(core_raw.numpy().astype(np.float32) - cml_core_raw.astype(np.float32)).max())
del cml_r, state_r; gc.collect()

# Load norm+proj model
cml_np = ct.models.MLModel(OUT_DIR + "/E_norm_proj.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)

# PyTorch ref for norm+proj
with torch.no_grad():
    core_for_norm = core_raw.reshape(1, SEQ_LEN, attn.value_dim)
    z_for_norm = z.reshape(1, SEQ_LEN, attn.value_dim)
    cn = attn.core_norm_stage.norm(
        core_for_norm.reshape(-1, attn.head_v_dim),
        z_for_norm.reshape(-1, attn.head_v_dim)
    ).reshape(1, SEQ_LEN, attn.value_dim)
    cf = cn.transpose(1, 2).unsqueeze(2)
    ref_np = attn.core_norm_stage.out_proj(cf.to(MODEL_DTYPE)).squeeze(2).transpose(1, 2)

# TEST 1: Norm+proj with PYTORCH core_raw input
core_pt = core_raw.reshape(1, SEQ_LEN, attn.value_dim).numpy().astype(np.float16)
z_np = z.reshape(1, SEQ_LEN, attn.value_dim).numpy().astype(np.float16)
out_pt = cml_np.predict({"core_raw": core_pt, "z": z_np})
cml_normed_from_pt = list(out_pt.values())[0]
cos_pt = cosine(ref_np.numpy(), cml_normed_from_pt)
print("\nTEST 1: Norm+proj with PYTORCH input -> cos=%.10f" % cos_pt)

# TEST 2: Norm+proj with ANE core_raw input (reshape (1,256,32,128) -> (1,256,4096))
cml_core_reshaped = cml_core_raw.reshape(1, SEQ_LEN, attn.value_dim).astype(np.float16)
out_ane = cml_np.predict({"core_raw": cml_core_reshaped, "z": z_np})
cml_normed_from_ane = list(out_ane.values())[0]
cos_ane = cosine(ref_np.numpy(), cml_normed_from_ane)
print("TEST 2: Norm+proj with ANE input (reshape) -> cos=%.10f" % cos_ane)

# TEST 3: Check if output cos between test1 and test2 differs
cos_12 = cosine(cml_normed_from_pt, cml_normed_from_ane)
diff_12 = np.abs(cml_normed_from_pt.astype(np.float32) - cml_normed_from_ane.astype(np.float32))
print("TEST 3: cos(test1_out, test2_out) = %.10f, max_diff=%.6f" % (cos_12, diff_12.max()))

# Channel analysis for norm+proj with PyTorch input
diff = np.abs(ref_np.numpy().astype(np.float32) - cml_normed_from_pt.astype(np.float32))
ch_err = diff.reshape(-1, diff.shape[-1]).mean(axis=0)
worst5 = np.argsort(ch_err)[-5:][::-1]
print("\nChannel error (norm+proj with PyTorch input):")
for c in worst5:
    print("  ch=%d mean_abs=%.6f" % (c, ch_err[c]))

del cml_np; gc.collect()
print("\nDone.")
