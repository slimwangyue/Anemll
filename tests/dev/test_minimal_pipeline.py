#!/usr/bin/env python3
"""Minimal test: load models EXACTLY like chat_server.py and run 1 token.

CRITICAL: Loading order must match chat_server.py:
  1. Embeddings
  2. LM Head  (BEFORE FFN chunks!)
  3. FFN chunks (combined dedup with function_name="infer")

Loading FFN before lm_head causes segfaults because ANE resource allocation
is order-dependent. chat_server.py loads lm_head second and it works.
"""
import sys, os
sys.path.insert(0, "/Users/yw68/Anemll")
import numpy as np
import coremltools as ct
import time

STABLE_DIR = "/Users/yw68/Anemll/qwen3_5_stable_models"
cu = ct.ComputeUnit.CPU_AND_NE
CTX = 1024
NUM_CHUNKS = 4


def _load_model(path, compute_unit, function_name=None):
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, compute_unit)
    kwargs = {"compute_units": compute_unit}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def _find_model(base_dir, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


print("=== Loading EXACTLY like chat_server.py (with prefill) ===", flush=True)

# 1. Embeddings (same as chat_server)
print("1. Loading embeddings...", flush=True)
t0 = time.time()
embed = _load_model(_find_model(STABLE_DIR, "embeddings"), cu)
print(f"   {time.time()-t0:.0f}s", flush=True)

# 2. LM Head BEFORE FFN (same as chat_server)
print("2. Loading lm_head...", flush=True)
t0 = time.time()
try:
    lmhead_path = _find_model(STABLE_DIR, "lm_head_logits")
    lm = _load_model(lmhead_path, cu)
    lmhead_mode = "logits"
    print(f"   logits lm_head, {time.time()-t0:.0f}s", flush=True)
except FileNotFoundError:
    lmhead_path = _find_model(STABLE_DIR, "lm_head")
    lm = _load_model(lmhead_path, cu)
    lmhead_mode = "argmax"
    print(f"   argmax lm_head, {time.time()-t0:.0f}s", flush=True)

# 3. FFN chunks: infer + prefill (EXACTLY like chat_server)
print("3. Loading FFN chunks (infer + prefill)...", flush=True)
combined_dir = os.path.join(STABLE_DIR, "combined_LUT4_dedup")
use_combined = os.path.isdir(combined_dir)

ffns = []       # infer models
prefills = []   # prefill models
for ci in range(NUM_CHUNKS):
    if use_combined:
        path = _find_model(combined_dir, f"chunk{ci}")
        if path.endswith(".mlmodelc"):
            use_combined = False
    if use_combined:
        # infer
        t0 = time.time()
        m_infer = _load_model(path, cu, function_name="infer")
        print(f"   chunk {ci} infer  (combined) {time.time()-t0:.0f}s", flush=True)
        # prefill
        t0 = time.time()
        m_prefill = _load_model(path, cu, function_name="prefill")
        print(f"   chunk {ci} prefill (combined) {time.time()-t0:.0f}s", flush=True)
    else:
        path = _find_model(STABLE_DIR, f"ffn_LUT4_chunk{ci}")
        t0 = time.time()
        m_infer = _load_model(path, cu)
        print(f"   chunk {ci} infer  (separate) {time.time()-t0:.0f}s", flush=True)
        m_prefill = None
    ffns.append(m_infer)
    prefills.append(m_prefill)

# Detect shapes (same as chat_server _detect_shapes)
spec = ffns[0].get_spec()
inp_map = {}
fn_inputs = None
if use_combined and hasattr(spec.description, 'functions'):
    for fn in spec.description.functions:
        if fn.name == "infer":
            fn_inputs = fn.input
            break
if fn_inputs is None:
    fn_inputs = spec.description.input
for inp in fn_inputs:
    try:
        inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
    except Exception:
        pass
print(f"   inp_map keys: {list(inp_map.keys())}", flush=True)

# Run 1 token through full pipeline (using pre-allocated buffers like chat_server)
print("\nRunning 1 token through pipeline...", flush=True)
_tok_buf = np.zeros((1, 1), dtype=np.int32)
_mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
_pos_buf = np.zeros(1, dtype=np.int32)

_tok_buf[0, 0] = 1
hidden = list(embed.predict({"input_ids": _tok_buf}).values())[0]
print(f"  embed predict OK: {hidden.shape}", flush=True)

states = [m.make_state() for m in ffns]

_mask_buf[:, :, :, :] = -65504.0
_mask_buf[:, :, :, :1] = 0
_pos_buf[0] = 0

for ci in range(NUM_CHUNKS):
    inp = {
        "hidden_states": hidden.astype(np.float16),
        "position_ids": _pos_buf,
        "causal_mask": _mask_buf,
        "current_pos": _pos_buf,
        "linear_conv_state": np.zeros(inp_map["linear_conv_state"], dtype=np.float16),
        "linear_recurrent_state": np.zeros(inp_map["linear_recurrent_state"], dtype=np.float16),
    }
    out = ffns[ci].predict(inp, state=states[ci])
    hidden = out["output_hidden_states"]
    print(f"  chunk {ci} predict OK", flush=True)

print("  lm_head predict...", flush=True)
lm_out = lm.predict({"hidden_states": hidden.astype(np.float16)})
print(f"  lm_head output keys: {list(lm_out.keys())}", flush=True)
if "output_logits" in lm_out:
    argmax = int(np.argmax(lm_out["output_logits"].flatten()))
elif "argmax_idx" in lm_out:
    argmax = int(lm_out["argmax_idx"].flatten()[0])
else:
    argmax = -1
print(f"  OK! argmax={argmax}", flush=True)

# Run 10 more tokens to verify multi-step stability
print("\nRunning 10 more tokens...", flush=True)
lin_convs = [np.zeros(inp_map["linear_conv_state"], dtype=np.float16) for _ in range(NUM_CHUNKS)]
lin_recs = [np.zeros(inp_map["linear_recurrent_state"], dtype=np.float16) for _ in range(NUM_CHUNKS)]

for step in range(1, 11):
    _tok_buf[0, 0] = argmax
    hidden = list(embed.predict({"input_ids": _tok_buf}).values())[0]
    _mask_buf[:, :, :, :] = -65504.0
    _mask_buf[:, :, :, :step + 1] = 0
    _pos_buf[0] = step

    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": _pos_buf,
            "causal_mask": _mask_buf,
            "current_pos": _pos_buf,
            "linear_conv_state": lin_convs[ci],
            "linear_recurrent_state": lin_recs[ci],
        }
        out = ffns[ci].predict(inp, state=states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_convs[ci] = out['linear_conv_state_out']
            lin_recs[ci] = out['linear_recurrent_state_out']

    lm_out = lm.predict({"hidden_states": hidden.astype(np.float16)})
    if "output_logits" in lm_out:
        argmax = int(np.argmax(lm_out["output_logits"].flatten()))
    elif "argmax_idx" in lm_out:
        argmax = int(lm_out["argmax_idx"].flatten()[0])
    print(f"  step {step}: tok={argmax}", flush=True)

print("\nALL OK — 11 tokens generated!", flush=True)
