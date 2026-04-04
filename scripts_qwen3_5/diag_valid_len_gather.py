#!/usr/bin/env python3
"""Test whether the valid_len one-hot gather in conv_stage is truly dynamic.

Loads chunk0 only and runs prefill with two different valid_len values:
  A) valid_len = 16  (the actual prompt length)
  B) valid_len = 512 (= BATCH_SIZE, i.e. "all valid, no padding")

If the conv_state outputs are IDENTICAL, the one-hot gather is compiled
with static indices (frozen at the tracing value), confirming the bug.

Also runs sequential (infer) with 16 tokens as the reference.
"""
import sys, os, time
import numpy as np
import coremltools as ct

MODEL_DIR = "/Users/yw68/Anemll/qwen3_5_stable_models"
CTX = 2048
BATCH_SIZE = 512

PROMPT_TOKENS = [248045, 846, 198, 95975, 120282, 126114, 97255, 248046,
                 198, 248045, 74455, 198, 248068, 271, 248069, 271]

cu = ct.ComputeUnit.CPU_AND_NE

def find_model(base, prefix):
    for ext in (".mlpackage", ".mlmodelc"):
        p = os.path.join(base, prefix + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model matching {prefix} in {base}")

# ── Load models ──────────────────────────────────────────────────────

combined_dir = os.path.join(MODEL_DIR, "combined_LUT6_dedup")
embed_path = find_model(MODEL_DIR, "embeddings")

print("Loading embeddings…", flush=True)
embed = ct.models.MLModel(embed_path, compute_units=cu)

print("Loading chunk0…", flush=True)
t0 = time.time()
chunk0_path = find_model(combined_dir, "chunk0")
m_infer = ct.models.MLModel(chunk0_path, compute_units=cu, function_name="infer")
m_prefill = ct.models.MLModel(chunk0_path, compute_units=cu, function_name="prefill")
print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

# ── Detect state shapes ────────────────────────────────────────────

spec = m_infer.get_spec()
fn_descs = list(spec.description.functions) if hasattr(spec.description, 'functions') and spec.description.functions else [spec.description]
infer_desc = fn_descs[0]
for fn in fn_descs:
    if fn.name == 'infer':
        infer_desc = fn
        break
inp_shapes = {}
for inp in infer_desc.input:
    inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
out_shapes = {}
for o in infer_desc.output:
    out_shapes[o.name] = tuple(o.type.multiArrayType.shape)

conv_shape = inp_shapes.get('linear_conv_state')
rec_shape = inp_shapes.get('linear_recurrent_state')
print(f"conv_state input shape: {conv_shape}")
print(f"rec_state input shape:  {rec_shape}")

# Also check prefill input shapes
pf_desc = None
for fn in fn_descs:
    if fn.name == 'prefill':
        pf_desc = fn
        break
if pf_desc:
    pf_inp_shapes = {}
    for inp in pf_desc.input:
        pf_inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
    print(f"prefill conv_state input shape: {pf_inp_shapes.get('linear_conv_state')}")
    print(f"prefill rec_state input shape:  {pf_inp_shapes.get('linear_recurrent_state')}")
    print(f"prefill valid_len input shape:  {pf_inp_shapes.get('valid_len')}")

# ── Helper: run single prefill call ─────────────────────────────────

def run_prefill(embed_model, prefill_model, infer_model, valid_len_val, tag):
    """Run chunk0 prefill with the given valid_len and return conv/rec states."""
    state = infer_model.make_state()
    lin_conv = np.zeros(conv_shape, dtype=np.float16)
    lin_rec  = np.zeros(rec_shape, dtype=np.float16)

    # Embed full batch
    input_ids = np.zeros((1, BATCH_SIZE), dtype=np.int32)
    input_ids[0, :len(PROMPT_TOKENS)] = PROMPT_TOKENS
    hidden = list(embed_model.predict({"input_ids": input_ids}).values())[0]

    # Causal mask
    mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
    for i in range(len(PROMPT_TOKENS)):
        mask[0, 0, i, :i + 1] = 0
    # Padding rows stay all -inf

    pos_ids = np.zeros((BATCH_SIZE,), dtype=np.int32)
    pos_ids[:len(PROMPT_TOKENS)] = np.arange(len(PROMPT_TOKENS), dtype=np.int32)

    cur_pos = np.array([0], dtype=np.int32)
    valid_len_arr = np.array([valid_len_val], dtype=np.int32)

    inp = {
        "hidden_states": hidden.astype(np.float16),
        "position_ids": pos_ids,
        "causal_mask": mask,
        "current_pos": cur_pos,
        "linear_conv_state": lin_conv,
        "linear_recurrent_state": lin_rec,
        "valid_len": valid_len_arr,
    }
    t0 = time.time()
    out = prefill_model.predict(inp, state=state)
    dt = time.time() - t0
    c_out = out.get('linear_conv_state_out')
    r_out = out.get('linear_recurrent_state_out')
    h_out = out.get('output_hidden_states')
    print(f"[{tag}] prefill took {dt:.2f}s, valid_len={valid_len_val}")
    if c_out is not None:
        print(f"  conv_state_out: shape={c_out.shape}, max={np.abs(c_out).max():.4f}, "
              f"mean={np.abs(c_out.astype(np.float32)).mean():.6f}")
    if r_out is not None:
        print(f"  rec_state_out:  shape={r_out.shape}, max={np.abs(r_out).max():.4f}")
    return c_out, r_out, h_out


def run_sequential(embed_model, infer_model):
    """Run chunk0 with 16 tokens sequentially."""
    state = infer_model.make_state()
    lin_conv = np.zeros(conv_shape, dtype=np.float16)
    lin_rec  = np.zeros(rec_shape, dtype=np.float16)

    tok_buf = np.zeros((1, 1), dtype=np.int32)
    mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    pos_buf = np.zeros((1,), dtype=np.int32)

    for i, tok_id in enumerate(PROMPT_TOKENS):
        tok_buf[0, 0] = tok_id
        hidden = list(embed_model.predict({"input_ids": tok_buf}).values())[0]
        mask_buf[:, :, :, :] = -65504.0
        mask_buf[:, :, :, :i + 1] = 0
        pos_buf[0] = i

        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_buf,
            "causal_mask": mask_buf,
            "current_pos": pos_buf,
            "linear_conv_state": lin_conv,
            "linear_recurrent_state": lin_rec,
        }
        out = infer_model.predict(inp, state=state)
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_conv = out['linear_conv_state_out']
            lin_rec  = out['linear_recurrent_state_out']

    print(f"[sequential] 16 tokens done")
    print(f"  conv_state_out: shape={lin_conv.shape}, max={np.abs(lin_conv).max():.4f}, "
          f"mean={np.abs(lin_conv.astype(np.float32)).mean():.6f}")
    print(f"  rec_state_out:  shape={lin_rec.shape}, max={np.abs(lin_rec).max():.4f}")
    return lin_conv, lin_rec


# ── Run tests ────────────────────────────────────────────────────────

print("\n=== TEST 1: Sequential (16 tokens) ===", flush=True)
seq_conv, seq_rec = run_sequential(embed, m_infer)

print("\n=== TEST 2: Prefill with valid_len=16 ===", flush=True)
pf16_conv, pf16_rec, _ = run_prefill(embed, m_prefill, m_infer, 16, "vl=16")

print("\n=== TEST 3: Prefill with valid_len=512 ===", flush=True)
pf512_conv, pf512_rec, _ = run_prefill(embed, m_prefill, m_infer, 512, "vl=512")

print("\n=== TEST 4: Prefill with valid_len=1 ===", flush=True)
pf1_conv, pf1_rec, _ = run_prefill(embed, m_prefill, m_infer, 1, "vl=1")

# ── Compare ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("COMPARISONS")
print("="*60)

def compare(name, a, b, a_tag, b_tag):
    if a is None or b is None:
        print(f"  {name}: one or both are None")
        return
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    diff = np.abs(a32 - b32)
    identical = np.array_equal(a, b)
    print(f"\n  {name}: {a_tag} vs {b_tag}")
    print(f"    identical  = {identical}")
    print(f"    max_abs_diff = {diff.max():.6e}")
    print(f"    mean_abs_diff = {diff.mean():.6e}")

compare("conv_state", seq_conv, pf16_conv, "sequential", "prefill(vl=16)")
compare("conv_state", seq_conv, pf512_conv, "sequential", "prefill(vl=512)")
compare("conv_state", pf16_conv, pf512_conv, "prefill(vl=16)", "prefill(vl=512)")
compare("conv_state", pf16_conv, pf1_conv, "prefill(vl=16)", "prefill(vl=1)")

compare("rec_state",  seq_rec,  pf16_rec, "sequential", "prefill(vl=16)")
compare("rec_state",  seq_rec,  pf512_rec, "sequential", "prefill(vl=512)")
compare("rec_state",  pf16_rec, pf512_rec, "prefill(vl=16)", "prefill(vl=512)")
compare("rec_state",  pf16_conv, pf1_conv, "prefill(vl=16)", "prefill(vl=1)")

print("\nDone.", flush=True)
