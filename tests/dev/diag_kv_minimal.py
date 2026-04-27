#!/usr/bin/env python3
"""Minimal KV cache test: load only chunk1 (has Full attention at layer 3).
Designed to work with limited boot drive space (~1.5GB).
"""
import sys, os, numpy as np, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import coremltools as ct

MODEL_DIR = '/Volumes/MySSD/Anemll/qwen3_5_4B_milestone_3.4_fix'
HF_PATH = '/Volumes/MySSD/Anemll/models/Qwen__Qwen3.5-4B'
COMPUTE = ct.ComputeUnit.CPU_AND_NE
BS = 256  # batch prefill size
CTX = 4096

def find_model(base_dir, name):
    """Find model, preferring .mlmodelc."""
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model for {name} in {base_dir}")

def load_model(path, cu):
    """Load model from .mlmodelc or .mlpackage."""
    # Clean stale temps first
    import glob, shutil
    sys_tmp = '/var/folders/r5/dn4v9jhx1cvg33xnvt8nmjxr0000gn/T'
    for p in glob.glob(os.path.join(sys_tmp, "*.mlmodelc")):
        try: shutil.rmtree(p)
        except: pass
    for p in glob.glob(os.path.join(sys_tmp, "TemporaryItems", "NSIRD_Python_*")):
        try: shutil.rmtree(p)
        except: pass

    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, cu)
    return ct.models.MLModel(path, compute_units=cu)

# Only load what we need: embed_prefill + chunk1 prefill
print("Loading embed_prefill...", flush=True)
embed_pf = load_model(find_model(MODEL_DIR, "embed_prefill"), COMPUTE)
print(f"  OK", flush=True)

# Load chunk1 (layers 3-6: 1 Full + 3 Linear, has k_cache/v_cache)
CHUNK_IDX = 1
print(f"Loading prefill chunk {CHUNK_IDX}...", flush=True)
pf_path = find_model(MODEL_DIR, f"prefill_LUT4_chunk{CHUNK_IDX}")
print(f"  path: {pf_path}")
t0 = time.time()
prefill = load_model(pf_path, COMPUTE)
print(f"  OK ({time.time()-t0:.1f}s)", flush=True)

# Create state and detect shapes
state = prefill.make_state()

# Find state names
state_names = []
spec_path = pf_path.replace('.mlmodelc', '.mlpackage')
if os.path.exists(spec_path):
    spec = ct.utils.load_spec(spec_path)
    fn = spec.mlProgram.functions.get('main')
    if fn:
        block = list(fn.block_specializations.values())[0]
        for op in block.operations:
            if op.type == 'read_state':
                for o in op.outputs:
                    state_names.append(o.name.replace('read_state_', ''))
# Default KV state names
kv_names = ['k_cache', 'v_cache']
print(f"KV state names: {kv_names}", flush=True)

# Get hidden dim from embed
test_ids = np.zeros((1, BS), dtype=np.int32)
embed_out = list(embed_pf.predict({'input_ids': test_ids}).values())[0]
hidden_dim = embed_out.shape[-1]
print(f"hidden_dim={hidden_dim}, embed shape={embed_out.shape}", flush=True)

# Create block1 input (position 0..BS-1)
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(os.path.join(HF_PATH, 'tokenizer.json'))
prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
paragraph = 'Explain general relativity in simple terms. How does mass curve spacetime? '
while len(tokenizer.encode(prompt).ids) < BS + 50:
    prompt += paragraph
prompt += '<|im_end|>\n<|im_start|>assistant\n'
all_ids = tokenizer.encode(prompt).ids
block1_ids = all_ids[:BS]
block2_ids = all_ids[BS:BS+50]
valid_len2 = len(block2_ids)
print(f"block1={len(block1_ids)} tokens, block2={valid_len2} tokens", flush=True)

# Embed block1
input_ids = np.zeros((1, BS), dtype=np.int32)
input_ids[0, :BS] = block1_ids
hidden = list(embed_pf.predict({'input_ids': input_ids}).values())[0]

# Build mask for block1
mask = np.full((1, 1, BS, CTX), -65504.0, dtype=np.float16)
for i in range(BS):
    mask[0, 0, i, :i+1] = 0

pos_ids = np.arange(0, BS, dtype=np.int32)
cur_pos = np.zeros(1, dtype=np.int32)
vl = np.array([BS], dtype=np.int32)

# Conv/rec state shapes from chunk1: conv=(4, 1024, 32), rec=(4, 32, 128, 128)
num_layers_in_chunk = 4
conv_shape = (num_layers_in_chunk, 1024, 32)  # from detect_shapes output
rec_shape = (num_layers_in_chunk, 32, 128, 128)
lin_conv = np.zeros(conv_shape, dtype=np.float16)
lin_rec = np.zeros(rec_shape, dtype=np.float16)

print("\nRunning block1 prefill on chunk1...", flush=True)
t0 = time.time()
inp = {
    'hidden_states': hidden.astype(np.float16),
    'position_ids': pos_ids,
    'causal_mask': mask,
    'current_pos': cur_pos,
    'linear_conv_state': lin_conv,
    'linear_recurrent_state': lin_rec,
    'valid_len': vl,
}
out = prefill.predict(inp, state=state)
print(f"  done ({time.time()-t0:.1f}s)", flush=True)

# Snapshot KV after block1
kv_b1 = {}
for sn in kv_names:
    kv_b1[sn] = state.read_state(name=sn).copy()
    print(f"  {sn} shape={kv_b1[sn].shape}, dtype={kv_b1[sn].dtype}", flush=True)
    # Show non-zero pattern
    nz = np.count_nonzero(kv_b1[sn])
    print(f"  {sn} non-zero elements: {nz}/{kv_b1[sn].size}", flush=True)

# Block2 prefill (valid_len2 tokens at position BS..BS+valid_len2-1)
input_ids2 = np.zeros((1, BS), dtype=np.int32)
input_ids2[0, :valid_len2] = block2_ids
hidden2 = list(embed_pf.predict({'input_ids': input_ids2}).values())[0]
hidden2[:, valid_len2:, :] = 0.0

mask2 = np.full((1, 1, BS, CTX), -65504.0, dtype=np.float16)
for i in range(valid_len2):
    mask2[0, 0, i, :BS+i+1] = 0
for i in range(valid_len2, BS):
    mask2[0, 0, i, 0] = 0.0

pos_ids2 = np.zeros(BS, dtype=np.int32)
pos_ids2[:valid_len2] = np.arange(BS, BS+valid_len2, dtype=np.int32)
cur_pos2 = np.array([BS], dtype=np.int32)
vl2 = np.array([valid_len2], dtype=np.int32)

lin_conv2 = out.get('linear_conv_state_out', lin_conv)
lin_rec2 = out.get('linear_recurrent_state_out', lin_rec)

print(f"\nRunning block2 prefill on chunk1 ({valid_len2} valid tokens)...", flush=True)
t0 = time.time()
inp2 = {
    'hidden_states': hidden2.astype(np.float16),
    'position_ids': pos_ids2,
    'causal_mask': mask2,
    'current_pos': cur_pos2,
    'linear_conv_state': lin_conv2,
    'linear_recurrent_state': lin_rec2,
    'valid_len': vl2,
}
out2 = prefill.predict(inp2, state=state)
print(f"  done ({time.time()-t0:.1f}s)", flush=True)

# Check: did block2 preserve block1's KV values (positions 0..BS-1)?
print()
print("=" * 60)
print("CHECK: Did block2 preserve block1 KV (positions 0-255)?")
print("=" * 60)
all_ok = True
for sn in kv_names:
    before = kv_b1[sn]
    after = state.read_state(name=sn)
    # Compare positions 0..BS-1 (block1 wrote here)
    if before.ndim == 4:
        # shape [num_heads, head_dim, ctx, 1] or similar
        # Find the dimension that has CTX
        for dim in range(before.ndim):
            if before.shape[dim] == CTX:
                break
        slc_b = [slice(None)] * before.ndim
        slc_b[dim] = slice(0, BS)
        slc_a = [slice(None)] * after.ndim
        slc_a[dim] = slice(0, BS)
        b = before[tuple(slc_b)]
        a = after[tuple(slc_a)]
    else:
        b, a = before, after

    eq = np.array_equal(b, a)
    if eq:
        print(f"  chunk{CHUNK_IDX} {sn}: OK (block1 values preserved)")
    else:
        all_ok = False
        d = np.abs(b.astype(np.float64) - a.astype(np.float64))
        changed = np.count_nonzero(d > 0)
        print(f"  chunk{CHUNK_IDX} {sn}: MODIFIED! max_diff={d.max():.6f} changed={changed}/{b.size}")

# Check: did block2 write to positions BS..BS+valid_len2-1?
print()
print("=" * 60)
print(f"CHECK: Did block2 write new KV (positions {BS}-{BS+valid_len2-1})?")
print("=" * 60)
for sn in kv_names:
    after = state.read_state(name=sn)
    if after.ndim == 4:
        for dim in range(after.ndim):
            if after.shape[dim] == CTX:
                break
        slc = [slice(None)] * after.ndim
        slc[dim] = slice(BS, BS + valid_len2)
        block2_kv = after[tuple(slc)]
    else:
        block2_kv = after

    nz = np.count_nonzero(block2_kv)
    total = block2_kv.size
    if nz > 0:
        print(f"  chunk{CHUNK_IDX} {sn}: OK ({nz}/{total} non-zero at block2 positions)")
    else:
        all_ok = False
        print(f"  chunk{CHUNK_IDX} {sn}: EMPTY! Block2 KV not written to correct positions!")

# Check: are positions BS+valid_len2..CTX-1 still zero?
print()
print("=" * 60)
print(f"CHECK: Positions {BS+valid_len2}-{CTX-1} still zero (untouched)?")
print("=" * 60)
for sn in kv_names:
    after = state.read_state(name=sn)
    if after.ndim == 4:
        for dim in range(after.ndim):
            if after.shape[dim] == CTX:
                break
        slc = [slice(None)] * after.ndim
        slc[dim] = slice(BS + valid_len2, CTX)
        rest_kv = after[tuple(slc)]
    else:
        rest_kv = after

    nz = np.count_nonzero(rest_kv)
    if nz == 0:
        print(f"  chunk{CHUNK_IDX} {sn}: OK (zero)")
    else:
        all_ok = False
        print(f"  chunk{CHUNK_IDX} {sn}: CONTAMINATED! {nz} non-zero values in unused region!")

print()
if all_ok:
    print("PASS: KV cache fix verified!")
else:
    print("FAIL: KV cache bug still present!")
