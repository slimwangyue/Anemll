#!/usr/bin/env python3
"""Parity check: milestone1 vs milestone1_3 (subprocess-isolated).

Runs each model set in a separate subprocess to avoid OOM,
saves tokens to temp files, then compares.
"""
import sys, os, json, subprocess, tempfile

TOKENIZER_DIR = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
M1_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1"
M13_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"

WORKER_SCRIPT = r'''
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

model_dir = sys.argv[1]
out_file = sys.argv[2]
tokenizer_dir = sys.argv[3]
max_tokens = int(sys.argv[4])

CTX = 1024
NUM_CHUNKS = 4
LABEL = "LUT4"

cu = ct.ComputeUnit.CPU_AND_NE
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

print(f"Loading models from {model_dir}...")
t0 = time.time()
embed = ct.models.MLModel(os.path.join(model_dir, "embeddings.mlpackage"), compute_units=cu)
lmhead = ct.models.MLModel(os.path.join(model_dir, "lm_head.mlpackage"), compute_units=cu)
ffns = []
for ci in range(NUM_CHUNKS):
    ffns.append(ct.models.MLModel(
        os.path.join(model_dir, f"ffn_{LABEL}_chunk{ci}.mlpackage"), compute_units=cu))
print(f"  Loaded in {time.time()-t0:.0f}s")

spec = ffns[0].get_spec()
inp_map = {}
for inp in spec.description.input:
    try:
        inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
    except Exception:
        pass
has_linear = 'linear_conv_state' in inp_map
states = [m.make_state() for m in ffns]
if has_linear:
    lin_convs = [np.zeros(inp_map['linear_conv_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
    lin_recs = [np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
else:
    lin_convs = [None] * NUM_CHUNKS
    lin_recs = [None] * NUM_CHUNKS

def step(tok_id, pos):
    tok = np.array([[tok_id]], dtype=np.int32)
    hidden = list(embed.predict({"input_ids": tok}).values())[0]
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :pos + 1] = 0
    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
        }
        if lin_convs[ci] is not None:
            inp["linear_conv_state"] = lin_convs[ci]
            inp["linear_recurrent_state"] = lin_recs[ci]
        out = ffns[ci].predict(inp, state=states[ci])
        hidden = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            lin_convs[ci] = out['linear_conv_state_out']
            lin_recs[ci] = out['linear_recurrent_state_out']
    lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if "logits" in lm_out:
        return int(np.argmax(lm_out["logits"].flatten()))
    return int(lm_out["argmax_idx"].flatten()[0])

stop_ids = set()
for sid in ["<|im_end|>", "<|endoftext|>"]:
    tids = tokenizer.encode(sid, add_special_tokens=False)
    if tids:
        stop_ids.add(tids[0])

prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt)
print(f"Prompt: {len(input_ids)} tokens, generating {max_tokens}...")

t0 = time.time()
last = None
for i, tid in enumerate(input_ids):
    last = step(tid, i)
    if (i+1) % 5 == 0:
        print(f"  prefill {i+1}/{len(input_ids)}")

generated = [last]
pos = len(input_ids)
for gi in range(max_tokens - 1):
    if pos >= CTX - 1:
        break
    nxt = step(generated[-1], pos)
    generated.append(nxt)
    pos += 1
    if nxt in stop_ids:
        break
    if (gi+1) % 5 == 0:
        print(f"  decode {gi+1}/{max_tokens}")

elapsed = time.time() - t0
text = tokenizer.decode(generated, skip_special_tokens=True)
print(f"Done: {len(generated)} tokens in {elapsed:.1f}s ({len(generated)/elapsed:.1f} tok/s)")
print(f"Output: {text}")

with open(out_file, 'w') as f:
    json.dump({"tokens": generated, "text": text}, f)
print(f"Saved to {out_file}")
'''


def run_worker(model_dir, out_file, max_tokens=30):
    """Run generation in a subprocess."""
    worker_path = "/tmp/_parity_worker.py"
    with open(worker_path, 'w') as f:
        f.write(WORKER_SCRIPT)
    
    cmd = [
        sys.executable, worker_path,
        model_dir, out_file, TOKENIZER_DIR, str(max_tokens)
    ]
    print(f"\n{'='*60}")
    print(f"Running: {model_dir}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=False)
    max_tokens = 30

    m1_out = "/tmp/parity_m1.json"
    m13_out = "/tmp/parity_m13.json"

    # Run M1
    rc = run_worker(M1_DIR, m1_out, max_tokens)
    if rc != 0:
        print(f"ERROR: M1 worker exited with code {rc}")
        sys.exit(1)

    # Run M1.3
    rc = run_worker(M13_DIR, m13_out, max_tokens)
    if rc != 0:
        print(f"ERROR: M1.3 worker exited with code {rc}")
        sys.exit(1)

    # Compare
    with open(m1_out) as f:
        m1 = json.load(f)
    with open(m13_out) as f:
        m13 = json.load(f)

    m1_tokens = m1["tokens"]
    m13_tokens = m13["tokens"]

    print(f"\n{'='*70}")
    print(f"  PARITY: M1 (static) vs M1.3 (tensor-value slice)")
    print(f"{'='*70}")
    print(f"  M1  output: {m1['text']}")
    print(f"  M1.3 output: {m13['text']}")

    total = min(len(m1_tokens), len(m13_tokens))
    matches = sum(1 for a, b in zip(m1_tokens, m13_tokens) if a == b)
    pct = 100 * matches / total if total > 0 else 0

    print(f"\n  Token match: {matches}/{total} ({pct:.1f}%)")

    diverge_pos = -1
    for i in range(total):
        if m1_tokens[i] != m13_tokens[i]:
            if diverge_pos < 0:
                diverge_pos = i
            t1 = tokenizer.decode([m1_tokens[i]])
            t13 = tokenizer.decode([m13_tokens[i]])
            print(f"    pos {i}: M1={m1_tokens[i]} '{t1}' vs M1.3={m13_tokens[i]} '{t13}'")

    if diverge_pos < 0:
        print(f"\n  100% MATCH — PERFECT PARITY")
    else:
        print(f"\n  First divergence at position {diverge_pos}")
        if pct >= 90:
            print(f"  {pct:.0f}% match — acceptable for fp16 quantized models")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
