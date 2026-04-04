#!/usr/bin/env python3
"""Re-export prefill chunk 0 with chunk_size=16 (reduced from 64 to cut op count)."""
import sys
import os
import time

sys.path.insert(0, '/Users/yw68/Anemll')

import torch
import numpy as np
import coremltools as ct

# Set env for config
os.environ['QWEN35_NUM_CHUNKS'] = '4'
os.environ['QWEN35_HF_MODEL'] = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'
os.environ['QWEN35_STATIC_PREFILL_CTX'] = '2048'

from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

model_dir = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'
output_dir = '/Users/yw68/Anemll/qwen3_5_4chunk_lut6_bs512_ctx2048'

print("Loading model...", flush=True)
context_length = 2048
cfg = Qwen35Config.from_json(os.path.join(model_dir, "config.json"))
cfg.context_length = context_length
cfg.state_length = context_length
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(model_dir), f"Failed to load weights"
model.eval()
for p in model.parameters():
    p.requires_grad = False

chunk_index = 0
total_chunks = 4
bucket = 512
context_length = 2048

# Config
LUT_BITS = 6
FFN_PER_CHANNEL = 4
FFN_LABEL = f"LUT{LUT_BITS}"

prefill_path = os.path.join(
    output_dir,
    f"prefill_{FFN_LABEL}_chunk{chunk_index}_bs{bucket}.mlpackage",
)

print(f"Exporting chunk {chunk_index} prefill_bs{bucket} (chunk_size=16)...", flush=True)
t0 = time.time()
conv = Qwen35Converter(
    model,
    context_length=context_length,
    batch_size=bucket,
    num_chunks=total_chunks,
    lut_bits=LUT_BITS,
    per_channel=FFN_PER_CHANNEL,
)
mlmodel = conv.convert_part_2_prefill_exact(
    model,
    chunk_idx=chunk_index,
    total_chunks=total_chunks,
    exact_seq_len=bucket,
)

# Count ops before saving
spec = mlmodel.get_spec()
prog = spec.mlProgram
total_ops = 0
op_counts = {}
for func in prog.functions.values():
    for block in func.block_specializations.values():
        for op in block.operations:
            total_ops += 1
            op_type = op.type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

print(f"\nTotal ops: {total_ops}")
for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"  {op}: {cnt}")

# Save - backup old first
import shutil
backup_path = prefill_path + ".bak_cs64"
if os.path.exists(prefill_path) and not os.path.exists(backup_path):
    print(f"\nBacking up old model to {backup_path}")
    shutil.move(prefill_path, backup_path)
elif os.path.exists(prefill_path):
    shutil.rmtree(prefill_path)

mlmodel.save(prefill_path)
dt = time.time() - t0
print(f"\nSaved to {prefill_path} ({dt:.1f}s)")
print(f"Size: {sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(prefill_path) for f in fns) / 1e6:.1f} MB")
