#!/usr/bin/env python3
"""Quick test: export 2-layer, 4-layer, 9-layer prefill chunks using the actual converter.
Skips LUT6 to be fast. Just checks Mac load (predict) status."""
import sys
import os
import time
import shutil

sys.path.insert(0, '/Users/yw68/Anemll')
os.environ['QWEN35_NUM_CHUNKS'] = '4'
os.environ['QWEN35_HF_MODEL'] = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'
os.environ['QWEN35_STATIC_PREFILL_CTX'] = '2048'

import torch
import numpy as np
import coremltools as ct

from anemll.ane_converter.qwen3_5_converter import Qwen35Converter, ane_conv_state_shape
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM, MODEL_DTYPE, TEST_DEVICE

model_dir = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'
output_base = '/Users/yw68/Anemll/tests/dev/ane_op_test_models'

print("Loading model...", flush=True)
cfg = Qwen35Config.from_json(os.path.join(model_dir, "config.json"))
cfg.context_length = 2048
cfg.state_length = 2048
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(model_dir), "Failed to load weights"
model.eval()
for p in model.parameters():
    p.requires_grad = False
print("Model loaded.", flush=True)

total_model_layers = len(model.model.layers)
print(f"Total model layers: {total_model_layers}")

# Test configs: (target_layers, total_chunks) 
# For chunk_idx=0: start=0, end = base + (1 if 0 < rem else 0) where base,rem = divmod(total, chunks)
test_configs = []
for target_layers in [1, 2, 4, 9]:
    # We want chunk 0 to have exactly target_layers
    # base = total_model_layers // total_chunks, rem = total_model_layers % total_chunks
    # end_layer for chunk 0 = base + (1 if 0 < rem else 0)
    # If rem > 0: end_layer = base + 1 = target_layers -> base = target_layers - 1 -> total_chunks = total_model_layers / (target_layers - 1)
    # If rem = 0: end_layer = base = target_layers -> total_chunks = total_model_layers / target_layers
    total_chunks = total_model_layers // target_layers
    base, rem = divmod(total_model_layers, total_chunks)
    actual_layers = base + (1 if 0 < rem else 0)
    test_configs.append((target_layers, total_chunks, actual_layers))
    print(f"  target={target_layers}: total_chunks={total_chunks}, actual chunk0 layers={actual_layers}")

for target_layers, total_chunks, actual_layers in test_configs:
    name = f"test_real_qwen_prefill_{actual_layers}L"
    output_path = os.path.join(output_base, f"{name}.mlpackage")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    print(f"\n=== {actual_layers}-layer prefill (no LUT6) ===", flush=True)
    t0 = time.time()

    # Temporarily override num_hidden_layers for export
    conv = Qwen35Converter(
        model,
        context_length=2048,
        batch_size=512,
        num_chunks=total_chunks,
        lut_bits=0,    # NO LUT6 for speed
        per_channel=4,
    )
    try:
        mlmodel = conv.convert_part_2_prefill_exact(
            model,
            chunk_idx=0,
            total_chunks=total_chunks,
            exact_seq_len=512,
            block_start=0,
        )
    except Exception as e:
        print(f"  EXPORT FAILED: {e}", flush=True)
        # Override total_layers manually  
        # Try a different approach: set end_layer explicitly
        continue

    spec = mlmodel.get_spec()
    prog = spec.mlProgram
    total_ops = sum(1 for func in prog.functions.values()
                    for block in func.block_specializations.values()
                    for op in block.operations)
    dt = time.time() - t0
    print(f"  Ops: {total_ops}, Time: {dt:.1f}s", flush=True)

    mlmodel.save(output_path)
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fns in os.walk(output_path) for f in fns) / 1e6
    print(f"  Saved: {size_mb:.1f} MB", flush=True)

print("\nDone.", flush=True)
