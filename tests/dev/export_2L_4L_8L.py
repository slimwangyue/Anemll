#!/usr/bin/env python3
"""Export 2L and 4L real Qwen3.5 prefill chunks to verify the State identity fix
and measure op counts for different chunk configurations."""
import sys, os, time, shutil
sys.path.insert(0, '/Users/yw68/Anemll')
os.environ['QWEN35_NUM_CHUNKS'] = '4'
os.environ['QWEN35_HF_MODEL'] = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'
os.environ['QWEN35_STATIC_PREFILL_CTX'] = '2048'

import torch, numpy as np, coremltools as ct
from collections import Counter
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

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

total_model_layers = len(model.model.layers)
print(f"Model loaded. Total layers: {total_model_layers}", flush=True)

# Test 2L, 4L, and 8L chunks
for target_layers in [2, 4, 8]:
    total_chunks = total_model_layers // target_layers
    base, rem = divmod(total_model_layers, total_chunks)
    actual_layers = base + (1 if 0 < rem else 0)
    
    name = f"test_real_qwen_prefill_{actual_layers}L"
    output_path = os.path.join(output_base, f"{name}.mlpackage")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    print(f"\n{'='*60}", flush=True)
    print(f"=== {actual_layers}-layer prefill (no LUT6) [total_chunks={total_chunks}] ===", flush=True)
    t0 = time.time()

    conv = Qwen35Converter(
        model,
        context_length=2048,
        batch_size=512,
        num_chunks=total_chunks,
        lut_bits=0,
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
        import traceback
        traceback.print_exc()
        continue

    spec = mlmodel.get_spec()
    prog = spec.mlProgram
    op_counts = Counter()
    for func in prog.functions.values():
        for block in func.block_specializations.values():
            for op in block.operations:
                op_counts[op.type] += 1

    total_ops = sum(op_counts.values())
    dt = time.time() - t0
    print(f"  Ops: {total_ops}, Time: {dt:.0f}s", flush=True)
    print(f"  Top ops:", flush=True)
    for op_type, count in op_counts.most_common(10):
        print(f"    {op_type}: {count}", flush=True)

    mlmodel.save(output_path)
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fns in os.walk(output_path) for f in fns) / 1e6
    print(f"  Saved: {size_mb:.0f} MB", flush=True)

    # Mac loading test
    for cu_name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)]:
        t0 = time.time()
        try:
            m = ct.models.MLModel(output_path, compute_units=cu)
            print(f"  Mac {cu_name}: OK ({time.time()-t0:.1f}s)", flush=True)
            del m
        except Exception as e:
            print(f"  Mac {cu_name}: FAIL ({time.time()-t0:.1f}s) - {e}", flush=True)

print("\nDone.", flush=True)
