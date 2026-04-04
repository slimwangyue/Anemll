#!/usr/bin/env python3
"""Export chunk0 for 5/6/7 chunk configs with LUT6 to find optimal chunk count."""
import sys, os, time, shutil, gc
sys.path.insert(0, '/Users/yw68/Anemll')

import torch, numpy as np, coremltools as ct
from collections import Counter
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

model_dir = '/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B'
output_base = '/Users/yw68/Anemll/tests/dev/ane_op_test_models'

# Which chunk configs to test (from CLI or default)
configs = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [5, 6, 7]

print("Loading model...", flush=True)
cfg = Qwen35Config.from_json(os.path.join(model_dir, "config.json"))
cfg.context_length = 2048
cfg.state_length = 2048
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(model_dir), "Failed to load weights"
model.eval()
for p in model.parameters():
    p.requires_grad = False

total_layers = len(model.model.layers)
print(f"Model loaded. Total layers: {total_layers}", flush=True)

for num_chunks in configs:
    base, rem = divmod(total_layers, num_chunks)
    # chunk0 layers
    layers_chunk0 = base + (1 if 0 < rem else 0)
    
    name = f"test_prefill_{num_chunks}chunks_LUT6"
    output_path = os.path.join(output_base, f"{name}.mlpackage")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    print(f"\n{'='*60}", flush=True)
    print(f"=== {num_chunks} chunks: chunk0 has {layers_chunk0} layers ===", flush=True)
    
    # Show all chunk sizes
    for ci in range(num_chunks):
        s = ci * base + min(ci, rem)
        e = s + base + (1 if ci < rem else 0)
        print(f"    chunk {ci}: layers {s}-{e-1} ({e-s} layers)", flush=True)
    
    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=2048, batch_size=512,
        num_chunks=num_chunks, lut_bits=6, per_channel=4,
    )
    
    try:
        mlmodel = conv.convert_part_2_prefill_exact(
            model, chunk_idx=0, total_chunks=num_chunks, exact_seq_len=512,
        )
    except Exception as e:
        print(f"  EXPORT FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        continue

    spec = mlmodel.get_spec()
    prog = spec.mlProgram
    op_counts = Counter()
    for func in prog.functions.values():
        for block in func.block_specializations.values():
            for op in block.operations:
                op_counts[op.type] += 1
    total_ops = sum(op_counts.values())

    mlmodel.save(output_path)
    dt = time.time() - t0
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fns in os.walk(output_path) for f in fns) / 1e6
    print(f"  Ops: {total_ops}, Size: {size_mb:.0f} MB, Time: {dt:.0f}s", flush=True)
    print(f"  LUT ops: {op_counts.get('constexpr_lut_to_dense', 0)}", flush=True)

    # Mac loading test
    for cu_name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
        t1 = time.time()
        try:
            m = ct.models.MLModel(output_path, compute_units=cu)
            print(f"  Mac {cu_name}: OK ({time.time()-t1:.1f}s)", flush=True)
            del m
        except Exception as e:
            print(f"  Mac {cu_name}: FAIL ({time.time()-t1:.1f}s) - {e}", flush=True)

    del mlmodel, conv, spec
    gc.collect()

print("\nDone.", flush=True)
