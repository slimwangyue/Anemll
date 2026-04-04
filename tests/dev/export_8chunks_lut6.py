#!/usr/bin/env python3
"""Export all 8 LUT6 prefill chunks (4 layers each) for iPhone ANE production."""
import sys, os, gc, time, traceback, shutil

sys.path.insert(0, '/Users/yw68/Anemll')
sys.path.insert(0, '/Users/yw68/Anemll/scripts_qwen3_5')

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from collections import Counter

model_dir = '/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B'
output_dir = '/Users/yw68/Anemll/qwen3_5_8chunk_lut6_bs512_ctx2048'
context_length = 2048
bucket = 512
num_chunks = 8
lut_bits = 6
per_channel = 4

# Parse args
start_chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else num_chunks
skip_existing = '--skip-existing' in sys.argv

os.makedirs(output_dir, exist_ok=True)

print(f'=== Exporting {num_chunks} LUT6 prefill chunks ({32//num_chunks}L each) ===')
print(f'  Output: {output_dir}')
print(f'  LUT6 bits={lut_bits}, per_channel={per_channel}')
print(f'  Chunks: {start_chunk} to {end_chunk-1}')
print(f'Loading model...', flush=True)

cfg = Qwen35Config.from_json(os.path.join(model_dir, 'config.json'))
cfg.context_length = context_length
cfg.state_length = context_length
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(model_dir), "Failed to load weights"
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f'Model loaded. Total layers: {len(model.model.layers)}', flush=True)

total_time = 0
for ci in range(start_chunk, end_chunk):
    out_path = os.path.join(output_dir, f'prefill_LUT6_chunk{ci}_bs{bucket}.mlpackage')
    if skip_existing and os.path.exists(out_path):
        print(f'\n[skip] chunk {ci} (already exists at {out_path})')
        continue
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    print(f'\n{"="*60}', flush=True)
    print(f'=== Chunk {ci}/{num_chunks} ===', flush=True)
    t0 = time.time()

    conv = Qwen35Converter(
        model, context_length=context_length, batch_size=bucket,
        num_chunks=num_chunks, lut_bits=lut_bits, per_channel=per_channel,
    )
    try:
        mlmodel = conv.convert_part_2_prefill_exact(
            model, chunk_idx=ci, total_chunks=num_chunks, exact_seq_len=bucket,
        )
    except Exception as e:
        print(f'  EXPORT FAILED chunk {ci}: {e}', flush=True)
        traceback.print_exc()
        continue

    # Count ops
    spec = mlmodel.get_spec()
    prog = spec.mlProgram
    op_counts = Counter()
    for func in prog.functions.values():
        for block in func.block_specializations.values():
            for op in block.operations:
                op_counts[op.type] += 1
    total_ops = sum(op_counts.values())

    mlmodel.save(out_path)
    dt = time.time() - t0
    total_time += dt
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fns in os.walk(out_path) for f in fns) / 1e6
    print(f'  Ops: {total_ops}, Size: {size_mb:.0f} MB, Time: {dt:.0f}s', flush=True)
    print(f'  LUT ops: {op_counts.get("constexpr_lut_to_dense", 0)}', flush=True)

    del mlmodel, conv, spec
    gc.collect()

print(f'\n{"="*60}')
print(f'All done! Total time: {total_time:.0f}s ({total_time/60:.1f} min)')
print(f'Output: {output_dir}')

# List exported files
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.mlpackage'):
        path = os.path.join(output_dir, f)
        size = sum(os.path.getsize(os.path.join(dp, fn))
                   for dp, _, fns in os.walk(path) for fn in fns) / 1e6
        print(f'  {f}: {size:.0f} MB')
