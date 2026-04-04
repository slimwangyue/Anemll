#!/usr/bin/env python3
"""Export a single prefill chunk for debugging."""
import sys, os, gc, time, traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'scripts_qwen3_5'))

from config import FFN_LABEL, LUT_BITS, FFN_PER_CHANNEL, PER_CHANNEL
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

model_dir = '/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B'
output_dir = '/Users/yw68/Anemll/qwen3_5_4chunk_lut6_bs512_ctx2048'
context_length = 2048
bucket = 512
num_chunks = 4
chunk_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
all_chunks = len(sys.argv) > 2 and sys.argv[2] == '--all'

print(f'Loading model...')
cfg = Qwen35Config.from_json(os.path.join(model_dir, 'config.json'))
cfg.context_length = context_length
cfg.state_length = context_length
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(model_dir)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f'Loaded')

chunks_to_export = range(num_chunks) if all_chunks else [chunk_index]
for ci in chunks_to_export:
    out_path = os.path.join(output_dir, f'prefill_{FFN_LABEL}_chunk{ci}_bs{bucket}.mlpackage')
    if os.path.exists(out_path):
        print(f'  [skip] chunk {ci} (already exists)')
        continue
    print(f'Exporting chunk {ci} prefill_bs{bucket}...')
    conv = Qwen35Converter(
        model, context_length=context_length, batch_size=bucket,
        num_chunks=num_chunks, lut_bits=LUT_BITS, per_channel=FFN_PER_CHANNEL,
    )
    try:
        mlmodel = conv.convert_part_2_prefill_exact(
            model, chunk_idx=ci, total_chunks=num_chunks, exact_seq_len=bucket,
        )
        mlmodel.save(out_path)
        print(f'Saved to {out_path}')
        del mlmodel, conv
        gc.collect()
    except Exception as e:
        traceback.print_exc()
        print(f'FAILED chunk {ci}: {e}')
print('Done.')
