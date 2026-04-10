#!/usr/bin/env python3
"""Verify type-separated chunking: test 4 representative chunks on ANE."""
import os, sys, time, warnings, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts_gemma4'))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from config import *
from export import load_model
import coremltools as ct

print(f'NUM_CHUNKS={NUM_CHUNKS}')
for ci, (s,e) in enumerate(CHUNK_RANGES):
    print(f'  Chunk {ci}: layers {s}-{e-1}')

model = load_model(os.path.expanduser(DEFAULT_HF_MODEL), CTX)
model.model.config.force_rotation_mode = False
converter = Gemma4Converter(model=model, batch_size=BATCH_SIZE, context_length=CTX, lut_bits=None, per_channel=PER_CHANNEL, num_chunks=NUM_CHUNKS)

# Test chunks 0 (5 local), 1 (1 global), 2 (5 local), 8 (3 shared)
for ci in [0, 1, 2, 8]:
    start, end = CHUNK_RANGES[ci]
    layer_types = [model.model.config.layer_types[i] for i in range(start, end)]
    has_global = any(t == 'full_attention' for t in layer_types)
    
    print(f'\n=== Chunk {ci}: layers {start}-{end-1} (global={has_global}) ===')
    t0 = time.time()
    ml = converter.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS, force_rotation=False, start_layer=start, end_layer=end)
    
    spec = ml.get_spec()
    op_counts = {}
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                op_counts[op.type] = op_counts.get(op.type, 0) + 1
    total = sum(op_counts.values())
    print(f'  {total} ops in {time.time()-t0:.1f}s')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f'chunk{ci}.mlpackage')
        ml.save(path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            ane_fail = any('error code: -14' in str(x.message) for x in w)
            status = "FAIL: error -14" if ane_fail else "PASS: ANE OK!"
            print(f'  {status}')

print('\n=== All tests done ===')
