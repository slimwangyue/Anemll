#!/usr/bin/env python3
"""Analyze the 4L model structure."""
import coremltools as ct
from collections import Counter

model_path = '/Users/yw68/Anemll/tests/dev/ane_op_test_models/test_real_qwen_prefill_4L.mlpackage'
print('Loading 4L model with CPU_AND_NE...', flush=True)
model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
print('Loaded OK', flush=True)

spec = model.get_spec()
prog = spec.mlProgram

op_counts = Counter()
for func in prog.functions.values():
    for block in func.block_specializations.values():
        for op in block.operations:
            op_counts[op.type] += 1

total = sum(op_counts.values())
print(f'\nTotal ops: {total}')
print('\nTop ops:')
for op_type, count in op_counts.most_common(20):
    print(f'  {op_type}: {count}')

print(f'\nInputs: {len(spec.description.input)}')
for inp in spec.description.input:
    print(f'  {inp.name}')
print(f'Outputs: {len(spec.description.output)}')
for out in spec.description.output:
    print(f'  {out.name}')
print(f'States: {len(spec.description.state)}')
for st in spec.description.state:
    print(f'  {st.name}')
    
# Also check the model spec file size
import os
spec_path = os.path.join(model_path, 'Data', 'com.apple.CoreML', 'model.mlmodel')
if os.path.exists(spec_path):
    print(f'\nSpec file size: {os.path.getsize(spec_path) / 1e6:.1f} MB')
    
# Weight file sizes
weights_dir = os.path.join(model_path, 'Data', 'com.apple.CoreML', 'weights')
if os.path.isdir(weights_dir):
    total_w = sum(os.path.getsize(os.path.join(weights_dir, f)) for f in os.listdir(weights_dir))
    print(f'Weights total: {total_w / 1e6:.1f} MB')
