#!/usr/bin/env python3
import coremltools as ct
import sys

spec = ct.utils.load_spec('/Users/yw68/Anemll/qwen3_5_6chunk_models/lm_head.mlpackage')
for i in spec.description.input:
    print(f'IN: {i.name} shape={list(i.type.multiArrayType.shape)}')
for o in spec.description.output:
    print(f'OUT: {o.name} shape={list(o.type.multiArrayType.shape)}')

# Also check last chunk prefill output
spec2 = ct.utils.load_spec('/Users/yw68/Anemll/qwen3_5_6chunk_models/combined_LUT4_dedup/chunk5.mlpackage')
fns = list(spec2.description.functions.functions.keys()) if hasattr(spec2.description, 'functions') and spec2.description.functions.functions else []
print(f'\nChunk5 functions: {fns}')

# Check the prefill function of the last chunk
if hasattr(spec2, 'description') and hasattr(spec2.description, 'functions'):
    for fname, fdesc in spec2.description.functions.functions.items():
        if fname == 'prefill':
            for o in fdesc.description.output:
                print(f'  Prefill OUT: {o.name} shape={list(o.type.multiArrayType.shape)}')
sys.stdout.flush()
