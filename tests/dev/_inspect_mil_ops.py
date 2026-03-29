"""Inspect MIL op types and names in exported CoreML model chunks."""
import coremltools as ct
import os
import glob
from collections import Counter

model_dir = '/Users/yw68/Anemll/qwen3_5_stable_models_repro'
chunks = sorted(glob.glob(os.path.join(model_dir, 'ffn_LUT4_chunk*.mlpackage')))
if not chunks:
    print('No chunks found')
    exit(1)

print('Loading:', chunks[0])
m = ct.models.MLModel(chunks[0], compute_units=ct.ComputeUnit.CPU_ONLY)
spec = m.get_spec()
prog = spec.mlProgram
for fn_name in prog.functions:
    fn = prog.functions[fn_name]
    print(f'Function: {fn_name}')
    for block in fn.block_specializations:
        blk_spec = fn.block_specializations[block]
        ops = blk_spec.operations
        print(f'  Block {block}: {len(ops)} ops')
        op_types = Counter(op.type for op in ops)
        for t, c in op_types.most_common(30):
            print(f'    {t}: {c}')
        # Show ALL conv ops with output names (to identify projection names)
        print('\n  -- conv ops (showing weight names) --')
        for op in ops:
            if op.type == 'conv':
                out_name = op.outputs[0].name if op.outputs else '?'
                # Get input names
                input_names = []
                for inp_name in op.inputs:
                    inp = op.inputs[inp_name]
                    if hasattr(inp, 'name'):
                        input_names.append(f'{inp_name}={inp.name}')
                print(f'    conv -> {out_name}  inputs: {input_names}')
        # Show constexpr_lut_to_dense ops (quantized weights)
        print(f'\n  -- constexpr_lut_to_dense count: {op_types.get("constexpr_lut_to_dense", 0)} --')
        lut_ops = [op for op in ops if op.type == 'constexpr_lut_to_dense']
        for op in lut_ops[:20]:
            out_name = op.outputs[0].name if op.outputs else '?'
            print(f'    lut_to_dense -> {out_name}')
