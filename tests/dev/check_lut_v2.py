"""
Validate LUT quantization by converting proto spec to MIL program.
"""
import coremltools as ct
from coremltools.converters.mil.frontend.milproto.load import load as milproto_load
import os, math

bundle = '/Users/yw68/Anemll/qwen3_5_stable_models'
models = ['prefill_LUT4_chunk0', 'prefill_LUT4_chunk1', 'prefill_LUT4_chunk2']

for name in models:
    path = os.path.join(bundle, f'{name}.mlpackage')
    spec = ct.utils.load_spec(path)
    spec_ver = spec.specificationVersion
    
    # Find weights directory
    weights_dir = os.path.join(path, 'Data', 'com.apple.CoreML', 'weights')
    if not os.path.isdir(weights_dir):
        weights_dir = ''
    
    try:
        prog = milproto_load(spec, spec_ver, file_weights_dir=weights_dir)
    except Exception as e:
        print(f"{name}: MIL load error: {e}")
        continue
    
    print(f"\n{name}:")
    for fn_name, fn_obj in prog.functions.items():
        ops = fn_obj.find_ops(op_type='constexpr_lut_to_dense')
        nbits_counts = {}
        for op in ops:
            lut = op.lut
            shape = tuple(lut.shape)
            # shape: (num_palettes, 1, 2^nbits, vector_size)
            n_entries = shape[-2] if len(shape) >= 2 else shape[0]
            nbits = int(round(math.log2(n_entries)))
            nbits_counts[nbits] = nbits_counts.get(nbits, 0) + 1

        if nbits_counts:
            for nb in sorted(nbits_counts):
                print(f"  {fn_name}: LUT{nb} x {nbits_counts[nb]} weights")
        else:
            print(f"  {fn_name}: FP16 (no LUT)")
