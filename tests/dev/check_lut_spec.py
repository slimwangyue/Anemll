"""
Validate LUT quantization by loading the spec and converting to MIL.
"""
import coremltools as ct
from coremltools.converters.mil.frontend.milproto import load as milproto_load
import os, math

bundle = '/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle'

# Just check chunk0 first
path = os.path.join(bundle, 'chunk0.mlpackage')
spec = ct.utils.load_spec(path)
print(f"Spec type: {spec.WhichOneof('Type')}")

# Convert spec to MIL program
prog = milproto_load(spec)
print(f"MIL program functions: {list(prog.functions.keys())}")

for fn_name, fn_obj in prog.functions.items():
    ops = fn_obj.find_ops(op_type='constexpr_lut_to_dense')
    nbits_counts = {}
    for op in ops:
        lut = op.lut
        shape = lut.shape
        n_entries = shape[-2] if len(shape) >= 2 else shape[0]
        nbits = int(round(math.log2(n_entries)))
        nbits_counts[nbits] = nbits_counts.get(nbits, 0) + 1

    if nbits_counts:
        for nb in sorted(nbits_counts):
            print(f"  {fn_name}: LUT{nb} x {nbits_counts[nb]} weight tensors")
    else:
        print(f"  {fn_name}: no LUT ops")
    break  # just first function
