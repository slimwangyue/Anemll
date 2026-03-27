"""Validate LUT bit-width by reading weight.bin metadata directly."""
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import os, json, struct

bundle = '/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle'

def check_nbits_from_weights(mlpackage_path):
    """Read Manifest.json to find weight blob, then decode constexpr_lut_to_dense metadata."""
    manifest_path = os.path.join(mlpackage_path, 'Manifest.json')
    if not os.path.exists(manifest_path):
        return None
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Find the model item
    for item in manifest.get('itemInfoEntries', {}).values():
        if item.get('path', '').endswith('.mlmodel'):
            model_path = os.path.join(mlpackage_path, 'Data', item['path'])
            if os.path.exists(model_path):
                return model_path
    return None


def check_nbits_via_load(path, fn_name=None):
    """Load model and check lut table sizes via spec inspection."""
    try:
        kwargs = {'compute_units': ct.ComputeUnit.CPU_ONLY}
        if fn_name:
            kwargs['function_name'] = fn_name
        m = ct.models.MLModel(path, **kwargs)
    except Exception:
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    
    spec = m.get_spec()
    prog = spec.mlProgram
    
    nbits_found = set()
    total_ops = 0
    
    for fn in prog.functions:
        func = prog.functions[fn]
        for block_name in func.block_specializations:
            block = func.block_specializations[block_name]
            for op in block.operations:
                if 'lut_to_dense' not in op.type:
                    continue
                total_ops += 1
                
                # Check the lut input — its type has shape info
                # The lut tensor shape: [num_palettes, 1, 2^nbits, vector_size]
                for inp_name in op.inputs:
                    if inp_name != 'lut':
                        continue
                    arg = op.inputs[inp_name]
                    for binding in arg.arguments:
                        if binding.HasField('name'):
                            # It's a reference to another op's output
                            # Find that op's output type
                            ref_name = binding.name
                            # Search for the op that produces this output
                            for other_op in block.operations:
                                for out in other_op.outputs:
                                    if out.name == ref_name:
                                        t = out.type
                                        if t.HasField('tensorType'):
                                            dims = []
                                            for d in t.tensorType.dimensions:
                                                if d.HasField('constant'):
                                                    dims.append(d.constant.size)
                                            if len(dims) >= 2:
                                                import math
                                                n_entries = dims[-2]
                                                nbits = int(round(math.log2(n_entries)))
                                                nbits_found.add(nbits)
                                            return nbits_found, total_ops
    
    return nbits_found, total_ops


models = ['chunk0', 'chunk1', 'chunk2', 'chunk3', 'embeddings', 'lm_head_logits']

for name in models:
    path = os.path.join(bundle, f'{name}.mlpackage')
    if not os.path.isdir(path):
        print(f"{name}: NOT FOUND")
        continue
    
    fn_name = 'infer' if name.startswith('chunk') else None
    nbits_found, total_ops = check_nbits_via_load(path, fn_name)
    
    if nbits_found:
        bits_str = ', '.join(f'LUT{n}' for n in sorted(nbits_found))
        print(f"{name}: {bits_str} ({total_ops} quantized weight ops)")
    elif total_ops > 0:
        print(f"{name}: {total_ops} LUT ops (nbits not resolved from spec)")
    else:
        print(f"{name}: FP16 (no LUT ops)")
