"""
Validate LUT quantization of deployed models.

Approach: Load model via coremltools, use _mil_program to inspect 
constexpr_lut_to_dense ops and read their lut.shape to determine nbits.
"""
import coremltools as ct
import os, math

bundle = '/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle'
models = ['chunk0', 'chunk1', 'chunk2', 'chunk3', 'embeddings', 'lm_head_logits']

for name in models:
    path = os.path.join(bundle, f'{name}.mlpackage')
    if not os.path.isdir(path):
        print(f"{name}: NOT FOUND")
        continue

    fn_name = 'infer' if name.startswith('chunk') else None
    try:
        kwargs = {'compute_units': ct.ComputeUnit.CPU_ONLY}
        if fn_name:
            kwargs['function_name'] = fn_name
        m = ct.models.MLModel(path, **kwargs)
    except Exception:
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)

    # Try _mil_program (private but reliable)
    mil_prog = getattr(m, '_mil_program', None)
    if mil_prog is None:
        print(f"{name}: cannot access MIL program")
        continue

    nbits_counts = {}
    for fn_obj in mil_prog.functions.values():
        for op in fn_obj.find_ops(op_type='constexpr_lut_to_dense'):
            lut = op.lut
            if hasattr(lut, 'shape'):
                shape = lut.shape
                # shape: (num_palettes, 1, 2^nbits, vector_size) or (1, 1, 2^nbits, nbytes)
                n_entries = shape[-2] if len(shape) >= 2 else shape[0]
                nbits = int(round(math.log2(n_entries)))
                nbits_counts[nbits] = nbits_counts.get(nbits, 0) + 1
        break  # only check one function

    if nbits_counts:
        parts = []
        for nb in sorted(nbits_counts):
            parts.append(f"LUT{nb}: {nbits_counts[nb]} weight tensors")
        print(f"{name}: {', '.join(parts)}")
    else:
        print(f"{name}: no LUT ops found (FP16)")
