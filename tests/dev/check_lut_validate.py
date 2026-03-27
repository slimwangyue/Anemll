"""Validate LUT bit-width of deployed models using coremltools Python API."""
import coremltools as ct
import os, sys

bundle = '/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle'
models = ['chunk0', 'chunk1', 'chunk2', 'chunk3', 'embeddings', 'lm_head_logits']

for name in models:
    path = os.path.join(bundle, f'{name}.mlpackage')
    if not os.path.isdir(path):
        print(f"{name}: NOT FOUND")
        continue

    fn_name = None
    # Multi-function: specify function_name
    try:
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY, function_name='infer')
        fn_name = 'infer'
    except Exception:
        try:
            m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
            fn_name = 'main'
        except Exception as e:
            print(f"{name}: LOAD ERROR - {e}")
            continue

    # Use get_spec to count weight sizes and infer nbits from lut shape
    spec = m.get_spec()
    prog = spec.mlProgram
    
    # Iterate through all functions and blocks to find constexpr_lut_to_dense ops
    nbits_counts = {}
    total_lut_ops = 0
    
    for fn in prog.functions:
        func = prog.functions[fn]
        for block_name in func.block_specializations:
            block = func.block_specializations[block_name]
            for op in block.operations:
                if 'lut_to_dense' in op.type:
                    total_lut_ops += 1
                    # The lut input shape tells us nbits: lut has 2^nbits entries
                    # Check output type for shape info
                    for out in op.outputs:
                        t = out.type
                        if t.HasField('tensorType'):
                            dims = [d.constant.size for d in t.tensorType.dimensions]

    # Alternative: use the weight metadata from coremltools
    # The simplest way: check a weight's lut table size
    # 2^4 = 16 (LUT4), 2^6 = 64 (LUT6), 2^8 = 256 (LUT8)
    
    # Read the mlmodel weight file directly to determine nbits
    # For each constexpr_lut_to_dense, the lut_shape tells us nbits
    print(f"\n{name}.mlpackage (loaded as '{fn_name}'):")
    print(f"  constexpr_lut_to_dense ops: {total_lut_ops}")
    
    # Best approach: use mil program
    try:
        prog_mil = ct.models.utils.get_mil_internal(m)
        if prog_mil is not None:
            for fn_obj in prog_mil.functions.values():
                for op in fn_obj.find_ops(op_type='constexpr_lut_to_dense'):
                    lut_val = op.lut.val
                    if lut_val is not None:
                        lut_shape = lut_val.shape
                        # shape is typically (..., 2^nbits, vector_size)
                        # or (num_palettes, 1, 2^nbits, vector_size)
                        n_entries = lut_shape[-2] if len(lut_shape) >= 2 else lut_shape[0]
                        import math
                        nbits = int(math.log2(n_entries))
                        nbits_counts[nbits] = nbits_counts.get(nbits, 0) + 1
                break  # just check one function
    except Exception as e:
        print(f"  MIL inspection failed: {e}")
    
    if nbits_counts:
        for nbits, count in sorted(nbits_counts.items()):
            print(f"  LUT{nbits}: {count} weight tensors")
    else:
        if total_lut_ops > 0:
            print("  (could not extract nbits from MIL)")
        else:
            print("  No LUT quantization (FP16)")
