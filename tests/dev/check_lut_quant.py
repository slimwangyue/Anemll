"""Check LUT quantization bit-width of deployed models."""
import coremltools as ct
import sys

bundle = '/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle'
models = ['chunk0', 'chunk1', 'chunk2', 'chunk3', 'embeddings', 'lm_head_logits']

for name in models:
    path = f'{bundle}/{name}.mlpackage'
    try:
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = m.get_spec()
        fns = list(spec.mlProgram.functions) if spec.WhichOneof('Type') == 'mlProgram' else ['?']

        # Scan ops for constexpr_lut_to_dense
        lut_counts = {}  # nbits -> count
        other_constexpr = {}
        for fn_name in fns:
            fn = spec.mlProgram.functions[fn_name]
            for block in fn.block_specializations.values():
                for op in block.operations:
                    if 'lut_to_dense' in op.type:
                        # Extract nbits from attributes
                        nbits = '?'
                        for attr_name in op.attributes:
                            if 'bit' in attr_name.lower() or attr_name == 'nbits':
                                attr_val = op.attributes[attr_name]
                                if attr_val.HasField('immediateValue'):
                                    iv = attr_val.immediateValue
                                    if iv.HasField('integer'):
                                        nbits = iv.integer.value
                        lut_counts[nbits] = lut_counts.get(nbits, 0) + 1
                    elif 'constexpr' in op.type:
                        other_constexpr[op.type] = other_constexpr.get(op.type, 0) + 1

        print(f"\n{name}.mlpackage:")
        print(f"  functions: {fns}")
        if lut_counts:
            for nbits, count in sorted(lut_counts.items(), key=lambda x: str(x[0])):
                label = f"LUT{nbits}" if isinstance(nbits, int) else f"LUT({nbits})"
                print(f"  {label}: {count} ops")
        else:
            print("  No LUT quantization ops found (FP16)")
        if other_constexpr:
            for op_type, count in other_constexpr.items():
                print(f"  {op_type}: {count} ops")

    except Exception as e:
        print(f"\n{name}: ERROR - {e}", file=sys.stderr)
