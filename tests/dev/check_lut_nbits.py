"""Extract LUT nbits from deployed models."""
import coremltools as ct

bundle = '/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle'
m = ct.models.MLModel(f'{bundle}/chunk0.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
spec = m.get_spec()
fn = spec.mlProgram.functions['infer']

for block in fn.block_specializations.values():
    for op in block.operations:
        if 'lut_to_dense' in op.type:
            print("Op:", op.type)
            print("Inputs:")
            for inp_name in op.inputs:
                arg = op.inputs[inp_name]
                for binding in arg.arguments:
                    if binding.HasField('value'):
                        iv = binding.value.immediateValue
                        wof = iv.WhichOneof('value')
                        if wof == 'tensor':
                            t = iv.tensor
                            if t.HasField('ints'):
                                vals = list(t.ints.values)[:5]
                                print(f"  {inp_name}: ints={vals}")
                            elif t.HasField('bytes'):
                                print(f"  {inp_name}: bytes len={len(t.bytes.values)}")
                            elif t.HasField('floats'):
                                vals = list(t.floats.values)[:5]
                                print(f"  {inp_name}: floats={vals}")
                            else:
                                print(f"  {inp_name}: tensor (other)")
                        else:
                            print(f"  {inp_name}: {wof}")
                    elif binding.HasField('name'):
                        print(f"  {inp_name}: ref={binding.name}")
            break
    break
