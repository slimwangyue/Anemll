import coremltools as ct
import os

bundle = '/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle'

for ci in [0, 5]:
    path = os.path.join(bundle, f'chunk{ci}.mlpackage')
    if not os.path.exists(path):
        print(f"chunk{ci}.mlpackage not found")
        continue
    
    for fn_name in ['prefill', 'infer']:
        try:
            m = ct.models.MLModel(path, function_name=fn_name)
            spec = m.get_spec()
            prog = spec.mlProgram
            
            print(f"=== CHUNK{ci} {fn_name} ===")
            if fn_name in prog.functions:
                fn = prog.functions[fn_name]
                for inp in fn.inputs:
                    shape_str = str(inp.type).replace('\n', ' ').strip()
                    # Extract just the shape numbers
                    print(f"  IN:  {inp.name:30s} {shape_str[:150]}")
                
                for blk_name in fn.block_specializations:
                    block = fn.block_specializations[blk_name]
                    for out in block.outputs:
                        out_str = str(out.type).replace('\n', ' ').strip()
                        print(f"  OUT: {out.name:30s} {out_str[:150]}")
            print()
        except Exception as e:
            print(f"  ERROR loading chunk{ci} {fn_name}: {e}")
            print()
