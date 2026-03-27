import coremltools as ct

m = ct.models.MLModel('/Users/yw68/Anemll/qwen3_5_6chunk_models/lm_head.mlpackage')
spec = m.get_spec()

print("=== LM HEAD SPEC ===")
for i in spec.description.input:
    mt = i.type.multiArrayType
    print(f"  Input: {i.name}  shape={list(mt.shape)}  dtype={mt.dataType}")

for o in spec.description.output:
    mt = o.type.multiArrayType
    print(f"  Output: {o.name}  shape={list(mt.shape)}  dtype={mt.dataType}")
