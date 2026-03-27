"""Compare old iOS models vs new stable models."""
import coremltools as ct

old_bundle = '/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle'
new_path = '/Users/yw68/Anemll/qwen3_5_stable_models'

print('=== OLD MODELS (current iOS) ===')
m = ct.models.MLModel(f'{old_bundle}/chunk0.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY, function_name='infer')
spec = m.get_spec()
fn = spec.mlProgram.functions['infer']
inputs = [inp.name for inp in fn.inputs]
print(f'chunk0 infer inputs: {inputs}')
print(f'chunk0 functions: {list(spec.mlProgram.functions.keys())}')

m2 = ct.models.MLModel(f'{old_bundle}/lm_head.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
spec2 = m2.get_spec()
outs = [(o.name, list(o.type.multiArrayType.shape)) for o in spec2.description.output]
print(f'lm_head outputs: {outs}')

m3 = ct.models.MLModel(f'{old_bundle}/embeddings.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
spec3 = m3.get_spec()
emb_ins = [(i.name, list(i.type.multiArrayType.shape)) for i in spec3.description.input]
print(f'embeddings inputs: {emb_ins}')

print()
print('=== NEW STABLE MODELS ===')
m4 = ct.models.MLModel(f'{new_path}/combined_LUT4_dedup/chunk0.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY, function_name='infer')
spec4 = m4.get_spec()
fn4 = spec4.mlProgram.functions['infer']
new_inputs = [inp.name for inp in fn4.inputs]
print(f'chunk0 infer inputs: {new_inputs}')
print(f'chunk0 functions: {list(spec4.mlProgram.functions.keys())}')

m5 = ct.models.MLModel(f'{new_path}/lm_head_logits.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
spec5 = m5.get_spec()
outs5 = [(o.name, list(o.type.multiArrayType.shape)) for o in spec5.description.output]
print(f'lm_head_logits outputs: {outs5}')

m6 = ct.models.MLModel(f'{new_path}/embeddings.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
spec6 = m6.get_spec()
emb_ins6 = [(i.name, list(i.type.multiArrayType.shape)) for i in spec6.description.input]
print(f'embeddings inputs: {emb_ins6}')

print()
print('=== DIFFERENCES ===')
old_set = set(inputs)
new_set = set(new_inputs)
added = new_set - old_set
removed = old_set - new_set
if added:
    print(f'New inputs added: {added}')
if removed:
    print(f'Inputs removed: {removed}')
if not added and not removed:
    print('Same input names')
