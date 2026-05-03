#!/usr/bin/env python3
"""Re-quantize FP16 vision encoders with fixed seed, then combine for proper dedup."""
import coremltools as ct
import coremltools.optimize.coreml as cto_coreml
import numpy as np
import os, time, shutil

output_dir = 'qwen3_5_stable_lut4ffn_lut6em_fp16/lut6_per_res'
fp16_dir = 'qwen3_5_stable_lut4ffn_lut6em_fp16'

res_to_file = {
    ('448', '448'): 'vision_encoder.mlpackage',
    ('448', '672'): 'vision_encoder_448x672.mlpackage',
    ('448', '896'): 'vision_encoder_448x896.mlpackage',
    ('672', '448'): 'vision_encoder_672x448.mlpackage',
    ('896', '448'): 'vision_encoder_896x448.mlpackage',
}

config = cto_coreml.OptimizationConfig(
    global_config=cto_coreml.OpPalettizerConfig(
        mode='kmeans',
        nbits=6,
        granularity='per_grouped_channel',
        group_size=8,
        num_kmeans_workers=1,
    )
)

for (h, w), fname in res_to_file.items():
    fp16_path = os.path.join(fp16_dir, fname)
    out_path = os.path.join(output_dir, f'vision_encoder_{h}x{w}_seedfix.mlpackage')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    print(f'Quantizing {h}x{w} from {fname}...')
    np.random.seed(42)
    model = ct.models.MLModel(fp16_path)
    t0 = time.time()
    quantized = cto_coreml.palettize_weights(model, config)
    elapsed = time.time() - t0
    quantized.save(out_path)
    sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fns in os.walk(out_path) for f in fns)
    print(f'  -> {sz / 1e6:.1f} MB in {elapsed:.1f}s')

print('\nAll quantized. Now combining with dedup...')

desc = ct.utils.MultiFunctionDescriptor()
default_fn = None
for (h, w) in res_to_file:
    fn_name = f'f_{h}x{w}'
    pkg = os.path.join(output_dir, f'vision_encoder_{h}x{w}_seedfix.mlpackage')
    desc.add_function(pkg, 'main', fn_name)
    if default_fn is None:
        default_fn = fn_name
desc.default_function_name = default_fn

combined_path = os.path.join(output_dir, 'vision_encoder_multi_lut6_dedup.mlpackage')
if os.path.exists(combined_path):
    shutil.rmtree(combined_path)

print(f'Saving combined -> {combined_path}')
t0 = time.time()
ct.utils.save_multifunction(desc, combined_path)
elapsed = time.time() - t0

sz = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fns in os.walk(combined_path) for f in fns)
old_path = os.path.join(output_dir, 'vision_encoder_multi_lut6.mlpackage')
old_sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fns in os.walk(old_path) for f in fns)
print(f'\nDone in {elapsed:.1f}s')
print(f'New combined (seed-fixed): {sz / 1e6:.1f} MB')
print(f'Old combined (no seed):    {old_sz / 1e6:.1f} MB')
print(f'Reduction: {(1 - sz/old_sz)*100:.1f}%')
