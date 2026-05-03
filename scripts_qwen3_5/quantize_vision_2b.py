#!/usr/bin/env python3
"""Quantize all 3 vision encoders for 2B model to LUT6."""
import coremltools as ct
import coremltools.optimize.coreml as cto_coreml
import time, os

out_dir = '/Volumes/MySSD/Anemll/qwen3_5_2b_mrope'
models = [
    ('vision_encoder.mlpackage', 'vision_encoder_lut6.mlpackage'),
    ('vision_encoder_448x896.mlpackage', 'vision_encoder_448x896_lut6.mlpackage'),
    ('vision_encoder_896x448.mlpackage', 'vision_encoder_896x448_lut6.mlpackage'),
]

config = cto_coreml.OptimizationConfig(
    global_config=cto_coreml.OpPalettizerConfig(
        mode="kmeans",
        nbits=6,
        granularity="per_grouped_channel",
        group_size=8,
        num_kmeans_workers=1,
    )
)

for src_name, dst_name in models:
    src = os.path.join(out_dir, src_name)
    dst = os.path.join(out_dir, dst_name)
    if os.path.exists(dst):
        print(f"SKIP {dst_name} (already exists)")
        continue
    print(f"\nQuantizing {src_name} -> {dst_name} ...")
    model = ct.models.MLModel(src)
    src_size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(src) for f in fs)
    t0 = time.time()
    q = cto_coreml.palettize_weights(model, config)
    q.save(dst)
    elapsed = time.time() - t0
    dst_size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(dst) for f in fs)
    print(f"  {src_size/1e6:.1f} MB -> {dst_size/1e6:.1f} MB ({dst_size*100//src_size}%) in {elapsed:.1f}s")

print("\nDone! All 3 LUT6 quantizations complete.")
