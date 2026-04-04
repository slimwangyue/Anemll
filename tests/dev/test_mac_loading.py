#!/usr/bin/env python3
"""Test loading production prefill models on Mac with different compute units."""
import coremltools as ct
import time
import os, sys

models_to_test = [
    # (path, description)
    ("qwen3_5_4chunk_lut6_bs512_ctx2048/prefill_LUT6_chunk0_bs512.mlpackage", "Production LUT6 chunk0 (22K ops)"),
    ("tests/dev/ane_op_test_models/test_real_qwen_prefill_4L.mlpackage", "4L real no-LUT6 (11K ops)"),
    ("tests/dev/ane_op_test_models/test_real_9layer_state.mlpackage", "9L test simple (580 ops)"),
    ("tests/dev/ane_op_test_models/test_mixed_attn_1layer.mlpackage", "1L mixed attn (92 ops)"),
]

for path, desc in models_to_test:
    full_path = os.path.join("/Users/yw68/Anemll", path)
    if not os.path.exists(full_path):
        print(f"\n{desc}: NOT FOUND ({path})")
        continue
    
    size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(full_path) for f in fns) / 1e6
    
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"  Size: {size_mb:.0f} MB")

    for cu_name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), 
                         ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                         ("ALL", ct.ComputeUnit.ALL)]:
        t0 = time.time()
        try:
            model = ct.models.MLModel(full_path, compute_units=cu)
            dt = time.time() - t0
            print(f"  {cu_name}: OK ({dt:.1f}s)")
            del model
        except Exception as e:
            dt = time.time() - t0
            err_str = str(e)[:200]
            print(f"  {cu_name}: FAIL ({dt:.1f}s) - {err_str}")
    sys.stdout.flush()

print("\nDone.")
