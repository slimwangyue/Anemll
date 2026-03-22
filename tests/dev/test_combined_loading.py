#!/usr/bin/env python3
"""Test if combined dedup models can be loaded without function_name,
allowing both infer and prefill functions from a single model load."""
import coremltools as ct
import time
import inspect
import numpy as np

cu = ct.ComputeUnit.CPU_AND_NE
path = "qwen3_5_stable_models/combined_LUT4_dedup/chunk0.mlpackage"

print("=== Test 1: Load WITHOUT function_name ===")
try:
    t0 = time.time()
    m = ct.models.MLModel(path, compute_units=cu)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Type: {type(m)}")

    spec = m.get_spec()
    if hasattr(spec, "description") and hasattr(spec.description, "functions"):
        fns = [f.name for f in spec.description.functions]
        print(f"  Functions: {fns}")
    else:
        print("  No functions attribute found")

    try:
        state = m.make_state()
        print(f"  make_state() OK")
    except Exception as e:
        print(f"  make_state() failed: {e}")

    sig = inspect.signature(m.predict)
    print(f"  predict params: {list(sig.parameters.keys())}")
    del m
except Exception as e:
    print(f"  Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print('=== Test 2: Load WITH function_name="infer" ===')
try:
    t0 = time.time()
    m2 = ct.models.MLModel(path, compute_units=cu, function_name="infer")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    state2 = m2.make_state()
    print(f"  make_state() OK")

    sig = inspect.signature(m2.predict)
    print(f"  predict params: {list(sig.parameters.keys())}")

    # List input names
    spec2 = m2.get_spec()
    fn_inputs = None
    if hasattr(spec2.description, "functions"):
        for fn in spec2.description.functions:
            if fn.name == "infer":
                fn_inputs = fn.input
                break
    if fn_inputs is None:
        fn_inputs = spec2.description.input
    print(f"  Infer inputs: {[inp.name for inp in fn_inputs]}")

    del m2
except Exception as e:
    print(f"  Failed: {e}")

print()
print('=== Test 3: Load WITH function_name="prefill" ===')
try:
    t0 = time.time()
    m3 = ct.models.MLModel(path, compute_units=cu, function_name="prefill")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    state3 = m3.make_state()
    print(f"  make_state() OK")

    spec3 = m3.get_spec()
    fn_inputs = None
    if hasattr(spec3.description, "functions"):
        for fn in spec3.description.functions:
            if fn.name == "prefill":
                fn_inputs = fn.input
                break
    if fn_inputs is None:
        fn_inputs = spec3.description.input
    print(f"  Prefill inputs: {[inp.name for inp in fn_inputs]}")
    for inp in fn_inputs:
        try:
            shape = tuple(inp.type.multiArrayType.shape)
            print(f"    {inp.name}: {shape}")
        except Exception:
            try:
                shape_range = inp.type.multiArrayType.shapeRange
                print(f"    {inp.name}: shapeRange={shape_range}")
            except Exception:
                print(f"    {inp.name}: (could not read shape)")

    del m3
except Exception as e:
    print(f"  Failed: {e}")

print("\nDone.")
