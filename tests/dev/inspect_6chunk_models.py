#!/usr/bin/env python3
"""Inspect the 6-chunk iOS models to understand their input/output signatures."""
import coremltools as ct
import os

BUNDLE = "/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle"

def inspect_model(path, label):
    print(f"\n{'='*60}")
    print(f"{label}: {os.path.basename(path)}")
    print(f"{'='*60}")
    spec = ct.utils.load_spec(path)
    
    # Check if multi-function
    if hasattr(spec.description, 'functions') and spec.description.functions:
        for fn in spec.description.functions:
            print(f"\n  Function: {fn.name}")
            print(f"    Inputs:")
            for inp in fn.input:
                shape = list(inp.type.multiArrayType.shape) if inp.type.HasField('multiArrayType') else "?"
                dtype = inp.type.multiArrayType.dataType if inp.type.HasField('multiArrayType') else "?"
                print(f"      {inp.name}: shape={shape}, dtype={dtype}")
            print(f"    Outputs:")
            for out in fn.output:
                shape = list(out.type.multiArrayType.shape) if out.type.HasField('multiArrayType') else "?"
                print(f"      {out.name}: shape={shape}")
        # Check states
        if hasattr(spec.description, 'stateDescriptions') and spec.description.stateDescriptions:
            print(f"\n    States:")
            for sd in spec.description.stateDescriptions:
                shape = list(sd.arrayType.shape) if sd.HasField('arrayType') else "?"
                print(f"      {sd.name}: shape={shape}")
    else:
        print(f"  Inputs:")
        for inp in spec.description.input:
            shape = list(inp.type.multiArrayType.shape) if inp.type.HasField('multiArrayType') else "?"
            dtype = inp.type.multiArrayType.dataType if inp.type.HasField('multiArrayType') else "?"
            print(f"    {inp.name}: shape={shape}, dtype={dtype}")
        print(f"  Outputs:")
        for out in spec.description.output:
            shape = list(out.type.multiArrayType.shape) if out.type.HasField('multiArrayType') else "?"
            print(f"    {out.name}: shape={shape}")

# Inspect all models
for name in ["embeddings", "lm_head"]:
    path = os.path.join(BUNDLE, f"{name}.mlpackage")
    if os.path.exists(path):
        inspect_model(path, name)

for ci in range(6):
    path = os.path.join(BUNDLE, f"chunk{ci}.mlpackage")
    if os.path.exists(path):
        inspect_model(path, f"FFN chunk{ci}")
