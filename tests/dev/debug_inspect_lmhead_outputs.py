#!/usr/bin/env python3
"""Inspect lm_head output schema and runtime output shapes for a model directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np


def find_model(base: Path, name: str) -> Path:
    for ext in (".mlmodelc", ".mlpackage"):
        p = base / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def load_model(path: Path):
    if path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(path), ct.ComputeUnit.CPU_AND_NE)
    return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    base = Path(args.model_dir)
    lm_path = find_model(base, "lm_head")
    lm = load_model(lm_path)

    spec = lm.get_spec()
    print("lm_head_path:", lm_path)
    print("outputs from spec:")
    for o in spec.description.output:
        typ = o.type.WhichOneof("Type")
        if typ == "multiArrayType":
            shp = [int(x) for x in o.type.multiArrayType.shape]
            print(f"  - {o.name}: shape={shp}, dtype={o.type.multiArrayType.dataType}")
        else:
            print(f"  - {o.name}: {typ}")

    hidden = np.zeros((1, 1, 2560), dtype=np.float16)
    out = lm.predict({"hidden_states": hidden})
    print("runtime output keys:", sorted(out.keys()))
    for k, v in out.items():
        arr = np.asarray(v)
        print(f"  - {k}: shape={list(arr.shape)}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
