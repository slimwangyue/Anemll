#!/usr/bin/env python3
"""Minimal ANE model load sanity check for milestone1_3 models."""

import os
import time

import coremltools as ct

MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"
CU = ct.ComputeUnit.CPU_AND_NE


def load_model(path: str, name: str) -> None:
    print(f"Loading {name}: {path}", flush=True)
    t0 = time.time()
    _m = ct.models.MLModel(path, compute_units=CU)
    print(f"  OK in {time.time() - t0:.2f}s", flush=True)


def main() -> None:
    load_model(os.path.join(MODEL_DIR, "embeddings.mlpackage"), "embeddings")
    for i in range(4):
        load_model(
            os.path.join(MODEL_DIR, f"prefill_LUT4_chunk{i}.mlpackage"),
            f"prefill_LUT4_chunk{i}",
        )
    load_model(os.path.join(MODEL_DIR, "lm_head.mlpackage"), "lm_head")
    print("Done", flush=True)


if __name__ == "__main__":
    main()
