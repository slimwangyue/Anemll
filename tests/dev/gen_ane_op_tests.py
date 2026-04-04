"""Generate minimal CoreML models to test which ops load on ANE.

Creates 3 tiny models, each exercising exactly one suspected ANE-hostile op:
  1. test_less.mlpackage    – uses `less` comparison
  2. test_one_hot.mlpackage – uses `one_hot`
  3. test_gather.mlpackage  – uses `gather` (index-based)

Also creates a baseline model with only ANE-friendly ops for sanity.
"""

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import os, sys

OUT_DIR = "/Users/yw68/Anemll/tests/dev/ane_op_test_models"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ = 512
HIDDEN = 64  # keep models tiny


def save_model(prog, name):
    model = ct.convert(
        prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    model.save(path)
    print(f"Saved {path}")
    return path


# ── 0. Baseline: matmul only (should always load on ANE) ──
@mb.program(
    input_specs=[mb.TensorSpec(shape=(1, SEQ, HIDDEN), dtype=types.fp16)],
    opset_version=ct.target.iOS18,
)
def baseline(x):
    # Simple add + mul — pure ANE-friendly ops
    y = mb.add(x=x, y=np.float16(1.0))
    return mb.mul(x=y, y=np.float16(0.5))


save_model(baseline, "test_baseline")


# ── 1. Test `less` op ──
@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(1, SEQ, HIDDEN), dtype=types.fp16),
        mb.TensorSpec(shape=(1,), dtype=types.int32),  # valid_len
    ],
    opset_version=ct.target.iOS18,
)
def test_less(x, valid_len):
    positions = mb.const(val=np.arange(SEQ, dtype=np.int32))
    mask = mb.less(x=positions, y=valid_len)  # <-- the suspect op
    mask_fp = mb.cast(x=mask, dtype="fp16")
    mask_3d = mb.reshape(x=mask_fp, shape=np.array([1, SEQ, 1], dtype=np.int32))
    return mb.mul(x=x, y=mask_3d)


save_model(test_less, "test_less")


# ── 2. Test `one_hot` op ──
@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(1, SEQ, HIDDEN), dtype=types.fp16),
        mb.TensorSpec(shape=(1,), dtype=types.int32),  # index
    ],
    opset_version=ct.target.iOS18,
)
def test_one_hot(x, index):
    oh = mb.one_hot(
        indices=index,
        one_hot_vector_size=SEQ,
        axis=-1,
        on_value=np.int32(1),
        off_value=np.int32(0),
    )  # shape (1, SEQ)
    oh_fp = mb.cast(x=oh, dtype="fp16")
    # Use one_hot as a selector: (1, 1, SEQ) @ (1, SEQ, HIDDEN) -> (1, 1, HIDDEN)
    oh_3d = mb.reshape(x=oh_fp, shape=np.array([1, 1, SEQ], dtype=np.int32))
    return mb.matmul(x=oh_3d, y=x)  # (1, 1, HIDDEN)


save_model(test_one_hot, "test_one_hot")


# ── 3. Test `gather` op (the exact pattern from RoPE embedding) ──
@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(1, SEQ, HIDDEN), dtype=types.fp16),
        mb.TensorSpec(shape=(SEQ,), dtype=types.int32),  # indices
    ],
    opset_version=ct.target.iOS18,
)
def test_gather(x, indices):
    # Gather along axis=1, analogous to RoPE cos/sin embedding lookup
    # x: (1, 4096, 64) gathered with indices: (512,) -> (1, 512, 64) in the real model
    table = mb.const(val=np.random.randn(1, 4096, HIDDEN).astype(np.float16))
    out = mb.gather(x=table, indices=indices, axis=1)  # <-- the suspect op
    return mb.add(x=x, y=out)


save_model(test_gather, "test_gather")


# ── 4. Test `one_hot` with int type (conv_stage pattern, shape (4, 516)) ──
@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(1, SEQ, HIDDEN), dtype=types.fp16),
        mb.TensorSpec(shape=(4,), dtype=types.int32),  # conv_pos indices
    ],
    opset_version=ct.target.iOS18,
)
def test_one_hot_conv(x, conv_pos):
    oh = mb.one_hot(
        indices=conv_pos,
        one_hot_vector_size=516,
        axis=-1,
        on_value=np.int32(1),
        off_value=np.int32(0),
    )  # shape (4, 516), matches real conv_stage pattern
    oh_fp = mb.cast(x=oh, dtype="fp16")
    # Just multiply to make the output depend on it
    # Reshape to broadcast: (4, 516) -> (1, 4, 516) -> matmul not needed, just return x
    reduced = mb.reduce_sum(x=oh_fp, axes=np.array([-1], dtype=np.int32))  # (4,)
    reduced_3d = mb.reshape(x=reduced, shape=np.array([1, 1, 4], dtype=np.int32))
    # tile to match hidden
    tiled = mb.tile(x=reduced_3d, reps=np.array([1, SEQ, HIDDEN // 4], dtype=np.int32))
    return mb.mul(x=x, y=tiled)


save_model(test_one_hot_conv, "test_one_hot_conv")


# ── 5. Combined: less + one_hot (both together) ──
@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(1, SEQ, HIDDEN), dtype=types.fp16),
        mb.TensorSpec(shape=(1,), dtype=types.int32),  # valid_len
    ],
    opset_version=ct.target.iOS18,
)
def test_less_and_one_hot(x, valid_len):
    # less
    positions = mb.const(val=np.arange(SEQ, dtype=np.int32))
    mask = mb.less(x=positions, y=valid_len)
    mask_fp = mb.cast(x=mask, dtype="fp16")
    mask_3d = mb.reshape(x=mask_fp, shape=np.array([1, SEQ, 1], dtype=np.int32))
    masked = mb.mul(x=x, y=mask_3d)
    # one_hot readout
    idx = mb.sub(x=valid_len, y=np.int32(1))
    oh = mb.one_hot(indices=idx, one_hot_vector_size=SEQ, axis=-1,
                    on_value=np.int32(1), off_value=np.int32(0))
    oh_fp = mb.cast(x=oh, dtype="fp16")
    oh_3d = mb.reshape(x=oh_fp, shape=np.array([1, 1, SEQ], dtype=np.int32))
    return mb.matmul(x=oh_3d, y=masked)


save_model(test_less_and_one_hot, "test_less_and_one_hot")

print(f"\nAll models saved to {OUT_DIR}")
print("Models:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.mlpackage'):
        print(f"  {f}")
