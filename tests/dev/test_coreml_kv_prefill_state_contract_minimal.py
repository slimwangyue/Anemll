#!/usr/bin/env python3
"""Minimal CoreML probe for prefill-style KV state writes.

This tests writing a full contiguous prefill span at current_pos=0 with
Qwen3.5-like KV cache shapes, comparing stacked stateful, split stateful, and
split stateless contracts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch


NUM_KV_HEADS = 4
STATE_LENGTH = 256
HEAD_DIM = 256
SEQ_LEN = 256


def _metric(name: str, got: np.ndarray, ref: np.ndarray) -> str:
    got_t = torch.from_numpy(got.astype(np.float32))
    ref_t = torch.from_numpy(ref.astype(np.float32))
    diff = (got_t - ref_t).abs()
    got_f = got_t.reshape(-1)
    ref_f = ref_t.reshape(-1)
    denom = got_f.norm() * ref_f.norm()
    cos = 0.0 if float(denom) == 0.0 else float(torch.dot(got_f, ref_f) / denom)
    return (
        f"{name}: shape={tuple(got.shape)} "
        f"max_abs={float(diff.max()):.6f} "
        f"mean_abs={float(diff.mean()):.6f} "
        f"cos={cos:.6f}"
    )


def _save_and_reload_cpu(mlmodel: ct.models.MLModel, prefix: str) -> ct.models.MLModel:
    tmpdir = Path(tempfile.mkdtemp(prefix=prefix))
    pkg = tmpdir / "model.mlpackage"
    mlmodel.save(str(pkg))
    return ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)


def _make_inputs():
    base = np.linspace(-1.0, 1.0, NUM_KV_HEADS * SEQ_LEN * HEAD_DIM, dtype=np.float32).reshape(
        1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM
    )
    key_states = np.sin(base * 3.1).astype(np.float16)
    value_states = np.cos(base * 2.3).astype(np.float16)
    current_pos = np.array([0], dtype=np.int32)
    k_ref = np.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=np.float16)
    v_ref = np.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=np.float16)
    k_ref[:, 0:SEQ_LEN, :] = key_states.squeeze(0)
    v_ref[:, 0:SEQ_LEN, :] = value_states.squeeze(0)
    return key_states, value_states, current_pos, k_ref, v_ref


def _convert_stacked_stateful() -> ct.models.MLModel:
    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "kv_cache_0",
                torch.zeros((2, NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=torch.float16),
            )
            self.states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(2, NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=np.float16
                    ),
                    name="kv_cache_0",
                )
            ]

        def forward(self, key_states, value_states, current_pos):
            pos = current_pos
            self.kv_cache_0[0:1, :, pos : pos + SEQ_LEN, :] = key_states
            self.kv_cache_0[1:2, :, pos : pos + SEQ_LEN, :] = value_states
            return self.kv_cache_0[0:1].squeeze(0), self.kv_cache_0[1:2].squeeze(0)

    wrapper = Wrapper().eval()
    key_states = torch.zeros((1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float16)
    value_states = torch.zeros((1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)
    traced = torch.jit.trace(wrapper, (key_states, value_states, current_pos))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="key_states", shape=key_states.shape, dtype=np.float16),
            ct.TensorType(name="value_states", shape=value_states.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="key_cache_out", dtype=np.float16),
            ct.TensorType(name="value_cache_out", dtype=np.float16),
        ],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    return _save_and_reload_cpu(mlmodel, "kv_prefill_stacked_")


def _convert_split_stateful() -> ct.models.MLModel:
    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "k_cache", torch.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=torch.float16)
            )
            self.register_buffer(
                "v_cache", torch.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=torch.float16)
            )
            self.states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=np.float16
                    ),
                    name="k_cache",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=np.float16
                    ),
                    name="v_cache",
                ),
            ]

        def forward(self, key_states, value_states, current_pos):
            pos = current_pos
            self.k_cache[:, pos : pos + SEQ_LEN, :] = key_states.squeeze(0)
            self.v_cache[:, pos : pos + SEQ_LEN, :] = value_states.squeeze(0)
            return self.k_cache.clone(), self.v_cache.clone()

    wrapper = Wrapper().eval()
    key_states = torch.zeros((1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float16)
    value_states = torch.zeros((1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)
    traced = torch.jit.trace(wrapper, (key_states, value_states, current_pos))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="key_states", shape=key_states.shape, dtype=np.float16),
            ct.TensorType(name="value_states", shape=value_states.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="key_cache_out", dtype=np.float16),
            ct.TensorType(name="value_cache_out", dtype=np.float16),
        ],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    return _save_and_reload_cpu(mlmodel, "kv_prefill_split_")


def _convert_split_stateless() -> ct.models.MLModel:
    class Wrapper(torch.nn.Module):
        def forward(self, k_cache, v_cache, key_states, value_states, current_pos):
            pos = current_pos
            k = k_cache.clone()
            v = v_cache.clone()
            k[:, pos : pos + SEQ_LEN, :] = key_states.squeeze(0)
            v[:, pos : pos + SEQ_LEN, :] = value_states.squeeze(0)
            return k, v

    wrapper = Wrapper().eval()
    k_cache = torch.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=torch.float16)
    v_cache = torch.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=torch.float16)
    key_states = torch.zeros((1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float16)
    value_states = torch.zeros((1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.float16)
    current_pos = torch.zeros((1,), dtype=torch.int32)
    traced = torch.jit.trace(wrapper, (k_cache, v_cache, key_states, value_states, current_pos))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="k_cache", shape=k_cache.shape, dtype=np.float16),
            ct.TensorType(name="v_cache", shape=v_cache.shape, dtype=np.float16),
            ct.TensorType(name="key_states", shape=key_states.shape, dtype=np.float16),
            ct.TensorType(name="value_states", shape=value_states.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="key_cache_out", dtype=np.float16),
            ct.TensorType(name="value_cache_out", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    return _save_and_reload_cpu(mlmodel, "kv_prefill_split_stateless_")


def main() -> None:
    key_states, value_states, current_pos, k_ref, v_ref = _make_inputs()

    stacked = _convert_stacked_stateful()
    out = stacked.predict(
        {"key_states": key_states, "value_states": value_states, "current_pos": current_pos},
        stacked.make_state(),
    )
    print(_metric("stacked stateful k_cache", out["key_cache_out"], k_ref))
    print(_metric("stacked stateful v_cache", out["value_cache_out"], v_ref))

    split = _convert_split_stateful()
    out = split.predict(
        {"key_states": key_states, "value_states": value_states, "current_pos": current_pos},
        split.make_state(),
    )
    print(_metric("split stateful k_cache", out["key_cache_out"], k_ref))
    print(_metric("split stateful v_cache", out["value_cache_out"], v_ref))

    stateless = _convert_split_stateless()
    out = stateless.predict(
        {
            "k_cache": np.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=np.float16),
            "v_cache": np.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=np.float16),
            "key_states": key_states,
            "value_states": value_states,
            "current_pos": current_pos,
        }
    )
    print(_metric("split stateless k_cache", out["key_cache_out"], k_ref))
    print(_metric("split stateless v_cache", out["value_cache_out"], v_ref))


if __name__ == "__main__":
    main()
