#!/usr/bin/env python3
"""Minimal CoreML probe for alternative KV-cache state contracts.

This intentionally avoids loading the full model. We only test whether CoreML
preserves single-position KV writes for Qwen3.5-like cache shapes.
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


def _make_inputs(pos: int):
    base = np.linspace(-1.0, 1.0, NUM_KV_HEADS * HEAD_DIM, dtype=np.float32).reshape(
        1, NUM_KV_HEADS, 1, HEAD_DIM
    )
    key_states = np.sin(base * 3.1).astype(np.float16)
    value_states = np.cos(base * 2.3).astype(np.float16)
    current_pos = np.array([pos], dtype=np.int32)

    k_ref = np.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=np.float16)
    v_ref = np.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=np.float16)
    k_ref[:, pos : pos + 1, :] = key_states.squeeze(0)
    v_ref[:, pos : pos + 1, :] = value_states.squeeze(0)
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
            self.kv_cache_0[0:1, :, pos : pos + 1, :] = key_states
            self.kv_cache_0[1:2, :, pos : pos + 1, :] = value_states
            return self.kv_cache_0[0:1].squeeze(0), self.kv_cache_0[1:2].squeeze(0)

    wrapper = Wrapper().eval()
    key_states = torch.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=torch.float16)
    value_states = torch.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=torch.float16)
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
    return _save_and_reload_cpu(mlmodel, "kv_state_stacked_")


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
            self.k_cache[:, pos : pos + 1, :] = key_states.squeeze(0)
            self.v_cache[:, pos : pos + 1, :] = value_states.squeeze(0)
            return self.k_cache.clone(), self.v_cache.clone()

    wrapper = Wrapper().eval()
    key_states = torch.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=torch.float16)
    value_states = torch.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=torch.float16)
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
    return _save_and_reload_cpu(mlmodel, "kv_state_split_")


def _convert_split_stateless() -> ct.models.MLModel:
    class Wrapper(torch.nn.Module):
        def forward(self, k_cache, v_cache, key_states, value_states, current_pos):
            pos = current_pos
            k = k_cache.clone()
            v = v_cache.clone()
            k[:, pos : pos + 1, :] = key_states.squeeze(0)
            v[:, pos : pos + 1, :] = value_states.squeeze(0)
            return k, v

    wrapper = Wrapper().eval()
    k_cache = torch.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=torch.float16)
    v_cache = torch.zeros((NUM_KV_HEADS, STATE_LENGTH, HEAD_DIM), dtype=torch.float16)
    key_states = torch.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=torch.float16)
    value_states = torch.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=torch.float16)
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
    return _save_and_reload_cpu(mlmodel, "kv_state_split_stateless_")


def _convert_split_stateful_masked() -> ct.models.MLModel:
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
            pos = current_pos.to(torch.long)
            pos_mask = torch.nn.functional.one_hot(pos, num_classes=STATE_LENGTH).to(torch.float16)
            pos_mask = pos_mask.view(1, STATE_LENGTH, 1)
            inv_mask = 1.0 - pos_mask
            key_update = key_states.squeeze(0) * pos_mask
            value_update = value_states.squeeze(0) * pos_mask
            self.k_cache.copy_(self.k_cache * inv_mask + key_update)
            self.v_cache.copy_(self.v_cache * inv_mask + value_update)
            return self.k_cache.clone(), self.v_cache.clone()

    wrapper = Wrapper().eval()
    key_states = torch.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=torch.float16)
    value_states = torch.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=torch.float16)
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
    return _save_and_reload_cpu(mlmodel, "kv_state_split_masked_")


def _run_one(pos: int) -> None:
    key_states, value_states, current_pos, k_ref, v_ref = _make_inputs(pos)
    print(f"position={pos}")

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

    split_masked = _convert_split_stateful_masked()
    out = split_masked.predict(
        {"key_states": key_states, "value_states": value_states, "current_pos": current_pos},
        split_masked.make_state(),
    )
    print(_metric("split masked stateful k_cache", out["key_cache_out"], k_ref))
    print(_metric("split masked stateful v_cache", out["value_cache_out"], v_ref))

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


def main() -> None:
    _run_one(0)
    _run_one(37)


if __name__ == "__main__":
    main()
