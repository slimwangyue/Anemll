#!/usr/bin/env python3
"""Deep split compare for Qwen3.5 chunk-4 first-token divergence.

This evaluates the three subranges inside 24:31:
  - 24:27
  - 27:28
  - 28:31

For each subrange it measures:
  1. PT input -> CoreML vs PT
  2. CoreML-composed input -> CoreML vs PT
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import List, Tuple

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer

from anemll.models.qwen3_5_model import MODEL_DTYPE, Qwen35Config, Qwen35ForCausalLM, TEST_DEVICE


PROMPT = "What is 2 + 2? Return only the number."


def _build_input_ids(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    try:
        out = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(out, "input_ids"):
            return out.input_ids
        if isinstance(out, torch.Tensor):
            return out
    except Exception:
        pass
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = a_flat.norm() * b_flat.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / denom)


def _metric(name: str, got: torch.Tensor, ref: torch.Tensor) -> str:
    diff = (got.float() - ref.float()).abs()
    return (
        f"{name}: max_abs={float(diff.max()):.6f} "
        f"mean_abs={float(diff.mean()):.6f} "
        f"cos={_cosine(got, ref):.6f}"
    )


def _make_local_states(cfg: Qwen35Config, local_num_layers: int):
    kv = torch.zeros(
        (2 * local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
        dtype=MODEL_DTYPE,
        device=TEST_DEVICE,
    )
    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    conv = torch.zeros((local_num_layers, conv_dim, conv_kernel), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    rec = torch.zeros(
        (
            local_num_layers,
            cfg.text_config.linear_num_value_heads,
            cfg.text_config.linear_key_head_dim,
            cfg.text_config.linear_value_head_dim,
        ),
        dtype=MODEL_DTYPE,
        device=TEST_DEVICE,
    )
    return kv, conv, rec


def _run_pt_subrange(
    model: Qwen35ForCausalLM,
    hidden_in: torch.Tensor,
    position_ids: torch.Tensor,
    causal_mask: torch.Tensor,
    current_pos: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    local_num_layers = end - start
    kv, conv, rec = _make_local_states(model.config, local_num_layers)
    with torch.no_grad():
        return model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden_in.clone(),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            kv_cache_0=kv,
            linear_conv_state=conv,
            linear_recurrent_state=rec,
            start_layer=start,
            end_layer=end,
            apply_final_norm=False,
        )


def _convert_subrange(model: Qwen35ForCausalLM, start: int, end: int, context_length: int) -> ct.models.MLModel:
    local_num_layers = end - start
    has_linear = any(
        model.model.layers[i].layer_type == "linear_attention"
        for i in range(start, end)
    )
    has_full = any(
        model.model.layers[i].layer_type == "full_attention"
        for i in range(start, end)
    )

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            cfg = model.config
            self.model = model
            self.start = start
            self.end = end
            self.states = []
            if has_full:
                self.register_buffer(
                    "kv_cache_0",
                    torch.zeros(
                        (2 * local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                        dtype=MODEL_DTYPE,
                        device=TEST_DEVICE,
                    ),
                )
                self.states.append(
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=(2 * local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                            dtype=np.float16,
                        ),
                        name="kv_cache_0",
                    )
                )
            else:
                self.kv_cache_0 = torch.zeros((1, 1, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
            if has_linear:
                conv_dim = (
                    cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                    + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
                )
                conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
                self.register_buffer(
                    "linear_conv_state",
                    torch.zeros((local_num_layers, conv_dim, conv_kernel), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                )
                self.register_buffer(
                    "linear_recurrent_state",
                    torch.zeros(
                        (
                            local_num_layers,
                            cfg.text_config.linear_num_value_heads,
                            cfg.text_config.linear_key_head_dim,
                            cfg.text_config.linear_value_head_dim,
                        ),
                        dtype=MODEL_DTYPE,
                        device=TEST_DEVICE,
                    ),
                )
                self.states.extend(
                    [
                        ct.StateType(
                            wrapped_type=ct.TensorType(
                                shape=(local_num_layers, conv_dim, conv_kernel),
                                dtype=np.float16,
                            ),
                            name="linear_conv_state",
                        ),
                        ct.StateType(
                            wrapped_type=ct.TensorType(
                                shape=(
                                    local_num_layers,
                                    cfg.text_config.linear_num_value_heads,
                                    cfg.text_config.linear_key_head_dim,
                                    cfg.text_config.linear_value_head_dim,
                                ),
                                dtype=np.float16,
                            ),
                            name="linear_recurrent_state",
                        ),
                    ]
                )
            else:
                self.linear_conv_state = None
                self.linear_recurrent_state = None

        def forward(self, hidden_states, position_ids, causal_mask, current_pos):
            return self.model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                kv_cache_0=self.kv_cache_0,
                linear_conv_state=self.linear_conv_state,
                linear_recurrent_state=self.linear_recurrent_state,
                start_layer=self.start,
                end_layer=self.end,
                apply_final_norm=False,
            )

    wrapper = Wrapper().eval()
    hidden_states = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, context_length), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    tmpdir = Path(tempfile.mkdtemp(prefix=f"qwen35_subrange_{start}_{end}_"))
    pkg = tmpdir / "model.mlpackage"
    mlmodel.save(str(pkg))
    return ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    input_ids = _build_input_ids(tokenizer, PROMPT)
    token_id = int(input_ids[0, 0].item())
    print(f"token0_id={token_id}")

    cfg = Qwen35Config.from_json(str(Path(args.model_path) / "config.json"))
    model = Qwen35ForCausalLM(cfg).half().eval()
    ok = model.load_pretrained_weights(args.model_path)
    if not ok:
        raise RuntimeError("Failed to load repo weights")

    token_input = torch.tensor([[token_id]], dtype=torch.int32)
    position_ids = torch.tensor([0], dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, args.context_length), dtype=torch.float16)
    current_pos = torch.tensor(0, dtype=torch.int32)

    with torch.no_grad():
        hidden = model.model.embed_tokens(token_input).to(torch.float16)

    # Advance exactly to layer 24.
    for start, end in [(0, 8), (8, 16), (16, 24)]:
        hidden = _run_pt_subrange(model, hidden, position_ids, causal_mask, current_pos, start, end)
    hidden_24 = hidden.clone()
    print("got_hidden_24")

    pt_24_27 = _run_pt_subrange(model, hidden_24, position_ids, causal_mask, current_pos, 24, 27)
    pt_27_28_from_pt = _run_pt_subrange(model, pt_24_27, position_ids, causal_mask, current_pos, 27, 28)
    pt_28_31_from_pt = _run_pt_subrange(model, pt_27_28_from_pt, position_ids, causal_mask, current_pos, 28, 31)
    print("pt_refs_ready")

    ml_24_27 = _convert_subrange(model, 24, 27, args.context_length)
    ml_27_28 = _convert_subrange(model, 27, 28, args.context_length)
    ml_28_31 = _convert_subrange(model, 28, 31, args.context_length)
    print("coreml_subranges_built")

    base_inputs = {
        "position_ids": np.array([0], dtype=np.int32),
        "causal_mask": causal_mask.numpy(),
        "current_pos": np.array([0], dtype=np.int32),
    }

    st_24_27 = ml_24_27.make_state()
    out_24_27 = ml_24_27.predict({"hidden_states": hidden_24.numpy(), **base_inputs}, st_24_27)["output_hidden_states"]
    cm_24_27 = torch.from_numpy(out_24_27)
    print(_metric("24:27 PTinput->CoreML vs PT", cm_24_27, pt_24_27))

    st_27_28_a = ml_27_28.make_state()
    out_27_28_from_pt = ml_27_28.predict({"hidden_states": pt_24_27.numpy(), **base_inputs}, st_27_28_a)["output_hidden_states"]
    cm_27_28_from_pt = torch.from_numpy(out_27_28_from_pt)
    print(_metric("27:28 PTinput->CoreML vs PT", cm_27_28_from_pt, pt_27_28_from_pt))

    st_27_28_b = ml_27_28.make_state()
    out_27_28_from_cm = ml_27_28.predict({"hidden_states": out_24_27, **base_inputs}, st_27_28_b)["output_hidden_states"]
    cm_27_28_from_cm = torch.from_numpy(out_27_28_from_cm)
    print(_metric("27:28 CoreMLinput->CoreML vs PT", cm_27_28_from_cm, pt_27_28_from_pt))

    st_28_31_a = ml_28_31.make_state()
    out_28_31_from_pt = ml_28_31.predict({"hidden_states": pt_27_28_from_pt.numpy(), **base_inputs}, st_28_31_a)["output_hidden_states"]
    cm_28_31_from_pt = torch.from_numpy(out_28_31_from_pt)
    print(_metric("28:31 PTinput->CoreML vs PT", cm_28_31_from_pt, pt_28_31_from_pt))

    st_28_31_b = ml_28_31.make_state()
    out_28_31_from_cm = ml_28_31.predict({"hidden_states": out_27_28_from_cm, **base_inputs}, st_28_31_b)["output_hidden_states"]
    cm_28_31_from_cm = torch.from_numpy(out_28_31_from_cm)
    print(_metric("28:31 CoreMLinput->CoreML vs PT", cm_28_31_from_cm, pt_28_31_from_pt))


if __name__ == "__main__":
    main()
