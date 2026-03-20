#!/usr/bin/env python3
"""Targeted chunk-4 subrange compare for Qwen3.5 exported decode path.

This probes:
1. PT input -> CoreML 24:31  vs PT 24:31
2. PT 24:31 output -> CoreML 31:32 vs PT 31:32
3. CoreML 24:31 output -> CoreML 31:32 vs PT 31:32

That lets us distinguish an internal subrange issue from an interface issue.
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


def _convert_subrange(
    model: Qwen35ForCausalLM,
    start_layer: int,
    end_layer: int,
    context_length: int,
) -> ct.models.MLModel:
    local_num_layers = end_layer - start_layer
    has_linear = any(
        model.model.layers[i].layer_type == "linear_attention"
        for i in range(start_layer, end_layer)
    )

    class FFNWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            cfg = model.config
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.local_num_layers = local_num_layers
            self.register_buffer(
                "kv_cache_0",
                torch.zeros(
                    (2 * local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                    dtype=MODEL_DTYPE,
                    device=TEST_DEVICE,
                ),
            )
            self.states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(2 * local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                        dtype=np.float16,
                    ),
                    name="kv_cache_0",
                )
            ]
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
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                apply_final_norm=False,
            )

    wrapper = FFNWrapper().eval()
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
    tmpdir = Path(tempfile.mkdtemp(prefix=f"qwen35_subrange_{start_layer}_{end_layer}_"))
    pkg = tmpdir / "model.mlpackage"
    mlmodel.save(str(pkg))
    return ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="What is 2 + 2? Return only the number.")
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    input_ids = _build_input_ids(tokenizer, args.prompt)
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
    current_pos_scalar = torch.tensor(0, dtype=torch.int32)

    with torch.no_grad():
        hidden = model.model.embed_tokens(token_input).to(torch.float16)

    # March PyTorch up to layer 24 using the exact local-state export contract.
    chunk_ranges: List[Tuple[int, int]] = [(0, 8), (8, 16), (16, 24)]
    for start, end in chunk_ranges:
        local_layers = end - start
        conv_dim = (
            cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
        )
        conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
        kv = torch.zeros((2 * local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE)
        conv = torch.zeros((local_layers, conv_dim, conv_kernel), dtype=MODEL_DTYPE)
        rec = torch.zeros(
            (
                local_layers,
                cfg.text_config.linear_num_value_heads,
                cfg.text_config.linear_key_head_dim,
                cfg.text_config.linear_value_head_dim,
            ),
            dtype=MODEL_DTYPE,
        )
        with torch.no_grad():
            hidden = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos_scalar,
                kv_cache_0=kv,
                linear_conv_state=conv,
                linear_recurrent_state=rec,
                start_layer=start,
                end_layer=end,
                apply_final_norm=False,
            )
    hidden_24 = hidden.clone()
    print("got_hidden_24")

    # PyTorch references for 24:31 and 31:32.
    def run_pt_subrange(hidden_in: torch.Tensor, start: int, end: int) -> torch.Tensor:
        local_layers = end - start
        conv_dim = (
            cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
        )
        conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
        kv = torch.zeros((2 * local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE)
        conv = torch.zeros((local_layers, conv_dim, conv_kernel), dtype=MODEL_DTYPE)
        rec = torch.zeros(
            (
                local_layers,
                cfg.text_config.linear_num_value_heads,
                cfg.text_config.linear_key_head_dim,
                cfg.text_config.linear_value_head_dim,
            ),
            dtype=MODEL_DTYPE,
        )
        with torch.no_grad():
            return model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden_in.clone(),
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos_scalar,
                kv_cache_0=kv,
                linear_conv_state=conv,
                linear_recurrent_state=rec,
                start_layer=start,
                end_layer=end,
                apply_final_norm=False,
            )

    pt_24_31 = run_pt_subrange(hidden_24, 24, 31)
    print("pt_24_31_done")
    pt_31_32_from_pt = run_pt_subrange(pt_24_31, 31, 32)
    print("pt_31_32_done")

    compute_unit = ct.ComputeUnit.CPU_ONLY
    ml_24_31 = _convert_subrange(model, 24, 31, args.context_length)
    ml_31_32 = _convert_subrange(model, 31, 32, args.context_length)
    print("coreml_subranges_built")

    st_24_31 = ml_24_31.make_state()
    st_31_32_a = ml_31_32.make_state()
    st_31_32_b = ml_31_32.make_state()

    base_inputs = {
        "position_ids": np.array([0], dtype=np.int32),
        "causal_mask": causal_mask.numpy(),
        "current_pos": np.array([0], dtype=np.int32),
    }

    out_24_31 = ml_24_31.predict(
        {"hidden_states": hidden_24.numpy(), **base_inputs},
        st_24_31,
    )["output_hidden_states"]
    coreml_24_31 = torch.from_numpy(out_24_31)
    print(_metric("24:31 PT->CoreML vs PT", coreml_24_31, pt_24_31))

    out_31_32_from_pt = ml_31_32.predict(
        {"hidden_states": pt_24_31.numpy(), **base_inputs},
        st_31_32_a,
    )["output_hidden_states"]
    coreml_31_32_from_pt = torch.from_numpy(out_31_32_from_pt)
    print(_metric("31:32 PTinput->CoreML vs PT", coreml_31_32_from_pt, pt_31_32_from_pt))

    out_31_32_from_coreml = ml_31_32.predict(
        {"hidden_states": out_24_31, **base_inputs},
        st_31_32_b,
    )["output_hidden_states"]
    coreml_31_32_from_coreml = torch.from_numpy(out_31_32_from_coreml)
    print(_metric("31:32 CoreMLinput->CoreML vs PT", coreml_31_32_from_coreml, pt_31_32_from_pt))


if __name__ == "__main__":
    main()
