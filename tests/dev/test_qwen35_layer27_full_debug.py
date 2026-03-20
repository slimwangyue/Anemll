#!/usr/bin/env python3
"""Debug full-attention layer 27 internals against CoreML on the first prompt token."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

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


def _save_and_reload_cpu(mlmodel: ct.models.MLModel, prefix: str) -> ct.models.MLModel:
    tmpdir = Path(tempfile.mkdtemp(prefix=prefix))
    pkg = tmpdir / "model.mlpackage"
    mlmodel.save(str(pkg))
    return ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)


def _convert_qkv_debug(model: Qwen35ForCausalLM, context_length: int) -> ct.models.MLModel:
    layer = model.model.layers[27]

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = layer

        def forward(self, hidden_states, current_pos):
            x = self.layer.input_layernorm(hidden_states)
            q, k, v, gate = self.layer.self_attn.get_new_kv_cache(x, current_pos)
            return x, q, k, v, gate

    wrapper = Wrapper().eval()
    hidden_states = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    traced = torch.jit.trace(wrapper, (hidden_states, current_pos))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="normed_x", dtype=np.float16),
            ct.TensorType(name="query_states", dtype=np.float16),
            ct.TensorType(name="key_states", dtype=np.float16),
            ct.TensorType(name="value_states", dtype=np.float16),
            ct.TensorType(name="gate", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    return _save_and_reload_cpu(mlmodel, "qwen35_l27_qkv_")


def _convert_forward_debug(model: Qwen35ForCausalLM, context_length: int) -> ct.models.MLModel:
    layer = model.model.layers[27]
    cfg = model.config

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = layer
            self.register_buffer(
                "kv_cache_0",
                torch.zeros((2, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            )
            self.states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(2, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                        dtype=np.float16,
                    ),
                    name="kv_cache_0",
                )
            ]

        def forward(self, hidden_states, query_states, key_states, value_states, gate, causal_mask, current_pos):
            pos = current_pos
            self.kv_cache_0[0:1, :, pos:pos + 1, :] = key_states
            self.kv_cache_0[1:2, :, pos:pos + 1, :] = value_states
            key_cache = self.kv_cache_0[0:1].squeeze(0)
            value_cache = self.kv_cache_0[1:2].squeeze(0)
            attn_out = self.layer.self_attn.forward_regular(
                hidden_states=hidden_states,
                query_states=query_states,
                kv_cache_layer=(key_cache, value_cache),
                causal_mask=causal_mask,
                gate=gate,
            )
            hidden_out = hidden_states + attn_out
            post = self.layer.post_attention_layernorm(hidden_out)
            mlp_out = self.layer.mlp(post)
            final_out = hidden_out + mlp_out
            return attn_out, hidden_out, final_out

    wrapper = Wrapper().eval()
    hidden_states = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    query_states = torch.zeros((1, model.config.num_attention_heads, 1, model.config.head_dim), dtype=torch.float16, device=TEST_DEVICE)
    key_states = torch.zeros((1, model.config.num_key_value_heads, 1, model.config.head_dim), dtype=torch.float16, device=TEST_DEVICE)
    value_states = torch.zeros((1, model.config.num_key_value_heads, 1, model.config.head_dim), dtype=torch.float16, device=TEST_DEVICE)
    gate = torch.zeros((1, 1, model.config.num_attention_heads * model.config.head_dim), dtype=torch.float16, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, context_length), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    traced = torch.jit.trace(wrapper, (hidden_states, query_states, key_states, value_states, gate, causal_mask, current_pos))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="query_states", shape=query_states.shape, dtype=np.float16),
            ct.TensorType(name="key_states", shape=key_states.shape, dtype=np.float16),
            ct.TensorType(name="value_states", shape=value_states.shape, dtype=np.float16),
            ct.TensorType(name="gate", shape=gate.shape, dtype=np.float16),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="attn_out", dtype=np.float16),
            ct.TensorType(name="hidden_out", dtype=np.float16),
            ct.TensorType(name="final_out", dtype=np.float16),
        ],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    return _save_and_reload_cpu(mlmodel, "qwen35_l27_fwd_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--mode", choices=["all", "qkv", "forward"], default="all")
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
    current_pos = torch.tensor([0], dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, args.context_length), dtype=torch.float16)

    with torch.no_grad():
        hidden = model.model.embed_tokens(token_input).to(torch.float16)
    for start, end in [(0, 8), (8, 16), (16, 24), (24, 27)]:
        hidden = _run_pt_subrange(model, hidden, position_ids, causal_mask, current_pos.squeeze(0), start, end)
    hidden_27 = hidden.clone()
    print("got_hidden_27")

    layer = model.model.layers[27]
    with torch.no_grad():
        normed_x = layer.input_layernorm(hidden_27)
        pt_q, pt_k, pt_v, pt_gate = layer.self_attn.get_new_kv_cache(normed_x, current_pos)

        kv = torch.zeros((2, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE)
        kv[0:1, :, 0:1, :] = pt_k
        kv[1:2, :, 0:1, :] = pt_v
        key_cache = kv[0:1].squeeze(0)
        value_cache = kv[1:2].squeeze(0)
        pt_attn_out = layer.self_attn.forward_regular(
            hidden_states=normed_x,
            query_states=pt_q,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
            gate=pt_gate,
        )
        pt_hidden_out = hidden_27 + pt_attn_out
        pt_post = layer.post_attention_layernorm(pt_hidden_out)
        pt_final_out = pt_hidden_out + layer.mlp(pt_post)
    print("pt_layer27_ready")

    if args.mode in ("all", "qkv"):
        qkv_model = _convert_qkv_debug(model, args.context_length)
        qkv_out = qkv_model.predict({"hidden_states": hidden_27.numpy(), "current_pos": current_pos.numpy()})
        cm_normed_x = torch.from_numpy(qkv_out["normed_x"])
        cm_q = torch.from_numpy(qkv_out["query_states"])
        cm_k = torch.from_numpy(qkv_out["key_states"])
        cm_v = torch.from_numpy(qkv_out["value_states"])
        cm_gate = torch.from_numpy(qkv_out["gate"])

        print(_metric("layer27 normed_x", cm_normed_x, normed_x))
        print(_metric("layer27 query_states", cm_q, pt_q))
        print(_metric("layer27 key_states", cm_k, pt_k))
        print(_metric("layer27 value_states", cm_v, pt_v))
        print(_metric("layer27 gate", cm_gate, pt_gate))

    if args.mode in ("all", "forward"):
        fwd_model = _convert_forward_debug(model, args.context_length)
        fwd_out = fwd_model.predict(
            {
                "hidden_states": normed_x.numpy(),
                "query_states": pt_q.numpy(),
                "key_states": pt_k.numpy(),
                "value_states": pt_v.numpy(),
                "gate": pt_gate.numpy(),
                "causal_mask": causal_mask.numpy(),
                "current_pos": current_pos.numpy(),
            },
            fwd_model.make_state(),
        )
        cm_attn_out = torch.from_numpy(fwd_out["attn_out"])
        cm_hidden_out = torch.from_numpy(fwd_out["hidden_out"])
        cm_final_out = torch.from_numpy(fwd_out["final_out"])

        print(_metric("layer27 attn_out", cm_attn_out, pt_attn_out))
        print(_metric("layer27 hidden_out", cm_hidden_out, pt_hidden_out))
        print(_metric("layer27 final_out", cm_final_out, pt_final_out))


if __name__ == "__main__":
    main()
