#!/usr/bin/env python3
"""Inspect layer-27 full-attention forward_regular internals vs CoreML."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer

from anemll.models.qwen3_5_model import MODEL_DTYPE, Qwen35Config, Qwen35ForCausalLM, TEST_DEVICE, _repeat_kv


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
        f"{name}: shape={tuple(got.shape)} "
        f"max_abs={float(diff.max()):.6f} "
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


def _convert_forward_internals(model: Qwen35ForCausalLM, context_length: int) -> ct.models.MLModel:
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
            n_rep = self.layer.self_attn.num_heads // self.layer.self_attn.num_kv_heads
            rep_k = _repeat_kv(key_cache.unsqueeze(0), n_rep)
            rep_v = _repeat_kv(value_cache.unsqueeze(0), n_rep)

            attn_weights = (
                torch.matmul(query_states.to(MODEL_DTYPE), rep_k.transpose(-1, -2).to(MODEL_DTYPE))
                * self.layer.self_attn.scale
            )
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn_core = torch.matmul(attn_probs, rep_v.to(MODEL_DTYPE))
            attn_flat = attn_core.transpose(1, 2).contiguous().flatten(2, 3)
            gated = attn_flat * torch.sigmoid(gate.to(attn_flat.dtype))
            proj = self.layer.self_attn.o_proj(gated.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
            hidden_out = hidden_states + proj
            post = self.layer.post_attention_layernorm(hidden_out)
            final_out = hidden_out + self.layer.mlp(post)
            return key_cache, value_cache, rep_k, rep_v, attn_probs, attn_flat, gated, proj, final_out

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
            ct.TensorType(name="key_cache", dtype=np.float16),
            ct.TensorType(name="value_cache", dtype=np.float16),
            ct.TensorType(name="rep_k", dtype=np.float16),
            ct.TensorType(name="rep_v", dtype=np.float16),
            ct.TensorType(name="attn_probs", dtype=np.float16),
            ct.TensorType(name="attn_flat", dtype=np.float16),
            ct.TensorType(name="gated", dtype=np.float16),
            ct.TensorType(name="proj", dtype=np.float16),
            ct.TensorType(name="final_out", dtype=np.float16),
        ],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    return _save_and_reload_cpu(mlmodel, "qwen35_l27_fwdint_")


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
        n_rep = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
        pt_rep_k = _repeat_kv(key_cache.unsqueeze(0), n_rep)
        pt_rep_v = _repeat_kv(value_cache.unsqueeze(0), n_rep)
        pt_attn_weights = (
            torch.matmul(pt_q.to(MODEL_DTYPE), pt_rep_k.transpose(-1, -2).to(MODEL_DTYPE))
            * layer.self_attn.scale
        )
        pt_attn_weights = pt_attn_weights + causal_mask.to(MODEL_DTYPE)
        pt_attn_probs = torch.softmax(pt_attn_weights, dim=-1)
        pt_attn_core = torch.matmul(pt_attn_probs, pt_rep_v.to(MODEL_DTYPE))
        pt_attn_flat = pt_attn_core.transpose(1, 2).contiguous().flatten(2, 3)
        pt_gated = pt_attn_flat * torch.sigmoid(pt_gate.to(pt_attn_flat.dtype))
        pt_proj = layer.self_attn.o_proj(pt_gated.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
        pt_hidden_out = normed_x + pt_proj
        pt_post = layer.post_attention_layernorm(pt_hidden_out)
        pt_final_out = pt_hidden_out + layer.mlp(pt_post)
    print("pt_internals_ready")

    model_cm = _convert_forward_internals(model, args.context_length)
    out = model_cm.predict(
        {
            "hidden_states": normed_x.numpy(),
            "query_states": pt_q.numpy(),
            "key_states": pt_k.numpy(),
            "value_states": pt_v.numpy(),
            "gate": pt_gate.numpy(),
            "causal_mask": causal_mask.numpy(),
            "current_pos": current_pos.numpy(),
        },
        model_cm.make_state(),
    )

    print(_metric("layer27 key_cache", torch.from_numpy(out["key_cache"]), key_cache))
    print(_metric("layer27 value_cache", torch.from_numpy(out["value_cache"]), value_cache))
    print(_metric("layer27 rep_k", torch.from_numpy(out["rep_k"]), pt_rep_k))
    print(_metric("layer27 rep_v", torch.from_numpy(out["rep_v"]), pt_rep_v))
    print(_metric("layer27 attn_probs", torch.from_numpy(out["attn_probs"]), pt_attn_probs))
    print(_metric("layer27 attn_flat", torch.from_numpy(out["attn_flat"]), pt_attn_flat))
    print(_metric("layer27 gated", torch.from_numpy(out["gated"]), pt_gated))
    print(_metric("layer27 proj", torch.from_numpy(out["proj"]), pt_proj))
    print(_metric("layer27 final_out", torch.from_numpy(out["final_out"]), pt_final_out))


if __name__ == "__main__":
    main()
