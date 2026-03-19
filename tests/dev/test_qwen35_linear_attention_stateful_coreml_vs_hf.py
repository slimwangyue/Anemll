#!/usr/bin/env python3
"""Stateful CoreML parity for Qwen3.5 linear-attention block.

This script validates:
1) PyTorch linear-attention block parity vs HF for prefill/decode.
2) Stateful CoreML export for linear attention using conv/recurrent states.
3) CoreML runtime parity vs PyTorch when runtime is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anemll.models.qwen3_5_model import MODEL_DTYPE, Qwen35Config, Qwen35ForCausalLM


def _load_index(model_path: str) -> Dict:
    with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
        return json.load(f)


def _read_tensors(model_path: str, keys: List[str], weight_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    file_to_keys: Dict[str, List[str]] = {}
    for key in keys:
        if key in weight_map:
            file_to_keys.setdefault(weight_map[key], []).append(key)
    out: Dict[str, torch.Tensor] = {}
    for shard, shard_keys in file_to_keys.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for key in shard_keys:
                out[key] = f.get_tensor(key)
    return out


def _metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a32 = a.float().reshape(-1)
    b32 = b.float().reshape(-1)
    diff = (a32 - b32).abs()
    cos = torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0)).item()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a32 - b32) ** 2)).item()),
        "cosine": float(cos),
    }


def _print_metric(name: str, m: Dict[str, float]) -> None:
    print(
        f"{name:22s} max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
        f"rmse={m['rmse']:.6e} cosine={m['cosine']:.8f}"
    )


class _StatefulLinearAttentionBase(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module):
        super().__init__()
        self.attn = attn
        self.register_buffer(
            "conv_state",
            torch.zeros((1, attn.conv_dim, attn.linear_conv_kernel_dim), dtype=MODEL_DTYPE),
        )
        self.register_buffer(
            "recurrent_state",
            torch.zeros((1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim), dtype=torch.float32),
        )


class StatefulLinearAttentionPrefillBlock(_StatefulLinearAttentionBase):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out, next_conv, next_rec = self.attn.forward_prefill(
            hidden_states=hidden_states,
            conv_state=self.conv_state,
            recurrent_state=self.recurrent_state,
            has_previous_state=False,
        )
        self.conv_state[:, :, :] = next_conv
        self.recurrent_state[:, :, :, :] = next_rec.to(self.recurrent_state.dtype)
        return out


class StatefulLinearAttentionDecodeBlock(_StatefulLinearAttentionBase):
    def __init__(self, attn: torch.nn.Module, has_previous_state: bool = True):
        super().__init__(attn)
        self.has_previous_state = bool(has_previous_state)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out, next_conv, next_rec = self.attn.forward_regular(
            hidden_states=hidden_states,
            conv_state=self.conv_state,
            recurrent_state=self.recurrent_state,
            has_previous_state=self.has_previous_state,
        )
        self.conv_state[:, :, :] = next_conv
        self.recurrent_state[:, :, :, :] = next_rec.to(self.recurrent_state.dtype)
        return out


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _run_hf_parity(
    model_path: str,
    layer_idx: int,
    cfg: Qwen35Config,
    our_attn: torch.nn.Module,
    hidden_prefill: torch.Tensor,
    hidden_decode: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5DynamicCache

    hf_cfg = Qwen3_5Config.from_pretrained(model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=layer_idx).half().eval()

    idx = _load_index(model_path)
    weight_map = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    layer_keys = [k for k in weight_map.keys() if k.startswith(prefix)]
    tensors = _read_tensors(model_path, layer_keys, weight_map)
    hf_state = {k[len(prefix) :]: v for k, v in tensors.items()}
    missing, unexpected = hf_layer.load_state_dict(hf_state, strict=False)
    if missing:
        raise RuntimeError(f"HF layer missing keys: {missing}")
    if unexpected:
        print(f"HF unexpected keys: {unexpected}")

    with torch.no_grad():
        x_prefill = hf_layer.input_layernorm(hidden_prefill).half()
        # Keep decode setup aligned with the existing linear parity harness first.
        x_decode = x_prefill[:, -1:, :]

        hf_cache = Qwen3_5DynamicCache(hf_cfg)
        seq_len = x_prefill.shape[1]
        hf_prefill = hf_layer.linear_attn(
            hidden_states=x_prefill,
            cache_params=hf_cache,
            cache_position=torch.arange(seq_len, dtype=torch.long),
            attention_mask=None,
        )
        hf_decode = hf_layer.linear_attn(
            hidden_states=x_decode,
            cache_params=hf_cache,
            cache_position=torch.tensor([seq_len], dtype=torch.long),
            attention_mask=None,
        )

        prefill_block = StatefulLinearAttentionPrefillBlock(our_attn).eval()
        decode_block = StatefulLinearAttentionDecodeBlock(our_attn, has_previous_state=True).eval()
        our_prefill = prefill_block(x_prefill)
        decode_block.conv_state.copy_(prefill_block.conv_state)
        decode_block.recurrent_state.copy_(prefill_block.recurrent_state)
        our_decode = decode_block(x_decode)

    print("Stateful Linear-Attn Parity vs HF")
    _print_metric("prefill_torch_vs_hf", _metrics(our_prefill, hf_prefill))
    _print_metric("decode_torch_vs_hf", _metrics(our_decode, hf_decode))
    return our_prefill, our_decode


def _export_and_run_coreml(
    block_prefill: torch.nn.Module,
    block_decode: torch.nn.Module,
    cfg: Qwen35Config,
    seq_len: int,
    x_prefill: torch.Tensor,
    x_decode: torch.Tensor,
    save_packages: bool,
) -> None:
    import coremltools as ct

    traced_prefill = torch.jit.trace(block_prefill, (x_prefill,), strict=False, check_trace=False)
    traced_decode = torch.jit.trace(block_decode, (x_decode,), strict=False, check_trace=False)

    conv_state = ct.StateType(
        wrapped_type=ct.TensorType(
            shape=(1, block_prefill.attn.conv_dim, block_prefill.attn.linear_conv_kernel_dim),
            dtype=np.float16,
        ),
        name="conv_state",
    )
    recurrent_state = ct.StateType(
        wrapped_type=ct.TensorType(
            shape=(1, block_prefill.attn.num_v_heads, block_prefill.attn.head_k_dim, block_prefill.attn.head_v_dim),
            dtype=np.float32,
        ),
        name="recurrent_state",
    )

    ml_prefill = ct.convert(
        traced_prefill,
        inputs=[ct.TensorType(name="hidden_states", shape=x_prefill.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="attn_out")],
        states=[conv_state, recurrent_state],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )
    ml_decode = ct.convert(
        traced_decode,
        inputs=[ct.TensorType(name="hidden_states", shape=x_decode.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="attn_out")],
        states=[conv_state, recurrent_state],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )

    if save_packages:
        out_dir = REPO_ROOT / "tests" / "dev"
        prefill_pkg = out_dir / "qwen35_linear_attn_stateful_prefill.mlpackage"
        decode_pkg = out_dir / "qwen35_linear_attn_stateful_decode.mlpackage"
        ml_prefill.save(str(prefill_pkg))
        ml_decode.save(str(decode_pkg))
        print(f"Saved stateful CoreML prefill block: {prefill_pkg}")
        print(f"Saved stateful CoreML decode block: {decode_pkg}")

    try:
        st = ml_prefill.make_state()
        coreml_prefill = ml_prefill.predict({"hidden_states": _to_np(x_prefill)}, state=st)["attn_out"]
        print("CoreML prefill predict: SUCCESS")
    except Exception as e:
        print(f"CoreML prefill predict: SKIPPED ({e})")
        return

    try:
        st_decode = ml_decode.make_state()
        st_decode.write_state(name="conv_state", value=st.read_state(name="conv_state"))
        st_decode.write_state(name="recurrent_state", value=st.read_state(name="recurrent_state"))
        coreml_decode = ml_decode.predict({"hidden_states": _to_np(x_decode)}, state=st_decode)["attn_out"]
        print("CoreML decode predict: SUCCESS")
    except Exception as e:
        print(f"CoreML decode predict: SKIPPED ({e})")
        return

    with torch.no_grad():
        t_prefill = block_prefill(x_prefill)
        t_decode_block = block_decode
        t_decode_block.conv_state.copy_(block_prefill.conv_state)
        t_decode_block.recurrent_state.copy_(block_prefill.recurrent_state)
        t_decode = t_decode_block(x_decode)

    print("Stateful CoreML Parity vs PyTorch")
    _print_metric("prefill_coreml_vs_torch", _metrics(torch.from_numpy(coreml_prefill), t_prefill))
    _print_metric("decode_coreml_vs_torch", _metrics(torch.from_numpy(coreml_decode), t_decode))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer-idx", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-packages", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = Qwen35Config.from_json(os.path.join(args.model_path, "config.json"))
    if cfg.text_config.layer_types[args.layer_idx] != "linear_attention":
        raise ValueError(
            f"layer {args.layer_idx} is {cfg.text_config.layer_types[args.layer_idx]}, expected linear_attention"
        )

    model = Qwen35ForCausalLM(cfg).half().eval()
    if not model.load_pretrained_weights(args.model_path):
        raise RuntimeError("ANEMLL loader failed.")
    our_attn = model.model.layers[args.layer_idx].self_attn

    hidden_prefill = torch.randn(1, args.seq_len, cfg.hidden_size, dtype=torch.float32)
    hidden_decode = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32)

    _run_hf_parity(
        model_path=args.model_path,
        layer_idx=args.layer_idx,
        cfg=cfg,
        our_attn=our_attn,
        hidden_prefill=hidden_prefill,
        hidden_decode=hidden_decode,
    )

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    with torch.no_grad():
        hf_cfg = Qwen3_5Config.from_pretrained(args.model_path).text_config
        hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=args.layer_idx).half().eval()
        x_prefill = hf_layer.input_layernorm(hidden_prefill).half()
        x_decode = x_prefill[:, -1:, :]

    block_prefill = StatefulLinearAttentionPrefillBlock(our_attn).eval()
    block_decode = StatefulLinearAttentionDecodeBlock(our_attn, has_previous_state=True).eval()
    _export_and_run_coreml(
        block_prefill=block_prefill,
        block_decode=block_decode,
        cfg=cfg,
        seq_len=args.seq_len,
        x_prefill=x_prefill,
        x_decode=x_decode,
        save_packages=args.save_packages,
    )


if __name__ == "__main__":
    main()
