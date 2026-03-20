#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import coremltools as ct

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from tests.dev.tmp_linear_layout_probe import LayoutStage
from tests.dev.tmp_linear_proj_probe import CausalConvStage, QKVProjStage
from tests.dev.tmp_linear_stage_probe import NormOutStage, RecurrentCoreStage, metrics


def metric_line(name: str, a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> str:
    m = metrics(a, b)
    return (
        f"{name:18s} max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
        f"rmse={m['rmse']:.6e} cosine={m['cosine']:.8f}"
    )


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def convert_model(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    specs: list[ct.TensorType],
    output_names: list[str],
    package_name: str,
    compute_units: ct.ComputeUnit,
    pass_pipeline=None,
) -> ct.models.MLModel:
    traced = torch.jit.trace(module, inputs, strict=False, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=specs,
        outputs=[ct.TensorType(name=name) for name in output_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=pass_pipeline,
    )
    pkg = os.path.join("/tmp", package_name)
    shutil.rmtree(pkg, ignore_errors=True)
    mlmodel.save(pkg)
    return ct.models.MLModel(pkg, compute_units=compute_units)


def time_predict(model: ct.models.MLModel, feed: dict[str, np.ndarray], warmup: int, iters: int):
    for _ in range(warmup):
        model.predict(feed)
    start = time.perf_counter()
    out = None
    for _ in range(iters):
        out = model.predict(feed)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    return out, elapsed_ms


class ProjConvLayoutGroup(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.proj = QKVProjStage(attn)
        self.conv = CausalConvStage(attn, seq_len)
        self.layout = LayoutStage(attn, seq_len)

    def forward(self, hidden_states: torch.Tensor, conv_state: torch.Tensor):
        mixed_qkv, z, b, a = self.proj(hidden_states)
        conv_out, next_conv = self.conv(mixed_qkv, conv_state)
        query, key, value, beta, g, z_out = self.layout(conv_out, b, a, z)
        return query, key, value, beta, g, z_out, next_conv


class CoreNormGroup(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.core = RecurrentCoreStage(attn, seq_len)
        self.norm = NormOutStage(attn, seq_len)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        z_out: torch.Tensor,
        recurrent_state: torch.Tensor,
    ):
        core, next_rec = self.core(query, key, value, beta, g, recurrent_state)
        attn_out = self.norm(core, z_out)
        return attn_out, next_rec


class ProjConvGroup(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.proj = QKVProjStage(attn)
        self.conv = CausalConvStage(attn, seq_len)

    def forward(self, hidden_states: torch.Tensor, conv_state: torch.Tensor):
        mixed_qkv, z, b, a = self.proj(hidden_states)
        conv_out, next_conv = self.conv(mixed_qkv, conv_state)
        return conv_out, b, a, z, next_conv


class LayoutCoreNormGroup(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.layout = LayoutStage(attn, seq_len)
        self.core = RecurrentCoreStage(attn, seq_len)
        self.norm = NormOutStage(attn, seq_len)

    def forward(
        self,
        conv_out: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        z: torch.Tensor,
        recurrent_state: torch.Tensor,
    ):
        query, key, value, beta, g, z_out = self.layout(conv_out, b, a, z)
        core, next_rec = self.core(query, key, value, beta, g, recurrent_state)
        attn_out = self.norm(core, z_out)
        return attn_out, next_rec


class ProjConvLayoutCoreGroup(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module, seq_len: int):
        super().__init__()
        self.pcl = ProjConvLayoutGroup(attn, seq_len)
        self.core = RecurrentCoreStage(attn, seq_len)

    def forward(self, hidden_states: torch.Tensor, conv_state: torch.Tensor, recurrent_state: torch.Tensor):
        query, key, value, beta, g, z_out, next_conv = self.pcl(hidden_states, conv_state)
        core, next_rec = self.core(query, key, value, beta, g, recurrent_state)
        return core, z_out, next_conv, next_rec


@dataclass
class VariantResult:
    name: str
    prefill_ms: float
    decode_ms: float
    prefill_metric: str
    decode_metric: str
    conv_prefill_metric: str
    rec_prefill_metric: str


def run_variant(
    name: str,
    prefill_modules: list[tuple[torch.nn.Module, tuple[torch.Tensor, ...], list[ct.TensorType], list[str], str]],
    decode_modules: list[tuple[torch.nn.Module, tuple[torch.Tensor, ...], list[ct.TensorType], list[str], str]],
    prefill_feeds: list[dict[str, np.ndarray]],
    decode_feeds_builder,
    torch_prefill_attn: torch.Tensor,
    torch_decode_attn: torch.Tensor,
    torch_prefill_conv: torch.Tensor,
    torch_prefill_rec: torch.Tensor,
    warmup: int,
    iters: int,
    compute_units: ct.ComputeUnit,
    pass_pipeline=None,
) -> VariantResult:
    def resolve_value(v, curr, outputs):
        if isinstance(v, str):
            return curr[v]
        if isinstance(v, tuple) and len(v) == 2:
            idx, key = v
            return outputs[idx][key]
        return v

    prefill_models = [
        convert_model(mod, inputs, specs, outputs, pkg, compute_units, pass_pipeline=pass_pipeline)
        for mod, inputs, specs, outputs, pkg in prefill_modules
    ]
    decode_models = [
        convert_model(mod, inputs, specs, outputs, pkg, compute_units, pass_pipeline=pass_pipeline)
        for mod, inputs, specs, outputs, pkg in decode_modules
    ]

    prefill_last = None
    prefill_outputs: list[dict[str, np.ndarray]] = []
    total_prefill_ms = 0.0
    curr = None
    for model, feed in zip(prefill_models, prefill_feeds):
        actual_feed = (
            feed
            if curr is None
            else {k: resolve_value(v, curr, prefill_outputs) for k, v in feed.items()}
        )
        curr, ms = time_predict(model, actual_feed, warmup, iters)
        total_prefill_ms += ms
        prefill_outputs.append(curr)
    prefill_last = curr

    total_decode_ms = 0.0
    curr = None
    decode_outputs: list[dict[str, np.ndarray]] = []
    decode_feeds = decode_feeds_builder(prefill_outputs)
    for model, feed in zip(decode_models, decode_feeds):
        actual_feed = (
            feed
            if curr is None
            else {k: resolve_value(v, curr, decode_outputs) for k, v in feed.items()}
        )
        curr, ms = time_predict(model, actual_feed, warmup, iters)
        total_decode_ms += ms
        decode_outputs.append(curr)
    decode_last = curr
    merged_prefill = {}
    for out in prefill_outputs:
        merged_prefill.update(out)

    return VariantResult(
        name=name,
        prefill_ms=total_prefill_ms,
        decode_ms=total_decode_ms,
        prefill_metric=metric_line("prefill_attn", prefill_last["attn_out"], torch_prefill_attn),
        decode_metric=metric_line("decode_attn", decode_last["attn_out"], torch_decode_attn),
        conv_prefill_metric=metric_line("prefill_conv", merged_prefill["next_conv"], torch_prefill_conv),
        rec_prefill_metric=metric_line("prefill_rec", merged_prefill["next_rec"], torch_prefill_rec),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compute-unit", default="ALL")
    parser.add_argument("--pass-pipeline", choices=["default", "empty"], default="default")
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    model_path = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
    layer_idx = 30
    seq_len = args.seq_len

    cfg = Qwen35Config.from_json(os.path.join(model_path, "config.json"))
    model = Qwen35ForCausalLM(cfg).half().eval()
    if not model.load_pretrained_weights(model_path):
        raise RuntimeError("failed to load weights")
    attn = model.model.layers[layer_idx].self_attn

    hf_cfg = Qwen3_5Config.from_pretrained(model_path).text_config
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, layer_idx=layer_idx).half().eval()

    torch.manual_seed(7)
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32)
    hidden_decode = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32)
    x = hf_layer.input_layernorm(hidden).half()
    x_decode = hf_layer.input_layernorm(hidden_decode).half()
    conv_state = torch.zeros((1, attn.conv_dim, attn.linear_conv_kernel_dim), dtype=torch.float16)
    recurrent_state = torch.zeros((1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim), dtype=torch.float16)

    with torch.no_grad():
        full_prefill, full_next_conv, full_next_rec = attn.forward_prefill(
            hidden_states=x,
            conv_state=conv_state,
            recurrent_state=recurrent_state.to(torch.float32),
            has_previous_state=False,
            force_recurrent=True,
            force_fp16_math=True,
        )
        full_decode, _, _ = attn.forward_regular(
            hidden_states=x_decode,
            conv_state=full_next_conv,
            recurrent_state=full_next_rec,
            has_previous_state=True,
            force_recurrent=True,
            force_fp16_math=True,
        )

        pcl = ProjConvLayoutGroup(attn, seq_len).eval()
        cn = CoreNormGroup(attn, seq_len).eval()
        pc = ProjConvGroup(attn, seq_len).eval()
        lcn = LayoutCoreNormGroup(attn, seq_len).eval()
        pclc = ProjConvLayoutCoreGroup(attn, seq_len).eval()
        norm = NormOutStage(attn, seq_len).eval()

        pcl_d = ProjConvLayoutGroup(attn, 1).eval()
        cn_d = CoreNormGroup(attn, 1).eval()
        pc_d = ProjConvGroup(attn, 1).eval()
        lcn_d = LayoutCoreNormGroup(attn, 1).eval()
        pclc_d = ProjConvLayoutCoreGroup(attn, 1).eval()
        norm_d = NormOutStage(attn, 1).eval()

        q, k, v, beta, g, z_out, next_conv = pcl(x, conv_state)
        group_prefill_attn, group_prefill_rec = cn(q, k, v, beta, g, z_out, recurrent_state)
        qd, kd, vd, betad, gd, z_out_d, _ = pcl_d(x_decode, next_conv)
        group_decode_attn, _ = cn_d(qd, kd, vd, betad, gd, z_out_d, group_prefill_rec)

    compute_units = getattr(ct.ComputeUnit, args.compute_unit)
    pass_pipeline = None if args.pass_pipeline == "default" else ct.PassPipeline.EMPTY

    results = []

    if args.only in ("", "2stage_pcl_cn"):
        results.append(
        run_variant(
            name="2stage_pcl_cn",
            prefill_modules=[
                (
                    pcl,
                    (x, conv_state),
                    [
                        ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
                    ],
                    ["query", "key", "value", "beta", "g", "z_out", "next_conv"],
                    "qwen35_lin_grp_pcl_prefill.mlpackage",
                ),
                (
                    cn,
                    (q, k, v, beta, g, z_out, recurrent_state),
                    [
                        ct.TensorType(name="query", shape=q.shape, dtype=np.float16),
                        ct.TensorType(name="key", shape=k.shape, dtype=np.float16),
                        ct.TensorType(name="value", shape=v.shape, dtype=np.float16),
                        ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
                        ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
                        ct.TensorType(name="z_out", shape=z_out.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
                    ],
                    ["attn_out", "next_rec"],
                    "qwen35_lin_grp_cn_prefill.mlpackage",
                ),
            ],
            decode_modules=[
                (
                    pcl_d,
                    (x_decode, next_conv),
                    [
                        ct.TensorType(name="hidden_states", shape=x_decode.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=next_conv.shape, dtype=np.float16),
                    ],
                    ["query", "key", "value", "beta", "g", "z_out", "next_conv"],
                    "qwen35_lin_grp_pcl_decode.mlpackage",
                ),
                (
                    cn_d,
                    (qd, kd, vd, betad, gd, z_out_d, group_prefill_rec),
                    [
                        ct.TensorType(name="query", shape=qd.shape, dtype=np.float16),
                        ct.TensorType(name="key", shape=kd.shape, dtype=np.float16),
                        ct.TensorType(name="value", shape=vd.shape, dtype=np.float16),
                        ct.TensorType(name="beta", shape=betad.shape, dtype=np.float16),
                        ct.TensorType(name="g", shape=gd.shape, dtype=np.float16),
                        ct.TensorType(name="z_out", shape=z_out_d.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=group_prefill_rec.shape, dtype=np.float16),
                    ],
                    ["attn_out", "next_rec"],
                    "qwen35_lin_grp_cn_decode.mlpackage",
                ),
            ],
            prefill_feeds=[
                {"hidden_states": to_np(x), "conv_state": to_np(conv_state)},
                {
                    "query": "query",
                    "key": "key",
                    "value": "value",
                    "beta": "beta",
                    "g": "g",
                    "z_out": "z_out",
                    "recurrent_state": to_np(recurrent_state),
                },
            ],
            decode_feeds_builder=lambda ps: [
                {"hidden_states": to_np(x_decode), "conv_state": ps[0]["next_conv"]},
                {
                    "query": "query",
                    "key": "key",
                    "value": "value",
                    "beta": "beta",
                    "g": "g",
                    "z_out": "z_out",
                    "recurrent_state": ps[-1]["next_rec"],
                },
            ],
            torch_prefill_attn=full_prefill,
            torch_decode_attn=full_decode,
            torch_prefill_conv=full_next_conv,
            torch_prefill_rec=full_next_rec,
            warmup=args.warmup,
            iters=args.iters,
            compute_units=compute_units,
            pass_pipeline=pass_pipeline,
        )
        )

    if args.only in ("", "2stage_pc_lcn"):
        results.append(
        run_variant(
            name="2stage_pc_lcn",
            prefill_modules=[
                (
                    pc,
                    (x, conv_state),
                    [
                        ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
                    ],
                    ["conv_out", "b", "a", "z", "next_conv"],
                    "qwen35_lin_grp_pc_prefill.mlpackage",
                ),
                (
                    lcn,
                    (pc(x, conv_state)[0], pc(x, conv_state)[1], pc(x, conv_state)[2], pc(x, conv_state)[3], recurrent_state),
                    [
                        ct.TensorType(name="conv_out", shape=pc(x, conv_state)[0].shape, dtype=np.float16),
                        ct.TensorType(name="b", shape=pc(x, conv_state)[1].shape, dtype=np.float16),
                        ct.TensorType(name="a", shape=pc(x, conv_state)[2].shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=pc(x, conv_state)[3].shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
                    ],
                    ["attn_out", "next_rec"],
                    "qwen35_lin_grp_lcn_prefill.mlpackage",
                ),
            ],
            decode_modules=[
                (
                    pc_d,
                    (x_decode, next_conv),
                    [
                        ct.TensorType(name="hidden_states", shape=x_decode.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=next_conv.shape, dtype=np.float16),
                    ],
                    ["conv_out", "b", "a", "z", "next_conv"],
                    "qwen35_lin_grp_pc_decode.mlpackage",
                ),
                (
                    lcn_d,
                    (pc_d(x_decode, next_conv)[0], pc_d(x_decode, next_conv)[1], pc_d(x_decode, next_conv)[2], pc_d(x_decode, next_conv)[3], group_prefill_rec),
                    [
                        ct.TensorType(name="conv_out", shape=pc_d(x_decode, next_conv)[0].shape, dtype=np.float16),
                        ct.TensorType(name="b", shape=pc_d(x_decode, next_conv)[1].shape, dtype=np.float16),
                        ct.TensorType(name="a", shape=pc_d(x_decode, next_conv)[2].shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=pc_d(x_decode, next_conv)[3].shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=group_prefill_rec.shape, dtype=np.float16),
                    ],
                    ["attn_out", "next_rec"],
                    "qwen35_lin_grp_lcn_decode.mlpackage",
                ),
            ],
            prefill_feeds=[
                {"hidden_states": to_np(x), "conv_state": to_np(conv_state)},
                {
                    "conv_out": "conv_out",
                    "b": "b",
                    "a": "a",
                    "z": "z",
                    "recurrent_state": to_np(recurrent_state),
                },
            ],
            decode_feeds_builder=lambda ps: [
                {"hidden_states": to_np(x_decode), "conv_state": ps[0]["next_conv"]},
                {
                    "conv_out": "conv_out",
                    "b": "b",
                    "a": "a",
                    "z": "z",
                    "recurrent_state": ps[-1]["next_rec"],
                },
            ],
            torch_prefill_attn=full_prefill,
            torch_decode_attn=full_decode,
            torch_prefill_conv=full_next_conv,
            torch_prefill_rec=full_next_rec,
            warmup=args.warmup,
            iters=args.iters,
            compute_units=compute_units,
            pass_pipeline=pass_pipeline,
        )
        )

    if args.only in ("", "2stage_pclc_n"):
        results.append(
        run_variant(
            name="2stage_pclc_n",
            prefill_modules=[
                (
                    pclc,
                    (x, conv_state, recurrent_state),
                    [
                        ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
                    ],
                    ["core", "z_out", "next_conv", "next_rec"],
                    "qwen35_lin_grp_pclc_prefill.mlpackage",
                ),
                (
                    norm,
                    (pclc(x, conv_state, recurrent_state)[0], pclc(x, conv_state, recurrent_state)[1]),
                    [
                        ct.TensorType(name="core", shape=pclc(x, conv_state, recurrent_state)[0].shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=pclc(x, conv_state, recurrent_state)[1].shape, dtype=np.float16),
                    ],
                    ["attn_out"],
                    "qwen35_lin_grp_norm_prefill.mlpackage",
                ),
            ],
            decode_modules=[
                (
                    pclc_d,
                    (x_decode, next_conv, group_prefill_rec),
                    [
                        ct.TensorType(name="hidden_states", shape=x_decode.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=next_conv.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=group_prefill_rec.shape, dtype=np.float16),
                    ],
                    ["core", "z_out", "next_conv", "next_rec"],
                    "qwen35_lin_grp_pclc_decode.mlpackage",
                ),
                (
                    norm_d,
                    (pclc_d(x_decode, next_conv, group_prefill_rec)[0], pclc_d(x_decode, next_conv, group_prefill_rec)[1]),
                    [
                        ct.TensorType(name="core", shape=pclc_d(x_decode, next_conv, group_prefill_rec)[0].shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=pclc_d(x_decode, next_conv, group_prefill_rec)[1].shape, dtype=np.float16),
                    ],
                    ["attn_out"],
                    "qwen35_lin_grp_norm_decode.mlpackage",
                ),
            ],
            prefill_feeds=[
                {
                    "hidden_states": to_np(x),
                    "conv_state": to_np(conv_state),
                    "recurrent_state": to_np(recurrent_state),
                },
                {"core": "core", "z": "z_out"},
            ],
            decode_feeds_builder=lambda ps: [
                {
                    "hidden_states": to_np(x_decode),
                    "conv_state": ps[0]["next_conv"],
                    "recurrent_state": ps[0]["next_rec"],
                },
                {"core": "core", "z": "z_out"},
            ],
            torch_prefill_attn=full_prefill,
            torch_decode_attn=full_decode,
            torch_prefill_conv=full_next_conv,
            torch_prefill_rec=full_next_rec,
            warmup=args.warmup,
            iters=args.iters,
            compute_units=compute_units,
            pass_pipeline=pass_pipeline,
        )
        )

    if args.only in ("", "3stage_pcl_c_n"):
        results.append(
        run_variant(
            name="3stage_pcl_c_n",
            prefill_modules=[
                (
                    pcl,
                    (x, conv_state),
                    [
                        ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
                    ],
                    ["query", "key", "value", "beta", "g", "z_out", "next_conv"],
                    "qwen35_lin_grp3_pcl_prefill.mlpackage",
                ),
                (
                    RecurrentCoreStage(attn, seq_len).eval(),
                    (q, k, v, beta, g, recurrent_state),
                    [
                        ct.TensorType(name="query", shape=q.shape, dtype=np.float16),
                        ct.TensorType(name="key", shape=k.shape, dtype=np.float16),
                        ct.TensorType(name="value", shape=v.shape, dtype=np.float16),
                        ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
                        ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
                    ],
                    ["core", "next_rec"],
                    "qwen35_lin_grp3_core_prefill.mlpackage",
                ),
                (
                    norm,
                    (group_prefill_rec.new_zeros((1, seq_len, attn.value_dim)).reshape(1, seq_len, attn.num_v_heads, attn.head_v_dim), z_out),
                    [
                        ct.TensorType(name="core", shape=group_prefill_rec.new_zeros((1, seq_len, attn.value_dim)).reshape(1, seq_len, attn.num_v_heads, attn.head_v_dim).shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=z_out.shape, dtype=np.float16),
                    ],
                    ["attn_out"],
                    "qwen35_lin_grp3_norm_prefill.mlpackage",
                ),
            ],
            decode_modules=[
                (
                    pcl_d,
                    (x_decode, next_conv),
                    [
                        ct.TensorType(name="hidden_states", shape=x_decode.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=next_conv.shape, dtype=np.float16),
                    ],
                    ["query", "key", "value", "beta", "g", "z_out", "next_conv"],
                    "qwen35_lin_grp3_pcl_decode.mlpackage",
                ),
                (
                    RecurrentCoreStage(attn, 1).eval(),
                    (qd, kd, vd, betad, gd, group_prefill_rec),
                    [
                        ct.TensorType(name="query", shape=qd.shape, dtype=np.float16),
                        ct.TensorType(name="key", shape=kd.shape, dtype=np.float16),
                        ct.TensorType(name="value", shape=vd.shape, dtype=np.float16),
                        ct.TensorType(name="beta", shape=betad.shape, dtype=np.float16),
                        ct.TensorType(name="g", shape=gd.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=group_prefill_rec.shape, dtype=np.float16),
                    ],
                    ["core", "next_rec"],
                    "qwen35_lin_grp3_core_decode.mlpackage",
                ),
                (
                    norm_d,
                    (group_prefill_rec.new_zeros((1, 1, attn.value_dim)).reshape(1, 1, attn.num_v_heads, attn.head_v_dim), z_out_d),
                    [
                        ct.TensorType(name="core", shape=group_prefill_rec.new_zeros((1, 1, attn.value_dim)).reshape(1, 1, attn.num_v_heads, attn.head_v_dim).shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=z_out_d.shape, dtype=np.float16),
                    ],
                    ["attn_out"],
                    "qwen35_lin_grp3_norm_decode.mlpackage",
                ),
            ],
            prefill_feeds=[
                {"hidden_states": to_np(x), "conv_state": to_np(conv_state)},
                {
                    "query": "query",
                    "key": "key",
                    "value": "value",
                    "beta": "beta",
                    "g": "g",
                    "recurrent_state": to_np(recurrent_state),
                },
                {"core": "core", "z": (0, "z_out")},
            ],
            decode_feeds_builder=lambda ps: [
                {"hidden_states": to_np(x_decode), "conv_state": ps[0]["next_conv"]},
                {
                    "query": "query",
                    "key": "key",
                    "value": "value",
                    "beta": "beta",
                    "g": "g",
                    "recurrent_state": ps[1]["next_rec"],
                },
                {"core": "core", "z": (0, "z_out")},
            ],
            torch_prefill_attn=full_prefill,
            torch_decode_attn=full_decode,
            torch_prefill_conv=full_next_conv,
            torch_prefill_rec=full_next_rec,
            warmup=args.warmup,
            iters=args.iters,
            compute_units=compute_units,
            pass_pipeline=pass_pipeline,
        )
        )

    if args.only in ("", "4stage_pc_l_c_n"):
        results.append(
        run_variant(
            name="4stage_pc_l_c_n",
            prefill_modules=[
                (
                    pc,
                    (x, conv_state),
                    [
                        ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
                    ],
                    ["conv_out", "b", "a", "z", "next_conv"],
                    "qwen35_lin_grp4_pc_prefill.mlpackage",
                ),
                (
                    LayoutStage(attn, seq_len).eval(),
                    (pc(x, conv_state)[0], pc(x, conv_state)[1], pc(x, conv_state)[2], pc(x, conv_state)[3]),
                    [
                        ct.TensorType(name="conv_out", shape=pc(x, conv_state)[0].shape, dtype=np.float16),
                        ct.TensorType(name="b", shape=pc(x, conv_state)[1].shape, dtype=np.float16),
                        ct.TensorType(name="a", shape=pc(x, conv_state)[2].shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=pc(x, conv_state)[3].shape, dtype=np.float16),
                    ],
                    ["query", "key", "value", "beta", "g", "z_out"],
                    "qwen35_lin_grp4_l_prefill.mlpackage",
                ),
                (
                    RecurrentCoreStage(attn, seq_len).eval(),
                    (q, k, v, beta, g, recurrent_state),
                    [
                        ct.TensorType(name="query", shape=q.shape, dtype=np.float16),
                        ct.TensorType(name="key", shape=k.shape, dtype=np.float16),
                        ct.TensorType(name="value", shape=v.shape, dtype=np.float16),
                        ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
                        ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
                    ],
                    ["core", "next_rec"],
                    "qwen35_lin_grp4_c_prefill.mlpackage",
                ),
                (
                    norm,
                    (group_prefill_rec.new_zeros((1, seq_len, attn.value_dim)).reshape(1, seq_len, attn.num_v_heads, attn.head_v_dim), z_out),
                    [
                        ct.TensorType(name="core", shape=group_prefill_rec.new_zeros((1, seq_len, attn.value_dim)).reshape(1, seq_len, attn.num_v_heads, attn.head_v_dim).shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=z_out.shape, dtype=np.float16),
                    ],
                    ["attn_out"],
                    "qwen35_lin_grp4_n_prefill.mlpackage",
                ),
            ],
            decode_modules=[
                (
                    pc_d,
                    (x_decode, next_conv),
                    [
                        ct.TensorType(name="hidden_states", shape=x_decode.shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=next_conv.shape, dtype=np.float16),
                    ],
                    ["conv_out", "b", "a", "z", "next_conv"],
                    "qwen35_lin_grp4_pc_decode.mlpackage",
                ),
                (
                    LayoutStage(attn, 1).eval(),
                    (pc_d(x_decode, next_conv)[0], pc_d(x_decode, next_conv)[1], pc_d(x_decode, next_conv)[2], pc_d(x_decode, next_conv)[3]),
                    [
                        ct.TensorType(name="conv_out", shape=pc_d(x_decode, next_conv)[0].shape, dtype=np.float16),
                        ct.TensorType(name="b", shape=pc_d(x_decode, next_conv)[1].shape, dtype=np.float16),
                        ct.TensorType(name="a", shape=pc_d(x_decode, next_conv)[2].shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=pc_d(x_decode, next_conv)[3].shape, dtype=np.float16),
                    ],
                    ["query", "key", "value", "beta", "g", "z_out"],
                    "qwen35_lin_grp4_l_decode.mlpackage",
                ),
                (
                    RecurrentCoreStage(attn, 1).eval(),
                    (qd, kd, vd, betad, gd, group_prefill_rec),
                    [
                        ct.TensorType(name="query", shape=qd.shape, dtype=np.float16),
                        ct.TensorType(name="key", shape=kd.shape, dtype=np.float16),
                        ct.TensorType(name="value", shape=vd.shape, dtype=np.float16),
                        ct.TensorType(name="beta", shape=betad.shape, dtype=np.float16),
                        ct.TensorType(name="g", shape=gd.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=group_prefill_rec.shape, dtype=np.float16),
                    ],
                    ["core", "next_rec"],
                    "qwen35_lin_grp4_c_decode.mlpackage",
                ),
                (
                    norm_d,
                    (group_prefill_rec.new_zeros((1, 1, attn.value_dim)).reshape(1, 1, attn.num_v_heads, attn.head_v_dim), z_out_d),
                    [
                        ct.TensorType(name="core", shape=group_prefill_rec.new_zeros((1, 1, attn.value_dim)).reshape(1, 1, attn.num_v_heads, attn.head_v_dim).shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=z_out_d.shape, dtype=np.float16),
                    ],
                    ["attn_out"],
                    "qwen35_lin_grp4_n_decode.mlpackage",
                ),
            ],
            prefill_feeds=[
                {"hidden_states": to_np(x), "conv_state": to_np(conv_state)},
                {"conv_out": "conv_out", "b": "b", "a": "a", "z": "z"},
                {
                    "query": "query",
                    "key": "key",
                    "value": "value",
                    "beta": "beta",
                    "g": "g",
                    "recurrent_state": to_np(recurrent_state),
                },
                {"core": "core", "z": (1, "z_out")},
            ],
            decode_feeds_builder=lambda ps: [
                {"hidden_states": to_np(x_decode), "conv_state": ps[0]["next_conv"]},
                {"conv_out": "conv_out", "b": "b", "a": "a", "z": "z"},
                {
                    "query": "query",
                    "key": "key",
                    "value": "value",
                    "beta": "beta",
                    "g": "g",
                    "recurrent_state": ps[2]["next_rec"],
                },
                {"core": "core", "z": (1, "z_out")},
            ],
            torch_prefill_attn=full_prefill,
            torch_decode_attn=full_decode,
            torch_prefill_conv=full_next_conv,
            torch_prefill_rec=full_next_rec,
            warmup=args.warmup,
            iters=args.iters,
            compute_units=compute_units,
            pass_pipeline=pass_pipeline,
        )
        )

    if args.only in ("", "4stage_p_c_l_cn"):
        results.append(
        run_variant(
            name="4stage_p_c_l_cn",
            prefill_modules=[
                (
                    QKVProjStage(attn).eval(),
                    (x,),
                    [ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16)],
                    ["mixed_qkv", "z", "b", "a"],
                    "qwen35_lin_grp4b_p_prefill.mlpackage",
                ),
                (
                    CausalConvStage(attn, seq_len).eval(),
                    (pc(x, conv_state)[0], conv_state),
                    [
                        ct.TensorType(name="mixed_qkv", shape=pc(x, conv_state)[0].shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
                    ],
                    ["conv_out", "next_conv"],
                    "qwen35_lin_grp4b_c_prefill.mlpackage",
                ),
                (
                    LayoutStage(attn, seq_len).eval(),
                    (pc(x, conv_state)[0], pc(x, conv_state)[1], pc(x, conv_state)[2], pc(x, conv_state)[3]),
                    [
                        ct.TensorType(name="conv_out", shape=pc(x, conv_state)[0].shape, dtype=np.float16),
                        ct.TensorType(name="b", shape=pc(x, conv_state)[1].shape, dtype=np.float16),
                        ct.TensorType(name="a", shape=pc(x, conv_state)[2].shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=pc(x, conv_state)[3].shape, dtype=np.float16),
                    ],
                    ["query", "key", "value", "beta", "g", "z_out"],
                    "qwen35_lin_grp4b_l_prefill.mlpackage",
                ),
                (
                    cn,
                    (q, k, v, beta, g, z_out, recurrent_state),
                    [
                        ct.TensorType(name="query", shape=q.shape, dtype=np.float16),
                        ct.TensorType(name="key", shape=k.shape, dtype=np.float16),
                        ct.TensorType(name="value", shape=v.shape, dtype=np.float16),
                        ct.TensorType(name="beta", shape=beta.shape, dtype=np.float16),
                        ct.TensorType(name="g", shape=g.shape, dtype=np.float16),
                        ct.TensorType(name="z_out", shape=z_out.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
                    ],
                    ["attn_out", "next_rec"],
                    "qwen35_lin_grp4b_cn_prefill.mlpackage",
                ),
            ],
            decode_modules=[
                (
                    QKVProjStage(attn).eval(),
                    (x_decode,),
                    [ct.TensorType(name="hidden_states", shape=x_decode.shape, dtype=np.float16)],
                    ["mixed_qkv", "z", "b", "a"],
                    "qwen35_lin_grp4b_p_decode.mlpackage",
                ),
                (
                    CausalConvStage(attn, 1).eval(),
                    (pc_d(x_decode, next_conv)[0], next_conv),
                    [
                        ct.TensorType(name="mixed_qkv", shape=pc_d(x_decode, next_conv)[0].shape, dtype=np.float16),
                        ct.TensorType(name="conv_state", shape=next_conv.shape, dtype=np.float16),
                    ],
                    ["conv_out", "next_conv"],
                    "qwen35_lin_grp4b_c_decode.mlpackage",
                ),
                (
                    LayoutStage(attn, 1).eval(),
                    (pc_d(x_decode, next_conv)[0], pc_d(x_decode, next_conv)[1], pc_d(x_decode, next_conv)[2], pc_d(x_decode, next_conv)[3]),
                    [
                        ct.TensorType(name="conv_out", shape=pc_d(x_decode, next_conv)[0].shape, dtype=np.float16),
                        ct.TensorType(name="b", shape=pc_d(x_decode, next_conv)[1].shape, dtype=np.float16),
                        ct.TensorType(name="a", shape=pc_d(x_decode, next_conv)[2].shape, dtype=np.float16),
                        ct.TensorType(name="z", shape=pc_d(x_decode, next_conv)[3].shape, dtype=np.float16),
                    ],
                    ["query", "key", "value", "beta", "g", "z_out"],
                    "qwen35_lin_grp4b_l_decode.mlpackage",
                ),
                (
                    cn_d,
                    (qd, kd, vd, betad, gd, z_out_d, group_prefill_rec),
                    [
                        ct.TensorType(name="query", shape=qd.shape, dtype=np.float16),
                        ct.TensorType(name="key", shape=kd.shape, dtype=np.float16),
                        ct.TensorType(name="value", shape=vd.shape, dtype=np.float16),
                        ct.TensorType(name="beta", shape=betad.shape, dtype=np.float16),
                        ct.TensorType(name="g", shape=gd.shape, dtype=np.float16),
                        ct.TensorType(name="z_out", shape=z_out_d.shape, dtype=np.float16),
                        ct.TensorType(name="recurrent_state", shape=group_prefill_rec.shape, dtype=np.float16),
                    ],
                    ["attn_out", "next_rec"],
                    "qwen35_lin_grp4b_cn_decode.mlpackage",
                ),
            ],
            prefill_feeds=[
                {"hidden_states": to_np(x)},
                {"mixed_qkv": "mixed_qkv", "conv_state": to_np(conv_state)},
                {"conv_out": "conv_out", "b": (0, "b"), "a": (0, "a"), "z": (0, "z")},
                {
                    "query": "query",
                    "key": "key",
                    "value": "value",
                    "beta": "beta",
                    "g": "g",
                    "z_out": "z_out",
                    "recurrent_state": to_np(recurrent_state),
                },
            ],
            decode_feeds_builder=lambda ps: [
                {"hidden_states": to_np(x_decode)},
                {"mixed_qkv": "mixed_qkv", "conv_state": ps[1]["next_conv"]},
                {"conv_out": "conv_out", "b": (0, "b"), "a": (0, "a"), "z": (0, "z")},
                {
                    "query": "query",
                    "key": "key",
                    "value": "value",
                    "beta": "beta",
                    "g": "g",
                    "z_out": "z_out",
                    "recurrent_state": ps[-1]["next_rec"],
                },
            ],
            torch_prefill_attn=full_prefill,
            torch_decode_attn=full_decode,
            torch_prefill_conv=full_next_conv,
            torch_prefill_rec=full_next_rec,
            warmup=args.warmup,
            iters=args.iters,
            compute_units=compute_units,
            pass_pipeline=pass_pipeline,
        )
        )

    for result in results:
        print(f"=== {result.name} ===")
        print(f"prefill_mean_ms={result.prefill_ms:.3f} decode_mean_ms={result.decode_ms:.3f}")
        print(result.prefill_metric)
        print(result.decode_metric)
        print(result.conv_prefill_metric)
        print(result.rec_prefill_metric)


if __name__ == "__main__":
    main()
