#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import coremltools as ct

from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from tests.dev.tmp_linear_layout_probe import LayoutStage
from tests.dev.tmp_linear_proj_probe import CausalConvStage, QKVProjStage
from tests.dev.tmp_linear_stage_probe import NormOutStage, RecurrentCoreStage, metrics, print_metric


def convert_model(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    specs: list[ct.TensorType],
    output_names: list[str],
    package_name: str,
) -> ct.models.MLModel:
    traced = torch.jit.trace(module, inputs, strict=False, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=specs,
        outputs=[ct.TensorType(name=name) for name in output_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )
    pkg = os.path.join("/tmp", package_name)
    shutil.rmtree(pkg, ignore_errors=True)
    mlmodel.save(pkg)
    return ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.ALL)


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def main() -> None:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    model_path = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
    layer_idx = 30
    seq_len = 32

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
        proj_stage = QKVProjStage(attn).eval()
        conv_stage = CausalConvStage(attn, seq_len).eval()
        layout_stage = LayoutStage(attn, seq_len).eval()
        core_stage = RecurrentCoreStage(attn, seq_len).eval()
        norm_stage = NormOutStage(attn, seq_len).eval()
        proj_stage_dec = QKVProjStage(attn).eval()
        conv_stage_dec = CausalConvStage(attn, 1).eval()
        layout_stage_dec = LayoutStage(attn, 1).eval()
        core_stage_dec = RecurrentCoreStage(attn, 1).eval()
        norm_stage_dec = NormOutStage(attn, 1).eval()

        tmixed, tz0, tb0, ta0 = proj_stage(x)
        tconv_out, tnext_conv = conv_stage(tmixed, conv_state)
        tq, tk, tv, tb, tg, tz = layout_stage(tconv_out, tb0, ta0, tz0)
        tcore, tnext_rec = core_stage(tq, tk, tv, tb, tg, recurrent_state)
        tout = norm_stage(tcore, tz)

        tmixed_d, tz0_d, tb0_d, ta0_d = proj_stage_dec(x_decode)
        tconv_out_d, tnext_conv_d = conv_stage_dec(tmixed_d, tnext_conv)
        tq_d, tk_d, tv_d, tb_d, tg_d, tz_d = layout_stage_dec(tconv_out_d, tb0_d, ta0_d, tz0_d)
        tcore_d, tnext_rec_d = core_stage_dec(tq_d, tk_d, tv_d, tb_d, tg_d, tnext_rec)
        tout_d = norm_stage_dec(tcore_d, tz_d)

    m_proj = convert_model(
        proj_stage,
        (x,),
        [ct.TensorType(name="hidden_states", shape=x.shape, dtype=np.float16)],
        ["mixed_qkv", "z", "b", "a"],
        "qwen35_linear_pipe_proj.mlpackage",
    )
    m_conv = convert_model(
        conv_stage,
        (tmixed, conv_state),
        [
            ct.TensorType(name="mixed_qkv", shape=tmixed.shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=conv_state.shape, dtype=np.float16),
        ],
        ["conv_out", "next_conv"],
        "qwen35_linear_pipe_conv.mlpackage",
    )
    m_layout = convert_model(
        layout_stage,
        (tconv_out, tb0, ta0, tz0),
        [
            ct.TensorType(name="conv_out", shape=tconv_out.shape, dtype=np.float16),
            ct.TensorType(name="b", shape=tb0.shape, dtype=np.float16),
            ct.TensorType(name="a", shape=ta0.shape, dtype=np.float16),
            ct.TensorType(name="z", shape=tz0.shape, dtype=np.float16),
        ],
        ["query", "key", "value", "beta", "g", "z_out"],
        "qwen35_linear_pipe_layout.mlpackage",
    )
    m_core = convert_model(
        core_stage,
        (tq, tk, tv, tb, tg, recurrent_state),
        [
            ct.TensorType(name="query", shape=tq.shape, dtype=np.float16),
            ct.TensorType(name="key", shape=tk.shape, dtype=np.float16),
            ct.TensorType(name="value", shape=tv.shape, dtype=np.float16),
            ct.TensorType(name="beta", shape=tb.shape, dtype=np.float16),
            ct.TensorType(name="g", shape=tg.shape, dtype=np.float16),
            ct.TensorType(name="recurrent_state", shape=recurrent_state.shape, dtype=np.float16),
        ],
        ["core", "next_rec"],
        "qwen35_linear_pipe_core.mlpackage",
    )
    m_norm = convert_model(
        norm_stage,
        (tcore, tz),
        [
            ct.TensorType(name="core", shape=tcore.shape, dtype=np.float16),
            ct.TensorType(name="z", shape=tz.shape, dtype=np.float16),
        ],
        ["attn_out"],
        "qwen35_linear_pipe_norm.mlpackage",
    )
    m_proj_d = convert_model(
        proj_stage_dec,
        (x_decode,),
        [ct.TensorType(name="hidden_states", shape=x_decode.shape, dtype=np.float16)],
        ["mixed_qkv", "z", "b", "a"],
        "qwen35_linear_pipe_proj_dec.mlpackage",
    )
    m_conv_d = convert_model(
        conv_stage_dec,
        (tmixed_d, tnext_conv),
        [
            ct.TensorType(name="mixed_qkv", shape=tmixed_d.shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=tnext_conv.shape, dtype=np.float16),
        ],
        ["conv_out", "next_conv"],
        "qwen35_linear_pipe_conv_dec.mlpackage",
    )
    m_layout_d = convert_model(
        layout_stage_dec,
        (tconv_out_d, tb0_d, ta0_d, tz0_d),
        [
            ct.TensorType(name="conv_out", shape=tconv_out_d.shape, dtype=np.float16),
            ct.TensorType(name="b", shape=tb0_d.shape, dtype=np.float16),
            ct.TensorType(name="a", shape=ta0_d.shape, dtype=np.float16),
            ct.TensorType(name="z", shape=tz0_d.shape, dtype=np.float16),
        ],
        ["query", "key", "value", "beta", "g", "z_out"],
        "qwen35_linear_pipe_layout_dec.mlpackage",
    )
    m_core_d = convert_model(
        core_stage_dec,
        (tq_d, tk_d, tv_d, tb_d, tg_d, tnext_rec),
        [
            ct.TensorType(name="query", shape=tq_d.shape, dtype=np.float16),
            ct.TensorType(name="key", shape=tk_d.shape, dtype=np.float16),
            ct.TensorType(name="value", shape=tv_d.shape, dtype=np.float16),
            ct.TensorType(name="beta", shape=tb_d.shape, dtype=np.float16),
            ct.TensorType(name="g", shape=tg_d.shape, dtype=np.float16),
            ct.TensorType(name="recurrent_state", shape=tnext_rec.shape, dtype=np.float16),
        ],
        ["core", "next_rec"],
        "qwen35_linear_pipe_core_dec.mlpackage",
    )
    m_norm_d = convert_model(
        norm_stage_dec,
        (tcore_d, tz_d),
        [
            ct.TensorType(name="core", shape=tcore_d.shape, dtype=np.float16),
            ct.TensorType(name="z", shape=tz_d.shape, dtype=np.float16),
        ],
        ["attn_out"],
        "qwen35_linear_pipe_norm_dec.mlpackage",
    )

    p0 = m_proj.predict({"hidden_states": to_np(x)})
    p1 = m_conv.predict({"mixed_qkv": p0["mixed_qkv"], "conv_state": to_np(conv_state)})
    p2 = m_layout.predict({"conv_out": p1["conv_out"], "b": p0["b"], "a": p0["a"], "z": p0["z"]})
    p3 = m_core.predict(
        {
            "query": p2["query"],
            "key": p2["key"],
            "value": p2["value"],
            "beta": p2["beta"],
            "g": p2["g"],
            "recurrent_state": to_np(recurrent_state),
        }
    )
    p4 = m_norm.predict({"core": p3["core"], "z": p2["z_out"]})

    d0 = m_proj_d.predict({"hidden_states": to_np(x_decode)})
    d1 = m_conv_d.predict({"mixed_qkv": d0["mixed_qkv"], "conv_state": p1["next_conv"]})
    d2 = m_layout_d.predict({"conv_out": d1["conv_out"], "b": d0["b"], "a": d0["a"], "z": d0["z"]})
    d3 = m_core_d.predict(
        {
            "query": d2["query"],
            "key": d2["key"],
            "value": d2["value"],
            "beta": d2["beta"],
            "g": d2["g"],
            "recurrent_state": p3["next_rec"],
        }
    )
    d4 = m_norm_d.predict({"core": d3["core"], "z": d2["z_out"]})

    print("Pipeline Prefill vs Torch")
    print_metric("attn_out", metrics(p4["attn_out"], tout))
    print_metric("next_conv", metrics(p1["next_conv"], tnext_conv))
    print_metric("next_rec", metrics(p3["next_rec"], tnext_rec))
    print("Pipeline Decode vs Torch")
    print_metric("attn_out", metrics(d4["attn_out"], tout_d))
    print_metric("next_conv", metrics(d1["next_conv"], tnext_conv_d))
    print_metric("next_rec", metrics(d3["next_rec"], tnext_rec_d))


if __name__ == "__main__":
    main()
