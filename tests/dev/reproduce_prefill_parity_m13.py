#!/usr/bin/env python3
"""Reproduce runbook-style prefill parity on milestone1_3 naming.

Runs two-phase parity:
1) PyTorch references (embed + chunk1..chunk4 outputs)
2) CoreML prefill comparison in isolated and cascaded modes

Designed for model dirs with:
- embeddings.mlpackage
- prefill_LUT4_chunk{0,1,2,3}.mlpackage
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from anemll.models.qwen3_5_model import (
    MODEL_DTYPE,
    Qwen35Config,
    Qwen35ForCausalLM,
    ane_conv_state_shape,
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.reshape(-1).astype(np.float64)
    b_f = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


def make_prompt_ids(tokenizer: AutoTokenizer, prompt: str, seq_len: int) -> torch.Tensor:
    text = prompt
    while True:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
        if ids.shape[1] >= seq_len:
            return ids[:, :seq_len]
        text = text + " " + prompt


def make_causal_mask(seq_len: int, ctx_len: int) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, ctx_len), float("-inf"), dtype=torch.float16)
    for r in range(seq_len):
        mask[0, 0, r, : r + 1] = 0
    return mask


def chunk_ranges(num_layers: int, num_chunks: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(num_chunks):
        end = start + (num_layers // num_chunks) if i < num_chunks - 1 else num_layers
        ranges.append((start, end))
        start = end
    return ranges


def generate_torch_refs(
    hf_model_path: Path,
    prompt: str,
    seq_len: int,
    ctx_len: int,
    num_chunks: int,
) -> Dict[str, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_path), use_fast=False)
    input_ids = make_prompt_ids(tokenizer, prompt, seq_len)

    cfg = Qwen35Config.from_json(str(hf_model_path / "config.json"))
    cfg.context_length = ctx_len
    cfg.state_length = ctx_len

    model = Qwen35ForCausalLM(cfg).half().eval()
    ok = model.load_pretrained_weights(str(hf_model_path))
    if not ok:
        raise RuntimeError("Failed to load HF weights into repo model")

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids.to(torch.int32)).to(torch.float16)
    refs: Dict[str, np.ndarray] = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "embed": hidden.numpy(),
    }

    position_ids = torch.arange(seq_len, dtype=torch.int32)
    causal_mask = make_causal_mask(seq_len, ctx_len)
    current_pos = torch.tensor([0], dtype=torch.int32)

    ranges = chunk_ranges(cfg.num_hidden_layers, num_chunks)
    with torch.no_grad():
        for idx, (start, end) in enumerate(ranges, start=1):
            local_layers = end - start
            k_cache = torch.zeros(
                (local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE,
            )
            v_cache = torch.zeros(
                (local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE,
            )
            conv_dim = (
                cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
            )
            conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
            ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)
            conv = torch.zeros((local_layers, ane_d1, ane_d2), dtype=MODEL_DTYPE)
            rec = torch.zeros(
                (
                    local_layers,
                    cfg.text_config.linear_num_value_heads,
                    cfg.text_config.linear_key_head_dim,
                    cfg.text_config.linear_value_head_dim,
                ),
                dtype=MODEL_DTYPE,
            )

            hidden = model.model.process_layers_prefill_export_local_state(
                hidden_states=hidden,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                kv_cache_0=None,
                k_cache=k_cache,
                v_cache=v_cache,
                linear_conv_state=conv,
                linear_recurrent_state=rec,
                start_layer=start,
                end_layer=end,
                apply_final_norm=False,
                expected_batch_size=1,
                expected_seq_len=seq_len,
            )
            refs[f"torch_chunk{idx}"] = hidden.numpy()

    del model
    gc.collect()
    return refs


def extract_hidden_output(outputs: Dict[str, np.ndarray]) -> np.ndarray:
    for key in ("output_hidden_states", "hidden_states", "out"):
        if key in outputs:
            return outputs[key]
    return list(outputs.values())[0]


def build_chunk_inputs(spec, hidden_input: np.ndarray, seq_len: int, ctx_len: int) -> Dict[str, np.ndarray]:
    inp: Dict[str, np.ndarray] = {}
    for item in spec.description.input:
        if not item.type.HasField("multiArrayType"):
            continue
        name = item.name
        shape = tuple(item.type.multiArrayType.shape)

        if name == "hidden_states":
            inp[name] = hidden_input.astype(np.float16).reshape(shape)
        elif "position" in name:
            pos = np.arange(seq_len, dtype=np.int32)
            inp[name] = pos.reshape(shape)
        elif "causal_mask" in name or name == "mask":
            mask = np.full(shape, -65504.0, dtype=np.float16)
            for r in range(shape[-2]):
                mask[..., r, : r + 1] = 0
            inp[name] = mask
        elif name == "current_pos":
            inp[name] = np.zeros(shape, dtype=np.int32)
        elif name == "linear_conv_state":
            inp[name] = np.zeros(shape, dtype=np.float16)
        elif name == "linear_recurrent_state":
            inp[name] = np.zeros(shape, dtype=np.float16)
        else:
            inp[name] = np.zeros(shape, dtype=np.float16)
    return inp


def compare_arrays(ref: np.ndarray, got: np.ndarray) -> Dict[str, float]:
    if ref.shape != got.shape:
        min_seq = min(ref.shape[1], got.shape[1])
        ref = ref[:, :min_seq, :]
        got = got[:, :min_seq, :]

    diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "cosine": cosine(ref, got),
        "p99": float(np.percentile(diff.reshape(-1), 99)),
    }


def run_coreml_parity(
    model_dir: Path,
    refs: Dict[str, np.ndarray],
    seq_len: int,
    ctx_len: int,
    num_chunks: int,
) -> Dict[str, List[Dict[str, object]]]:
    print("[phase2] loading embeddings model", flush=True)
    embed_model = ct.models.MLModel(
        str(model_dir / "embeddings.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    print("[phase2] running embeddings predict", flush=True)
    embed_out = embed_model.predict({"input_ids": refs["input_ids"].astype(np.int32)})
    coreml_embed = extract_hidden_output(embed_out)
    print(f"[phase2] embedding out shape={coreml_embed.shape}", flush=True)
    del embed_model, embed_out
    gc.collect()

    isolated_results: List[Dict[str, object]] = []
    cascaded_results: List[Dict[str, object]] = []

    prev_coreml = coreml_embed.copy()

    for idx in range(1, num_chunks + 1):
        print(f"[phase2] loading chunk {idx}", flush=True)
        model = ct.models.MLModel(
            str(model_dir / f"prefill_LUT4_chunk{idx-1}.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        spec = model.get_spec()

        isolated_input = coreml_embed if idx == 1 else refs[f"torch_chunk{idx-1}"]
        iso_inp = build_chunk_inputs(spec, isolated_input, seq_len, ctx_len)
        print(f"[phase2] chunk {idx} isolated predict", flush=True)
        iso_out = model.predict(iso_inp, state=model.make_state())
        iso_hidden = extract_hidden_output(iso_out)
        iso_metrics = compare_arrays(refs[f"torch_chunk{idx}"], iso_hidden)
        isolated_results.append(
            {
                "chunk": idx,
                "input_source": "coreml_embed" if idx == 1 else f"torch_chunk{idx-1}",
                "output_shape": list(iso_hidden.shape),
                **iso_metrics,
            }
        )

        cas_inp = build_chunk_inputs(spec, prev_coreml, seq_len, ctx_len)
        print(f"[phase2] chunk {idx} cascaded predict", flush=True)
        cas_out = model.predict(cas_inp, state=model.make_state())
        cas_hidden = extract_hidden_output(cas_out)
        cas_metrics = compare_arrays(refs[f"torch_chunk{idx}"], cas_hidden)
        cascaded_results.append(
            {
                "chunk": idx,
                "input_source": "coreml_embed" if idx == 1 else f"coreml_chunk{idx-1}",
                "output_shape": list(cas_hidden.shape),
                **cas_metrics,
            }
        )

        prev_coreml = cas_hidden
        print(f"[phase2] chunk {idx} done", flush=True)

        del model
        gc.collect()

    return {
        "embedding": compare_arrays(refs["embed"], coreml_embed),
        "isolated": isolated_results,
        "cascaded": cascaded_results,
    }


def save_refs(refs: Dict[str, np.ndarray], ref_dir: Path) -> None:
    ref_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in refs.items():
        np.save(str(ref_dir / f"{key}.npy"), arr)


def load_refs(ref_dir: Path, num_chunks: int) -> Dict[str, np.ndarray]:
    refs: Dict[str, np.ndarray] = {
        "input_ids": np.load(str(ref_dir / "input_ids.npy")),
        "embed": np.load(str(ref_dir / "embed.npy")),
    }
    for idx in range(1, num_chunks + 1):
        refs[f"torch_chunk{idx}"] = np.load(str(ref_dir / f"torch_chunk{idx}.npy"))
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce prefill parity for milestone1_3")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument(
        "--prompt",
        default="Explain stack and heap memory in one paragraph and give one debugging tip.",
    )
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both")
    parser.add_argument("--ref-dir", default="/tmp/qwen35_prefill_parity_m13")
    parser.add_argument("--out", default="tests/dev/reproduce_prefill_parity_m13_report.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    hf_model = Path(args.hf_model)

    ref_dir = Path(args.ref_dir)

    if args.phase in ("1", "both"):
        refs = generate_torch_refs(
            hf_model_path=hf_model,
            prompt=args.prompt,
            seq_len=args.seq_len,
            ctx_len=args.ctx,
            num_chunks=args.num_chunks,
        )
        save_refs(refs, ref_dir)
        print(f"Saved references to {ref_dir}")
        if args.phase == "1":
            return

    refs = load_refs(ref_dir, args.num_chunks)

    report = {
        "model_dir": str(model_dir),
        "hf_model": str(hf_model),
        "seq_len": args.seq_len,
        "ctx": args.ctx,
        "num_chunks": args.num_chunks,
        "ref_dir": str(ref_dir),
        "results": run_coreml_parity(
            model_dir=model_dir,
            refs=refs,
            seq_len=args.seq_len,
            ctx_len=args.ctx,
            num_chunks=args.num_chunks,
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nEmbedding parity:")
    print(report["results"]["embedding"])

    print("\nIsolated chunk parity:")
    for row in report["results"]["isolated"]:
        print(row)

    print("\nCascaded chunk parity:")
    for row in report["results"]["cascaded"]:
        print(row)

    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
