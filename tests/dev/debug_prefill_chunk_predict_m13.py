#!/usr/bin/env python3
"""Single prefill predict probe for milestone1_3 chunk0 on ANE."""

import os
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"
HF_MODEL = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
SEQ = 256
CTX = 256


def build_ids(tokenizer):
    prompt = "Explain stack and heap memory in one paragraph and give one debugging tip."
    text = prompt
    while True:
        ids = tokenizer(text, return_tensors="np", add_special_tokens=True)["input_ids"]
        if ids.shape[1] >= SEQ:
            return ids[:, :SEQ].astype(np.int32)
        text = text + " " + prompt


def main() -> None:
    cu = ct.ComputeUnit.CPU_AND_NE

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=False)
    input_ids = build_ids(tokenizer)

    embed = ct.models.MLModel(os.path.join(MODEL_DIR, "embeddings.mlpackage"), compute_units=cu)
    hidden = list(embed.predict({"input_ids": input_ids}).values())[0]
    print("embed out", hidden.shape)

    chunk = ct.models.MLModel(
        os.path.join(MODEL_DIR, "prefill_LUT4_chunk0.mlpackage"), compute_units=cu
    )
    spec = chunk.get_spec()
    inp = {}
    for item in spec.description.input:
        if not item.type.HasField("multiArrayType"):
            continue
        name = item.name
        shape = tuple(item.type.multiArrayType.shape)
        if name == "hidden_states":
            inp[name] = hidden.astype(np.float16).reshape(shape)
        elif "position" in name:
            inp[name] = np.arange(SEQ, dtype=np.int32).reshape(shape)
        elif "causal_mask" in name or name == "mask":
            mask = np.full(shape, -65504.0, dtype=np.float16)
            for r in range(shape[-2]):
                mask[..., r, : r + 1] = 0
            inp[name] = mask
        elif name == "current_pos":
            inp[name] = np.zeros(shape, dtype=np.int32)
        elif name in ("linear_conv_state", "linear_recurrent_state"):
            inp[name] = np.zeros(shape, dtype=np.float16)
        else:
            inp[name] = np.zeros(shape, dtype=np.float16)
        print("input", name, inp[name].shape, inp[name].dtype)

    out = chunk.predict(inp, state=chunk.make_state())
    first_key = next(iter(out))
    print("output key", first_key, "shape", out[first_key].shape)


if __name__ == "__main__":
    main()
