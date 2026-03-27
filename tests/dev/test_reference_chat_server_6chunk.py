#!/usr/bin/env python3
"""Run the known reference ChatEngine on 6-chunk models for think on/off controls."""

import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE)

from scripts_qwen3_5.chat_server import ChatEngine


def run_case(engine, prompt, think):
    chunks = []
    for evt in engine.chat_stream(
        prompt,
        max_tokens=256,
        enable_thinking=think,
        repetition_guard=True,
        repetition_penalty=1.08,
        presence_penalty=0.05,
        frequency_penalty=0.08,
    ):
        if evt.get("type") == "token":
            chunks.append(evt.get("text", ""))
        if evt.get("type") == "done":
            break
    return "".join(chunks)


def main():
    model_dir = os.path.join(BASE, "qwen3_5_6chunk_models")
    if not os.path.isdir(model_dir):
        print(f"Missing model dir: {model_dir}")
        return 2

    print("Loading reference ChatEngine (6-chunk)...")
    engine = ChatEngine(model_dir=model_dir, hf_path=os.path.join(BASE, "qwen3_5_stable_models"), ctx=1024, num_chunks=6)
    engine.load()

    prompts = [
        "What is the capital of China?",
        "What is the capital of USA?",
    ]

    for think in (False, True):
        print("\n" + "=" * 72)
        print(f"REFERENCE 6-CHUNK think={'ON' if think else 'OFF'}")
        print("=" * 72)
        engine.reset()
        for p in prompts:
            text = run_case(engine, p, think)
            show = text.replace("\n", " ")[:220]
            print(f"Q: {p}")
            print(f"A: {show}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
