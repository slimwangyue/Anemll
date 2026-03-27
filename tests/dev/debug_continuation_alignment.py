#!/usr/bin/env python3
"""Check continuation-token alignment between chat_server and validated harness logic."""

import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts_qwen3_5.chat_server import ChatEngine
from tests.dev._test_multiround_conversation import _build_continuation, _get_template_tokens


HF_PATH = "/Users/yw68/Anemll/qwen3_5_stable_models"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, use_fast=False)
    tpl = _get_template_tokens(tokenizer)

    eng = ChatEngine(model_dir="/tmp", hf_path=HF_PATH, ctx=1024, num_chunks=4)
    eng.tokenizer = tokenizer
    eng._build_template_tokens()

    user_msg = "How does it compare to a queue?"

    for has_stop in (False, True):
        for thinking in (True, False):
            ref = _build_continuation(tokenizer, tpl, user_msg, has_stop)
            got = eng._build_continuation_tokens(user_msg, enable_thinking=thinking, has_stop_token=has_stop)

            if thinking:
                status = "MATCH" if ref == got else "DIFF"
                print(f"thinking={thinking} has_stop={has_stop} => {status} (ref={len(ref)}, got={len(got)})")
            else:
                print(
                    f"thinking={thinking} has_stop={has_stop} => got_len={len(got)} "
                    f"(harness uses thinking=True path only)"
                )


if __name__ == "__main__":
    main()
