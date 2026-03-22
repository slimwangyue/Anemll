#!/usr/bin/env python3
"""End-to-end generation test with LUT6+argmax LM head.

Loads all models in a memory-efficient way (one at a time) and runs
3 prompts to verify generation quality.
"""
import sys, os, gc, time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)  # must be first for config.py

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
from config import DEFAULT_OUTPUT, DEFAULT_HF_MODEL, CTX, NUM_CHUNKS

MAX_TOKENS = 30


def find_model(base, name):
    for ext in [".mlmodelc", ".mlpackage"]:
        p = os.path.join(base, name + ext)
        if os.path.exists(p):
            return p
    return None


class LightEngine:
    """Memory-efficient inference engine that loads models incrementally."""

    def __init__(self, export_dir, tokenizer):
        self.export_dir = export_dir
        self.tokenizer = tokenizer
        self.cu = ct.ComputeUnit.CPU_AND_NE

        # Load models one at a time, let GC reclaim between loads
        print("  Loading embed...")
        self.embed = ct.models.MLModel(find_model(export_dir, "embeddings"), compute_units=self.cu)
        gc.collect()

        print("  Loading lm_head...")
        self.lmhead = ct.models.MLModel(find_model(export_dir, "lm_head"), compute_units=self.cu)
        gc.collect()

        self.ffns = []
        for ci in range(NUM_CHUNKS):
            print(f"  Loading ffn chunk {ci}...")
            m = ct.models.MLModel(find_model(export_dir, f"ffn_LUT4_chunk{ci}"), compute_units=self.cu)
            self.ffns.append(m)
            gc.collect()

        # Init states
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [np.zeros((8, 1024, 32), dtype=np.float16) for _ in range(NUM_CHUNKS)]
        self.lin_recs = [np.zeros((8, 32, 128, 128), dtype=np.float16) for _ in range(NUM_CHUNKS)]
        self.pos = 0

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [np.zeros((8, 1024, 32), dtype=np.float16) for _ in range(NUM_CHUNKS)]
        self.lin_recs = [np.zeros((8, 32, 128, 128), dtype=np.float16) for _ in range(NUM_CHUNKS)]
        self.pos = 0

    def step(self, tok_id):
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :self.pos + 1] = 0

        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([self.pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([self.pos], dtype=np.int32),
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            next_id = int(np.argmax(lm_out["logits"].flatten()))
        else:
            next_id = int(lm_out["argmax_idx"].flatten()[0])

        self.pos += 1
        return next_id

    def generate(self, prompt, max_tokens=30):
        self.reset()
        stop_ids = set()
        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(self.tokenizer.eos_token_id)
        for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
            tok = self.tokenizer.convert_tokens_to_ids(name)
            if tok is not None and tok != self.tokenizer.unk_token_id:
                stop_ids.add(tok)

        msgs = [{"role": "user", "content": prompt}]
        tpl = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_dict=False)
        if isinstance(tpl, dict):
            prompt_ids = list(tpl["input_ids"])
        else:
            prompt_ids = list(tpl)

        # Prefill
        for tid in prompt_ids:
            if self.pos >= CTX:
                break
            last_tok = self.step(tid)

        # Decode
        gen_ids = [last_tok]
        t0 = time.time()
        for _ in range(max_tokens - 1):
            if last_tok in stop_ids or self.pos >= CTX:
                break
            last_tok = self.step(last_tok)
            gen_ids.append(last_tok)

        elapsed = time.time() - t0
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        tps = (len(gen_ids) - 1) / elapsed if elapsed > 0 and len(gen_ids) > 1 else 0

        # Strip <think>...</think>
        clean = text
        if "<think>" in clean:
            end_tag = clean.find("</think>")
            if end_tag >= 0:
                clean = clean[end_tag + len("</think>"):].strip()

        return clean, len(gen_ids), tps


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3.5-4B end-to-end generation test")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT,
                        help="Directory with exported .mlpackage files")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer dir (default: same as --model-dir)")
    parser.add_argument("--tokens", type=int, default=MAX_TOKENS,
                        help="Max tokens to generate per prompt")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_dir

    EXPORT_DIR = args.model_dir
    MODEL_PATH = args.tokenizer

    print("=" * 60)
    print("  Qwen3.5-4B End-to-End Generation — LUT6+Argmax")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load engine
    print("\nLoading models (incremental)...")
    t0 = time.time()
    engine = LightEngine(EXPORT_DIR, tokenizer)
    print(f"  All models loaded in {time.time()-t0:.1f}s")

    # Check LM head type
    spec = engine.lmhead.get_spec()
    outputs = [o.name for o in spec.description.output]
    print(f"\n  LM head outputs: {outputs}")

    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "What is 2+2?",
    ]

    print("\n" + "-" * 60)
    for prompt in prompts:
        text, n_tokens, tps = engine.generate(prompt, args.tokens)
        print(f"\n  Q: {prompt}")
        print(f"  A: {text[:300]}")
        print(f"  Tokens: {n_tokens}, Decode: {tps:.1f} tok/s")

    print("\n" + "=" * 60)
    print("  END-TO-END VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
