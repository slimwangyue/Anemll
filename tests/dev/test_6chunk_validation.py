#!/usr/bin/env python3
"""
Comprehensive validation of inference logic with both 4-chunk and 6-chunk models.
Tests prompts of varying lengths (short to long) and verifies correct answers.

This mirrors the EXACT same inference logic as the Swift iOS app:
  - Sequential prefill (token-by-token stepKVOnly + step for last)
  - Same causal mask filling
  - Same buffer management
  - No output backings (matching the fix we applied)
"""
import sys, os, time
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

CTX = 1024

# ── Test prompts with expected answer keywords ──
TEST_CASES = [
    # Short prompts (~10-20 tokens)
    ("What is 2+2?", ["4", "four"]),
    ("What is the capital of France?", ["Paris"]),
    ("What color is the sky?", ["blue"]),
    # Medium prompts (~20-40 tokens)
    ("Who wrote Romeo and Juliet? Please answer briefly.", ["Shakespeare"]),
    ("What is the largest planet in our solar system? Answer in one word.", ["Jupiter"]),
    ("Name the first president of the United States of America.", ["Washington"]),
    # Longer prompts (~40-80 tokens)
    (
        "I am a student studying geography. I have a test tomorrow about European capitals. "
        "Can you tell me what the capital of Germany is? Please be concise.",
        ["Berlin"]
    ),
    (
        "My friend asked me a trivia question and I want to make sure I get it right. "
        "The question is: what is the chemical symbol for water? Just give me the answer.",
        ["H2O"]
    ),
    (
        "I am writing a report for school about famous scientists. "
        "Who developed the theory of general relativity? "
        "Please provide just the name.",
        ["Einstein"]
    ),
]


def load_model(path, cu, function_name=None):
    kwargs = {"compute_units": cu}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def find_model(base_dir, name):
    for ext in (".mlpackage", ".mlmodelc"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


class InferenceEngine:
    """Mirrors the Swift AnemllInferenceProvider logic exactly."""

    def __init__(self, model_dir, hf_path, num_chunks, chunk_dir=None):
        self.ctx = CTX
        self.num_chunks = num_chunks
        cu = ct.ComputeUnit.CPU_AND_NE

        self.tok = AutoTokenizer.from_pretrained(hf_path, use_fast=False)

        # Build stop IDs
        self.stop_ids = set()
        if self.tok.eos_token_id is not None:
            self.stop_ids.add(self.tok.eos_token_id)
        for name in ["<|im_end|>", "<|endoftext|>"]:
            t_id = self.tok.convert_tokens_to_ids(name)
            if t_id is not None and t_id != self.tok.unk_token_id:
                self.stop_ids.add(t_id)

        # Load models
        self.embed = load_model(find_model(model_dir, "embeddings"), cu)

        # LM head
        try:
            self.lmhead = load_model(find_model(model_dir, "lm_head_logits"), cu)
        except FileNotFoundError:
            self.lmhead = load_model(find_model(model_dir, "lm_head"), cu)
        spec = self.lmhead.get_spec()
        out_names = [o.name for o in spec.description.output]
        split_keys = sorted([n for n in out_names if n.startswith("logits") and n[6:].isdigit()])
        if split_keys:
            self.logits_keys = split_keys
            self.logits_key = None
            self.lmhead_mode = "logits"
        else:
            self.logits_keys = None
            self.logits_key = "output_logits" if "output_logits" in out_names else "logits"
            self.lmhead_mode = "logits" if self.logits_key in out_names else "argmax"

        # FFN chunks
        if chunk_dir is None:
            chunk_dir = model_dir
        self.ffns = []
        for ci in range(num_chunks):
            path = find_model(chunk_dir, f"chunk{ci}")
            m = load_model(path, cu, function_name="infer")
            self.ffns.append(m)

        # Detect state shapes — PER-CHUNK (layer counts may differ!)
        self.conv_shapes = []
        self.rec_shapes = []
        for ci in range(num_chunks):
            shapes = {}
            fn_spec = self.ffns[ci].get_spec()
            if hasattr(fn_spec.description, 'functions'):
                for fn in fn_spec.description.functions:
                    if fn.name == "infer":
                        for inp in fn.input:
                            try:
                                shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                            except:
                                pass
                        break
            if not shapes:
                for inp in fn_spec.description.input:
                    try:
                        shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except:
                        pass
            self.conv_shapes.append(shapes.get('linear_conv_state', (8, 1024, 32)))
            self.rec_shapes.append(shapes.get('linear_recurrent_state', (8, 32, 128, 128)))

        # Pre-allocate buffers
        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)

    def _extract_logits(self, lm_out):
        if self.logits_keys:
            parts = [lm_out[k].flatten().astype(np.float32) for k in self.logits_keys]
            return np.concatenate(parts)
        return lm_out[self.logits_key].flatten().astype(np.float32)

    def _reset_states(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [
            np.zeros(self.conv_shapes[ci], dtype=np.float16)
            for ci in range(self.num_chunks)]
        self.lin_recs = [
            np.zeros(self.rec_shapes[ci], dtype=np.float16)
            for ci in range(self.num_chunks)]
        self.pos = 0

    def _step_kv_only(self, tok_id, pos):
        self._tok_buf[0, 0] = tok_id
        hidden = list(self.embed.predict({"input_ids": self._tok_buf}).values())[0]

        self._mask_buf[:, :, :, :] = -65504.0
        self._mask_buf[:, :, :, :pos + 1] = 0
        self._pos_buf[0] = pos

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": self._pos_buf,
                "causal_mask": self._mask_buf,
                "current_pos": self._pos_buf,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

    def _step(self, tok_id, pos):
        self._tok_buf[0, 0] = tok_id
        hidden = list(self.embed.predict({"input_ids": self._tok_buf}).values())[0]

        self._mask_buf[:, :, :, :] = -65504.0
        self._mask_buf[:, :, :, :pos + 1] = 0
        self._pos_buf[0] = pos

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": self._pos_buf,
                "causal_mask": self._mask_buf,
                "current_pos": self._pos_buf,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if self.lmhead_mode == "logits":
            logits = self._extract_logits(lm_out)
            return int(np.argmax(logits))
        return int(lm_out["argmax_idx"].flatten()[0])

    def run(self, user_msg, max_tokens=200, think=False):
        """Run full inference, returns generated text."""
        # Build prompt (matching Swift QwenChatPromptBuilder format)
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        if think:
            prompt += "<think>\n"
        else:
            # Empty think block forces model to skip thinking (matches chat_server.py)
            prompt += "<think>\n\n</think>\n\n"

        prompt_ids = self.tok.encode(prompt)

        self._reset_states()

        # Prefill (sequential, same as Swift for short prompts)
        n = len(prompt_ids)
        for ti in range(n):
            if ti == n - 1:
                first_token = self._step(prompt_ids[ti], self.pos)
            else:
                self._step_kv_only(prompt_ids[ti], self.pos)
            self.pos += 1

        # Decode
        generated = [first_token]
        current = first_token
        for _ in range(max_tokens - 1):
            if self.pos >= self.ctx - 1:
                break
            if current in self.stop_ids:
                break
            next_id = self._step(current, self.pos)
            self.pos += 1
            generated.append(next_id)
            current = next_id
            if current in self.stop_ids:
                break

        text = self.tok.decode(generated, skip_special_tokens=True)
        return text, len(prompt_ids), len(generated)


def run_validation(engine, label, think=False):
    """Run all test cases and report pass/fail."""
    print(f"\n{'='*70}")
    print(f"  VALIDATION: {label} (think={'ON' if think else 'OFF'})")
    print(f"{'='*70}")

    passed = 0
    failed = 0
    results = []

    for i, (question, keywords) in enumerate(TEST_CASES):
        t0 = time.time()
        text, n_prompt, n_gen = engine.run(question, max_tokens=200, think=think)
        elapsed = time.time() - t0
        tps = n_gen / max(elapsed, 1e-9)

        # Check if any expected keyword is in the output
        text_lower = text.lower()
        found = any(kw.lower() in text_lower for kw in keywords)
        status = "PASS" if found else "FAIL"
        if found:
            passed += 1
        else:
            failed += 1

        # Show truncated output
        display = text.replace('\n', ' ')[:120]
        print(f"  [{status}] Q{i+1}: \"{question[:50]}...\"")
        print(f"       prompt={n_prompt}tok gen={n_gen}tok {elapsed:.1f}s ({tps:.0f}t/s)")
        print(f"       A: {display}")
        if not found:
            print(f"       EXPECTED: {keywords}")
        results.append((status, question, text))

    print(f"\n  SUMMARY: {passed}/{passed+failed} passed")
    return passed, failed, results


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # ── 4-chunk models ──
    model_dir_4 = os.path.join(base_dir, "qwen3_5_stable_models")
    chunk_dir_4 = os.path.join(model_dir_4, "combined_LUT4_dedup")

    print("Loading 4-chunk models...")
    engine4 = InferenceEngine(model_dir_4, model_dir_4, num_chunks=4, chunk_dir=chunk_dir_4)
    print("4-chunk models loaded.")

    p4_think, f4_think, _ = run_validation(engine4, "4-CHUNK (think=ON)", think=True)
    p4_nothink, f4_nothink, _ = run_validation(engine4, "4-CHUNK (think=OFF)", think=False)

    # ── 6-chunk models ──
    model_dir_6 = os.path.join(base_dir, "qwen3_5_6chunk_models")
    chunk_dir_6 = os.path.join(model_dir_6, "combined_LUT4_dedup")
    if os.path.isdir(model_dir_6):
        print("\nLoading 6-chunk models...")
        engine6 = InferenceEngine(model_dir_6, model_dir_4, num_chunks=6, chunk_dir=chunk_dir_6)
        print("6-chunk models loaded.")

        p6_think, f6_think, _ = run_validation(engine6, "6-CHUNK (think=ON)", think=True)
        p6_nothink, f6_nothink, _ = run_validation(engine6, "6-CHUNK (think=OFF)", think=False)
    else:
        print(f"\n6-chunk models not found at {model_dir_6}, skipping.")
        p6_think = f6_think = p6_nothink = f6_nothink = 0

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    total = len(TEST_CASES)
    print(f"  4-chunk think=ON:  {p4_think}/{total}")
    print(f"  4-chunk think=OFF: {p4_nothink}/{total}")
    if os.path.isdir(os.path.join(base_dir, "qwen3_5_6chunk_models")):
        print(f"  6-chunk think=ON:  {p6_think}/{total}")
        print(f"  6-chunk think=OFF: {p6_nothink}/{total}")

    all_passed = (f4_think + f4_nothink + f6_think + f6_nothink) == 0
    if all_passed:
        print("\n  ALL TESTS PASSED!")
    else:
        print(f"\n  FAILURES DETECTED - check output above")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
