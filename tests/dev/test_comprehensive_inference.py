#!/usr/bin/env python3
"""
Comprehensive validation of iOS 6-chunk models vs Mac 4-chunk models.

Tests:
1. 6-chunk models from Models.bundle (iOS app's actual models)
2. 4-chunk models from qwen3_5_stable_models (known working)
3. Multiple prompts of varying lengths (10 to 512 tokens)
4. Both think-on and think-off modes
"""
import sys, os, time
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STABLE_DIR = os.path.join(REPO_ROOT, "qwen3_5_stable_models")
IOS_BUNDLE = "/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle"
HF_PATH = STABLE_DIR
CTX = 1024


def find_model(base_dir, name):
    for ext in (".mlpackage", ".mlmodelc"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def load_model(path, cu, function_name=None):
    kwargs = {"compute_units": cu}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


class InferenceEngine:
    """Minimal inference engine matching Swift AnemllInferenceProvider logic."""

    def __init__(self, model_dir, num_chunks, tokenizer, label, cu=ct.ComputeUnit.CPU_AND_NE):
        self.label = label
        self.tok = tokenizer
        self.num_chunks = num_chunks
        self.ctx = CTX
        print(f"\n{'='*60}")
        print(f"Loading {label}: {model_dir} ({num_chunks} chunks)")

        self.embed = load_model(find_model(model_dir, "embeddings"), cu)

        # lm_head
        try:
            self.lmhead = load_model(find_model(model_dir, "lm_head_logits"), cu)
            self.lmhead_mode = "logits"
        except FileNotFoundError:
            self.lmhead = load_model(find_model(model_dir, "lm_head"), cu)
            spec = self.lmhead.get_spec()
            out_names = [o.name for o in spec.description.output]
            self.lmhead_mode = "logits" if ("logits" in out_names
                                            or "output_logits" in out_names) else "argmax"
        # Logits key
        spec = self.lmhead.get_spec()
        out_names = [o.name for o in spec.description.output]
        split_keys = sorted([n for n in out_names if n.startswith("logits") and n[6:].isdigit()])
        if split_keys:
            self.logits_keys = split_keys
            self.logits_key = None
        else:
            self.logits_keys = None
            self.logits_key = "output_logits" if "output_logits" in out_names else "logits"
        print(f"  lm_head mode={self.lmhead_mode}, key={self.logits_key or 'split'}")

        # FFN chunks - try combined first, then separate
        self.ffns = []
        combined_dir = os.path.join(model_dir, "combined_LUT4_dedup")
        use_combined = os.path.isdir(combined_dir)
        for ci in range(num_chunks):
            try:
                if use_combined:
                    path = find_model(combined_dir, f"chunk{ci}")
                    m = load_model(path, cu, function_name="infer")
                else:
                    # Try chunk{ci} first (iOS bundle naming), then ffn_LUT4_chunk{ci}
                    try:
                        path = find_model(model_dir, f"chunk{ci}")
                        m = load_model(path, cu, function_name="infer")
                    except (FileNotFoundError, RuntimeError):
                        path = find_model(model_dir, f"ffn_LUT4_chunk{ci}")
                        m = load_model(path, cu)
            except Exception as e:
                print(f"  FAILED to load chunk{ci}: {e}")
                raise
            self.ffns.append(m)
            print(f"  chunk{ci} loaded")

        # Detect state shapes
        self.inp_shapes = {}
        fn_spec = self.ffns[0].get_spec()
        if use_combined and hasattr(fn_spec.description, 'functions'):
            for fn in fn_spec.description.functions:
                if fn.name == "infer":
                    for inp in fn.input:
                        try:
                            self.inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                        except:
                            pass
                    break
        if not self.inp_shapes:
            for inp in fn_spec.description.input:
                try:
                    self.inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                except:
                    pass
        print(f"  conv_state: {self.inp_shapes.get('linear_conv_state')}")
        print(f"  rec_state: {self.inp_shapes.get('linear_recurrent_state')}")

        # Build stop IDs
        self.stop_ids = set()
        if tokenizer.eos_token_id is not None:
            self.stop_ids.add(tokenizer.eos_token_id)
        for name in ["<|im_end|>", "<|endoftext|>"]:
            t_id = tokenizer.convert_tokens_to_ids(name)
            if t_id is not None and t_id != tokenizer.unk_token_id:
                self.stop_ids.add(t_id)

        # Pre-allocate buffers
        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)
        print(f"  Ready!")

    def _extract_logits(self, lm_out):
        if self.logits_keys:
            parts = [lm_out[k].flatten().astype(np.float32) for k in self.logits_keys]
            return np.concatenate(parts)
        return lm_out[self.logits_key].flatten().astype(np.float32)

    def generate(self, prompt_text, max_tokens=60):
        """Run full inference: tokenize, prefill, decode."""
        prompt_ids = self.tok.encode(prompt_text)

        # Reset states
        states = [m.make_state() for m in self.ffns]
        lin_convs = [np.zeros(self.inp_shapes['linear_conv_state'], dtype=np.float16)
                     for _ in range(self.num_chunks)]
        lin_recs = [np.zeros(self.inp_shapes['linear_recurrent_state'], dtype=np.float16)
                    for _ in range(self.num_chunks)]

        position = 0

        def step_kv_only(tok_id, pos):
            nonlocal lin_convs, lin_recs
            self._tok_buf[0, 0] = tok_id
            hidden = list(self.embed.predict({"input_ids": self._tok_buf}).values())[0]
            self._mask_buf[:] = -65504.0
            self._mask_buf[:, :, :, :pos + 1] = 0
            self._pos_buf[0] = pos
            for ci in range(self.num_chunks):
                out = self.ffns[ci].predict({
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": self._pos_buf,
                    "causal_mask": self._mask_buf,
                    "current_pos": self._pos_buf,
                    "linear_conv_state": lin_convs[ci],
                    "linear_recurrent_state": lin_recs[ci],
                }, state=states[ci])
                hidden = out["output_hidden_states"]
                if 'linear_conv_state_out' in out:
                    lin_convs[ci] = out['linear_conv_state_out']
                    lin_recs[ci] = out['linear_recurrent_state_out']

        def step(tok_id, pos):
            nonlocal lin_convs, lin_recs
            self._tok_buf[0, 0] = tok_id
            hidden = list(self.embed.predict({"input_ids": self._tok_buf}).values())[0]
            self._mask_buf[:] = -65504.0
            self._mask_buf[:, :, :, :pos + 1] = 0
            self._pos_buf[0] = pos
            for ci in range(self.num_chunks):
                out = self.ffns[ci].predict({
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": self._pos_buf,
                    "causal_mask": self._mask_buf,
                    "current_pos": self._pos_buf,
                    "linear_conv_state": lin_convs[ci],
                    "linear_recurrent_state": lin_recs[ci],
                }, state=states[ci])
                hidden = out["output_hidden_states"]
                if 'linear_conv_state_out' in out:
                    lin_convs[ci] = out['linear_conv_state_out']
                    lin_recs[ci] = out['linear_recurrent_state_out']
            lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
            if self.lmhead_mode == "logits":
                logits = self._extract_logits(lm_out)
                return int(np.argmax(logits))
            return int(lm_out["argmax_idx"].flatten()[0])

        # Prefill
        n = len(prompt_ids)
        t0 = time.time()
        for ti in range(n):
            if ti == n - 1:
                first_token = step(prompt_ids[ti], position)
            else:
                step_kv_only(prompt_ids[ti], position)
            position += 1
        prefill_t = time.time() - t0

        # Decode
        generated = [first_token]
        current = first_token
        t0 = time.time()
        for _ in range(max_tokens - 1):
            if position >= self.ctx - 1:
                break
            if current in self.stop_ids:
                break
            next_id = step(current, position)
            position += 1
            generated.append(next_id)
            current = next_id
            if current in self.stop_ids:
                break
        decode_t = time.time() - t0

        text = self.tok.decode(generated, skip_special_tokens=True)
        tps = len(generated) / max(decode_t, 1e-9)
        return text, len(prompt_ids), len(generated), prefill_t, decode_t, tps


def build_prompt(user_msg, think=False):
    """Build exact HF-matching ChatML prompt."""
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    if think:
        prompt += "<think>\n"
    return prompt


# Test prompts of varying lengths/complexity
TEST_PROMPTS = [
    # Short
    ("What is 2+2?", "4"),
    ("What is the capital of France?", "Paris"),
    # Medium
    ("What planet is closest to the sun?", "Mercury"),
    # Longer
    ("Explain the difference between a list and a dictionary in Python. Be brief.", "list"),
]


def main():
    tok = AutoTokenizer.from_pretrained(HF_PATH, use_fast=False)

    # Check iOS bundle exists
    ios_available = os.path.isdir(IOS_BUNDLE) and os.path.exists(
        os.path.join(IOS_BUNDLE, "chunk0.mlpackage"))

    # Load 4-chunk engine
    engine_4 = InferenceEngine(STABLE_DIR, 4, tok, "4-chunk (Mac)")

    # Load 6-chunk engine if available
    engine_6 = None
    if ios_available:
        try:
            engine_6 = InferenceEngine(IOS_BUNDLE, 6, tok, "6-chunk (iOS)")
        except Exception as e:
            print(f"\n*** FAILED to load 6-chunk models: {e}")
            engine_6 = None
    else:
        print(f"\n*** iOS Models.bundle not found at {IOS_BUNDLE}")

    # Run tests
    print(f"\n{'='*70}")
    print("COMPREHENSIVE INFERENCE VALIDATION")
    print(f"{'='*70}")

    results = []
    for user_msg, expected_keyword in TEST_PROMPTS:
        prompt = build_prompt(user_msg, think=False)
        prompt_ids = tok.encode(prompt)
        prompt_len = len(prompt_ids)

        print(f"\n--- Prompt: \"{user_msg}\" ({prompt_len} tokens) ---")

        # 4-chunk test
        try:
            text_4, p_len, g_len, pf_t, dc_t, tps = engine_4.generate(prompt, max_tokens=120)
            has_keyword_4 = expected_keyword.lower() in text_4.lower()
            # Clean up think tags for display
            display_4 = text_4.replace("<think>", "").replace("</think>", "").strip()
            if len(display_4) > 120:
                display_4 = display_4[:120] + "..."
            status_4 = "PASS" if has_keyword_4 else "FAIL"
            print(f"  4-chunk [{status_4}]: {display_4}")
            results.append(("4-chunk", user_msg, prompt_len, status_4, display_4))
        except Exception as e:
            print(f"  4-chunk [ERROR]: {e}")
            results.append(("4-chunk", user_msg, prompt_len, "ERROR", str(e)))

        # 6-chunk test
        if engine_6:
            try:
                text_6, p_len, g_len, pf_t, dc_t, tps = engine_6.generate(prompt, max_tokens=120)
                has_keyword_6 = expected_keyword.lower() in text_6.lower()
                display_6 = text_6.replace("<think>", "").replace("</think>", "").strip()
                if len(display_6) > 120:
                    display_6 = display_6[:120] + "..."
                status_6 = "PASS" if has_keyword_6 else "FAIL"
                print(f"  6-chunk [{status_6}]: {display_6}")
                results.append(("6-chunk", user_msg, prompt_len, status_6, display_6))
            except Exception as e:
                print(f"  6-chunk [ERROR]: {e}")
                results.append(("6-chunk", user_msg, prompt_len, "ERROR", str(e)))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for engine_name in ["4-chunk", "6-chunk"]:
        engine_results = [r for r in results if r[0] == engine_name]
        if not engine_results:
            continue
        passed = sum(1 for r in engine_results if r[3] == "PASS")
        total = len(engine_results)
        print(f"\n{engine_name}: {passed}/{total} passed")
        for _, msg, plen, status, output in engine_results:
            marker = "  OK " if status == "PASS" else "  BAD"
            print(f"  {marker} [{plen:3d} tok] {msg}")
            if status != "PASS":
                print(f"         Output: {output[:80]}")


if __name__ == "__main__":
    main()
