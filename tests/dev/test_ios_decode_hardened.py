#!/usr/bin/env python3
"""
Mac-side validation that mirrors the latest Swift decode hardening logic:
- think OFF: suppress model-generated <think> ... </think> blocks
- hidden-think cap to avoid permanent blank output
- logit penalties and repetition detection
- non-empty fallback emission
"""

import argparse
import os
import time
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

CTX = 1024

TESTS = [
    ("What is the capital of China?", ["beijing"]),
    ("What is the capital of USA?", ["washington"]),
    ("What is 2+2?", ["4", "four"]),
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


class Engine:
    def __init__(self, model_dir, hf_path, num_chunks, chunk_dir=None):
        self.ctx = CTX
        self.num_chunks = num_chunks
        cu = ct.ComputeUnit.CPU_AND_NE

        self.tok = AutoTokenizer.from_pretrained(hf_path, use_fast=False)

        self.stop_ids = set()
        if self.tok.eos_token_id is not None:
            self.stop_ids.add(self.tok.eos_token_id)
        for name in ["<|im_end|>", "<|endoftext|>"]:
            t_id = self.tok.convert_tokens_to_ids(name)
            if t_id is not None and t_id != self.tok.unk_token_id:
                self.stop_ids.add(t_id)

        self.think_id = self.tok.convert_tokens_to_ids("<think>")
        self.endthink_id = self.tok.convert_tokens_to_ids("</think>")

        self.embed = load_model(find_model(model_dir, "embeddings"), cu)

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

        if chunk_dir is None:
            chunk_dir = model_dir
        self.ffns = []
        for ci in range(num_chunks):
            path = find_model(chunk_dir, f"chunk{ci}")
            self.ffns.append(load_model(path, cu, function_name="infer"))

        self.conv_shapes = []
        self.rec_shapes = []
        for ci in range(num_chunks):
            shapes = {}
            fn_spec = self.ffns[ci].get_spec()
            if hasattr(fn_spec.description, "functions"):
                for fn in fn_spec.description.functions:
                    if fn.name == "infer":
                        for inp in fn.input:
                            try:
                                shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                            except Exception:
                                pass
                        break
            if not shapes:
                for inp in fn_spec.description.input:
                    try:
                        shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except Exception:
                        pass
            self.conv_shapes.append(shapes.get("linear_conv_state", (8, 1024, 32)))
            self.rec_shapes.append(shapes.get("linear_recurrent_state", (8, 32, 128, 128)))

        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)

    def _extract_logits(self, lm_out):
        if self.logits_keys:
            parts = [lm_out[k].flatten().astype(np.float32) for k in self.logits_keys]
            return np.concatenate(parts)
        return lm_out[self.logits_key].flatten().astype(np.float32)

    def _reset(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [np.zeros(self.conv_shapes[ci], dtype=np.float16) for ci in range(self.num_chunks)]
        self.lin_recs = [np.zeros(self.rec_shapes[ci], dtype=np.float16) for ci in range(self.num_chunks)]
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
            if "linear_conv_state_out" in out:
                self.lin_convs[ci] = out["linear_conv_state_out"]
                self.lin_recs[ci] = out["linear_recurrent_state_out"]

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
            if "linear_conv_state_out" in out:
                self.lin_convs[ci] = out["linear_conv_state_out"]
                self.lin_recs[ci] = out["linear_recurrent_state_out"]

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if self.lmhead_mode == "logits":
            logits = self._extract_logits(lm_out)
            return int(np.argmax(logits)), logits
        return int(lm_out["argmax_idx"].flatten()[0]), None

    @staticmethod
    def _apply_penalties(logits, generated_ids, repetition_penalty=1.08, presence_penalty=0.05, frequency_penalty=0.08):
        if not generated_ids:
            return int(np.argmax(logits))

        adjusted = logits.copy()
        token_counts = {}
        for tid in generated_ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        for tid, count in token_counts.items():
            if tid < 0 or tid >= len(adjusted):
                continue
            if repetition_penalty != 1.0:
                if adjusted[tid] > 0:
                    adjusted[tid] /= repetition_penalty
                else:
                    adjusted[tid] *= repetition_penalty
            adjusted[tid] -= presence_penalty
            adjusted[tid] -= frequency_penalty * count

        return int(np.argmax(adjusted))

    @staticmethod
    def _has_repetition(generated_ids, window_size=80, ngram_size=5, threshold=3):
        if len(generated_ids) < ngram_size:
            return False
        window = generated_ids[-window_size:]
        counts = {}
        for i in range(len(window) - ngram_size + 1):
            gram = tuple(window[i:i + ngram_size])
            counts[gram] = counts.get(gram, 0) + 1
            if counts[gram] >= threshold:
                return True
        return False

    @staticmethod
    def _clean(decoded):
        return (
            decoded.replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|im_start|>", "")
            .replace("<think>", "")
            .replace("</think>", "")
            .replace("assistant\n", "")
            .replace("assistant:", "")
            .replace("user\n", "")
            .replace("user:", "")
            .strip()
        )

    def _final_visible_text(self, all_generated_ids, think_mode_enabled):
        decoded = self.tok.decode(all_generated_ids, skip_special_tokens=False)
        if think_mode_enabled:
            idx = decoded.find("</think>")
            if idx >= 0:
                decoded = decoded[idx + len("</think>"):]
        return self._clean(decoded)

    def run(self, user_msg, think=False):
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        if think:
            prompt += "<think>\n"

        prompt_ids = self.tok.encode(prompt)
        self._reset()

        n = len(prompt_ids)
        for ti in range(n):
            if ti == n - 1:
                first_token, _ = self._step(prompt_ids[ti], self.pos)
            else:
                self._step_kv_only(prompt_ids[ti], self.pos)
            self.pos += 1

        current = first_token
        generated_visible = []
        all_generated = []
        suppressing_think = (not think and self.think_id is not None and current == self.think_id)
        hidden_think_count = 0

        if current not in self.stop_ids:
            all_generated.append(current)
            if not suppressing_think:
                generated_visible.append(current)

        max_tokens = 512 if think else 384
        for _ in range(max_tokens - 1):
            if self.pos >= self.ctx - 1:
                break
            if current in self.stop_ids:
                break

            raw_next, logits = self._step(current, self.pos)
            nxt = raw_next
            if logits is not None:
                nxt = self._apply_penalties(logits, all_generated)

            self.pos += 1
            current = nxt

            if current in self.stop_ids:
                break

            all_generated.append(current)

            if self._has_repetition(all_generated):
                break

            if not think:
                if suppressing_think:
                    hidden_think_count += 1
                    if self.endthink_id is not None and current == self.endthink_id:
                        suppressing_think = False
                        hidden_think_count = 0
                    if hidden_think_count >= 96:
                        suppressing_think = False
                        hidden_think_count = 0
                    continue
                if self.think_id is not None and current == self.think_id:
                    suppressing_think = True
                    hidden_think_count = 0
                    continue

            generated_visible.append(current)

        visible = self._final_visible_text(generated_visible, think_mode_enabled=think)
        if not visible.strip():
            visible = self._final_visible_text(all_generated, think_mode_enabled=think)

        return {
            "text": visible,
            "first_token": first_token,
            "prompt_tokens": len(prompt_ids),
            "visible_tokens": len(generated_visible),
            "all_tokens": len(all_generated),
        }


def run_suite(engine, label, think):
    print("\n" + "=" * 72)
    print(f"{label} think={'ON' if think else 'OFF'}")
    print("=" * 72)
    passed = 0

    for q, kws in TESTS:
        t0 = time.time()
        out = engine.run(q, think=think)
        elapsed = time.time() - t0
        text = out["text"]
        ok_nonempty = bool(text.strip())
        ok_kw = any(k in text.lower() for k in kws)
        ok = ok_nonempty and ok_kw
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {q}")
        print(
            f"  first={out['first_token']} prompt={out['prompt_tokens']} "
            f"visible={out['visible_tokens']} all={out['all_tokens']} {elapsed:.1f}s"
        )
        show = text.replace("\n", " ")[:180]
        print(f"  text: {show}")
        if not ok:
            print(f"  expected keywords: {kws}")
    return passed, len(TESTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=6, choices=[4, 6])
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if args.chunks == 6:
        model_dir = os.path.join(base, "qwen3_5_6chunk_models")
        chunk_dir = os.path.join(model_dir, "combined_LUT4_dedup")
    else:
        model_dir = os.path.join(base, "qwen3_5_stable_models")
        chunk_dir = os.path.join(model_dir, "combined_LUT4_dedup")

    tok_dir = os.path.join(base, "qwen3_5_stable_models")

    if not os.path.isdir(model_dir):
        print(f"Missing model dir: {model_dir}")
        return 2

    print(f"Loading {args.chunks}-chunk engine...")
    engine = Engine(model_dir, tok_dir, num_chunks=args.chunks, chunk_dir=chunk_dir)
    print("Engine loaded.")

    p_off, n = run_suite(engine, "VALIDATION", think=False)
    p_on, _ = run_suite(engine, "VALIDATION", think=True)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"think OFF: {p_off}/{n}")
    print(f"think ON:  {p_on}/{n}")

    # Strict success: at least non-empty + keyword for all OFF prompts.
    return 0 if p_off == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
