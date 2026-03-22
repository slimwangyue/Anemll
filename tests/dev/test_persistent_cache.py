#!/usr/bin/env python3
"""Compare reset-replay vs persistent-cache multi-turn conversation.

Mode A (old/naive): Reset states + replay full history each turn
Mode B (incremental): Persistent cache + only prefill new tokens each turn

Verifies:
  - Mode B decode throughput stays flat across turns
  - Mode A throughput drops as history grows
  - Both produce correct text
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, time, gc, copy
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

MODEL_DIR = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3"
HF_MODEL = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 1024
NUM_CHUNKS = 4
MAX_GEN = 30
LABEL = "LUT4"

PROMPTS = [
    "What is the capital of France?",
    "Tell me a fun fact about it.",
    "What language do they speak there?",
]


def build_stop_ids(tokenizer):
    stop = set()
    if tokenizer.eos_token_id is not None:
        stop.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop.add(tok)
    return stop


def build_continuation_tokens(tokenizer, user_msg, enable_thinking=True):
    """Build incremental token sequence for a follow-up turn."""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_toks = tokenizer.encode("\n", add_special_tokens=False)
    think = tokenizer.convert_tokens_to_ids("<think>")
    user_toks = tokenizer.encode("user", add_special_tokens=False)
    asst_toks = tokenizer.encode("assistant", add_special_tokens=False)

    tokens = nl_toks[:]  # after previous <|im_end|>
    tokens += [im_start] + user_toks + nl_toks
    tokens += tokenizer.encode(user_msg, add_special_tokens=False)
    tokens += [im_end] + nl_toks
    tokens += [im_start] + asst_toks + nl_toks
    if enable_thinking:
        tokens += [think] + nl_toks
    return tokens


class Engine:
    def __init__(self, model_dir, cu):
        self.cu = cu
        self.embed = ct.models.MLModel(
            os.path.join(model_dir, "embeddings.mlpackage"), compute_units=cu)
        self.lmhead = ct.models.MLModel(
            os.path.join(model_dir, "lm_head.mlpackage"), compute_units=cu)
        self.ffns = []
        for ci in range(NUM_CHUNKS):
            m = ct.models.MLModel(
                os.path.join(model_dir, f"ffn_{LABEL}_chunk{ci}.mlpackage"),
                compute_units=cu)
            self.ffns.append(m)

        spec = self.ffns[0].get_spec()
        self.inp_map = {}
        for inp in spec.description.input:
            try:
                self.inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.has_linear = 'linear_conv_state' in self.inp_map
        self.reset()

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
                              for _ in range(NUM_CHUNKS)]
            self.lin_recs = [np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
                             for _ in range(NUM_CHUNKS)]
        else:
            self.lin_convs = [None] * NUM_CHUNKS
            self.lin_recs = [None] * NUM_CHUNKS
        self.pos = 0

    def step(self, tok_id, pos):
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if self.lin_convs[ci] is not None:
                inp["linear_conv_state"] = self.lin_convs[ci]
                inp["linear_recurrent_state"] = self.lin_recs[ci]
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    def prefill_and_decode(self, token_ids, stop_ids, max_gen):
        """Prefill from self.pos, then decode max_gen tokens.
        Returns (generated_ids, prefill_time_s, decode_time_s)."""
        # Prefill
        t_pf = time.perf_counter()
        for tid in token_ids:
            if self.pos >= CTX - 1:
                break
            last = self.step(tid, self.pos)
            self.pos += 1
        t_pf = time.perf_counter() - t_pf

        # Decode
        gen = [last]
        t_dc = time.perf_counter()
        for _ in range(max_gen - 1):
            if self.pos >= CTX - 1:
                break
            nxt = self.step(gen[-1], self.pos)
            self.pos += 1
            gen.append(nxt)
            if nxt in stop_ids:
                break
        t_dc = time.perf_counter() - t_dc

        return gen, t_pf, t_dc


def main():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=False)
    stop_ids = build_stop_ids(tokenizer)
    cu = ct.ComputeUnit.CPU_AND_NE

    print("=" * 75)
    print("  Persistent Cache vs Reset-Replay  —  Multi-Turn Comparison")
    print("=" * 75)

    # ── Load engine ──
    print("\nLoading models (CPU_AND_NE)...")
    t0 = time.time()
    eng = Engine(MODEL_DIR, cu)
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Warmup
    eng.step(1, 0)
    eng.reset()

    # ══════════════════════════════════════════════════════════════════
    # Mode A: Reset + Full Replay each turn
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print("  MODE A: Reset + Full Replay (old behavior)")
    print("─" * 75)

    conversation_a = []
    results_a = []

    for ti, user_msg in enumerate(PROMPTS):
        conversation_a.append({"role": "user", "content": user_msg})

        # Full template from scratch
        input_ids = tokenizer.apply_chat_template(
            conversation_a, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=True)
        if hasattr(input_ids, 'input_ids'):
            input_ids = input_ids.input_ids
        ids = input_ids[0].tolist()

        # Reset everything
        eng.reset()

        gen, t_pf, t_dc = eng.prefill_and_decode(ids, stop_ids, MAX_GEN)
        text = tokenizer.decode(gen, skip_special_tokens=True)

        n_decode = len(gen)
        decode_tps = n_decode / t_dc if t_dc > 0 else 0

        results_a.append({
            'turn': ti + 1,
            'prefill_tokens': len(ids),
            'decode_tokens': n_decode,
            'prefill_s': t_pf,
            'decode_s': t_dc,
            'decode_tps': decode_tps,
            'text': text,
            'gen_ids': gen,
        })

        # Add assistant response for next turn's template
        conversation_a.append({"role": "assistant", "content": "<think>\n" + text})

        print(f"\n  Turn {ti+1}: \"{user_msg}\"")
        print(f"    Prefill: {len(ids)} tok in {t_pf*1000:.0f}ms")
        print(f"    Decode:  {n_decode} tok in {t_dc*1000:.0f}ms = {decode_tps:.1f} tok/s")
        print(f"    Total time: {(t_pf+t_dc)*1000:.0f}ms")
        print(f"    Text: {text[:120]}")

    # ══════════════════════════════════════════════════════════════════
    # Mode B: Persistent Cache + Incremental Prefill
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print("  MODE B: Persistent Cache + Incremental Prefill (new behavior)")
    print("─" * 75)

    eng.reset()
    results_b = []

    for ti, user_msg in enumerate(PROMPTS):
        if ti == 0:
            # First turn: full template
            msgs = [{"role": "user", "content": user_msg}]
            input_ids = tokenizer.apply_chat_template(
                msgs, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=True)
            if hasattr(input_ids, 'input_ids'):
                input_ids = input_ids.input_ids
            new_tokens = input_ids[0].tolist()
        else:
            # Subsequent turns: only incremental tokens
            new_tokens = build_continuation_tokens(tokenizer, user_msg)

        gen, t_pf, t_dc = eng.prefill_and_decode(new_tokens, stop_ids, MAX_GEN)
        text = tokenizer.decode(gen, skip_special_tokens=True)

        n_decode = len(gen)
        decode_tps = n_decode / t_dc if t_dc > 0 else 0

        results_b.append({
            'turn': ti + 1,
            'prefill_tokens': len(new_tokens),
            'decode_tokens': n_decode,
            'prefill_s': t_pf,
            'decode_s': t_dc,
            'decode_tps': decode_tps,
            'text': text,
            'gen_ids': gen,
            'pos_after': eng.pos,
        })

        print(f"\n  Turn {ti+1}: \"{user_msg}\"")
        print(f"    Prefill: {len(new_tokens)} tok in {t_pf*1000:.0f}ms  (pos {eng.pos - n_decode - len(new_tokens)} → {eng.pos})")
        print(f"    Decode:  {n_decode} tok in {t_dc*1000:.0f}ms = {decode_tps:.1f} tok/s")
        print(f"    Total time: {(t_pf+t_dc)*1000:.0f}ms")
        print(f"    Text: {text[:120]}")

    # ══════════════════════════════════════════════════════════════════
    # Comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("  COMPARISON")
    print("=" * 75)

    print(f"\n  {'Turn':<6} {'Mode':<12} {'Prefill':>10} {'Decode':>10} {'Decode':>12} {'Total':>10}")
    print(f"  {'':6} {'':12} {'tokens':>10} {'tokens':>10} {'tok/s':>12} {'ms':>10}")
    print(f"  {'─'*62}")

    for ti in range(len(PROMPTS)):
        a = results_a[ti]
        b = results_b[ti]
        print(f"  {ti+1:<6} {'A:reset':<12} {a['prefill_tokens']:>10} {a['decode_tokens']:>10} "
              f"{a['decode_tps']:>11.1f} {(a['prefill_s']+a['decode_s'])*1000:>10.0f}")
        print(f"  {'':6} {'B:persist':<12} {b['prefill_tokens']:>10} {b['decode_tokens']:>10} "
              f"{b['decode_tps']:>11.1f} {(b['prefill_s']+b['decode_s'])*1000:>10.0f}")

    # Speedup analysis
    print(f"\n  Speedup Analysis:")
    for ti in range(len(PROMPTS)):
        a = results_a[ti]
        b = results_b[ti]
        time_a = a['prefill_s'] + a['decode_s']
        time_b = b['prefill_s'] + b['decode_s']
        prefill_saved = a['prefill_tokens'] - b['prefill_tokens']
        speedup = time_a / time_b if time_b > 0 else 0
        print(f"    Turn {ti+1}: {speedup:.1f}x faster total  "
              f"({prefill_saved} prefill tokens saved, "
              f"decode: {a['decode_tps']:.1f} → {b['decode_tps']:.1f} tok/s)")

    # Throughput stability check
    print(f"\n  Decode Throughput Stability:")
    a_tps = [r['decode_tps'] for r in results_a]
    b_tps = [r['decode_tps'] for r in results_b]
    a_drop = (a_tps[0] - a_tps[-1]) / a_tps[0] * 100 if a_tps[0] > 0 else 0
    b_drop = (b_tps[0] - b_tps[-1]) / b_tps[0] * 100 if b_tps[0] > 0 else 0
    print(f"    Mode A (reset): {a_tps[0]:.1f} → {a_tps[-1]:.1f} tok/s  "
          f"({'dropped' if a_drop > 5 else 'stable'} {abs(a_drop):.0f}%)")
    print(f"    Mode B (persist): {b_tps[0]:.1f} → {b_tps[-1]:.1f} tok/s  "
          f"({'dropped' if b_drop > 5 else 'stable'} {abs(b_drop):.0f}%)")

    # Quality check
    print(f"\n  Output Quality:")
    for ti in range(len(PROMPTS)):
        a_text = results_a[ti]['text'][:80]
        b_text = results_b[ti]['text'][:80]
        match = results_a[ti]['gen_ids'][:5] == results_b[ti]['gen_ids'][:5]
        print(f"    Turn {ti+1} first-5-tok match: {'YES' if match else 'NO'}")
        print(f"      A: {a_text}")
        print(f"      B: {b_text}")

    # Verdict
    print(f"\n  {'─'*62}")
    b_stable = abs(b_drop) < 10
    b_faster = all(
        (results_b[ti]['prefill_s'] + results_b[ti]['decode_s']) <
        (results_a[ti]['prefill_s'] + results_a[ti]['decode_s'])
        for ti in range(1, len(PROMPTS)))
    print(f"  Persistent cache stable throughput: {'PASS' if b_stable else 'FAIL'}")
    print(f"  Persistent cache faster on turn 2+: {'PASS' if b_faster else 'FAIL'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
