#!/usr/bin/env python3
"""Qwen3.5-4B Milestone 1: Multi-round conversation validation.

Validates that:
  1) Dedup combined models produce IDENTICAL tokens to separate models
  2) Fresh vs incremental mode match for both engine types
  3) All 3 conversation turns pass at 100% token accuracy

Usage:
    python tests/dev/qwen35_validate.py \\
        --model-dir /path/to/exported \\
        --tokenizer /path/to/Qwen3.5-4B \\
        --tokens 40

    # Skip separate model validation (only test dedup):
    python tests/dev/qwen35_validate.py --model-dir /path/to/exported \\
        --tokenizer /path/to/Qwen3.5-4B --skip-separate
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import gc, time, argparse
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import torch

# ── Config ──
BATCH_SIZE = 256   # prefill input length
CTX = 1024         # KV cache / context length
NUM_CHUNKS = 4
LUT_BITS = 4

CONVERSATION_TURNS = [
    "What is a stack in computer science?",
    "How does it compare to a queue?",
    "Give me a Python example of each.",
]


# ── Helpers ──

def _ensure_ids(tpl_out):
    if hasattr(tpl_out, 'input_ids'):
        ids = tpl_out.input_ids
    elif isinstance(tpl_out, torch.Tensor):
        ids = tpl_out
    else:
        ids = torch.tensor(tpl_out)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.to(torch.int32)


def _build_stop_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)
    return stop_ids


def _get_template_tokens(tokenizer):
    return {
        "im_start": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end":   tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "think":    tokenizer.convert_tokens_to_ids("<think>"),
        "nl":       198,
        "user":     tokenizer.encode("user", add_special_tokens=False),
        "assistant": tokenizer.encode("assistant", add_special_tokens=False),
    }


def _build_continuation(tokenizer, tpl_tokens, user_msg, has_stop_token):
    t = tpl_tokens
    msg_tokens = tokenizer.encode(user_msg, add_special_tokens=False)
    continuation = []
    if not has_stop_token:
        continuation += [t["im_end"], t["nl"]]
    else:
        continuation += [t["nl"]]
    continuation += [t["im_start"]] + t["user"] + [t["nl"]]
    continuation += msg_tokens
    continuation += [t["im_end"], t["nl"]]
    continuation += [t["im_start"]] + t["assistant"] + [t["nl"]]
    continuation += [t["think"], t["nl"]]
    return continuation


# ── CoreML Engine: Separate models ──

class SeparateEngine:
    def __init__(self, model_dir, label, compute_unit):
        self.embed = ct.models.MLModel(
            os.path.join(model_dir, "embeddings.mlpackage"), compute_units=compute_unit)
        self.lmhead = ct.models.MLModel(
            os.path.join(model_dir, "lm_head.mlpackage"), compute_units=compute_unit)
        self.ffns = []
        for ci in range(NUM_CHUNKS):
            m = ct.models.MLModel(
                os.path.join(model_dir, f"ffn_{label}_chunk{ci}.mlpackage"),
                compute_units=compute_unit)
            self.ffns.append(m)
        spec = self.ffns[0].get_spec()
        self.inp_map = {}
        for inp in spec.description.input:
            try:
                self.inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.has_linear = 'linear_conv_state' in self.inp_map
        self.reset_all()

    def reset_all(self):
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
                              for _ in range(NUM_CHUNKS)]
            self.lin_recs = [np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
                             for _ in range(NUM_CHUNKS)]
        else:
            self.lin_convs = [None] * NUM_CHUNKS
            self.lin_recs = [None] * NUM_CHUNKS

    def _step(self, tok_id, pos):
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

    def prefill_and_decode(self, token_ids, start_pos, max_gen, stop_ids):
        t0 = time.time()
        for i, tid in enumerate(token_ids):
            pos = start_pos + i
            if pos >= CTX:
                break
            last_next = self._step(tid, pos)
        prefill_end_pos = start_pos + len(token_ids)
        t_prefill = (time.time() - t0) * 1000
        tokens = [last_next]
        t_dec = time.time()
        for gi in range(max_gen - 1):
            pos = prefill_end_pos + gi
            if pos >= CTX - 1:
                break
            next_id = self._step(tokens[-1], pos)
            tokens.append(next_id)
            if next_id in stop_ids:
                break
        t_decode = (time.time() - t_dec) * 1000
        end_pos = prefill_end_pos + len(tokens)
        return tokens, end_pos, t_prefill, t_decode

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        gc.collect()


# ── CoreML Engine: Dedup (combined multi-function) models ──

class DedupEngine:
    def __init__(self, combined_dir, model_dir, compute_unit):
        self.embed = ct.models.MLModel(
            os.path.join(model_dir, "embeddings.mlpackage"), compute_units=compute_unit)
        self.lmhead = ct.models.MLModel(
            os.path.join(model_dir, "lm_head.mlpackage"), compute_units=compute_unit)
        self.ffns = []
        for ci in range(NUM_CHUNKS):
            m = ct.models.MLModel(
                os.path.join(combined_dir, f"chunk{ci}.mlpackage"),
                compute_units=compute_unit, function_name="infer")
            self.ffns.append(m)
        spec = self.ffns[0].get_spec()
        self.inp_map = {}
        fn_inputs = None
        for fn in spec.description.functions:
            if fn.name == "infer":
                fn_inputs = fn.input
                break
        if fn_inputs is None:
            fn_inputs = spec.description.input
        for inp in fn_inputs:
            try:
                self.inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass
        self.has_linear = 'linear_conv_state' in self.inp_map
        self.reset_all()

    def reset_all(self):
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
                              for _ in range(NUM_CHUNKS)]
            self.lin_recs = [np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
                             for _ in range(NUM_CHUNKS)]
        else:
            self.lin_convs = [None] * NUM_CHUNKS
            self.lin_recs = [None] * NUM_CHUNKS

    def _step(self, tok_id, pos):
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

    def prefill_and_decode(self, token_ids, start_pos, max_gen, stop_ids):
        t0 = time.time()
        for i, tid in enumerate(token_ids):
            pos = start_pos + i
            if pos >= CTX:
                break
            last_next = self._step(tid, pos)
        prefill_end_pos = start_pos + len(token_ids)
        t_prefill = (time.time() - t0) * 1000
        tokens = [last_next]
        t_dec = time.time()
        for gi in range(max_gen - 1):
            pos = prefill_end_pos + gi
            if pos >= CTX - 1:
                break
            next_id = self._step(tokens[-1], pos)
            tokens.append(next_id)
            if next_id in stop_ids:
                break
        t_decode = (time.time() - t_dec) * 1000
        end_pos = prefill_end_pos + len(tokens)
        return tokens, end_pos, t_prefill, t_decode

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        gc.collect()


# ── Run modes ──

def run_fresh(engine, tokenizer, turns, max_gen, stop_ids, engine_name):
    conversation = []
    results = []
    for ti, user_msg in enumerate(turns):
        print(f"\n  {engine_name} Turn {ti+1} [fresh]: {user_msg[:60]}")
        conversation.append({"role": "user", "content": user_msg})
        input_ids = _ensure_ids(tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True))
        prompt_len = input_ids.shape[1]
        if prompt_len + max_gen > CTX:
            input_ids = input_ids[:, -(CTX - max_gen):]
            prompt_len = input_ids.shape[1]
        engine.reset_all()
        token_list = input_ids[0].tolist()
        gen_tokens, end_pos, pf_ms, dc_ms = engine.prefill_and_decode(
            token_list, 0, max_gen, stop_ids)
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        response_for_template = "<think>\n" + raw_text
        conversation.append({"role": "assistant", "content": response_for_template})
        results.append({
            'turn': ti + 1, 'prompt_len': prompt_len,
            'tokens': gen_tokens, 'text': raw_text,
            'prefill_ms': pf_ms, 'decode_ms': dc_ms, 'end_pos': end_pos,
        })
        print(f"    [{prompt_len} tok prompt, end_pos={end_pos}] {raw_text[:120]}")
    return results


def run_incremental(engine, tokenizer, tpl_tokens, turns, max_gen, stop_ids, engine_name):
    conversation = []
    results = []
    current_pos = 0
    for ti, user_msg in enumerate(turns):
        print(f"\n  {engine_name} Turn {ti+1} [incremental]: {user_msg[:60]}")
        if ti == 0:
            conversation.append({"role": "user", "content": user_msg})
            input_ids = _ensure_ids(tokenizer.apply_chat_template(
                conversation, return_tensors="pt", add_generation_prompt=True,
                enable_thinking=True))
            new_tokens = input_ids[0].tolist()
            new_start = 0
        else:
            has_stop = any(t in stop_ids for t in results[-1]['tokens'][-1:])
            new_tokens = _build_continuation(tokenizer, tpl_tokens, user_msg, has_stop)
            new_start = current_pos
            conversation.append({"role": "user", "content": user_msg})
        prompt_len = len(new_tokens)
        if new_start + prompt_len + max_gen > CTX:
            excess = (new_start + prompt_len + max_gen) - CTX
            new_tokens = new_tokens[excess:]
            new_start += excess
            prompt_len = len(new_tokens)
        gen_tokens, end_pos, pf_ms, dc_ms = engine.prefill_and_decode(
            new_tokens, new_start, max_gen, stop_ids)
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        response_for_template = "<think>\n" + raw_text
        conversation.append({"role": "assistant", "content": response_for_template})
        current_pos = end_pos
        results.append({
            'turn': ti + 1, 'prompt_len': prompt_len,
            'new_start': new_start,
            'tokens': gen_tokens, 'text': raw_text,
            'prefill_ms': pf_ms, 'decode_ms': dc_ms, 'end_pos': end_pos,
        })
        print(f"    [start={new_start}, {prompt_len} tok, end_pos={end_pos}] {raw_text[:120]}")
    return results


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Validate Qwen3.5-4B multi-round conversation")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory with exported .mlpackage files")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to HuggingFace model directory (for tokenizer)")
    parser.add_argument("--tokens", type=int, default=40,
                        help="Max tokens to generate per turn (default: 40)")
    parser.add_argument("--skip-separate", action="store_true",
                        help="Skip separate model validation (only test dedup)")
    args = parser.parse_args()

    label = f"LUT{LUT_BITS}"
    combined_dir = os.path.join(args.model_dir, f"combined_{label}_dedup")
    compute_unit = ct.ComputeUnit.CPU_AND_NE

    # Verify required files
    required = ["embeddings.mlpackage", "lm_head.mlpackage"]
    for ci in range(NUM_CHUNKS):
        required.append(f"ffn_{label}_chunk{ci}.mlpackage")
        required.append(os.path.join(f"combined_{label}_dedup", f"chunk{ci}.mlpackage"))
    missing = [f for f in required if not os.path.exists(os.path.join(args.model_dir, f))]
    if missing:
        print("ERROR: Missing files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun qwen35_export.py and qwen35_combine.py first.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)
    tpl_tokens = _get_template_tokens(tokenizer)

    print("=" * 80)
    print("  Qwen3.5-4B Multi-Round Validation — Milestone 1")
    print(f"  Model dir: {args.model_dir}")
    print(f"  Config: {label}, {NUM_CHUNKS} chunks, CTX={CTX}")
    print(f"  Tokens/turn: {args.tokens}, Turns: {len(CONVERSATION_TURNS)}")
    print("=" * 80)

    all_configs = []

    # ── Separate models ──
    if not args.skip_separate:
        print(f"\n{'='*60}")
        print(f"  Separate {label} — fresh")
        print(f"{'='*60}")
        sep_engine = SeparateEngine(args.model_dir, label, compute_unit)
        sep_fresh = run_fresh(sep_engine, tokenizer, CONVERSATION_TURNS,
                              args.tokens, stop_ids, f"Separate {label}")
        all_configs.append((f"Separate {label} fresh", sep_fresh))

        print(f"\n{'='*60}")
        print(f"  Separate {label} — incremental")
        print(f"{'='*60}")
        sep_engine.reset_all()
        sep_engine.states = [m.make_state() for m in sep_engine.ffns]
        if sep_engine.has_linear:
            sep_engine.lin_convs = [np.zeros(sep_engine.inp_map['linear_conv_state'], dtype=np.float16)
                                    for _ in range(NUM_CHUNKS)]
            sep_engine.lin_recs = [np.zeros(sep_engine.inp_map['linear_recurrent_state'], dtype=np.float16)
                                   for _ in range(NUM_CHUNKS)]
        sep_inc = run_incremental(sep_engine, tokenizer, tpl_tokens,
                                  CONVERSATION_TURNS, args.tokens, stop_ids, f"Separate {label}")
        all_configs.append((f"Separate {label} incremental", sep_inc))
        sep_engine.cleanup()

    # ── Dedup models ──
    print(f"\n{'='*60}")
    print(f"  Dedup {label} — fresh")
    print(f"{'='*60}")
    dedup_engine = DedupEngine(combined_dir, args.model_dir, compute_unit)
    dedup_fresh = run_fresh(dedup_engine, tokenizer, CONVERSATION_TURNS,
                            args.tokens, stop_ids, f"Dedup {label}")
    all_configs.append((f"Dedup {label} fresh", dedup_fresh))

    print(f"\n{'='*60}")
    print(f"  Dedup {label} — incremental")
    print(f"{'='*60}")
    dedup_engine2 = DedupEngine(combined_dir, args.model_dir, compute_unit)
    dedup_inc = run_incremental(dedup_engine2, tokenizer, tpl_tokens,
                                CONVERSATION_TURNS, args.tokens, stop_ids, f"Dedup {label}")
    all_configs.append((f"Dedup {label} incremental", dedup_inc))
    dedup_engine.cleanup()
    dedup_engine2.cleanup()

    # ── Results table ──
    num_turns = len(CONVERSATION_TURNS)
    ref_name, ref_results = all_configs[0]

    print(f"\n{'='*80}")
    print(f"  RESULTS")
    print(f"{'='*80}")

    for ti in range(num_turns):
        ref = ref_results[ti]
        print(f"\n  Turn {ti+1} (prompt: {ref['prompt_len']} tok, end_pos: {ref.get('end_pos','?')})")
        print(f"    {'Config':<40} {'Prefill(ms)':>11} {'Decode(ms)':>11} {'Match':>14}")
        print(f"    {'-'*78}")
        print(f"    {ref_name:<40} {ref['prefill_ms']:>11.0f} {ref['decode_ms']:>11.0f} {'---':>14}")
        for cname, cresults in all_configs[1:]:
            cr = cresults[ti]
            matches = sum(1 for a, b in zip(ref['tokens'], cr['tokens']) if a == b)
            total = min(len(ref['tokens']), len(cr['tokens']))
            pct = 100 * matches / total if total > 0 else 0
            status = f"{matches}/{total} ({pct:.0f}%)"
            print(f"    {cname:<40} {cr['prefill_ms']:>11.0f} {cr['decode_ms']:>11.0f} {status:>14}")

    print(f"\n  — Generated Text —")
    for ti in range(num_turns):
        print(f"\n  Turn {ti+1}: {CONVERSATION_TURNS[ti]}")
        for cname, cresults in all_configs:
            print(f"    [{cname}] {cresults[ti]['text'][:200]}")

    # ── Verdict ──
    print(f"\n{'='*80}")
    print("  VERDICT")
    print(f"{'='*80}")

    all_pass = True
    checks = 0

    # Dedup fresh vs incremental
    for ti in range(num_turns):
        f_toks = dedup_fresh[ti]['tokens']
        i_toks = dedup_inc[ti]['tokens']
        matches = sum(1 for a, b in zip(f_toks, i_toks) if a == b)
        total = min(len(f_toks), len(i_toks))
        pct = 100 * matches / total if total > 0 else 0
        status = "PASS" if pct == 100 else "FAIL"
        if pct < 100:
            all_pass = False
        checks += 1
        print(f"  Turn {ti+1}: dedup fresh vs incremental = {matches}/{total} ({pct:.0f}%) [{status}]")

    # Dedup vs separate (fresh)
    if not args.skip_separate:
        print()
        for ti in range(num_turns):
            s_toks = sep_fresh[ti]['tokens']
            d_toks = dedup_fresh[ti]['tokens']
            matches = sum(1 for a, b in zip(s_toks, d_toks) if a == b)
            total = min(len(s_toks), len(d_toks))
            pct = 100 * matches / total if total > 0 else 0
            status = "PASS" if pct == 100 else "FAIL"
            if pct < 100:
                all_pass = False
            checks += 1
            print(f"  Turn {ti+1}: separate vs dedup (fresh) = {matches}/{total} ({pct:.0f}%) [{status}]")

        # Separate fresh vs incremental
        print()
        for ti in range(num_turns):
            f_toks = sep_fresh[ti]['tokens']
            i_toks = sep_inc[ti]['tokens']
            matches = sum(1 for a, b in zip(f_toks, i_toks) if a == b)
            total = min(len(f_toks), len(i_toks))
            pct = 100 * matches / total if total > 0 else 0
            status = "PASS" if pct == 100 else "FAIL"
            if pct < 100:
                all_pass = False
            checks += 1
            print(f"  Turn {ti+1}: separate fresh vs incremental = {matches}/{total} ({pct:.0f}%) [{status}]")

        # Separate incremental vs dedup incremental
        print()
        for ti in range(num_turns):
            s_toks = sep_inc[ti]['tokens']
            d_toks = dedup_inc[ti]['tokens']
            matches = sum(1 for a, b in zip(s_toks, d_toks) if a == b)
            total = min(len(s_toks), len(d_toks))
            pct = 100 * matches / total if total > 0 else 0
            status = "PASS" if pct == 100 else "FAIL"
            if pct < 100:
                all_pass = False
            checks += 1
            print(f"  Turn {ti+1}: separate vs dedup (incremental) = {matches}/{total} ({pct:.0f}%) [{status}]")

    if all_pass:
        print(f"\n  ALL {checks} CHECKS PASS ✓")
    else:
        print(f"\n  SOME CHECKS FAILED — investigation needed")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
