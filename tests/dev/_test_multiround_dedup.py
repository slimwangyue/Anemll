#!/usr/bin/env python3
"""Multi-round conversation validation for DEDUP (weight-shared) models.

Reuses the same 3-turn conversation as _test_multiround_conversation.py
and validates that:
  1) DEDUP combined models produce IDENTICAL tokens to separate models.
  2) Fresh vs incremental mode match for DEDUP models.

Prerequisites:
  - Separate decode LUT4 chunks already exported in --export-dir
    (from _test_lut_vs_nolut_textgen.py or _test_multiround_conversation.py)
  - This script exports prefill chunks if missing, then combines with dedup.

Usage:
    python tests/dev/_test_multiround_dedup.py --tokens 40
    python tests/dev/_test_multiround_dedup.py --tokens 40 --skip-prefill-export
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, time, argparse
import numpy as np
import torch
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from anemll.utils.combine_models import _save_multifunction_dedup
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 256
NUM_CHUNKS = 4
CHUNKS = [(0, 8), (8, 16), (16, 24), (24, 32)]
LAYERS_PER_CHUNK = 8

CONVERSATION_TURNS = [
    "What is a stack in computer science?",
    "How does it compare to a queue?",
    "Give me a Python example of each.",
]


# ── Helpers (same as _test_multiround_conversation.py) ───────────────

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


# ── CoreML engine for SEPARATE models ────────────────────────────────

class SeparateEngine:
    def __init__(self, out_dir, label, compute_unit):
        self.embed = ct.models.MLModel(
            os.path.join(out_dir, "embeddings.mlpackage"), compute_units=compute_unit)
        self.lmhead = ct.models.MLModel(
            os.path.join(out_dir, "lm_head.mlpackage"), compute_units=compute_unit)
        self.ffns = []
        for ci in range(NUM_CHUNKS):
            m = ct.models.MLModel(
                os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage"),
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
        for m in self.ffns: del m
        gc.collect()


# ── CoreML engine for DEDUP (combined) models ───────────────────────

class DedupEngine:
    """Loads combined multifunction .mlpackage with function_name='infer'."""
    def __init__(self, combined_dir, out_dir, compute_unit):
        self.embed = ct.models.MLModel(
            os.path.join(out_dir, "embeddings.mlpackage"), compute_units=compute_unit)
        self.lmhead = ct.models.MLModel(
            os.path.join(out_dir, "lm_head.mlpackage"), compute_units=compute_unit)
        self.ffns = []
        for ci in range(NUM_CHUNKS):
            m = ct.models.MLModel(
                os.path.join(combined_dir, f"chunk{ci}.mlpackage"),
                compute_units=compute_unit, function_name="infer")
            self.ffns.append(m)

        # Get input spec from the infer function (multifunction: per-function I/O)
        spec = self.ffns[0].get_spec()
        self.inp_map = {}
        # For multifunction models, inputs are under spec.description.functions
        fn_inputs = None
        for fn in spec.description.functions:
            if fn.name == "infer":
                fn_inputs = fn.input
                break
        if fn_inputs is None:
            fn_inputs = spec.description.input  # fallback for single-function
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
        for m in self.ffns: del m
        gc.collect()


# ── Modes (fresh / incremental) ─────────────────────────────────────

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


# ── Export & Combine ─────────────────────────────────────────────────

def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def export_prefill_chunks(model, cfg, out_dir, lut_bits, per_channel=8):
    """Export prefill chunks (decode chunks assumed to already exist)."""
    label = f"LUT{lut_bits}" if lut_bits else "fp16"
    for ci in range(NUM_CHUNKS):
        pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage")
        if os.path.exists(pf_path):
            print(f"  Prefill chunk {ci} exists, skipping.")
            continue
        t0 = time.time()
        print(f"  Exporting prefill chunk {ci} ({label})...")
        conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                               num_chunks=NUM_CHUNKS, lut_bits=lut_bits,
                               per_channel=per_channel)
        ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
        ml.save(pf_path)
        del ml, conv; gc.collect()
        print(f"    Done ({time.time()-t0:.1f}s)")


def combine_dedup(out_dir, label):
    """Combine decode+prefill per chunk with ANEMLL-Dedup. Returns combined dir."""
    combined_dir = os.path.join(out_dir, f"combined_{label}_dedup")
    os.makedirs(combined_dir, exist_ok=True)
    total_size = 0.0
    for ci in range(NUM_CHUNKS):
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if os.path.exists(combined_path):
            sz = dir_size_mb(combined_path)
            total_size += sz
            print(f"  Chunk {ci} combined already exists ({sz:.1f}MB), skipping.")
            continue
        dec_path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage")
        if not os.path.exists(dec_path):
            raise FileNotFoundError(f"Decode chunk missing: {dec_path}")
        if not os.path.exists(pf_path):
            raise FileNotFoundError(f"Prefill chunk missing: {pf_path}")
        sources = [
            (dec_path, "main", "infer"),
            (pf_path, "main", "prefill"),
        ]
        t0 = time.time()
        print(f"  Combining chunk {ci} with dedup...")
        _save_multifunction_dedup(sources, combined_path,
                                  dedup_weights=True, verbose=False)
        dt = time.time() - t0
        sz = dir_size_mb(combined_path)
        total_size += sz
        print(f"    Done ({dt:.1f}s) — {sz:.1f} MB")
    return combined_dir, total_size


# ── Comparison ───────────────────────────────────────────────────────

def compare_results(configs, tokenizer):
    if not configs:
        return
    num_turns = len(configs[0][1])
    ref_name, ref_results = configs[0]

    print(f"\n{'='*80}")
    print(f"  MULTI-ROUND RESULTS")
    print(f"{'='*80}")

    for ti in range(num_turns):
        ref = ref_results[ti]
        print(f"\n  Turn {ti+1} (prompt: {ref['prompt_len']} tok, end_pos: {ref.get('end_pos','?')})")
        print(f"    {'Config':<40} {'Prefill(ms)':>11} {'Decode(ms)':>11} {'Match':>14}")
        print(f"    {'-'*78}")
        print(f"    {ref_name:<40} {ref['prefill_ms']:>11.0f} {ref['decode_ms']:>11.0f} {'---':>14}")
        for cname, cresults in configs[1:]:
            cr = cresults[ti]
            matches = sum(1 for a, b in zip(ref['tokens'], cr['tokens']) if a == b)
            total = min(len(ref['tokens']), len(cr['tokens']))
            pct = 100 * matches / total if total > 0 else 0
            status = f"{matches}/{total} ({pct:.0f}%)"
            print(f"    {cname:<40} {cr['prefill_ms']:>11.0f} {cr['decode_ms']:>11.0f} {status:>14}")
            if pct < 100:
                for pos, (a, b) in enumerate(zip(ref['tokens'], cr['tokens'])):
                    if a != b:
                        a_str = tokenizer.decode([a])
                        b_str = tokenizer.decode([b])
                        print(f"      1st diff at tok {pos}: [{a_str}]({a}) vs [{b_str}]({b})")
                        break

    print(f"\n  -- Generated Text --")
    for ti in range(num_turns):
        print(f"\n  Turn {ti+1}: {CONVERSATION_TURNS[ti]}")
        for cname, cresults in configs:
            print(f"    [{cname}] {cresults[ti]['text'][:200]}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--export-dir", type=str, default="/tmp/lut_vs_nolut_export")
    parser.add_argument("--skip-prefill-export", action="store_true",
                        help="Skip exporting prefill chunks (assume they exist)")
    parser.add_argument("--skip-combine", action="store_true",
                        help="Skip combining (assume combined dir exists)")
    parser.add_argument("--skip-separate", action="store_true",
                        help="Skip running separate models (only run dedup)")
    parser.add_argument("--lut", type=int, default=4)
    args = parser.parse_args()

    max_gen = args.tokens
    out_dir = args.export_dir
    lut_bits = args.lut
    label = f"LUT{lut_bits}" if lut_bits else "fp16"
    compute_unit = ct.ComputeUnit.CPU_AND_NE
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)
    tpl_tokens = _get_template_tokens(tokenizer)

    print("=" * 80)
    print("  Multi-Round Conversation: Separate vs DEDUP (Weight-Shared) Models")
    print(f"  Config: {label}, {NUM_CHUNKS} chunks, CTX={CTX}")
    print(f"  Tokens/turn: {max_gen}, Turns: {len(CONVERSATION_TURNS)}")
    print("=" * 80)

    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX

    # ── 1. Export prefill chunks if needed ──
    if not args.skip_prefill_export:
        missing = [ci for ci in range(NUM_CHUNKS)
                   if not os.path.exists(os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage"))]
        if missing:
            print(f"\n  Need to export prefill chunks: {missing}")
            model = Qwen35ForCausalLM(cfg)
            assert model.load_pretrained_weights(MODEL_PATH)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            export_prefill_chunks(model, cfg, out_dir, lut_bits)
            del model; gc.collect()
        else:
            print("\n  All prefill chunks exist.")

    # ── 2. Combine with dedup ──
    if not args.skip_combine:
        print(f"\n{'='*60}")
        print("  Combining with ANEMLL-Dedup")
        print(f"{'='*60}")
        combined_dir, dedup_total = combine_dedup(out_dir, label)
    else:
        combined_dir = os.path.join(out_dir, f"combined_{label}_dedup")
        dedup_total = sum(dir_size_mb(os.path.join(combined_dir, f"chunk{ci}.mlpackage"))
                         for ci in range(NUM_CHUNKS))

    # ── 3. Size comparison ──
    print(f"\n{'='*60}")
    print("  SIZE COMPARISON")
    print(f"{'='*60}")
    sep_dec = sum(dir_size_mb(os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage"))
                  for ci in range(NUM_CHUNKS))
    sep_pf = sum(dir_size_mb(os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage"))
                 for ci in range(NUM_CHUNKS))
    embed_sz = dir_size_mb(os.path.join(out_dir, "embeddings.mlpackage"))
    lmhead_sz = dir_size_mb(os.path.join(out_dir, "lm_head.mlpackage"))
    sep_total = sep_dec + sep_pf + embed_sz + lmhead_sz
    dedup_full = dedup_total + embed_sz + lmhead_sz
    print(f"  Separate decode:  {sep_dec:.1f} MB ({NUM_CHUNKS} chunks)")
    print(f"  Separate prefill: {sep_pf:.1f} MB ({NUM_CHUNKS} chunks)")
    print(f"  Embed + LMHead:   {embed_sz + lmhead_sz:.1f} MB")
    print(f"  TOTAL separate:   {sep_total:.1f} MB")
    print(f"  TOTAL dedup:      {dedup_full:.1f} MB  ({(1-dedup_full/sep_total)*100:.1f}% saving)")

    all_configs = []

    # ── 4. Separate models: fresh ──
    if not args.skip_separate:
        print(f"\n{'='*60}")
        print(f"  Separate {label} -- fresh")
        print(f"{'='*60}")
        sep_engine = SeparateEngine(out_dir, label, compute_unit)
        sep_fresh = run_fresh(sep_engine, tokenizer, CONVERSATION_TURNS,
                              max_gen, stop_ids, f"Separate {label}")
        all_configs.append((f"Separate {label} fresh", sep_fresh))

        # ── 5. Separate models: incremental ──
        print(f"\n{'='*60}")
        print(f"  Separate {label} -- incremental")
        print(f"{'='*60}")
        sep_engine.reset_all()
        # Need fresh states for incremental
        sep_states_bak = sep_engine.states
        sep_engine.states = [m.make_state() for m in sep_engine.ffns]
        if sep_engine.has_linear:
            sep_engine.lin_convs = [np.zeros(sep_engine.inp_map['linear_conv_state'], dtype=np.float16)
                                    for _ in range(NUM_CHUNKS)]
            sep_engine.lin_recs = [np.zeros(sep_engine.inp_map['linear_recurrent_state'], dtype=np.float16)
                                   for _ in range(NUM_CHUNKS)]
        sep_inc = run_incremental(sep_engine, tokenizer, tpl_tokens,
                                  CONVERSATION_TURNS, max_gen, stop_ids, f"Separate {label}")
        all_configs.append((f"Separate {label} incremental", sep_inc))
        sep_engine.cleanup()

    # ── 6. Dedup models: fresh ──
    print(f"\n{'='*60}")
    print(f"  Dedup {label} -- fresh")
    print(f"{'='*60}")
    dedup_engine = DedupEngine(combined_dir, out_dir, compute_unit)
    dedup_fresh = run_fresh(dedup_engine, tokenizer, CONVERSATION_TURNS,
                            max_gen, stop_ids, f"Dedup {label}")
    all_configs.append((f"Dedup {label} fresh", dedup_fresh))

    # ── 7. Dedup models: incremental ──
    print(f"\n{'='*60}")
    print(f"  Dedup {label} -- incremental")
    print(f"{'='*60}")
    dedup_engine2 = DedupEngine(combined_dir, out_dir, compute_unit)
    dedup_inc = run_incremental(dedup_engine2, tokenizer, tpl_tokens,
                                CONVERSATION_TURNS, max_gen, stop_ids, f"Dedup {label}")
    all_configs.append((f"Dedup {label} incremental", dedup_inc))
    dedup_engine.cleanup()
    dedup_engine2.cleanup()

    # ── 8. Summary ──
    compare_results(all_configs, tokenizer)

    # ── 9. Verdict ──
    print(f"\n{'='*80}")
    print("  VERDICT")
    print(f"{'='*80}")

    all_pass = True

    # Check dedup fresh vs incremental
    for ti in range(len(CONVERSATION_TURNS)):
        f_toks = dedup_fresh[ti]['tokens']
        i_toks = dedup_inc[ti]['tokens']
        matches = sum(1 for a, b in zip(f_toks, i_toks) if a == b)
        total = min(len(f_toks), len(i_toks))
        pct = 100 * matches / total if total > 0 else 0
        status = "PASS" if pct == 100 else "FAIL"
        if pct < 100:
            all_pass = False
        print(f"  Turn {ti+1}: dedup fresh vs incremental = {matches}/{total} ({pct:.0f}%) [{status}]")

    # Check dedup vs separate (fresh)
    if not args.skip_separate:
        print()
        for ti in range(len(CONVERSATION_TURNS)):
            s_toks = sep_fresh[ti]['tokens']
            d_toks = dedup_fresh[ti]['tokens']
            matches = sum(1 for a, b in zip(s_toks, d_toks) if a == b)
            total = min(len(s_toks), len(d_toks))
            pct = 100 * matches / total if total > 0 else 0
            status = "PASS" if pct == 100 else "FAIL"
            if pct < 100:
                all_pass = False
            print(f"  Turn {ti+1}: separate vs dedup (fresh) = {matches}/{total} ({pct:.0f}%) [{status}]")

    if all_pass:
        print("\n  ALL CHECKS PASS -- dedup models match separate models!")
    else:
        print("\n  SOME CHECKS DIVERGE -- investigation needed.")

    print(f"\nDone. Export dir: {out_dir}")
    print(f"Combined dir: {combined_dir}")


if __name__ == "__main__":
    main()
