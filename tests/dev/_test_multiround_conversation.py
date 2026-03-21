#!/usr/bin/env python3
"""Multi-round conversation validation for Qwen3.5-4B with stateless linear attention.

Tests two state-management strategies:

  "fresh"       -- Reset ALL states each turn, re-prefill from position 0.
                   Gold-standard baseline (always correct).

  "incremental" -- Keep ALL states across turns.  Only prefill NEW tokens
                   (turn-separator + new user message) at the current cache
                   position.  Most efficient approach.

Both modes should produce *identical* output for every turn because:
  - The KV cache sees the same tokens at the same positions.
  - The linear recurrent state accumulates from the same token sequence.

Key fixes vs previous versions:
  - Qwen3.5 chat template adds <think>\\n in the generation prompt.  When
    feeding the response back we prepend <think>\\n so the re-tokenized
    prompt matches the original cache content exactly.
  - Incremental mode constructs turn-continuation tokens from known special
    token IDs (no re-tokenization roundtrip issues).

Usage:
    python tests/dev/_test_multiround_conversation.py --tokens 40
    python tests/dev/_test_multiround_conversation.py --tokens 40 --skip-export
    python tests/dev/_test_multiround_conversation.py --tokens 40 --skip-pytorch
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


# ── Helpers ──────────────────────────────────────────────────────────

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
    """Pre-compute the special token IDs used in Qwen chat template."""
    return {
        "im_start": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end":   tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "think":    tokenizer.convert_tokens_to_ids("<think>"),
        "nl":       198,  # \n
        "user":     tokenizer.encode("user", add_special_tokens=False),
        "assistant": tokenizer.encode("assistant", add_special_tokens=False),
    }


def _build_continuation(tokenizer, tpl_tokens, user_msg, has_stop_token):
    """Build the exact token sequence for a new turn (to append at current_pos).

    If the previous generation ended WITHOUT a stop token, prepends <|im_end|>\\n.
    Then appends: <|im_start|>user\\n{msg}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n

    Returns list of int token IDs.
    """
    t = tpl_tokens
    msg_tokens = tokenizer.encode(user_msg, add_special_tokens=False)

    continuation = []
    if not has_stop_token:
        continuation += [t["im_end"], t["nl"]]  # close previous assistant turn
    else:
        continuation += [t["nl"]]  # newline after stop token

    continuation += [t["im_start"]] + t["user"] + [t["nl"]]
    continuation += msg_tokens
    continuation += [t["im_end"], t["nl"]]
    continuation += [t["im_start"]] + t["assistant"] + [t["nl"]]
    continuation += [t["think"], t["nl"]]  # <think>\n (thinking prompt)
    return continuation


# ── PyTorch engine ───────────────────────────────────────────────────

class PyTorchEngine:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        ane_d1, ane_d2 = ane_conv_state_shape(
            cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim,
            max(1, int(cfg.text_config.linear_conv_kernel_dim)))
        self.ane_d1 = ane_d1
        self.ane_d2 = ane_d2
        self.reset_all()

    def reset_all(self):
        self.pt_k = [torch.zeros(LAYERS_PER_CHUNK, self.cfg.num_key_value_heads, CTX,
                                  self.cfg.head_dim, dtype=MODEL_DTYPE) for _ in range(NUM_CHUNKS)]
        self.pt_v = [torch.zeros(LAYERS_PER_CHUNK, self.cfg.num_key_value_heads, CTX,
                                  self.cfg.head_dim, dtype=MODEL_DTYPE) for _ in range(NUM_CHUNKS)]
        self.pt_c = [torch.zeros(LAYERS_PER_CHUNK, self.ane_d1, self.ane_d2, dtype=MODEL_DTYPE)
                     for _ in range(NUM_CHUNKS)]
        self.pt_r = [torch.zeros(LAYERS_PER_CHUNK, self.cfg.text_config.linear_num_value_heads,
                                  self.cfg.text_config.linear_key_head_dim,
                                  self.cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)
                     for _ in range(NUM_CHUNKS)]

    def _step(self, tok_id, pos):
        tok = torch.tensor([[tok_id]], dtype=torch.int32)
        hidden = self.model.model.embed_tokens(tok).to(MODEL_DTYPE)
        mask = torch.full((1, 1, 1, CTX), float("-inf"), dtype=MODEL_DTYPE)
        mask[:, :, :, :pos + 1] = 0
        for ci, (s, e) in enumerate(CHUNKS):
            hidden = self.model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden,
                position_ids=torch.tensor([pos], dtype=torch.int32),
                causal_mask=mask,
                current_pos=torch.tensor([pos], dtype=torch.int32),
                kv_cache_0=None,
                k_cache=self.pt_k[ci], v_cache=self.pt_v[ci],
                linear_conv_state=self.pt_c[ci],
                linear_recurrent_state=self.pt_r[ci],
                start_layer=s, end_layer=e,
                apply_final_norm=(ci == NUM_CHUNKS - 1))
        logits = self.model.lm_head(
            hidden.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
        return torch.argmax(logits, dim=-1).item()

    def prefill_and_decode(self, token_ids, start_pos, max_gen, stop_ids):
        with torch.no_grad():
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


# ── CoreML engine ────────────────────────────────────────────────────

class CoreMLEngine:
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
        for m in self.ffns:
            del m
        gc.collect()


# ── Mode: fresh ──────────────────────────────────────────────────────

def run_fresh(engine, tokenizer, turns, max_gen, stop_ids, engine_name):
    """Reset all states each turn, re-encode full history, prefill from 0.

    KEY: prepend '<think>\\n' to the assistant response before feeding it back
    so the template re-encoding aligns with what was actually in the cache
    (the Qwen3.5 template adds <think>\\n as part of the generation prompt).
    """
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

        # Decode response; prepend <think>\n so re-tokenization aligns next turn
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        response_for_template = "<think>\n" + raw_text
        conversation.append({"role": "assistant", "content": response_for_template})

        results.append({
            'turn': ti + 1, 'prompt_len': prompt_len,
            'tokens': gen_tokens, 'text': raw_text,
            'prefill_ms': pf_ms, 'decode_ms': dc_ms,
            'end_pos': end_pos,
        })
        print(f"    [{prompt_len} tok prompt, end_pos={end_pos}] {raw_text[:120]}")
    return results


# ── Mode: incremental ───────────────────────────────────────────────

def run_incremental(engine, tokenizer, tpl_tokens, turns, max_gen, stop_ids, engine_name):
    """Keep all states across turns.  Only prefill NEW tokens at current_pos.

    After each generation, construct the turn-continuation tokens manually
    (no re-tokenization) and prefill them at the current cache position.
    """
    conversation = []
    results = []
    current_pos = 0

    for ti, user_msg in enumerate(turns):
        print(f"\n  {engine_name} Turn {ti+1} [incremental]: {user_msg[:60]}")

        if ti == 0:
            # First turn: full template prefill (same as fresh)
            conversation.append({"role": "user", "content": user_msg})
            input_ids = _ensure_ids(tokenizer.apply_chat_template(
                conversation, return_tensors="pt", add_generation_prompt=True,
                enable_thinking=True))
            new_tokens = input_ids[0].tolist()
            new_start = 0
        else:
            # Subsequent turns: construct continuation from known token IDs
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
            print(f"    Trimmed {excess} tokens to fit CTX={CTX}")

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
            'prefill_ms': pf_ms, 'decode_ms': dc_ms,
            'end_pos': end_pos,
        })
        print(f"    [start={new_start}, {prompt_len} tok, end_pos={end_pos}] {raw_text[:120]}")
    return results


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
        print(f"    {'Config':<35} {'Prefill(ms)':>11} {'Decode(ms)':>11} {'Match':>10}")
        print(f"    {'-'*69}")
        print(f"    {ref_name:<35} {ref['prefill_ms']:>11.0f} {ref['decode_ms']:>11.0f} {'---':>10}")
        for cname, cresults in configs[1:]:
            cr = cresults[ti]
            matches = sum(1 for a, b in zip(ref['tokens'], cr['tokens']) if a == b)
            total = min(len(ref['tokens']), len(cr['tokens']))
            pct = 100 * matches / total if total > 0 else 0
            status = f"{matches}/{total} ({pct:.0f}%)"
            print(f"    {cname:<35} {cr['prefill_ms']:>11.0f} {cr['decode_ms']:>11.0f} {status:>10}")
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


# ── Export ───────────────────────────────────────────────────────────

def export_if_needed(model, cfg, out_dir, lut_bits):
    label = f"LUT{lut_bits}" if lut_bits else "fp16"
    embed_path = os.path.join(out_dir, "embeddings.mlpackage")
    if not os.path.exists(embed_path):
        print("  Exporting embeddings...")
        conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                               num_chunks=NUM_CHUNKS, lut_bits=None)
        ml = conv.convert_part_1(model)
        ml.save(embed_path)
        del ml, conv; gc.collect()

    lmhead_path = os.path.join(out_dir, "lm_head.mlpackage")
    if not os.path.exists(lmhead_path):
        print("  Exporting lm_head...")
        conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                               num_chunks=NUM_CHUNKS, lut_bits=None)
        ml = conv.convert_part_3(model, argmax_in_model=False)
        ml.save(lmhead_path)
        del ml, conv; gc.collect()

    for ci in range(NUM_CHUNKS):
        chunk_path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        if not os.path.exists(chunk_path):
            t0 = time.time()
            print(f"  Exporting decode chunk {ci} ({label})...")
            conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                                   num_chunks=NUM_CHUNKS, lut_bits=lut_bits,
                                   per_channel=8)
            ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
            ml.save(chunk_path)
            del ml, conv; gc.collect()
            print(f"    Done ({time.time()-t0:.1f}s)")
        else:
            print(f"  Chunk {ci} exists, skipping.")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--export-dir", type=str, default="/tmp/lut_vs_nolut_export")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-pytorch", action="store_true")
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
    print("  Multi-Round Conversation Validation -- Qwen3.5-4B Stateless")
    print(f"  Config: {label}, {NUM_CHUNKS} chunks, CTX={CTX}")
    print(f"  Tokens/turn: {max_gen}, Turns: {len(CONVERSATION_TURNS)}")
    print("=" * 80)

    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX

    model = None
    if not args.skip_export or not args.skip_pytorch:
        model = Qwen35ForCausalLM(cfg)
        assert model.load_pretrained_weights(MODEL_PATH)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    if not args.skip_export:
        export_if_needed(model, cfg, out_dir, lut_bits)

    all_configs = []

    # ── PyTorch: fresh + incremental ──
    if not args.skip_pytorch:
        print(f"\n{'='*60}")
        print("  PyTorch -- fresh")
        print(f"{'='*60}")
        pt_fresh_engine = PyTorchEngine(model, cfg)
        pt_fresh = run_fresh(pt_fresh_engine, tokenizer, CONVERSATION_TURNS,
                             max_gen, stop_ids, "PyTorch")
        all_configs.append(("PyTorch fresh", pt_fresh))
        del pt_fresh_engine

        print(f"\n{'='*60}")
        print("  PyTorch -- incremental")
        print(f"{'='*60}")
        pt_inc_engine = PyTorchEngine(model, cfg)
        pt_inc = run_incremental(pt_inc_engine, tokenizer, tpl_tokens,
                                 CONVERSATION_TURNS, max_gen, stop_ids, "PyTorch")
        all_configs.append(("PyTorch incremental", pt_inc))
        del pt_inc_engine

    if model is not None:
        del model
        gc.collect()

    # ── CoreML: fresh ──
    print(f"\n{'='*60}")
    print(f"  CoreML {label} -- fresh")
    print(f"{'='*60}")
    cml_fresh_engine = CoreMLEngine(out_dir, label, compute_unit)
    cml_fresh = run_fresh(cml_fresh_engine, tokenizer, CONVERSATION_TURNS,
                          max_gen, stop_ids, f"CoreML {label}")
    all_configs.append((f"CoreML {label} fresh", cml_fresh))
    cml_fresh_engine.cleanup()

    # ── CoreML: incremental ──
    print(f"\n{'='*60}")
    print(f"  CoreML {label} -- incremental")
    print(f"{'='*60}")
    cml_inc_engine = CoreMLEngine(out_dir, label, compute_unit)
    cml_inc = run_incremental(cml_inc_engine, tokenizer, tpl_tokens,
                              CONVERSATION_TURNS, max_gen, stop_ids, f"CoreML {label}")
    all_configs.append((f"CoreML {label} incremental", cml_inc))
    cml_inc_engine.cleanup()

    # ── Summary ──
    compare_results(all_configs, tokenizer)

    # ── Pass/Fail ──
    print(f"\n{'='*80}")
    print("  VERDICT")
    print(f"{'='*80}")

    # Check fresh vs incremental for CoreML
    all_pass = True
    for ti in range(len(CONVERSATION_TURNS)):
        f_toks = cml_fresh[ti]['tokens']
        i_toks = cml_inc[ti]['tokens']
        matches = sum(1 for a, b in zip(f_toks, i_toks) if a == b)
        total = min(len(f_toks), len(i_toks))
        pct = 100 * matches / total if total > 0 else 0
        status = "PASS" if pct == 100 else "FAIL"
        if pct < 100:
            all_pass = False
        print(f"  Turn {ti+1}: fresh vs incremental = {matches}/{total} ({pct:.0f}%) [{status}]")

    if all_pass:
        print("\n  ALL TURNS MATCH -- multi-round incremental inference is correct!")
    else:
        print("\n  SOME TURNS DIVERGE -- investigation needed.")

    print(f"\nDone. Export dir: {out_dir}")


if __name__ == "__main__":
    main()
