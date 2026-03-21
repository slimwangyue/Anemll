#!/usr/bin/env python3
"""Experiment: Accuracy impact of LUT4 quantization on embed and lm_head.

Tests 4 configurations using 3-turn conversation (fresh mode):
  A) Baseline:  fp16 embed + fp16 lmhead  (current)
  B) LUT4-embed: LUT4 embed + fp16 lmhead
  C) LUT4-lmhead: fp16 embed + LUT4 lmhead
  D) LUT4-both: LUT4 embed + LUT4 lmhead

All configs share the same LUT4 FFN chunks.
Also reports size savings including tie_word_embeddings dedup (Option 2).

Usage:
    python tests/dev/_test_embed_lmhead_lut4.py --tokens 40
    python tests/dev/_test_embed_lmhead_lut4.py --tokens 40 --skip-export
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, time, argparse
import numpy as np
import torch
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 256
NUM_CHUNKS = 4
CHUNKS = [(0, 8), (8, 16), (16, 24), (24, 32)]

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


def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


# ── CoreML engine with swappable embed/lmhead ───────────────────────

class FlexEngine:
    """Engine that accepts any embed + lmhead + LUT4 FFN chunks."""

    def __init__(self, embed_path, lmhead_path, ffn_dir, ffn_label,
                 compute_unit, config_name="default"):
        self.name = config_name
        self.embed = ct.models.MLModel(embed_path, compute_units=compute_unit)
        self.lmhead = ct.models.MLModel(lmhead_path, compute_units=compute_unit)
        self.ffns = []
        for ci in range(NUM_CHUNKS):
            m = ct.models.MLModel(
                os.path.join(ffn_dir, f"ffn_{ffn_label}_chunk{ci}.mlpackage"),
                compute_units=compute_unit)
            self.ffns.append(m)

        # Get input spec
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


# ── Run ──────────────────────────────────────────────────────────────

def run_fresh(engine, tokenizer, turns, max_gen, stop_ids):
    """Run 3-turn conversation from scratch each turn (fresh mode)."""
    conversation = []
    results = []
    for ti, user_msg in enumerate(turns):
        print(f"\n    Turn {ti+1} [{engine.name}]: {user_msg[:60]}")
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
        print(f"      [{prompt_len} tok, end_pos={end_pos}] {raw_text[:120]}")
    return results


# ── Export LUT4 versions ─────────────────────────────────────────────

def export_lut4_embed(model, out_dir, per_channel=8):
    """Export embeddings with LUT4 quantization."""
    path = os.path.join(out_dir, "embeddings_LUT4.mlpackage")
    if os.path.exists(path):
        sz = dir_size_mb(path)
        print(f"  LUT4 embed already exists ({sz:.1f} MB), skipping.")
        return path
    t0 = time.time()
    print("  Exporting LUT4 embeddings...")
    conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                           num_chunks=NUM_CHUNKS, lut_bits=4,
                           per_channel=per_channel)
    ml = conv.convert_part_1(model)
    ml.save(path)
    sz = dir_size_mb(path)
    print(f"    Done ({time.time()-t0:.1f}s) — {sz:.1f} MB")
    del ml, conv
    gc.collect()
    return path


def export_lut4_lmhead(model, out_dir, per_channel=8):
    """Export lm_head with LUT4 quantization."""
    path = os.path.join(out_dir, "lm_head_LUT4.mlpackage")
    if os.path.exists(path):
        sz = dir_size_mb(path)
        print(f"  LUT4 lm_head already exists ({sz:.1f} MB), skipping.")
        return path
    t0 = time.time()
    print("  Exporting LUT4 lm_head...")
    conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                           num_chunks=NUM_CHUNKS, lut_bits=4,
                           per_channel=per_channel)
    ml = conv.convert_part_3(model)
    ml.save(path)
    sz = dir_size_mb(path)
    print(f"    Done ({time.time()-t0:.1f}s) — {sz:.1f} MB")
    del ml, conv
    gc.collect()
    return path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--export-dir", type=str, default="/tmp/lut_vs_nolut_export")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip exporting LUT4 embed/lmhead (assume they exist)")
    parser.add_argument("--per-channel", type=int, default=8)
    args = parser.parse_args()

    max_gen = args.tokens
    out_dir = args.export_dir
    compute_unit = ct.ComputeUnit.CPU_AND_NE
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)

    print("=" * 80)
    print("  Experiment: LUT4 Quantization on Embed / LM Head")
    print(f"  Tokens/turn: {max_gen}, Turns: {len(CONVERSATION_TURNS)}, CTX={CTX}")
    print("=" * 80)

    # Paths for fp16 (existing)
    fp16_embed = os.path.join(out_dir, "embeddings.mlpackage")
    fp16_lmhead = os.path.join(out_dir, "lm_head.mlpackage")

    # ── 1. Export LUT4 embed and lm_head ──
    lut4_embed_path = os.path.join(out_dir, "embeddings_LUT4.mlpackage")
    lut4_lmhead_path = os.path.join(out_dir, "lm_head_LUT4.mlpackage")

    if not args.skip_export:
        cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
        cfg.context_length = CTX
        cfg.state_length = CTX

        # Export embed (load model fresh, release after)
        if not os.path.exists(lut4_embed_path):
            model = Qwen35ForCausalLM(cfg)
            assert model.load_pretrained_weights(MODEL_PATH)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            export_lut4_embed(model, out_dir, args.per_channel)
            del model; gc.collect()
        else:
            print(f"  LUT4 embed exists ({dir_size_mb(lut4_embed_path):.1f} MB), skipping.")

        # Export lm_head (load model fresh, release after)
        if not os.path.exists(lut4_lmhead_path):
            model = Qwen35ForCausalLM(cfg)
            assert model.load_pretrained_weights(MODEL_PATH)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            export_lut4_lmhead(model, out_dir, args.per_channel)
            del model; gc.collect()
        else:
            print(f"  LUT4 lm_head exists ({dir_size_mb(lut4_lmhead_path):.1f} MB), skipping.")

    # ── 2. Size comparison ──
    print(f"\n{'='*80}")
    print("  SIZE COMPARISON")
    print(f"{'='*80}")

    fp16_embed_sz = dir_size_mb(fp16_embed)
    fp16_lmhead_sz = dir_size_mb(fp16_lmhead)
    lut4_embed_sz = dir_size_mb(lut4_embed_path)
    lut4_lmhead_sz = dir_size_mb(lut4_lmhead_path)
    ffn_total = sum(dir_size_mb(os.path.join(out_dir, f"ffn_LUT4_chunk{ci}.mlpackage"))
                    for ci in range(NUM_CHUNKS))

    baseline_total = fp16_embed_sz + fp16_lmhead_sz + ffn_total

    configs_info = [
        ("A) fp16 embed + fp16 lmhead (baseline)", fp16_embed_sz + fp16_lmhead_sz + ffn_total),
        ("B) LUT4 embed + fp16 lmhead",            lut4_embed_sz + fp16_lmhead_sz + ffn_total),
        ("C) fp16 embed + LUT4 lmhead",             fp16_embed_sz + lut4_lmhead_sz + ffn_total),
        ("D) LUT4 embed + LUT4 lmhead",             lut4_embed_sz + lut4_lmhead_sz + ffn_total),
        ("E) tie_weight dedup (fp16, no dup)",       fp16_embed_sz + ffn_total),  # share embed=lmhead
    ]

    print(f"\n  {'Component':<35} {'fp16':>10} {'LUT4':>10} {'Saving':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Embeddings':<35} {fp16_embed_sz:>9.1f}M {lut4_embed_sz:>9.1f}M {fp16_embed_sz-lut4_embed_sz:>9.1f}M")
    print(f"  {'LM Head':<35} {fp16_lmhead_sz:>9.1f}M {lut4_lmhead_sz:>9.1f}M {fp16_lmhead_sz-lut4_lmhead_sz:>9.1f}M")
    print(f"  {'FFN (4 chunks, LUT4)':<35} {ffn_total:>9.1f}M {'---':>10} {'---':>10}")
    print()
    for label, sz in configs_info:
        saving = baseline_total - sz
        pct = 100 * saving / baseline_total if baseline_total > 0 else 0
        print(f"  {label:<45} {sz:>8.1f} MB  (save {saving:>7.1f} MB, {pct:>5.1f}%)")

    # ── 3. Run all 4 configs (fresh mode, 3 turns) ──
    print(f"\n{'='*80}")
    print("  ACCURACY COMPARISON (3-turn fresh mode)")
    print(f"{'='*80}")

    configs = [
        ("A) fp16+fp16 (baseline)", fp16_embed, fp16_lmhead),
        ("B) LUT4 embed + fp16 lmhead", lut4_embed_path, fp16_lmhead),
        ("C) fp16 embed + LUT4 lmhead", fp16_embed, lut4_lmhead_path),
        ("D) LUT4 embed + LUT4 lmhead", lut4_embed_path, lut4_lmhead_path),
    ]

    all_results = []

    for config_name, embed_path, lmhead_path in configs:
        print(f"\n  --- {config_name} ---")
        engine = FlexEngine(
            embed_path=embed_path,
            lmhead_path=lmhead_path,
            ffn_dir=out_dir,
            ffn_label="LUT4",
            compute_unit=compute_unit,
            config_name=config_name,
        )
        results = run_fresh(engine, tokenizer, CONVERSATION_TURNS, max_gen, stop_ids)
        all_results.append((config_name, results))
        engine.cleanup()

    # ── 4. Compare all configs vs baseline ──
    print(f"\n{'='*80}")
    print("  RESULTS SUMMARY")
    print(f"{'='*80}")

    ref_name, ref_results = all_results[0]  # baseline
    num_turns = len(CONVERSATION_TURNS)

    for ti in range(num_turns):
        ref = ref_results[ti]
        print(f"\n  Turn {ti+1}: \"{CONVERSATION_TURNS[ti]}\"")
        print(f"    {'Config':<45} {'Match':>12} {'Time(ms)':>10}")
        print(f"    {'-'*69}")
        for cname, cresults in all_results:
            cr = cresults[ti]
            if cname == ref_name:
                total = len(ref['tokens'])
                print(f"    {cname:<45} {'baseline':>12} {cr['prefill_ms']+cr['decode_ms']:>10.0f}")
            else:
                matches = sum(1 for a, b in zip(ref['tokens'], cr['tokens']) if a == b)
                total = min(len(ref['tokens']), len(cr['tokens']))
                pct = 100 * matches / total if total > 0 else 0
                status = f"{matches}/{total} ({pct:.0f}%)"
                print(f"    {cname:<45} {status:>12} {cr['prefill_ms']+cr['decode_ms']:>10.0f}")
                if pct < 100:
                    for pos, (a, b) in enumerate(zip(ref['tokens'], cr['tokens'])):
                        if a != b:
                            a_str = tokenizer.decode([a])
                            b_str = tokenizer.decode([b])
                            print(f"      1st diff at token {pos}: baseline=[{a_str}]({a}) vs [{b_str}]({b})")
                            break

    # ── 5. Overall verdict ──
    print(f"\n{'='*80}")
    print("  VERDICT")
    print(f"{'='*80}")

    for cname, cresults in all_results[1:]:
        total_match = 0
        total_tok = 0
        for ti in range(num_turns):
            ref = ref_results[ti]
            cr = cresults[ti]
            matches = sum(1 for a, b in zip(ref['tokens'], cr['tokens']) if a == b)
            total = min(len(ref['tokens']), len(cr['tokens']))
            total_match += matches
            total_tok += total
        overall_pct = 100 * total_match / total_tok if total_tok > 0 else 0
        # Find size from configs_info
        label_map = {"B)": 1, "C)": 2, "D)": 3}
        idx = next((i for i, (l, _) in enumerate(configs_info) if l.startswith(cname[:2])), None)
        sz = configs_info[idx][1] if idx is not None else 0
        saving = baseline_total - sz
        print(f"  {cname}")
        print(f"    Accuracy: {total_match}/{total_tok} ({overall_pct:.1f}%) match vs baseline")
        print(f"    Size:     {sz:.0f} MB (saves {saving:.0f} MB / {100*saving/baseline_total:.1f}%)")
        print()

    print(f"  Option 2: tie_word_embeddings dedup (same weights, 0% accuracy change)")
    dedup_sz = configs_info[4][1]
    dedup_saving = baseline_total - dedup_sz
    print(f"    Size:     {dedup_sz:.0f} MB (saves {dedup_saving:.0f} MB / {100*dedup_saving/baseline_total:.1f}%)")
    print()

    # ── 6. Generated text side-by-side ──
    print(f"\n{'='*80}")
    print("  GENERATED TEXT (side-by-side)")
    print(f"{'='*80}")
    for ti in range(num_turns):
        print(f"\n  Turn {ti+1}: \"{CONVERSATION_TURNS[ti]}\"")
        for cname, cresults in all_results:
            cr = cresults[ti]
            text_preview = cr['text'][:200].replace('\n', ' ')
            print(f"    [{cname[:20]}] {text_preview}")

    print(f"\nDone. Export dir: {out_dir}")


if __name__ == "__main__":
    main()
