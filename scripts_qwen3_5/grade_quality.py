#!/usr/bin/env python3
"""Qwen3.5-4B Quality Grading — 3-way comparison: PyTorch FP16 vs CoreML LUT6 vs LUT4.

Generates responses for diverse prompts across all three backends, prints them
side-by-side for honest quality assessment.

Usage:
    python scripts_qwen3_5/grade_quality.py \
        --hf-model /path/to/Qwen3.5-4B \
        --lut6-dir /path/to/9chunk_lut6 \
        --lut4-dir /path/to/9chunk_lut4 \
        --tokens 200
"""
import sys, os, gc, time, argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import torch

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES
from anemll.models.qwen3_5_model import Qwen35ForCausalLM as Qwen35Model, Qwen35Config

# ── Prompts for quality grading ──────────────────────────────────────
GRADING_PROMPTS = [
    # 1. Chinese language / culture
    "教我做红烧鱼",
    # 2. Reasoning / logic
    "A farmer has 17 sheep. All but 9 run away. How many are left? Explain your reasoning step by step.",
    # 3. Code generation
    "Write a Python function to find the longest palindromic substring in a given string. Include docstring and examples.",
    # 4. Creative writing
    "Write a short poem about the stars, in the style of Li Bai.",
    # 5. Multi-turn context (single prompt that references prior context)
    "Explain what a neural network is, then give a simple analogy a 10-year-old would understand.",
]

def _build_stop_ids(tokenizer):
    """Build stop token set dynamically from the tokenizer."""
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tok = tokenizer.convert_tokens_to_ids(name)
        if tok is not None and tok != tokenizer.unk_token_id:
            stop_ids.add(tok)
    return stop_ids


def _sample_top_p(logits, temperature=0.6, top_p=0.95,
                  generated_ids=None, repetition_penalty=1.1):
    """Temperature + top-p (nucleus) sampling with repetition penalty."""
    logits = logits.astype(np.float64)
    # Apply repetition penalty
    if generated_ids and repetition_penalty != 1.0:
        for tid in set(generated_ids):
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty
    if temperature <= 0:
        return int(np.argmax(logits))
    logits /= temperature
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        mask = np.zeros_like(probs, dtype=bool)
        mask[sorted_idx[:cutoff]] = True
        probs[~mask] = 0.0
        probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def _wrap_prompt(tokenizer, user_msg):
    """Tokenize a single-turn prompt using Qwen chat template."""
    messages = [{"role": "user", "content": user_msg}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=True)
    if hasattr(input_ids, 'input_ids'):
        input_ids = input_ids.input_ids
    return input_ids


# ── PyTorch FP16 Backend ─────────────────────────────────────────────

def run_pytorch_fp16(hf_path, tokenizer, prompts, max_tokens):
    """Run repo's own Qwen3.5 model in FP16 greedy auto-regressive mode.

    Uses full-sequence forward passes (no KV cache) — slower but correct
    reference for quality comparison.
    """
    print("\n" + "=" * 70)
    print("  Loading PyTorch FP16 model (repo implementation)...")
    print("=" * 70)
    cfg = Qwen35Config.from_json(os.path.join(hf_path, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35Model(cfg)
    assert model.load_pretrained_weights(hf_path), f"Failed to load weights from {hf_path}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    results = []
    for i, prompt in enumerate(prompts):
        input_ids = _wrap_prompt(tokenizer, prompt)
        ids_list = input_ids[0].tolist() if hasattr(input_ids, 'tolist') else list(input_ids[0])
        print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}... ({len(ids_list)} tok prompt)")
        t0 = time.time()
        gen_ids = []
        seq = list(ids_list)  # current sequence
        with torch.no_grad():
            for _ in range(max_tokens):
                if len(seq) >= CTX:
                    break
                # Full-sequence forward (no KV cache, just causal mask)
                inp = torch.tensor([seq], dtype=torch.long)
                seq_len = len(seq)
                pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
                # Causal mask: (1, 1, seq_len, seq_len)
                mask = torch.full((1, 1, seq_len, seq_len), -65504.0, dtype=torch.float32)
                mask = torch.triu(mask, diagonal=1)  # upper triangle = -inf
                logits = model(inp, pos, causal_mask=mask, IN_PREFILL=True)
                # Take the last token's logits
                next_id = int(logits[0, -1].argmax(-1))
                gen_ids.append(next_id)
                if next_id in STOP_IDS:
                    break
                seq.append(next_id)
        elapsed = time.time() - t0
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        results.append({"prompt": prompt, "text": text, "tokens": len(gen_ids),
                        "time": elapsed})
        print(f"    {len(gen_ids)} tok in {elapsed:.1f}s")
    del model; gc.collect()
    return results


# ── CoreML Backend ───────────────────────────────────────────────────

def _extract_logits(lm_out):
    """Extract full logits array from lm_head output (may be split into logitsN)."""
    keys = sorted(lm_out.keys())
    if len(keys) == 1:
        return list(lm_out.values())[0].flatten()
    # Split logits: sort by numeric suffix to get correct order
    parts = []
    for k in sorted(keys, key=lambda x: int(x.replace("logits", ""))):
        parts.append(lm_out[k].flatten())
    return np.concatenate(parts)


class CoreMLEngine:
    """Minimal CoreML inference engine for dedup models."""

    def __init__(self, model_dir, label, num_chunks, probe_dir=None):
        self.num_chunks = num_chunks
        cu = ct.ComputeUnit.CPU_AND_NE
        # probe_dir: directory with separate .mlpackage files for shape probing
        # (can differ from model_dir, e.g. use LUT6 separate models for LUT4)
        if probe_dir is None:
            probe_dir = model_dir

        # Embeddings
        embed_path = os.path.join(model_dir, "embeddings.mlpackage")
        print(f"  Loading embeddings...")
        self.embed = ct.models.MLModel(embed_path, compute_units=ct.ComputeUnit.CPU_ONLY)

        # LM Head
        lm_path = os.path.join(model_dir, "lm_head_logits.mlpackage")
        if not os.path.exists(lm_path):
            lm_path = os.path.join(model_dir, "lm_head.mlpackage")
        print(f"  Loading lm_head...")
        self.lmhead = ct.models.MLModel(lm_path, compute_units=ct.ComputeUnit.CPU_ONLY)

        # FFN chunks — prefer dedup combined, fall back to separate
        dedup_dir = os.path.join(model_dir, f"combined_{label}_dedup")
        self.ffns = []
        if os.path.isdir(dedup_dir):
            for ci in range(num_chunks):
                p = os.path.join(dedup_dir, f"chunk{ci}.mlpackage")
                print(f"  Loading dedup chunk {ci}...")
                m = ct.models.MLModel(p, compute_units=cu, function_name="infer")
                self.ffns.append(m)
        else:
            for ci in range(num_chunks):
                p = os.path.join(model_dir, f"ffn_{label}_chunk{ci}.mlpackage")
                print(f"  Loading separate chunk {ci}...")
                m = ct.models.MLModel(p, compute_units=cu)
                self.ffns.append(m)

        # Detect linear attention states — probe from separate models or dedup function spec
        # Use load_spec (no compilation) to avoid disk/memory overhead
        self.has_linear = False
        self.inp_maps = []
        for ci in range(num_chunks):
            # Try separate models in probe_dir and model_dir
            sep_path = None
            for search_dir in [probe_dir, model_dir]:
                if not os.path.isdir(search_dir):
                    continue
                for probe_label in [label, "LUT6", "LUT4"]:
                    candidate = os.path.join(search_dir, f"ffn_{probe_label}_chunk{ci}.mlpackage")
                    if os.path.exists(candidate):
                        sep_path = candidate
                        break
                if sep_path:
                    break
            inp_map = {}
            if sep_path:
                spec = ct.utils.load_spec(sep_path)
                for inp in spec.description.input:
                    if inp.type.WhichOneof('Type') == 'multiArrayType':
                        inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
            else:
                # Fall back to dedup model's per-function inputs (like validate.py)
                spec = self.ffns[ci].get_spec()
                fn_inputs = None
                for fn in spec.description.functions:
                    if fn.name == "infer":
                        fn_inputs = fn.input
                        break
                if fn_inputs is None:
                    fn_inputs = spec.description.input
                for inp in fn_inputs:
                    try:
                        inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except Exception:
                        pass
            self.inp_maps.append(inp_map)
            if 'linear_conv_state' in inp_map:
                self.has_linear = True

        self.states = None
        self.lin_convs = None
        self.lin_recs = None
        self.reset_all()

    def reset_all(self):
        self.states = [m.make_state() for m in self.ffns]
        if self.has_linear:
            self.lin_convs = [np.zeros(self.inp_maps[ci]['linear_conv_state'], dtype=np.float16)
                              for ci in range(self.num_chunks)]
            self.lin_recs = [np.zeros(self.inp_maps[ci]['linear_recurrent_state'], dtype=np.float16)
                             for ci in range(self.num_chunks)]
        else:
            self.lin_convs = [None] * self.num_chunks
            self.lin_recs = [None] * self.num_chunks

    def _step(self, tok_id, pos):
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        for ci in range(self.num_chunks):
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
        logits = _extract_logits(lm_out)
        return logits

    def generate(self, token_ids, max_tokens, stop_ids, temperature=0.6, top_p=0.95,
                 repetition_penalty=1.1):
        """Prefill then decode. Returns list of generated token IDs."""
        self.reset_all()
        # Prefill
        for i, tid in enumerate(token_ids):
            logits = self._step(tid, i)
        first_id = _sample_top_p(logits, temperature, top_p,
                                  generated_ids=[], repetition_penalty=repetition_penalty)
        # Decode
        pos = len(token_ids)
        tokens = [first_id]
        for _ in range(max_tokens - 1):
            if pos >= CTX - 1:
                break
            logits = self._step(tokens[-1], pos)
            nxt = _sample_top_p(logits, temperature, top_p,
                                generated_ids=tokens, repetition_penalty=repetition_penalty)
            tokens.append(nxt)
            pos += 1
            if nxt in stop_ids:
                break
        return tokens


def run_coreml(model_dir, label, num_chunks, tokenizer, prompts, max_tokens, name,
               stop_ids, temperature=0.6, top_p=0.95, repetition_penalty=1.1,
               probe_dir=None):
    """Run CoreML engine and return results."""
    print(f"\n{'=' * 70}")
    print(f"  Loading CoreML {name}...")
    print(f"{'=' * 70}")
    engine = CoreMLEngine(model_dir, label, num_chunks, probe_dir=probe_dir)
    results = []
    for i, prompt in enumerate(prompts):
        input_ids = _wrap_prompt(tokenizer, prompt)
        ids_list = input_ids[0].tolist() if hasattr(input_ids, 'tolist') else list(input_ids[0])
        print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}... ({len(ids_list)} tok prompt)")
        t0 = time.time()
        gen_ids = engine.generate(ids_list, max_tokens, stop_ids,
                                  temperature=temperature, top_p=top_p,
                                  repetition_penalty=repetition_penalty)
        elapsed = time.time() - t0
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        results.append({"prompt": prompt, "text": text, "tokens": len(gen_ids),
                        "time": elapsed})
        print(f"    {len(gen_ids)} tok in {elapsed:.1f}s")
    del engine; gc.collect()
    return results


# ── Formatting and display ───────────────────────────────────────────

def _clean_answer(text):
    """Extract the visible answer after </think> tags."""
    if "</think>" in text:
        parts = text.split("</think>")
        answer = parts[-1].strip()
        think = "</think>".join(parts[:-1]).strip()
        # Remove leading <think> if present
        if think.startswith("<think>"):
            think = think[len("<think>"):].strip()
        return think, answer
    return "", text.strip()


def display_results(all_results, prompts):
    """Display all results side-by-side for grading."""
    backends = list(all_results.keys())

    print("\n" + "=" * 80)
    print("  QUALITY GRADING — SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    for pi, prompt in enumerate(prompts):
        print(f"\n{'─' * 80}")
        print(f"  PROMPT {pi+1}: {prompt}")
        print(f"{'─' * 80}")

        for backend in backends:
            r = all_results[backend][pi]
            think, answer = _clean_answer(r["text"])
            tok_s = r["tokens"] / r["time"] if r["time"] > 0 else 0
            print(f"\n  ┌─ {backend} ({r['tokens']} tok, {r['time']:.1f}s, {tok_s:.1f} tok/s)")
            if think:
                # Show first 200 chars of thinking
                think_preview = think[:200] + ("..." if len(think) > 200 else "")
                print(f"  │ [think] {think_preview}")
            # Show answer (limit to 500 chars for display)
            ans_lines = answer[:500].split("\n")
            for line in ans_lines:
                print(f"  │ {line}")
            if len(answer) > 500:
                print(f"  │ ... ({len(answer)} chars total)")
            print(f"  └─")

    # Summary table
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Backend':<25s} {'Avg tok/s':>10s} {'Avg tokens':>12s}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 12}")
    for backend in backends:
        res = all_results[backend]
        avg_tps = np.mean([r["tokens"] / r["time"] for r in res if r["time"] > 0])
        avg_tok = np.mean([r["tokens"] for r in res])
        print(f"  {backend:<25s} {avg_tps:>10.1f} {avg_tok:>12.0f}")


def main():
    parser = argparse.ArgumentParser(description="Quality grading: PyTorch FP16 vs CoreML LUT6 vs LUT4")
    parser.add_argument("--hf-model", required=True, help="Path to HuggingFace Qwen3.5-4B")
    parser.add_argument("--lut6-dir", required=True, help="Path to 9-chunk LUT6 model dir")
    parser.add_argument("--lut4-dir", required=True, help="Path to 9-chunk LUT4 model dir")
    parser.add_argument("--tokens", type=int, default=200, help="Max tokens per response")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch FP16 (use if already known)")
    parser.add_argument("--num-chunks", type=int, default=NUM_CHUNKS, help="Number of chunks")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--rep-penalty", type=float, default=1.1, help="Repetition penalty (1.0=none)")
    args = parser.parse_args()

    print("=" * 80)
    print("  Qwen3.5-4B Quality Grading — 3-way Comparison")
    print(f"  PyTorch FP16 vs CoreML LUT6 vs CoreML LUT4")
    print(f"  Max tokens: {args.tokens}, Prompts: {len(GRADING_PROMPTS)}")
    print(f"  Temperature: {args.temperature}, Top-p: {args.top_p}, Rep-penalty: {args.rep_penalty}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    stop_ids = _build_stop_ids(tokenizer)
    print(f"  Stop IDs: {stop_ids}")

    all_results = {}

    # 1. PyTorch FP16
    if not args.skip_pytorch:
        try:
            all_results["PyTorch FP16"] = run_pytorch_fp16(
                args.hf_model, tokenizer, GRADING_PROMPTS, args.tokens)
        except Exception as e:
            print(f"  WARNING: PyTorch FP16 failed: {e}")
            print(f"  Skipping PyTorch FP16 (user baseline: 4.5/5)")

    # 2. CoreML LUT6
    if os.path.isdir(args.lut6_dir):
        all_results["CoreML LUT6 (9-chunk)"] = run_coreml(
            args.lut6_dir, "LUT6", args.num_chunks, tokenizer,
            GRADING_PROMPTS, args.tokens, "LUT6 9-chunk",
            stop_ids, temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.rep_penalty)

    # 3. CoreML LUT4 (use LUT6 dir for shape probing if LUT4 separate models missing)
    if os.path.isdir(args.lut4_dir):
        all_results["CoreML LUT4 (9-chunk)"] = run_coreml(
            args.lut4_dir, "LUT4", args.num_chunks, tokenizer,
            GRADING_PROMPTS, args.tokens, "LUT4 9-chunk",
            stop_ids, temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.rep_penalty, probe_dir=args.lut6_dir)

    display_results(all_results, GRADING_PROMPTS)


if __name__ == "__main__":
    main()
