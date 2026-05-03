#!/usr/bin/env python3
"""Quick multi-round validation using pre-compiled .mlmodelc models.
Avoids runtime compilation that requires boot drive space.
"""
import sys, os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import time, argparse
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import torch

from config import CTX, NUM_CHUNKS

CONVERSATION_TURNS = [
    "What is a stack in computer science?",
    "How does it compare to a queue?",
    "Give me a Python example of each.",
]


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


def _argmax_from_lm_out(lm_out):
    if "logits" in lm_out:
        return int(np.argmax(lm_out["logits"].flatten()))
    if "argmax_idx" in lm_out:
        return int(lm_out["argmax_idx"].flatten()[0])
    split_keys = sorted([k for k in lm_out if k.startswith("logits")],
                        key=lambda k: int(k.replace("logits", "")))
    if split_keys:
        full = np.concatenate([lm_out[k].flatten() for k in split_keys])
        return int(np.argmax(full))
    raise KeyError(f"Cannot find logits in lm_head output: {list(lm_out.keys())}")


class DedupEngine:
    """Load dedup models from .mlpackage (or .mlmodelc if available)."""

    def __init__(self, model_dir, label, num_chunks, compute_unit):
        combined_dir = os.path.join(model_dir, f"combined_{label}_dedup")

        # embed + lmhead - prefer .mlpackage for multifunction support
        for ext in [".mlpackage"]:
            el_path = os.path.join(model_dir, f"embed_lmhead_combined{ext}")
            if os.path.exists(el_path):
                print(f"  Loading embed from {el_path} (CPU_ONLY)...")
                self.embed = ct.models.MLModel(el_path, compute_units=ct.ComputeUnit.CPU_ONLY,
                                               function_name="embed")
                print(f"  Loading lmhead from {el_path} (CPU_ONLY)...")
                self.lmhead = ct.models.MLModel(el_path, compute_units=ct.ComputeUnit.CPU_ONLY,
                                                function_name="lmhead")
                break
        else:
            raise FileNotFoundError(f"No embed_lmhead_combined in {model_dir}")

        # FFN chunks
        self.ffns = []
        self.num_chunks = num_chunks
        for ci in range(num_chunks):
            path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
            print(f"  Loading dedup chunk {ci} from {path} ({compute_unit})...")
            m = ct.models.MLModel(path, compute_units=compute_unit, function_name="infer")
            self.ffns.append(m)

        # Get input shapes
        self.inp_maps = []
        for ci in range(num_chunks):
            spec = self.ffns[ci].get_spec()
            imap = {}
            fn_inputs = None
            for fn in spec.description.functions:
                if fn.name == "infer":
                    fn_inputs = fn.input
                    break
            if fn_inputs is None:
                fn_inputs = spec.description.input
            for inp in fn_inputs:
                try:
                    imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                except Exception:
                    pass
            self.inp_maps.append(imap)
        self.inp_map = self.inp_maps[0]
        self.has_linear = 'linear_conv_state' in self.inp_map
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
                "position_ids": np.array([[pos], [pos], [pos]], dtype=np.int32),
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
        return _argmax_from_lm_out(lm_out)


def run_conversation(engine, tokenizer, max_gen):
    stop_ids = _build_stop_ids(tokenizer)
    tpl_tokens = _get_template_tokens(tokenizer)
    conversation = []
    accumulated_tokens = []

    for ti, user_msg in enumerate(CONVERSATION_TURNS):
        print(f"\n{'='*60}")
        print(f"Turn {ti+1}: {user_msg}")
        print('='*60)

        conversation.append({"role": "user", "content": user_msg})
        if ti == 0:
            input_ids = _ensure_ids(tokenizer.apply_chat_template(
                conversation, return_tensors="pt", add_generation_prompt=True,
                enable_thinking=True))
            token_list = input_ids[0].tolist()
        else:
            has_stop = any(t in stop_ids for t in gen_tokens[-1:])
            continuation = _build_continuation(tokenizer, tpl_tokens, user_msg, has_stop)
            token_list = accumulated_tokens + gen_tokens + continuation

        prompt_len = len(token_list)
        if prompt_len + max_gen > CTX:
            token_list = token_list[-(CTX - max_gen):]
            prompt_len = len(token_list)

        engine.reset_all()

        # Prefill
        t0 = time.time()
        for i, tid in enumerate(token_list):
            pos = i
            if pos >= CTX:
                break
            last_next = engine._step(tid, pos)
        prefill_end_pos = len(token_list)
        t_prefill = time.time() - t0

        # Decode
        gen_tokens = [last_next]
        t_dec = time.time()
        for gi in range(max_gen - 1):
            pos = prefill_end_pos + gi
            if pos >= CTX - 1:
                break
            next_id = engine._step(gen_tokens[-1], pos)
            gen_tokens.append(next_id)
            if next_id in stop_ids:
                break
        t_decode = time.time() - t_dec

        # Fill last position for KV cache continuity
        fill_pos = prefill_end_pos + len(gen_tokens) - 1
        if gen_tokens and fill_pos < CTX:
            engine._step(gen_tokens[-1], fill_pos)

        accumulated_tokens = list(token_list)
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        response_for_template = "<think>\n" + raw_text
        conversation.append({"role": "assistant", "content": response_for_template})

        tok_sec = len(gen_tokens) / t_decode if t_decode > 0 else 0
        print(f"  Prompt: {prompt_len} tokens")
        print(f"  Generated: {len(gen_tokens)} tokens in {t_decode*1000:.0f}ms ({tok_sec:.1f} tok/s)")
        print(f"  Prefill: {t_prefill*1000:.0f}ms")
        print(f"\n  Output:\n  {raw_text[:500]}")
        if len(raw_text) > 500:
            print(f"  ...[truncated, {len(raw_text)} chars total]")


def main():
    parser = argparse.ArgumentParser(description="Validate with compiled .mlmodelc models")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory with compiled models")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=120,
                        help="Max tokens per turn")
    parser.add_argument("--label", type=str, default="LUT4")
    parser.add_argument("--chunks", type=int, default=None)
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use CPU_ONLY to skip ANE compilation (saves boot drive space)")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_dir

    num_chunks = args.chunks if args.chunks else NUM_CHUNKS
    compute_unit = ct.ComputeUnit.CPU_ONLY if args.cpu_only else ct.ComputeUnit.CPU_AND_NE

    print(f"Loading compiled models from {args.model_dir}")
    print(f"  Label: {args.label}, Chunks: {num_chunks}, CTX: {CTX}")
    print(f"  Tokens/turn: {args.tokens}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    engine = DedupEngine(args.model_dir, args.label, num_chunks, compute_unit)
    run_conversation(engine, tokenizer, args.tokens)


if __name__ == "__main__":
    main()
