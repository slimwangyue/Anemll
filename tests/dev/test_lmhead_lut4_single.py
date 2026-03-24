#!/usr/bin/env python3
"""Run a SINGLE lm_head config and save token output to JSON.

Called by test_lmhead_lut4_sweep_runner.py as a subprocess to avoid
memory pressure / segfaults when running multiple configs in one process.

Usage:
    python tests/dev/test_lmhead_lut4_single.py \
        --lmhead-path /path/to/lm_head.mlpackage \
        --label LUT4_gs1 \
        --tokens 40 \
        --output /tmp/lmhead_groupsize_sweep/results/LUT4_gs1.json
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, json, time, argparse
import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
STABLE_DIR = "/Users/yw68/Anemll/qwen3_5_stable_models"
CTX = 1024
NUM_CHUNKS = 4

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


def _load_model(path, compute_unit, function_name=None):
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, compute_unit)
    kwargs = {"compute_units": compute_unit}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def _find_model(base_dir, name):
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def _detect_shapes(ffn_model, use_combined):
    inp_map = {}
    spec = ffn_model.get_spec()
    fn_inputs = None
    if use_combined and hasattr(spec.description, 'functions'):
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
    return inp_map


class Engine:
    def __init__(self, lmhead_path, compute_unit, lmhead_compute_unit=None):
        cu = compute_unit
        lmhead_cu = lmhead_compute_unit or cu
        # Loading order MUST match chat_server.py: embed → lm_head → FFN(infer+prefill)
        # Loading prefill functions is REQUIRED — without them, lm_head.predict segfaults
        print(f"  Loading embeddings...", flush=True)
        t0 = time.time()
        self.embed = _load_model(_find_model(STABLE_DIR, "embeddings"), cu)
        print(f"    {time.time()-t0:.0f}s", flush=True)

        print(f"  Loading lm_head (cu={lmhead_cu})...", flush=True)
        t0 = time.time()
        self.lmhead = _load_model(lmhead_path, lmhead_cu)
        print(f"    {time.time()-t0:.0f}s", flush=True)

        combined_dir = os.path.join(STABLE_DIR, "combined_LUT4_dedup")
        use_combined = os.path.isdir(combined_dir)

        self.ffns = []
        self.prefills = []
        for ci in range(NUM_CHUNKS):
            if use_combined:
                path = _find_model(combined_dir, f"chunk{ci}")
                if path.endswith(".mlmodelc"):
                    use_combined = False
            if use_combined:
                t0 = time.time()
                m_infer = _load_model(path, cu, function_name="infer")
                print(f"    chunk {ci} infer  (combined) {time.time()-t0:.0f}s", flush=True)
                t0 = time.time()
                m_prefill = _load_model(path, cu, function_name="prefill")
                print(f"    chunk {ci} prefill (combined) {time.time()-t0:.0f}s", flush=True)
            else:
                path = _find_model(STABLE_DIR, f"ffn_LUT4_chunk{ci}")
                t0 = time.time()
                m_infer = _load_model(path, cu)
                print(f"    chunk {ci} (separate) {time.time()-t0:.0f}s", flush=True)
                m_prefill = None
            self.ffns.append(m_infer)
            self.prefills.append(m_prefill)

        self.inp_map = _detect_shapes(self.ffns[0], use_combined)

        # Pre-allocate reusable buffers (same as chat_server.py)
        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)

        self.reset()

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [
            np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
            for _ in range(NUM_CHUNKS)]
        self.lin_recs = [
            np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
            for _ in range(NUM_CHUNKS)]

    def step(self, tok_id, pos):
        # Use pre-allocated buffers (same as chat_server._step)
        self._tok_buf[0, 0] = tok_id
        hidden = list(self.embed.predict({"input_ids": self._tok_buf}).values())[0]

        self._mask_buf[:, :, :, :] = -65504.0
        self._mask_buf[:, :, :, :pos + 1] = 0

        self._pos_buf[0] = pos

        for ci in range(NUM_CHUNKS):
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
        if "logits" in lm_out or "output_logits" in lm_out:
            key = "output_logits" if "output_logits" in lm_out else "logits"
            return int(np.argmax(lm_out[key].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    def generate(self, token_ids, start_pos, max_gen, stop_ids):
        for i, tid in enumerate(token_ids):
            pos = start_pos + i
            if pos >= CTX:
                break
            last_next = self.step(tid, pos)
        prefill_end_pos = start_pos + len(token_ids)
        tokens = [last_next]
        for gi in range(max_gen - 1):
            pos = prefill_end_pos + gi
            if pos >= CTX - 1:
                break
            next_id = self.step(tokens[-1], pos)
            tokens.append(next_id)
            if next_id in stop_ids:
                break
        return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmhead-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lmhead-gpu", action="store_true",
                        help="Load lm_head on CPU_AND_GPU (for LUT4 models that hang on ANE)")
    parser.add_argument("--all-gpu", action="store_true",
                        help="Load ALL models on CPU_AND_GPU (when ANE is stuck)")
    args = parser.parse_args()

    if args.all_gpu:
        cu = ct.ComputeUnit.CPU_AND_GPU
        lmhead_cu = None  # same as cu
    else:
        cu = ct.ComputeUnit.CPU_AND_NE
        lmhead_cu = ct.ComputeUnit.CPU_AND_GPU if args.lmhead_gpu else None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)

    print(f"\n=== {args.label} (cu={cu}) ===", flush=True)
    t_start = time.time()
    engine = Engine(args.lmhead_path, cu, lmhead_compute_unit=lmhead_cu)
    load_time = time.time() - t_start
    print(f"  Models loaded in {load_time:.0f}s", flush=True)

    conversation = []
    results = []
    for ti, user_msg in enumerate(CONVERSATION_TURNS):
        print(f"  Turn {ti+1}: {user_msg[:50]}", flush=True)
        conversation.append({"role": "user", "content": user_msg})
        input_ids = _ensure_ids(tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True))
        prompt_len = input_ids.shape[1]
        if prompt_len + args.tokens > CTX:
            input_ids = input_ids[:, -(CTX - args.tokens):]
            prompt_len = input_ids.shape[1]
        engine.reset()
        t0 = time.time()
        gen_tokens = engine.generate(
            input_ids[0].tolist(), 0, args.tokens, stop_ids)
        elapsed = time.time() - t0
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": "<think>\n" + raw_text})
        results.append({
            'turn': ti + 1,
            'prompt_len': prompt_len,
            'tokens': gen_tokens,
            'text': raw_text,
            'elapsed': round(elapsed, 2),
        })
        print(f"    [{prompt_len} tok, {elapsed:.1f}s] {raw_text[:80]}", flush=True)

    output = {
        'label': args.label,
        'lmhead_path': args.lmhead_path,
        'load_time': round(load_time, 1),
        'results': results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
