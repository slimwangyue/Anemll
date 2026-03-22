#!/usr/bin/env python3
"""Validate dynamic KV slicing: batch prefill at different positions must match
sequential (one-token) decode.

Test plan:
  1. Sequential baseline: process all prompt tokens one-by-one and generate N decode tokens.
  2. Batch prefill: process full BATCH_SIZE blocks via prefill models, then decode.
  3. Compare: all generated tokens must match exactly.

Usage:
    python tests/dev/qwen35_dynamic_kv_validate.py \
        --model-dir /Users/yw68/Anemll_remote_run/qwen35_milestone1_2 \
        --tokenizer /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time, argparse
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

BATCH_SIZE = 256
CTX = 1024
NUM_CHUNKS = 4
NUM_DECODE_TOKENS = 20  # number of tokens to generate after prefill


def load_model(base_dir, name, cu, function_name=None):
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            kwargs = {"compute_units": cu}
            if function_name:
                kwargs["function_name"] = function_name
            if ext == ".mlmodelc":
                return ct.models.CompiledMLModel(p, cu)
            return ct.models.MLModel(p, **kwargs)
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def get_input_shapes(model):
    """Extract input shapes from model spec."""
    spec = model.get_spec()
    shapes = {}
    for inp in spec.description.input:
        try:
            shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
        except Exception:
            pass
    return shapes


class InferenceEngine:
    def __init__(self, model_dir, tokenizer_path):
        cu = ct.ComputeUnit.CPU_AND_NE
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        print("Loading embeddings...")
        self.embed = load_model(model_dir, "embeddings", cu)
        print("Loading lm_head...")
        self.lmhead = load_model(model_dir, "lm_head", cu)
        print("Loading decode chunks...")
        self.ffns = [load_model(model_dir, f"ffn_LUT4_chunk{i}", cu) for i in range(NUM_CHUNKS)]
        print("Loading prefill chunks (ALL compute units)...")
        cu_pf = ct.ComputeUnit.ALL  # Prefill needs ALL due to dynamic kv_write_end on ANE
        self.prefills = [load_model(model_dir, f"prefill_LUT4_chunk{i}", cu_pf) for i in range(NUM_CHUNKS)]

        self.inp_map = get_input_shapes(self.ffns[0])
        self.model_dir = model_dir

    def _make_states(self):
        states = [m.make_state() for m in self.ffns]
        lin_convs = [np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
        lin_recs = [np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
        return states, lin_convs, lin_recs

    def step(self, tok_id, pos, states, lin_convs, lin_recs):
        """Single-token decode. Returns next_token_id."""
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        kv_write_end = np.zeros((pos + 1,), dtype=np.int32)

        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
                "kv_write_end": kv_write_end,
            }
            out = self.ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    def batch_prefill(self, token_ids, block_start, states, lin_convs, lin_recs):
        """Process BATCH_SIZE tokens through prefill models. Returns next_token_id."""
        assert len(token_ids) == BATCH_SIZE
        end_step = block_start + BATCH_SIZE

        input_ids = np.array([token_ids], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": input_ids}).values())[0]

        mask = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
        for i in range(BATCH_SIZE):
            mask[0, 0, i, :block_start + i + 1] = 0

        pos_ids = np.arange(block_start, end_step, dtype=np.int32)
        current_pos = np.array([block_start], dtype=np.int32)
        kv_write_end = np.zeros((end_step,), dtype=np.int32)

        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": current_pos,
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
                "kv_write_end": kv_write_end,
            }
            out = self.prefills[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])


def run_sequential(engine, prompt_tokens, n_decode):
    """Process all tokens one-by-one. Returns list of generated token ids."""
    states, lin_convs, lin_recs = engine._make_states()
    pos = 0
    last_next = None
    t0 = time.time()
    for tok_id in prompt_tokens:
        last_next = engine.step(tok_id, pos, states, lin_convs, lin_recs)
        pos += 1
    t_prefill = time.time() - t0

    generated = [last_next]
    t0 = time.time()
    for _ in range(n_decode - 1):
        if pos >= CTX - 1:
            break
        next_id = engine.step(generated[-1], pos, states, lin_convs, lin_recs)
        pos += 1
        generated.append(next_id)
    t_decode = time.time() - t0

    print(f"  Sequential: {len(prompt_tokens)} prompt ({t_prefill:.1f}s) + "
          f"{len(generated)} decode ({t_decode:.1f}s, {len(generated)/max(t_decode,1e-9):.1f} tok/s)")
    return generated


def run_batch_prefill(engine, prompt_tokens, n_decode):
    """Batch prefill + decode. Returns list of generated token ids."""
    # Create states from decode models — shared between prefill and decode
    states, lin_convs, lin_recs = engine._make_states()

    pos = 0
    last_next = None
    t0 = time.time()
    n_batch = 0

    # Batch prefill as many blocks as possible
    while len(prompt_tokens) - n_batch >= BATCH_SIZE and pos + BATCH_SIZE <= CTX:
        batch = prompt_tokens[n_batch:n_batch + BATCH_SIZE]
        last_next = engine.batch_prefill(batch, block_start=pos, states=states,
                                          lin_convs=lin_convs, lin_recs=lin_recs)
        n_batch += BATCH_SIZE
        pos += BATCH_SIZE

    # Remaining prompt tokens via sequential decode
    for tok_id in prompt_tokens[n_batch:]:
        last_next = engine.step(tok_id, pos, states, lin_convs, lin_recs)
        pos += 1
    t_prefill = time.time() - t0

    generated = [last_next]
    t0 = time.time()
    for _ in range(n_decode - 1):
        if pos >= CTX - 1:
            break
        next_id = engine.step(generated[-1], pos, states, lin_convs, lin_recs)
        pos += 1
        generated.append(next_id)
    t_decode = time.time() - t0

    n_seq = len(prompt_tokens) - n_batch
    print(f"  Batch: {n_batch} batch + {n_seq} seq ({t_prefill:.1f}s) + "
          f"{len(generated)} decode ({t_decode:.1f}s, {len(generated)/max(t_decode,1e-9):.1f} tok/s)")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Validate dynamic KV slicing parity")
    parser.add_argument("--model-dir", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1_2")
    parser.add_argument("--tokenizer", default="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B")
    parser.add_argument("--n-decode", type=int, default=NUM_DECODE_TOKENS)
    args = parser.parse_args()

    engine = InferenceEngine(args.model_dir, args.tokenizer)

    # Build a prompt that's > BATCH_SIZE tokens to test multi-block prefill
    messages = [{"role": "user", "content": "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, gravitational time dilation, and the field equations. Also discuss experimental evidence and modern applications. Please be thorough. Provide extensive historical context about Einstein's development of general relativity. Discuss the precession of Mercury's orbit, gravitational lensing, gravitational waves detected by LIGO, and the recent Event Horizon Telescope image of a black hole. Elaborate on each topic with mathematical formulations where applicable. Also compare and contrast general relativity with Newtonian gravity. Discuss the implications for cosmology, including the expansion of the universe and dark energy."}]
    input_ids = engine.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=True)
    if hasattr(input_ids, 'input_ids'):
        input_ids = input_ids.input_ids
    prompt_tokens = input_ids[0].tolist()
    print(f"\nPrompt: {len(prompt_tokens)} tokens ({len(prompt_tokens)//BATCH_SIZE} full blocks)")

    print(f"\n{'='*70}")
    print(f"  VALIDATION: Sequential vs Batch Prefill")
    print(f"  Prompt: {len(prompt_tokens)} tokens, Decode: {args.n_decode} tokens")
    print(f"  Blocks: {len(prompt_tokens)//BATCH_SIZE} full ({BATCH_SIZE} each)")
    print(f"{'='*70}")

    print("\n[1/2] Sequential (baseline):")
    seq_tokens = run_sequential(engine, prompt_tokens, args.n_decode)

    print("\n[2/2] Batch prefill:")
    batch_tokens = run_batch_prefill(engine, prompt_tokens, args.n_decode)

    # Compare
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")

    match = seq_tokens == batch_tokens
    n = min(len(seq_tokens), len(batch_tokens))
    first_mismatch = -1
    for i in range(n):
        if seq_tokens[i] != batch_tokens[i]:
            first_mismatch = i
            break

    if match:
        seq_text = engine.tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"  ✅ PARITY: 100% match ({n} tokens)")
        print(f"  Decoded text: {seq_text[:200]}...")
    else:
        print(f"  ❌ MISMATCH at token {first_mismatch}:")
        print(f"     Sequential: {seq_tokens[max(0,first_mismatch-2):first_mismatch+3]}")
        print(f"     Batch:      {batch_tokens[max(0,first_mismatch-2):first_mismatch+3]}")
        matching = first_mismatch
        pct = matching / n * 100
        print(f"     {matching}/{n} tokens matched ({pct:.1f}%)")
        seq_text = engine.tokenizer.decode(seq_tokens[:first_mismatch+5], skip_special_tokens=True)
        batch_text = engine.tokenizer.decode(batch_tokens[:first_mismatch+5], skip_special_tokens=True)
        print(f"     Sequential text: ...{seq_text[-100:]}")
        print(f"     Batch text:      ...{batch_text[-100:]}")

    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
