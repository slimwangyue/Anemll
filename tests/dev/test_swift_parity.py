#!/usr/bin/env python3
"""
Minimal parity test: run the EXACT same inference logic as the Swift iOS app
using the 4-chunk models on Mac. This isolates whether the bug is in the
inference logic or in the 6-chunk models themselves.

Tests both think-on (with <think>) and think-off (without) prompt formats.
"""
import sys, os, time
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "qwen3_5_stable_models"))
HF_PATH = MODEL_DIR
CTX = 1024
NUM_CHUNKS = 4

def load_model(path, cu, function_name=None):
    kwargs = {"compute_units": cu}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)

def find_model(base_dir, name):
    # Prefer .mlpackage over .mlmodelc (some .mlmodelc are nested inside .mlpackage)
    for ext in (".mlpackage", ".mlmodelc"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def main():
    cu = ct.ComputeUnit.CPU_AND_NE
    tok = AutoTokenizer.from_pretrained(HF_PATH, use_fast=False)

    # Build prompts - both think-on and think-off
    prompt_no_think = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "What is the capital of France<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    prompt_think = prompt_no_think + "<think>\n"

    # Use HF apply_chat_template as reference
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France"},
    ]
    ref_text_think = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=True
    )
    ref_text_nothink = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=False
    )
    print("=== HF REFERENCE (think=True) ===")
    print(repr(ref_text_think))
    ref_ids_think = tok.encode(ref_text_think)
    print(f"IDs ({len(ref_ids_think)}): {ref_ids_think}")

    print("\n=== HF REFERENCE (think=False) ===")
    print(repr(ref_text_nothink))
    ref_ids_nothink = tok.encode(ref_text_nothink)
    print(f"IDs ({len(ref_ids_nothink)}): {ref_ids_nothink}")

    print("\n=== SWIFT FORMAT (no think) ===")
    print(repr(prompt_no_think))
    swift_ids = tok.encode(prompt_no_think)
    print(f"IDs ({len(swift_ids)}): {swift_ids}")
    print(f"Matches HF think=True (prefix): {swift_ids == ref_ids_think[:len(swift_ids)]}")

    print("\n=== SWIFT FORMAT (think) ===")
    print(repr(prompt_think))
    swift_think_ids = tok.encode(prompt_think)
    print(f"IDs ({len(swift_think_ids)}): {swift_think_ids}")
    print(f"Matches HF think=True: {swift_think_ids == ref_ids_think}")

    # ── Load models ──
    print("\n=== LOADING MODELS ===")
    combined_dir = os.path.join(MODEL_DIR, "combined_LUT4_dedup")
    use_combined = os.path.isdir(combined_dir)

    embed = load_model(find_model(MODEL_DIR, "embeddings"), cu)
    print("  embeddings loaded")

    try:
        lmhead = load_model(find_model(MODEL_DIR, "lm_head_logits"), cu)
        lmhead_mode = "logits"
    except FileNotFoundError:
        lmhead = load_model(find_model(MODEL_DIR, "lm_head"), cu)
        spec = lmhead.get_spec()
        out_names = [o.name for o in spec.description.output]
        lmhead_mode = "logits" if "logits" in out_names else "argmax"
    print(f"  lm_head loaded (mode={lmhead_mode})")

    # Detect logits key
    spec = lmhead.get_spec()
    out_names = [o.name for o in spec.description.output]
    split_keys = sorted([n for n in out_names if n.startswith("logits") and n[6:].isdigit()])
    if split_keys:
        logits_keys = split_keys
        print(f"  Split logits: {len(split_keys)}-way")
    else:
        logits_keys = None
        logits_key = "output_logits" if "output_logits" in out_names else "logits"
        print(f"  Single logits key: {logits_key}")

    ffns = []
    for ci in range(NUM_CHUNKS):
        if use_combined:
            path = find_model(combined_dir, f"chunk{ci}")
            m = load_model(path, cu, function_name="infer")
        else:
            path = find_model(MODEL_DIR, f"ffn_LUT4_chunk{ci}")
            m = load_model(path, cu)
        ffns.append(m)
        print(f"  chunk{ci} infer loaded")

    # Detect state shapes
    inp_shapes = {}
    fn_spec = ffns[0].get_spec()
    if use_combined and hasattr(fn_spec.description, 'functions'):
        for fn in fn_spec.description.functions:
            if fn.name == "infer":
                for inp in fn.input:
                    try:
                        inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except:
                        pass
                break
    if not inp_shapes:
        for inp in fn_spec.description.input:
            try:
                inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
            except:
                pass
    print(f"  State shapes: conv={inp_shapes.get('linear_conv_state')}, "
          f"rec={inp_shapes.get('linear_recurrent_state')}")

    # ── Run inference (mimics Swift step-by-step) ──
    def extract_logits(lm_out):
        if logits_keys:
            parts = [lm_out[k].flatten().astype(np.float32) for k in logits_keys]
            return np.concatenate(parts)
        return lm_out[logits_key].flatten().astype(np.float32)

    def run_inference(prompt_ids, label, max_tokens=50):
        print(f"\n{'='*60}")
        print(f"=== INFERENCE: {label} ({len(prompt_ids)} tokens) ===")
        print(f"{'='*60}")

        # Reset states (mirrors Swift resetStates)
        states = [m.make_state() for m in ffns]
        lin_convs = [np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16)
                     for _ in range(NUM_CHUNKS)]
        lin_recs = [np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16)
                    for _ in range(NUM_CHUNKS)]

        # Pre-allocate buffers (mirrors Swift)
        tok_buf = np.zeros((1, 1), dtype=np.int32)
        mask_buf = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        pos_buf = np.zeros(1, dtype=np.int32)

        position = 0

        def step_kv_only(tok_id, pos):
            nonlocal lin_convs, lin_recs
            tok_buf[0, 0] = tok_id
            hidden = list(embed.predict({"input_ids": tok_buf}).values())[0]

            mask_buf[:, :, :, :] = -65504.0
            mask_buf[:, :, :, :pos + 1] = 0
            pos_buf[0] = pos

            for ci in range(NUM_CHUNKS):
                inp = {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": pos_buf,
                    "causal_mask": mask_buf,
                    "current_pos": pos_buf,
                    "linear_conv_state": lin_convs[ci],
                    "linear_recurrent_state": lin_recs[ci],
                }
                out = ffns[ci].predict(inp, state=states[ci])
                hidden = out["output_hidden_states"]
                if 'linear_conv_state_out' in out:
                    lin_convs[ci] = out['linear_conv_state_out']
                    lin_recs[ci] = out['linear_recurrent_state_out']

        def step(tok_id, pos):
            nonlocal lin_convs, lin_recs
            tok_buf[0, 0] = tok_id
            hidden = list(embed.predict({"input_ids": tok_buf}).values())[0]

            mask_buf[:, :, :, :] = -65504.0
            mask_buf[:, :, :, :pos + 1] = 0
            pos_buf[0] = pos

            for ci in range(NUM_CHUNKS):
                inp = {
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": pos_buf,
                    "causal_mask": mask_buf,
                    "current_pos": pos_buf,
                    "linear_conv_state": lin_convs[ci],
                    "linear_recurrent_state": lin_recs[ci],
                }
                out = ffns[ci].predict(inp, state=states[ci])
                hidden = out["output_hidden_states"]
                if 'linear_conv_state_out' in out:
                    lin_convs[ci] = out['linear_conv_state_out']
                    lin_recs[ci] = out['linear_recurrent_state_out']

            lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
            if lmhead_mode == "logits":
                logits = extract_logits(lm_out)
                return int(np.argmax(logits))
            return int(lm_out["argmax_idx"].flatten()[0])

        # ── PREFILL (sequential, same as Swift for < 32 tokens) ──
        t0 = time.time()
        n = len(prompt_ids)
        for ti in range(n):
            if ti == n - 1:
                first_token = step(prompt_ids[ti], position)
            else:
                step_kv_only(prompt_ids[ti], position)
            position += 1
        prefill_time = time.time() - t0
        print(f"[prefill] {n} tokens in {prefill_time*1000:.0f}ms, "
              f"first_token={first_token} ({tok.decode([first_token])!r})")

        # ── DECODE ──
        stop_ids = set()
        if tok.eos_token_id is not None:
            stop_ids.add(tok.eos_token_id)
        for name in ["<|im_end|>", "<|endoftext|>"]:
            t_id = tok.convert_tokens_to_ids(name)
            if t_id is not None and t_id != tok.unk_token_id:
                stop_ids.add(t_id)

        generated = [first_token]
        current = first_token
        t0 = time.time()
        for gi in range(max_tokens - 1):
            if position >= CTX - 1:
                break
            if current in stop_ids:
                break
            next_id = step(current, position)
            position += 1
            generated.append(next_id)
            current = next_id
            if current in stop_ids:
                break
        decode_time = time.time() - t0

        text = tok.decode(generated, skip_special_tokens=False)
        tps = len(generated) / max(decode_time, 1e-9)
        print(f"[decode] {len(generated)} tokens in {decode_time:.1f}s "
              f"({tps:.1f} tok/s)")
        print(f"[output] {text}")
        print(f"[token_ids] {generated[:20]}...")
        return text

    # Test 1: With <think> (should work like chat_server)
    run_inference(swift_think_ids, "with <think>", max_tokens=80)

    # Test 2: Without <think> (Swift no-think format)
    run_inference(swift_ids, "without <think>", max_tokens=80)


if __name__ == "__main__":
    main()
