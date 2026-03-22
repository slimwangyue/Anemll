#!/usr/bin/env python3
"""Profile prefill per-component timing to identify bottlenecks.

Measures: embed, mask creation, FFN chunks (x4), lm_head, numpy overhead.
"""
import sys, os, time, gc
import numpy as np
import coremltools as ct

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts_qwen3_5")
sys.path.insert(0, _SCRIPTS_DIR)  # must be first — repo root has empty config.py
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from config import DEFAULT_OUTPUT, CTX, NUM_CHUNKS
from transformers import AutoTokenizer

MODEL_DIR = DEFAULT_OUTPUT
COMBINED_DIR = os.path.join(MODEL_DIR, "combined_LUT4_dedup")
N_PROFILE_TOKENS = 20  # profile this many tokens

def find_model(base, name):
    for ext in [".mlmodelc", ".mlpackage"]:
        p = os.path.join(base, name + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    cu = ct.ComputeUnit.CPU_AND_NE

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    embed = ct.models.MLModel(find_model(MODEL_DIR, "embeddings"), compute_units=cu)
    lmhead = ct.models.MLModel(find_model(MODEL_DIR, "lm_head"), compute_units=cu)
    gc.collect()

    ffns = []
    for ci in range(NUM_CHUNKS):
        path = find_model(COMBINED_DIR, f"chunk{ci}")
        m = ct.models.MLModel(path, compute_units=cu, function_name="infer")
        ffns.append(m)
        gc.collect()

    # Init states
    states = [m.make_state() for m in ffns]

    # Read shapes
    spec = ffns[0].get_spec()
    inp_map = {}
    for fn in spec.description.functions:
        if fn.name == "infer":
            for inp in fn.input:
                try:
                    inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
                except Exception:
                    pass
            break

    lin_convs = [np.zeros(inp_map['linear_conv_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]
    lin_recs = [np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16) for _ in range(NUM_CHUNKS)]

    # Tokenize a prompt
    msgs = [{"role": "user", "content": "What is the capital of France? Tell me about its history."}]
    tpl = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    prompt_ids = list(tpl)[:N_PROFILE_TOKENS]
    print(f"\nProfiling {len(prompt_ids)} tokens...")
    print(f"Shapes: conv={inp_map['linear_conv_state']}, rec={inp_map['linear_recurrent_state']}")

    # Warmup 2 tokens
    print("\nWarmup (2 tokens)...")
    for i, tok_id in enumerate(prompt_ids[:2]):
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :i + 1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([i], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([i], dtype=np.int32),
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
            }
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    pos = 2
    print("Warmup done.\n")

    # Profile remaining tokens
    timings = {
        "embed": [],
        "mask_create": [],
        "np_prep": [],  # numpy array creation for inputs
        "ffn_0": [],
        "ffn_1": [],
        "ffn_2": [],
        "ffn_3": [],
        "ffn_state_copy": [],
        "lmhead": [],
        "lmhead_argmax": [],
        "total": [],
    }

    for ti, tok_id in enumerate(prompt_ids[2:]):
        t_total = time.perf_counter()

        # Embed
        t0 = time.perf_counter()
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]
        timings["embed"].append(time.perf_counter() - t0)

        # Mask
        t0 = time.perf_counter()
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0
        timings["mask_create"].append(time.perf_counter() - t0)

        # FFN chunks
        for ci in range(NUM_CHUNKS):
            t0 = time.perf_counter()
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
                "linear_conv_state": lin_convs[ci],
                "linear_recurrent_state": lin_recs[ci],
            }
            timings["np_prep"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            out = ffns[ci].predict(inp, state=states[ci])
            timings[f"ffn_{ci}"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']
            timings["ffn_state_copy"].append(time.perf_counter() - t0)

        # LM head
        t0 = time.perf_counter()
        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        timings["lmhead"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        if "logits" in lm_out:
            next_id = int(np.argmax(lm_out["logits"].flatten()))
        else:
            next_id = int(lm_out["argmax_idx"].flatten()[0])
        timings["lmhead_argmax"].append(time.perf_counter() - t0)

        timings["total"].append(time.perf_counter() - t_total)
        pos += 1

    # Report
    n = len(timings["total"])
    print(f"{'Component':<20} {'Mean (ms)':>10} {'% of total':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 62)
    total_mean = np.mean(timings["total"]) * 1000
    for key in ["embed", "mask_create", "np_prep", "ffn_0", "ffn_1", "ffn_2", "ffn_3",
                "ffn_state_copy", "lmhead", "lmhead_argmax", "total"]:
        vals = timings[key]
        if not vals:
            continue
        mean_ms = np.mean(vals) * 1000
        min_ms = np.min(vals) * 1000
        max_ms = np.max(vals) * 1000
        if key == "np_prep" or key == "ffn_state_copy":
            # These are per-chunk, so sum them for total per token
            mean_ms_total = mean_ms * 4
            pct = mean_ms_total / total_mean * 100
            print(f"{key + ' (x4)':<20} {mean_ms_total:>10.2f} {pct:>9.1f}% {min_ms:>10.3f} {max_ms:>10.3f}")
        elif key == "total":
            print("-" * 62)
            print(f"{'TOTAL per token':<20} {mean_ms:>10.2f} {'100.0':>9}% {min_ms:>10.3f} {max_ms:>10.3f}")
            print(f"\n  Effective tok/s: {1000/mean_ms:.1f}")
        else:
            pct = mean_ms / total_mean * 100
            print(f"{key:<20} {mean_ms:>10.2f} {pct:>9.1f}% {min_ms:>10.3f} {max_ms:>10.3f}")

    # Summary
    ffn_total = sum(np.mean(timings[f"ffn_{ci}"]) for ci in range(4)) * 1000
    non_ffn = total_mean - ffn_total
    lmhead_ms = np.mean(timings["lmhead"]) * 1000
    embed_ms = np.mean(timings["embed"]) * 1000

    print(f"\n--- Summary ---")
    print(f"  FFN chunks total: {ffn_total:.1f}ms ({ffn_total/total_mean*100:.1f}%)")
    print(f"  LM head:          {lmhead_ms:.1f}ms ({lmhead_ms/total_mean*100:.1f}%)")
    print(f"  Embed:            {embed_ms:.1f}ms ({embed_ms/total_mean*100:.1f}%)")
    print(f"  Other overhead:   {non_ffn - lmhead_ms - embed_ms:.1f}ms ({(non_ffn - lmhead_ms - embed_ms)/total_mean*100:.1f}%)")
    print(f"\n  Potential speedup if lmhead skipped during prefill:")
    without_lm = total_mean - lmhead_ms
    print(f"    {1000/without_lm:.1f} tok/s (was {1000/total_mean:.1f} tok/s, "
          f"{total_mean/without_lm:.2f}x faster)")


if __name__ == "__main__":
    main()
