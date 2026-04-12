#!/usr/bin/env python3
"""
End-to-end latency comparison: 9-chunk baseline vs F-isolated 16-chunk layout.

Measures REAL decode and prefill latency with CoreML models on ANE.

Usage:
  python tests/dev/f_isolated_latency_comparison.py                       # full run
  python tests/dev/f_isolated_latency_comparison.py --skip-export         # skip export, measure only
  python tests/dev/f_isolated_latency_comparison.py --export-only         # export only, no measurement
  python tests/dev/f_isolated_latency_comparison.py --skip-existing       # skip already-exported chunks
"""

import argparse
import gc
import os
import resource
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = "/Volumes/MySSD/Anemll"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
os.chdir(REPO_ROOT)

import numpy as np
import torch

torch.set_grad_enabled(False)

import coremltools as ct

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

# ─── Paths ────────────────────────────────────────────────────────────────────
HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
BASELINE_DIR = os.path.join(
    REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp32", "combined_LUT4_dedup"
)
MODEL_DIR = os.path.join(REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp32")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "f_isolated_experiment")
COMPUTE_UNIT = ct.ComputeUnit.CPU_AND_NE

# ─── F-isolated 16-chunk layout ──────────────────────────────────────────────
# Every FLLL chunk is split into F (1 linear-attn layer) + LLL (3 full-attn layers)
# Chunk 0 (LLL) and chunk 15 (F) are structurally identical to baseline 0 and 8.
F_ISOLATED_RANGES = [
    (0, 3),    # chunk  0: layers 0-2   (LLL) — same as baseline chunk 0
    (3, 4),    # chunk  1: layer  3     (F)
    (4, 7),    # chunk  2: layers 4-6   (LLL)
    (7, 8),    # chunk  3: layer  7     (F)
    (8, 11),   # chunk  4: layers 8-10  (LLL)
    (11, 12),  # chunk  5: layer  11    (F)
    (12, 15),  # chunk  6: layers 12-14 (LLL)
    (15, 16),  # chunk  7: layer  15    (F)
    (16, 19),  # chunk  8: layers 16-18 (LLL)
    (19, 20),  # chunk  9: layer  19    (F)
    (20, 23),  # chunk 10: layers 20-22 (LLL)
    (23, 24),  # chunk 11: layer  23    (F)
    (24, 27),  # chunk 12: layers 24-26 (LLL)
    (27, 28),  # chunk 13: layer  27    (F)
    (28, 31),  # chunk 14: layers 28-30 (LLL)
    (31, 32),  # chunk 15: layer  31    (F)
]
F_ISOLATED_NUM = len(F_ISOLATED_RANGES)  # 16

# ─── Measurement config ──────────────────────────────────────────────────────
WARMUP_TOKENS = 5     # warmup steps before measuring
PROMPT_TOKENS = 20    # prompt length for prefill measurement
DECODE_TOKENS = 40    # tokens to generate for decode measurement
NUM_RUNS = 3          # repeat measurements

# ── Prompt for tokenization ──
TEST_PROMPT = "Explain the concept of neural architecture search in simple terms."


# =============================================================================
#  Phase 1: Export F-isolated chunks
# =============================================================================

def export_f_isolated_chunks(skip_existing=False):
    """Export all 16 F-isolated chunks using the standard converter."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
    from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

    # Check which chunks need exporting
    to_export = []
    for ci, (start, end) in enumerate(F_ISOLATED_RANGES):
        path = os.path.join(ARTIFACT_DIR, f"chunk{ci}.mlpackage")
        if skip_existing and os.path.exists(path):
            print(f"  [skip] chunk {ci}: layers {start}-{end - 1} (exists)")
        else:
            to_export.append((ci, start, end))

    if not to_export:
        print("All F-isolated chunks already exported.")
        return

    print(f"\nLoading model weights for export ({len(to_export)} chunks to export)...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    for ci, start, end in to_export:
        nlayers = end - start
        ltype = "F" if nlayers == 1 and (start in [3, 7, 11, 15, 19, 23, 27, 31]) else "LLL"
        print(f"\n  Exporting chunk {ci}: layers {start}-{end - 1} ({ltype}, {nlayers} layers)...")
        t0 = time.time()

        conv = Qwen35Converter(
            model,
            context_length=CTX,
            batch_size=BATCH_SIZE,
            num_chunks=F_ISOLATED_NUM,
            lut_bits=4,
            per_channel=4,
            compute_precision="float32",
        )
        ml = conv.convert_part_2(
            model,
            chunk_idx=ci,
            total_chunks=F_ISOLATED_NUM,
            override_start_layer=start,
            override_end_layer=end,
        )
        out_path = os.path.join(ARTIFACT_DIR, f"chunk{ci}.mlpackage")
        ml.save(out_path)
        del ml, conv
        gc.collect()
        print(f"    Saved chunk{ci}.mlpackage in {time.time() - t0:.1f}s")

    del model
    gc.collect()
    print("\nAll F-isolated chunks exported.")


# =============================================================================
#  Inference Engine (generic for any chunk layout)
# =============================================================================

class ChunkEngine:
    """Loads and runs inference for an arbitrary chunk layout."""

    def __init__(self, name, chunk_paths, embed_path, lmhead_path,
                 compute_unit, num_chunks, is_multifunction=False):
        self.name = name
        self.num_chunks = num_chunks
        self.is_multifunction = is_multifunction

        print(f"\n  Loading {name} engine ({num_chunks} chunks)...")
        t0 = time.time()

        # Embed + LMHead
        self.embed = ct.models.MLModel(embed_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.lmhead = ct.models.MLModel(lmhead_path, compute_units=ct.ComputeUnit.CPU_ONLY)

        # FFN chunks
        self.ffns = []
        for ci in range(num_chunks):
            if is_multifunction:
                m = ct.models.MLModel(
                    chunk_paths[ci], compute_units=compute_unit,
                    function_name="infer"
                )
            else:
                m = ct.models.MLModel(chunk_paths[ci], compute_units=compute_unit)
            self.ffns.append(m)

        # Detect input shapes per chunk
        self.inp_maps = []
        for ci in range(num_chunks):
            spec = self.ffns[ci].get_spec()
            imap = {}
            if is_multifunction:
                fn_inputs = None
                for fn in spec.description.functions:
                    if fn.name == "infer":
                        fn_inputs = fn.input
                        break
                if fn_inputs is None:
                    fn_inputs = spec.description.input
            else:
                fn_inputs = spec.description.input
            for inp in fn_inputs:
                try:
                    imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                except Exception:
                    pass
            self.inp_maps.append(imap)

        # Per-chunk linear state detection
        self.has_linear_per_chunk = [
            "linear_conv_state" in self.inp_maps[ci] for ci in range(num_chunks)
        ]

        self.reset_all()
        print(f"    Loaded in {time.time() - t0:.1f}s")

    def reset_all(self):
        """Reset all KV cache states and linear states."""
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = []
        self.lin_recs = []
        for ci in range(self.num_chunks):
            if self.has_linear_per_chunk[ci]:
                self.lin_convs.append(
                    np.zeros(self.inp_maps[ci]["linear_conv_state"], dtype=np.float16)
                )
                self.lin_recs.append(
                    np.zeros(self.inp_maps[ci]["linear_recurrent_state"], dtype=np.float16)
                )
            else:
                self.lin_convs.append(None)
                self.lin_recs.append(None)

    def _step(self, tok_id, pos):
        """Single-token step through all chunks."""
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, : pos + 1] = 0

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

            if "linear_conv_state_out" in out:
                self.lin_convs[ci] = out["linear_conv_state_out"]
                self.lin_recs[ci] = out["linear_recurrent_state_out"]

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        logits = list(lm_out.values())[0]
        return int(np.argmax(logits.reshape(-1)))

    def _step_timed(self, tok_id, pos):
        """Single-token step with per-component timing."""
        tok = np.array([[tok_id]], dtype=np.int32)

        # Embed
        t_embed_start = time.perf_counter()
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]
        t_embed = time.perf_counter() - t_embed_start

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, : pos + 1] = 0

        # Chunks
        chunk_times = []
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

            t_chunk_start = time.perf_counter()
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            chunk_times.append(time.perf_counter() - t_chunk_start)

            hidden = out["output_hidden_states"]
            if "linear_conv_state_out" in out:
                self.lin_convs[ci] = out["linear_conv_state_out"]
                self.lin_recs[ci] = out["linear_recurrent_state_out"]

        # LMHead
        t_lm_start = time.perf_counter()
        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        t_lmhead = time.perf_counter() - t_lm_start

        logits = list(lm_out.values())[0]
        tok_out = int(np.argmax(logits.reshape(-1)))
        return tok_out, t_embed, chunk_times, t_lmhead

    def generate(self, prompt_tokens, max_decode):
        """Full prefill + decode. Returns (tokens, prefill_ms, decode_ms, per_step_details)."""
        self.reset_all()

        # Warmup
        for i in range(min(WARMUP_TOKENS, len(prompt_tokens))):
            self._step(prompt_tokens[i], i)

        # Re-reset for clean measurement
        self.reset_all()

        # Prefill
        t_prefill_start = time.perf_counter()
        cpu_prefill_start = time.process_time()
        for i, tid in enumerate(prompt_tokens):
            last_next = self._step(tid, i)
        t_prefill = (time.perf_counter() - t_prefill_start) * 1000
        cpu_prefill = (time.process_time() - cpu_prefill_start) * 1000

        prefill_end_pos = len(prompt_tokens)

        # Decode with per-step timing
        decode_tokens = [last_next]
        step_details = []
        t_decode_start = time.perf_counter()
        cpu_decode_start = time.process_time()

        for gi in range(max_decode - 1):
            pos = prefill_end_pos + gi
            if pos >= CTX - 1:
                break
            tok_out, t_embed, chunk_times, t_lmhead = self._step_timed(
                decode_tokens[-1], pos
            )
            step_details.append({
                "embed_ms": t_embed * 1000,
                "chunk_ms": [t * 1000 for t in chunk_times],
                "lmhead_ms": t_lmhead * 1000,
                "total_ms": (t_embed + sum(chunk_times) + t_lmhead) * 1000,
            })
            decode_tokens.append(tok_out)

        t_decode = (time.perf_counter() - t_decode_start) * 1000
        cpu_decode = (time.process_time() - cpu_decode_start) * 1000

        return {
            "tokens": decode_tokens,
            "prefill_ms": t_prefill,
            "prefill_cpu_ms": cpu_prefill,
            "decode_ms": t_decode,
            "decode_cpu_ms": cpu_decode,
            "num_prompt": len(prompt_tokens),
            "num_decode": len(decode_tokens),
            "step_details": step_details,
        }

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        del self.ffns
        gc.collect()


# =============================================================================
#  Phase 2: Load engines
# =============================================================================

def load_baseline_engine():
    """Load the 9-chunk baseline from combined dedup models."""
    chunk_paths = [
        os.path.join(BASELINE_DIR, f"chunk{ci}.mlpackage")
        for ci in range(NUM_CHUNKS)
    ]
    embed_path = os.path.join(MODEL_DIR, "embed_single.mlpackage")
    lmhead_path = os.path.join(MODEL_DIR, "lm_head_nosplit.mlpackage")

    return ChunkEngine(
        name="Baseline (9-chunk)",
        chunk_paths=chunk_paths,
        embed_path=embed_path,
        lmhead_path=lmhead_path,
        compute_unit=COMPUTE_UNIT,
        num_chunks=NUM_CHUNKS,
        is_multifunction=True,
    )


def load_f_isolated_engine():
    """Load the 16-chunk F-isolated variant from exported models."""
    chunk_paths = [
        os.path.join(ARTIFACT_DIR, f"chunk{ci}.mlpackage")
        for ci in range(F_ISOLATED_NUM)
    ]
    embed_path = os.path.join(MODEL_DIR, "embed_single.mlpackage")
    lmhead_path = os.path.join(MODEL_DIR, "lm_head_nosplit.mlpackage")

    return ChunkEngine(
        name="F-isolated (16-chunk)",
        chunk_paths=chunk_paths,
        embed_path=embed_path,
        lmhead_path=lmhead_path,
        compute_unit=COMPUTE_UNIT,
        num_chunks=F_ISOLATED_NUM,
        is_multifunction=False,
    )


# =============================================================================
#  Phase 3: Measurement
# =============================================================================

def get_prompt_tokens(prompt_tokens_count=PROMPT_TOKENS):
    """Tokenize the test prompt."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    chat = [{"role": "user", "content": TEST_PROMPT}]
    ids = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True)
    token_ids = ids[0].tolist()
    # Trim or pad to prompt_tokens_count
    if len(token_ids) > prompt_tokens_count:
        token_ids = token_ids[:prompt_tokens_count]
    return token_ids, tokenizer


def run_measurements(engine, prompt_tokens, num_runs=NUM_RUNS, decode_tokens=DECODE_TOKENS):
    """Run multiple measurement rounds, return aggregated results."""
    all_results = []
    for run_idx in range(num_runs):
        print(f"    Run {run_idx + 1}/{num_runs}...", end="", flush=True)
        result = engine.generate(prompt_tokens, decode_tokens)
        all_results.append(result)
        print(f" decode={result['decode_ms']:.0f}ms "
              f"({result['num_decode']} tok), "
              f"prefill={result['prefill_ms']:.0f}ms")

    # Aggregate
    prefill_times = [r["prefill_ms"] for r in all_results]
    decode_times = [r["decode_ms"] for r in all_results]
    prefill_cpu = [r["prefill_cpu_ms"] for r in all_results]
    decode_cpu = [r["decode_cpu_ms"] for r in all_results]
    num_decode = [r["num_decode"] for r in all_results]

    # Use median for best stability
    import statistics
    median_idx = sorted(range(len(decode_times)),
                        key=lambda i: decode_times[i])[len(decode_times) // 2]
    best_result = all_results[median_idx]

    # Per-step breakdown from median run
    step_details = best_result["step_details"]
    if step_details:
        avg_embed = statistics.mean(s["embed_ms"] for s in step_details)
        avg_lmhead = statistics.mean(s["lmhead_ms"] for s in step_details)
        avg_total = statistics.mean(s["total_ms"] for s in step_details)
        # Per-chunk breakdown
        n_chunks = len(step_details[0]["chunk_ms"])
        avg_per_chunk = []
        for ci in range(n_chunks):
            avg_per_chunk.append(
                statistics.mean(s["chunk_ms"][ci] for s in step_details)
            )
    else:
        avg_embed = avg_lmhead = avg_total = 0
        avg_per_chunk = []

    return {
        "prefill_ms_median": statistics.median(prefill_times),
        "prefill_ms_all": prefill_times,
        "prefill_cpu_ms_median": statistics.median(prefill_cpu),
        "decode_ms_median": statistics.median(decode_times),
        "decode_ms_all": decode_times,
        "decode_cpu_ms_median": statistics.median(decode_cpu),
        "num_prompt": best_result["num_prompt"],
        "num_decode_median": int(statistics.median(num_decode)),
        "tokens": best_result["tokens"],
        # Per-step breakdown
        "avg_step_total_ms": avg_total,
        "avg_embed_ms": avg_embed,
        "avg_lmhead_ms": avg_lmhead,
        "avg_per_chunk_ms": avg_per_chunk,
        "step_details": step_details,
    }


# =============================================================================
#  Phase 4: Report
# =============================================================================

def print_report(baseline_results, f_isolated_results, baseline_chunks, f_isolated_chunks):
    """Print comprehensive comparison report."""
    b = baseline_results
    f = f_isolated_results

    # Derived metrics
    b_decode_toks_per_s = (b["num_decode_median"] / b["decode_ms_median"]) * 1000 if b["decode_ms_median"] > 0 else 0
    f_decode_toks_per_s = (f["num_decode_median"] / f["decode_ms_median"]) * 1000 if f["decode_ms_median"] > 0 else 0
    b_prefill_toks_per_s = (b["num_prompt"] / b["prefill_ms_median"]) * 1000 if b["prefill_ms_median"] > 0 else 0
    f_prefill_toks_per_s = (f["num_prompt"] / f["prefill_ms_median"]) * 1000 if f["prefill_ms_median"] > 0 else 0

    b_decode_per_tok = b["decode_ms_median"] / b["num_decode_median"] if b["num_decode_median"] > 0 else 0
    f_decode_per_tok = f["decode_ms_median"] / f["num_decode_median"] if f["num_decode_median"] > 0 else 0

    b_cpu_frac_decode = (b["decode_cpu_ms_median"] / b["decode_ms_median"]) if b["decode_ms_median"] > 0 else 1
    f_cpu_frac_decode = (f["decode_cpu_ms_median"] / f["decode_ms_median"]) if f["decode_ms_median"] > 0 else 1
    b_ane_frac_decode = max(0, 1 - b_cpu_frac_decode)
    f_ane_frac_decode = max(0, 1 - f_cpu_frac_decode)

    b_cpu_frac_prefill = (b["prefill_cpu_ms_median"] / b["prefill_ms_median"]) if b["prefill_ms_median"] > 0 else 1
    f_cpu_frac_prefill = (f["prefill_cpu_ms_median"] / f["prefill_ms_median"]) if f["prefill_ms_median"] > 0 else 1

    print("\n" + "=" * 80)
    print("  END-TO-END LATENCY COMPARISON: 9-CHUNK vs F-ISOLATED 16-CHUNK")
    print("=" * 80)

    # ── Main comparison table ──
    print("\n┌─────────────────────────────┬──────────────────┬──────────────────┬──────────┐")
    print("│ Metric                      │ Baseline (9-ch)  │ F-isolated (16)  │ Δ        │")
    print("├─────────────────────────────┼──────────────────┼──────────────────┼──────────┤")

    def row(label, bval, fval, unit="", higher_better=False):
        if isinstance(bval, float):
            bs = f"{bval:.1f}{unit}"
            fs = f"{fval:.1f}{unit}"
            if bval > 0:
                delta_pct = ((fval - bval) / bval) * 100
                if higher_better:
                    delta_pct = -delta_pct  # flip sign so positive = good
                ds = f"{delta_pct:+.1f}%"
            else:
                ds = "N/A"
        else:
            bs = f"{bval}{unit}"
            fs = f"{fval}{unit}"
            ds = ""
        print(f"│ {label:<27s} │ {bs:>16s} │ {fs:>16s} │ {ds:>8s} │")

    row("Chunk count", 9, 16, "")
    row("Decode latency (ms)", b["decode_ms_median"], f["decode_ms_median"], " ms")
    row("Decode tok/s", b_decode_toks_per_s, f_decode_toks_per_s, "", higher_better=True)
    row("Decode per-token (ms)", b_decode_per_tok, f_decode_per_tok, " ms")
    row("Decode tokens generated", float(b["num_decode_median"]), float(f["num_decode_median"]), "")
    row("Prefill latency (ms)", b["prefill_ms_median"], f["prefill_ms_median"], " ms")
    row("Prefill tok/s", b_prefill_toks_per_s, f_prefill_toks_per_s, "", higher_better=True)
    row("Decode CPU fraction", b_cpu_frac_decode, f_cpu_frac_decode, "")
    row("Decode ANE fraction (est)", b_ane_frac_decode, f_ane_frac_decode, "")
    row("Prefill CPU fraction", b_cpu_frac_prefill, f_cpu_frac_prefill, "")

    print("└─────────────────────────────┴──────────────────┴──────────────────┴──────────┘")

    # ── Per-step breakdown ──
    print("\n── Per-step breakdown (median run) ──")
    print(f"  {'Component':<16s}  {'Baseline':>10s}  {'F-isolated':>10s}  {'Δ':>8s}")
    print(f"  {'─' * 16}  {'─' * 10}  {'─' * 10}  {'─' * 8}")

    def step_row(label, bval, fval):
        if bval > 0:
            d = f"{((fval - bval) / bval) * 100:+.1f}%"
        else:
            d = "N/A"
        print(f"  {label:<16s}  {bval:>9.2f}ms  {fval:>9.2f}ms  {d:>8s}")

    step_row("Embed", b["avg_embed_ms"], f["avg_embed_ms"])
    step_row("All chunks", sum(b["avg_per_chunk_ms"]), sum(f["avg_per_chunk_ms"]))
    step_row("LM Head", b["avg_lmhead_ms"], f["avg_lmhead_ms"])
    step_row("Total step", b["avg_step_total_ms"], f["avg_step_total_ms"])

    # ── Per-chunk latency breakdown ──
    print("\n── Baseline per-chunk decode latency (median run) ──")
    for ci, t_ms in enumerate(b["avg_per_chunk_ms"]):
        s, e = CHUNK_RANGES[ci]
        nlayers = e - s
        ltype = "LLL" if ci == 0 else ("F" if ci == 8 else "FLLL")
        print(f"  chunk {ci:2d} [{s:2d}-{e - 1:2d}] ({ltype:4s}, {nlayers}L): {t_ms:>8.2f} ms")
    print(f"  {'Total':>34s}: {sum(b['avg_per_chunk_ms']):>8.2f} ms")

    print("\n── F-isolated per-chunk decode latency (median run) ──")
    for ci, t_ms in enumerate(f["avg_per_chunk_ms"]):
        s, e = F_ISOLATED_RANGES[ci]
        nlayers = e - s
        ltype = "F" if nlayers == 1 and s in [3, 7, 11, 15, 19, 23, 27, 31] else "LLL"
        print(f"  chunk {ci:2d} [{s:2d}-{e - 1:2d}] ({ltype:3s}, {nlayers}L): {t_ms:>8.2f} ms")
    print(f"  {'Total':>34s}: {sum(f['avg_per_chunk_ms']):>8.2f} ms")

    # ── Group F vs L chunks in F-isolated ──
    f_chunk_total = 0.0
    l_chunk_total = 0.0
    f_chunk_count = 0
    l_chunk_count = 0
    for ci, t_ms in enumerate(f["avg_per_chunk_ms"]):
        s, e = F_ISOLATED_RANGES[ci]
        nlayers = e - s
        is_f = nlayers == 1 and s in [3, 7, 11, 15, 19, 23, 27, 31]
        if is_f:
            f_chunk_total += t_ms
            f_chunk_count += 1
        else:
            l_chunk_total += t_ms
            l_chunk_count += 1

    print(f"\n  F-chunks total ({f_chunk_count} chunks): {f_chunk_total:.2f} ms")
    print(f"  L-chunks total ({l_chunk_count} chunks): {l_chunk_total:.2f} ms")
    if f_chunk_count > 0:
        print(f"  Avg F-chunk: {f_chunk_total / f_chunk_count:.2f} ms")
    if l_chunk_count > 0:
        print(f"  Avg L-chunk: {l_chunk_total / l_chunk_count:.2f} ms")

    # ── Correctness check ──
    print("\n── Correctness check ──")
    b_toks = b["tokens"]
    f_toks = f["tokens"]
    min_len = min(len(b_toks), len(f_toks))
    matches = sum(1 for i in range(min_len) if b_toks[i] == f_toks[i])
    print(f"  Token match: {matches}/{min_len} ({100 * matches / min_len:.0f}%)")
    if matches < min_len:
        first_diverge = next(i for i in range(min_len) if b_toks[i] != f_toks[i])
        print(f"  First divergence at token {first_diverge}: baseline={b_toks[first_diverge]}, f-isolated={f_toks[first_diverge]}")

    # ── State boundary analysis ──
    print("\n── State boundary analysis ──")
    print(f"  Baseline: {NUM_CHUNKS} chunk boundaries per token")
    print(f"  F-isolated: {F_ISOLATED_NUM} chunk boundaries per token")
    extra_boundaries = F_ISOLATED_NUM - NUM_CHUNKS
    overhead_per_tok = f_decode_per_tok - b_decode_per_tok if b_decode_per_tok > 0 else 0
    print(f"  Extra boundaries: {extra_boundaries} per token")
    if overhead_per_tok != 0:
        per_boundary = overhead_per_tok / extra_boundaries if extra_boundaries > 0 else 0
        print(f"  Overhead per extra boundary: {per_boundary:.2f} ms")

    # ── All run times ──
    print("\n── All run times ──")
    print(f"  Baseline decode:    {[f'{t:.0f}' for t in b['decode_ms_all']]} ms")
    print(f"  F-isolated decode:  {[f'{t:.0f}' for t in f['decode_ms_all']]} ms")
    print(f"  Baseline prefill:   {[f'{t:.0f}' for t in b['prefill_ms_all']]} ms")
    print(f"  F-isolated prefill: {[f'{t:.0f}' for t in f['prefill_ms_all']]} ms")

    # ── Final recommendation ──
    decode_change = ((f["decode_ms_median"] - b["decode_ms_median"]) / b["decode_ms_median"]) * 100
    prefill_change = ((f["prefill_ms_median"] - b["prefill_ms_median"]) / b["prefill_ms_median"]) * 100

    print("\n" + "=" * 80)
    print("  RECOMMENDATION")
    print("=" * 80)
    print(f"\n  Decode latency change: {decode_change:+.1f}%")
    print(f"  Prefill latency change: {prefill_change:+.1f}%")
    print(f"  ANE utilization change: {b_ane_frac_decode:.0%} → {f_ane_frac_decode:.0%}")

    if decode_change < -5:
        print("\n  ✅ F-isolated layout IMPROVES decode latency significantly.")
        print("     Recommendation: ADOPT the F-isolated layout.")
    elif decode_change < 5:
        if f_ane_frac_decode > b_ane_frac_decode + 0.1:
            print("\n  ⚡ F-isolated layout is latency-neutral but improves ANE utilization.")
            print("     Recommendation: ADOPT if ANE offload (CPU savings) matters.")
        else:
            print("\n  ≈ F-isolated layout is roughly equivalent in latency.")
            print("     Recommendation: KEEP current 9-chunk layout (simpler).")
    else:
        print(f"\n  ❌ F-isolated layout HURTS decode latency by {decode_change:.1f}%.")
        if f_ane_frac_decode > b_ane_frac_decode + 0.2:
            print("     However, ANE utilization improved significantly.")
            print("     Recommendation: Consider partial split (only worst chunks).")
        else:
            print("     Recommendation: KEEP current 9-chunk layout.")

    print()


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="9-chunk vs F-isolated 16-chunk latency comparison"
    )
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export phase (use existing models)")
    parser.add_argument("--export-only", action="store_true",
                        help="Export only, no measurement")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip already-exported chunks")
    parser.add_argument("--tokens", type=int, default=DECODE_TOKENS,
                        help=f"Decode tokens to generate (default: {DECODE_TOKENS})")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help=f"Number of measurement runs (default: {NUM_RUNS})")
    parser.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS,
                        help=f"Prompt length for prefill (default: {PROMPT_TOKENS})")
    args = parser.parse_args()

    decode_tokens = args.tokens
    num_runs = args.runs
    prompt_tokens_count = args.prompt_tokens

    print("=" * 70)
    print("  F-ISOLATED LATENCY COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"  Baseline: 9-chunk layout ({BASELINE_DIR})")
    print(f"  Variant:  16-chunk F-isolated ({ARTIFACT_DIR})")
    print(f"  Settings: CTX={CTX}, BATCH_SIZE={BATCH_SIZE}, CU=CPU_AND_NE")
    print(f"  Measure:  {prompt_tokens_count} prompt tokens, {decode_tokens} decode tokens, {num_runs} runs")

    # Phase 1: Export
    if not args.skip_export:
        print("\n" + "─" * 70)
        print("  PHASE 1: Export F-isolated chunks")
        print("─" * 70)
        export_f_isolated_chunks(skip_existing=args.skip_existing)
    else:
        print("\n  [skip-export] Using existing F-isolated models")

    if args.export_only:
        print("\n  [export-only] Done.")
        return

    # Phase 2: Load engines
    print("\n" + "─" * 70)
    print("  PHASE 2: Load inference engines")
    print("─" * 70)

    # Get prompt tokens
    prompt_tokens, tokenizer = get_prompt_tokens(prompt_tokens_count)
    print(f"\n  Prompt tokens: {len(prompt_tokens)}")

    baseline_engine = load_baseline_engine()
    f_isolated_engine = load_f_isolated_engine()

    # Phase 3: Measure
    print("\n" + "─" * 70)
    print("  PHASE 3: Latency measurements")
    print("─" * 70)

    print(f"\n  Measuring Baseline (9-chunk)...")
    baseline_results = run_measurements(baseline_engine, prompt_tokens,
                                        num_runs=num_runs, decode_tokens=decode_tokens)

    print(f"\n  Measuring F-isolated (16-chunk)...")
    f_isolated_results = run_measurements(f_isolated_engine, prompt_tokens,
                                          num_runs=num_runs, decode_tokens=decode_tokens)

    # Phase 4: Report
    print_report(baseline_results, f_isolated_results, CHUNK_RANGES, F_ISOLATED_RANGES)

    # Cleanup
    baseline_engine.cleanup()
    f_isolated_engine.cleanup()


if __name__ == "__main__":
    main()
