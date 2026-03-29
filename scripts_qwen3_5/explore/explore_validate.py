#!/usr/bin/env python3
"""Qwen3.5-4B — Exploratory validation: compare new configs against stable baseline.

Runs multi-round conversation on both stable and experimental configs,
comparing token-for-token correctness, latency, and memory.

Usage:
    # Validate a specific config against stable baseline:
    python scripts_qwen3_5/explore/explore_validate.py --config batch512_ctx1024

    # Validate all configs:
    python scripts_qwen3_5/explore/explore_validate.py --config all

    # More tokens per turn:
    python scripts_qwen3_5/explore/explore_validate.py --config batch256_ctx2048 --tokens 60
"""
import sys, os, gc, time, json, argparse, traceback
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts_qwen3_5.explore.explore_config import CONFIGS, get_config, STABLE, HF_MODEL

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
    if path.endswith(".mlmodelc") and function_name is None:
        return ct.models.CompiledMLModel(path, compute_unit)
    # For .mlmodelc with function_name, fall back to .mlpackage
    if path.endswith(".mlmodelc") and function_name is not None:
        pkg = path.replace(".mlmodelc", ".mlpackage")
        if os.path.exists(pkg):
            path = pkg
        else:
            # Try without function_name on compiled
            return ct.models.CompiledMLModel(path, compute_unit)
    kwargs = {"compute_units": compute_unit}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def _find_model(base_dir, name):
    """Find model path. Prefer .mlmodelc for non-combined models (faster loading)."""
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
    """CoreML inference engine for a Qwen3.5 config."""

    def __init__(self, model_dir, ctx, num_chunks, compute_unit, skip_prefill=False, lut_bits=6):
        cu = compute_unit
        self.ctx = ctx
        self.num_chunks = num_chunks
        ffn_label = f"LUT{lut_bits}"

        # Loading order: embed → lm_head → FFN (infer only when skip_prefill)
        print(f"    Loading embeddings...", flush=True)
        t0 = time.time()
        self.embed = _load_model(_find_model(model_dir, "embeddings"), cu)
        print(f"      {time.time()-t0:.0f}s", flush=True)

        print(f"    Loading lm_head...", flush=True)
        t0 = time.time()
        # Try lm_head_logits first, fall back to lm_head
        try:
            self.lmhead = _load_model(_find_model(model_dir, "lm_head_logits"), cu)
        except FileNotFoundError:
            self.lmhead = _load_model(_find_model(model_dir, "lm_head"), cu)
        print(f"      {time.time()-t0:.0f}s", flush=True)

        combined_dir = os.path.join(model_dir, f"combined_{ffn_label}_dedup")
        use_combined = os.path.isdir(combined_dir)

        # Check if combined dir has .mlpackage (needed for function_name)
        if use_combined:
            test_path = os.path.join(combined_dir, "chunk0.mlpackage")
            if not os.path.exists(test_path):
                # Only .mlmodelc available - can't use function_name, fall back
                print(f"    Combined dir has no .mlpackage, using separate models", flush=True)
                use_combined = False

        self.ffns = []
        self.prefills = []
        for ci in range(num_chunks):
            if use_combined:
                path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
                t0 = time.time()
                m_infer = _load_model(path, cu, function_name="infer")
                print(f"      chunk {ci} infer {time.time()-t0:.0f}s", flush=True)
                if skip_prefill:
                    m_prefill = None
                else:
                    t0 = time.time()
                    m_prefill = _load_model(path, cu, function_name="prefill")
                    print(f"      chunk {ci} prefill {time.time()-t0:.0f}s", flush=True)
            else:
                path = _find_model(model_dir, f"ffn_{ffn_label}_chunk{ci}")
                t0 = time.time()
                m_infer = _load_model(path, cu)
                print(f"      chunk {ci} (separate) {time.time()-t0:.0f}s", flush=True)
                m_prefill = None
            self.ffns.append(m_infer)
            self.prefills.append(m_prefill)

        self.inp_map = _detect_shapes(self.ffns[0], use_combined)

        # Pre-allocate buffers
        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)

        self.reset()

    def reset(self):
        self.states = [m.make_state() for m in self.ffns]
        if 'linear_conv_state' in self.inp_map:
            self.lin_convs = [np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
                              for _ in range(self.num_chunks)]
            self.lin_recs = [np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
                             for _ in range(self.num_chunks)]
        else:
            self.lin_convs = [None] * self.num_chunks
            self.lin_recs = [None] * self.num_chunks

    def _step(self, tok_id, pos):
        self._tok_buf[0, 0] = tok_id
        hidden = list(self.embed.predict({"input_ids": self._tok_buf}).values())[0]

        self._mask_buf[:, :, :, :] = -65504.0
        self._mask_buf[:, :, :, :pos + 1] = 0
        self._pos_buf[0] = pos

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": self._pos_buf,
                "causal_mask": self._mask_buf,
                "current_pos": self._pos_buf,
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
        if "logits" in lm_out or "output_logits" in lm_out:
            key = "output_logits" if "output_logits" in lm_out else "logits"
            return int(np.argmax(lm_out[key].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    def generate(self, token_ids, start_pos, max_gen, stop_ids):
        for i, tid in enumerate(token_ids):
            pos = start_pos + i
            if pos >= self.ctx:
                break
            last_next = self._step(tid, pos)
        prefill_end = start_pos + len(token_ids)
        tokens = [last_next]
        for gi in range(max_gen - 1):
            pos = prefill_end + gi
            if pos >= self.ctx - 1:
                break
            next_id = self._step(tokens[-1], pos)
            tokens.append(next_id)
            if next_id in stop_ids:
                break
        return tokens

    def cleanup(self):
        del self.embed, self.lmhead
        for m in self.ffns:
            del m
        for m in self.prefills:
            if m is not None:
                del m
        gc.collect()


def run_conversation(engine, tokenizer, stop_ids, turns, max_gen, ctx):
    """Run multi-round conversation and return results."""
    conversation = []
    results = []
    for ti, user_msg in enumerate(turns):
        conversation.append({"role": "user", "content": user_msg})
        input_ids = _ensure_ids(tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=True))
        prompt_len = input_ids.shape[1]
        if prompt_len + max_gen > ctx:
            input_ids = input_ids[:, -(ctx - max_gen):]
            prompt_len = input_ids.shape[1]
        engine.reset()
        t0 = time.time()
        gen_tokens = engine.generate(input_ids[0].tolist(), 0, max_gen, stop_ids)
        elapsed = time.time() - t0
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": "<think>\n" + raw_text})
        results.append({
            'turn': ti + 1, 'prompt_len': prompt_len,
            'tokens': gen_tokens, 'text': raw_text,
            'elapsed': round(elapsed, 2),
            'tok_per_sec': round(len(gen_tokens) / elapsed, 1) if elapsed > 0 else 0,
        })
    return results


def compare_results(baseline_results, exp_results, baseline_name, exp_name):
    """Compare two sets of conversation results."""
    report = {"baseline": baseline_name, "experiment": exp_name, "turns": []}
    all_match = True
    for br, er in zip(baseline_results, exp_results):
        bt = br['tokens'][:len(er['tokens'])]
        et = er['tokens'][:len(bt)]
        match = sum(1 for a, b in zip(bt, et) if a == b)
        total = max(len(bt), len(et))
        pct = match / total * 100 if total > 0 else 100
        first_diff = next((i for i, (a, b) in enumerate(zip(bt, et)) if a != b), None)
        turn_report = {
            "turn": br['turn'],
            "baseline_prompt": br['prompt_len'],
            "exp_prompt": er['prompt_len'],
            "match_pct": round(pct, 1),
            "first_diff_pos": first_diff,
            "baseline_elapsed": br['elapsed'],
            "exp_elapsed": er['elapsed'],
            "baseline_tok_s": br['tok_per_sec'],
            "exp_tok_s": er['tok_per_sec'],
        }
        if pct < 100:
            all_match = False
        report["turns"].append(turn_report)
    report["all_match"] = all_match
    return report


def _run_single_engine(model_dir, ctx, num_chunks, compute_unit, tokenizer,
                       stop_ids, max_gen, label):
    """Load one engine, run conversation, cleanup, return results dict.

    Runs in a subprocess to guarantee full memory release on macOS/ANE.
    """
    import subprocess as sp, tempfile, pickle

    # Serialize args and run in a child process to avoid ANE memory leaks
    script = f'''
import sys, os, gc, time, json, pickle
sys.path.insert(0, {repr(_REPO_ROOT)})
os.environ["QWEN35_HF_MODEL"] = os.environ.get("QWEN35_HF_MODEL", "")

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

# Re-import everything we need
from scripts_qwen3_5.explore.explore_validate import (
    Engine, run_conversation, _build_stop_ids, _ensure_ids, CONVERSATION_TURNS
)

cu = ct.ComputeUnit.CPU_AND_NE
tokenizer = AutoTokenizer.from_pretrained({repr(str(tokenizer.name_or_path))}, use_fast=False)
stop_ids = _build_stop_ids(tokenizer)

print(f"  Loading {{'{label}'}} (CTX={ctx})...", flush=True)
t0 = time.time()
engine = Engine({repr(model_dir)}, {ctx}, {num_chunks}, cu, skip_prefill=True)
load_time = time.time() - t0
print(f"    Loaded in {{load_time:.0f}}s", flush=True)

print(f"  Running {{'{label}'}} conversation ({max_gen} tok/turn)...", flush=True)
results = run_conversation(engine, tokenizer, stop_ids,
                           CONVERSATION_TURNS, {max_gen}, {ctx})
for r in results:
    print(f"    Turn {{r['turn']}}: [{{r['prompt_len']}} tok, {{r['elapsed']:.1f}}s] {{r['text'][:80]}}", flush=True)

engine.cleanup()
del engine; gc.collect()

# Write results to temp file
out = {{"label": "{label}", "load_time": round(load_time, 1), "results": results}}
with open(sys.argv[1], "w") as f:
    json.dump(out, f)
print(f"  {{'{label}'}} done.", flush=True)
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as sf:
        sf.write(script)
        script_path = sf.name

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as rf:
        result_path = rf.name

    env = os.environ.copy()
    proc = sp.run(
        [sys.executable, script_path, result_path],
        env=env, timeout=900,
        capture_output=True, text=True,
    )
    # Print subprocess output so we see progress
    if proc.stdout:
        print(proc.stdout, end='', flush=True)
    os.unlink(script_path)

    if proc.returncode != 0:
        print(f"  Subprocess exited with code {proc.returncode}", flush=True)
        if proc.stderr:
            # Print last 2000 chars of stderr for diagnostics
            stderr_tail = proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr
            print(f"  STDERR:\n{stderr_tail}", flush=True)
        try:
            os.unlink(result_path)
        except OSError:
            pass
        return None

    with open(result_path) as f:
        data = json.load(f)
    os.unlink(result_path)
    return data


def validate_config(cfg_name, tokenizer, stop_ids, max_gen, compute_unit):
    """Validate one experimental config against stable baseline.

    Each engine runs in a separate subprocess to avoid OOM on 16 GB Macs.
    """
    cfg = get_config(cfg_name)
    model_dir = cfg["output_dir"]

    # Check if models exist
    if not os.path.isdir(model_dir):
        print(f"\n  SKIP {cfg_name}: output dir not found ({model_dir})")
        print(f"  Run: python scripts_qwen3_5/explore/explore_export.py --config {cfg_name}")
        return None

    # Check for embeddings (.mlmodelc or .mlpackage)
    has_embed = any(os.path.exists(os.path.join(model_dir, f"embeddings{ext}"))
                    for ext in (".mlmodelc", ".mlpackage"))
    # Check for lm_head (may be lm_head or lm_head_logits)
    has_lmhead = any(os.path.exists(os.path.join(model_dir, f"{n}{ext}"))
                     for n in ("lm_head", "lm_head_logits")
                     for ext in (".mlmodelc", ".mlpackage"))
    # Check for FFN chunks (combined or separate)
    ffn_label = f"LUT{cfg['LUT_BITS']}"
    combined_dir = os.path.join(model_dir, f"combined_{ffn_label}_dedup")
    has_ffn = os.path.isdir(combined_dir) and all(
        os.path.exists(os.path.join(combined_dir, f"chunk{ci}.mlpackage"))
        for ci in range(cfg["NUM_CHUNKS"])
    )
    if not has_ffn:
        has_ffn = all(os.path.exists(os.path.join(model_dir, f"ffn_{ffn_label}_chunk{ci}.mlpackage"))
                      for ci in range(cfg["NUM_CHUNKS"]))
    missing = []
    if not has_embed: missing.append("embeddings")
    if not has_lmhead: missing.append("lm_head")
    if not has_ffn: missing.append("ffn_chunks")
    if missing:
        print(f"\n  SKIP {cfg_name}: missing: {missing}")
        return None

    print(f"\n{'='*70}")
    print(f"  VALIDATING: {cfg_name}")
    print(f"  Batch={cfg['BATCH_SIZE']}  CTX={cfg['CTX']}  Models: {model_dir}")
    print(f"  (Each engine runs in a subprocess to avoid OOM)")
    print(f"{'='*70}")

    # Phase 1: Run baseline in subprocess
    baseline_data = _run_single_engine(
        STABLE["output_dir"], STABLE["CTX"], STABLE["NUM_CHUNKS"],
        compute_unit, tokenizer, stop_ids, max_gen, "STABLE baseline")

    if baseline_data is None:
        print(f"  FAILED: baseline subprocess crashed")
        return {"config": cfg_name, "status": "BASELINE_CRASHED", "error": "subprocess OOM or crash"}

    # Phase 2: Run experiment in subprocess
    exp_data = _run_single_engine(
        model_dir, cfg["CTX"], cfg["NUM_CHUNKS"],
        compute_unit, tokenizer, stop_ids, max_gen, f"EXPERIMENTAL {cfg_name}")

    if exp_data is None:
        print(f"  FAILED: {cfg_name} subprocess crashed")
        return {"config": cfg_name, "status": "EXP_CRASHED", "error": "subprocess OOM or crash"}

    baseline_results = baseline_data["results"]
    exp_results = exp_data["results"]
    baseline_load = baseline_data["load_time"]
    exp_load = exp_data["load_time"]

    # Compare
    report = compare_results(baseline_results, exp_results, "stable", cfg_name)
    report["config"] = cfg_name
    report["status"] = "PASS" if report["all_match"] else "DIVERGED"
    report["baseline_load_s"] = baseline_load
    report["exp_load_s"] = exp_load
    report["cfg"] = {"BATCH_SIZE": cfg["BATCH_SIZE"], "CTX": cfg["CTX"]}

    # Print comparison
    print(f"\n  COMPARISON: {cfg_name} vs stable")
    print(f"  {'Turn':>6} {'Match':>8} {'Base (s)':>10} {'Exp (s)':>10} {'Base tok/s':>12} {'Exp tok/s':>12}")
    print(f"  {'-'*60}")
    for t in report["turns"]:
        print(f"  {t['turn']:>6} {t['match_pct']:>7.0f}% {t['baseline_elapsed']:>10.1f} {t['exp_elapsed']:>10.1f} {t['baseline_tok_s']:>12.1f} {t['exp_tok_s']:>12.1f}")

    status_str = "PASS ✓" if report["all_match"] else "DIVERGED (expected for different CTX)"
    print(f"\n  Status: {status_str}")
    print(f"  Load time: baseline={baseline_load:.0f}s  exp={exp_load:.0f}s")

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate exploratory Qwen3.5-4B configs")
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--tokens", type=int, default=20,
                        help="Max tokens to generate per turn")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to this path")
    args = parser.parse_args()

    if args.list:
        from scripts_qwen3_5.explore.explore_config import list_configs
        list_configs()
        return

    if not args.config:
        print("ERROR: --config required")
        sys.exit(1)

    cu = ct.ComputeUnit.CPU_AND_NE
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=False)
    stop_ids = _build_stop_ids(tokenizer)

    if args.config == "all":
        configs = list(CONFIGS.keys())
    else:
        configs = [args.config]

    all_reports = []
    for cfg_name in configs:
        report = validate_config(cfg_name, tokenizer, stop_ids, args.tokens, cu)
        if report:
            all_reports.append(report)

    # Summary
    print(f"\n{'='*70}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'Status':<20} {'Load (s)':<10} {'Notes'}")
    print(f"  {'-'*65}")
    for r in all_reports:
        notes = ""
        if r["status"] == "PASS":
            notes = "100% token match with baseline"
        elif r["status"] == "DIVERGED":
            mismatches = [t for t in r.get("turns", []) if t.get("match_pct", 100) < 100]
            if mismatches:
                notes = f"Diverged at turn {mismatches[0]['turn']}"
        elif "error" in r:
            notes = r["error"][:50]
        print(f"  {r['config']:<25} {r['status']:<20} {r.get('exp_load_s', '?'):<10} {notes}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_reports, f, indent=2)
        print(f"\n  Report saved to {args.output}")


if __name__ == "__main__":
    main()
