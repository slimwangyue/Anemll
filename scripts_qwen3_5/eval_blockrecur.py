#!/usr/bin/env python3
"""End-to-end evaluation: block-recursive BC=16 batched prefill quality.

Compares three configurations:
  1. sequential  — no prefill models (token-by-token, ground truth)
  2. cs32-old    — old row-by-row cs=32 batched prefill
  3. blockrecur  — block-recursive BC=16 batched prefill

All prompts are >32 tokens to guarantee batched prefill is triggered
(PREFILL_CROSSOVER=32 in chat_server.py).

Usage:
    python scripts_qwen3_5/eval_blockrecur.py
    python scripts_qwen3_5/eval_blockrecur.py --configs sequential,blockrecur
    python scripts_qwen3_5/eval_blockrecur.py --only-config blockrecur
"""
import subprocess
import sys
import os
import time
import json
import urllib.request
import urllib.error
import signal
import textwrap
import argparse
import re

PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "python")
CHAT_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_server.py")
TOKENIZER = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"
MAX_TOKENS = 200

# ── Model configurations ──
# sequential: use cs32 decode models but NO prefill models → forces sequential
# cs32-old:   old row-by-row cs32 models with batched prefill
# blockrecur: new block-recursive BC=16 models with batched prefill
CONFIGS = {
    "sequential": {
        "model_dir": "/Users/yw68/Anemll/qwen3_5_cs32_models",
        "ffn_dir": "__sequential__",  # Special marker: create decode-only dir
        "description": "Sequential token-by-token (ground truth, no batched prefill)",
    },
    "cs32-old": {
        "model_dir": "/Users/yw68/Anemll/qwen3_5_cs32_models",
        "ffn_dir": "/Users/yw68/Anemll/qwen3_5_cs32_models/combined_LUT6_dedup",
        "description": "Old row-by-row cs=32 batched prefill",
    },
    "blockrecur": {
        "model_dir": "/Users/yw68/Anemll/qwen3_5_blockrecur_full",
        "ffn_dir": "/Users/yw68/Anemll/qwen3_5_blockrecur_full/combined_LUT6_dedup",
        "description": "Block-recursive BC=16 batched prefill",
    },
}

# ── Prompts ── all >32 tokens to guarantee batched prefill triggers
PROMPTS = [
    # 1. Medium English (~45 tokens) — factual recall
    "Explain the key differences between TCP and UDP protocols in computer networking. "
    "Include at least three specific differences and when you would use each one.",

    # 2. Long English (~65 tokens) — reasoning/analysis
    "A farmer has a rectangular field that is 120 meters long and 80 meters wide. "
    "He wants to build a fence around the entire field and also divide it into four "
    "equal sections with internal fences parallel to the shorter side. How many meters "
    "of fencing material does he need in total?",

    # 3. Chinese (~50 tokens) — creative writing
    "请用三段话描述一个未来城市的生活场景。第一段描述交通，第二段描述住宅，"
    "第三段描述人们的日常工作和娱乐方式。每段至少写三句话。",

    # 4. Structured/list (~55 tokens)
    "List the top 5 largest countries in the world by land area. For each country, "
    "provide its approximate area in square kilometers, its capital city, and the "
    "continent it is primarily located on. Format your answer as a numbered list.",

    # 5. Code generation (~60 tokens)
    "Write a Python function called 'merge_sorted_lists' that takes two sorted "
    "lists of integers and returns a single sorted list containing all elements "
    "from both lists. Use the merge step of merge sort, not the built-in sort. "
    "Include a brief docstring explaining the time complexity.",

    # 6. Multi-step reasoning (~70 tokens)
    "If a train leaves Station A at 9:00 AM traveling east at 80 km/h, and another "
    "train leaves Station B (which is 500 km east of A) at 10:00 AM traveling west "
    "at 120 km/h, at what time will the two trains meet? Show your work step by step "
    "and express the answer in hours and minutes.",
]


def kill_port(port):
    """Kill any process listening on the given port."""
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        if out:
            for pid in out.split("\n"):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            time.sleep(1)
    except subprocess.CalledProcessError:
        pass


def wait_for_server(url, timeout=600):
    """Wait for server to report ready."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.urlopen(f"{url}/api/status", timeout=2)
            data = json.loads(req.read().decode())
            if data.get("ready"):
                return True
        except Exception:
            pass
        elapsed = time.time() - t0
        if int(elapsed) % 30 == 0 and int(elapsed) > 0:
            print(f"  ({int(elapsed)}s)", end="", flush=True)
        time.sleep(2)
    return False


def send_chat(url, message, max_tokens=200):
    """Send a chat message and collect SSE response."""
    payload = json.dumps({
        "message": message,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "frequency_penalty": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/api/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.time()
    response = urllib.request.urlopen(req, timeout=300)
    text_parts = []
    meta_info = {}
    first_token_time = None

    buf = b""
    while True:
        chunk = response.read(1)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event_data, buf = buf.split(b"\n\n", 1)
            for line in event_data.decode("utf-8", errors="replace").split("\n"):
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        continue
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "token":
                        if first_token_time is None:
                            first_token_time = time.time()
                        text_parts.append(ev.get("text", ""))
                    elif ev.get("type") == "done":
                        meta_info = ev

    t_end = time.time()
    full_text = "".join(text_parts)

    # Strip thinking tags
    if "<think>" in full_text:
        end_tag = full_text.find("</think>")
        if end_tag >= 0:
            full_text = full_text[end_tag + len("</think>"):].strip()

    decode_tokens = meta_info.get("decode_tokens", 0)
    decode_elapsed = meta_info.get("elapsed", 0)
    stop_reason = meta_info.get("stop_reason", "unknown")
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else 0
    total_ms = (t_end - t_start) * 1000
    decode_tps = (decode_tokens / decode_elapsed) if decode_elapsed > 0 else 0

    return {
        "text": full_text,
        "decode_tokens": decode_tokens,
        "decode_elapsed_s": decode_elapsed,
        "decode_tps": decode_tps,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "stop_reason": stop_reason,
    }


def reset_chat(url):
    """Reset the chat session."""
    try:
        req = urllib.request.Request(f"{url}/api/reset", method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def _prepare_sequential_dir(model_dir):
    """Create a temp dir with symlinks to decode-only models (no prefill).
    This forces the chat server into sequential mode."""
    seq_dir = "/tmp/eval_sequential_models"
    if os.path.isdir(seq_dir):
        import shutil
        shutil.rmtree(seq_dir)
    os.makedirs(seq_dir)
    # Symlink everything EXCEPT prefill models
    for name in os.listdir(model_dir):
        if "prefill" in name.lower() or name == "combined_LUT6_dedup":
            continue
        src = os.path.join(model_dir, name)
        dst = os.path.join(seq_dir, name)
        os.symlink(src, dst)
    return seq_dir


def start_server(model_dir, ffn_dir=None):
    """Start chat server. If ffn_dir is None, no combined dir → sequential only."""
    # Handle sequential marker
    actual_model_dir = model_dir
    actual_ffn_dir = ffn_dir
    if ffn_dir == "__sequential__":
        actual_model_dir = _prepare_sequential_dir(model_dir)
        actual_ffn_dir = None

    cmd = [
        PYTHON, CHAT_SERVER,
        "--model-dir", actual_model_dir,
        "--tokenizer", TOKENIZER,
        "--ctx", "2048",
        "--num-chunks", "6",
        "--port", str(PORT),
    ]
    if actual_ffn_dir:
        cmd.extend(["--ffn-dir", actual_ffn_dir])
    log_path = f"/tmp/chat_server_eval_{model_dir.split('/')[-1]}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd="/Users/yw68/Anemll",
    )
    proc._log_file = log_file
    proc._log_path = log_path
    return proc


def detect_repetition(text, min_ngram=3, threshold=3):
    """Detect repeated ngrams in text. Returns count of repeated patterns."""
    words = text.split()
    if len(words) < min_ngram * 2:
        return 0
    ngram_counts = {}
    for i in range(len(words) - min_ngram + 1):
        ngram = tuple(words[i:i + min_ngram])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    return sum(1 for c in ngram_counts.values() if c >= threshold)


def first_n_tokens(text, n=10):
    """Return first n whitespace-separated tokens."""
    return " ".join(text.split()[:n])


def run_evaluation(selected_configs):
    results = {}

    for label in selected_configs:
        cfg = CONFIGS[label]
        print(f"\n{'='*70}")
        print(f"  CONFIG: {label}")
        print(f"  {cfg['description']}")
        print(f"  model_dir: {cfg['model_dir']}")
        print(f"  ffn_dir:   {cfg.get('ffn_dir', 'None (sequential)')}")
        print(f"{'='*70}")

        # Verify model dir exists
        if not os.path.isdir(cfg["model_dir"]):
            print(f"  ERROR: model_dir does not exist: {cfg['model_dir']}")
            results[label] = {"error": "model_dir not found"}
            continue
        ffn_dir = cfg.get("ffn_dir")
        if ffn_dir and ffn_dir != "__sequential__" and not os.path.isdir(ffn_dir):
            print(f"  ERROR: ffn_dir does not exist: {ffn_dir}")
            results[label] = {"error": "ffn_dir not found"}
            continue

        kill_port(PORT)
        time.sleep(1)

        print(f"\n  Starting server for {label}...")
        proc = start_server(cfg["model_dir"], cfg.get("ffn_dir"))

        print(f"  Waiting for server...", end="", flush=True)
        if not wait_for_server(BASE_URL, timeout=600):
            print(" TIMEOUT!")
            proc.kill()
            proc.wait()
            results[label] = {"error": "Server failed to start"}
            # Print log for debugging
            if hasattr(proc, '_log_path') and os.path.exists(proc._log_path):
                with open(proc._log_path) as f:
                    print("  Server log (last 10 lines):")
                    for line in f.readlines()[-10:]:
                        print(f"    {line.rstrip()}")
            continue
        print(" READY!")

        # Print server log snippet to confirm prefill mode
        if hasattr(proc, '_log_path') and os.path.exists(proc._log_path):
            with open(proc._log_path) as f:
                for line in f:
                    if "prefill=" in line:
                        print(f"  Server mode: {line.strip()}")
                        break

        label_results = []
        for i, prompt in enumerate(PROMPTS):
            print(f"\n  [{i+1}/{len(PROMPTS)}] Q: {prompt[:80]}...")
            reset_chat(BASE_URL)
            time.sleep(0.5)

            try:
                r = send_chat(BASE_URL, prompt, max_tokens=MAX_TOKENS)
                rep_count = detect_repetition(r["text"])
                r["repetition_count"] = rep_count
                r["first_10"] = first_n_tokens(r["text"])
                label_results.append({"prompt": prompt, **r})
                print(f"  A: {r['text'][:200]}{'...' if len(r['text'])>200 else ''}")
                print(f"  Tokens: {r['decode_tokens']}, "
                      f"TTFT: {r['ttft_ms']:.0f}ms, "
                      f"Decode: {r['decode_tps']:.1f} tok/s, "
                      f"Stop: {r['stop_reason']}, "
                      f"Reps: {rep_count}")
            except Exception as e:
                print(f"  ERROR: {e}")
                label_results.append({"prompt": prompt, "error": str(e)})

        results[label] = label_results

        print(f"\n  Stopping server for {label}...")
        proc.kill()
        proc.wait()
        if hasattr(proc, '_log_file'):
            proc._log_file.close()
        kill_port(PORT)
        time.sleep(2)

    # ── Detailed comparison ──
    print("\n\n")
    print("=" * 80)
    print("  DETAILED COMPARISON")
    print("=" * 80)

    ref_label = "sequential" if "sequential" in results else None
    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'─'*80}")
        print(f"  Q: {prompt[:100]}...")
        print(f"{'─'*80}")

        for label in selected_configs:
            if label not in results or isinstance(results[label], dict):
                print(f"  [{label}] ERROR: {results.get(label, {}).get('error', 'N/A')}")
                continue
            r = results[label][i]
            if "error" in r:
                print(f"  [{label}] ERROR: {r['error']}")
                continue

            wrapped = textwrap.fill(r["text"], width=70,
                                    initial_indent="    ",
                                    subsequent_indent="    ")
            print(f"\n  [{label}]")
            print(f"    TTFT: {r['ttft_ms']:.0f}ms | "
                  f"Decode: {r['decode_tps']:.1f} t/s | "
                  f"Tokens: {r['decode_tokens']} | "
                  f"Stop: {r['stop_reason']} | "
                  f"Reps: {r['repetition_count']}")
            print(wrapped)

            # Show first-token agreement with reference
            if ref_label and label != ref_label and ref_label in results:
                ref_r = results[ref_label][i]
                if "error" not in ref_r:
                    ref_first = first_n_tokens(ref_r["text"], 5)
                    cur_first = first_n_tokens(r["text"], 5)
                    match = "MATCH" if ref_first == cur_first else "DIFFER"
                    print(f"    First-5 vs ref: {match}")
                    if match == "DIFFER":
                        print(f"      ref:  {ref_first}")
                        print(f"      this: {cur_first}")

    # ── Summary tables ──
    active = [l for l in selected_configs if l in results and not isinstance(results[l], dict)]
    if len(active) < 1:
        print("\nNo valid results to summarize.")
        return results

    # TTFT table
    print(f"\n\n{'='*80}")
    print("  TIME TO FIRST TOKEN (ms)")
    print(f"{'='*80}")
    header = f"  {'Prompt':<40}"
    for l in active:
        header += f" {l:>14}"
    print(header)
    print(f"  {'─'*40}" + "".join(f" {'─'*14}" for _ in active))
    for i, prompt in enumerate(PROMPTS):
        short = prompt[:37] + "..." if len(prompt) > 37 else prompt
        row = f"  {short:<40}"
        for l in active:
            r = results[l][i]
            if "error" in r:
                row += f" {'ERROR':>14}"
            else:
                row += f" {r['ttft_ms']:>12.0f}ms"
        print(row)

    # Decode speed table
    print(f"\n  DECODE SPEED (tok/s)")
    print(f"  {'─'*40}" + "".join(f" {'─'*14}" for _ in active))
    for i, prompt in enumerate(PROMPTS):
        short = prompt[:37] + "..." if len(prompt) > 37 else prompt
        row = f"  {short:<40}"
        for l in active:
            r = results[l][i]
            if "error" in r:
                row += f" {'ERROR':>14}"
            else:
                row += f" {r['decode_tps']:>11.1f}t/s"
        print(row)

    # Quality summary
    print(f"\n  QUALITY SUMMARY")
    print(f"  {'─'*40}" + "".join(f" {'─'*14}" for _ in active))
    print(f"  {'Metric':<40}" + "".join(f" {l:>14}" for l in active))
    print(f"  {'─'*40}" + "".join(f" {'─'*14}" for _ in active))

    for l in active:
        valid = [r for r in results[l] if "error" not in r]
        eos_count = sum(1 for r in valid if r["stop_reason"] == "eos")
        rep_count = sum(1 for r in valid if r["stop_reason"] == "repetition")
        len_count = sum(1 for r in valid if r["stop_reason"] == "length")
        total_reps = sum(r.get("repetition_count", 0) for r in valid)
        avg_ttft = sum(r["ttft_ms"] for r in valid) / max(len(valid), 1)
        avg_tps = sum(r["decode_tps"] for r in valid) / max(len(valid), 1)

    # Print per-config summary
    for l in active:
        valid = [r for r in results[l] if "error" not in r]
        eos_count = sum(1 for r in valid if r["stop_reason"] == "eos")
        rep_count = sum(1 for r in valid if r["stop_reason"] == "repetition")
        len_count = sum(1 for r in valid if r["stop_reason"] == "length")
        total_reps = sum(r.get("repetition_count", 0) for r in valid)
        avg_ttft = sum(r["ttft_ms"] for r in valid) / max(len(valid), 1)
        avg_tps = sum(r["decode_tps"] for r in valid) / max(len(valid), 1)
        print(f"\n  [{l}]")
        print(f"    Prompts OK: {len(valid)}/{len(PROMPTS)}")
        print(f"    Stop reasons: eos={eos_count}, repetition={rep_count}, length={len_count}")
        print(f"    Total repeated ngrams: {total_reps}")
        print(f"    Avg TTFT: {avg_ttft:.0f}ms")
        print(f"    Avg decode: {avg_tps:.1f} tok/s")

    # First-token agreement with reference
    if ref_label and ref_label in active:
        print(f"\n  FIRST-TOKEN AGREEMENT vs '{ref_label}' (reference)")
        print(f"  {'─'*60}")
        for l in active:
            if l == ref_label:
                continue
            matches = 0
            total = 0
            for i in range(len(PROMPTS)):
                ref_r = results[ref_label][i]
                cur_r = results[l][i]
                if "error" in ref_r or "error" in cur_r:
                    continue
                total += 1
                if first_n_tokens(ref_r["text"], 1) == first_n_tokens(cur_r["text"], 1):
                    matches += 1
            print(f"    {l}: {matches}/{total} first tokens match ({100*matches/max(total,1):.0f}%)")

    print(f"\n{'='*80}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*80}")

    # Save raw results to JSON
    out_path = "/tmp/eval_blockrecur_results.json"
    serializable = {}
    for k, v in results.items():
        if isinstance(v, list):
            serializable[k] = v
        else:
            serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Raw results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate block-recursive BC=16 batched prefill quality")
    parser.add_argument("--configs", default=None,
                        help="Comma-separated config names (default: all)")
    parser.add_argument("--only-config", default=None,
                        help="Run only a single config")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens per response")
    args = parser.parse_args()

    global MAX_TOKENS
    MAX_TOKENS = args.max_tokens

    if args.only_config:
        selected = [args.only_config]
    elif args.configs:
        selected = [c.strip() for c in args.configs.split(",")]
    else:
        selected = list(CONFIGS.keys())

    for c in selected:
        if c not in CONFIGS:
            print(f"ERROR: Unknown config '{c}'. Available: {list(CONFIGS.keys())}")
            sys.exit(1)

    print("=" * 70)
    print("  Block-Recursive BC=16 End-to-End Evaluation")
    print(f"  Configs: {', '.join(selected)}")
    print(f"  Prompts: {len(PROMPTS)} (all >32 tokens → forces batched prefill)")
    print(f"  Max tokens: {MAX_TOKENS}")
    print("=" * 70)

    run_evaluation(selected)


if __name__ == "__main__":
    main()
