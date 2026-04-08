#!/usr/bin/env python3
"""Compare cs=16 vs cs=32 model quality and latency.

Starts the chat server for each model set, sends identical prompts,
captures output text and timing. Prints a side-by-side comparison.
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

PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "python")
CHAT_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_server.py")
TOKENIZER = "/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"

MODELS = {
    "cs=16": {
        "model_dir": "/Users/yw68/Anemll/qwen3_5_stable_models_6chunk",
        "ffn_dir": "/Users/yw68/Anemll/qwen3_5_stable_models_6chunk/combined_LUT6_dedup",
    },
    "cs=32": {
        "model_dir": "/Users/yw68/Anemll/qwen3_5_cs32_models",
        "ffn_dir": "/Users/yw68/Anemll/qwen3_5_cs32_models/combined_LUT6_dedup",
    },
}

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "教我做红烧鱼",
    "Write a haiku about the ocean.",
    "What are the three laws of thermodynamics?",
]

MAX_TOKENS = 200  # enough for quality comparison


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
    """Wait for the server to respond to /api/status."""
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
        if int(elapsed) % 30 == 0:
            print(f"({int(elapsed)}s)", end="", flush=True)
        time.sleep(2)
    return False


def send_chat(url, message, max_tokens=200, enable_thinking=False):
    """Send a chat message and collect SSE response.

    Returns dict with text, timing, and token count info.
    """
    payload = json.dumps({
        "message": message,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/api/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.time()
    response = urllib.request.urlopen(req, timeout=180)
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

    # Parse meta from "done" event:
    #   {"type":"done","decode_tokens":N,"end_pos":P,"elapsed":T,"stop_reason":"eos"}
    decode_tokens = meta_info.get("decode_tokens", 0)
    decode_elapsed = meta_info.get("elapsed", 0)
    stop_reason = meta_info.get("stop_reason", "unknown")

    # Time to first token = prefill time (approx)
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
    req = urllib.request.Request(f"{url}/api/reset", method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def start_server(model_dir, ffn_dir):
    """Start chat server as a subprocess. Returns the process."""
    cmd = [
        PYTHON, CHAT_SERVER,
        "--model-dir", model_dir,
        "--tokenizer", TOKENIZER,
        "--ffn-dir", ffn_dir,
        "--ctx", "2048",
        "--port", str(PORT),
    ]
    # Send stdout/stderr to a log file to avoid pipe buffer deadlock
    log_path = f"/tmp/chat_server_{model_dir.split('/')[-1]}.log"
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


def run_comparison():
    results = {}

    for label, cfg in MODELS.items():
        print(f"\n{'='*70}")
        print(f"  TESTING: {label}")
        print(f"  model_dir: {cfg['model_dir']}")
        print(f"  ffn_dir:   {cfg['ffn_dir']}")
        print(f"{'='*70}")

        # Kill any existing server
        kill_port(PORT)
        time.sleep(1)

        # Start server
        print(f"\n  Starting server for {label}...")
        proc = start_server(cfg["model_dir"], cfg["ffn_dir"])

        # Wait for ready
        print(f"  Waiting for server to be ready...", end="", flush=True)
        if not wait_for_server(BASE_URL, timeout=600):
            print(" TIMEOUT!")
            proc.kill()
            proc.wait()
            results[label] = {"error": "Server failed to start"}
            continue
        print(" READY!")

        # Run prompts
        label_results = []
        for i, prompt in enumerate(PROMPTS):
            print(f"\n  [{i+1}/{len(PROMPTS)}] Q: {prompt}")

            # Reset between prompts for fair comparison
            reset_chat(BASE_URL)
            time.sleep(0.5)

            try:
                r = send_chat(BASE_URL, prompt, max_tokens=MAX_TOKENS,
                             enable_thinking=False)
                # Strip <think>...</think> if present
                text = r["text"]
                if "<think>" in text:
                    end_tag = text.find("</think>")
                    if end_tag >= 0:
                        text = text[end_tag + len("</think>"):].strip()
                r["text"] = text

                label_results.append({"prompt": prompt, **r})
                print(f"  A: {text[:200]}{'...' if len(text)>200 else ''}")
                print(f"  Tokens: {r['decode_tokens']}, "
                      f"TTFT: {r['ttft_ms']:.0f}ms, "
                      f"Decode: {r['decode_tps']:.1f} tok/s, "
                      f"Total: {r['total_ms']:.0f}ms, "
                      f"Stop: {r['stop_reason']}")
            except Exception as e:
                print(f"  ERROR: {e}")
                label_results.append({"prompt": prompt, "error": str(e)})

        results[label] = label_results

        # Kill server
        print(f"\n  Stopping server for {label}...")
        proc.kill()
        proc.wait()
        if hasattr(proc, '_log_file'):
            proc._log_file.close()
            # Print server log tail for debugging
            with open(proc._log_path) as f:
                lines = f.readlines()
            if lines:
                print(f"  Server log (last 5 lines):")
                for line in lines[-5:]:
                    print(f"    {line.rstrip()}")
        kill_port(PORT)
        time.sleep(2)

    # ── Print comparison ──
    print("\n\n")
    print("=" * 80)
    print("  COMPARISON: cs=16 vs cs=32")
    print("=" * 80)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'─'*80}")
        print(f"  Q: {prompt}")
        print(f"{'─'*80}")

        for label in MODELS:
            if label not in results or isinstance(results[label], dict):
                print(f"  [{label}] ERROR: Server failed")
                continue
            r = results[label][i]
            if "error" in r:
                print(f"  [{label}] ERROR: {r['error']}")
                continue

            text = r["text"]
            # Wrap long text
            wrapped = textwrap.fill(text, width=70, initial_indent="    ",
                                    subsequent_indent="    ")
            print(f"\n  [{label}]")
            print(f"    TTFT: {r['ttft_ms']:.0f}ms | "
                  f"Decode: {r['decode_tps']:.1f} tok/s | "
                  f"Tokens: {r['decode_tokens']} | "
                  f"Total: {r['total_ms']:.0f}ms | "
                  f"Stop: {r['stop_reason']}")
            print(wrapped)

    # ── Summary table ──
    print(f"\n\n{'='*80}")
    print("  LATENCY SUMMARY (Time to First Token)")
    print(f"{'='*80}")
    print(f"  {'Prompt':<45} {'cs=16':>12} {'cs=32':>12} {'Δ':>8}")
    print(f"  {'─'*45} {'─'*12} {'─'*12} {'─'*8}")

    for i, prompt in enumerate(PROMPTS):
        short = prompt[:42] + "..." if len(prompt) > 42 else prompt
        vals = {}
        for label in MODELS:
            if label not in results or isinstance(results[label], dict):
                vals[label] = None
                continue
            r = results[label][i]
            vals[label] = r.get("ttft_ms")

        if vals.get("cs=16") is not None and vals.get("cs=32") is not None:
            delta = vals["cs=32"] - vals["cs=16"]
            print(f"  {short:<45} {vals['cs=16']:>10.0f}ms {vals['cs=32']:>10.0f}ms {delta:>+7.0f}ms")
        else:
            print(f"  {short:<45} {'N/A':>12} {'N/A':>12} {'N/A':>8}")

    # Decode TPS summary
    print(f"\n  DECODE SPEED (tok/s)")
    print(f"  {'Prompt':<45} {'cs=16':>12} {'cs=32':>12}")
    print(f"  {'─'*45} {'─'*12} {'─'*12}")
    for i, prompt in enumerate(PROMPTS):
        short = prompt[:42] + "..." if len(prompt) > 42 else prompt
        vals = {}
        for label in MODELS:
            if label not in results or isinstance(results[label], dict):
                vals[label] = None
                continue
            r = results[label][i]
            vals[label] = r.get("decode_tps")

        if vals.get("cs=16") is not None and vals.get("cs=32") is not None:
            print(f"  {short:<45} {vals['cs=16']:>9.1f}t/s {vals['cs=32']:>9.1f}t/s")
        else:
            print(f"  {short:<45} {'N/A':>12} {'N/A':>12}")

    print(f"\n{'='*80}")
    print("  COMPARISON COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_comparison()
