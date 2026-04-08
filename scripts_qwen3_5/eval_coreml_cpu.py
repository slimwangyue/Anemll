#!/usr/bin/env python3
"""Evaluate CoreML models on CPU-only compute unit.

Assumes chat_server is already running on the target port with --compute-unit cpu.

Usage:
    python scripts_qwen3_5/eval_coreml_cpu.py --port 8081 --label lut6-cpu
"""
import sys, os, time, json, urllib.request, textwrap, argparse

BASE_URL = None
MAX_TOKENS = 200

PROMPTS = [
    "Explain the key differences between TCP and UDP protocols in computer networking. "
    "Include at least three specific differences and when you would use each one.",

    "A farmer has a rectangular field that is 120 meters long and 80 meters wide. "
    "He wants to build a fence around the entire field and also divide it into four "
    "equal sections with internal fences parallel to the shorter side. How many meters "
    "of fencing material does he need in total?",

    "请用三段话描述一个未来城市的生活场景。第一段描述交通，第二段描述住宅，"
    "第三段描述人们的日常工作和娱乐方式。每段至少写三句话。",

    "List the top 5 largest countries in the world by land area. For each country, "
    "provide its approximate area in square kilometers, its capital city, and the "
    "continent it is primarily located on. Format your answer as a numbered list.",

    "Write a Python function called 'merge_sorted_lists' that takes two sorted "
    "lists of integers and returns a single sorted list containing all elements "
    "from both lists. Use the merge step of merge sort, not the built-in sort. "
    "Include a brief docstring explaining the time complexity.",

    "If a train leaves Station A at 9:00 AM traveling east at 80 km/h, and another "
    "train leaves Station B (which is 500 km east of A) at 10:00 AM traveling west "
    "at 120 km/h, at what time will the two trains meet? Show your work step by step "
    "and express the answer in hours and minutes.",
]


def send_chat(message, max_tokens=200):
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
        f"{BASE_URL}/api/chat/stream",
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

    if "<think>" in full_text:
        end_tag = full_text.find("</think>")
        if end_tag >= 0:
            full_text = full_text[end_tag + len("</think>"):].strip()

    decode_tokens = meta_info.get("decode_tokens", 0)
    decode_elapsed = meta_info.get("elapsed", 0)
    stop_reason = meta_info.get("stop_reason", "unknown")
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else 0
    decode_tps = (decode_tokens / decode_elapsed) if decode_elapsed > 0 else 0

    return {
        "text": full_text,
        "decode_tokens": decode_tokens,
        "decode_elapsed_s": decode_elapsed,
        "decode_tps": decode_tps,
        "ttft_ms": ttft_ms,
        "stop_reason": stop_reason,
    }


def reset_chat():
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/reset", method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def detect_repetition(text, min_ngram=3, threshold=3):
    words = text.split()
    if len(words) < min_ngram * 2:
        return 0
    ngram_counts = {}
    for i in range(len(words) - min_ngram + 1):
        ngram = tuple(words[i:i + min_ngram])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    return sum(1 for c in ngram_counts.values() if c >= threshold)


def main():
    global BASE_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--label", default="coreml-cpu",
                        help="Label for this run (e.g. lut6-cpu, fp16-cpu)")
    args = parser.parse_args()
    BASE_URL = f"http://127.0.0.1:{args.port}"

    # Check server is ready
    try:
        status = json.loads(urllib.request.urlopen(
            f"{BASE_URL}/api/status", timeout=5).read())
        if not status.get("ready"):
            print("Server not ready!")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot connect to server at {BASE_URL}: {e}")
        sys.exit(1)

    print(f"Running eval: {args.label} (server at {BASE_URL})")
    print(f"{'='*70}")

    results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] Q: {prompt[:80]}...")
        reset_chat()
        time.sleep(0.5)

        r = send_chat(prompt, max_tokens=MAX_TOKENS)
        rep_count = detect_repetition(r["text"])
        r["repetition_count"] = rep_count
        results.append({"prompt": prompt, **r})
        print(f"  A: {r['text'][:200]}{'...' if len(r['text'])>200 else ''}")
        print(f"  Tokens: {r['decode_tokens']}, "
              f"TTFT: {r['ttft_ms']:.0f}ms, "
              f"Decode: {r['decode_tps']:.1f} tok/s, "
              f"Stop: {r['stop_reason']}, "
              f"Reps: {rep_count}")

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: {args.label}")
    print(f"{'='*70}")
    rep_stops = sum(1 for r in results if r.get("stop_reason") == "repetition")
    print(f"  Repetition stops: {rep_stops}/{len(results)}")
    avg_tps = sum(r["decode_tps"] for r in results) / len(results)
    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    print(f"  Avg decode: {avg_tps:.1f} tok/s")
    print(f"  Avg TTFT: {avg_ttft:.0f}ms")

    # Save
    out_path = f"/tmp/eval_{args.label}_results.json"
    with open(out_path, "w") as f:
        json.dump({"label": args.label, "results": results}, f, indent=2,
                  ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
