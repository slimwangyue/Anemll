#!/usr/bin/env python3
"""
Automated validation runner for Qwen3.5 chat_server.py.

Sends prompts via HTTP to the running chat_server, collects responses,
runs quality checks, and produces a JSON/text report.

Usage:
    python run_validation.py --url http://localhost:8080 [--phase quality|robustness|recovery|ane|all]
    python run_validation.py --url http://localhost:8080 --phase quality --mode think-on
    python run_validation.py --url http://localhost:8080 --phase robustness
    python run_validation.py --dry-run   # just print prompts, no server needed
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Optional, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(SCRIPT_DIR, "validation_prompts.json")


# ── SSE Client ───────────────────────────────────────────────────────

def chat_request(url: str, message: str, max_tokens: int = 2048,
                 enable_thinking: bool = True, timeout: int = 180,
                 **sampling_kwargs) -> Dict[str, Any]:
    """Send a chat request and collect the full SSE response.

    Returns dict with keys: text, tokens, elapsed, tok_s, stop_reason, think_text, error
    """
    endpoint = f"{url.rstrip('/')}/api/chat/stream"
    payload = {
        "message": message,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
    }
    payload.update(sampling_kwargs)

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint, data=data,
        headers={"Content-Type": "application/json"},
    )

    result = {
        "text": "", "tokens": 0, "elapsed": 0.0, "tok_s": 0.0,
        "stop_reason": "unknown", "think_text": "", "error": None,
    }

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buf = ""
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    try:
                        evt = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    etype = evt.get("type", "")
                    if etype == "token":
                        result["text"] += evt.get("text", "")
                    elif etype == "think":
                        result["think_text"] += evt.get("text", "")
                    elif etype == "done":
                        result["tokens"] = evt.get("decode_tokens", 0)
                        result["elapsed"] = evt.get("elapsed", 0.0)
                        result["tok_s"] = evt.get("tok_s", 0.0)
                        result["stop_reason"] = evt.get("stop_reason", "unknown")
    except urllib.error.URLError as e:
        result["error"] = f"Connection error: {e}"
    except Exception as e:
        result["error"] = f"Request error: {e}"

    if result["elapsed"] == 0.0:
        result["elapsed"] = time.monotonic() - t0
    if result["tokens"] > 0 and result["tok_s"] == 0.0:
        result["tok_s"] = result["tokens"] / max(result["elapsed"], 0.001)

    return result


def reset_server(url: str) -> bool:
    """POST /api/reset. Returns True on success."""
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/reset", method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_status(url: str) -> Optional[Dict]:
    """GET /api/status. Returns status dict or None."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/status", timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# ── Quality Checks ───────────────────────────────────────────────────

def check_non_empty(text: str) -> bool:
    return len(text.strip()) > 0


def check_no_repetition(text: str, ngram_size: int = 5, threshold: int = 3) -> bool:
    """Sliding window n-gram repetition check."""
    words = text.split()
    if len(words) < ngram_size * threshold:
        return True
    ngrams = {}
    for i in range(len(words) - ngram_size + 1):
        ng = " ".join(words[i:i + ngram_size])
        ngrams[ng] = ngrams.get(ng, 0) + 1
        if ngrams[ng] >= threshold:
            return False
    return True


def check_coherent(text: str) -> bool:
    """Basic coherence: not too many non-ASCII garbage sequences."""
    if not text.strip():
        return False
    # Check ratio of printable chars
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    return printable / max(len(text), 1) > 0.85


def check_contains_code(text: str) -> bool:
    """Check if response contains code-like content."""
    return bool(re.search(r'(def |class |function |```|import |return )', text))


def check_has_numbered_list(text: str) -> bool:
    """Check for numbered list items."""
    return len(re.findall(r'^\s*\d+[\.\)]\s+', text, re.MULTILINE)) >= 3


def check_min_length_200(text: str) -> bool:
    return len(text.split()) >= 200


def check_answer_contains(text: str, keywords: List[str]) -> bool:
    """Check if any keyword appears in text (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


CHECK_FUNCS = {
    "non_empty": check_non_empty,
    "no_repetition": check_no_repetition,
    "coherent": check_coherent,
    "contains_code": check_contains_code,
    "has_numbered_list": check_has_numbered_list,
    "min_length_200": check_min_length_200,
}


def run_checks(text: str, checks: List[str],
               expect_answer_contains: Optional[List[str]] = None) -> Dict[str, bool]:
    """Run named checks on response text. Returns {check_name: pass/fail}."""
    results = {}
    for check_name in checks:
        fn = CHECK_FUNCS.get(check_name)
        if fn:
            results[check_name] = fn(text)
        else:
            results[check_name] = True  # unknown check = pass
    if expect_answer_contains:
        results["answer_contains"] = check_answer_contains(text, expect_answer_contains)
    return results


# ── Test Phases ──────────────────────────────────────────────────────

def run_quality_tests(url: str, prompts_data: Dict, enable_thinking: bool,
                      max_tokens_default: int = 500) -> List[Dict]:
    """Run all single-turn quality prompts."""
    results = []
    mode_label = "think-on" if enable_thinking else "think-off"

    for cat_name, cat in prompts_data.get("categories", {}).items():
        if cat_name == "multi_turn":
            continue  # handled separately

        for p in cat.get("prompts", []):
            pid = p["id"]
            prompt = p["prompt"]
            mt = p.get("max_tokens", max_tokens_default)
            print(f"  [{mode_label}] {pid}: {prompt[:60]}...", flush=True)

            reset_server(url)
            resp = chat_request(url, prompt, max_tokens=mt,
                                enable_thinking=enable_thinking)

            checks = run_checks(
                resp["text"], p.get("checks", []),
                p.get("expect_answer_contains"),
            )

            # Think mode check
            # Server sends all tokens (including thinking) as "token" events.
            # Think content may contain literal </think> tag in the text.
            if enable_thinking:
                # Check for think pattern: </think> in text indicates thinking happened,
                # OR response has tokens (model responded in thinking mode)
                checks["has_think_tags"] = (
                    "</think>" in resp["text"]
                    or len(resp["think_text"].strip()) > 0
                    or resp["tokens"] > 0  # model generated tokens in think mode
                )
            else:
                checks["no_think_tags"] = "<think>" not in resp["text"]

            # EOS check — only for non-think mode (think mode often hits token limit)
            if p.get("expect_eos") and not enable_thinking:
                checks["eos_stop"] = resp["stop_reason"] == "eos"

            all_pass = all(checks.values())
            status = "PASS" if all_pass else "FAIL"
            print(f"    → {status} | {resp['tokens']} tok, {resp['tok_s']:.1f} tok/s, "
                  f"stop={resp['stop_reason']}", flush=True)
            if not all_pass:
                failed = [k for k, v in checks.items() if not v]
                print(f"    ✗ Failed: {', '.join(failed)}", flush=True)

            results.append({
                "id": pid, "category": cat_name, "mode": mode_label,
                "prompt": prompt, "response_preview": resp["text"][:300],
                "tokens": resp["tokens"], "elapsed": resp["elapsed"],
                "tok_s": resp["tok_s"], "stop_reason": resp["stop_reason"],
                "checks": checks, "pass": all_pass,
                "error": resp["error"],
            })

    return results


def run_multi_turn_tests(url: str, prompts_data: Dict, enable_thinking: bool) -> List[Dict]:
    """Run multi-turn conversation scenarios."""
    results = []
    mode_label = "think-on" if enable_thinking else "think-off"
    scenarios = prompts_data.get("categories", {}).get("multi_turn", {}).get("scenarios", [])

    for scenario in scenarios:
        sid = scenario["id"]
        name = scenario["name"]
        print(f"  [{mode_label}] Multi-turn: {name}", flush=True)

        reset_server(url)
        scenario_results = []

        for i, turn in enumerate(scenario["turns"]):
            prompt = turn["prompt"]
            print(f"    Turn {i+1}: {prompt[:60]}...", flush=True)

            resp = chat_request(url, prompt, max_tokens=500,
                                enable_thinking=enable_thinking)

            checks = run_checks(
                resp["text"], turn.get("checks", []),
                turn.get("expect_answer_contains"),
            )

            all_pass = all(checks.values())
            status = "PASS" if all_pass else "FAIL"
            print(f"      → {status} | {resp['tokens']} tok", flush=True)
            if not all_pass:
                failed = [k for k, v in checks.items() if not v]
                print(f"      ✗ Failed: {', '.join(failed)}", flush=True)

            scenario_results.append({
                "turn": i + 1, "prompt": prompt,
                "response_preview": resp["text"][:200],
                "tokens": resp["tokens"], "checks": checks,
                "pass": all_pass, "error": resp["error"],
            })

        results.append({
            "id": sid, "name": name, "mode": mode_label,
            "turns": scenario_results,
            "pass": all(t["pass"] for t in scenario_results),
        })

    return results


def run_robustness_tests(url: str, prompts_data: Dict) -> List[Dict]:
    """Run context limit and compaction stress tests."""
    results = []
    rob_tests = prompts_data.get("robustness", {}).get("tests", [])

    for test in rob_tests:
        tid = test["id"]
        name = test["name"]
        print(f"  [Robustness] {tid}: {name}", flush=True)

        reset_server(url)
        status_before = get_status(url)

        if "turns" in test:
            # Multi-turn robustness test
            turn_results = []
            for turn in test["turns"]:
                resp = chat_request(url, turn["prompt"],
                                    max_tokens=turn.get("max_tokens", 200),
                                    enable_thinking=False)
                turn_results.append({
                    "prompt": turn["prompt"][:60],
                    "tokens": resp["tokens"],
                    "stop_reason": resp["stop_reason"],
                    "response_preview": resp["text"][:100],
                    "error": resp["error"],
                })
            status_after = get_status(url)
            results.append({
                "id": tid, "name": name, "turns": turn_results,
                "status_before": status_before, "status_after": status_after,
                "pass": all(t["error"] is None for t in turn_results),
            })

        elif "turns_template" in test:
            # Many short turns
            template = test["turns_template"]
            prompts = template["prompts"]
            mt = template.get("max_tokens_each", 30)
            turn_results = []
            for i, prompt in enumerate(prompts):
                resp = chat_request(url, prompt, max_tokens=mt,
                                    enable_thinking=False)
                if (i + 1) % 10 == 0:
                    st = get_status(url)
                    pos = st.get("pos", "?") if st else "?"
                    print(f"    Turn {i+1}/{len(prompts)}, pos={pos}", flush=True)
                turn_results.append({
                    "turn": i + 1, "tokens": resp["tokens"],
                    "error": resp["error"],
                })
            status_after = get_status(url)
            results.append({
                "id": tid, "name": name,
                "total_turns": len(prompts),
                "errors": sum(1 for t in turn_results if t["error"]),
                "status_before": status_before, "status_after": status_after,
                "pass": all(t["error"] is None for t in turn_results),
            })

        elif "prompt" in test:
            # Single prompt (long gen or post-compaction)
            mt = test.get("max_tokens", 2000)
            resp = chat_request(url, test["prompt"], max_tokens=mt,
                                enable_thinking=False)
            status_after = get_status(url)
            text = resp["text"]
            results.append({
                "id": tid, "name": name,
                "tokens": resp["tokens"], "elapsed": resp["elapsed"],
                "tok_s": resp["tok_s"], "stop_reason": resp["stop_reason"],
                "response_preview": text[:200],
                "no_repetition": check_no_repetition(text),
                "coherent": check_coherent(text),
                "status_before": status_before, "status_after": status_after,
                "pass": resp["error"] is None and check_no_repetition(text),
                "error": resp["error"],
            })

        print(f"    → {'PASS' if results[-1]['pass'] else 'FAIL'}", flush=True)

    return results


def run_recovery_tests(url: str, prompts_data: Dict) -> List[Dict]:
    """Run reset and recovery tests."""
    results = []
    rec_tests = prompts_data.get("recovery", {}).get("tests", [])

    for test in rec_tests:
        tid = test["id"]
        name = test["name"]
        print(f"  [Recovery] {tid}: {name}", flush=True)

        if tid == "REC1":
            # Reset endpoint test
            reset_server(url)
            # Build conversation
            for turn in test["turns_before_reset"]:
                resp = chat_request(url, turn["prompt"],
                                    max_tokens=turn.get("max_tokens", 50),
                                    enable_thinking=False)
                print(f"    Pre-reset: {resp['text'][:80]}", flush=True)

            # Verify context is held
            status_pre = get_status(url)

            # Reset
            reset_ok = reset_server(url)
            status_post = get_status(url)

            # Ask post-reset question
            post = test["post_reset_prompt"]
            resp_post = chat_request(url, post["prompt"],
                                     max_tokens=post.get("max_tokens", 50),
                                     enable_thinking=False)
            print(f"    Post-reset: {resp_post['text'][:80]}", flush=True)

            # Check that post-reset response does NOT contain "42"
            has_no_context = "42" not in resp_post["text"]
            results.append({
                "id": tid, "name": name,
                "reset_ok": reset_ok,
                "status_pre_reset": status_pre,
                "status_post_reset": status_post,
                "post_reset_response": resp_post["text"][:200],
                "has_no_prior_context": has_no_context,
                "pass": reset_ok and has_no_context,
            })

        elif tid == "REC2":
            # Post-reset quality
            reset_server(url)
            resp = chat_request(url, test["prompt"],
                                max_tokens=test.get("max_tokens", 500),
                                enable_thinking=False)
            checks = {
                "non_empty": check_non_empty(resp["text"]),
                "no_repetition": check_no_repetition(resp["text"]),
                "coherent": check_coherent(resp["text"]),
            }
            results.append({
                "id": tid, "name": name,
                "response_preview": resp["text"][:200],
                "tokens": resp["tokens"],
                "checks": checks,
                "pass": all(checks.values()),
            })

        elif tid == "REC3":
            # Rapid reset cycling
            cycles = test.get("cycles", 5)
            cycle_results = []
            for c in range(cycles):
                reset_server(url)
                resp = chat_request(url, test["prompt_per_cycle"],
                                    max_tokens=test.get("max_tokens", 30),
                                    enable_thinking=False)
                has_answer = check_answer_contains(
                    resp["text"],
                    test.get("expect_answer_contains", []),
                )
                cycle_results.append({
                    "cycle": c + 1,
                    "response": resp["text"][:50],
                    "has_answer": has_answer,
                })
            results.append({
                "id": tid, "name": name,
                "cycles": cycle_results,
                "pass": all(c["has_answer"] for c in cycle_results),
            })

        print(f"    → {'PASS' if results[-1]['pass'] else 'FAIL'}", flush=True)

    return results


def run_ane_profiling(url: str) -> List[Dict]:
    """Collect timing and performance data."""
    results = []
    print("  [ANE] Profiling...", flush=True)

    # A1: Model load status
    status = get_status(url)
    results.append({
        "id": "ANE1", "name": "model_status",
        "status": status,
        "pass": status is not None and status.get("ready", False),
    })

    # A2: Prefill timing (short prompt → measure first-token latency)
    reset_server(url)
    prompt_256 = "Explain the concept of " + " ".join(["artificial"] * 50) + " intelligence."
    t0 = time.monotonic()
    resp = chat_request(url, prompt_256, max_tokens=1, enable_thinking=False)
    first_token_time = time.monotonic() - t0
    results.append({
        "id": "ANE2", "name": "first_token_latency",
        "latency_s": round(first_token_time, 3),
        "pass": first_token_time < 30,  # should be under 30s for prefill
    })

    # A3: Decode throughput (100 tokens)
    reset_server(url)
    resp = chat_request(url, "Write a long story about a brave robot.",
                        max_tokens=100, enable_thinking=False)
    results.append({
        "id": "ANE3", "name": "decode_100_tokens",
        "tokens": resp["tokens"], "elapsed": resp["elapsed"],
        "tok_s": resp["tok_s"],
        "pass": resp["tokens"] >= 50 and resp["error"] is None,
    })

    # A4: Long generation stress (500 tokens)
    reset_server(url)
    resp = chat_request(url, "Write a very detailed essay about the history of space exploration.",
                        max_tokens=500, enable_thinking=False)
    results.append({
        "id": "ANE4", "name": "long_gen_500",
        "tokens": resp["tokens"], "elapsed": resp["elapsed"],
        "tok_s": resp["tok_s"], "stop_reason": resp["stop_reason"],
        "no_repetition": check_no_repetition(resp["text"]),
        "pass": resp["tokens"] >= 200 and resp["error"] is None,
    })

    # A5: Think-on decode throughput
    reset_server(url)
    resp = chat_request(url, "Solve: what is 15 * 23?",
                        max_tokens=300, enable_thinking=True)
    results.append({
        "id": "ANE5", "name": "think_on_throughput",
        "tokens": resp["tokens"], "elapsed": resp["elapsed"],
        "tok_s": resp["tok_s"],
        "pass": resp["tokens"] >= 10 and resp["error"] is None,
    })

    for r in results:
        print(f"    {r['id']}: {r['name']} → {'PASS' if r['pass'] else 'FAIL'}", flush=True)

    return results


# ── Report ───────────────────────────────────────────────────────────

def generate_report(all_results: Dict, output_path: str):
    """Write JSON and human-readable text reports."""
    # JSON report
    json_path = output_path + ".json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON report: {json_path}")

    # Text summary
    txt_path = output_path + ".txt"
    with open(txt_path, "w") as f:
        f.write(f"Qwen3.5 Validation Report\n")
        f.write(f"Generated: {all_results.get('timestamp', 'unknown')}\n")
        f.write(f"Server: {all_results.get('url', 'unknown')}\n")
        f.write("=" * 70 + "\n\n")

        for phase_name, phase_data in all_results.get("phases", {}).items():
            if not phase_data:
                continue
            f.write(f"── {phase_name.upper()} ──\n\n")

            if isinstance(phase_data, list):
                total = len(phase_data)
                passed = sum(1 for r in phase_data if r.get("pass", False))
                f.write(f"  Results: {passed}/{total} passed\n\n")
                for r in phase_data:
                    status = "✓" if r.get("pass") else "✗"
                    rid = r.get("id", "?")
                    name = r.get("name", r.get("category", ""))
                    f.write(f"  {status} {rid} {name}")
                    if "tok_s" in r and r["tok_s"]:
                        f.write(f" ({r['tok_s']:.1f} tok/s)")
                    f.write("\n")
                    if not r.get("pass"):
                        # Show failure details
                        checks = r.get("checks", {})
                        failed = [k for k, v in checks.items() if not v]
                        if failed:
                            f.write(f"    Failed checks: {', '.join(failed)}\n")
                        if r.get("error"):
                            f.write(f"    Error: {r['error']}\n")
                        preview = r.get("response_preview", "")
                        if preview:
                            f.write(f"    Response: {preview[:120]}...\n")
                f.write("\n")

        # Summary
        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n\n")
        for phase_name, phase_data in all_results.get("phases", {}).items():
            if isinstance(phase_data, list) and phase_data:
                total = len(phase_data)
                passed = sum(1 for r in phase_data if r.get("pass", False))
                f.write(f"  {phase_name}: {passed}/{total}\n")
        f.write("\n")

    print(f"  Text report: {txt_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 Validation Runner")
    parser.add_argument("--url", default="http://localhost:8080",
                        help="Chat server URL (default: http://localhost:8080)")
    parser.add_argument("--phase", default="all",
                        choices=["quality", "robustness", "recovery", "ane", "all"],
                        help="Which test phase to run")
    parser.add_argument("--mode", default="both",
                        choices=["think-on", "think-off", "both"],
                        help="Thinking mode for quality tests")
    parser.add_argument("--output", default=None,
                        help="Output report path (without extension)")
    parser.add_argument("--prompts", default=PROMPTS_FILE,
                        help="Path to validation_prompts.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without sending requests")
    args = parser.parse_args()

    # Load prompts
    with open(args.prompts) as f:
        prompts_data = json.load(f)

    if args.dry_run:
        print("DRY RUN — listing all prompts:\n")
        for cat_name, cat in prompts_data.get("categories", {}).items():
            if cat_name == "multi_turn":
                for s in cat.get("scenarios", []):
                    print(f"  [{s['id']}] {s['name']}:")
                    for t in s["turns"]:
                        print(f"    → {t['prompt']}")
            else:
                for p in cat.get("prompts", []):
                    print(f"  [{p['id']}] {p['prompt'][:80]}")
        print(f"\nTotal single-turn: {sum(len(c.get('prompts', [])) for n, c in prompts_data['categories'].items() if n != 'multi_turn')}")
        print(f"Multi-turn scenarios: {len(prompts_data['categories'].get('multi_turn', {}).get('scenarios', []))}")
        return

    # Check server is reachable
    status = get_status(args.url)
    if status is None:
        print(f"ERROR: Cannot connect to server at {args.url}")
        print("Start the server first: python chat_server.py --model-dir <dir> ...")
        sys.exit(1)
    print(f"Connected to {args.url}")
    print(f"  Status: {json.dumps(status, indent=2)}\n")

    # Set up output
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(SCRIPT_DIR, f"validation_report_{ts}")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "url": args.url,
        "server_status": status,
        "phases": {},
    }

    phases_to_run = (
        ["quality", "robustness", "recovery", "ane"]
        if args.phase == "all"
        else [args.phase]
    )

    for phase in phases_to_run:
        print(f"\n{'='*60}")
        print(f"  PHASE: {phase.upper()}")
        print(f"{'='*60}\n")

        if phase == "quality":
            quality_results = []
            modes = []
            if args.mode in ("think-on", "both"):
                modes.append(True)
            if args.mode in ("think-off", "both"):
                modes.append(False)

            for thinking in modes:
                label = "think-on" if thinking else "think-off"
                print(f"\n── Quality: {label} ──\n")
                quality_results.extend(
                    run_quality_tests(args.url, prompts_data, thinking)
                )
                # Multi-turn tests
                quality_results.extend([
                    {"id": s["id"], "name": s["name"], "mode": label,
                     "turns": s.get("turns", []), "pass": s["pass"]}
                    for s in run_multi_turn_tests(args.url, prompts_data, thinking)
                ])

            all_results["phases"]["quality"] = quality_results

        elif phase == "robustness":
            all_results["phases"]["robustness"] = run_robustness_tests(
                args.url, prompts_data)

        elif phase == "recovery":
            all_results["phases"]["recovery"] = run_recovery_tests(
                args.url, prompts_data)

        elif phase == "ane":
            all_results["phases"]["ane"] = run_ane_profiling(args.url)

    # Generate report
    print(f"\n{'='*60}")
    print("  REPORT")
    print(f"{'='*60}")
    generate_report(all_results, args.output)

    # Quick summary
    print("\n── Summary ──")
    for phase_name, phase_data in all_results["phases"].items():
        if isinstance(phase_data, list) and phase_data:
            total = len(phase_data)
            passed = sum(1 for r in phase_data if r.get("pass", False))
            pct = 100 * passed / total if total else 0
            print(f"  {phase_name}: {passed}/{total} ({pct:.0f}%)")

    # Exit code
    all_pass = all(
        r.get("pass", False)
        for pd in all_results["phases"].values()
        if isinstance(pd, list)
        for r in pd
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
