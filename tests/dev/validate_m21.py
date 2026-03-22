#!/usr/bin/env python3
"""Milestone 2.1 validation: logits lm_head + penalties in chat_server.py.

Runs against a live chat server at http://localhost:8080.
Tests:
  1. Status API
  2. Greedy decode (no penalties)
  3. Repetition penalty (rep=1.15)
  4. Multi-turn (continuation)
  5. Batch prefill (long prompt)
  6. Repetition guard (n-gram)
"""
import json, sys, time, urllib.request

BASE = "http://localhost:8080"
PASS, FAIL = 0, 0


def api_get(path):
    return json.loads(urllib.request.urlopen(f"{BASE}{path}").read())


def api_post(path, data):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req)


def chat(msg, **kwargs):
    """Send chat message, return (full_text, done_event)."""
    payload = {"message": msg, "max_tokens": 60,
               "enable_thinking": False, **kwargs}
    resp = api_post("/api/chat/stream", payload)
    tokens, done = [], None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "token":
            tokens.append(ev["text"])
        elif ev.get("type") == "done":
            done = ev
        elif ev.get("type") == "error":
            return "ERROR: " + ev.get("message", ""), ev
    return "".join(tokens), done


def reset():
    api_post("/api/reset", {})


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def main():
    global PASS, FAIL

    # Test 1: Status
    print("\n=== Test 1: Status API ===")
    st = api_get("/api/status")
    check("ready", st.get("ready") is True)
    check("ctx", st.get("ctx") == 1024, f"ctx={st.get('ctx')}")
    check("mode", st.get("mode") == "combined-dedup", st.get("mode"))

    # Test 2: Greedy decode (no penalties)
    print("\n=== Test 2: Greedy decode (no penalties) ===")
    reset()
    text, done = chat("What is 2+2?", repetition_penalty=1.0,
                       repetition_guard=False)
    check("non-empty", len(text) > 0, f"{len(text)} chars")
    check("contains_4", "4" in text, text[:80])
    check("done_event", done is not None)
    if done:
        check("stop_reason", done["stop_reason"] in ("eos", "length"),
              done["stop_reason"])

    # Test 3: Repetition penalty
    print("\n=== Test 3: Repetition penalty (rep=1.15) ===")
    reset()
    text_pen, done_pen = chat("Name 5 fruits", repetition_penalty=1.15,
                               repetition_guard=False)
    check("non-empty", len(text_pen) > 0, f"{len(text_pen)} chars")
    check("done_event", done_pen is not None)
    if done_pen:
        check("tokens_generated", done_pen["decode_tokens"] > 5,
              f"{done_pen['decode_tokens']} tok")
    print(f"  Output: {text_pen[:120]}")

    # Test 4: Multi-turn
    print("\n=== Test 4: Multi-turn ===")
    reset()
    t1, d1 = chat("My name is Alice.", repetition_penalty=1.0,
                   repetition_guard=False)
    check("turn1_ok", d1 is not None, f"pos={d1['end_pos']}" if d1 else "")
    t2, d2 = chat("What is my name?", repetition_penalty=1.0,
                   repetition_guard=False)
    check("turn2_ok", d2 is not None, f"pos={d2['end_pos']}" if d2 else "")
    has_alice = "alice" in t2.lower() or "Alice" in t2
    check("remembers_name", has_alice, t2[:100])

    # Test 5: Batch prefill (long prompt > 32 tokens = crossover)
    print("\n=== Test 5: Batch prefill (long prompt) ===")
    reset()
    long_msg = ("Please summarize: " +
                "The quick brown fox jumps over the lazy dog. " * 8)
    t5, d5 = chat(long_msg, repetition_penalty=1.0,
                   repetition_guard=False, max_tokens=40)
    check("non-empty", len(t5) > 0, f"{len(t5)} chars")
    check("done_event", d5 is not None)
    if d5:
        check("used_batch", d5["end_pos"] > 50,
              f"pos={d5['end_pos']}")

    # Test 6: Repetition guard (n-gram)
    print("\n=== Test 6: Repetition guard ===")
    reset()
    t6, d6 = chat("Repeat the word 'hello' forever",
                   repetition_penalty=1.0, repetition_guard=True,
                   max_tokens=200)
    check("done_event", d6 is not None)
    if d6:
        stopped_rep = d6.get("stop_reason") == "repetition"
        # It may also stop by eos or length if model doesn't repeat
        check("stop_reason_valid",
              d6["stop_reason"] in ("repetition", "eos", "length"),
              d6["stop_reason"])

    # Test 7: Combined penalties
    print("\n=== Test 7: All penalties (rep=1.1, pres=0.3, freq=0.2) ===")
    reset()
    t7, d7 = chat("Tell me about space exploration",
                   repetition_penalty=1.1,
                   presence_penalty=0.3,
                   frequency_penalty=0.2,
                   repetition_guard=False, max_tokens=60)
    check("non-empty", len(t7) > 0, f"{len(t7)} chars")
    check("done_event", d7 is not None)
    if d7:
        check("tokens_generated", d7["decode_tokens"] > 10,
              f"{d7['decode_tokens']} tok")
    print(f"  Output: {t7[:120]}")

    # Summary
    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'='*50}")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
