#!/usr/bin/env python3
"""Test cache overflow handling in chat_server.py.

Sends multiple turns to fill the 1024-token KV cache and verifies
that overflow is handled gracefully without crashing.
"""
import urllib.request
import json
import time
import sys

URL = "http://localhost:8080"

def chat(msg, max_tokens=100, enable_thinking=False):
    """Send a chat message and return the response text and metadata."""
    data = json.dumps({
        "message": msg,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
    }).encode()
    req = urllib.request.Request(
        f"{URL}/api/chat/stream",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=300)
    body = resp.read().decode()

    tokens = []
    meta = {}
    for line in body.strip().split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if ev["type"] == "token":
            tokens.append(ev["text"])
        elif ev["type"] == "done":
            meta = ev
        elif ev["type"] == "error":
            return f"ERROR: {ev['message']}", ev
    text = "".join(tokens)
    return text, meta

def status():
    req = urllib.request.Request(f"{URL}/api/status")
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read().decode())

def reset():
    req = urllib.request.Request(f"{URL}/api/reset", method="POST")
    urllib.request.urlopen(req, timeout=10)

# ── Test ──

print("=" * 60)
print("Cache overflow test")
print("=" * 60)

reset()
s = status()
print(f"\nAfter reset: pos={s['pos']}, turns={s['turns']}")

questions = [
    "What is the capital of France?",
    "Tell me about the Eiffel Tower in three sentences.",
    "What language do they speak there?",
    "Name three famous French painters.",
    "What is French cuisine known for?",
    "Tell me about the French Revolution briefly.",
    "Who was Napoleon Bonaparte?",
    "What are the major rivers in France?",
    "Describe the French Alps.",
    "What is the population of Paris?",
    "Tell me about French wine regions.",
    "What sports are popular in France?",
]

for i, q in enumerate(questions):
    s = status()
    print(f"\n--- Turn {i+1}: \"{q[:50]}\" ---")
    print(f"  Before: pos={s['pos']}/{s['ctx']} ({s['cache_pct']}%), "
          f"turns={s['turns']}")

    t0 = time.time()
    text, meta = chat(q, max_tokens=80, enable_thinking=False)
    elapsed = time.time() - t0

    if isinstance(meta, dict) and "end_pos" in meta:
        print(f"  After:  pos={meta['end_pos']}, "
              f"decode={meta['decode_tokens']} tok, "
              f"elapsed={elapsed:.1f}s")
        # Show first 60 chars of response
        clean = text.strip()
        if clean.startswith("<think>"):
            end = clean.find("</think>")
            if end >= 0:
                clean = clean[end+8:].strip()
        print(f"  Response: \"{clean[:80]}...\"" if len(clean) > 80
              else f"  Response: \"{clean}\"")
    else:
        print(f"  Result: {text[:100]}")

    if "ERROR" in str(text):
        print("  *** Got error, but server didn't crash! ***")

s = status()
print(f"\n{'='*60}")
print(f"Final: pos={s['pos']}/{s['ctx']} ({s['cache_pct']}%), "
      f"turns={s['turns']}")
print(f"Server is {'ready' if s['ready'] else 'NOT ready'}")
print("=" * 60)
