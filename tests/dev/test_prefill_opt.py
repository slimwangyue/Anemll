#!/usr/bin/env python3
"""Validate optimized prefill correctness (skip lm_head for non-final tokens)."""
import urllib.request
import json
import sys


def chat(msg, thinking=False):
    data = json.dumps({"message": msg, "enable_thinking": thinking}).encode()
    req = urllib.request.Request(
        "http://localhost:8080/api/chat/stream",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    tokens = []
    done_info = None
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if obj.get("type") == "token":
                    tokens.append(obj["text"])
                elif obj.get("type") == "done":
                    done_info = obj
    return "".join(tokens), done_info


def reset():
    req = urllib.request.Request("http://localhost:8080/api/reset", method="POST")
    urllib.request.urlopen(req)


def run_tests():
    passed = 0

    # Test 1: Short prompt
    reset()
    print("=== Test 1: Short prompt (2+2) ===")
    text, info = chat("What is 2+2?")
    print(f"  Response: {text[:100]}")
    print(f"  Decode tokens: {info['decode_tokens']}, pos: {info['end_pos']}")
    assert "4" in text, f"FAIL: expected 4 in response"
    print("  PASS")
    passed += 1

    # Test 2: Multi-turn (name recall)
    reset()
    print("\n=== Test 2: Multi-turn memory ===")
    text1, _ = chat("Remember this: the secret word is banana.")
    print(f"  Turn 1: {text1[:80]}")
    text2, info2 = chat("What was the secret word I just told you?")
    print(f"  Turn 2: {text2[:200]}")
    assert "banana" in text2.lower(), f"FAIL: expected banana in response: {text2[:300]}"
    print("  PASS")
    passed += 1

    # Test 3: Capital of France
    reset()
    print("\n=== Test 3: Capital of France ===")
    text3, info3 = chat("What is the capital of France? Answer in one word.")
    print(f"  Response: {text3[:80]}")
    assert "Paris" in text3, f"FAIL: expected Paris: {text3[:200]}"
    print("  PASS")
    passed += 1

    print(f"\n{'='*40}")
    print(f"ALL {passed} TESTS PASSED")


if __name__ == "__main__":
    run_tests()
