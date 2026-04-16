#!/usr/bin/env python3
"""
Systematic sampling parameter sweep for Qwen3.5 chat_server.py.

Measures repetition, coherence, and quality across diverse prompts
to find optimal sampling configs for think-on and think-off modes.

Usage:
    python scripts_qwen3_5/sweep_sampling.py --url http://localhost:8080 --stage baseline
    python scripts_qwen3_5/sweep_sampling.py --url http://localhost:8080 --stage freq
    python scripts_qwen3_5/sweep_sampling.py --url http://localhost:8080 --stage rep
    python scripts_qwen3_5/sweep_sampling.py --url http://localhost:8080 --stage pres
    python scripts_qwen3_5/sweep_sampling.py --url http://localhost:8080 --stage temp
    python scripts_qwen3_5/sweep_sampling.py --url http://localhost:8080 --stage final
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── SSE Client (reused from run_validation.py) ──────────────────────

def chat_request(url: str, message: str, max_tokens: int = 500,
                 enable_thinking: bool = False, timeout: int = 300,
                 **sampling_kwargs) -> Dict[str, Any]:
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
    except Exception as e:
        result["error"] = str(e)
    if result["elapsed"] == 0.0:
        result["elapsed"] = time.monotonic() - t0
    if result["tokens"] > 0 and result["tok_s"] == 0.0:
        result["tok_s"] = result["tokens"] / max(result["elapsed"], 0.001)
    return result


def reset_server(url: str) -> bool:
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/reset", method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Repetition Scoring ───────────────────────────────────────────────

def compute_repetition_score(text: str) -> Dict[str, Any]:
    """Compute a detailed repetition analysis of generated text.

    Returns dict with:
      - rep_score: float 0-1 (0=no repetition, 1=severe)
      - ngram_3_dup: count of 3-gram duplicates
      - ngram_5_dup: count of 5-gram duplicates
      - ngram_8_dup: count of 8-gram duplicates
      - max_ngram_repeat: highest repeat count of any 5-gram
      - line_dup_ratio: fraction of duplicate lines
      - longest_loop: length of longest exactly-repeated phrase
      - has_obvious_loop: bool
      - detail: string summary
    """
    words = text.split()
    n_words = len(words)

    # N-gram analysis
    def count_dup_ngrams(words, n):
        if len(words) < n:
            return 0, 0
        ngrams = Counter()
        for i in range(len(words) - n + 1):
            ng = " ".join(words[i:i + n])
            ngrams[ng] += 1
        dup_count = sum(c - 1 for c in ngrams.values() if c > 1)
        max_repeat = max(ngrams.values()) if ngrams else 0
        return dup_count, max_repeat

    ng3_dup, ng3_max = count_dup_ngrams(words, 3)
    ng5_dup, ng5_max = count_dup_ngrams(words, 5)
    ng8_dup, ng8_max = count_dup_ngrams(words, 8)

    # Line-level duplication
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    n_lines = len(lines)
    unique_lines = len(set(lines))
    line_dup_ratio = 1.0 - (unique_lines / max(n_lines, 1))

    # Longest repeated phrase (sliding window, 5-20 words)
    longest_loop = 0
    for size in range(5, min(21, n_words // 2 + 1)):
        ngrams = Counter()
        for i in range(n_words - size + 1):
            ng = " ".join(words[i:i + size])
            ngrams[ng] += 1
        max_c = max(ngrams.values()) if ngrams else 0
        if max_c >= 3:
            longest_loop = max(longest_loop, size)

    # Detect consecutive identical sentences
    sentences = re.split(r'[.!?。！？]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    consec_dup = 0
    for i in range(1, len(sentences)):
        if sentences[i] == sentences[i - 1]:
            consec_dup += 1

    # Composite score (0-1)
    # Weighted formula: heavier on 5-gram and 8-gram which indicate real loops
    if n_words < 10:
        rep_score = 0.0
    else:
        # Normalize per 100 words
        ng5_rate = (ng5_dup / max(n_words - 4, 1)) * 100
        ng8_rate = (ng8_dup / max(n_words - 7, 1)) * 100
        rep_score = min(1.0,
            0.15 * min(ng5_rate / 5.0, 1.0) +  # 5-gram rate
            0.30 * min(ng8_rate / 3.0, 1.0) +  # 8-gram rate (heavier)
            0.20 * line_dup_ratio +              # line duplication
            0.20 * min(longest_loop / 15.0, 1.0) +  # loop length
            0.15 * min(consec_dup / 3.0, 1.0)   # consecutive sentence dups
        )

    has_loop = (ng8_max >= 3) or (longest_loop >= 10) or (consec_dup >= 2)

    return {
        "rep_score": round(rep_score, 4),
        "ngram_3_dup": ng3_dup,
        "ngram_5_dup": ng5_dup,
        "ngram_8_dup": ng8_dup,
        "max_5gram_repeat": ng5_max,
        "max_8gram_repeat": ng8_max,
        "line_dup_ratio": round(line_dup_ratio, 4),
        "longest_loop": longest_loop,
        "consec_sentence_dup": consec_dup,
        "has_obvious_loop": has_loop,
        "n_words": n_words,
    }


def check_coherent(text: str) -> bool:
    if not text.strip():
        return False
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    return printable / max(len(text), 1) > 0.85


def check_on_topic(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# ── Test Prompt Suite ────────────────────────────────────────────────

# FULL prompt set (for final validation)
SWEEP_PROMPTS_FULL = {
    "S1_en_short": {
        "prompt": "What are the main differences between Python and Rust?",
        "max_tokens": 500,
        "lang": "en", "category": "short",
        "topic_kw": ["python", "rust"],
    },
    "S2_en_reason": {
        "prompt": "A farmer has 17 sheep. All but 9 die. How many are left?",
        "max_tokens": 200,
        "lang": "en", "category": "reasoning",
        "topic_kw": ["9"],
    },
    "S3_en_code": {
        "prompt": "Write a Python function that checks if a number is prime.",
        "max_tokens": 500,
        "lang": "en", "category": "coding",
        "topic_kw": ["def", "prime"],
    },
    "S4_en_list": {
        "prompt": "List 10 tips for writing clean code, numbered 1-10.",
        "max_tokens": 500,
        "lang": "en", "category": "instruction",
        "topic_kw": ["1.", "2.", "3."],
    },
    "L1_en_long": {
        "prompt": "Write a detailed comparison of TCP vs UDP, covering at least 5 aspects including use cases, advantages, and trade-offs.",
        "max_tokens": 800,
        "lang": "en", "category": "long",
        "topic_kw": ["tcp", "udp"],
    },
    "L2_en_long": {
        "prompt": "Explain the complete lifecycle of an HTTP request from when a user types a URL to when the page is rendered, covering DNS, TCP handshake, TLS, HTTP, and rendering.",
        "max_tokens": 800,
        "lang": "en", "category": "long",
        "topic_kw": ["dns", "tcp", "http"],
    },
    "CH1_zh_short": {
        "prompt": "请解释什么是人工智能，以及它在日常生活中的应用。",
        "max_tokens": 500,
        "lang": "zh", "category": "short",
        "topic_kw": ["人工智能", "AI", "应用"],
    },
    "CH2_zh_long": {
        "prompt": "请详细比较Python和Java两种编程语言的优缺点，至少从5个方面进行分析，包括性能、生态、学习难度、应用场景和社区支持。",
        "max_tokens": 800,
        "lang": "zh", "category": "long",
        "topic_kw": ["python", "java", "Python", "Java"],
    },
    "REP1_history": {
        "prompt": "What happened during the French Revolution? Give a comprehensive overview.",
        "max_tokens": 600,
        "lang": "en", "category": "long",
        "topic_kw": ["revolution", "france", "french"],
    },
    "REP2_analysis": {
        "prompt": "Please analyze the development of artificial intelligence from the 1950s to today, covering major milestones and breakthroughs in chronological order.",
        "max_tokens": 800,
        "lang": "en", "category": "long",
        "topic_kw": ["ai", "artificial", "intelligence"],
    },
}

# FAST probe set: 4 key prompts covering short, long, Chinese, repetition-prone
# Used for parameter scanning to keep sweep time reasonable (~6 tok/s ANE)
SWEEP_PROMPTS_FAST = {
    "F1_en_short": {
        "prompt": "What are the main differences between Python and Rust?",
        "max_tokens": 400,
        "lang": "en", "category": "short",
        "topic_kw": ["python", "rust"],
    },
    "F2_en_long": {
        "prompt": "Write a detailed comparison of TCP vs UDP, covering at least 5 aspects including use cases, advantages, and trade-offs.",
        "max_tokens": 600,
        "lang": "en", "category": "long",
        "topic_kw": ["tcp", "udp"],
    },
    "F3_zh_long": {
        "prompt": "请详细比较Python和Java两种编程语言的优缺点，至少从5个方面进行分析。",
        "max_tokens": 600,
        "lang": "zh", "category": "long",
        "topic_kw": ["python", "java", "Python", "Java"],
    },
    "F4_rep_stress": {
        "prompt": "Please analyze the development of artificial intelligence from the 1950s to today, covering major milestones and breakthroughs in chronological order.",
        "max_tokens": 600,
        "lang": "en", "category": "long",
        "topic_kw": ["ai", "artificial", "intelligence"],
    },
}

# PROBE set: ultra-fast for parameter scanning (~3 min per config at 7 tok/s)
# 3 prompts × 250 tok + 3 multi-turn × 200 tok ≈ 1350 tok ≈ 3.2 min
SWEEP_PROMPTS_PROBE = {
    "P1_en_long": {
        "prompt": "Write a detailed comparison of TCP vs UDP, covering use cases and trade-offs.",
        "max_tokens": 250,
        "lang": "en", "category": "long",
        "topic_kw": ["tcp", "udp"],
    },
    "P2_zh_long": {
        "prompt": "请详细比较Python和Java两种编程语言的优缺点，至少从5个方面进行分析。",
        "max_tokens": 250,
        "lang": "zh", "category": "long",
        "topic_kw": ["python", "java", "Python", "Java"],
    },
    "P3_rep_stress": {
        "prompt": "Please analyze the development of artificial intelligence from the 1950s to today, covering major milestones.",
        "max_tokens": 250,
        "lang": "en", "category": "long",
        "topic_kw": ["ai", "artificial", "intelligence"],
    },
}

MULTI_TURN_PROBE = [
    {"prompt": "What is a derivative in calculus?", "max_tokens": 200},
    {"prompt": "Now explain the chain rule with an example.", "max_tokens": 250},
    {"prompt": "Can you also explain integration by parts?", "max_tokens": 300},
]

# Default prompt set used in sweeps
SWEEP_PROMPTS = SWEEP_PROMPTS_FAST

# Multi-turn scenario (accumulates context, tests multi-turn repetition)
MULTI_TURN_SCENARIO = [
    {"prompt": "What is a derivative in calculus?", "max_tokens": 400},
    {"prompt": "How do I find the derivative of x squared?", "max_tokens": 400},
    {"prompt": "Now explain the chain rule with an example.", "max_tokens": 500},
    {"prompt": "Can you also explain integration by parts?", "max_tokens": 500},
]


# ── Sweep Runner ─────────────────────────────────────────────────────

def run_single_prompt(url, prompt_id, prompt_info, enable_thinking, sampling,
                      verbose=False):
    """Run a single prompt with given sampling params. Returns result dict."""
    msg = prompt_info["prompt"]
    max_tok = prompt_info["max_tokens"]
    topic_kw = prompt_info.get("topic_kw", [])

    resp = chat_request(url, msg, max_tokens=max_tok,
                        enable_thinking=enable_thinking, **sampling)

    text = resp["text"]
    rep = compute_repetition_score(text)
    coherent = check_coherent(text)
    on_topic = check_on_topic(text, topic_kw) if topic_kw else True

    result = {
        "id": prompt_id,
        "tokens": resp["tokens"],
        "tok_s": round(resp["tok_s"], 1),
        "stop": resp["stop_reason"],
        "rep_score": rep["rep_score"],
        "has_loop": rep["has_obvious_loop"],
        "ng5_dup": rep["ngram_5_dup"],
        "ng8_dup": rep["ngram_8_dup"],
        "max_5g": rep["max_5gram_repeat"],
        "line_dup": rep["line_dup_ratio"],
        "longest_loop": rep["longest_loop"],
        "coherent": coherent,
        "on_topic": on_topic,
        "error": resp["error"],
    }

    if verbose:
        flag = "LOOP" if rep["has_obvious_loop"] else ("warn" if rep["rep_score"] > 0.15 else "ok")
        print(f"  {prompt_id:16s} tok={resp['tokens']:4d} stop={resp['stop_reason']:6s} "
              f"rep={rep['rep_score']:.3f} ng5={rep['ngram_5_dup']:3d} ng8={rep['ngram_8_dup']:3d} "
              f"loop={rep['longest_loop']:2d} [{flag}]"
              f"{'  !! '+resp['error'] if resp['error'] else ''}")

    return result


ACTIVE_MULTI_TURN = MULTI_TURN_SCENARIO

def run_multi_turn(url, enable_thinking, sampling, verbose=False):
    """Run multi-turn scenario, measuring per-turn repetition."""
    results = []
    for i, turn in enumerate(ACTIVE_MULTI_TURN):
        resp = chat_request(url, turn["prompt"], max_tokens=turn["max_tokens"],
                            enable_thinking=enable_thinking, **sampling)
        rep = compute_repetition_score(resp["text"])
        r = {
            "turn": i + 1,
            "tokens": resp["tokens"],
            "tok_s": round(resp["tok_s"], 1),
            "stop": resp["stop_reason"],
            "rep_score": rep["rep_score"],
            "has_loop": rep["has_obvious_loop"],
            "ng5_dup": rep["ngram_5_dup"],
            "ng8_dup": rep["ngram_8_dup"],
            "longest_loop": rep["longest_loop"],
            "coherent": check_coherent(resp["text"]),
            "error": resp["error"],
        }
        results.append(r)
        if verbose:
            flag = "LOOP" if rep["has_obvious_loop"] else ("warn" if rep["rep_score"] > 0.15 else "ok")
            print(f"  MT turn {i+1}/{len(ACTIVE_MULTI_TURN)} tok={resp['tokens']:4d} "
                  f"rep={rep['rep_score']:.3f} ng5={rep['ngram_5_dup']:3d} [{flag}]")
    return results


def run_config(url, config_name, sampling, enable_thinking, verbose=True):
    """Run full suite for one config. Returns aggregate results."""
    mode = "think-on" if enable_thinking else "think-off"
    print(f"\n{'='*70}")
    print(f"Config: {config_name}  mode={mode}")
    print(f"Params: {sampling}")
    print(f"{'='*70}")

    # Reset before each config
    reset_server(url)
    time.sleep(0.5)

    # Single-turn tests
    st_results = []
    for pid, pinfo in SWEEP_PROMPTS.items():
        reset_server(url)
        time.sleep(0.3)
        r = run_single_prompt(url, pid, pinfo, enable_thinking, sampling, verbose)
        st_results.append(r)

    # Multi-turn test
    reset_server(url)
    time.sleep(0.3)
    print(f"\n  -- Multi-turn scenario --")
    mt_results = run_multi_turn(url, enable_thinking, sampling, verbose)

    # Aggregate
    all_rep_scores = [r["rep_score"] for r in st_results]
    mt_rep_scores = [r["rep_score"] for r in mt_results]
    n_loops_st = sum(1 for r in st_results if r["has_loop"])
    n_loops_mt = sum(1 for r in mt_results if r["has_loop"])
    n_coherent = sum(1 for r in st_results if r["coherent"])
    n_on_topic = sum(1 for r in st_results if r["on_topic"])

    agg = {
        "config_name": config_name,
        "mode": mode,
        "sampling": sampling,
        "n_single_turn": len(st_results),
        "n_multi_turn": len(mt_results),
        "avg_rep_score_st": round(sum(all_rep_scores) / max(len(all_rep_scores), 1), 4),
        "max_rep_score_st": round(max(all_rep_scores) if all_rep_scores else 0.0, 4),
        "loops_st": n_loops_st,
        "avg_rep_score_mt": round(sum(mt_rep_scores) / max(len(mt_rep_scores), 1), 4),
        "max_rep_score_mt": round(max(mt_rep_scores) if mt_rep_scores else 0.0, 4),
        "loops_mt": n_loops_mt,
        "coherent_ratio": round(n_coherent / max(len(st_results), 1), 3),
        "on_topic_ratio": round(n_on_topic / max(len(st_results), 1), 3),
        "total_loops": n_loops_st + n_loops_mt,
        "single_turn_details": st_results,
        "multi_turn_details": mt_results,
    }

    print(f"\n  Summary: avg_rep_st={agg['avg_rep_score_st']:.4f} "
          f"max_rep_st={agg['max_rep_score_st']:.4f} "
          f"loops_st={n_loops_st}/{len(st_results)} "
          f"avg_rep_mt={agg['avg_rep_score_mt']:.4f} "
          f"loops_mt={n_loops_mt}/{len(mt_results)} "
          f"coherent={agg['coherent_ratio']:.3f} "
          f"on_topic={agg['on_topic_ratio']:.3f}")

    return agg


# ── Stage Definitions ────────────────────────────────────────────────

def get_baseline_configs():
    """Official Qwen recommendations as baseline. Think-off first (faster)."""
    return {
        "baseline_think_off": {
            "sampling": {
                "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "presence_penalty": 1.5, "repetition_penalty": 1.0,
                "frequency_penalty": 0.0,
            },
            "enable_thinking": False,
        },
        "baseline_think_on": {
            "sampling": {
                "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "presence_penalty": 1.5, "repetition_penalty": 1.0,
                "frequency_penalty": 0.0,
            },
            "enable_thinking": True,
        },
    }


def get_freq_sweep_configs(enable_thinking):
    """Frequency penalty sweep, other params held at baseline."""
    mode = "on" if enable_thinking else "off"
    base = get_baseline_configs()[f"baseline_think_{mode}"]["sampling"].copy()
    configs = {}
    for fp in [0.0, 0.02, 0.05, 0.08, 0.1]:
        name = f"freq_{fp}_think_{mode}"
        s = base.copy()
        s["frequency_penalty"] = fp
        configs[name] = {"sampling": s, "enable_thinking": enable_thinking}
    return configs


def get_rep_sweep_configs(enable_thinking, best_freq=0.0):
    """Repetition penalty sweep with best freq penalty applied."""
    mode = "on" if enable_thinking else "off"
    base = get_baseline_configs()[f"baseline_think_{mode}"]["sampling"].copy()
    base["frequency_penalty"] = best_freq
    configs = {}
    for rp in [1.0, 1.02, 1.05, 1.08, 1.1, 1.12]:
        name = f"rep_{rp}_freq_{best_freq}_think_{mode}"
        s = base.copy()
        s["repetition_penalty"] = rp
        configs[name] = {"sampling": s, "enable_thinking": enable_thinking}
    return configs


def get_pres_sweep_configs(enable_thinking, best_freq=0.0, best_rep=1.0):
    """Presence penalty sweep with best freq+rep applied."""
    mode = "on" if enable_thinking else "off"
    base = get_baseline_configs()[f"baseline_think_{mode}"]["sampling"].copy()
    base["frequency_penalty"] = best_freq
    base["repetition_penalty"] = best_rep
    configs = {}
    for pp in [0.0, 0.5, 1.0, 1.5, 2.0]:
        name = f"pres_{pp}_freq_{best_freq}_rep_{best_rep}_think_{mode}"
        s = base.copy()
        s["presence_penalty"] = pp
        configs[name] = {"sampling": s, "enable_thinking": enable_thinking}
    return configs


def get_temp_sweep_configs(enable_thinking, best_freq=0.0, best_rep=1.0, best_pres=1.5):
    """Temperature/top_p/top_k sweep with best penalties applied."""
    mode = "on" if enable_thinking else "off"
    base = get_baseline_configs()[f"baseline_think_{mode}"]["sampling"].copy()
    base["frequency_penalty"] = best_freq
    base["repetition_penalty"] = best_rep
    base["presence_penalty"] = best_pres
    configs = {}

    # Temperature sweep
    temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if enable_thinking else [0.5, 0.6, 0.7, 0.8]
    for t in temps:
        name = f"temp_{t}_think_{mode}"
        s = base.copy()
        s["temperature"] = t
        configs[name] = {"sampling": s, "enable_thinking": enable_thinking}

    # Top-p sweep (best temp to be determined, use baseline default)
    for tp in [0.7, 0.8, 0.9, 0.95]:
        name = f"topp_{tp}_think_{mode}"
        s = base.copy()
        s["top_p"] = tp
        configs[name] = {"sampling": s, "enable_thinking": enable_thinking}

    return configs


# ── Report Generation ────────────────────────────────────────────────

def make_comparison_table(results: List[Dict]) -> str:
    """Generate a compact comparison table from sweep results."""
    lines = []
    header = (f"{'Config':<45s} {'AvgRep':>7s} {'MaxRep':>7s} "
              f"{'Loops':>5s} {'MTAvg':>7s} {'MTMax':>7s} "
              f"{'MTLoop':>6s} {'Coher':>6s} {'Topic':>6s}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        line = (f"{r['config_name']:<45s} "
                f"{r['avg_rep_score_st']:>7.4f} {r['max_rep_score_st']:>7.4f} "
                f"{r['loops_st']:>5d} {r['avg_rep_score_mt']:>7.4f} "
                f"{r['max_rep_score_mt']:>7.4f} {r['loops_mt']:>6d} "
                f"{r['coherent_ratio']:>6.3f} {r['on_topic_ratio']:>6.3f}")
        lines.append(line)
    return "\n".join(lines)


def save_results(results: List[Dict], output_path: str):
    """Save full results to JSON and summary to text."""
    # JSON (strip response text to save space, keep scores)
    with open(output_path + ".json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Text summary
    with open(output_path + ".txt", "w") as f:
        f.write(f"Sampling Sweep Results - {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(make_comparison_table(results) + "\n\n")

        # Per-config details
        for r in results:
            f.write(f"\n--- {r['config_name']} ({r['mode']}) ---\n")
            f.write(f"Params: {json.dumps(r['sampling'])}\n")
            f.write(f"Single-turn: avg_rep={r['avg_rep_score_st']:.4f} "
                    f"max_rep={r['max_rep_score_st']:.4f} loops={r['loops_st']}\n")
            for st in r["single_turn_details"]:
                flag = "LOOP" if st["has_loop"] else ("WARN" if st["rep_score"] > 0.15 else "OK")
                f.write(f"  {st['id']:16s} tok={st['tokens']:4d} "
                        f"rep={st['rep_score']:.3f} ng5={st['ng5_dup']:3d} "
                        f"loop={st['longest_loop']:2d} [{flag}]\n")
            f.write(f"Multi-turn: avg_rep={r['avg_rep_score_mt']:.4f} "
                    f"max_rep={r['max_rep_score_mt']:.4f} loops={r['loops_mt']}\n")
            for mt in r["multi_turn_details"]:
                flag = "LOOP" if mt["has_loop"] else ("WARN" if mt["rep_score"] > 0.15 else "OK")
                f.write(f"  Turn {mt['turn']} tok={mt['tokens']:4d} "
                        f"rep={mt['rep_score']:.3f} ng5={mt['ng5_dup']:3d} [{flag}]\n")

    print(f"\nSaved: {output_path}.json, {output_path}.txt")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sampling parameter sweep")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--stage", required=True,
                        choices=["baseline", "freq", "rep", "pres", "temp", "final",
                                 "freq_on", "freq_off", "rep_on", "rep_off",
                                 "pres_on", "pres_off", "temp_on", "temp_off",
                                 "final_on", "final_off"])
    parser.add_argument("--output", default=None,
                        help="Output path prefix (default: sweep_<stage>)")
    parser.add_argument("--best-freq", type=float, default=0.0,
                        help="Best frequency_penalty from prior stage")
    parser.add_argument("--best-rep", type=float, default=1.0,
                        help="Best repetition_penalty from prior stage")
    parser.add_argument("--best-pres", type=float, default=1.5,
                        help="Best presence_penalty from prior stage")
    parser.add_argument("--best-temp", type=float, default=None,
                        help="Best temperature from prior stage")
    parser.add_argument("--custom-config", type=str, default=None,
                        help="JSON string with custom sampling config to test")
    parser.add_argument("--think-on", action="store_true", default=False,
                        help="Run think-on mode (default think-off)")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Use full prompt set (default: fast 4-prompt probe)")
    parser.add_argument("--probe", action="store_true", default=False,
                        help="Use ultra-fast probe set (3 ST + 3 MT, ~3 min/config)")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Select prompt set
    global SWEEP_PROMPTS, ACTIVE_MULTI_TURN
    if args.full:
        SWEEP_PROMPTS = SWEEP_PROMPTS_FULL
        ACTIVE_MULTI_TURN = MULTI_TURN_SCENARIO
        print("Using FULL prompt set (10 prompts)")
    elif args.probe:
        SWEEP_PROMPTS = SWEEP_PROMPTS_PROBE
        ACTIVE_MULTI_TURN = MULTI_TURN_PROBE
        print("Using PROBE set (3 ST + 3 MT, ~3 min/config)")
    else:
        SWEEP_PROMPTS = SWEEP_PROMPTS_FAST
        ACTIVE_MULTI_TURN = MULTI_TURN_SCENARIO
        print("Using FAST probe set (4 prompts)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output = args.output or os.path.join(script_dir, f"sweep_{args.stage}")

    # Build config set based on stage
    configs = {}

    if args.stage == "baseline":
        configs = get_baseline_configs()

    elif args.stage in ("freq", "freq_on", "freq_off"):
        think = args.stage == "freq_on" or (args.stage == "freq" and args.think_on)
        configs = get_freq_sweep_configs(think)

    elif args.stage in ("rep", "rep_on", "rep_off"):
        think = args.stage == "rep_on" or (args.stage == "rep" and args.think_on)
        configs = get_rep_sweep_configs(think, best_freq=args.best_freq)

    elif args.stage in ("pres", "pres_on", "pres_off"):
        think = args.stage == "pres_on" or (args.stage == "pres" and args.think_on)
        configs = get_pres_sweep_configs(think, best_freq=args.best_freq,
                                         best_rep=args.best_rep)

    elif args.stage in ("temp", "temp_on", "temp_off"):
        think = args.stage == "temp_on" or (args.stage == "temp" and args.think_on)
        configs = get_temp_sweep_configs(think, best_freq=args.best_freq,
                                         best_rep=args.best_rep,
                                         best_pres=args.best_pres)

    elif args.stage in ("final", "final_on", "final_off"):
        # Final stage: test a custom config
        if args.custom_config:
            cfg = json.loads(args.custom_config)
            think = args.stage == "final_on" or (args.stage == "final" and args.think_on)
            configs["final_custom"] = {"sampling": cfg, "enable_thinking": think}
        else:
            print("ERROR: --custom-config required for final stage")
            sys.exit(1)

    if not configs:
        print("No configs to test. Check stage and flags.")
        sys.exit(1)

    print(f"Sweep: {args.stage} — {len(configs)} configs to test")
    print(f"Server: {args.url}")
    print(f"Output: {output}")

    all_results = []
    for name, cfg in configs.items():
        result = run_config(args.url, name, cfg["sampling"],
                           cfg["enable_thinking"], verbose=args.verbose)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(make_comparison_table(all_results))

    save_results(all_results, output)
    print(f"\nDone! {len(configs)} configs tested.")


if __name__ == "__main__":
    main()
