#!/usr/bin/env python3
"""Shared exact-prefill bucket helpers for Qwen3.5 chunk models."""

from __future__ import annotations

from typing import Iterable

PREFILL_BUCKETS_DESC = (256, 128, 64, 32)
PREFILL_CROSSOVER = 32
EXACT_PREFILL_CTX = 2048


def bucket_function_name(bucket: int) -> str:
    return f"prefill_bs{bucket}"


def parse_prefill_bucket_name(name: str) -> int | None:
    if not name.startswith("prefill_bs"):
        return None
    suffix = name.replace("prefill_bs", "", 1)
    return int(suffix) if suffix.isdigit() else None


def available_prefill_buckets(function_names: Iterable[str]) -> list[int]:
    buckets = []
    for name in function_names:
        bucket = parse_prefill_bucket_name(name)
        if bucket is not None:
            buckets.append(bucket)
    return sorted(set(buckets), reverse=True)


def greedy_prefill_plan(token_count: int, buckets: Iterable[int]) -> tuple[list[int], int]:
    ordered = sorted(set(int(b) for b in buckets), reverse=True)
    plan: list[int] = []
    remaining = token_count
    while remaining > 0:
        match = next((bucket for bucket in ordered if bucket <= remaining), None)
        if match is None:
            break
        plan.append(match)
        remaining -= match
    return plan, remaining
