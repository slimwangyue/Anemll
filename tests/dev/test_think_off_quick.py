#!/usr/bin/env python3
"""Quick test: 6-chunk think=OFF with 3 questions to verify empty think block works."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.dev.test_6chunk_validation import InferenceEngine, run_validation, TEST_CASES

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_dir_4 = os.path.join(base_dir, "qwen3_5_stable_models")
model_dir_6 = os.path.join(base_dir, "qwen3_5_6chunk_models")
chunk_dir_6 = os.path.join(model_dir_6, "combined_LUT4_dedup")

print("Loading 6-chunk models...")
engine = InferenceEngine(model_dir_6, model_dir_4, num_chunks=6, chunk_dir=chunk_dir_6)
print("Loaded. Running 3 questions with think=OFF...\n")

for q, kw in TEST_CASES[:3]:
    text, n_prompt, n_gen = engine.run(q, max_tokens=100, think=False)
    found = any(k.lower() in text.lower() for k in kw)
    status = "PASS" if found else "FAIL"
    display = text[:120].replace('\n', '\\n')
    print(f"[{status}] {q}")
    print(f"  prompt={n_prompt}tok gen={n_gen}tok")
    print(f"  A: {display}")
    if not found:
        print(f"  EXPECTED: {kw}")
    print()
