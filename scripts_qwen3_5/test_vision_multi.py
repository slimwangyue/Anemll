#!/usr/bin/env python3
"""Quick test: verify the multi-function vision encoder loads and runs all 5 functions.

Usage:
    python3 scripts_qwen3_5/test_vision_multi.py
    python3 scripts_qwen3_5/test_vision_multi.py --model-dir qwen3_5_stable_lut4ffn_lut6em_fp16
"""
import sys, os, argparse, json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import coremltools as ct
from export_vision import SUPPORTED_RESOLUTIONS


def _cleanup_ane_temp():
    tmp = "/tmp/coreml_"
    import glob
    for p in glob.glob(tmp + "*"):
        try:
            import shutil; shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def test_multi_function_vision(model_dir):
    pkg_path = os.path.join(model_dir, "vision_encoder_multi.mlpackage")
    if not os.path.exists(pkg_path):
        print(f"ERROR: {pkg_path} not found — run combine_vision.py first.")
        return False

    print(f"Testing: {pkg_path}")
    print()

    all_ok = True
    for h, w in SUPPORTED_RESOLUTIONS:
        fn_name = f"f_{h}x{w}"
        expected_tokens = (h // 32) * (w // 32)
        print(f"Loading function {fn_name} ({h}×{w}, {expected_tokens} tokens)...")
        try:
            _cleanup_ane_temp()
            model = ct.models.MLModel(
                pkg_path,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                function_name=fn_name,
            )
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            all_ok = False
            continue

        # Build dummy pixel_values [1, 3, 2, H, W]
        T = 2
        pixel_values = np.zeros((1, 3, T, h, w), dtype=np.float16)

        try:
            out = model.predict({"pixel_values": pixel_values})
            embeds = list(out.values())[0]
            shape = embeds.shape  # expected (1, expected_tokens, 2560)
            ok = (shape == (1, expected_tokens, 2560))
            status = "OK" if ok else f"WRONG SHAPE (expected (1, {expected_tokens}, 2560))"
            print(f"  {fn_name}: {shape} → {status}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  PREDICT FAILED: {e}")
            all_ok = False

    print()
    if all_ok:
        print("All 5 functions PASSED.")
    else:
        print("Some functions FAILED.")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.path.join(_REPO_ROOT, "qwen3_5_stable_lut4ffn_lut6em_fp16"))
    args = parser.parse_args()

    success = test_multi_function_vision(args.model_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
