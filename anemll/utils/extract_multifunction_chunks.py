#!/usr/bin/env python3
import argparse
from pathlib import Path

import coremltools as ct


def extract_function(src_path: Path, src_function_name: str, dst_path: Path) -> None:
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(str(src_path), src_function_name=src_function_name, target_function_name="main")
    desc.default_function_name = "main"
    ct.utils.save_multifunction(desc, str(dst_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract single-function chunks from combined FFN_PF multifunction packages.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--lut", type=int, default=0)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    lut_suffix = f"_lut{args.lut}" if args.lut else ""
    input_dir = args.input_dir

    for idx in range(1, args.chunk + 1):
        combined = input_dir / f"{args.prefix}_FFN_PF{lut_suffix}_chunk_{idx:02d}of{args.chunk:02d}.mlpackage"
        infer_out = input_dir / f"{args.prefix}_FFN{lut_suffix}_chunk_{idx:02d}of{args.chunk:02d}.mlpackage"
        prefill_out = input_dir / f"{args.prefix}_prefill{lut_suffix}_chunk_{idx:02d}of{args.chunk:02d}.mlpackage"
        if not combined.exists():
            raise FileNotFoundError(f"Missing combined chunk: {combined}")
        if args.skip_existing and infer_out.exists() and prefill_out.exists():
            print(f"Skipping chunk {idx:02d}: outputs already exist")
            continue
        print(f"Extracting chunk {idx:02d} from {combined.name}")
        if infer_out.exists():
            print(f"  Reusing existing {infer_out.name}")
        else:
            extract_function(combined, "infer", infer_out)
            print(f"  Saved {infer_out.name}")
        if prefill_out.exists():
            print(f"  Reusing existing {prefill_out.name}")
        else:
            extract_function(combined, "prefill", prefill_out)
            print(f"  Saved {prefill_out.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
