#!/usr/bin/env python3
"""Compare FP16 vs FP32 MIL programs at protobuf level."""
import coremltools as ct
from collections import Counter

path16 = "/Users/yw68/Anemll_remote_run/qwen35_milestone1_3/ffn_LUT4_chunk0.mlpackage"
path32 = "/tmp/qwen35_fp32_chunks/ffn_LUT4_chunk0.mlpackage"

spec16 = ct.utils.load_spec(path16)
spec32 = ct.utils.load_spec(path32)

prog16 = spec16.mlProgram
prog32 = spec32.mlProgram


def count_ops(prog, label):
    op_counter = Counter()
    cast_detail = []
    total = 0
    for fname, func in prog.functions.items():
        for bname, block in func.block_specializations.items():
            for op in block.operations:
                op_counter[op.type] += 1
                total += 1
                if op.type == "cast":
                    for attr_name, attr in op.attributes.items():
                        if attr_name == "dtype":
                            try:
                                val = attr.immediate.tensor.strings.values[0]
                            except Exception:
                                val = "?"
                            cast_detail.append(val)
    print(f"[{label}] {total} ops")
    for op_type, cnt in op_counter.most_common(25):
        print(f"  {op_type:<25} {cnt:>5}")
    if cast_detail:
        print(f"  Cast targets: {Counter(cast_detail)}")
    return op_counter


c16 = count_ops(prog16, "FP16")
print()
c32 = count_ops(prog32, "FP32")

print("\nDifferences (FP32 - FP16):")
all_ops = set(c16.keys()) | set(c32.keys())
for op in sorted(all_ops):
    if c16[op] != c32[op]:
        diff = c32[op] - c16[op]
        print(f"  {op:<25} FP16={c16[op]:>5}  FP32={c32[op]:>5}  diff={diff:>+5}")
