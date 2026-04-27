#!/usr/bin/env python3
"""Post-combine protobuf fix: inject cast ops between read_state and slice_update.

The combine step's save_multifunction re-runs MIL optimization passes that strip
the cast ops needed for correct KV cache write positions on ANE. This script
directly modifies the combined mlpackage protobufs to re-inject those casts,
bypassing MIL optimization entirely.

Usage:
    python scripts_qwen3_5/fix_combined_protobuf.py --model-dir qwen3_5_4B_milestone_3.4_fix
"""
import os
import sys
import argparse
import time
import copy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from config import NUM_CHUNKS, FFN_LABEL, DEFAULT_OUTPUT

import coremltools as ct

# CoreML proto data types
DT_FLOAT16 = 10
DT_FLOAT32 = 11
DT_STRING = 2


def _make_const_op(name, string_value):
    """Create a const op that produces a string value (for dtype args)."""
    from coremltools.proto import MIL_pb2

    op = MIL_pb2.Operation()
    op.type = "const"

    # Output
    out = op.outputs.add()
    out.name = name
    out.type.tensorType.dataType = DT_STRING

    # Attributes: name
    name_attr = op.attributes["name"]
    name_attr.type.tensorType.dataType = DT_STRING
    name_attr.immediateValue.tensor.strings.values.append(name)

    # Attributes: val
    val_attr = op.attributes["val"]
    val_attr.type.tensorType.dataType = DT_STRING
    val_attr.immediateValue.tensor.strings.values.append(string_value)

    return op


def _make_cast_op(x_name, dtype_const_name, output_name, output_dtype, output_type_proto):
    """Create a cast op: output = cast(x, dtype)."""
    from coremltools.proto import MIL_pb2

    op = MIL_pb2.Operation()
    op.type = "cast"

    # Input: x
    x_arg = op.inputs["x"].arguments.add()
    x_arg.name = x_name

    # Input: dtype (reference to const)
    dtype_arg = op.inputs["dtype"].arguments.add()
    dtype_arg.name = dtype_const_name

    # Output with correct type
    out = op.outputs.add()
    out.name = output_name
    out.type.CopyFrom(output_type_proto)
    out.type.tensorType.dataType = output_dtype

    # Attribute: name (every op needs this in CoreML protobuf)
    name_attr = op.attributes["name"]
    name_attr.type.tensorType.dataType = DT_STRING
    name_attr.immediateValue.tensor.strings.values.append(output_name)

    return op


def _fix_function_kv_casts(block, fn_name, counter):
    """Fix read_state → slice_update patterns in a function block.

    Returns the number of casts injected.
    """
    ops = list(block.operations)

    # Build output_name → op_index map AND output_name → output type map
    output_to_idx = {}
    output_to_type = {}
    for i, op in enumerate(ops):
        for o in op.outputs:
            output_to_idx[o.name] = i
            output_to_type[o.name] = o.type

    # Find slice_update ops where x comes directly from read_state for k_cache/v_cache
    fixes = []  # (slice_update_idx, read_state_idx, cache_name)

    for i, op in enumerate(ops):
        if op.type != "slice_update":
            continue

        # Get x input name
        x_name = None
        for arg in op.inputs["x"].arguments:
            if arg.HasField("name"):
                x_name = arg.name
                break
        if x_name is None:
            continue

        # Check if x comes from read_state
        if x_name not in output_to_idx:
            continue
        src_idx = output_to_idx[x_name]
        src_op = ops[src_idx]
        if src_op.type != "read_state":
            continue

        # Get the cache name (k_cache or v_cache)
        cache_name = None
        for arg in src_op.inputs["input"].arguments:
            if arg.HasField("name"):
                cache_name = arg.name
                break

        if cache_name not in ("k_cache", "v_cache"):
            continue

        # Check if write_state follows shortly after
        # Find the write_state that references this cache
        write_idx = None
        for j in range(i + 1, min(i + 5, len(ops))):
            if ops[j].type == "write_state":
                for arg in ops[j].inputs["input"].arguments:
                    if arg.HasField("name") and arg.name == cache_name:
                        write_idx = j
                        break
            if write_idx is not None:
                break

        if write_idx is None:
            continue

        fixes.append((i, src_idx, write_idx, cache_name, x_name))

    if not fixes:
        return 0

    # Build new operations list with injected casts
    new_ops = []
    injected = 0
    skip_indices = set()

    for fix_idx, (su_idx, rs_idx, ws_idx, cache_name, rs_output_name) in enumerate(fixes):
        su_op = ops[su_idx]
        ws_op = ops[ws_idx]

        # Get read_state output type (FP16)
        rs_op = ops[rs_idx]
        rs_output_type = rs_op.outputs[0].type

        # Get slice_update output name and update input name
        su_output_name = su_op.outputs[0].name
        update_name = None
        for arg in su_op.inputs["update"].arguments:
            if arg.HasField("name"):
                update_name = arg.name
                break

        # Create unique names
        c = counter[0]
        counter[0] += 1
        state_cast_const = f"_fix_cast_fp32_const_{c}"
        state_cast_output = f"_fix_state_cast_{c}"
        value_cast_output = f"_fix_value_cast_{c}"
        result_cast_const = f"_fix_cast_fp16_const_{c}"
        result_cast_output = f"_fix_result_cast_{c}"

        # Create FP32 output type (copy from read_state output, change dtype)
        fp32_state_type = copy.deepcopy(rs_output_type)
        fp32_state_type.tensorType.dataType = DT_FLOAT32

        # Create FP32 type for value cast (copy from slice_update update input's type)
        # We need to find the update tensor's type - use slice_update output type with FP32
        fp32_su_type = copy.deepcopy(rs_output_type)
        fp32_su_type.tensorType.dataType = DT_FLOAT32

        # Store fix info for insertion
        fixes[fix_idx] = (su_idx, rs_idx, ws_idx, cache_name, rs_output_name,
                          su_output_name, update_name,
                          state_cast_const, state_cast_output,
                          value_cast_output, result_cast_const, result_cast_output,
                          fp32_state_type, fp32_su_type, rs_output_type)

    # Now rebuild the operations list
    fix_map = {}  # su_idx -> fix tuple
    for fix in fixes:
        fix_map[fix[0]] = fix

    ws_fix_map = {}  # ws_idx -> fix tuple
    for fix in fixes:
        ws_fix_map[fix[2]] = fix

    for i, op in enumerate(ops):
        if i in fix_map:
            fix = fix_map[i]
            (su_idx, rs_idx, ws_idx, cache_name, rs_output_name,
             su_output_name, update_name,
             state_cast_const, state_cast_output,
             value_cast_output, result_cast_const, result_cast_output,
             fp32_state_type, fp32_su_type, rs_output_type) = fix

            # Insert BEFORE slice_update:
            # 1. const for fp32 dtype
            new_ops.append(_make_const_op(state_cast_const, "fp32"))

            # 2. cast read_state output (x) to fp32
            new_ops.append(_make_cast_op(
                rs_output_name, state_cast_const,
                state_cast_output, DT_FLOAT32, fp32_state_type
            ))

            # 3. cast update value to fp32 (ios18.slice_update requires x and update same dtype)
            if update_name and update_name in output_to_type:
                update_type = output_to_type[update_name]
                fp32_update_type = copy.deepcopy(update_type)
                fp32_update_type.tensorType.dataType = DT_FLOAT32
                new_ops.append(_make_cast_op(
                    update_name, state_cast_const,
                    value_cast_output, DT_FLOAT32, fp32_update_type
                ))

            # 4. Modified slice_update - update both x and update inputs
            new_su = copy.deepcopy(op)
            for arg in new_su.inputs["x"].arguments:
                if arg.HasField("name") and arg.name == rs_output_name:
                    arg.name = state_cast_output
            if update_name:
                for arg in new_su.inputs["update"].arguments:
                    if arg.HasField("name") and arg.name == update_name:
                        arg.name = value_cast_output
            # Change output type to FP32
            new_su.outputs[0].type.tensorType.dataType = DT_FLOAT32
            new_ops.append(new_su)

            # 5. const for fp16 dtype
            new_ops.append(_make_const_op(result_cast_const, "fp16"))

            # 6. cast result back to fp16
            fp16_result_type = copy.deepcopy(rs_output_type)
            fp16_result_type.tensorType.dataType = DT_FLOAT16
            new_ops.append(_make_cast_op(
                su_output_name, result_cast_const,
                result_cast_output, DT_FLOAT16, fp16_result_type
            ))

            injected += 1

        elif i in ws_fix_map:
            fix = ws_fix_map[i]
            result_cast_output = fix[11]
            su_output_name = fix[5]

            # Update write_state data input to use cast output
            new_ws = copy.deepcopy(op)
            for arg in new_ws.inputs["data"].arguments:
                if arg.HasField("name") and arg.name == su_output_name:
                    arg.name = result_cast_output
            new_ops.append(new_ws)

        else:
            new_ops.append(op)

    # Replace operations in block
    del block.operations[:]
    for op in new_ops:
        block.operations.append(op)

    return injected


def fix_chunk(chunk_path, chunk_idx):
    """Fix a single combined chunk mlpackage."""
    spec = ct.utils.load_spec(chunk_path)
    prog = spec.mlProgram

    total_injected = 0
    counter = [0]

    for fn_name, fn in prog.functions.items():
        block = None
        block_key = None
        for k in fn.block_specializations:
            block = fn.block_specializations[k]
            block_key = k
            break
        if block is None:
            continue

        n = _fix_function_kv_casts(block, fn_name, counter)
        if n > 0:
            print(f"    {fn_name}: injected {n} cast chain(s)")
        total_injected += n

    if total_injected > 0:
        import shutil
        import tempfile
        # Find the weights directory inside the mlpackage
        from coremltools.models.utils import _try_get_weights_dir_path
        weights_dir = _try_get_weights_dir_path(chunk_path)
        # Save to temp path, then replace
        tmp_path = chunk_path.replace(".mlpackage", "_fixed.mlpackage")
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        ct.utils.save_spec(spec, tmp_path, weights_dir=weights_dir)
        # Replace original
        shutil.rmtree(chunk_path)
        shutil.move(tmp_path, chunk_path)

    return total_injected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--separate", action="store_true",
                        help="Also fix separate prefill .mlpackage files")
    args = parser.parse_args()

    label = FFN_LABEL
    combined_dir = os.path.join(args.model_dir, f"combined_{label}_dedup")

    print("=" * 70)
    print("  Post-combine protobuf cast injection")
    print(f"  Dir: {combined_dir}")
    print("=" * 70)

    total_fixed = 0
    for ci in range(NUM_CHUNKS):
        pkg = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if not os.path.exists(pkg):
            print(f"  chunk {ci}: not found, skip")
            continue

        t0 = time.time()
        n = fix_chunk(pkg, ci)
        elapsed = time.time() - t0

        if n > 0:
            print(f"  chunk {ci}: fixed ({n} cast chains, {elapsed:.1f}s)")
            total_fixed += n
        else:
            print(f"  chunk {ci}: OK (no fix needed, {elapsed:.1f}s)")

    if args.separate:
        print()
        print("── Separate prefill models ──")
        for ci in range(NUM_CHUNKS):
            pkg = os.path.join(args.model_dir, f"prefill_{label}_chunk{ci}.mlpackage")
            if not os.path.exists(pkg):
                print(f"  prefill chunk {ci}: not found, skip")
                continue

            t0 = time.time()
            n = fix_chunk(pkg, ci)
            elapsed = time.time() - t0

            if n > 0:
                print(f"  prefill chunk {ci}: fixed ({n} cast chains, {elapsed:.1f}s)")
                total_fixed += n
            else:
                print(f"  prefill chunk {ci}: OK (no fix needed, {elapsed:.1f}s)")

    print(f"\n  Total: {total_fixed} cast chains injected")
    if total_fixed > 0:
        print(f"\n  Re-compile with:")
        print(f"    python scripts_qwen3_5/compile.py --model-dir {args.model_dir}")


if __name__ == "__main__":
    main()
