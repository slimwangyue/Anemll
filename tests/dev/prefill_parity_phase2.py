#!/usr/bin/env python3
"""Phase 2: Compare CoreML prefill chunks against saved PyTorch reference outputs.

Prerequisites:
    - Run prefill_parity_phase1.py first (or verify /tmp/qwen35_prefill_parity/ has .npy files)
    - Exported .mlpackage / .mlmodelc models in EXPORT_DIR

Usage:
    python -u prefill_parity_phase2.py                       # All chunks, ANE
    python -u prefill_parity_phase2.py --chunk 1             # Just chunk 1
    python -u prefill_parity_phase2.py --cpu-gpu             # CPU+GPU fallback
    python -u prefill_parity_phase2.py --cascaded            # Feed CoreML output forward
    python -u prefill_parity_phase2.py --compiled            # Use .mlmodelc (faster)
    python -u prefill_parity_phase2.py --chunk 1 --cpu-gpu   # Single chunk on CPU+GPU
"""
import argparse
import gc
import os
import sys

import coremltools as ct
import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


def _load_model(path: str, compute, compiled: bool):
    """Load a CoreML model, handling .mlpackage vs .mlmodelc."""
    if compiled:
        mlmodelc = path.replace(".mlpackage", ".mlmodelc")
        if os.path.exists(mlmodelc):
            return ct.models.CompiledMLModel(mlmodelc, compute)
        # Fall through to .mlpackage
    return ct.models.MLModel(path, compute_units=compute)


def _build_inputs(spec, hidden: np.ndarray, seq_len: int) -> dict:
    """Build input dict from model spec, skipping StateType inputs."""
    inp_dict = {}
    for inp in spec.description.input:
        name = inp.name
        if not inp.type.HasField("multiArrayType"):
            continue  # StateType handled via make_state()
        shape = tuple(inp.type.multiArrayType.shape)
        if name == "hidden_states":
            inp_dict[name] = hidden.astype(np.float16).reshape(shape)
        elif "position" in name:
            inp_dict[name] = np.arange(seq_len, dtype=np.int32).reshape(shape)
        elif "causal_mask" in name or "mask" in name:
            mask = np.full(shape, -65504.0, dtype=np.float16)
            for r in range(shape[-2]):
                mask[..., r, : r + 1] = 0
            inp_dict[name] = mask
        elif "current_pos" in name:
            inp_dict[name] = np.zeros(shape, dtype=np.int32)
        else:
            print(f"  WARNING: unknown input '{name}' shape={shape}, using zeros")
            inp_dict[name] = np.zeros(shape, dtype=np.float16)
    return inp_dict


def _compare(name: str, torch_ref: np.ndarray, cml_out: np.ndarray):
    """Print parity metrics between PyTorch reference and CoreML output."""
    # Handle shape mismatch (last chunk returns fewer tokens)
    if cml_out.shape != torch_ref.shape:
        print(f"  Shape mismatch: torch={torch_ref.shape} cml={cml_out.shape}")
        if cml_out.shape[1] < torch_ref.shape[1]:
            torch_ref = torch_ref[:, : cml_out.shape[1], :]
            print(f"  Comparing first {cml_out.shape[1]} token(s)")
        else:
            min_seq = min(torch_ref.shape[1], cml_out.shape[1])
            torch_ref = torch_ref[:, :min_seq, :]
            cml_out = cml_out[:, :min_seq, :]

    diff = np.abs(torch_ref.astype(np.float32) - cml_out.astype(np.float32))
    cos = _cosine(torch_ref, cml_out)
    flat = diff.flatten()

    print(f"{name}: max_abs={diff.max():.4f}  mean_abs={diff.mean():.6f}  cosine={cos:.10f}")
    print(
        f"  p99={np.percentile(flat, 99):.4f}  "
        f"p95={np.percentile(flat, 95):.4f}  "
        f"p50={np.percentile(flat, 50):.6f}"
    )

    # Top-5 worst positions
    worst = np.argsort(flat)[-5:][::-1]
    for w in worst:
        pos = np.unravel_index(w, diff.shape)
        print(
            f"  worst: pos={pos} torch={torch_ref[pos]:.4f} "
            f"cml={cml_out[pos]:.4f} diff={diff[pos]:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Prefill parity Phase 2: CoreML vs PyTorch")
    parser.add_argument("--chunk", type=int, default=0, help="Run only this chunk (1-4), 0=all")
    parser.add_argument("--cascaded", action="store_true", help="Feed CoreML output to next chunk")
    parser.add_argument("--cpu-gpu", action="store_true", help="Use CPU_AND_GPU instead of ANE")
    parser.add_argument("--compiled", action="store_true", help="Prefer .mlmodelc over .mlpackage")
    parser.add_argument("--tmp-dir", default="/tmp/qwen35_prefill_parity")
    parser.add_argument(
        "--export-dir", default="/Users/yw68/Anemll_remote_run/qwen35_chunk4_export"
    )
    parser.add_argument("--prefix", default="qwen35")
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    compute = ct.ComputeUnit.CPU_AND_GPU if args.cpu_gpu else ct.ComputeUnit.CPU_AND_NE
    mode = "CPU_AND_GPU" if args.cpu_gpu else "CPU_AND_NE"
    print(f"Compute: {mode}")
    print(f"Export dir: {args.export_dir}")
    print(f"Ref dir: {args.tmp_dir}")

    # Load PyTorch references
    input_ids = np.load(f"{args.tmp_dir}/input_ids.npy")
    torch_embed = np.load(f"{args.tmp_dir}/embed_out.npy")
    torch_chunks = {}
    for i in range(args.num_chunks):
        f = f"{args.tmp_dir}/torch_chunk{i+1}.npy"
        if os.path.exists(f):
            torch_chunks[i + 1] = np.load(f)
    print(f"Loaded {len(torch_chunks)} PyTorch chunk references + embed")

    # === Embeddings ===
    print("\n--- Embeddings ---")
    embed_path = os.path.join(args.export_dir, f"{args.prefix}_embeddings.mlpackage")
    embed_model = _load_model(embed_path, compute, args.compiled)
    embed_out = embed_model.predict({"input_ids": input_ids.astype(np.int32)})
    cml_embed = list(embed_out.values())[0]
    _compare("EMBED", torch_embed, cml_embed)
    del embed_model
    gc.collect()

    # === Prefill Chunks ===
    chunks_to_run = range(1, args.num_chunks + 1) if args.chunk == 0 else [args.chunk]
    prev_hidden = cml_embed.copy()

    for ci in chunks_to_run:
        chunk_name = f"{args.prefix}_prefill_chunk_{ci:02d}of{args.num_chunks:02d}.mlpackage"
        chunk_path = os.path.join(args.export_dir, chunk_name)
        print(f"\n--- Chunk {ci}: {chunk_name} ---")

        # Input: cascaded uses CoreML output; isolated uses PyTorch ref
        if args.cascaded or ci == 1:
            hidden_input = cml_embed.copy() if ci == 1 else prev_hidden.copy()
            src = "CoreML-cascaded" if ci > 1 else "CoreML-embed"
        else:
            hidden_input = torch_embed.copy() if ci == 1 else torch_chunks.get(ci - 1, prev_hidden).copy()
            src = "PyTorch-isolated"
        print(f"  Input: {src} shape={hidden_input.shape}")

        model = _load_model(chunk_path, compute, args.compiled)
        spec = model.get_spec()

        # Print inputs for diagnostics
        for inp in spec.description.input:
            if inp.type.HasField("multiArrayType"):
                print(f"  input: {inp.name} shape={tuple(inp.type.multiArrayType.shape)}")
            elif inp.type.HasField("stateType"):
                print(f"  state: {inp.name} shape={tuple(inp.type.stateType.multiArrayType.shape)}")

        inp_dict = _build_inputs(spec, hidden_input, args.seq_len)
        state = model.make_state()

        try:
            out = model.predict(inp_dict, state=state)
        except RuntimeError as e:
            err_str = str(e)
            print(f"  PREDICT FAILED: {err_str[:200]}")
            if not args.cpu_gpu and "ANE" in err_str:
                print("  Auto-retrying with CPU_AND_GPU...")
                del model, state
                gc.collect()
                model = _load_model(chunk_path, ct.ComputeUnit.CPU_AND_GPU, args.compiled)
                state = model.make_state()
                out = model.predict(inp_dict, state=state)
            else:
                raise

        cml_hidden = out[list(out.keys())[0]]
        print(f"  Output: shape={cml_hidden.shape} max={np.abs(cml_hidden).max():.4f}")
        prev_hidden = cml_hidden.copy()

        if ci in torch_chunks:
            _compare(f"CHUNK {ci}", torch_chunks[ci], cml_hidden)
        else:
            print(f"  No PyTorch reference for chunk {ci} — skipping comparison")

        del model, state
        gc.collect()

    print("\n=== PARITY VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()
