#!/usr/bin/env python3
"""Qwen3.5-4B Milestone 2.1 — Step 2: Combine chunks with ANEMLL-Dedup.

Combines each decode + prefill into a single multi-function .mlpackage
with shared (deduplicated) weights.

Usage:
    python scripts_qwen3_5/combine.py --input /path/to/exported
    python scripts_qwen3_5/combine.py --skip-existing
"""
import os, time, argparse, sys, shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)  # must be first for config.py

from config import BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, FFN_LABEL, DEFAULT_OUTPUT
from anemll.utils.combine_models import _save_multifunction_dedup


def dir_size_mb(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def combine_embed_lmhead(input_dir, skip_existing=False):
    """Combine embeddings + lm_head_nosplit into a multi-function mlpackage with cross-model weight dedup.

    Uses fixed-shape embed variants (embed_single [1,1] and embed_prefill [1,BS])
    to avoid unknown MIL dimensions that cause E5ML stride errors on ANE.
    Falls back to EnumeratedShapes embeddings.mlpackage if fixed variants don't exist.
    """
    import coremltools as ct
    from anemll.utils.dedup_weights import prepare_dedup_sources, dedup_cross_model_blobs

    embed_single_path = os.path.join(input_dir, "embed_single.mlpackage")
    embed_prefill_path = os.path.join(input_dir, "embed_prefill.mlpackage")
    embed_legacy_path = os.path.join(input_dir, "embeddings.mlpackage")
    lmhead_path = os.path.join(input_dir, "lm_head_nosplit.mlpackage")
    combined_path = os.path.join(input_dir, "embed_lmhead_combined.mlpackage")

    if skip_existing and os.path.exists(combined_path):
        sz = dir_size_mb(combined_path)
        print(f"  [skip] embed_lmhead_combined ({sz:.1f} MB)")
        return combined_path

    if not os.path.exists(lmhead_path):
        print(f"  ERROR: {lmhead_path} not found")
        return None

    # Prefer fixed-shape variants (ANE-safe), fall back to legacy
    use_fixed = os.path.exists(embed_single_path) and os.path.exists(embed_prefill_path)
    if use_fixed:
        print(f"  Using fixed-shape embed variants (ANE-safe)...")
        sources = [
            (embed_single_path, "main", "embed"),
            (embed_prefill_path, "main", "embed_prefill"),
            (lmhead_path, "main", "lmhead"),
        ]
    else:
        if not os.path.exists(embed_legacy_path):
            print(f"  ERROR: neither embed_single/embed_prefill nor embeddings.mlpackage found")
            return None
        print(f"  WARNING: Using legacy EnumeratedShapes embeddings (may cause E5ML stride warnings on ANE)")
        sources = [
            (embed_legacy_path, "main", "embed"),
            (lmhead_path, "main", "lmhead"),
        ]

    print(f"  Combining {len(sources)} functions into multifunction model with dedup...")
    t0 = time.time()

    try:
        with prepare_dedup_sources(sources, verbose=True, preflight=False) as deduped:
            desc = ct.utils.MultiFunctionDescriptor()
            for path, src_fn, tgt_fn in deduped:
                desc.add_function(path, src_fn, tgt_fn)
            desc.default_function_name = "embed"
            ct.utils.save_multifunction(desc, combined_path)

        print("  Running cross-model blob dedup post-processing...")
        fn_names = [tgt for _, _, tgt in sources]
        total_saved = 0
        # Dedup all pairs of functions
        for i in range(len(fn_names)):
            for j in range(i + 1, len(fn_names)):
                saved = dedup_cross_model_blobs(combined_path, fn_names[i], fn_names[j], verbose=True)
                total_saved += saved
        if total_saved > 0:
            print(f"  Cross-model dedup saved {total_saved / 1e6:.1f} MB total")
        else:
            print("  Cross-model dedup: no savings (blobs may already be shared or differ)")
    except Exception as e:
        print(f"  Dedup failed ({e}), trying without dedup...")
        if os.path.exists(combined_path):
            shutil.rmtree(combined_path)
        desc = ct.utils.MultiFunctionDescriptor()
        for path, src_fn, tgt_fn in sources:
            desc.add_function(path, src_fn, tgt_fn)
        desc.default_function_name = "embed"
        ct.utils.save_multifunction(desc, combined_path)

    sz = dir_size_mb(combined_path)
    print(f"  Saved embed_lmhead_combined ({time.time()-t0:.1f}s) — {sz:.1f} MB")
    return combined_path


def main():
    parser = argparse.ArgumentParser(description="Combine Qwen3.5-4B chunks (Milestone 2.1)")
    parser.add_argument("--input", default=DEFAULT_OUTPUT,
                        help="Directory with exported .mlpackage files")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-chunk", type=int, default=None,
                        help="Combine only the specified chunk index")
    parser.add_argument("--label", type=str, default=None,
                        help="Override FFN label (e.g. 'LUT4'). Default: from config.py")
    parser.add_argument("--combine-embed-lmhead", action="store_true",
                        help="Also combine embeddings + lm_head_nosplit into embed_lmhead_combined.mlpackage")
    args = parser.parse_args()

    label = args.label if args.label else FFN_LABEL
    combined_dir = os.path.join(args.input, f"combined_{label}_dedup")
    os.makedirs(combined_dir, exist_ok=True)

    # Verify sources
    # missing = []
    # for ci in range(NUM_CHUNKS):
    #     dec = os.path.join(args.input, f"ffn_{label}_chunk{ci}.mlpackage")
    #     if not os.path.exists(dec):
    #         missing.append(dec)
    #     pf = os.path.join(args.input, f"prefill_{label}_chunk{ci}.mlpackage")
    #     if not os.path.exists(pf):
    #         missing.append(pf)
    # if missing:
    #     print("ERROR: Missing source files:")
    #     for m in missing:
    #         print(f"  {m}")
    #     return 1

    print("=" * 70)
    print("  Qwen3.5-4B ANEMLL-Dedup Combine — Milestone 2.1")
    print(f"  Functions per chunk: infer + prefill")
    if args.only_chunk is not None:
        print(f"  Only chunk: {args.only_chunk}")
    print("=" * 70)

    t_total = time.time()
    total_size = 0.0
    chunk_indices = [args.only_chunk] if args.only_chunk is not None else list(range(NUM_CHUNKS))
    for ci in chunk_indices:
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if args.skip_existing and os.path.exists(combined_path):
            sz = dir_size_mb(combined_path)
            total_size += sz
            print(f"  [skip] chunk {ci} ({sz:.1f} MB)")
            continue

        dec_path = os.path.join(args.input, f"ffn_{label}_chunk{ci}.mlpackage")
        sources = [
            (dec_path, "main", "infer"),
        ]
        # Support both old naming (prefill_{label}_chunk{ci}.mlpackage) and
        # new bucket-based naming (prefill_{label}_chunk{ci}_bs{bucket}.mlpackage).
        pf_path_old = os.path.join(args.input, f"prefill_{label}_chunk{ci}.mlpackage")
        pf_path_bs = os.path.join(args.input, f"prefill_{label}_chunk{ci}_bs{BATCH_SIZE}.mlpackage")
        if os.path.exists(pf_path_bs):
            sources.append((pf_path_bs, "main", "prefill"))
        elif os.path.exists(pf_path_old):
            sources.append((pf_path_old, "main", "prefill"))
        else:
            print(f"  ERROR: No prefill found for chunk {ci}")
            return 1

        print(f"  Combining chunk {ci} (infer, prefill)...")
        t0 = time.time()
        _save_multifunction_dedup(sources, combined_path, dedup_weights=True, verbose=False)
        sz = dir_size_mb(combined_path)
        total_size += sz
        print(f"    Done ({time.time()-t0:.1f}s) — {sz:.1f} MB")

    print(f"\n  Total combined: {total_size:.1f} MB")

    # Optionally combine embed + lm_head_nosplit
    if args.combine_embed_lmhead:
        print(f"\n  Combining embed + lm_head_nosplit...")
        result = combine_embed_lmhead(args.input, skip_existing=args.skip_existing)
        if result is None:
            return 1

    print(f"  Elapsed: {time.time()-t_total:.1f}s")
    print(f"\nNext: python scripts_qwen3_5/compile.py --model-dir {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
