#!/usr/bin/env python3
"""
Export LUT6 prefill chunks, combine with existing LUT6 decode chunks,
and deploy to iOS Models.bundle.

The ffn_LUT4_chunk{0..3}.mlpackage files are actually LUT6 (confusing names).
The prefill_LUT4_chunk{0..3}.mlpackage files are LUT4.
This script exports LUT6 prefill chunks, combines them with the LUT6 decode
chunks, and copies the results to the iOS app's Models.bundle.
"""
import gc, time, os, sys, shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import from scripts_qwen3_5
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

BATCH_SIZE = 256
CTX = 1024
NUM_CHUNKS = 4
LUT6_BITS = 6
PER_CHANNEL = 8

STABLE_DIR = os.path.join(_REPO_ROOT, "qwen3_5_stable_models")
HF_MODEL = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
IOS_BUNDLE = "/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle"

from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from anemll.utils.combine_models import _save_multifunction_dedup


def dir_size_mb(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def main():
    print("=" * 70)
    print("  LUT6 Prefill Export + Combine + Deploy")
    print(f"  LUT bits: {LUT6_BITS} | Batch: {BATCH_SIZE} | CTX: {CTX}")
    print(f"  Stable dir: {STABLE_DIR}")
    print(f"  iOS bundle: {IOS_BUNDLE}")
    print("=" * 70)

    # --- Step 1: Export LUT6 prefill chunks ---
    print("\n[1/3] Exporting LUT6 prefill chunks...")
    print("  Loading model weights...")
    t_load = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL), f"Failed to load weights from {HF_MODEL}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t_load:.1f}s")

    for ci in range(NUM_CHUNKS):
        pf_path = os.path.join(STABLE_DIR, f"prefill_LUT6_chunk{ci}.mlpackage")
        if os.path.exists(pf_path):
            print(f"  [skip] prefill LUT6 chunk {ci} already exists")
            continue
        print(f"  Exporting prefill chunk {ci} (LUT6)...")
        t0 = time.time()
        conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                               num_chunks=NUM_CHUNKS, lut_bits=LUT6_BITS, per_channel=PER_CHANNEL)
        ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
        ml.save(pf_path)
        del ml, conv; gc.collect()
        print(f"  Saved prefill chunk {ci} ({time.time()-t0:.1f}s, {dir_size_mb(pf_path):.0f} MB)")

    del model; gc.collect()

    # --- Step 2: Combine LUT6 decode + LUT6 prefill ---
    print("\n[2/3] Combining LUT6 decode + prefill chunks...")
    combined_dir = os.path.join(STABLE_DIR, "combined_LUT6_dedup")
    os.makedirs(combined_dir, exist_ok=True)

    # Verify all sources exist
    missing = []
    for ci in range(NUM_CHUNKS):
        # Decode: ffn_LUT4_chunk{i} (actually LUT6!)
        dec = os.path.join(STABLE_DIR, f"ffn_LUT4_chunk{ci}.mlpackage")
        if not os.path.exists(dec):
            missing.append(dec)
        # Prefill: newly exported LUT6
        pf = os.path.join(STABLE_DIR, f"prefill_LUT6_chunk{ci}.mlpackage")
        if not os.path.exists(pf):
            missing.append(pf)
    if missing:
        print("ERROR: Missing source files:")
        for m in missing:
            print(f"  {m}")
        return 1

    for ci in range(NUM_CHUNKS):
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if os.path.exists(combined_path):
            print(f"  [skip] combined chunk {ci} already exists ({dir_size_mb(combined_path):.0f} MB)")
            continue

        dec_path = os.path.join(STABLE_DIR, f"ffn_LUT4_chunk{ci}.mlpackage")
        pf_path = os.path.join(STABLE_DIR, f"prefill_LUT6_chunk{ci}.mlpackage")
        sources = [
            (dec_path, "main", "infer"),
            (pf_path, "main", "prefill"),
        ]
        print(f"  Combining chunk {ci}...")
        t0 = time.time()
        _save_multifunction_dedup(sources, combined_path, dedup_weights=True, verbose=False)
        sz = dir_size_mb(combined_path)
        print(f"    Done ({time.time()-t0:.1f}s) — {sz:.0f} MB")

    # --- Step 3: Deploy to iOS ---
    print(f"\n[3/3] Deploying to iOS Models.bundle...")
    print(f"  Clearing {IOS_BUNDLE}...")
    for item in os.listdir(IOS_BUNDLE):
        full = os.path.join(IOS_BUNDLE, item)
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)

    # Copy combined LUT6 chunks
    for ci in range(NUM_CHUNKS):
        src = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        dst = os.path.join(IOS_BUNDLE, f"chunk{ci}.mlpackage")
        print(f"  Copying chunk{ci}.mlpackage ({dir_size_mb(src):.0f} MB)...")
        shutil.copytree(src, dst)

    # Copy embeddings
    src = os.path.join(STABLE_DIR, "embeddings.mlpackage")
    dst = os.path.join(IOS_BUNDLE, "embeddings.mlpackage")
    print(f"  Copying embeddings.mlpackage ({dir_size_mb(src):.0f} MB)...")
    shutil.copytree(src, dst)

    # Copy lm_head_logits
    src = os.path.join(STABLE_DIR, "lm_head_logits.mlpackage")
    dst = os.path.join(IOS_BUNDLE, "lm_head_logits.mlpackage")
    print(f"  Copying lm_head_logits.mlpackage ({dir_size_mb(src):.0f} MB)...")
    shutil.copytree(src, dst)

    # Copy tokenizer files
    for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = os.path.join(STABLE_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(IOS_BUNDLE, f))

    # Verify no symlinks
    symlinks = []
    for dp, dirs, files in os.walk(IOS_BUNDLE):
        for f in files + dirs:
            fp = os.path.join(dp, f)
            if os.path.islink(fp):
                symlinks.append(fp)
    if symlinks:
        print(f"\n  WARNING: {len(symlinks)} symlinks found in bundle!")
        for s in symlinks:
            print(f"    {s}")
    else:
        print("  No symlinks (safe for iOS)")

    print(f"\n  Total bundle size: {dir_size_mb(IOS_BUNDLE):.0f} MB")
    print("\n  DONE. Clean build in Xcode, then deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
