#!/usr/bin/env bash
# Export + Combine for Qwen3.5-4B with configurable chunk count.
#
# Usage:
#   ./scripts_qwen3_5/run_export_chunks.sh --chunks 6 --model /path/to/model
#   ./scripts_qwen3_5/run_export_chunks.sh --chunks 8 --model /path/to/model
#
# This does export + combine for stable exact prefill buckets:
#   infer + prefill_bs32 + prefill_bs64 + prefill_bs128 + prefill_bs256
# One multifunction .mlpackage is produced per chunk.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CHUNKS=""
MODEL=""
OUTPUT=""
SKIP_EXISTING=0
PREFILL_BUCKETS=(32 64 128 256)
CONTEXT_LENGTH=2048

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks) CHUNKS="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHUNKS" ]]; then
  echo "ERROR: Must provide --chunks N (e.g. --chunks 6)"
  exit 1
fi

if [[ -z "$MODEL" ]]; then
  # Try to find model automatically
  for candidate in \
    "$REPO_ROOT/models/Qwen__Qwen3.5-4B" \
    "$HOME/local_llm/models/Qwen__Qwen3.5-4B" \
    ; do
    if [[ -d "$candidate" ]]; then
      MODEL="$candidate"
      break
    fi
  done
fi

if [[ -z "$MODEL" ]]; then
  echo "ERROR: Must provide --model /path/to/Qwen3.5-4B"
  exit 1
fi

# Output defaults to qwen3_5_Nchunk_models/
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$REPO_ROOT/qwen3_5_${CHUNKS}chunk_models"
fi

# Export env vars for config.py
export QWEN35_NUM_CHUNKS="$CHUNKS"
export QWEN35_HF_MODEL="$MODEL"
export QWEN35_OUTPUT="$OUTPUT"
export QWEN35_STATIC_PREFILL_CTX="$CONTEXT_LENGTH"
export QWEN35_STATIC_PREFILL_BUCKETS="$(IFS=,; echo "${PREFILL_BUCKETS[*]}")"

echo "======================================================================"
echo "  Qwen3.5-4B — Export + Combine ($CHUNKS chunks, static prefill)"
echo "  Model:  $MODEL"
echo "  Output: $OUTPUT"
echo "  Chunks: $CHUNKS"
echo "  CTX:    $CONTEXT_LENGTH"
echo "  Buckets:${PREFILL_BUCKETS[*]}"
echo "======================================================================"

mkdir -p "$OUTPUT"

# Step 1: Export
echo ""
echo "── Step 1/2: Export exact bucket functions ($CHUNKS chunks) ──"
python3 - "$MODEL" "$OUTPUT" "$CHUNKS" "$SKIP_EXISTING" "$CONTEXT_LENGTH" "${PREFILL_BUCKETS[@]}" <<'PY'
import gc
import glob
import os
import shutil
import sys
import time

model_dir = sys.argv[1]
output_dir = sys.argv[2]
num_chunks = int(sys.argv[3])
skip_existing = bool(int(sys.argv[4]))
context_length = int(sys.argv[5])
buckets = [int(v) for v in sys.argv[6:]]

repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
scripts_dir = os.path.join(repo_root, "scripts_qwen3_5")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from config import FFN_LABEL, LUT_BITS, FFN_PER_CHANNEL
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


def copy_tokenizer_files(src_dir: str, dst_dir: str) -> None:
    patterns = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    copied = []
    for pattern in patterns:
        for src in glob.glob(os.path.join(src_dir, pattern)):
            dst = os.path.join(dst_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(os.path.basename(src))
    if copied:
        print(f"  Copied tokenizer files: {', '.join(copied)}")


os.makedirs(output_dir, exist_ok=True)
print("=" * 70)
print("  Exporting per-chunk multifunction sources")
print(f"  Chunks: {num_chunks} | CTX: {context_length} | Buckets: {buckets}")
print("=" * 70)

print("\nLoading model weights...")
t_load = time.time()
cfg = Qwen35Config.from_json(os.path.join(model_dir, "config.json"))
cfg.context_length = context_length
cfg.state_length = context_length
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(model_dir), f"Failed to load weights from {model_dir}"
model.eval()
for parameter in model.parameters():
    parameter.requires_grad = False
print(f"  Loaded in {time.time() - t_load:.1f}s")

for chunk_index in range(num_chunks):
    infer_path = os.path.join(output_dir, f"ffn_{FFN_LABEL}_chunk{chunk_index}.mlpackage")
    if skip_existing and os.path.exists(infer_path):
        print(f"  [skip] chunk {chunk_index} infer")
    else:
        print(f"  Exporting chunk {chunk_index} infer...")
        t0 = time.time()
        conv = Qwen35Converter(
            model,
            context_length=context_length,
            batch_size=1,
            num_chunks=num_chunks,
            lut_bits=LUT_BITS,
            per_channel=FFN_PER_CHANNEL,
        )
        mlmodel = conv.convert_part_2(model, chunk_idx=chunk_index, total_chunks=num_chunks)
        mlmodel.save(infer_path)
        del mlmodel, conv
        gc.collect()
        print(f"    Saved infer ({time.time() - t0:.1f}s)")

    for bucket in buckets:
        prefill_path = os.path.join(
            output_dir,
            f"prefill_{FFN_LABEL}_chunk{chunk_index}_bs{bucket}.mlpackage",
        )
        if skip_existing and os.path.exists(prefill_path):
            print(f"  [skip] chunk {chunk_index} prefill_bs{bucket}")
            continue

        print(f"  Exporting chunk {chunk_index} prefill_bs{bucket}...")
        t0 = time.time()
        conv = Qwen35Converter(
            model,
            context_length=context_length,
            batch_size=bucket,
            num_chunks=num_chunks,
            lut_bits=LUT_BITS,
            per_channel=FFN_PER_CHANNEL,
        )
        mlmodel = conv.convert_part_2_prefill_exact(
            model,
            chunk_idx=chunk_index,
            total_chunks=num_chunks,
            exact_seq_len=bucket,
        )
        mlmodel.save(prefill_path)
        del mlmodel, conv
        gc.collect()
        print(f"    Saved prefill_bs{bucket} ({time.time() - t0:.1f}s)")

del model
gc.collect()
copy_tokenizer_files(model_dir, output_dir)
PY

# Step 2: Combine (dedup)
echo ""
echo "── Step 2/2: Combine multifunction chunks ──"
python3 - "$OUTPUT" "$CHUNKS" "$SKIP_EXISTING" "${PREFILL_BUCKETS[@]}" <<'PY'
import os
import sys
import time

input_dir = sys.argv[1]
num_chunks = int(sys.argv[2])
skip_existing = bool(int(sys.argv[3]))
buckets = [int(v) for v in sys.argv[4:]]

repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
scripts_dir = os.path.join(repo_root, "scripts_qwen3_5")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from config import FFN_LABEL
from anemll.utils.combine_models import _save_multifunction_dedup


def dir_size_mb(path: str) -> float:
    total = 0
    for dp, _, fns in os.walk(path):
        for name in fns:
            full = os.path.join(dp, name)
            if not os.path.islink(full):
                total += os.path.getsize(full)
    return total / (1024 * 1024)


combined_dir = os.path.join(input_dir, f"combined_{FFN_LABEL}_dedup")
os.makedirs(combined_dir, exist_ok=True)

print("=" * 70)
print("  Combining exact-bucket multifunction chunks")
print(f"  Functions per chunk: infer + {', '.join(f'prefill_bs{b}' for b in buckets)}")
print("=" * 70)

total_size = 0.0
t_total = time.time()
for chunk_index in range(num_chunks):
    combined_path = os.path.join(combined_dir, f"chunk{chunk_index}.mlpackage")
    if skip_existing and os.path.exists(combined_path):
        size_mb = dir_size_mb(combined_path)
        total_size += size_mb
        print(f"  [skip] chunk {chunk_index} ({size_mb:.1f} MB)")
        continue

    sources = [
        (os.path.join(input_dir, f"ffn_{FFN_LABEL}_chunk{chunk_index}.mlpackage"), "main", "infer"),
    ]
    for bucket in buckets:
        sources.append(
            (
                os.path.join(input_dir, f"prefill_{FFN_LABEL}_chunk{chunk_index}_bs{bucket}.mlpackage"),
                "main",
                f"prefill_bs{bucket}",
            )
        )

    print(f"  Combining chunk {chunk_index}...")
    t0 = time.time()
    _save_multifunction_dedup(sources, combined_path, dedup_weights=True, verbose=False)
    size_mb = dir_size_mb(combined_path)
    total_size += size_mb
    print(f"    Done ({time.time() - t0:.1f}s) — {size_mb:.1f} MB")

print(f"\n  Total combined: {total_size:.1f} MB")
print(f"  Elapsed: {time.time() - t_total:.1f}s")
PY

echo ""
echo "======================================================================"
echo "  Export + Combine complete ($CHUNKS chunks, static prefill)"
echo "  Output: $OUTPUT"
echo "======================================================================"
du -sh "$OUTPUT"/*.mlpackage "$OUTPUT"/combined_LUT*_dedup/*.mlpackage 2>/dev/null | head -20 || true
echo ""
echo "Transfer to Mac, then compile:"
echo "  python scripts_qwen3_5/compile.py --model-dir $OUTPUT"
echo ""
echo "Combined chunk functions are:"
echo "  infer, prefill_bs32, prefill_bs64, prefill_bs128, prefill_bs256"
