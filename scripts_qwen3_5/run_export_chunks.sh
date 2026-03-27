#!/usr/bin/env bash
# Export + Combine for Qwen3.5-4B with configurable chunk count.
#
# Usage:
#   ./scripts_qwen3_5/run_export_chunks.sh --chunks 6 --model /path/to/model
#   ./scripts_qwen3_5/run_export_chunks.sh --chunks 8 --model /path/to/model
#
# This does export + combine (no compile — that requires macOS).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CHUNKS=""
MODEL=""
OUTPUT=""
SKIP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks) CHUNKS="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --skip-existing) SKIP="--skip-existing"; shift ;;
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

echo "======================================================================"
echo "  Qwen3.5-4B — Export + Combine ($CHUNKS chunks)"
echo "  Model:  $MODEL"
echo "  Output: $OUTPUT"
echo "  Chunks: $CHUNKS"
echo "======================================================================"

mkdir -p "$OUTPUT"

# Step 1: Export
echo ""
echo "── Step 1/2: Export ($CHUNKS chunks) ──"
python scripts_qwen3_5/export.py --model "$MODEL" --output "$OUTPUT" $SKIP

# Step 2: Combine (dedup)
echo ""
echo "── Step 2/2: Combine ──"
python scripts_qwen3_5/combine.py --input "$OUTPUT" $SKIP

echo ""
echo "======================================================================"
echo "  Export + Combine complete ($CHUNKS chunks)"
echo "  Output: $OUTPUT"
echo "======================================================================"
du -sh "$OUTPUT"/*.mlpackage "$OUTPUT"/combined_LUT4_dedup/*.mlpackage 2>/dev/null | head -20 || true
echo ""
echo "Transfer to Mac, then compile + validate:"
echo "  python scripts_qwen3_5/compile.py --model-dir $OUTPUT"
echo "  python scripts_qwen3_5/validate.py --model-dir $OUTPUT --tokens 40"
