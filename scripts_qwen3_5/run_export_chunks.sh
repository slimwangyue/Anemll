#!/usr/bin/env bash
# Export + Combine for Qwen3.5-4B with configurable chunk count.
#
# Usage:
#   ./scripts_qwen3_5/run_export_chunks.sh --chunks 6 --model /path/to/model
#   ./scripts_qwen3_5/run_export_chunks.sh --chunks 4 --model /path/to/model --only-chunk 2
#   ./scripts_qwen3_5/run_export_chunks.sh --chunks 4 --model /path/to/model --skip-existing
#
# Calls export.py (step 1) and combine.py (step 2) with the right env vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CHUNKS=""
MODEL=""
OUTPUT=""
SKIP_FLAG=""
ONLY_CHUNK_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks) CHUNKS="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --skip-existing) SKIP_FLAG="--skip-existing"; shift ;;
    --only-chunk) ONLY_CHUNK_FLAG="--only-chunk $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHUNKS" ]]; then
  echo "ERROR: Must provide --chunks N (e.g. --chunks 6)"
  exit 1
fi

if [[ -z "$MODEL" ]]; then
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

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$REPO_ROOT/qwen3_5_${CHUNKS}chunk_models"
fi

# Set env vars so config.py picks up the chunk count and paths
export QWEN35_NUM_CHUNKS="$CHUNKS"
export QWEN35_HF_MODEL="$MODEL"
export QWEN35_OUTPUT="$OUTPUT"

echo "======================================================================"
echo "  Qwen3.5-4B — Export + Combine ($CHUNKS chunks, static prefill)"
echo "  Model:  $MODEL"
echo "  Output: $OUTPUT"
echo "  Chunks: $CHUNKS"
echo "======================================================================"

# Step 1: Export
echo ""
echo "── Step 1/2: Export ──"
python "$SCRIPT_DIR/export.py" \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --static-prefill \
  $SKIP_FLAG $ONLY_CHUNK_FLAG

# Step 2: Combine (dedup)
echo ""
echo "── Step 2/2: Combine ──"
python "$SCRIPT_DIR/combine.py" \
  --input "$OUTPUT" \
  $SKIP_FLAG $ONLY_CHUNK_FLAG

echo ""
echo "======================================================================"
echo "  Export + Combine complete ($CHUNKS chunks, static prefill)"
echo "  Output: $OUTPUT"
echo "======================================================================"
du -sh "$OUTPUT"/*.mlpackage "$OUTPUT"/combined_LUT*_dedup/*.mlpackage 2>/dev/null | head -20 || true
echo ""
echo "Next — compile:"
echo "  python scripts_qwen3_5/compile.py --model-dir $OUTPUT"
