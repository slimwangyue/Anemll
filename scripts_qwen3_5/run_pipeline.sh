#!/usr/bin/env bash
# Qwen3.5-4B — Full Pipeline (export → combine → compile → validate)
#
# Usage:
#   ./scripts_qwen3_5/run_pipeline.sh --model /path/to/Qwen3.5-4B
#   ./scripts_qwen3_5/run_pipeline.sh --model /path/to/Qwen3.5-4B --output /path/to/output
#   ./scripts_qwen3_5/run_pipeline.sh --skip-existing
#
# Environment variables (optional):
#   QWEN35_HF_MODEL  — HuggingFace model path
#   QWEN35_OUTPUT    — Output directory (default: qwen3_5_stable_models/)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Defaults come from config.py via env vars; CLI overrides them
MODEL="${QWEN35_HF_MODEL:-}"
OUTPUT="${QWEN35_OUTPUT:-$REPO_ROOT/qwen3_5_stable_models}"
SKIP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --output)  OUTPUT="$2"; shift 2 ;;
    --skip-existing) SKIP="--skip-existing"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ERROR: Must provide --model /path/to/Qwen3.5-4B (or set QWEN35_HF_MODEL)"
  exit 1
fi

# Export env vars so config.py picks them up
export QWEN35_HF_MODEL="$MODEL"
export QWEN35_OUTPUT="$OUTPUT"

echo "======================================================================"
echo "  Qwen3.5-4B — Full Pipeline"
echo "  Model:  $MODEL"
echo "  Output: $OUTPUT"
echo "======================================================================"

# Step 1: Export
echo ""
echo "── Step 1/5: Export ──"
python scripts_qwen3_5/export.py --model "$MODEL" --output "$OUTPUT" $SKIP

# Step 2: Combine (dedup)
echo ""
echo "── Step 2/5: Combine ──"
python scripts_qwen3_5/combine.py --input "$OUTPUT" $SKIP

# Step 3: Compile
echo ""
echo "── Step 3/5: Compile ──"
python scripts_qwen3_5/compile.py --model-dir "$OUTPUT"

# Step 4: Validate
echo ""
echo "── Step 4/5: Validate ──"
python scripts_qwen3_5/validate.py --model-dir "$OUTPUT" --tokens 40

# Step 5: Done
echo ""
echo "── Step 5/5: Ready ──"
echo "All models exported, combined, compiled, and validated."
echo ""
echo "To start the chat server:"
echo "  python scripts_qwen3_5/chat_server.py --model-dir $OUTPUT"
echo ""
echo "To run profiling:"
echo "  python scripts_qwen3_5/profile.py --export-dir $OUTPUT --skip-cpu-compare"
echo ""
echo "Output directory:"
du -sh "$OUTPUT"/*.mlpackage "$OUTPUT"/combined_LUT*_dedup/*.mlpackage 2>/dev/null | head -20 || true
echo ""
echo "Pipeline complete!"
