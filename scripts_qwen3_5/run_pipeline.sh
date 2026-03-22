#!/usr/bin/env bash
# Qwen3.5-4B Milestone 1.1 — Full Pipeline
#
# Usage:
#   ./scripts_qwen3_5/run_pipeline.sh
#   ./scripts_qwen3_5/run_pipeline.sh --model /path/to/Qwen3.5-4B --output /path/to/output
#   ./scripts_qwen3_5/run_pipeline.sh --skip-existing
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B}"
OUTPUT="${OUTPUT:-/Users/yw68/qwen35_milestone1_1}"
SKIP=""
HF_PATH="$MODEL"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; HF_PATH="$2"; shift 2 ;;
    --output)  OUTPUT="$2"; shift 2 ;;
    --skip-existing) SKIP="--skip-existing"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "======================================================================"
echo "  Qwen3.5-4B Milestone 1.1 — Full Pipeline"
echo "  Model:  $MODEL"
echo "  Output: $OUTPUT"
echo "======================================================================"

# Step 1: Export
echo ""
echo "── Step 1/4: Export ──"
python scripts_qwen3_5/export.py --model "$MODEL" --output "$OUTPUT" $SKIP

# Step 2: Combine
echo ""
echo "── Step 2/4: Combine ──"
python scripts_qwen3_5/combine.py --input "$OUTPUT" $SKIP

# Step 3: Compile
echo ""
echo "── Step 3/4: Compile ──"
python scripts_qwen3_5/compile.py --model-dir "$OUTPUT"

# Step 4: Chat server
echo ""
echo "── Step 4/4: Ready ──"
echo "All models exported, combined, and compiled."
echo ""
echo "To start the chat server:"
echo "  python tests/dev/qwen35_chat_server.py --model-dir $OUTPUT --hf-model $HF_PATH"
echo ""
echo "Output directory: $OUTPUT"
ls -la "$OUTPUT"/*.mlpackage 2>/dev/null | head -20 || true
echo ""
echo "Pipeline complete!"
