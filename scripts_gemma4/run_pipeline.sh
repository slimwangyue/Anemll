#!/bin/bash
# Gemma4 E4B full pipeline: export → combine → compile → validate
#
# Usage:
#   cd /path/to/Anemll
#   bash scripts_gemma4/run_pipeline.sh [--model /path/to/model] [--output /path/to/output]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Defaults
MODEL="${GEMMA4_HF_MODEL:-$HOME/local_llm/models/google__gemma-4-E4B-it}"
OUTPUT="${GEMMA4_OUTPUT:-gemma4_E4B_lut4ffn_lut6em}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Gemma4 E4B Conversion Pipeline"
echo "============================================================"
echo "Model:  $MODEL"
echo "Output: $OUTPUT"
echo "============================================================"

# Step 1: Export
echo ""
echo ">>> Step 1/4: Export"
python scripts_gemma4/export.py \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --nosplit-lmhead \
    --lut-bits 4 \
    --per-channel 4

# Step 2: Combine
echo ""
echo ">>> Step 2/4: Combine"
python scripts_gemma4/combine.py \
    --model-dir "$OUTPUT" \
    --label LUT4 \
    --combine-embed-lmhead

# Step 3: Compile
echo ""
echo ">>> Step 3/4: Compile"
python scripts_gemma4/compile.py \
    --model-dir "$OUTPUT"

# Step 4: Validate
echo ""
echo ">>> Step 4/4: Validate"
python scripts_gemma4/validate.py \
    --model-dir "$OUTPUT" \
    --hf-model "$MODEL" \
    --tokens 40

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Output: $OUTPUT"
echo "============================================================"
ls -la "$OUTPUT"/ 2>/dev/null || true
