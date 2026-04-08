#!/usr/bin/env bash
# Qwen3.5-4B — Full Pipeline (export → combine → compile → validate)
#
# Produces a self-contained model directory with:
#   embed_lmhead_combined.mlpackage  (2 functions: embed + lmhead, cross-model dedup)
#   combined_LUT4_dedup/chunk{0..8}.mlpackage  (9 FFN chunks, infer + prefill)
#   tokenizer files
#
# Usage:
#   ./scripts_qwen3_5/run_pipeline.sh --model /path/to/Qwen3.5-4B
#   ./scripts_qwen3_5/run_pipeline.sh --model /path/to/Qwen3.5-4B --output /path/to/output
#   ./scripts_qwen3_5/run_pipeline.sh --skip-existing --parallel
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
PARALLEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --output)  OUTPUT="$2"; shift 2 ;;
    --skip-existing) SKIP="--skip-existing"; shift ;;
    --parallel) PARALLEL=true; shift ;;
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
echo "  Qwen3.5-4B — Stable Pipeline ([FLLL] 9-chunk, LUT4 FFN, LUT6 embed+lmhead)"
echo "  Model:  $MODEL"
echo "  Output: $OUTPUT"
echo "  Layout: embed_lmhead_combined.mlpackage + combined_LUT4_dedup/"
if $PARALLEL; then
  echo "  Export: PARALLEL (2 workers)"
fi
echo "======================================================================"

# ── Step 1: Export embeddings (LUT6) + lm_head_nosplit (LUT6) + FFN chunks (LUT4) ──
echo ""
echo "── Step 1/5: Export ──"
if $PARALLEL; then
  # Worker 1: embeddings + lm_head_nosplit + FFN chunks 0-3 (LUT4)
  # Worker 2: FFN chunks 4-8 (LUT4, ffn-only)
  echo "  Launching Worker 1 (embed + lm_head_nosplit + FFN chunks 0-3 LUT4)..."
  python scripts_qwen3_5/export.py --model "$MODEL" --output "$OUTPUT" \
    --chunks 0,1,2,3 --nosplit-lmhead --lut-bits 4 --per-channel 4 \
    $SKIP > "${OUTPUT}/export_worker1.log" 2>&1 &
  W1_PID=$!

  echo "  Launching Worker 2 (FFN chunks 4-8 LUT4, ffn-only)..."
  python scripts_qwen3_5/export.py --model "$MODEL" --output "$OUTPUT" \
    --chunks 4,5,6,7,8 --ffn-only --lut-bits 4 --per-channel 4 \
    $SKIP > "${OUTPUT}/export_worker2.log" 2>&1 &
  W2_PID=$!

  echo "  Worker 1 PID: $W1_PID  |  Worker 2 PID: $W2_PID"
  echo "  Logs: ${OUTPUT}/export_worker{1,2}.log"
  echo ""

  # Monitor loop
  _monitor_interval=60
  while kill -0 $W1_PID 2>/dev/null || kill -0 $W2_PID 2>/dev/null; do
    _w1_rss=$(ps -o rss= -p $W1_PID 2>/dev/null || echo 0)
    _w2_rss=$(ps -o rss= -p $W2_PID 2>/dev/null || echo 0)
    _w1_mb=$(( ${_w1_rss:-0} / 1024 ))
    _w2_mb=$(( ${_w2_rss:-0} / 1024 ))
    _total_mb=$(( _w1_mb + _w2_mb ))
    _ts=$(date '+%H:%M:%S')
    printf "  [%s] Memory — W1: %s MB  W2: %s MB  Total: %s MB\n" \
      "$_ts" "$_w1_mb" "$_w2_mb" "$_total_mb"

    _w1_last=$(grep -E '(Saved|Exporting|skip)' "${OUTPUT}/export_worker1.log" 2>/dev/null | tail -1 || true)
    _w2_last=$(grep -E '(Saved|Exporting|skip)' "${OUTPUT}/export_worker2.log" 2>/dev/null | tail -1 || true)
    [[ -n "$_w1_last" ]] && echo "    W1: $_w1_last"
    [[ -n "$_w2_last" ]] && echo "    W2: $_w2_last"
    sleep $_monitor_interval
  done

  wait $W1_PID; _rc1=$?
  wait $W2_PID; _rc2=$?
  if [[ $_rc1 -ne 0 ]]; then
    echo "ERROR: Worker 1 failed (exit $_rc1). See ${OUTPUT}/export_worker1.log"
    exit 1
  fi
  if [[ $_rc2 -ne 0 ]]; then
    echo "ERROR: Worker 2 failed (exit $_rc2). See ${OUTPUT}/export_worker2.log"
    exit 1
  fi
  echo "  Both export workers completed successfully."
else
  python scripts_qwen3_5/export.py --model "$MODEL" --output "$OUTPUT" \
    --nosplit-lmhead --lut-bits 4 --per-channel 4 $SKIP
fi

# ── Step 2: Combine FFN chunks (LUT4 dedup) + embed+lmhead (cross-model dedup) ──
echo ""
echo "── Step 2/5: Combine ──"
python scripts_qwen3_5/combine.py --input "$OUTPUT" --label LUT4 --combine-embed-lmhead $SKIP

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
echo "  python scripts_qwen3_5/chat_server.py --model-dir $OUTPUT --num-chunks 9 --ctx 2048"
echo ""
echo "Output directory:"
ls -la "$OUTPUT"/embed_lmhead_combined.mlpackage 2>/dev/null && echo "" || true
du -sh "$OUTPUT"/combined_LUT4_dedup/*.mlpackage 2>/dev/null | head -20 || true
echo ""
echo "Pipeline complete!"
