#!/usr/bin/env bash
# Qwen3.5-4B — V4+P2+D2 Production Pipeline (export → combine → compile → validate)
#
# Default features (Milestone 3.3):
#   V4: FP32 for kv_cache_state ops in F-layers, FP16 everywhere else
#   P2: Per-head attention splitting for ANE L2 cache residency (built into model code)
#   D2: Keep F-layer attention Q/K/V/O in FP16 (skip LUT4 for those)
#
# Produces a self-contained model directory with:
#   embed_single.mlpackage, embed_prefill.mlpackage, lm_head_nosplit.mlpackage
#   embed_lmhead_combined.mlpackage  (combined embed + lmhead)
#   ffn_LUT4_chunk{0..8}.mlpackage + prefill_LUT4_chunk{0..8}.mlpackage
#   combined_LUT4_dedup/chunk{0..8}.mlpackage  (9 FFN chunks, infer + prefill)
#   tokenizer files
#
# Prerequisites:
#   - HuggingFace model at models/Qwen__Qwen3.5-4B (or provide --model)
#   - Python venv with coremltools >= 9.0, transformers
#     Default: .venv_qwen35/bin/python
#
# Usage:
#   ./scripts_qwen3_5/run_pipeline_4B_V4.sh
#   ./scripts_qwen3_5/run_pipeline_4B_V4.sh --model /path/to/Qwen3.5-4B
#   ./scripts_qwen3_5/run_pipeline_4B_V4.sh --output /path/to/output
#   ./scripts_qwen3_5/run_pipeline_4B_V4.sh --skip-existing          # skip already-exported chunks
#   ./scripts_qwen3_5/run_pipeline_4B_V4.sh --skip-export             # reuse existing exports
#   ./scripts_qwen3_5/run_pipeline_4B_V4.sh --skip-validate           # skip validation at the end
#   ./scripts_qwen3_5/run_pipeline_4B_V4.sh --chunks 2,5              # export only specific chunks
#   ./scripts_qwen3_5/run_pipeline_4B_V4.sh --no-d2                   # disable D2 (all weights LUT4)
#
# Environment variables (optional):
#   QWEN35_HF_MODEL   — HuggingFace model path
#   QWEN35_PYTHON     — Python interpreter path
#   TMPDIR            — Temp directory (set to SSD path on low-disk boot volumes)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ──
MODEL="${QWEN35_HF_MODEL:-$REPO_ROOT/models/Qwen__Qwen3.5-4B}"
OUTPUT="${QWEN35_V4_OUTPUT:-$REPO_ROOT/qwen3_5_4B_v4_lut4}"
PYTHON="${QWEN35_PYTHON:-$REPO_ROOT/.venv_qwen35/bin/python}"
SKIP_EXISTING=""
SKIP_EXPORT=false
SKIP_VALIDATE=false
CHUNKS=""
TOKENS=40
LUT_BITS=4
PER_CHANNEL=4
NO_D2=""
E235=""
CTX=""
BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2 ;;
    --output)         OUTPUT="$2"; shift 2 ;;
    --python)         PYTHON="$2"; shift 2 ;;
    --skip-existing)  SKIP_EXISTING="--skip-existing"; shift ;;
    --skip-export)    SKIP_EXPORT=true; shift ;;
    --skip-validate)  SKIP_VALIDATE=true; shift ;;
    --no-d2)          NO_D2="--no-d2"; shift ;;
    --e235)           E235="--e235"; shift ;;
    --chunks)         CHUNKS="$2"; shift 2 ;;
    --tokens)         TOKENS="$2"; shift 2 ;;
    --lut-bits)       LUT_BITS="$2"; shift 2 ;;
    --per-channel)    PER_CHANNEL="$2"; shift 2 ;;
    --ctx)            CTX="$2"; shift 2 ;;
    --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Validate prerequisites ──
if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: Model directory not found: $MODEL"
  echo "  Download: huggingface-cli download Qwen/Qwen3.5-4B --local-dir $MODEL"
  echo "  Or provide --model /path/to/Qwen3.5-4B (or set QWEN35_HF_MODEL)"
  exit 1
fi

if ! "$PYTHON" --version &>/dev/null 2>&1; then
  echo "ERROR: Python not found: $PYTHON"
  echo "  Provide --python /path/to/python (or set QWEN35_PYTHON)"
  exit 1
fi

# Ensure TMPDIR is set for large exports
export TMPDIR="${TMPDIR:-/tmp}"

# Read CTX and BATCH_SIZE from config.py if not set via CLI
if [[ -z "$CTX" ]]; then
  CTX=$("$PYTHON" -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from config import CTX; print(CTX)")
fi
if [[ -z "$BATCH_SIZE" ]]; then
  BATCH_SIZE=$("$PYTHON" -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from config import BATCH_SIZE; print(BATCH_SIZE)")
fi

# ── Model info ──
NUM_LAYERS=32
NUM_CHUNKS=9
NUM_F_LAYERS=8
NUM_L_LAYERS=24

_POLICY="V4+P2+D2 (kv_cache→FP32, per-head attn, F-attn→FP16)"
[[ -n "$NO_D2" ]] && _POLICY="V4+P2 (kv_cache→FP32, per-head attn, all weights LUT${LUT_BITS})"

echo "======================================================================"
echo "  Qwen3.5-4B — V4+P2+D2 Production Pipeline"
echo "  Policy:     $_POLICY"
echo "  Model:      $MODEL"
echo "  Output:     $OUTPUT"
echo "  Python:     $PYTHON"
echo "  TMPDIR:     $TMPDIR"
echo "  Layers:     $NUM_LAYERS ($NUM_L_LAYERS L + $NUM_F_LAYERS F)"
echo "  Chunks:     $NUM_CHUNKS ([FLLL] 9-chunk: LLL,FLLL×7,F)"
echo "  CTX:        $CTX"
echo "  Batch:      $BATCH_SIZE"
echo "  Quant:      LUT${LUT_BITS} gs=${PER_CHANNEL}"
echo "  Layout:     embed_lmhead_combined + combined_LUT${LUT_BITS}_dedup/"
echo "======================================================================"

mkdir -p "$OUTPUT"

# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Export (embed + lmhead + FFN chunks with V4+P2+D2)
# ═══════════════════════════════════════════════════════════════════════
#
# Uses scripts_qwen3_5/export.py which:
#   - V4: keeps kv_cache_state ops in FP32 for F-layers (default ON)
#   - P2: per-head attention splitting (built into model code)
#   - D2: keeps F-layer attention Q/K/V/O in FP16 (default ON)
#   - Exports embed + lm_head + 9 decode + 9 prefill .mlpackage files
#
# Output: $OUTPUT/ffn_LUT4_chunk{0..8}.mlpackage, prefill_LUT4_chunk{0..8}.mlpackage,
#         embed_*.mlpackage, lm_head_nosplit.mlpackage

if $SKIP_EXPORT; then
  echo ""
  echo "── Step 1/4: Export SKIPPED (--skip-export) ──"
else
  echo ""
  echo "── Step 1/4: Export V4+P2+D2 (embed + lmhead + $NUM_CHUNKS FFN chunks × 2 phases) ──"

  _export_args="--model $MODEL --output $OUTPUT"
  _export_args="$_export_args --lut-bits $LUT_BITS --per-channel $PER_CHANNEL"
  _export_args="$_export_args --ctx $CTX --batch-size $BATCH_SIZE"
  _export_args="$_export_args --nosplit-lmhead"
  [[ -n "$SKIP_EXISTING" ]] && _export_args="$_export_args --skip-existing"
  [[ -n "$CHUNKS" ]] && _export_args="$_export_args --chunks $CHUNKS"
  [[ -n "$NO_D2" ]] && _export_args="$_export_args --no-d2"
  [[ -n "$E235" ]] && _export_args="$_export_args --e235"

  PYTHONUNBUFFERED=1 "$PYTHON" scripts_qwen3_5/export.py \
    $_export_args \
    2>&1 | tee "${OUTPUT}/export_v4.log"

  echo "  Export complete → $OUTPUT/"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Combine — decode+prefill → multifunction dedup models
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "── Step 2/4: Combine (dedup) ──"
"$PYTHON" scripts_qwen3_5/combine.py \
  --input "$OUTPUT" --label "LUT${LUT_BITS}" --combine-embed-lmhead \
  --batch-size "$BATCH_SIZE" \
  $SKIP_EXISTING \
  2>&1 | tee "${OUTPUT}/combine_v4.log"

echo "  Combined → $OUTPUT/combined_LUT${LUT_BITS}_dedup/"

# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Compile (.mlpackage → .mlmodelc for deployment)
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "── Step 3/4: Compile ──"
"$PYTHON" scripts_qwen3_5/compile.py \
  --model-dir "$OUTPUT" \
  2>&1 | tee "${OUTPUT}/compile_v4.log"

# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Validate
# ═══════════════════════════════════════════════════════════════════════

if $SKIP_VALIDATE; then
  echo ""
  echo "── Step 4/4: Validate SKIPPED (--skip-validate) ──"
else
  echo ""
  echo "── Step 4/4: Validate ──"
  "$PYTHON" scripts_qwen3_5/validate.py \
    --model-dir "$OUTPUT" \
    --tokens "$TOKENS" \
    --label "LUT${LUT_BITS}" \
    --skip-separate \
    2>&1 | tee "${OUTPUT}/validate_v4.log"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "======================================================================"
echo "  V4+P2+D2 Pipeline Complete!"
echo ""
echo "  Model directory:  $OUTPUT"
echo "  Combined dedup:   $OUTPUT/combined_LUT${LUT_BITS}_dedup/"
echo ""
echo "  To start the chat server:"
echo "    $PYTHON scripts_qwen3_5/chat_server.py \\"
echo "      --model-dir $OUTPUT \\"
echo "      --num-chunks $NUM_CHUNKS --ctx 4096 --port 8080"
echo ""
du -sh "$OUTPUT"/combined_LUT${LUT_BITS}_dedup/*.mlpackage 2>/dev/null | head -12 || true
echo ""
echo "  Policy: $_POLICY"
echo "======================================================================"
