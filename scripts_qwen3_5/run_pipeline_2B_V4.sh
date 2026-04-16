#!/usr/bin/env bash
# Qwen3.5-2B — V4+P2+D2 Production Pipeline (export → combine → compile → validate)
#
# Default features (Milestone 3.3):
#   V4: FP32 for kv_cache_state ops in F-layers, FP16 everywhere else
#   P2: Per-head attention splitting for ANE L2 cache residency (built into model code)
#   D2: Keep F-layer attention Q/K/V/O in FP16 (skip LUT4 for those)
#
# Produces a self-contained model directory with:
#   embed_single.mlpackage, embed_prefill.mlpackage, embed_lmhead_combined.mlpackage
#   combined_LUT4_dedup/chunk{0..6}.mlpackage  (7 FFN chunks, infer + prefill, V4 precision)
#   tokenizer files, meta.yaml
#
# Architecture: Qwen3.5-2B — 24 layers, [LLLFLLLFLLLFLLLFLLLFLLF]
#   - 18 L (linear_attention), 6 F (full_attention)
#   - full_attention_interval = 4 → F at layers {3,7,11,15,19,23}
#   - [FLLL] 7-chunk partition: [LLL, FLLL, FLLL, FLLL, FLLL, FLLL, F]
#   - hidden_size=2048, intermediate_size=6144, num_attention_heads=8, num_kv_heads=2
#
# Prerequisites:
#   - HuggingFace model at models/Qwen__Qwen3.5-2B (or provide --model)
#   - Python venv with coremltools >= 9.0, transformers
#     Default: .venv_qwen35/bin/python
#
# Usage:
#   ./scripts_qwen3_5/run_pipeline_2B_V4.sh
#   ./scripts_qwen3_5/run_pipeline_2B_V4.sh --model /path/to/Qwen3.5-2B
#   ./scripts_qwen3_5/run_pipeline_2B_V4.sh --output /path/to/output
#   ./scripts_qwen3_5/run_pipeline_2B_V4.sh --skip-existing          # skip already-exported chunks
#   ./scripts_qwen3_5/run_pipeline_2B_V4.sh --skip-export             # reuse existing exports
#   ./scripts_qwen3_5/run_pipeline_2B_V4.sh --skip-validate           # skip validation at the end
#   ./scripts_qwen3_5/run_pipeline_2B_V4.sh --chunks 2,5              # export only specific chunks
#   ./scripts_qwen3_5/run_pipeline_2B_V4.sh --no-d2                   # disable D2 (all weights LUT4)
#
# Environment variables (optional):
#   QWEN35_HF_MODEL   — HuggingFace model path
#   QWEN35_PYTHON     — Python interpreter path
#   TMPDIR            — Temp directory (set to SSD path on low-disk boot volumes)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Key env var: tell config.py to use 2B presets ──
export QWEN35_MODEL_SIZE=2B

# ── Defaults ──
MODEL="${QWEN35_HF_MODEL:-$REPO_ROOT/models/Qwen__Qwen3.5-2B}"
OUTPUT="${QWEN35_2B_OUTPUT:-$REPO_ROOT/qwen3_5_2b_v4_lut4}"
PYTHON="${QWEN35_PYTHON:-$REPO_ROOT/.venv_qwen35/bin/python}"
SKIP_EXISTING=""
SKIP_EXPORT=false
SKIP_VALIDATE=false
CHUNKS=""
TOKENS=40
LUT_BITS=4
PER_CHANNEL=4
NO_D2=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2 ;;
    --output)         OUTPUT="$2"; shift 2 ;;
    --python)         PYTHON="$2"; shift 2 ;;
    --skip-existing)  SKIP_EXISTING="--skip-existing"; shift ;;
    --skip-export)    SKIP_EXPORT=true; shift ;;
    --skip-validate)  SKIP_VALIDATE=true; shift ;;
    --no-d2)          NO_D2="--no-d2"; shift ;;
    --chunks)         CHUNKS="$2"; shift 2 ;;
    --tokens)         TOKENS="$2"; shift 2 ;;
    --lut-bits)       LUT_BITS="$2"; shift 2 ;;
    --per-channel)    PER_CHANNEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Validate prerequisites ──
if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: Model directory not found: $MODEL"
  echo "  Download: huggingface-cli download Qwen/Qwen3.5-2B --local-dir $MODEL"
  echo "  Or provide --model /path/to/Qwen3.5-2B (or set QWEN35_HF_MODEL)"
  exit 1
fi

if ! "$PYTHON" --version &>/dev/null 2>&1; then
  echo "ERROR: Python not found: $PYTHON"
  echo "  Provide --python /path/to/python (or set QWEN35_PYTHON)"
  exit 1
fi

# Ensure TMPDIR is set for large exports
export TMPDIR="${TMPDIR:-/tmp}"

# ── Model info ──
NUM_LAYERS=24
NUM_CHUNKS=7
NUM_F_LAYERS=6
NUM_L_LAYERS=18

_POLICY="V4+P2+D2 (kv_cache→FP32, per-head attn, F-attn→FP16)"
[[ -n "$NO_D2" ]] && _POLICY="V4+P2 (kv_cache→FP32, per-head attn, all weights LUT${LUT_BITS})"

echo "======================================================================"
echo "  Qwen3.5-2B — V4+P2+D2 Production Pipeline"
echo "  Policy:     $_POLICY"
echo "  Model:      $MODEL"
echo "  Output:     $OUTPUT"
echo "  Python:     $PYTHON"
echo "  TMPDIR:     $TMPDIR"
echo "  Layers:     $NUM_LAYERS ($NUM_L_LAYERS L + $NUM_F_LAYERS F)"
echo "  Chunks:     $NUM_CHUNKS ([FLLL] 7-chunk: LLL,FLLL,FLLL,FLLL,FLLL,FLLL,F)"
echo "  Quant:      LUT${LUT_BITS} gs=${PER_CHANNEL}"
echo "  Layout:     embed_lmhead_combined + combined_LUT${LUT_BITS}_dedup/"
echo "======================================================================"

mkdir -p "$OUTPUT"

# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Export (embed + lmhead + FFN chunks with V4+P2+D2)
# ═══════════════════════════════════════════════════════════════════════
#
# Uses scripts_qwen3_5/export.py (V4+D2 default ON).
# config.py reads QWEN35_MODEL_SIZE=2B to set NUM_CHUNKS=7, CHUNK_RANGES for 24 layers.
#
# Output: $OUTPUT/embed_*.mlpackage, ffn_LUT4_chunk{0..6}.mlpackage, prefill_LUT4_chunk{0..6}.mlpackage

if $SKIP_EXPORT; then
  echo ""
  echo "── Step 1/4: Export SKIPPED (--skip-export) ──"
else
  echo ""
  echo "── Step 1/4: Export V4+P2+D2 (embed + lmhead + $NUM_CHUNKS FFN chunks × 2 phases) ──"

  _export_args="--model $MODEL --output $OUTPUT"
  _export_args="$_export_args --lut-bits $LUT_BITS --per-channel $PER_CHANNEL"
  _export_args="$_export_args --nosplit-lmhead"
  [[ -n "$SKIP_EXISTING" ]] && _export_args="$_export_args --skip-existing"
  [[ -n "$CHUNKS" ]] && _export_args="$_export_args --chunks $CHUNKS"
  [[ -n "$NO_D2" ]] && _export_args="$_export_args --no-d2"

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
echo "  V4+P2+D2 Pipeline Complete — Qwen3.5-2B"
echo ""
echo "  Model directory: $OUTPUT"
echo "  Combined dedup:  $OUTPUT/combined_LUT${LUT_BITS}_dedup/"
echo ""
echo "  To start the chat server:"
echo "    QWEN35_MODEL_SIZE=2B $PYTHON scripts_qwen3_5/chat_server.py \\"
echo "      --model-dir $OUTPUT \\"
echo "      --num-chunks $NUM_CHUNKS --ctx 4096 --port 8080"
echo ""
du -sh "$OUTPUT"/combined_LUT${LUT_BITS}_dedup/*.mlpackage 2>/dev/null | head -10 || true
echo ""
echo "  Policy: $_POLICY"
echo "======================================================================"
