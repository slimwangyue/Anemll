#!/usr/bin/env bash
# Qwen3.5-4B — V4 Precision Pipeline (export → assemble → combine → compile → validate)
#
# V4 precision policy: everything FP16 except kv_cache_state ops in F layers (FP32).
# This achieves 57.5% fewer casts vs full FP32 while maintaining 100% self-consistency.
#
# Produces a self-contained model directory with:
#   embed_lmhead_combined.mlpackage  (embed + lmhead, from FP32 reference)
#   combined_LUT4_dedup/chunk{0..8}.mlpackage  (9 FFN chunks, infer + prefill, V4 precision)
#   tokenizer files
#
# Prerequisites:
#   - FP32 reference model already exported (embed_lmhead + tokenizer)
#     Default: qwen3_5_stable_lut4ffn_lut6em_fp32/
#   - Python venv with coremltools >= 9.0, transformers
#     Default: .venv_qwen35/bin/python
#
# Usage:
#   ./scripts_qwen3_5/run_pipeline_V4.sh
#   ./scripts_qwen3_5/run_pipeline_V4.sh --model /path/to/Qwen3.5-4B
#   ./scripts_qwen3_5/run_pipeline_V4.sh --output /path/to/output
#   ./scripts_qwen3_5/run_pipeline_V4.sh --skip-existing          # skip already-exported chunks
#   ./scripts_qwen3_5/run_pipeline_V4.sh --skip-export             # reuse existing V4 exports
#   ./scripts_qwen3_5/run_pipeline_V4.sh --skip-validate           # skip validation at the end
#   ./scripts_qwen3_5/run_pipeline_V4.sh --chunks 2,5              # export only specific chunks
#
# Environment variables (optional):
#   QWEN35_HF_MODEL   — HuggingFace model path
#   QWEN35_FP32_REF   — FP32 reference model directory (for embed/lmhead/tokenizer)
#   QWEN35_PYTHON     — Python interpreter path
#   TMPDIR            — Temp directory (set to SSD path on low-disk boot volumes)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ──
MODEL="${QWEN35_HF_MODEL:-$REPO_ROOT/models/Qwen__Qwen3.5-4B}"
OUTPUT="${QWEN35_V4_OUTPUT:-$REPO_ROOT/qwen3_5_v4_lut4ffn_lut6em}"
FP32_REF="${QWEN35_FP32_REF:-$REPO_ROOT/qwen3_5_stable_lut4ffn_lut6em_fp32}"
PYTHON="${QWEN35_PYTHON:-$REPO_ROOT/.venv_qwen35/bin/python}"
V4_ARTIFACT="$REPO_ROOT/artifacts/v4_all_chunks"
SKIP_EXISTING=""
SKIP_EXPORT=false
SKIP_VALIDATE=false
CHUNKS=""
TOKENS=40

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2 ;;
    --output)         OUTPUT="$2"; shift 2 ;;
    --fp32-ref)       FP32_REF="$2"; shift 2 ;;
    --python)         PYTHON="$2"; shift 2 ;;
    --skip-existing)  SKIP_EXISTING="--skip-existing"; shift ;;
    --skip-export)    SKIP_EXPORT=true; shift ;;
    --skip-validate)  SKIP_VALIDATE=true; shift ;;
    --chunks)         CHUNKS="$2"; shift 2 ;;
    --tokens)         TOKENS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Validate prerequisites ──
if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: Model directory not found: $MODEL"
  echo "  Provide --model /path/to/Qwen3.5-4B (or set QWEN35_HF_MODEL)"
  exit 1
fi

if [[ ! -d "$FP32_REF" ]]; then
  echo "ERROR: FP32 reference model not found: $FP32_REF"
  echo "  Run the standard pipeline first (run_pipeline.sh --fp32-compute), or set --fp32-ref"
  exit 1
fi

if ! "$PYTHON" --version &>/dev/null 2>&1; then
  echo "ERROR: Python not found: $PYTHON"
  echo "  Provide --python /path/to/python (or set QWEN35_PYTHON)"
  exit 1
fi

# Ensure TMPDIR is set for large exports
export TMPDIR="${TMPDIR:-/tmp}"

echo "======================================================================"
echo "  Qwen3.5-4B — V4 Precision Pipeline"
echo "  Policy:     L layers → FP16 | F layers → FP16 (kv_cache_state → FP32)"
echo "  Model:      $MODEL"
echo "  Output:     $OUTPUT"
echo "  FP32 ref:   $FP32_REF"
echo "  Artifacts:  $V4_ARTIFACT"
echo "  Python:     $PYTHON"
echo "  TMPDIR:     $TMPDIR"
echo "  Layout:     embed_lmhead_combined.mlpackage + combined_LUT4_dedup/"
echo "======================================================================"

mkdir -p "$OUTPUT"

# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Export FFN chunks with V4 precision policy
# ═══════════════════════════════════════════════════════════════════════
#
# Uses tests/dev/all_chunks_v4_kvcache_fp32.py which:
#   - Loads the HF model once
#   - For each chunk: creates a V4 op_selector that keeps kv_cache_state FP32
#   - Exports both decode and prefill .mlpackage files
#   - Saves audit logs (selected_ops_decode.json, selected_ops_prefill.json)
#
# Output: artifacts/v4_all_chunks/chunk_{0..8}/{decode,prefill}.mlpackage

if $SKIP_EXPORT; then
  echo ""
  echo "── Step 1/5: Export SKIPPED (--skip-export) ──"
  # Verify chunks exist
  _missing=0
  for ci in $(seq 0 8); do
    if [[ ! -d "$V4_ARTIFACT/chunk_${ci}/decode.mlpackage" ]] || \
       [[ ! -d "$V4_ARTIFACT/chunk_${ci}/prefill.mlpackage" ]]; then
      echo "  WARNING: chunk $ci missing from $V4_ARTIFACT/chunk_${ci}/"
      _missing=$((_missing + 1))
    fi
  done
  if [[ $_missing -gt 0 ]]; then
    echo "  $__missing chunks missing — consider running without --skip-export"
  fi
else
  echo ""
  echo "── Step 1/5: Export V4 FFN chunks (9 chunks × 2 phases) ──"

  _export_args="--skip-validate --tokens $TOKENS"
  [[ -n "$SKIP_EXISTING" ]] && _export_args="$_export_args --skip-existing"
  [[ -n "$CHUNKS" ]] && _export_args="$_export_args --chunks $CHUNKS"

  PYTHONUNBUFFERED=1 "$PYTHON" tests/dev/all_chunks_v4_kvcache_fp32.py \
    $_export_args \
    2>&1 | tee "${OUTPUT}/export_v4.log"

  echo "  V4 FFN export complete → $V4_ARTIFACT/chunk_{0..8}/"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Assemble — stage V4 chunks + embed/lmhead + tokenizer
# ═══════════════════════════════════════════════════════════════════════

ASSEMBLED="$V4_ARTIFACT/assembled"
echo ""
echo "── Step 2/5: Assemble staging directory ──"

mkdir -p "$ASSEMBLED"

# Symlink embed/lmhead from FP32 reference (these are precision-agnostic)
for f in embed_single.mlpackage embed_lmhead_combined.mlpackage embed_prefill.mlpackage lm_head_nosplit.mlpackage; do
  _src="$FP32_REF/$f"
  _dst="$ASSEMBLED/$f"
  if [[ -e "$_src" ]] && [[ ! -L "$_dst" ]]; then
    ln -sf "$(cd "$(dirname "$_src")" && pwd)/$(basename "$_src")" "$_dst"
    echo "  Linked $f"
  fi
done

# Copy tokenizer files
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  _src="$FP32_REF/$f"
  _dst="$ASSEMBLED/$f"
  if [[ -f "$_src" ]] && [[ ! -f "$_dst" ]]; then
    cp "$_src" "$_dst"
    echo "  Copied $f"
  fi
done

# Symlink V4 FFN decode + prefill chunks into assembled dir
for ci in $(seq 0 8); do
  _dec_src="$V4_ARTIFACT/chunk_${ci}/decode.mlpackage"
  _pf_src="$V4_ARTIFACT/chunk_${ci}/prefill.mlpackage"
  _dec_dst="$ASSEMBLED/ffn_LUT4_chunk${ci}.mlpackage"
  _pf_dst="$ASSEMBLED/prefill_LUT4_chunk${ci}.mlpackage"

  if [[ -d "$_dec_src" ]]; then
    ln -sf "$(cd "$(dirname "$_dec_src")" && pwd)/$(basename "$_dec_src")" "$_dec_dst" 2>/dev/null || true
  fi
  if [[ -d "$_pf_src" ]]; then
    ln -sf "$(cd "$(dirname "$_pf_src")" && pwd)/$(basename "$_pf_src")" "$_pf_dst" 2>/dev/null || true
  fi
done
echo "  Assembled: $ASSEMBLED"

# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Combine — decode+prefill → multifunction dedup models
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "── Step 3/5: Combine (dedup) ──"
"$PYTHON" scripts_qwen3_5/combine.py \
  --input "$ASSEMBLED" --label LUT4 --combine-embed-lmhead \
  $SKIP_EXISTING \
  2>&1 | tee "${OUTPUT}/combine_v4.log"

echo "  Combined → $ASSEMBLED/combined_LUT4_dedup/"

# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Compile (.mlpackage → .mlmodelc for deployment)
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "── Step 4/5: Compile ──"
"$PYTHON" scripts_qwen3_5/compile.py \
  --model-dir "$ASSEMBLED" \
  2>&1 | tee "${OUTPUT}/compile_v4.log"

# ═══════════════════════════════════════════════════════════════════════
#  Step 5: Validate
# ═══════════════════════════════════════════════════════════════════════

if $SKIP_VALIDATE; then
  echo ""
  echo "── Step 5/5: Validate SKIPPED (--skip-validate) ──"
else
  echo ""
  echo "── Step 5/5: Validate ──"
  "$PYTHON" scripts_qwen3_5/validate.py \
    --model-dir "$ASSEMBLED" \
    --tokens "$TOKENS" \
    2>&1 | tee "${OUTPUT}/validate_v4.log"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "======================================================================"
echo "  V4 Pipeline Complete!"
echo ""
echo "  Assembled model: $ASSEMBLED"
echo "  Combined dedup:  $ASSEMBLED/combined_LUT4_dedup/"
echo ""
echo "  To start the chat server:"
echo "    $PYTHON scripts_qwen3_5/chat_server.py \\"
echo "      --model-dir $ASSEMBLED \\"
echo "      --num-chunks 9 --ctx 2048 --port 8080"
echo ""
du -sh "$ASSEMBLED"/combined_LUT4_dedup/*.mlpackage 2>/dev/null | head -12 || true
echo ""
echo "  V4 precision policy:"
echo "    - 24 L layers (linear-attention): full FP16"
echo "    - 8 F layers (full-attention):    FP16, kv_cache_state → FP32"
echo "    - 57.5% fewer casts vs full FP32, 100% self-consistency"
echo "======================================================================"
