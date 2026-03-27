#!/bin/bash
# Deploy ANEMLL stable Qwen3.5-4B models to the iOS app's Models.bundle
#
# This copies the compiled models into the iOS app so it can compile/load them.
# The iOS app expects:
#   Models.bundle/embeddings.mlpackage
#   Models.bundle/lm_head_logits.mlpackage (or lm_head.mlpackage)
#   Models.bundle/chunk{0..3}.mlpackage
#   Models.bundle/tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
#
# Usage:
#   ./scripts/deploy_to_ios.sh [--ios-app-path /path/to/ios/app]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STABLE="$REPO_DIR/qwen3_5_stable_models"

# Default iOS app path
IOS_APP="${IOS_APP_PATH:-/Users/yw68/local_llm/local_llm}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ios-app-path) IOS_APP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Find or create Models.bundle inside the iOS app
BUNDLE="$IOS_APP/Models.bundle"
if [[ ! -d "$BUNDLE" ]]; then
    echo "Creating Models.bundle at $BUNDLE"
    mkdir -p "$BUNDLE"
fi

echo "============================================="
echo " Deploying ANEMLL models to iOS app"
echo " Source: $STABLE"
echo " Target: $BUNDLE"
echo "============================================="

# Copy embeddings
echo "Copying embeddings.mlpackage..."
rsync -a --delete "$STABLE/embeddings.mlpackage/" "$BUNDLE/embeddings.mlpackage/"

# Copy lm_head_logits (preferred by iOS app for sampling)
echo "Copying lm_head_logits.mlpackage..."
rsync -a --delete "$STABLE/lm_head_logits.mlpackage/" "$BUNDLE/lm_head_logits.mlpackage/"

# Copy combined FFN chunks (chunk0..chunk3)
for i in 0 1 2 3; do
    SRC="$STABLE/combined_LUT4_dedup/chunk${i}.mlpackage"
    DST="$BUNDLE/chunk${i}.mlpackage"
    if [[ -d "$SRC" ]]; then
        echo "Copying chunk${i}.mlpackage..."
        rsync -a --delete "$SRC/" "$DST/"
    else
        echo "WARNING: $SRC not found, skipping"
    fi
done

# Copy tokenizer files
echo "Copying tokenizer files..."
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    if [[ -f "$STABLE/$f" ]]; then
        cp "$STABLE/$f" "$BUNDLE/$f"
    fi
done

echo ""
echo "============================================="
echo " Deployment complete!"
echo ""
echo " Models in $BUNDLE:"
ls -1 "$BUNDLE" | sed 's/^/   /'
echo ""
echo " Total size: $(du -sh "$BUNDLE" | cut -f1)"
echo "============================================="
echo ""
echo "Next steps:"
echo "  1. Open the iOS Xcode project"
echo "  2. Ensure Models.bundle is in the project's Copy Bundle Resources"
echo "  3. Build and run — the app will compile .mlpackage to .mlmodelc on first launch"
echo "  4. The AnemllModelCompiler handles compilation automatically"
