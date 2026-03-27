#!/bin/bash
set -e

BUNDLE="/Users/yw68/local_llm/local_llm/local_llm/Resources/Models.bundle"
STABLE="/Users/yw68/Anemll/qwen3_5_stable_models"
BACKUP="${BUNDLE}_pre_lut6_backup_20260327"

echo "=== Deploying stable LUT6 models to iOS app ==="
echo "Source: $STABLE"
echo "Target: $BUNDLE"
echo ""

# Backup
if [ ! -d "$BACKUP" ]; then
    echo "Backing up current Models.bundle..."
    cp -a "$BUNDLE" "$BACKUP"
    echo "Backup: $BACKUP ($(du -sh "$BACKUP" | cut -f1))"
else
    echo "Backup already exists: $BACKUP"
fi

# Clear old models
echo ""
echo "Removing old models from bundle..."
rm -rf "$BUNDLE"/*

# Copy embeddings
echo "Copying embeddings.mlpackage..."
rsync -a "$STABLE/embeddings.mlpackage/" "$BUNDLE/embeddings.mlpackage/"

# Copy lm_head_logits (for sampling support)
echo "Copying lm_head_logits.mlpackage..."
rsync -a "$STABLE/lm_head_logits.mlpackage/" "$BUNDLE/lm_head_logits.mlpackage/"

# Copy combined FFN chunks (chunk0..chunk3)
for i in 0 1 2 3; do
    echo "Copying chunk${i}.mlpackage..."
    rsync -a "$STABLE/combined_LUT4_dedup/chunk${i}.mlpackage/" "$BUNDLE/chunk${i}.mlpackage/"
done

# Copy tokenizer files
echo "Copying tokenizer files..."
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    if [ -f "$STABLE/$f" ]; then
        cp "$STABLE/$f" "$BUNDLE/$f"
    fi
done

echo ""
echo "=== Deployment complete ==="
echo "Contents:"
ls -1 "$BUNDLE"
echo ""
echo "Total size: $(du -sh "$BUNDLE" | cut -f1)"
echo ""
echo "NOTE: The iOS app will need to recompile .mlpackage -> .mlmodelc on next launch."
echo "Any existing compiled caches should be cleared."
