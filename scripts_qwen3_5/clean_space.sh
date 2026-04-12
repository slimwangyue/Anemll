rm -rf ~/.Trash/*

rm -rf ~/Library/Developer/Xcode/DerivedData/*
rm -rf ~/Library/Developer/Xcode/Archives/*
xcrun simctl delete unavailable

pip cache purge
rm -rf ~/.cache/pip

du -sh ~/.cache/huggingface ~/.cache/torch 2>/dev/null
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch

find /private/tmp -maxdepth 1 \( \
  -name 'split_*' -o \
  -name 'single_func*' -o \
  -name 'combined_4chunk*' -o \
  -name 'ios_compiled' -o \
  -name 'ane_docs_check' -o \
  -name 'check_embed' -o \
  -name 'prefill_mils' -o \
  -name '*.mlmodelc' -o \
  -name '*.mlpackage' \
\) -exec rm -rf {} \;

sudo rm -rf ~/Library/Caches/com.apple.python3

find /var/folders/r5/dn4v9jhx1cvg33xnvt8nmjxr0000gn/T -mindepth 1 -maxdepth 1 -exec sudo rm -rf {} +