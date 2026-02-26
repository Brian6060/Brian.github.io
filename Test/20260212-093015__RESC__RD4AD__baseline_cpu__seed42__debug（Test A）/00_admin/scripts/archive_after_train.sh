#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${RUN_DIR:?RUN_DIR not set}"
BMAD_DIR="$RUN_DIR/02_src/BMAD"
OUT_DIR="$RUN_DIR/06_outputs"

mkdir -p "$OUT_DIR"
cd "$BMAD_DIR"

if [ -d "results" ]; then
  rm -rf "$OUT_DIR/results__after_train"
  cp -R "results" "$OUT_DIR/results__after_train"
fi

find "$BMAD_DIR" -maxdepth 7 -type f \( -name "*.ckpt" -o -name "*.pth" -o -name "*.pt" \) > "$OUT_DIR/weights__index.txt"

echo "ARCHIVE_DONE"
echo "weights index at: $OUT_DIR/weights__index.txt"
