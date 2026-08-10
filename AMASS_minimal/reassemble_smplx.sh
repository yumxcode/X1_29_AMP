#!/bin/bash
# Reassemble SMPLX_NEUTRAL.pkl from git-pushed chunks
# Run this before GMR retargeting on Gradmotion

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHUNKS_DIR="$REPO_ROOT/AMASS_minimal/smplx_parts"
OUTPUT_DIR="$REPO_ROOT/AMASS_minimal/smplx"
OUTPUT_FILE="$OUTPUT_DIR/SMPLX_NEUTRAL.pkl"

mkdir -p "$OUTPUT_DIR"

if [ -f "$OUTPUT_FILE" ]; then
    echo "[INFO] SMPLX_NEUTRAL.pkl already exists, skipping reassembly."
    exit 0
fi

echo "[INFO] Reassembling SMPLX_NEUTRAL.pkl from chunks..."
cat "$CHUNKS_DIR"/SMPLX_NEUTRAL.pkl.part* > "$OUTPUT_FILE"

# Verify size
SIZE=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || stat -f%z "$OUTPUT_FILE" 2>/dev/null)
echo "[INFO] Done. SMPLX_NEUTRAL.pkl size: $SIZE bytes"

if [ "$SIZE" -lt 100000000 ]; then
    echo "[ERROR] File too small, reassembly may have failed!"
    exit 1
fi

echo "[INFO] SMPLX_NEUTRAL.pkl ready at $OUTPUT_FILE"
