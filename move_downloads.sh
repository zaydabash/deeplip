#!/bin/bash
# Helper script to move downloaded GRID corpus files to downloads/ directory

SOURCE_DIR="$HOME/Downloads"
TARGET_DIR="downloads"

echo "Moving GRID corpus files from $SOURCE_DIR to $TARGET_DIR/..."

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Move alignments.zip
if [ -f "$SOURCE_DIR/alignments.zip" ]; then
    mv "$SOURCE_DIR/alignments.zip" "$TARGET_DIR/"
    echo "[OK] Moved alignments.zip"
else
    echo "[WARNING] alignments.zip not found in $SOURCE_DIR"
fi

# Move speaker files (s1.zip through s34.zip)
moved_count=0
for i in {1..34}; do
    file="$SOURCE_DIR/s${i}.zip"
    if [ -f "$file" ]; then
        mv "$file" "$TARGET_DIR/"
        echo "[OK] Moved s${i}.zip"
        moved_count=$((moved_count + 1))
    fi
done

# Move audio file if it exists
if [ -f "$SOURCE_DIR/audio_25k.zip" ]; then
    mv "$SOURCE_DIR/audio_25k.zip" "$TARGET_DIR/"
    echo "[OK] Moved audio_25k.zip"
fi

echo ""
echo "Summary:"
echo "  - Moved $moved_count speaker file(s)"
echo "  - Files are now in $TARGET_DIR/"
echo ""
echo "Next step: python3 download_grid_zenodo.py --extract downloads/"

