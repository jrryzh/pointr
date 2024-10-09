#!/bin/bash

# Define source and destination directories for testing
SOURCE_DIR="/mnt/test/data/shapenet/shapenetcorev2_render_output5"
DEST_DIR="/mnt/test/data/shapenet/shapenetcorev2_render_output5_core"

# Find and prepare to copy .obj and .txt files, preserving directory structure
find "$SOURCE_DIR" -type f \( -name "*.obj" -o -name "*.txt" \) | while read FILE; do
    # Determine the directory path in the source relative to the base
    REL_PATH=$(dirname "$FILE" | sed "s|^$SOURCE_DIR||")
    
    # # Print the relative directory path
    # echo "Source file: $FILE"
    # echo "Will be copied to: $DEST_DIR$REL_PATH/$(basename "$FILE")"

    # Uncomment the lines below to actually perform the operations
    mkdir -p "$DEST_DIR$REL_PATH"
    cp "$FILE" "$DEST_DIR$REL_PATH"
done
