#!/bin/bash
# --- Configuration ---
GPUS="6,7"
PDF_DIR=$1
CONFIG_FILE=$2
WINDOW_SIZE=128

# --- Execution ---
echo "=================================================="
echo "Starting Multi-GPU Extraction Pipeline"
echo "GPUs Assigned: $GPUS"
echo "PDF Source Dir: $PDF_DIR"
echo "Config File: $CONFIG_FILE"
echo "=================================================="

uv run scripts/run_sharded.py \
    --gpus "$GPUS" \
    --pdf "$PDF_DIR" \
    --config "$CONFIG_FILE" \
    --window-size "$WINDOW_SIZE" \