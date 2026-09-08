#!/usr/bin/env bash
set -euo pipefail

# --- Usage ---
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <PDF_DIR> <CONFIG_FILE>" >&2
    exit 1
fi

# --- Configuration ---
# Window size and all model/extraction params live in the YAML config (single
# source of truth). Only GPU assignment is set here, since it's host-specific.
PDF_DIR=$1
CONFIG_FILE=$2
GPUS="${GPUS:-5,6}"

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
    --config "$CONFIG_FILE"
