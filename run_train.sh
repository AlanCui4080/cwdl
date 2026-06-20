#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=".conda/bin/python"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" checkpoints
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/train_${STAMP}.log"

echo "=== CWDL training ==="
echo "log -> $LOG"

exec "$PY" train.py "$@" 2>&1 | tee "$LOG"
