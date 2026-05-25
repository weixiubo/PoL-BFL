#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/wxb/miniconda3/envs/wxb__veryfl_pol/bin/python}"
SESSION="${SESSION:-polbfl_20260518_prepare_femnist}"
LOG_DIR="$CODE_ROOT/experiments/results/reproduction/data_prep"
LOG_PATH="$LOG_DIR/prepare_femnist_hf.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session_exists=$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$CODE_ROOT' && \
   export PYTHONPATH='$CODE_ROOT':\${PYTHONPATH:-} && \
   exec '$PYTHON_BIN' experiments/scripts/tools/prepare_femnist_hf.py \
     --data-root '$CODE_ROOT/data/FEMNIST' \
     > '$LOG_PATH' 2>&1"

echo "session_started=$SESSION"
echo "log=$LOG_PATH"
