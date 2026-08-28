#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${1:-polbfl_strict_remote_smoke}"
PORT="${POL_VERIFIER_PORT:-18088}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_ROOT/experiments/results/reproduction/smoke}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

mkdir -p "$RUN_DIR"

export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export POL_DECENT_MODE=1
export POL_REQUIRE_REMOTE_VERIFIER=1
export POL_REMOTE_MODE=strict_replay
export POL_REMOTE_TIMEOUT_SEC="${POL_REMOTE_TIMEOUT_SEC:-240}"
export POL_VERIFIER_ENDPOINTS="http://127.0.0.1:$PORT"

"$PYTHON_BIN" -m server.committee.VerifierNode --host 127.0.0.1 --port "$PORT" \
  > "$RUN_DIR/verifier_node.log" 2>&1 &
VERIFIER_PID=$!

cleanup() {
  kill "$VERIFIER_PID" >/dev/null 2>&1 || true
  wait "$VERIFIER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 3

"$PYTHON_BIN" "$CODE_ROOT/experiments/reproducibility/run_repro_smoke.py" \
  --dataset MNIST \
  --model SimpleCNN \
  --rounds 1 \
  --num-clients 2 \
  --clients-per-round 2 \
  --data-distribution IID \
  --attacks free_riding_no_training \
  --baselines PoL_FL \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --gpu cpu
