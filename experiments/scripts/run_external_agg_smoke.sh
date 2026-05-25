#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/wxb/miniconda3/envs/wxb__veryfl_pol/bin/python}"
RUN_ID="${1:-polbfl_external_agg_smoke}"
PORT="${POL_AGGREGATOR_PORT:-18188}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_ROOT/experiments/results/reproduction/smoke}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

mkdir -p "$RUN_DIR"

export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export POL_AGGREGATOR_ENDPOINT="http://127.0.0.1:$PORT"
export POL_REQUIRE_EXTERNAL_AGGREGATOR=1

"$PYTHON_BIN" -m server.committee.AggregatorNode --host 127.0.0.1 --port "$PORT" \
  > "$RUN_DIR/aggregator_node.log" 2>&1 &
AGG_PID=$!

cleanup() {
  kill "$AGG_PID" >/dev/null 2>&1 || true
  wait "$AGG_PID" >/dev/null 2>&1 || true
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
