#!/usr/bin/env bash
set -euo pipefail

# Python binary (override with PYTHON_BIN if needed)
PYTHON_BIN="${PYTHON_BIN:-/home/wxb/miniconda3/envs/wxb__veryfl_pol/bin/python}"

# Usage: run_quick_rq4.sh <DATASET: MNIST|CIFAR10|CIFAR100> <TAG> <GPU_ID> [ROUNDS]
DATASET=${1:?DATASET required}
TAG=${2:?TAG required}
GPU=${3:?GPU_ID required}
ROUNDS=${4:-10}

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="experiments/logs/rq4_${STAMP}_${TAG}.log"
PIDF="experiments/logs/rq4_${STAMP}_${TAG}.pid"
mkdir -p experiments/logs
# Map GPU=cpu to CPU-only execution
if [ "$GPU" = "cpu" ]; then
  CVDEV=""
else
  CVDEV="$GPU"
fi
# CPU-only tuning to avoid oversubscription
if [ "$GPU" = "cpu" ]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
  export NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-2}"
fi



export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

{
  echo "==== RUN META $(date) ===="
  echo "TAG=$TAG GPU=$GPU DATASET=$DATASET ROUNDS=$ROUNDS"
  echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
} >> "$LOG"

CUDA_VISIBLE_DEVICES="$CVDEV" \
CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" \
NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-}" \
nohup "$PYTHON_BIN" experiments/scripts/runners/run_rq4_incentive.py \
  --dataset "$DATASET" \
  --num_rounds "$ROUNDS" \
  >> "$LOG" 2>&1 &

echo $! > "$PIDF"
echo "$LOG" > "experiments/logs/last_rq4_${TAG}.logpath"
echo "$PIDF" > "experiments/logs/last_rq4_${TAG}.pidpath"

echo "STARTED RQ4 dataset=$DATASET tag=$TAG pid=$(cat "$PIDF") log=$LOG"

