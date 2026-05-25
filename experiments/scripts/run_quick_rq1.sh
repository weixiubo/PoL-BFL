#!/usr/bin/env bash
set -euo pipefail

# Python binary (override with PYTHON_BIN if needed)
PYTHON_BIN="${PYTHON_BIN:-/home/wxb/miniconda3/envs/wxb__veryfl_pol/bin/python}"

# Usage: run_quick_rq1.sh <DATASET: MNIST|CIFAR10|CIFAR100> <TAG> <GPU_ID> [NUM_ROUNDS]
DATASET=${1:?DATASET required}
TAG=${2:?TAG required}
GPU=${3:?GPU_ID required}
NUM_ROUNDS=${4:-}

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="experiments/logs/rq1_${STAMP}_${TAG}.log"
PIDF="experiments/logs/rq1_${STAMP}_${TAG}.pid"
mkdir -p experiments/logs

# Defaults for quick obstacle-clearing runs
case "$DATASET" in
  MNIST)
    : "${NUM_ROUNDS:=5}"
    MODEL="SimpleCNN"
    ;;
  CIFAR10)
    : "${NUM_ROUNDS:=10}"
    MODEL="ResNet18"
    ;;
  CIFAR100)
    : "${NUM_ROUNDS:=10}"
    MODEL="ResNet18"
    ;;
  *) echo "Unknown DATASET: $DATASET" >&2; exit 1;;
esac

# Optional PoL quick overrides via env
: "${POL_VERIFICATION_RATE:=}"
: "${POL_DELTA_OVERRIDE:=}"
: "${POL_ALWAYS_VERIFY_LAST_K:=}"
: "${POL_RANDOM_Q:=}"
: "${POL_MIN_PAIR_SUCCESS_RATE:=}"

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

# Prepend run meta
{
  echo "==== RUN META $(date) ===="
  echo "TAG=$TAG GPU=$GPU DATASET=$DATASET ROUNDS=$NUM_ROUNDS MODEL=$MODEL"
  echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
} >> "$LOG"

CUDA_VISIBLE_DEVICES="$CVDEV" \
CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" \
NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-}" \
nohup "$PYTHON_BIN" experiments/scripts/runners/run_rq1_security.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --num_rounds "$NUM_ROUNDS" \
  >> "$LOG" 2>&1 &

echo $! > "$PIDF"
echo "$LOG" > "experiments/logs/last_rq1_${TAG}.logpath"
echo "$PIDF" > "experiments/logs/last_rq1_${TAG}.pidpath"

echo "STARTED RQ1 dataset=$DATASET tag=$TAG pid=$(cat "$PIDF") log=$LOG"

