#!/usr/bin/env bash
set -euo pipefail

# Parameter Scan Script for RQ1
# Usage: run_param_scan_rq1.sh <DATASET> <GPU_ID> <PARAM_NAME> <PARAM_VALUE> <TAG>
# Example: run_param_scan_rq1.sh MNIST 1 verification_rate 0.1 vr01

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET=${1:?DATASET required (MNIST|CIFAR10|CIFAR100)}
GPU=${2:?GPU_ID required}
PARAM_NAME=${3:?PARAM_NAME required (verification_rate|delta)}
PARAM_VALUE=${4:?PARAM_VALUE required}
TAG=${5:?TAG required}

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="experiments/logs/rq1_param_scan_${STAMP}_${TAG}.log"
PIDF="experiments/logs/rq1_param_scan_${STAMP}_${TAG}.pid"
mkdir -p experiments/logs

# Defaults for smoke parameter scan runs (10 rounds for speed)
case "$DATASET" in
  MNIST)
    NUM_ROUNDS=10
    MODEL="SimpleCNN"
    ;;
  CIFAR10)
    NUM_ROUNDS=10
    MODEL="ResNet18"
    ;;
  CIFAR100)
    NUM_ROUNDS=10
    MODEL="ResNet18"
    ;;
  *) echo "Unknown DATASET: $DATASET" >&2; exit 1;;
esac

# Map GPU=cpu to CPU-only execution
if [ "$GPU" = "cpu" ]; then
  CVDEV=""
else
  CVDEV="$GPU"
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Prepend run meta
{
  echo "==== PARAMETER SCAN RUN META $(date) ===="
  echo "TAG=$TAG GPU=$GPU DATASET=$DATASET ROUNDS=$NUM_ROUNDS MODEL=$MODEL"
  echo "PARAM_NAME=$PARAM_NAME PARAM_VALUE=$PARAM_VALUE"
  echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
} >> "$LOG"

# Build parameter argument
if [ "$PARAM_NAME" = "verification_rate" ]; then
  PARAM_ARG="--verification_rate $PARAM_VALUE"
elif [ "$PARAM_NAME" = "delta" ]; then
  PARAM_ARG="--pol_delta $PARAM_VALUE"
else
  echo "Unknown PARAM_NAME: $PARAM_NAME" >&2
  exit 1
fi

# Launch experiment
CUDA_VISIBLE_DEVICES="$CVDEV" \
CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" \
nohup "$PYTHON_BIN" experiments/scripts/runners/run_rq1_security.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --num_rounds "$NUM_ROUNDS" \
  $PARAM_ARG \
  >> "$LOG" 2>&1 &

echo $! > "$PIDF"
echo "$LOG" > "experiments/logs/last_param_scan_${TAG}.logpath"
echo "$PIDF" > "experiments/logs/last_param_scan_${TAG}.pidpath"

echo "STARTED Parameter Scan: $PARAM_NAME=$PARAM_VALUE dataset=$DATASET tag=$TAG pid=$(cat "$PIDF") log=$LOG"

