#!/usr/bin/env bash
set -euo pipefail

# Usage: run_quick_generic.sh <DATASET: MNIST|CIFAR10|CIFAR100> <TAG> <GPU_ID> <VR> <K_last> <Q_random> [NUM_ROUNDS=10]
DATASET=${1:?DATASET required}
TAG=${2:?TAG required}
GPU=${3:?GPU_ID required}
VR=${4:?verification_rate required}
K=${5:?K_last required}
Q=${6:?Q_random required}
NUM_ROUNDS=${7:-10}

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="experiments/logs/rq2_${STAMP}_${TAG}.log"
PIDF="experiments/logs/rq2_${STAMP}_${TAG}.pid"
mkdir -p experiments/logs

# Dataset-specific safe defaults (can be overridden by env)
case "$DATASET" in
  CIFAR10|CIFAR100)
    : "${PAIR_DELTA:=2.2}"   # pairwise delta (strict)
    : "${FINAL_DELTA:=35}"   # final state delta (looser)
    : "${SAVE_FREQ:=5}"
    ;;
  MNIST)
    : "${PAIR_DELTA:=6.0}"
    : "${FINAL_DELTA:=80}"
    : "${SAVE_FREQ:=4}"
    ;;
  *)
    echo "Unknown DATASET: $DATASET" >&2; exit 1;;
 esac

: "${MPSR:=1.0}"  # min_pair_success_rate
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

# Prepend run meta for reproducibility
{
  echo "==== RUN META $(date) ===="
  echo "TAG=$TAG GPU=$GPU DATASET=$DATASET VR=$VR K=$K Q=$Q ROUNDS=$NUM_ROUNDS"
  echo "PAIR_DELTA=$PAIR_DELTA FINAL_DELTA=$FINAL_DELTA MPSR=$MPSR SAVE_FREQ=$SAVE_FREQ"
  echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
} >> "$LOG"

CUDA_VISIBLE_DEVICES="$CVDEV" \
CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" \
POL_VERIFICATION_RATE="$VR" \
POL_SAVE_FREQ="$SAVE_FREQ" \
POL_MIN_PAIR_SUCCESS_RATE="$MPSR" \
POL_DELTA_OVERRIDE="$PAIR_DELTA" \
POL_FINAL_DELTA_OVERRIDE="$FINAL_DELTA" \
POL_ALWAYS_VERIFY_LAST_K="$K" \
POL_RANDOM_Q="$Q" \
NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-}" \
nohup /home/wxb/miniconda3/envs/wxb__veryfl_pol/bin/python \
  experiments/scripts/runners/run_rq2_ablation.py \
  --dataset "$DATASET" \
  --num_rounds "$NUM_ROUNDS" \
  --num_repetitions 1 \
  --variants pol_only \
  >> "$LOG" 2>&1 &

echo $! > "$PIDF"
echo "$LOG" > "experiments/logs/last_${TAG}.logpath"
echo "$PIDF" > "experiments/logs/last_${TAG}.pidpath"

echo "STARTED dataset=$DATASET tag=$TAG pid=$(cat "$PIDF") log=$LOG"

