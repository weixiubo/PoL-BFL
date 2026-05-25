#!/usr/bin/env bash
set -euo pipefail
# Run a quick CIFAR-10 RQ2 ablation with specific verification settings
# Usage: run_quick_cifar10.sh <tag> <gpu_id> <verification_rate> <K_last> <Q_random>
# Example: run_quick_cifar10.sh c10_vr05_k3q4 0 0.5 3 4

# Move to repo root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

mkdir -p experiments/logs
STAMP=$(date +%Y%m%d_%H%M%S)
TAG=${1:?tag}
GPU=${2:?gpu_id}
VR=${3:?verification_rate}
K=${4:?always_verify_last_k}
Q=${5:?random_q}

LOG="experiments/logs/rq2_${STAMP}_${TAG}.log"
PIDF="experiments/logs/rq2_${STAMP}_${TAG}.pid"

# Fixed calibration params for CIFAR-10 from prior runs
PAIR_DELTA=2.2
FINAL_DELTA=35
MPSR=1.0
SAVE_FREQ=5

# Ensure deterministic cuBLAS workspaces (required by PyTorch determinism on CUDA>=10.2)
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Prepend run meta into log for reproducibility
{
  echo "==== RUN META $(date) ===="
  echo "TAG=$TAG GPU=$GPU DATASET=CIFAR10 VR=$VR K=$K Q=$Q"
  echo "PAIR_DELTA=$PAIR_DELTA FINAL_DELTA=$FINAL_DELTA MPSR=$MPSR SAVE_FREQ=$SAVE_FREQ"
  echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
} >> "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" \
CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" \
POL_VERIFICATION_RATE="$VR" \
POL_SAVE_FREQ="$SAVE_FREQ" \
POL_MIN_PAIR_SUCCESS_RATE="$MPSR" \
POL_DELTA_OVERRIDE="$PAIR_DELTA" \
POL_FINAL_DELTA_OVERRIDE="$FINAL_DELTA" \
POL_ALWAYS_VERIFY_LAST_K="$K" \
POL_RANDOM_Q="$Q" \
nohup /home/wxb/miniconda3/envs/wxb__veryfl_pol/bin/python \
  experiments/scripts/runners/run_rq2_ablation.py \
  --dataset CIFAR10 \
  --num_rounds 10 \
  --num_repetitions 1 \
  --variants pol_only \
  >> "$LOG" 2>&1 &

# Save pid and pointers
echo $! > "$PIDF"
echo "$LOG" > "experiments/logs/last_${TAG}.logpath"
echo "$PIDF" > "experiments/logs/last_${TAG}.pidpath"

echo "STARTED tag=$TAG pid=$(cat "$PIDF") log=$LOG"
