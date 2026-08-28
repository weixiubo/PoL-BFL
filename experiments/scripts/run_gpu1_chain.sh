#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
PY="python"
LOGDIR="experiments/logs"
mkdir -p "$LOGDIR"

echo "[GPU1] Using python: $PY" | tee -a "$LOGDIR/gpu1_chain.log"

# RQ4 MNIST (resume after prior failure)
stdbuf -oL -eL "$PY" experiments/scripts/runners/run_rq4_incentive.py \
  --dataset MNIST --num_clients 20 --clients_per_round 10 --num_rounds 50 \
  >> "$LOGDIR/rq4_MNIST_full_gpu1.log" 2>&1

# CIFAR10 batch
stdbuf -oL -eL "$PY" experiments/scripts/runners/run_rq3_overhead.py \
  --dataset CIFAR10 --rounds 20 --num_clients 20 --clients_per_round 10 \
  >> "$LOGDIR/rq3_CIFAR10_full_gpu1.log" 2>&1

stdbuf -oL -eL "$PY" experiments/scripts/runners/run_rq2_ablation.py \
  --dataset CIFAR10 --num_rounds 100 --num_repetitions 1 \
  >> "$LOGDIR/rq2_CIFAR10_full_gpu1.log" 2>&1

stdbuf -oL -eL "$PY" experiments/scripts/runners/run_rq4_incentive.py \
  --dataset CIFAR10 --num_clients 20 --clients_per_round 10 --num_rounds 50 \
  >> "$LOGDIR/rq4_CIFAR10_full_gpu1.log" 2>&1

# CIFAR100 batch
stdbuf -oL -eL "$PY" experiments/scripts/runners/run_rq3_overhead.py \
  --dataset CIFAR100 --rounds 20 --num_clients 20 --clients_per_round 10 \
  >> "$LOGDIR/rq3_CIFAR100_full_gpu1.log" 2>&1

stdbuf -oL -eL "$PY" experiments/scripts/runners/run_rq2_ablation.py \
  --dataset CIFAR100 --num_rounds 100 --num_repetitions 1 \
  >> "$LOGDIR/rq2_CIFAR100_full_gpu1.log" 2>&1

stdbuf -oL -eL "$PY" experiments/scripts/runners/run_rq4_incentive.py \
  --dataset CIFAR100 --num_clients 20 --clients_per_round 10 --num_rounds 50 \
  >> "$LOGDIR/rq4_CIFAR100_full_gpu1.log" 2>&1

echo "[GPU1] Chain completed." | tee -a "$LOGDIR/gpu1_chain.log"

