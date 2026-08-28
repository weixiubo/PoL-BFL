#!/usr/bin/env bash

# RQ1 MNIST baselines smoke test (short run, validation-focused)
# Runs 3 rounds of byzantine_random_noise and free_riding_no_training across all baselines.
# This reduced-scale run verifies that each baseline produces finite metrics
# under the selected attacks.

set -euo pipefail

# Resolve to Code/ directory (this script lives in Code/experiments/scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_DIR}"

python3 experiments/scripts/runners/run_rq1_security.py \
  --dataset MNIST \
  --model SimpleCNN \
  --num_rounds 3 \
  --num_clients 20 \
  --clients_per_round 10 \
  --attacks byzantine_random_noise,free_riding_no_training \
  --baselines Vanilla_FL,Krum,Trimmed_Mean,Median,ShapleyFL,FoolsGold,PoL_FL \
  --output_dir experiments/results/rq1_smoke_mnist_baselines
