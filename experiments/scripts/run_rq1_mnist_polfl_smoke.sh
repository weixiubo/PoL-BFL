#!/usr/bin/env bash

# RQ1 MNIST PoL_FL smoke test (short run, clearance-focused)
# Runs 3 rounds of all 10 attacks with PoL_FL baseline only, using the
# cleared default configuration (delta=5.0, verification_rate=1.0).

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
  --attacks byzantine_random_noise,byzantine_label_flipping,byzantine_model_replacement,byzantine_gradient_inversion,byzantine_alie,byzantine_ipm,byzantine_minmax,free_riding_no_training,free_riding_lazy_training,free_riding_minimal_update \
  --baselines PoL_FL \
  --output_dir experiments/results/rq1_smoke_mnist_polfl

