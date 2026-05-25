#!/usr/bin/env bash

# RQ5 MNIST composability smoke test (short run, clearance-focused)
# This is NOT for paper-level numbers. It only checks that
# Krum/Median with and without PoL run end-to-end under label flipping.

set -euo pipefail

# Resolve to Code/ directory (this script lives in Code/experiments/scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_DIR}"

python3 experiments/scripts/runners/run_rq5_composability.py \
  --dataset MNIST \
  --model SimpleCNN \
  --num_rounds 3 \
  --attacks label_flipping \
  --baselines Krum,PoL_Krum,Median,PoL_Median \
  --output_dir experiments/results/rq5_mnist_smoke

