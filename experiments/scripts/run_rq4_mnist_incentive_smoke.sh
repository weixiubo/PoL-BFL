#!/usr/bin/env bash

# RQ4 MNIST incentive smoke test (short run, clearance-focused)
# Runs 3 rounds for each scenario to verify that the incentive
# logic and Sybil integration behave sanely and do not crash.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_DIR}"

SCENARIOS=(no_incentive fixed_reward dynamic_reward sybil_attack)

for sc in "${SCENARIOS[@]}"; do
  echo "[RQ4 smoke] Running scenario: ${sc}"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  python3 experiments/scripts/runners/run_rq4_incentive.py \
    --dataset MNIST \
    --model SimpleCNN \
    --num_clients 20 \
    --clients_per_round 10 \
    --num_rounds 3 \
    --scenario "${sc}" || exit 1
  echo "[RQ4 smoke] Scenario ${sc} done"
  echo
done

