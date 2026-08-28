#!/usr/bin/env bash
# Pre-launch smoke test suite for PoL-BFL experiments
# - Runs unit/component checks
# - Runs minimal end-to-end smoke for RQ1/RQ2/RQ3/RQ4 on MNIST
# Usage:
#   bash experiments/scripts/smoke_prelaunch.sh
# Optional env vars:
#   GPU0=0 GPU1=1 bash experiments/scripts/smoke_prelaunch.sh
set -euo pipefail

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# Resolve project root = PoL-BFL
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJ_ROOT"

LOG_DIR="$PROJ_ROOT/experiments/smoke_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"

# Python interpreter: override with PY=/path/to/python
PY_BIN="${PY:-python}"

note() { echo -e "${YELLOW}[SMOKE][INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[SMOKE][OK]${NC}   $*"; }
fail() { echo -e "${RED}[SMOKE][FAIL]${NC} $*"; }

run_step() {
  local name="$1"; shift
  note "Running: $name"
  set +e
  "$@" 2>&1 | tee -a "$LOG_DIR/${TS}__${name// /_}.log"
  local ec=${PIPESTATUS[0]}
  set -e
  if [[ $ec -ne 0 ]]; then
    fail "$name (exit $ec). See: $LOG_DIR/${TS}__${name// /_}.log"
    exit $ec
  fi
  ok "$name"
}

note "Project root: $PROJ_ROOT"
note "Logs: $LOG_DIR"

# 0) Environment probe
run_step "env_probe" bash -lc "$PY_BIN -V && $PY_BIN -c 'import sys; print(sys.executable)' && $PY_BIN -c 'import torch,sys;print(\"torch\", torch.__version__)'"

# 1) Unit/Component tests
run_step "pytest_checkpoint_cleaner" bash -lc "$PY_BIN -m pytest -q tests/test_checkpoint_cleaner.py"
run_step "pytest_deferred_cleanup" bash -lc "$PY_BIN -m pytest -q tests/test_deferred_cleanup.py"
run_step "pytest_zkp" bash -lc "$PY_BIN -m pytest -q tests/test_zkp.py"
run_step "pytest_pol_aggregator_setup" bash -lc "$PY_BIN -m pytest -q tests/test_rq1_smoke.py::test_pol_aggregator_setup"

# GPU assignment (defaults)
GPU0_DEFAULT="${GPU0:-0}"
GPU1_DEFAULT="${GPU1:-1}"
note "GPU0=${GPU0_DEFAULT}, GPU1=${GPU1_DEFAULT}"

# 2) RQ1 MNIST minimal (2 rounds; PoL + Vanilla; no_attack)
run_step "rq1_mnist_minimal" bash -lc "CUDA_VISIBLE_DEVICES=${GPU0_DEFAULT} $PY_BIN experiments/scripts/runners/run_rq1_security.py --dataset MNIST --num_rounds 2 --baselines PoL_FL,Vanilla_FL --attacks no_attack --verification_rate 0.3 --pol_delta 10.0"

# 3) RQ2 MNIST minimal (2 rounds; 1 repetition; two variants)
run_step "rq2_mnist_minimal" bash -lc "CUDA_VISIBLE_DEVICES=${GPU1_DEFAULT} $PY_BIN experiments/scripts/runners/run_rq2_ablation.py --dataset MNIST --num_rounds 2 --num_repetitions 1 --variants vanilla_fl,pol_only"

# 4) RQ3 MNIST minimal (2 rounds)
run_step "rq3_mnist_minimal" bash -lc "CUDA_VISIBLE_DEVICES=${GPU0_DEFAULT} $PY_BIN experiments/scripts/runners/run_rq3_overhead.py --dataset MNIST --rounds 2 --num_clients 10 --clients_per_round 5"

# 5) RQ4 MNIST minimal (3 rounds)
run_step "rq4_mnist_minimal" bash -lc "CUDA_VISIBLE_DEVICES=${GPU1_DEFAULT} $PY_BIN experiments/scripts/runners/run_rq4_incentive.py --dataset MNIST --num_rounds 3 --clients_per_round 10"

ok "All smoke steps completed. Review logs under: $LOG_DIR"

# Smoke pointers to expected outputs (best-effort)
echo "\nExpected outputs (if present):"
ls -1 "experiments/results/rq1_security" 2>/dev/null || true
ls -1d "experiments/results/rq2_ablation"/* 2>/dev/null | tail -n 3 || true
ls -1 "experiments/results/rq3_scalability" 2>/dev/null || true
ls -1 "experiments/results/rq4_incentive" 2>/dev/null || true

