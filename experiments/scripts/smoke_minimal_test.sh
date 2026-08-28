#!/usr/bin/env bash
POLBFL_ROOT="${POLBFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
#
# 缩减规模最小化测试脚本
# 验证PoL-BFL实验框架的完整性（3轮CIFAR-10）
#

set -euo pipefail

# Configuration
PROJECT_ROOT="$POLBFL_ROOT"
SCRIPT_DIR="$PROJECT_ROOT/experiments/scripts/runners"
LOG_DIR="$PROJECT_ROOT/experiments/logs/parameter_evaluation"
RESULT_DIR="$PROJECT_ROOT/experiments/results/parameter_evaluation"

# Create directories
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Environment setup
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/experiments/scripts/utils"
export POL_DATA_DIR="$PROJECT_ROOT/data"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "[START] PoL-BFL Smoke Minimal Test"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Log Dir: $LOG_DIR"
echo "Result Dir: $RESULT_DIR"
echo ""

# Test 1: RQ1 Minimal (3 rounds, CIFAR-10)
echo "Test 1: RQ1 Security (3 rounds, CIFAR-10)"
LOG_FILE="$LOG_DIR/test_rq1_minimal.log"
python "$SCRIPT_DIR/run_rq1_security.py" \
  --dataset CIFAR10 \
  --num_rounds 3 \
  --output_dir "$RESULT_DIR/test_rq1_minimal" \
  2>&1 | tee "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "[PASS] RQ1 Minimal Test PASSED"
else
    echo "[FAIL] RQ1 Minimal Test FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "[PASS] All minimal tests passed."
echo "=========================================="

