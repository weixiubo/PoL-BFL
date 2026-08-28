#!/usr/bin/env bash
POLBFL_ROOT="${POLBFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
#
# PoL-BFL 实验参数评估执行脚本
# 并行运行RQ1-RQ5的轻量化实验
# 使用双GPU并行加速
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

# Helper functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PASS] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAIL] $1"
}

run_experiment() {
    local rq=$1
    local dataset=$2
    local rounds=$3
    local gpu=$4
    local tag=$5

    local log_file="$LOG_DIR/${tag}.log"
    local result_subdir="$RESULT_DIR/${tag}"

    log_info "Starting $tag on GPU $gpu (${rq}, ${dataset}, ${rounds} rounds)"

    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/run_${rq}_*.py" \
        --dataset "$dataset" \
        --num_rounds "$rounds" \
        --output_dir "$result_subdir" \
        >> "$log_file" 2>&1 &

    local pid=$!
    echo $pid > "$LOG_DIR/${tag}.pid"

    log_info "$tag PID: $pid"
}

# Main execution
echo "=========================================="
echo "[START] PoL-BFL Tuning Experiments"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Log Dir: $LOG_DIR"
echo "Result Dir: $RESULT_DIR"
echo ""

# Reduced-scale validation (3 rounds)
log_info "Reduced-scale validation"
run_experiment "rq1" "CIFAR10" 3 0 "rq1_cifar10_smoke"
sleep 2

# RQ1 full coverage
log_info "RQ1 full coverage"
run_experiment "rq1" "MNIST" 10 0 "rq1_mnist_tuning"
sleep 2

# RQ2-RQ4 on GPU 1
log_info "RQ2-RQ4 on GPU 1"
run_experiment "rq2" "MNIST" 10 1 "rq2_mnist_tuning"
sleep 2
run_experiment "rq3" "MNIST" 5 1 "rq3_mnist_tuning"
sleep 2
run_experiment "rq4" "MNIST" 20 1 "rq4_mnist_tuning"

# Wait for all experiments
log_info "Waiting for all experiments to complete..."
wait

log_success "All experiments completed."
echo ""
echo "=========================================="
echo "[RESULT] Results Summary"
echo "=========================================="
echo "Results saved to: $RESULT_DIR"
echo "Logs saved to: $LOG_DIR"
