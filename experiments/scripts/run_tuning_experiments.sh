#!/usr/bin/env bash
#
# PoL-BFL 实验调参执行脚本
# 并行运行RQ1-RQ5的轻量化实验
# 使用双GPU并行加速
#

set -euo pipefail

# Configuration
PROJECT_ROOT="/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code"
SCRIPT_DIR="$PROJECT_ROOT/experiments/scripts/runners"
LOG_DIR="$PROJECT_ROOT/experiments/logs/tuning_2025-11-19"
RESULT_DIR="$PROJECT_ROOT/experiments/results/tuning_2025-11-19"

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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1"
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
echo "🚀 PoL-BFL Tuning Experiments"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Log Dir: $LOG_DIR"
echo "Result Dir: $RESULT_DIR"
echo ""

# Phase 1: Quick validation (3 rounds)
log_info "Phase 1: Quick Validation"
run_experiment "rq1" "CIFAR10" 3 0 "rq1_cifar10_quick"
sleep 2

# Phase 2: RQ1 Full Coverage
log_info "Phase 2: RQ1 Full Coverage"
run_experiment "rq1" "MNIST" 10 0 "rq1_mnist_tuning"
sleep 2

# Phase 3: RQ2-RQ4 on GPU1
log_info "Phase 3: RQ2-RQ4 on GPU1"
run_experiment "rq2" "MNIST" 10 1 "rq2_mnist_tuning"
sleep 2
run_experiment "rq3" "MNIST" 5 1 "rq3_mnist_tuning"
sleep 2
run_experiment "rq4" "MNIST" 20 1 "rq4_mnist_tuning"

# Wait for all experiments
log_info "Waiting for all experiments to complete..."
wait

log_success "All experiments completed!"
echo ""
echo "=========================================="
echo "📊 Results Summary"
echo "=========================================="
echo "Results saved to: $RESULT_DIR"
echo "Logs saved to: $LOG_DIR"

