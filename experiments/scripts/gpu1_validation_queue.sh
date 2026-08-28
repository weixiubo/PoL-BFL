#!/bin/bash
POLBFL_ROOT="${POLBFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# GPU 1 validation queue: MNIST RQ2, RQ3, and RQ4
# GPU: GPU1 (CUDA_VISIBLE_DEVICES=1)

set -e

# Environment setup
export CUDA_VISIBLE_DEVICES=1
export NUM_WORKERS_OVERRIDE=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export POL_VERIFICATION_RATE=0.3
export POL_FINAL_DELTA_OVERRIDE=50
export POL_ALWAYS_VERIFY_LAST_K=2
export POL_RANDOM_Q=3
export POL_MIN_PAIR_SUCCESS_RATE=0.99

# Paths
SCRIPT_DIR="$POLBFL_ROOT/experiments/scripts/runners"
LOG_DIR="/tmp/validation_queue"
PYTHON="python"

# Create log directory
mkdir -p "$LOG_DIR"

# Main log file
MAIN_LOG="$LOG_DIR/gpu1_validation_queue.log"

echo "========================================" | tee -a "$MAIN_LOG"
echo "GPU 1 validation queue started" | tee -a "$MAIN_LOG"
echo "Time: $(date)" | tee -a "$MAIN_LOG"
echo "GPU: GPU1 (CUDA_VISIBLE_DEVICES=1)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Function to run experiment
run_experiment() {
    local rq_type=$1
    local script_name=$2
    local exp_name=$3
    local args=$4

    echo "----------------------------------------" | tee -a "$MAIN_LOG"
    echo "Starting: $exp_name" | tee -a "$MAIN_LOG"
    echo "Time: $(date)" | tee -a "$MAIN_LOG"
    echo "----------------------------------------" | tee -a "$MAIN_LOG"

    local log_file="$LOG_DIR/${exp_name}.log"

    cd $POLBFL_ROOT

    if $PYTHON "$SCRIPT_DIR/$script_name" $args > "$log_file" 2>&1; then
        echo "[PASS] Completed: $exp_name" | tee -a "$MAIN_LOG"
        echo "   Log: $log_file" | tee -a "$MAIN_LOG"
    else
        echo "[FAIL] Failed: $exp_name" | tee -a "$MAIN_LOG"
        echo "   Log: $log_file" | tee -a "$MAIN_LOG"
    fi

    echo "" | tee -a "$MAIN_LOG"
}

# ========================================
# RQ2 - Ablation Study (5 experiments)
# ========================================
echo "========================================" | tee -a "$MAIN_LOG"
echo "RQ2 - Ablation Study (5 experiments)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# RQ2: Run all variants in one call (faster)
if [ -f "$SCRIPT_DIR/run_rq2_ablation.py" ]; then
    run_experiment "RQ2" "run_rq2_ablation.py" "rq2_mnist_all_variants" \
        "--dataset MNIST --num_rounds 20 --num_repetitions 1"
else
    echo "[WARNING]  RQ2 script not found, skipping RQ2 experiments" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
fi

# ========================================
# RQ3 - System Overhead (1 experiment, all methods)
# ========================================
echo "========================================" | tee -a "$MAIN_LOG"
echo "RQ3 - System Overhead (all methods)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# RQ3: Runs all methods (vanilla, pol_fl, pol_fl_zkp) in one call
if [ -f "$SCRIPT_DIR/run_rq3_overhead.py" ]; then
    run_experiment "RQ3" "run_rq3_overhead.py" "rq3_mnist_all_methods" \
        "--dataset MNIST --rounds 20"
else
    echo "[WARNING]  RQ3 script not found, skipping RQ3 experiments" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
fi

# ========================================
# RQ4 - Incentive Mechanism (4 experiments)
# ========================================
echo "========================================" | tee -a "$MAIN_LOG"
echo "RQ4 - Incentive Mechanism (4 experiments)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

if [ -f "$SCRIPT_DIR/run_rq4_incentive.py" ]; then
    run_experiment "RQ4" "run_rq4_incentive.py" "rq4_mnist_no_incentive" \
        "--dataset MNIST --num_rounds 50 --scenario no_incentive"

    run_experiment "RQ4" "run_rq4_incentive.py" "rq4_mnist_fixed_reward" \
        "--dataset MNIST --num_rounds 50 --scenario fixed_reward"

    run_experiment "RQ4" "run_rq4_incentive.py" "rq4_mnist_dynamic_reward" \
        "--dataset MNIST --num_rounds 50 --scenario dynamic_reward"

    run_experiment "RQ4" "run_rq4_incentive.py" "rq4_mnist_sybil_attack" \
        "--dataset MNIST --num_rounds 50 --scenario sybil_attack"
else
    echo "[WARNING]  RQ4 script not found, skipping RQ4 experiments" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
fi

# ========================================
# Summary
# ========================================
echo "========================================" | tee -a "$MAIN_LOG"
echo "GPU 1 validation queue completed" | tee -a "$MAIN_LOG"
echo "Time: $(date)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Count results
echo "Results Summary:" | tee -a "$MAIN_LOG"
echo "[PASS] Completed: $(grep -c "[PASS] Completed" "$MAIN_LOG" || echo 0)" | tee -a "$MAIN_LOG"
echo "[FAIL] Failed: $(grep -c "[FAIL] Failed" "$MAIN_LOG" || echo 0)" | tee -a "$MAIN_LOG"
echo "[WARNING]  Skipped: $(grep -c "[WARNING]  " "$MAIN_LOG" || echo 0)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo "All logs saved to: $LOG_DIR" | tee -a "$MAIN_LOG"
