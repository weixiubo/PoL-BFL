#!/bin/bash
POLBFL_ROOT="${POLBFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# GPU 0 CIFAR-10 RQ1 queue (9 attacks)
# GPU: GPU0 (CUDA_VISIBLE_DEVICES=0)

set -e

# Environment setup
export CUDA_VISIBLE_DEVICES=0
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
LOG_DIR="/tmp/rq1_validation"
PYTHON="python"

# Create log directory
mkdir -p "$LOG_DIR"

# Main log file
MAIN_LOG="$LOG_DIR/gpu0_rq1_cifar10.log"

echo "========================================" | tee -a "$MAIN_LOG"
echo "GPU 0 CIFAR-10 RQ1 queue started" | tee -a "$MAIN_LOG"
echo "Time: $(date)" | tee -a "$MAIN_LOG"
echo "GPU: GPU0 (CUDA_VISIBLE_DEVICES=0)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Attack list (9 attacks)
ATTACKS=(
    "no_attack"
    "byzantine_random_noise"
    "byzantine_label_flipping"
    "byzantine_model_replacement"
    "byzantine_gradient_inversion"
    "free_riding_no_training"
    "free_riding_lazy_training"
    "free_riding_minimal_update"
    "sybil_attack"
)

# Run each attack
for attack in "${ATTACKS[@]}"; do
    echo "----------------------------------------" | tee -a "$MAIN_LOG"
    echo "Starting: CIFAR-10 RQ1 - $attack" | tee -a "$MAIN_LOG"
    echo "Time: $(date)" | tee -a "$MAIN_LOG"
    echo "----------------------------------------" | tee -a "$MAIN_LOG"

    log_file="$LOG_DIR/rq1_cifar10_${attack}.log"

    cd $POLBFL_ROOT

    if $PYTHON "$SCRIPT_DIR/run_rq1_security.py" \
        --dataset CIFAR10 \
        --model ResNet18 \
        --num_rounds 20 \
        --attacks "$attack" \
        --baselines PoL_FL \
        > "$log_file" 2>&1; then
        echo "[PASS] Completed: $attack" | tee -a "$MAIN_LOG"
        echo "   Log: $log_file" | tee -a "$MAIN_LOG"

        # Extract key metrics
        echo "   Metrics:" | tee -a "$MAIN_LOG"
        grep -E "Final Accuracy|TPR_conditional" "$log_file" | tail -2 | sed 's/^/     /' | tee -a "$MAIN_LOG"
    else
        echo "[FAIL] Failed: $attack" | tee -a "$MAIN_LOG"
        echo "   Log: $log_file" | tee -a "$MAIN_LOG"
    fi

    echo "" | tee -a "$MAIN_LOG"
done

# Summary
echo "========================================" | tee -a "$MAIN_LOG"
echo "GPU 0 CIFAR-10 RQ1 queue completed" | tee -a "$MAIN_LOG"
echo "Time: $(date)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo "Results Summary:" | tee -a "$MAIN_LOG"
echo "[PASS] Completed: $(grep -c "[PASS] Completed" "$MAIN_LOG" || echo 0)" | tee -a "$MAIN_LOG"
echo "[FAIL] Failed: $(grep -c "[FAIL] Failed" "$MAIN_LOG" || echo 0)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo "All logs saved to: $LOG_DIR" | tee -a "$MAIN_LOG"
