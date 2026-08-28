#!/bin/bash

# PoL-BFL Smoke Multi-Dataset Validation
# Runs all RQ experiments on all datasets with reduced rounds for smoke validation
# Author: PoL-BFL Team
# Date: 2025-10-27

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/experiments/results"

# Logging
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/smoke_validation_$TIMESTAMP.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run experiment with error handling
run_experiment() {
    local rq_name="$1"
    local dataset="$2"
    local script_name="$3"
    local args="$4"

    log "[START] Starting $rq_name on $dataset (Smoke Mode)..."

    if python "$SCRIPT_DIR/$script_name" --dataset "$dataset" $args 2>&1 | tee -a "$LOG_FILE"; then
        log "[PASS] $rq_name on $dataset completed successfully"
    else
        log "[FAIL] $rq_name on $dataset failed"
        return 1
    fi
}

# Main execution
main() {
    log "[START] Starting PoL-BFL Smoke Multi-Dataset Validation"
    log "Project Root: $PROJECT_ROOT"
    log "Results Dir: $RESULTS_DIR"
    log "Log File: $LOG_FILE"

    # Check environment
    if ! command -v python &> /dev/null; then
        log "[FAIL] Python was not found. Activate the polbfl environment or set PYTHON_BIN."
        exit 1
    fi

    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/experiments/scripts/utils"
    cd "$PROJECT_ROOT"

    log "[RESULT] Python version: $(python --version)"
    log "[RESULT] PYTHONPATH: $PYTHONPATH"

    # RQ1: Security Evaluation (Smoke: 5 rounds)
    log "[SECURITY] Running RQ1: Security Evaluation (Smoke Mode)"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ1" "$dataset" "run_rq1_security.py" "--num_rounds 5"
    done

    # RQ2: Ablation Study (Smoke: 5 rounds, 1 repetition)
    log "[TEST] Running RQ2: Ablation Study (Smoke Mode)"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ2" "$dataset" "run_rq2_ablation.py" "--num_rounds 5 --num_repetitions 1"
    done

    # RQ3: System Overhead (Smoke: 3 rounds)
    log "[PERFORMANCE] Running RQ3: System Overhead (Smoke Mode)"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ3" "$dataset" "run_rq3_overhead.py" "--rounds 3"
    done

    # RQ4: Incentive Mechanism (Smoke: 5 rounds)
    log "[COST] Running RQ4: Incentive Mechanism (Smoke Mode)"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ4" "$dataset" "run_rq4_incentive.py" "--num_rounds 5"
    done

    log "[PASS] All smoke validation experiments completed successfully."
    log "[PATH] Results saved in: $RESULTS_DIR"
    log "[LOG] Full log available at: $LOG_FILE"

    # Summary
    log ""
    log "[RESULT] Smoke Validation Summary:"
    log "  - RQ1: 3 datasets × 5 rounds = 15 experiments"
    log "  - RQ2: 3 datasets × 5 rounds × 1 rep = 15 experiments"
    log "  - RQ3: 3 datasets × 3 rounds = 9 experiments"
    log "  - RQ4: 3 datasets × 5 rounds = 15 experiments"
    log "  - Total: 54 smoke experiments completed"
    log ""
    log "[START] Ready for full experiments. Use:"
    log "  ./run_all_experiments.sh (for full multi-dataset experiments)"
    log "  ./run_all_rq_multi_dataset.sh (alternative with more options)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Smoke validation of all RQ experiments on all datasets"
            echo ""
            echo "This script runs reduced experiments for smoke validation:"
            echo "  - RQ1: 5 rounds per dataset"
            echo "  - RQ2: 5 rounds, 1 repetition per dataset"
            echo "  - RQ3: 3 rounds per dataset"
            echo "  - RQ4: 5 rounds per dataset"
            echo ""
            echo "Options:"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            log "[FAIL] Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main

log "[COMPLETE] Smoke validation completed at $(date)"
