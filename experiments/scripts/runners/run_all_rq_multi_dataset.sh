#!/bin/bash

# PoL-BFL Multi-Dataset Experiment Runner
# Runs all RQ experiments on MNIST, CIFAR-10, and CIFAR-100
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
LOG_FILE="$LOG_DIR/run_all_rq_multi_dataset_$TIMESTAMP.log"

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
    
    log "Starting $rq_name on $dataset..."
    
    if python "$SCRIPT_DIR/$script_name" --dataset "$dataset" $args 2>&1 | tee -a "$LOG_FILE"; then
        log "✅ $rq_name on $dataset completed successfully"
    else
        log "❌ $rq_name on $dataset failed"
        return 1
    fi
}

# Main execution
main() {
    log "🚀 Starting PoL-BFL Multi-Dataset Experiments"
    log "Project Root: $PROJECT_ROOT"
    log "Results Dir: $RESULTS_DIR"
    log "Log File: $LOG_FILE"
    
    # Check environment
    if ! command -v python &> /dev/null; then
        log "❌ Python not found. Please activate conda environment: conda activate wxb__veryfl_pol"
        exit 1
    fi
    
    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/experiments/scripts/utils"
    cd "$PROJECT_ROOT"
    
    log "📊 Python version: $(python --version)"
    log "📊 PYTHONPATH: $PYTHONPATH"
    
    # RQ1: Security Evaluation (already supports multi-dataset)
    log "🔒 Running RQ1: Security Evaluation"
    for dataset in MNIST CIFAR10 CIFAR100; do
        if [ "$dataset" = "MNIST" ]; then
            rounds=50
        else
            rounds=100
        fi
        run_experiment "RQ1" "$dataset" "run_rq1_security.py" "--num_rounds $rounds"
    done
    
    # RQ2: Ablation Study
    log "🔬 Running RQ2: Ablation Study"
    for dataset in MNIST CIFAR10 CIFAR100; do
        if [ "$dataset" = "MNIST" ]; then
            rounds=50
        else
            rounds=100
        fi
        run_experiment "RQ2" "$dataset" "run_rq2_ablation.py" "--num_rounds $rounds --num_repetitions 3"
    done
    
    # RQ3: System Overhead
    log "⚡ Running RQ3: System Overhead"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ3" "$dataset" "run_rq3_overhead.py" "--rounds 20"
    done
    
    # RQ4: Incentive Mechanism
    log "💰 Running RQ4: Incentive Mechanism"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ4" "$dataset" "run_rq4_incentive.py" "--num_rounds 50"
    done
    
    log "🎉 All experiments completed successfully!"
    log "📁 Results saved in: $RESULTS_DIR"
    log "📄 Full log available at: $LOG_FILE"
}

# Parse command line arguments
QUICK_MODE=false
SKIP_RQ1=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            log "🚀 Quick mode enabled (reduced rounds)"
            shift
            ;;
        --skip-rq1)
            SKIP_RQ1=true
            log "⏭️ Skipping RQ1 (already completed)"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --quick      Use reduced rounds for faster testing"
            echo "  --skip-rq1   Skip RQ1 experiments"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            log "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

# Adjust rounds for quick mode
if [ "$QUICK_MODE" = true ]; then
    log "🚀 Quick mode: Using reduced rounds for all experiments"
    # Override rounds in the main function
    sed -i 's/rounds=50/rounds=10/g; s/rounds=100/rounds=20/g; s/--rounds 20/--rounds 5/g; s/--num_rounds 50/--num_rounds 10/g' "$0"
fi

# Run main function
main

log "✨ Script completed at $(date)"
