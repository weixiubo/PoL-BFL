#!/bin/bash

# PoL-BFL Quick Multi-Dataset Validation
# Runs all RQ experiments on all datasets with reduced rounds for quick validation
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
LOG_FILE="$LOG_DIR/quick_validation_$TIMESTAMP.log"

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
    
    log "🚀 Starting $rq_name on $dataset (Quick Mode)..."
    
    if python "$SCRIPT_DIR/$script_name" --dataset "$dataset" $args 2>&1 | tee -a "$LOG_FILE"; then
        log "✅ $rq_name on $dataset completed successfully"
    else
        log "❌ $rq_name on $dataset failed"
        return 1
    fi
}

# Main execution
main() {
    log "🚀 Starting PoL-BFL Quick Multi-Dataset Validation"
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
    
    # RQ1: Security Evaluation (Quick: 5 rounds)
    log "🔒 Running RQ1: Security Evaluation (Quick Mode)"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ1" "$dataset" "run_rq1_security.py" "--num_rounds 5"
    done
    
    # RQ2: Ablation Study (Quick: 5 rounds, 1 repetition)
    log "🔬 Running RQ2: Ablation Study (Quick Mode)"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ2" "$dataset" "run_rq2_ablation.py" "--num_rounds 5 --num_repetitions 1"
    done
    
    # RQ3: System Overhead (Quick: 3 rounds)
    log "⚡ Running RQ3: System Overhead (Quick Mode)"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ3" "$dataset" "run_rq3_overhead.py" "--rounds 3"
    done
    
    # RQ4: Incentive Mechanism (Quick: 5 rounds)
    log "💰 Running RQ4: Incentive Mechanism (Quick Mode)"
    for dataset in MNIST CIFAR10 CIFAR100; do
        run_experiment "RQ4" "$dataset" "run_rq4_incentive.py" "--num_rounds 5"
    done
    
    log "🎉 All quick validation experiments completed successfully!"
    log "📁 Results saved in: $RESULTS_DIR"
    log "📄 Full log available at: $LOG_FILE"
    
    # Summary
    log ""
    log "📊 Quick Validation Summary:"
    log "  - RQ1: 3 datasets × 5 rounds = 15 experiments"
    log "  - RQ2: 3 datasets × 5 rounds × 1 rep = 15 experiments"  
    log "  - RQ3: 3 datasets × 3 rounds = 9 experiments"
    log "  - RQ4: 3 datasets × 5 rounds = 15 experiments"
    log "  - Total: 54 quick experiments completed"
    log ""
    log "🚀 Ready for full experiments! Use:"
    log "  ./run_all_experiments.sh (for full multi-dataset experiments)"
    log "  ./run_all_rq_multi_dataset.sh (alternative with more options)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Quick validation of all RQ experiments on all datasets"
            echo ""
            echo "This script runs reduced experiments for quick validation:"
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
            log "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main

log "✨ Quick validation completed at $(date)"
