#!/bin/bash

# PoL-BFL Experiments Runner
# This script runs all RQ1-RQ4 experiments sequentially

set -e  # Exit on error

echo "======================================================================="
echo "PoL-BFL Experiments Runner"
echo "======================================================================="
echo ""

# Check if conda environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "polbfl" ]]; then
    echo "Error: activate the polbfl environment or set PYTHON_BIN."
    echo "Run: conda activate polbfl"
    exit 1
fi

echo "[PASS] Conda environment: ${CONDA_DEFAULT_ENV}"
echo ""

# Create results directory
mkdir -p results
echo "[PASS] Results directory created"
echo ""

# Function to run experiment with error handling
run_experiment() {
    local name=$1
    local script=$2

    echo "======================================================================="
    echo "Running ${name}..."
    echo "======================================================================="
    echo ""

    start_time=$(date +%s)

    if python ${script}; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo ""
        echo "[PASS] ${name} completed successfully in ${duration}s"
        echo ""
    else
        echo ""
        echo "[FAIL] ${name} failed."
        echo "Check the error messages above for details"
        exit 1
    fi
}

# Run experiments
echo "Starting experiments at $(date)"
echo ""

# RQ1: Security Evaluation (Multi-Dataset)
for dataset in MNIST CIFAR10 CIFAR100; do
    run_experiment "RQ1: Security Evaluation ($dataset)" "run_rq1_security.py --dataset $dataset"
done

# RQ2: Ablation Study (Multi-Dataset)
for dataset in MNIST CIFAR10 CIFAR100; do
    run_experiment "RQ2: Ablation Study ($dataset)" "run_rq2_ablation.py --dataset $dataset"
done

# RQ3: System Overhead (Multi-Dataset)
for dataset in MNIST CIFAR10 CIFAR100; do
    run_experiment "RQ3: System Overhead ($dataset)" "run_rq3_overhead.py --dataset $dataset"
done

# RQ4: Incentive Effectiveness (Multi-Dataset)
for dataset in MNIST CIFAR10 CIFAR100; do
    run_experiment "RQ4: Incentive Effectiveness ($dataset)" "run_rq4_incentive.py --dataset $dataset"
done

# Summary
echo "======================================================================="
echo "All Experiments Completed."
echo "======================================================================="
echo ""
echo "Results saved in:"
echo "  - results/rq1_security/ (MNIST, CIFAR10, CIFAR100)"
echo "  - results/rq2_ablation/ (MNIST, CIFAR10, CIFAR100)"
echo "  - results/rq3_overhead/ (MNIST, CIFAR10, CIFAR100)"
echo "  - results/rq4_incentive/ (MNIST, CIFAR10, CIFAR100)"
echo ""
echo "Completed at $(date)"
echo ""
