#!/bin/bash
# Run All Experiments with Multiple Seeds
#
# This script runs all RQ1-RQ5 experiments with multiple random seeds
# for statistical significance testing.
#
# Usage:
#   bash run_all_experiments.sh [num_seeds]
#
# Example:
#   bash run_all_experiments.sh 3  # Run with 3 different seeds

set -e  # Exit on error

# Configuration
NUM_SEEDS=${1:-3}  # Default: 3 seeds
SEEDS=(42 123 456 789 1024)  # Predefined seeds
RESULTS_BASE_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Running All PoL-BFL Experiments"
echo "=========================================="
echo "Number of seeds: $NUM_SEEDS"
echo "Results directory: $RESULTS_BASE_DIR"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Create results directory
mkdir -p "$RESULTS_BASE_DIR"

# Function to run experiment with multiple seeds
run_experiment_multi_seed() {
    local experiment=$1
    local script=$2
    local output_dir="$RESULTS_BASE_DIR/${experiment}_${TIMESTAMP}"

    echo ""
    echo "=========================================="
    echo "Running $experiment with $NUM_SEEDS seeds"
    echo "=========================================="

    mkdir -p "$output_dir"

    for i in $(seq 0 $((NUM_SEEDS - 1))); do
        seed=${SEEDS[$i]}
        echo ""
        echo "--- Seed $((i + 1))/$NUM_SEEDS: $seed ---"

        # Set random seed
        export RANDOM_SEED=$seed

        # Run experiment
        python "runners/$script" \
            --output_dir "$output_dir" \
            --seed "$seed" \
            2>&1 | tee "$output_dir/log_seed_${seed}.txt"

        # Rename output to include seed
        if [ -f "$output_dir/${experiment}_results.json" ]; then
            mv "$output_dir/${experiment}_results.json" \
               "$output_dir/${experiment}_results_seed_${seed}.json"
        fi
    done

    echo ""
    echo "[PASS] Completed $experiment with $NUM_SEEDS seeds"
}

# Function to aggregate results
aggregate_results() {
    local experiment=$1
    local output_dir="$RESULTS_BASE_DIR/${experiment}_${TIMESTAMP}"

    echo ""
    echo "=========================================="
    echo "Aggregating $experiment results"
    echo "=========================================="

    python aggregate_multi_seed_results.py \
        --experiment "$experiment" \
        --input_dir "$output_dir" \
        --pattern "${experiment}_results_seed_*.json"

    echo "[PASS] Aggregated $experiment results"
}

# Function to generate visualizations
generate_visualizations() {
    local experiment=$1
    local output_dir="$RESULTS_BASE_DIR/${experiment}_${TIMESTAMP}"

    echo ""
    echo "=========================================="
    echo "Generating $experiment visualizations"
    echo "=========================================="

    python visualize_results.py \
        --experiment "$experiment" \
        --input "$output_dir/${experiment}_aggregated.json" \
        --output_dir "$output_dir/figures"

    echo "[PASS] Generated $experiment visualizations"
}

# Function to generate tables
generate_tables() {
    local experiment=$1
    local output_dir="$RESULTS_BASE_DIR/${experiment}_${TIMESTAMP}"

    echo ""
    echo "=========================================="
    echo "Generating $experiment LaTeX tables"
    echo "=========================================="

    python generate_paper_tables.py \
        --experiment "$experiment" \
        --input "$output_dir/${experiment}_aggregated.json" \
        --output_dir "$output_dir/tables"

    echo "[PASS] Generated $experiment tables"
}

# Main execution
main() {
    # RQ1: Security Evaluation
    run_experiment_multi_seed "rq1" "run_rq1_security.py"
    aggregate_results "rq1"
    generate_visualizations "rq1"
    generate_tables "rq1"

    # RQ2: Ablation Study
    run_experiment_multi_seed "rq2" "run_rq2_ablation.py"
    aggregate_results "rq2"
    generate_visualizations "rq2"
    generate_tables "rq2"

    # RQ3: Overhead Analysis
    run_experiment_multi_seed "rq3" "run_rq3_overhead.py"
    aggregate_results "rq3"
    generate_visualizations "rq3"
    generate_tables "rq3"

    # RQ4: Incentive Mechanism
    run_experiment_multi_seed "rq4" "run_rq4_incentive.py"
    aggregate_results "rq4"
    generate_visualizations "rq4"
    generate_tables "rq4"

    # RQ5: Composability
    run_experiment_multi_seed "rq5" "run_rq5_composability.py"
    aggregate_results "rq5"
    generate_visualizations "rq5"
    generate_tables "rq5"

    # Generate summary report
    echo ""
    echo "=========================================="
    echo "Generating Summary Report"
    echo "=========================================="

    python generate_summary_report.py \
        --results_dir "$RESULTS_BASE_DIR" \
        --timestamp "$TIMESTAMP" \
        --output "$RESULTS_BASE_DIR/summary_${TIMESTAMP}.md"

    echo ""
    echo "=========================================="
    echo "All Experiments Completed."
    echo "=========================================="
    echo "Results saved to: $RESULTS_BASE_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Review aggregated results in *_aggregated.json files"
    echo "2. Check visualizations in figures/ directories"
    echo "3. Copy LaTeX tables from tables/ directories to paper"
    echo "4. Read summary report: $RESULTS_BASE_DIR/summary_${TIMESTAMP}.md"
    echo "=========================================="
}

# Run main
main

