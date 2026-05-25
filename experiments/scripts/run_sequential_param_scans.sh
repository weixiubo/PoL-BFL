#!/usr/bin/env bash
set -euo pipefail

# Sequential Parameter Scan Script
# Runs multiple parameter scan experiments sequentially on a single GPU
# Usage: run_sequential_param_scans.sh <GPU_ID>

GPU=${1:?GPU_ID required}
DATASET="MNIST"

echo "=========================================="
echo "Sequential Parameter Scan on GPU $GPU"
echo "Dataset: $DATASET"
echo "Started at: $(date)"
echo "=========================================="

# Function to wait for a process to complete
wait_for_completion() {
    local pid=$1
    local tag=$2
    echo "[$(date +%H:%M:%S)] Waiting for $tag (PID: $pid) to complete..."
    while kill -0 $pid 2>/dev/null; do
        sleep 30
    done
    echo "[$(date +%H:%M:%S)] $tag completed!"
}

# Function to run a parameter scan and wait
run_and_wait() {
    local param_name=$1
    local param_value=$2
    local tag=$3
    
    echo ""
    echo "=========================================="
    echo "[$(date +%H:%M:%S)] Starting: $param_name=$param_value (tag: $tag)"
    echo "=========================================="
    
    # Run the experiment
    bash experiments/scripts/run_param_scan_rq1.sh "$DATASET" "$GPU" "$param_name" "$param_value" "$tag"
    
    # Get the PID
    local pidfile="experiments/logs/last_param_scan_${tag}.pidpath"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$(cat "$pidfile")")
        wait_for_completion "$pid" "$tag"
    else
        echo "ERROR: PID file not found: $pidfile"
        exit 1
    fi
}

# Verification Rate Scan (5 experiments)
echo ""
echo "=========================================="
echo "Phase 1: Verification Rate Scan"
echo "=========================================="

# vr01 is already running, skip it
echo "[$(date +%H:%M:%S)] Skipping verification_rate=0.1 (already running)"

run_and_wait "verification_rate" "0.2" "vr02_scan"
run_and_wait "verification_rate" "0.3" "vr03_scan"
run_and_wait "verification_rate" "0.4" "vr04_scan"
run_and_wait "verification_rate" "0.5" "vr05_scan"

# Delta Scan (5 experiments)
echo ""
echo "=========================================="
echo "Phase 2: Delta Threshold Scan"
echo "=========================================="

run_and_wait "delta" "5.0" "d05_scan"
run_and_wait "delta" "10.0" "d10_scan"
run_and_wait "delta" "15.0" "d15_scan"
run_and_wait "delta" "20.0" "d20_scan"
run_and_wait "delta" "30.0" "d30_scan"

echo ""
echo "=========================================="
echo "All Parameter Scans Completed!"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Summary of experiments:"
echo "  - Verification Rate: 0.1, 0.2, 0.3, 0.4, 0.5"
echo "  - Delta: 5.0, 10.0, 15.0, 20.0, 30.0"
echo ""
echo "Log files are in: experiments/logs/rq1_param_scan_*"

