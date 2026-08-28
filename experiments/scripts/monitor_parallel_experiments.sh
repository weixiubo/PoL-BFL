#!/usr/bin/env bash
set -euo pipefail

# Monitor Parallel Experiments
# Shows the status of RQ1 and parameter scan experiments

echo "=========================================="
echo "Parallel Experiments Monitor"
echo "Time: $(date)"
echo "=========================================="
echo ""

# GPU Status
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
  awk -F', ' '{printf "GPU %s: %s%% util, %sMB / %sMB memory\n", $1, $3, $4, $5}'
echo ""

# Running Processes
echo "=== Running Python Processes ==="
ps aux | grep "run_rq1_security.py" | grep -v grep | \
  awk '{printf "PID %s: CPU %.1f%%, MEM %.1f%%, CMD: %s\n", $2, $3, $4, substr($0, index($0,$11))}'
echo ""

# RQ1 Default Config Progress
echo "=== RQ1 Default Config (GPU 0) ==="
RQ1_LOG=$(ls -t experiments/logs/rq1_*_smoke_validation_v2.log 2>/dev/null | head -1)
if [ -n "$RQ1_LOG" ]; then
    echo "Log: $RQ1_LOG"
    TOTAL_LINES=$(wc -l < "$RQ1_LOG")
    COMPLETED=$(grep -c "Running:.*vs" "$RQ1_LOG" || echo 0)
    echo "Progress: $COMPLETED/84 combinations completed"
    echo "Log lines: $TOTAL_LINES"
    LAST_RUNNING=$(grep "Running:.*vs" "$RQ1_LOG" | tail -1 || echo "N/A")
    echo "Last started: $LAST_RUNNING"
    LAST_ROUND=$(grep "Round.*/" "$RQ1_LOG" | tail -1 || echo "N/A")
    echo "Last round: $LAST_ROUND"
else
    echo "No RQ1 log found"
fi
echo ""

# Parameter Scan Progress
echo "=== Parameter Scan (GPU 1) ==="
PARAM_LOGS=$(ls -t experiments/logs/rq1_param_scan_*.log 2>/dev/null || echo "")
if [ -n "$PARAM_LOGS" ]; then
    CURRENT_SCAN=$(echo "$PARAM_LOGS" | head -1)
    echo "Current scan log: $CURRENT_SCAN"

    # Extract parameter info from log
    PARAM_INFO=$(grep "PARAM_NAME=" "$CURRENT_SCAN" | head -1 || echo "N/A")
    echo "Parameter: $PARAM_INFO"

    # Check progress
    TOTAL_LINES=$(wc -l < "$CURRENT_SCAN")
    COMPLETED=$(grep -c "Running:.*vs" "$CURRENT_SCAN" || echo 0)
    echo "Progress: $COMPLETED/84 combinations completed"
    echo "Log lines: $TOTAL_LINES"

    LAST_RUNNING=$(grep "Running:.*vs" "$CURRENT_SCAN" | tail -1 || echo "N/A")
    echo "Last started: $LAST_RUNNING"

    # List all completed scans
    echo ""
    echo "Completed scans:"
    for log in $PARAM_LOGS; do
        if ! ps aux | grep -q "$(basename "$log" .log)"; then
            PARAM=$(grep "PARAM_NAME=" "$log" | head -1 || echo "Unknown")
            COMBS=$(grep -c "Running:.*vs" "$log" || echo 0)
            echo "  - $(basename "$log"): $PARAM, $COMBS/84 combinations"
        fi
    done
else
    echo "No parameter scan logs found"
fi
echo ""

# Sequential scan script status
echo "=== Sequential Scan Script ==="
if ps aux | grep -q "run_sequential_param_scans.sh" | grep -v grep; then
    echo "Status: RUNNING"
    SEQ_LOG=$(ls -t experiments/logs/sequential_param_scans_*.log 2>/dev/null | head -1)
    if [ -n "$SEQ_LOG" ]; then
        echo "Log: $SEQ_LOG"
        echo "Last 5 lines:"
        tail -5 "$SEQ_LOG" | sed 's/^/  /'
    fi
else
    echo "Status: NOT RUNNING"
fi
echo ""

echo "=========================================="
echo "Monitor completed at $(date)"
echo "=========================================="

