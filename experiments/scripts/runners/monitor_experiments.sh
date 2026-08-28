#!/bin/bash

# Monitor running experiments
# Usage: bash monitor_experiments.sh

echo "=========================================="
echo "Experiment Monitor"
echo "Time: $(date)"
echo "=========================================="

# Check GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s%% util, %sMB / %sMB memory\n", $1, $3, $4, $5}'

# Check running processes
echo ""
echo "Running Python Processes:"
ps aux | grep "run_rq1_security.py" | grep -v grep | \
    awk '{printf "  PID %s: %s CPU, %s MEM\n", $2, $3, $4}'

# Check log files
echo ""
echo "Recent Experiment Progress:"
for LOG in /tmp/rq1_*.log; do
    if [ -f "$LOG" ]; then
        BASENAME=$(basename "$LOG" .log)
        LAST_LINE=$(tail -1 "$LOG" 2>/dev/null)

        # Extract round info if available
        ROUND_INFO=$(grep -oP "Round \d+/\d+" "$LOG" | tail -1)

        if [ -n "$ROUND_INFO" ]; then
            echo "  $BASENAME: $ROUND_INFO"
        else
            echo "  $BASENAME: Initializing..."
        fi
    fi
done

echo ""
echo "=========================================="

