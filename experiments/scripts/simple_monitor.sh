#!/bin/bash

# Simple experiment monitoring script

LOG_DIR="experiments/logs/tuning_2025-11-19"
RESULT_DIR="experiments/results/tuning_2025-11-19"

echo "=========================================="
echo "PoL-BFL Experiment Monitor"
echo "=========================================="

while true; do
    clear
    echo "=========================================="
    echo "PoL-BFL Experiment Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    
    echo ""
    echo "📊 GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    
    echo ""
    echo "📋 Running Processes:"
    ps aux | grep "run_rq" | grep -v grep | wc -l
    ps aux | grep "run_rq" | grep -v grep | awk '{print "  - " $NF}'
    
    echo ""
    echo "📁 Log Files:"
    for log in $LOG_DIR/*.log; do
        if [ -f "$log" ]; then
            size=$(du -h "$log" | cut -f1)
            lines=$(wc -l < "$log")
            name=$(basename "$log")
            echo "  $name: $size ($lines lines)"
            
            # Show last accuracy
            if grep -q "Test Accuracy:" "$log"; then
                acc=$(grep "Test Accuracy:" "$log" | tail -1 | awk '{print $NF}')
                echo "    └─ Latest Accuracy: $acc"
            fi
        fi
    done
    
    echo ""
    echo "📊 Result Directories:"
    for dir in $RESULT_DIR/*/; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            files=$(find "$dir" -type f | wc -l)
            size=$(du -sh "$dir" | cut -f1)
            echo "  $name: $files files, $size"
        fi
    done
    
    echo ""
    echo "⏳ Waiting 30 seconds... (Press Ctrl+C to exit)"
    sleep 30
done

