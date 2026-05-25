#!/bin/bash

# Wait for current experiments to complete and launch next batch
# Usage: bash wait_and_launch_next.sh

echo "=========================================="
echo "Waiting for current experiments to complete..."
echo "=========================================="

# Function to check if a process is running
is_running() {
    local PID=$1
    ps -p $PID > /dev/null 2>&1
    return $?
}

# Get PIDs of current experiments
GPU0_PID=$(ps aux | grep "CUDA_VISIBLE_DEVICES=0.*run_rq1_security.py.*CIFAR10.*byzantine_random_noise" | grep -v grep | awk '{print $2}' | head -1)
GPU1_PID=$(ps aux | grep "CUDA_VISIBLE_DEVICES=1.*run_rq1_security.py.*CIFAR100.*byzantine_random_noise" | grep -v grep | awk '{print $2}' | head -1)

echo "GPU 0 PID: $GPU0_PID (CIFAR10 + byzantine_random_noise)"
echo "GPU 1 PID: $GPU1_PID (CIFAR100 + byzantine_random_noise)"

# Wait for both to complete
while is_running $GPU0_PID || is_running $GPU1_PID; do
    sleep 60
    
    if is_running $GPU0_PID; then
        GPU0_STATUS="Running"
    else
        GPU0_STATUS="Completed"
    fi
    
    if is_running $GPU1_PID; then
        GPU1_STATUS="Running"
    else
        GPU1_STATUS="Completed"
    fi
    
    echo "[$(date +%H:%M:%S)] GPU 0: $GPU0_STATUS | GPU 1: $GPU1_STATUS"
done

echo ""
echo "=========================================="
echo "Both experiments completed!"
echo "=========================================="

# Check if experiments succeeded
if grep -q "Experiment completed successfully" /tmp/rq1_cifar10_byzantine_random_noise.log && \
   grep -q "Experiment completed successfully" /tmp/rq1_cifar100_byzantine_random_noise.log; then
    echo "✅ Both experiments succeeded"
    
    # Launch next batch
    echo ""
    echo "=========================================="
    echo "Launching next batch of experiments..."
    echo "=========================================="
    
    cd /home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code
    
    # GPU 0: CIFAR10 remaining attacks
    nohup bash experiments/scripts/runners/run_rq1_batch.sh 0 CIFAR10 > /tmp/rq1_cifar10_batch.log 2>&1 &
    echo "GPU 0: Started CIFAR10 batch (PID: $!)"
    
    # GPU 1: CIFAR100 remaining attacks
    nohup bash experiments/scripts/runners/run_rq1_batch.sh 1 CIFAR100 > /tmp/rq1_cifar100_batch.log 2>&1 &
    echo "GPU 1: Started CIFAR100 batch (PID: $!)"
    
else
    echo "❌ One or more experiments failed"
    echo "Check logs:"
    echo "  - /tmp/rq1_cifar10_byzantine_random_noise.log"
    echo "  - /tmp/rq1_cifar100_byzantine_random_noise.log"
    exit 1
fi

