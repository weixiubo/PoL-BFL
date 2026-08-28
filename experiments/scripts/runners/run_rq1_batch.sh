#!/bin/bash

# RQ1 Batch Runner - Run remaining experiments sequentially
# Usage: bash run_rq1_batch.sh <GPU_ID> <DATASET>

GPU_ID=$1
DATASET=$2

if [ -z "$GPU_ID" ] || [ -z "$DATASET" ]; then
    echo "Usage: bash run_rq1_batch.sh <GPU_ID> <DATASET>"
    echo "Example: bash run_rq1_batch.sh 0 CIFAR10"
    exit 1
fi

# Attack types (excluding lazy_training and sybil_attack)
ATTACKS=(
    "byzantine_label_flipping"
    "byzantine_model_replacement"
    "free_riding_no_training"
    "no_attack"
)

echo "=========================================="
echo "Starting RQ1 batch experiments"
echo "GPU: $GPU_ID"
echo "Dataset: $DATASET"
echo "Attacks: ${ATTACKS[@]}"
echo "=========================================="

# Run each attack sequentially
for ATTACK in "${ATTACKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $DATASET + $ATTACK"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        python experiments/scripts/runners/run_rq1_security.py \
        --dataset $DATASET \
        --num_rounds 20 \
        --attacks $ATTACK \
        --baselines PoL_FL \
        2>&1 | tee /tmp/rq1_${DATASET,,}_${ATTACK}.log

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Experiment failed with exit code $EXIT_CODE"
        echo "Check log: /tmp/rq1_${DATASET,,}_${ATTACK}.log"
        exit $EXIT_CODE
    fi

    echo "[PASS] Completed: $DATASET + $ATTACK"
done

echo ""
echo "=========================================="
echo "All experiments completed successfully."
echo "=========================================="

