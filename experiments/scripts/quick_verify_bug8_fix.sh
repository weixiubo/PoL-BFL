#!/bin/bash

# Quick verification script for Bug 8 fix
# Tests all attack types with 2 rounds to verify attacks are being applied

DATASET="MNIST"
MODEL="SimpleCNN"
NUM_ROUNDS=2
DEFENSE="PoL_FL"

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiments/logs"
LOG_FILE="${LOG_DIR}/quick_verify_bug8_${TIMESTAMP}.log"

mkdir -p ${LOG_DIR}

echo "========================================" | tee -a ${LOG_FILE}
echo "Quick Bug 8 Fix Verification" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
echo "Dataset: ${DATASET}" | tee -a ${LOG_FILE}
echo "Model: ${MODEL}" | tee -a ${LOG_FILE}
echo "Rounds: ${NUM_ROUNDS}" | tee -a ${LOG_FILE}
echo "Defense: ${DEFENSE}" | tee -a ${LOG_FILE}
echo "Log: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# List of attacks to test
ATTACKS=(
    "byzantine_random_noise"
    "byzantine_model_replacement"
    "byzantine_label_flipping"
    "byzantine_gradient_inversion"
)

# Test each attack
for ATTACK in "${ATTACKS[@]}"; do
    echo "Testing: ${ATTACK}" | tee -a ${LOG_FILE}
    echo "Started at: $(date)" | tee -a ${LOG_FILE}

    # Run the test
    python experiments/scripts/runners/run_rq1_security.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        --num_rounds ${NUM_ROUNDS} \
        --baselines ${DEFENSE} \
        --attacks ${ATTACK} \
        >> ${LOG_FILE} 2>&1

    echo "Completed at: $(date)" | tee -a ${LOG_FILE}

    # Check if attacks were applied (count in the entire log file)
    ATTACK_COUNT=$(grep "Applied.*attack" ${LOG_FILE} | grep -c "${ATTACK}" || echo "0")

    echo "Attack applications found: ${ATTACK_COUNT}" | tee -a ${LOG_FILE}

    if [ "${ATTACK_COUNT}" -gt 0 ] 2>/dev/null; then
        echo "✅ ${ATTACK}: PASS (${ATTACK_COUNT} attacks applied)" | tee -a ${LOG_FILE}
    else
        echo "❌ ${ATTACK}: FAIL (no attacks applied)" | tee -a ${LOG_FILE}
    fi

    echo "----------------------------------------" | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
done

echo "========================================" | tee -a ${LOG_FILE}
echo "Verification Complete!" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

# Summary
echo "" | tee -a ${LOG_FILE}
echo "Summary:" | tee -a ${LOG_FILE}
for ATTACK in "${ATTACKS[@]}"; do
    ATTACK_COUNT=$(grep "Applied.*attack" ${LOG_FILE} | grep -c "${ATTACK}" || echo "0")
    if [ "${ATTACK_COUNT}" -gt 0 ] 2>/dev/null; then
        echo "  ✅ ${ATTACK}: ${ATTACK_COUNT} attacks applied" | tee -a ${LOG_FILE}
    else
        echo "  ❌ ${ATTACK}: NO attacks applied" | tee -a ${LOG_FILE}
    fi
done

echo "" | tee -a ${LOG_FILE}
echo "Full log: ${LOG_FILE}" | tee -a ${LOG_FILE}

