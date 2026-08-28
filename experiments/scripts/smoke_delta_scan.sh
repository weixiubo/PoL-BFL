#!/bin/bash

# Smoke Delta Parameter Scan
# Evaluates delta values against the configured TPR and FPR thresholds.

DATASET="MNIST"
MODEL="SimpleCNN"
NUM_ROUNDS=2
DEFENSE="PoL_FL"
ATTACK="byzantine_random_noise"

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiments/logs"
LOG_FILE="${LOG_DIR}/smoke_delta_scan_${TIMESTAMP}.log"

mkdir -p ${LOG_DIR}

echo "========================================" | tee -a ${LOG_FILE}
echo "Smoke Delta Parameter Scan" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
echo "Dataset: ${DATASET}" | tee -a ${LOG_FILE}
echo "Model: ${MODEL}" | tee -a ${LOG_FILE}
echo "Rounds: ${NUM_ROUNDS}" | tee -a ${LOG_FILE}
echo "Defense: ${DEFENSE}" | tee -a ${LOG_FILE}
echo "Attack: ${ATTACK}" | tee -a ${LOG_FILE}
echo "Log: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Delta values to test
DELTAS=(10.0 15.0 20.0 30.0 50.0)

# Test each delta
for DELTA in "${DELTAS[@]}"; do
    echo "Testing Delta: ${DELTA}" | tee -a ${LOG_FILE}
    echo "Started at: $(date)" | tee -a ${LOG_FILE}

    # Run the test
    python experiments/scripts/runners/run_rq1_security.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        --num_rounds ${NUM_ROUNDS} \
        --baselines ${DEFENSE} \
        --attacks ${ATTACK} \
        --pol_delta ${DELTA} \
        >> ${LOG_FILE} 2>&1

    echo "Completed at: $(date)" | tee -a ${LOG_FILE}
    echo "----------------------------------------" | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
done

echo "========================================" | tee -a ${LOG_FILE}
echo "Delta Scan Complete." | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

# Extract and summarize results
echo "" | tee -a ${LOG_FILE}
echo "Summary:" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

python3 << EOF | tee -a ${LOG_FILE}
import re

log_file = '${LOG_FILE}'

with open(log_file, 'r') as f:
    content = f.read()

# Split by delta tests
sections = content.split('Testing Delta: ')

print("| Delta | TPR | FPR | Attacks Applied | Status |")
print("|-------|-----|-----|----------------|--------|")

for section in sections[1:]:  # Skip first empty section
    # Extract delta value
    delta_match = re.search(r'^([\d.]+)', section)
    if not delta_match:
        continue
    delta = delta_match.group(1)

    # Extract TPR and FPR
    tpr_match = re.search(r'TPR \(Detection Rate\): ([\d.]+)', section)
    fpr_match = re.search(r'FPR \(False Positive Rate\): ([\d.]+)', section)

    tpr = tpr_match.group(1) if tpr_match else "N/A"
    fpr = fpr_match.group(1) if fpr_match else "N/A"

    # Count attacks applied
    attacks_applied = len(re.findall(r'Applied random_noise attack', section))

    # Determine status
    try:
        tpr_val = float(tpr) if tpr != "N/A" else 0
        fpr_val = float(fpr) if fpr != "N/A" else 1

        if tpr_val >= 0.8 and fpr_val < 0.1:
            status = "[PASS] Meets target"
        elif tpr_val >= 0.8 and fpr_val < 0.2:
            status = "[PASS] Meets relaxed target"
        elif tpr_val >= 0.6:
            status = "[WARNING] Acceptable"
        else:
            status = "[FAIL] Poor"
    except:
        status = "[UNKNOWN] Unknown"

    print(f"| {delta} | {tpr} | {fpr} | {attacks_applied} | {status} |")

print("")
print("Recommendation:")

# Find best delta (highest TPR with FPR < 0.1)
best_delta = None
best_tpr = 0
best_fpr = 1

for section in sections[1:]:
    delta_match = re.search(r'^([\d.]+)', section)
    if not delta_match:
        continue
    delta = float(delta_match.group(1))

    tpr_match = re.search(r'TPR \(Detection Rate\): ([\d.]+)', section)
    fpr_match = re.search(r'FPR \(False Positive Rate\): ([\d.]+)', section)

    if tpr_match and fpr_match:
        tpr = float(tpr_match.group(1))
        fpr = float(fpr_match.group(1))

        # Prefer FPR < 0.1, then highest TPR
        if fpr < 0.1:
            if tpr > best_tpr or (tpr == best_tpr and fpr < best_fpr):
                best_delta = delta
                best_tpr = tpr
                best_fpr = fpr
        elif best_delta is None:  # No qualifying delta has been selected.
            if tpr > best_tpr or (tpr == best_tpr and fpr < best_fpr):
                best_delta = delta
                best_tpr = tpr
                best_fpr = fpr

if best_delta:
    print(f"  Recommended Delta: {best_delta}")
    print(f"  Expected TPR: {best_tpr}")
    print(f"  Expected FPR: {best_fpr}")

    if best_fpr < 0.1:
        print(f"  [PASS] Selected configuration satisfies FPR < 0.1")
    elif best_fpr < 0.2:
        print(f"  [PASS] Selected configuration satisfies FPR < 0.2")
    else:
        print(f"  [WARNING] Consider testing higher delta values to reduce FPR")
else:
    print("  [FAIL] No suitable delta found. Consider testing different values.")

EOF

echo "" | tee -a ${LOG_FILE}
echo "Full log: ${LOG_FILE}" | tee -a ${LOG_FILE}
