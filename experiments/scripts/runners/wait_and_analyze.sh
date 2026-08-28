#!/bin/bash

# Wait for experiments to complete and analyze results
# Usage: bash wait_and_analyze.sh <LOG1> <LOG2>

LOG1=$1
LOG2=$2

if [ -z "$LOG1" ] || [ -z "$LOG2" ]; then
    echo "Usage: bash wait_and_analyze.sh <LOG1> <LOG2>"
    echo "Example: bash wait_and_analyze.sh /tmp/rq1_cifar10_byzantine_random_noise.log /tmp/rq1_cifar100_byzantine_random_noise.log"
    exit 1
fi

echo "=========================================="
echo "Waiting for experiments to complete..."
echo "Log 1: $LOG1"
echo "Log 2: $LOG2"
echo "=========================================="

# Function to check if experiment is complete
is_complete() {
    local LOG=$1
    grep -q "Experiment completed successfully" "$LOG" 2>/dev/null
    return $?
}

# Wait loop
while true; do
    LOG1_COMPLETE=false
    LOG2_COMPLETE=false

    if is_complete "$LOG1"; then
        LOG1_COMPLETE=true
    fi

    if is_complete "$LOG2"; then
        LOG2_COMPLETE=true
    fi

    if $LOG1_COMPLETE && $LOG2_COMPLETE; then
        echo ""
        echo "[PASS] Both experiments completed."
        break
    fi

    # Show progress
    ROUND1=$(grep -oP "Round \d+/\d+" "$LOG1" | tail -1)
    ROUND2=$(grep -oP "Round \d+/\d+" "$LOG2" | tail -1)

    echo "[$(date +%H:%M:%S)] Exp1: $ROUND1 | Exp2: $ROUND2"

    sleep 60
done

echo ""
echo "=========================================="
echo "Analyzing results..."
echo "=========================================="

# Analyze results using Python
python3 << 'EOF'
import json
import sys

def analyze_experiment(results_file, experiment_name):
    """Analyze a single experiment result"""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Find the experiment
        exp = None
        for e in data['experiments']:
            if experiment_name in e['experiment_id']:
                exp = e
                break

        if not exp:
            print(f"[FAIL] Experiment not found: {experiment_name}")
            return False

        # Extract metrics
        tpr_cond = exp['detection_metrics']['TPR_conditional']
        tp_cond = exp['detection_metrics']['TP_conditional']
        fn_cond = exp['detection_metrics']['FN_conditional']
        total_verif = exp['detection_metrics']['total_malicious_verifications']
        fpr = exp['detection_metrics']['FPR']
        final_acc = exp['final_accuracy']

        print(f"\n{experiment_name}:")
        print(f"  TPR_conditional: {tpr_cond:.2%} ({tp_cond}/{total_verif})")
        print(f"  FN_conditional: {fn_cond}")
        print(f"  FPR: {fpr:.2%}")
        print(f"  Final Accuracy: {final_acc:.4f}")

        # Check quality thresholds
        issues = []
        if tpr_cond < 0.90:
            issues.append(f"TPR_conditional ({tpr_cond:.2%}) < 90%")
        if fpr > 0.08:
            issues.append(f"FPR ({fpr:.2%}) > 8%")

        if issues:
            print(f"  [WARNING]  Issues: {', '.join(issues)}")
            return False
        else:
            print(f"  [PASS] All metrics passed.")
            return True

    except Exception as e:
        print(f"[FAIL] Error analyzing {experiment_name}: {e}")
        return False

# Analyze both experiments
results_file = 'experiments/results/rq1_security/rq1_results.json'

exp1_ok = analyze_experiment(results_file, 'CIFAR10_byzantine_random_noise')
exp2_ok = analyze_experiment(results_file, 'CIFAR100_byzantine_random_noise')

print("\n" + "="*50)
if exp1_ok and exp2_ok:
    print("[PASS] Both experiments passed quality checks.")
    sys.exit(0)
else:
    print("[FAIL] One or more experiments failed quality checks")
    sys.exit(1)

EOF

ANALYSIS_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "[PASS] Analysis complete - Ready for next batch"
else
    echo "[FAIL] Analysis reported threshold violations."
fi
echo "=========================================="

exit $ANALYSIS_EXIT_CODE
