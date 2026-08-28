#!/bin/bash
# GPU 0 coarse parameter scan
# 负责: delta=[3.0, 5.0] 的所有组合

set -e

GPU_ID=0
DATASET="MNIST"
MODEL="SimpleCNN"
NUM_ROUNDS=2
OUTPUT_DIR="experiments/results/param_scan_coarse"

mkdir -p "$OUTPUT_DIR"

# GPU 0 负责的参数范围
DELTAS=(1.0 3.0)
VRS=(0.3 0.5 1.0)

ATTACKS=(
    "byzantine_random_noise"
    "byzantine_model_replacement"
    "byzantine_gradient_inversion"
    "free_riding_no_training"
)

LOG_FILE="$OUTPUT_DIR/scan_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "GPU $GPU_ID coarse parameter scan" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "负责 Delta: ${DELTAS[*]}" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

total_experiments=$((${#DELTAS[@]} * ${#VRS[@]} * ${#ATTACKS[@]}))
current=0

for delta in "${DELTAS[@]}"; do
    for vr in "${VRS[@]}"; do
        for attack in "${ATTACKS[@]}"; do
            current=$((current + 1))

            echo "" | tee -a "$LOG_FILE"
            echo "[GPU$GPU_ID $current/$total_experiments] Delta=$delta VR=$vr Attack=$attack" | tee -a "$LOG_FILE"
            echo "  开始: $(date)" | tee -a "$LOG_FILE"

            export CUDA_VISIBLE_DEVICES=$GPU_ID
            export CUBLAS_WORKSPACE_CONFIG=:4096:8
            export POL_DELTA_OVERRIDE=$delta
            export POL_VERIFICATION_RATE=$vr
            export POL_MIN_PAIR_SUCCESS_RATE=0.99
            export POL_ALWAYS_VERIFY_LAST_K=2
            export POL_RANDOM_Q=3

            output_subdir="${OUTPUT_DIR}/delta${delta}_vr${vr}_${attack}"
            mkdir -p "$output_subdir"

            start_time=$(date +%s)

            python experiments/scripts/runners/run_rq1_security.py \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --num_rounds "$NUM_ROUNDS" \
                --pol_delta "$delta" \
                --attack_type "$attack" \
                --output_dir "$output_subdir" \
                >> "$LOG_FILE" 2>&1

            end_time=$(date +%s)
            duration=$((end_time - start_time))

            echo "  完成: $(date) (耗时: ${duration}秒)" | tee -a "$LOG_FILE"

            if [ -f "$output_subdir/rq1_results.json" ]; then
                tpr=$(python3 -c "import json; data=json.load(open('$output_subdir/rq1_results.json')); print(f\"{data[0]['detection_metrics']['TPR']:.3f}\" if data else 'N/A')" 2>/dev/null || echo "N/A")
                fpr=$(python3 -c "import json; data=json.load(open('$output_subdir/rq1_results.json')); print(f\"{data[0]['detection_metrics']['FPR']:.3f}\" if data else 'N/A')" 2>/dev/null || echo "N/A")
                echo "  TPR=$tpr FPR=$fpr" | tee -a "$LOG_FILE"
            fi
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "GPU $GPU_ID 扫描完成。" | tee -a "$LOG_FILE"
echo "完成时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
