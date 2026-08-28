#!/bin/bash
POLBFL_ROOT="${POLBFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# GPU 1 extended parameter scan
# 负责: delta=[5.0, 10.0]

set -e

GPU_ID=1
DATASET="MNIST"
MODEL="SimpleCNN"
NUM_ROUNDS=2
OUTPUT_BASE="experiments/results/param_scan_coarse"

mkdir -p "$OUTPUT_BASE"

# GPU 1 负责的参数范围
DELTAS=(5.0 10.0)
VRS=(0.3 0.5 1.0)

# 代表性攻击
ATTACKS="byzantine_random_noise,byzantine_model_replacement,byzantine_gradient_inversion,free_riding_no_training"
BASELINES="PoL_FL"

LOG_FILE="$OUTPUT_BASE/scan_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "GPU $GPU_ID extended parameter scan" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "负责 Delta: ${DELTAS[*]}" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

total_experiments=$((${#DELTAS[@]} * ${#VRS[@]}))
current=0

for delta in "${DELTAS[@]}"; do
    for vr in "${VRS[@]}"; do
        current=$((current + 1))

        echo "" | tee -a "$LOG_FILE"
        echo "[GPU$GPU_ID $current/$total_experiments] Delta=$delta VR=$vr" | tee -a "$LOG_FILE"
        echo "  开始: $(date)" | tee -a "$LOG_FILE"

        # 设置环境变量
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        export CUBLAS_WORKSPACE_CONFIG=:4096:8

        output_subdir="${OUTPUT_BASE}/delta${delta}_vr${vr}"
        mkdir -p "$output_subdir"

        start_time=$(date +%s)

        cd $POLBFL_ROOT
        python experiments/scripts/runners/run_rq1_security.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --num_rounds "$NUM_ROUNDS" \
            --pol_delta "$delta" \
            --verification_rate "$vr" \
            --attacks "$ATTACKS" \
            --baselines "$BASELINES" \
            >> "$LOG_FILE" 2>&1

        end_time=$(date +%s)
        duration=$((end_time - start_time))

        echo "  完成: $(date) (耗时: ${duration}秒)" | tee -a "$LOG_FILE"

        # 移动结果文件
        mv experiments/results/rq1_security/rq1_results.json "$output_subdir/rq1_results_gpu${GPU_ID}.json" 2>/dev/null || true
        mv experiments/results/rq1_security/rq1_rounds_*.csv "$output_subdir/" 2>/dev/null || true
        mv experiments/results/rq1_security/config.json "$output_subdir/config_gpu${GPU_ID}.json" 2>/dev/null || true

        # 提取关键指标
        if [ -f "$output_subdir/rq1_results_gpu${GPU_ID}.json" ]; then
            python3 -c "
import json
with open('$output_subdir/rq1_results_gpu${GPU_ID}.json') as f:
    data = json.load(f)
    if data:
        for item in data:
            tpr = item['detection_metrics'].get('TPR', 0)
            fpr = item['detection_metrics'].get('FPR', 0)
            acc = item.get('final_accuracy', 0)
            attack = item['attack_type']
            print(f'  {attack:40s} TPR={tpr:.3f} FPR={fpr:.3f} Acc={acc:.3f}')
" | tee -a "$LOG_FILE"
        fi
    done
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "GPU $GPU_ID 扫描完成。" | tee -a "$LOG_FILE"
echo "完成时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
