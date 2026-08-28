#!/bin/bash
# PoL-BFL RQ1 coarse parameter scan
# Objective: identify a parameter range with a reduced-scale scan.
# 策略: 2轮缩减规模测试，4个代表性攻击

set -e

# 配置
DATASET="MNIST"
MODEL="SimpleCNN"
NUM_ROUNDS=2
GPU_ID=${1:-0}  # 默认使用GPU 0
OUTPUT_DIR="experiments/results/param_scan_coarse"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 参数范围
DELTAS=(3.0 5.0 10.0)
VRS=(0.3 0.5 1.0)

# 代表性攻击
ATTACKS=(
    "byzantine_random_noise"
    "byzantine_model_replacement"
    "byzantine_gradient_inversion"
    "free_riding_no_training"
)

# 日志文件
LOG_FILE="$OUTPUT_DIR/scan_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "PoL-BFL coarse parameter scan" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "数据集: $DATASET" | tee -a "$LOG_FILE"
echo "轮数: $NUM_ROUNDS" | tee -a "$LOG_FILE"
echo "Delta范围: ${DELTAS[*]}" | tee -a "$LOG_FILE"
echo "VR范围: ${VRS[*]}" | tee -a "$LOG_FILE"
echo "攻击类型: ${ATTACKS[*]}" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 计数器
total_experiments=$((${#DELTAS[@]} * ${#VRS[@]} * ${#ATTACKS[@]}))
current=0

# 主循环
for delta in "${DELTAS[@]}"; do
    for vr in "${VRS[@]}"; do
        for attack in "${ATTACKS[@]}"; do
            current=$((current + 1))

            echo "" | tee -a "$LOG_FILE"
            echo "[$current/$total_experiments] 运行实验:" | tee -a "$LOG_FILE"
            echo "  Delta: $delta" | tee -a "$LOG_FILE"
            echo "  VR: $vr" | tee -a "$LOG_FILE"
            echo "  Attack: $attack" | tee -a "$LOG_FILE"
            echo "  开始时间: $(date)" | tee -a "$LOG_FILE"

            # 设置环境变量
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            export CUBLAS_WORKSPACE_CONFIG=:4096:8
            export POL_DELTA_OVERRIDE=$delta
            export POL_VERIFICATION_RATE=$vr
            export POL_MIN_PAIR_SUCCESS_RATE=0.99
            export POL_ALWAYS_VERIFY_LAST_K=2
            export POL_RANDOM_Q=3

            # 运行实验
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
                2>&1 | tee -a "$LOG_FILE"

            end_time=$(date +%s)
            duration=$((end_time - start_time))

            echo "  完成时间: $(date)" | tee -a "$LOG_FILE"
            echo "  耗时: ${duration}秒 ($(($duration / 60))分钟)" | tee -a "$LOG_FILE"

            # 提取关键指标
            if [ -f "$output_subdir/rq1_results.json" ]; then
                tpr=$(python -c "import json; data=json.load(open('$output_subdir/rq1_results.json')); print(data[0]['detection_metrics']['TPR'] if data else 'N/A')")
                fpr=$(python -c "import json; data=json.load(open('$output_subdir/rq1_results.json')); print(data[0]['detection_metrics']['FPR'] if data else 'N/A')")
                acc=$(python -c "import json; data=json.load(open('$output_subdir/rq1_results.json')); print(data[0]['final_accuracy'] if data else 'N/A')")

                echo "  TPR: $tpr" | tee -a "$LOG_FILE"
                echo "  FPR: $fpr" | tee -a "$LOG_FILE"
                echo "  Accuracy: $acc" | tee -a "$LOG_FILE"
            fi
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Coarse parameter scan completed." | tee -a "$LOG_FILE"
echo "完成时间: $(date)" | tee -a "$LOG_FILE"
echo "总实验数: $total_experiments" | tee -a "$LOG_FILE"
echo "结果目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 生成汇总报告
python experiments/scripts/analyze_param_scan.py --input_dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/summary.json"

echo "汇总报告已生成: $OUTPUT_DIR/summary.json" | tee -a "$LOG_FILE"
