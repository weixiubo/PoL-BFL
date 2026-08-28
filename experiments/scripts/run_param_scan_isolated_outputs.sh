#!/bin/bash
#
# 修正后的参数扫描脚本 - 使用独立输出目录避免文件冲突
# 用于后续实验，避免并行运行时的文件移动问题
#

set -e  # Exit on error

# 配置
CONDA_ENV="polbfl"
GPU_ID=${1:-0}  # 从命令行参数获取GPU ID，默认为0
DATASET="MNIST"
MODEL="SimpleCNN"
NUM_ROUNDS=2
NUM_CLIENTS=20
CLIENTS_PER_ROUND=10

# 攻击类型（代表性攻击）
ATTACKS="byzantine_random_noise,byzantine_model_replacement,byzantine_gradient_inversion,free_riding_no_training"

# Baseline（只测试PoL_FL）
BASELINES="PoL_FL"

# 输出目录
OUTPUT_BASE="experiments/results/param_scan_coarse"
LOG_DIR="$OUTPUT_BASE"
mkdir -p "$LOG_DIR"

# 日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/scan_gpu${GPU_ID}_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "参数扫描实验 - GPU ${GPU_ID}" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 参数组合（根据GPU ID分配）
if [ "$GPU_ID" -eq 0 ]; then
    DELTAS=(1.0 3.0)
elif [ "$GPU_ID" -eq 1 ]; then
    DELTAS=(5.0 10.0)
else
    echo "错误: 不支持的GPU ID: $GPU_ID" | tee -a "$LOG_FILE"
    exit 1
fi

VRS=(0.3 0.5 1.0)

# 运行实验
for delta in "${DELTAS[@]}"; do
    for vr in "${VRS[@]}"; do
        echo "========================================" | tee -a "$LOG_FILE"
        echo "运行: Delta=${delta}, VR=${vr}" | tee -a "$LOG_FILE"
        echo "开始时间: $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"

        # 创建独立的临时输出目录（关键修正。）
        temp_output="experiments/results/rq1_security_temp_gpu${GPU_ID}_delta${delta}_vr${vr}"
        mkdir -p "$temp_output"

        # 创建最终输出目录
        output_subdir="${OUTPUT_BASE}/delta${delta}_vr${vr}"
        mkdir -p "$output_subdir"

        start_time=$(date +%s)

        # 运行实验，使用独立的临时输出目录
        python experiments/scripts/runners/run_rq1_security.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --num_rounds "$NUM_ROUNDS" \
            --num_clients "$NUM_CLIENTS" \
            --clients_per_round "$CLIENTS_PER_ROUND" \
            --pol_delta "$delta" \
            --verification_rate "$vr" \
            --attacks "$ATTACKS" \
            --baselines "$BASELINES" \
            --output_dir "$temp_output" \
            >> "$LOG_FILE" 2>&1

        end_time=$(date +%s)
        duration=$((end_time - start_time))

        echo "  完成: $(date) (耗时: ${duration}秒)" | tee -a "$LOG_FILE"

        # 移动结果文件到最终目录（使用文件锁确保原子性）
        (
            flock -x 200

            # 移动结果文件
            if [ -f "$temp_output/rq1_results.json" ]; then
                mv "$temp_output/rq1_results.json" "$output_subdir/rq1_results_gpu${GPU_ID}.json"
            fi

            if [ -f "$temp_output/config.json" ]; then
                mv "$temp_output/config.json" "$output_subdir/config_gpu${GPU_ID}.json"
            fi

            # 移动CSV文件
            if ls "$temp_output"/*.csv 1> /dev/null 2>&1; then
                mv "$temp_output"/*.csv "$output_subdir/"
            fi

            # 清理临时目录
            rm -rf "$temp_output"

        ) 200>/tmp/param_scan_gpu${GPU_ID}.lock

        # 提取关键指标
        if [ -f "$output_subdir/rq1_results_gpu${GPU_ID}.json" ]; then
            python3 -c "
import json
with open('$output_subdir/rq1_results_gpu${GPU_ID}.json') as f:
    data = json.load(f)
    if data:
        print('  关键指标:')
        for item in data:
            attack = item['attack_type']
            tpr = item['detection_metrics'].get('TPR', 0)
            fpr = item['detection_metrics'].get('FPR', 0)
            acc = item.get('final_accuracy', 0)
            print(f'    {attack:40s}  TPR={tpr:.3f} FPR={fpr:.3f} Acc={acc:.3f}')
" | tee -a "$LOG_FILE"
        fi

        echo "" | tee -a "$LOG_FILE"
    done
done

echo "========================================" | tee -a "$LOG_FILE"
echo "GPU ${GPU_ID} 所有实验完成." | tee -a "$LOG_FILE"
echo "结束时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

