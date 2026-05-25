#!/bin/bash
# 顺序运行 delta 参数扫描实验
# 用法: bash run_sequential_delta_scans.sh GPU_ID

GPU_ID=${1:-1}
DATASET="MNIST"
ROUNDS=10

echo "================================================================================"
echo "Delta 参数扫描 - 顺序执行"
echo "================================================================================"
echo "GPU: $GPU_ID"
echo "Dataset: $DATASET"
echo "Rounds: $ROUNDS"
echo "================================================================================"
echo ""

# Delta 值列表
DELTA_VALUES=(5.0 7.5 10.0 15.0 20.0)

for delta in "${DELTA_VALUES[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "🚀 启动实验: delta=$delta"
    echo "--------------------------------------------------------------------------------"
    
    # 生成标签
    TAG="delta$(echo $delta | tr '.' '_')_scan"
    
    # 启动实验
    bash experiments/scripts/run_param_scan_rq1.sh "$DATASET" "$GPU_ID" delta "$delta" "$TAG"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ delta=$delta 完成"
    else
        echo "❌ delta=$delta 失败 (exit code: $EXIT_CODE)"
        echo "⚠️ 继续下一个实验..."
    fi
    
    echo ""
    sleep 5
done

echo "================================================================================"
echo "所有 delta 扫描实验完成"
echo "================================================================================"
echo ""
echo "结果文件："
for delta in "${DELTA_VALUES[@]}"; do
    TAG="delta$(echo $delta | tr '.' '_')_scan"
    LOG_FILE=$(ls -t experiments/logs/rq1_param_scan_*_${TAG}.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILE" ]; then
        echo "  delta=$delta: $LOG_FILE"
    fi
done
echo ""

