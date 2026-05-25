#!/bin/bash

# 夜间执行进度检查脚本

BASE_DIR="/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code"
LOG_DIR="$BASE_DIR/experiments/logs/tuning_2025-11-19"
RESULT_DIR="$BASE_DIR/experiments/results/tuning_2025-11-19"

echo "=========================================="
echo "PoL-BFL 夜间执行进度检查"
echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 1. 进程状态
echo ""
echo "【进程状态】"
echo "-----------------------------------------"
RUNNING=$(ps aux | grep "run_rq" | grep -v grep | wc -l)
if [ $RUNNING -eq 0 ]; then
  echo "✓ 所有实验已完成"
else
  echo "✓ 正在运行 $RUNNING 个进程"
  ps aux | grep "run_rq" | grep -v grep | awk '{print "  - PID " $2 ": " $NF}'
fi

# 2. GPU 状态
echo ""
echo "【GPU 状态】"
echo "-----------------------------------------"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits | awk '{printf "GPU%d: %d/%d MB (GPU: %d%%, Mem: %d%%)\n", $1, $3, $4, $5, $6}'

# 3. 日志进度
echo ""
echo "【日志进度】"
echo "-----------------------------------------"

echo "RQ1 CIFAR-10:"
if [ -f "$LOG_DIR/rq1_cifar10_tuning.log" ]; then
  LINES=$(wc -l < "$LOG_DIR/rq1_cifar10_tuning.log")
  LAST_LINE=$(tail -1 "$LOG_DIR/rq1_cifar10_tuning.log")
  echo "  - 日志行数: $LINES"
  echo "  - 最后一行: $LAST_LINE"
else
  echo "  - 日志不存在"
fi

echo ""
echo "RQ2 CIFAR-10:"
if [ -f "$LOG_DIR/rq2_cifar10_tuning.log" ]; then
  LINES=$(wc -l < "$LOG_DIR/rq2_cifar10_tuning.log")
  LAST_LINE=$(tail -1 "$LOG_DIR/rq2_cifar10_tuning.log")
  echo "  - 日志行数: $LINES"
  echo "  - 最后一行: $LAST_LINE"
else
  echo "  - 日志不存在"
fi

# 4. 结果文件统计
echo ""
echo "【结果文件统计】"
echo "-----------------------------------------"

RQ1_CIFAR10_COUNT=$(find "$RESULT_DIR/rq1_cifar10_tuning" -name "*.csv" 2>/dev/null | wc -l)
echo "RQ1 CIFAR-10: $RQ1_CIFAR10_COUNT 个 CSV (预期: 77)"

RQ2_CIFAR10_COUNT=$(find "$RESULT_DIR" -path "*/rq2_ablation/*" -name "*cifar10*" -name "*.csv" 2>/dev/null | wc -l)
echo "RQ2 CIFAR-10: $RQ2_CIFAR10_COUNT 个 CSV (预期: 15)"

RQ1_MNIST_COUNT=$(find "$RESULT_DIR/rq1_mnist_tuning" -name "*.csv" 2>/dev/null | wc -l)
echo "RQ1 MNIST: $RQ1_MNIST_COUNT 个 CSV (预期: 16)"

RQ2_MNIST_COUNT=$(find "$RESULT_DIR/rq2_ablation/20251119_202046_360315" -name "rq2_rounds*.csv" 2>/dev/null | wc -l)
echo "RQ2 MNIST: $RQ2_MNIST_COUNT 个 CSV (预期: 15)"

# 5. 磁盘使用
echo ""
echo "【磁盘使用】"
echo "-----------------------------------------"
du -sh "$RESULT_DIR" 2>/dev/null | awk '{print "结果目录: " $1}'
du -sh "$LOG_DIR" 2>/dev/null | awk '{print "日志目录: " $1}'

# 6. 最近修改的文件
echo ""
echo "【最近修改的文件】"
echo "-----------------------------------------"
echo "最新的 CSV 文件:"
find "$RESULT_DIR" -name "*.csv" -type f -printf '%T@ %p\n' 2>/dev/null | \
  sort -rn | head -5 | awk '{print "  - " $2 " (" strftime("%H:%M:%S", $1) ")"}'

echo ""
echo "=========================================="
echo "检查完成"
echo "=========================================="

