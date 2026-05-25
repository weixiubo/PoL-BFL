#!/bin/bash

# 并行执行监控脚本
# 用法: bash monitor_parallel.sh

cd /home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code

echo "=========================================="
echo "并行执行监控 (2025-11-20)"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 监控函数
monitor_task() {
    local task_name=$1
    local log_file=$2
    local gpu=$3
    
    echo -e "${BLUE}【$task_name】${NC}"
    
    if [ ! -f "$log_file" ]; then
        echo -e "${RED}✗ 日志文件不存在: $log_file${NC}"
        return
    fi
    
    # 检查进程
    if [ "$gpu" = "GPU0" ]; then
        local process_count=$(ps aux | grep "run_rq1_security.py" | grep -v grep | wc -l)
    else
        local process_count=$(ps aux | grep "run_rq2_ablation.py" | grep -v grep | wc -l)
    fi
    
    if [ $process_count -gt 0 ]; then
        echo -e "${GREEN}✓ 进程运行中${NC}"
    else
        echo -e "${YELLOW}✗ 进程已停止${NC}"
    fi
    
    # 显示最后 5 行日志
    echo "最后日志:"
    tail -5 "$log_file" | sed 's/^/  /'
    
    # 显示文件大小
    local size=$(du -h "$log_file" | cut -f1)
    echo "日志大小: $size"
    
    echo ""
}

# 监控 GPU 使用
echo -e "${BLUE}【GPU 使用情况】${NC}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader
echo ""

# 监控 GPU0 任务
monitor_task "GPU0: MNIST 快速诊断" \
    "experiments/logs/tuning_2025-11-20/rq1_mnist_quick_test.log" \
    "GPU0"

# 监控 GPU1 任务
monitor_task "GPU1: RQ2 MNIST 补齐" \
    "experiments/logs/tuning_2025-11-20/rq2_mnist_full.log" \
    "GPU1"

# 显示结果文件
echo -e "${BLUE}【结果文件】${NC}"
echo "GPU0 结果:"
ls -lh experiments/results/tuning_2025-11-20/rq1_mnist_quick_test/*.csv 2>/dev/null | wc -l
echo "个 CSV 文件"

echo "GPU1 结果:"
ls -lh experiments/results/tuning_2025-11-20/rq2_mnist_full/*/*.csv 2>/dev/null | wc -l
echo "个 CSV 文件"

echo ""
echo "=========================================="
echo "监控完成 ($(date))"
echo "=========================================="

