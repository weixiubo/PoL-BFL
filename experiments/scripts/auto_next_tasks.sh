#!/usr/bin/env bash
# 自动启动后续任务脚本
# 在RQ1完成后自动启动RQ4和RQ5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo "自动任务启动脚本"
echo "=========================================="
echo "时间: $(date)"
echo ""

# 检查RQ1是否完成
check_rq1_complete() {
    local gpu=$1
    local result_dir="experiments/results/rq1_mnist_polfl_20r_gpu${gpu}"
    
    if [ ! -f "$result_dir/rq1_results.json" ]; then
        return 1
    fi
    
    # 检查是否有5个攻击的结果
    local count=$(python3 -c "
import json
with open('$result_dir/rq1_results.json') as f:
    data = json.load(f)
print(len(set(exp['attack_type'] for exp in data)))
" 2>/dev/null || echo "0")
    
    if [ "$count" -ge 5 ]; then
        return 0
    else
        return 1
    fi
}

# 等待RQ1完成
echo "等待RQ1实验完成..."
echo ""

while true; do
    gpu0_done=false
    gpu1_done=false
    
    if check_rq1_complete 0; then
        gpu0_done=true
    fi
    
    if check_rq1_complete 1; then
        gpu1_done=true
    fi
    
    if $gpu0_done && $gpu1_done; then
        echo "✅ RQ1实验已完成！"
        break
    fi
    
    echo "$(date +%H:%M:%S) - GPU0: $($gpu0_done && echo '✅' || echo '⏳') | GPU1: $($gpu1_done && echo '✅' || echo '⏳')"
    sleep 60  # 每分钟检查一次
done

echo ""
echo "=========================================="
echo "开始启动后续任务"
echo "=========================================="
echo ""

# 分析RQ1结果
echo "### 分析RQ1结果"
python3 experiments/scripts/analyze_rq1_results.py

# 启动RQ4 (GPU 0)
echo ""
echo "### 启动RQ4 (GPU 0)"
bash experiments/scripts/run_quick_rq4.sh MNIST rq4_auto 0 15

# 启动RQ5 (GPU 1)
echo ""
echo "### 启动RQ5 (GPU 1)"
bash experiments/scripts/run_rq5_mnist_smoke.sh 1

echo ""
echo "=========================================="
echo "所有后续任务已启动"
echo "=========================================="
echo "时间: $(date)"
echo ""
echo "监控命令:"
echo "  - RQ4: tail -f experiments/logs/rq4_*_rq4_auto.log"
echo "  - RQ5: tail -f experiments/logs/rq5_*.log"
echo ""

