#!/bin/bash
# 监控参数扫描进度

OUTPUT_BASE="experiments/results/param_scan_stage1"

echo "========================================"
echo "参数扫描进度监控"
echo "========================================"
echo "时间: $(date)"
echo ""

# 检查GPU状态
echo "GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s%% 利用率, %s/%s MB 内存\n", $1, $3, $4, $5}'
echo ""

# 检查运行进程
echo "运行进程:"
ps aux | grep "run_param_scan" | grep -v grep | awk '{print "  PID " $2 ": " $11 " " $12 " " $13}'
echo ""

# 检查完成的实验
echo "已完成的参数组合:"
completed=0
for dir in "$OUTPUT_BASE"/delta*_vr*/; do
    if [ -d "$dir" ]; then
        if [ -f "$dir/rq1_results_gpu0.json" ] || [ -f "$dir/rq1_results_gpu1.json" ]; then
            dirname=$(basename "$dir")
            completed=$((completed + 1))
            echo "  ✅ $dirname"
        fi
    fi
done

total=12  # 4 deltas × 3 VRs
echo ""
echo "进度: $completed/$total 组合完成 ($(( completed * 100 / total ))%)"
echo ""

# 显示最新日志
echo "最新日志 (GPU 0):"
tail -5 /tmp/param_scan_gpu0_v2.log 2>/dev/null | sed 's/^/  /'
echo ""

echo "最新日志 (GPU 1):"
tail -5 /tmp/param_scan_gpu1_v2.log 2>/dev/null | sed 's/^/  /'
echo ""

echo "========================================"

