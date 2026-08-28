#!/bin/bash
# 监控当前实验并在完成后启动参数扫描
# 智能调度脚本

set -e

PID_TO_MONITOR=588870
LOG_FILE="experiments/results/monitor_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "PoL-BFL 智能调度监控" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "监控PID: $PID_TO_MONITOR" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 检查进程是否存在
if ! ps -p $PID_TO_MONITOR > /dev/null; then
    echo "进程 $PID_TO_MONITOR 不存在，直接启动参数扫描" | tee -a "$LOG_FILE"
    LAUNCH_NOW=1
else
    echo "进程 $PID_TO_MONITOR 正在运行，等待完成..." | tee -a "$LOG_FILE"
    LAUNCH_NOW=0

    # 等待进程完成
    while ps -p $PID_TO_MONITOR > /dev/null; do
        echo "[$(date +%H:%M:%S)] 进程仍在运行，等待中..." | tee -a "$LOG_FILE"
        sleep 60  # 每分钟检查一次
    done

    echo "进程 $PID_TO_MONITOR 已完成。" | tee -a "$LOG_FILE"
    echo "完成时间: $(date)" | tee -a "$LOG_FILE"
fi

# 等待一小段时间确保资源释放
echo "等待5秒以确保GPU资源释放..." | tee -a "$LOG_FILE"
sleep 5

# 启动参数扫描
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "启动参数扫描实验" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# GPU 0 和 GPU 1 并行运行
echo "启动 GPU 0 参数扫描..." | tee -a "$LOG_FILE"
nohup bash experiments/scripts/run_param_scan_coarse_gpu0.sh > experiments/results/param_scan_gpu0.log 2>&1 &
GPU0_PID=$!
echo "GPU 0 PID: $GPU0_PID" | tee -a "$LOG_FILE"

echo "启动 GPU 1 参数扫描..." | tee -a "$LOG_FILE"
nohup bash experiments/scripts/run_param_scan_coarse_gpu1.sh > experiments/results/param_scan_gpu1.log 2>&1 &
GPU1_PID=$!
echo "GPU 1 PID: $GPU1_PID" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "参数扫描已启动。" | tee -a "$LOG_FILE"
echo "GPU 0 PID: $GPU0_PID" | tee -a "$LOG_FILE"
echo "GPU 1 PID: $GPU1_PID" | tee -a "$LOG_FILE"
echo "GPU 0 日志: experiments/results/param_scan_gpu0.log" | tee -a "$LOG_FILE"
echo "GPU 1 日志: experiments/results/param_scan_gpu1.log" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "监控命令:" | tee -a "$LOG_FILE"
echo "  tail -f experiments/results/param_scan_gpu0.log" | tee -a "$LOG_FILE"
echo "  tail -f experiments/results/param_scan_gpu1.log" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

