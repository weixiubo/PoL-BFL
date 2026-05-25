#!/bin/bash
# 自动启动下一批实验的脚本
# 等待 vr04 完成后，自动启动 vr05 和 delta 扫描

echo "================================================================================"
echo "自动实验启动脚本"
echo "================================================================================"
echo ""

# 检查 vr04 是否还在运行
check_vr04_running() {
    ps aux | grep "verification_rate 0.4" | grep -v grep | wc -l
}

# 等待 vr04 完成
echo "检查 vr04 状态..."
vr04_count=$(check_vr04_running)

if [ $vr04_count -gt 0 ]; then
    echo "⏳ vr04 还在运行中（$vr04_count 个进程）"
    echo "等待 vr04 完成..."
    
    while [ $(check_vr04_running) -gt 0 ]; do
        sleep 300  # 每 5 分钟检查一次
        echo "  $(date '+%H:%M:%S') - vr04 仍在运行..."
    done
    
    echo "✅ vr04 已完成！"
else
    echo "✅ vr04 已完成"
fi

echo ""
echo "================================================================================"
echo "启动下一批实验"
echo "================================================================================"
echo ""

# 检查 GPU 可用性
echo "检查 GPU 状态..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "选择实验启动策略："
echo "  1. 保守策略：只在 GPU 1 上顺序运行"
echo "  2. 激进策略：GPU 0 和 GPU 1 并行运行"
echo ""

# 默认使用保守策略
STRATEGY=${1:-"conservative"}

if [ "$STRATEGY" == "aggressive" ]; then
    echo "📊 使用激进策略：并行运行"
    echo ""
    
    # GPU 0: vr05
    echo "🚀 在 GPU 0 上启动 vr05 (verification_rate=0.5)..."
    nohup bash experiments/scripts/run_param_scan_rq1.sh MNIST 0 verification_rate 0.5 vr05_scan \
        > experiments/logs/rq1_param_scan_$(date +%Y%m%d_%H%M%S)_vr05_scan.log 2>&1 &
    VR05_PID=$!
    echo "  PID: $VR05_PID"
    
    sleep 5
    
    # GPU 1: delta 扫描
    echo "🚀 在 GPU 1 上启动 delta 扫描..."
    nohup bash experiments/scripts/run_sequential_delta_scans.sh 1 \
        > experiments/logs/sequential_delta_scans_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    DELTA_PID=$!
    echo "  PID: $DELTA_PID"
    
else
    echo "📊 使用保守策略：顺序运行"
    echo ""
    
    # 只在 GPU 1 上顺序运行
    echo "🚀 在 GPU 1 上启动 vr05 (verification_rate=0.5)..."
    bash experiments/scripts/run_param_scan_rq1.sh MNIST 1 verification_rate 0.5 vr05_scan \
        2>&1 | tee experiments/logs/rq1_param_scan_$(date +%Y%m%d_%H%M%S)_vr05_scan.log
    
    echo "✅ vr05 完成！"
    echo ""
    
    echo "🚀 启动 delta 扫描..."
    bash experiments/scripts/run_sequential_delta_scans.sh 1 \
        2>&1 | tee experiments/logs/sequential_delta_scans_$(date +%Y%m%d_%H%M%S).log
    
    echo "✅ delta 扫描完成！"
fi

echo ""
echo "================================================================================"
echo "所有实验已启动"
echo "================================================================================"
echo ""
echo "监控命令："
echo "  nvidia-smi"
echo "  ps aux | grep run_rq1_security.py"
echo "  tail -f experiments/logs/rq1_param_scan_*_vr05_scan.log"
echo ""

