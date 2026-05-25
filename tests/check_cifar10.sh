#!/bin/bash
# 快速检查CIFAR-10实验状态（已迁移到 scripts/ 目录）

echo "========================================================================"
echo "CIFAR-10 Experiment Status Check"
echo "========================================================================"
echo ""

# 检查进程
echo "1. Process Status:"
if ps aux | grep run_cifar10_paper | grep -v grep > /dev/null; then
    echo "   ✅ Experiment is RUNNING"
    ps aux | grep run_cifar10_paper | grep -v grep | awk '{print "   PID: " $2 ", CPU: " $3 "% , MEM: " $4 "%"}'
else
    echo "   ⚠️  Experiment is NOT running (may have completed or crashed)"
fi
echo ""

# 检查日志
echo "2. Latest Log (last 10 lines):"
if [ -f cifar10_experiment.log ]; then
    tail -10 cifar10_experiment.log | sed 's/^/   /'
else
    echo "   ⚠️  Log file not found"
fi
echo ""

# 检查准确率
echo "3. Accuracy Progress:"
if [ -f cifar10_experiment.log ]; then
    grep "Accuracy" cifar10_experiment.log | tail -5 | sed 's/^/   /'
else
    echo "   ⚠️  No accuracy data yet"
fi
echo ""

# 检查结果文件
echo "4. Results:"
if [ -f experiments/results/cifar10_paper/results.json ]; then
    echo "   ✅ Results file exists!"
    echo "   Size: $(ls -lh experiments/results/cifar10_paper/results.json | awk '{print $5}')"
else
    echo "   ⚠️  Results not generated yet"
fi
echo ""

# GPU状态
echo "5. GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/   /'
else
    echo "   ⚠️  nvidia-smi not available"
fi
echo ""

echo "========================================================================"
echo "To monitor in real-time: tail -f cifar10_experiment.log"
echo "========================================================================"

