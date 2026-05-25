#!/bin/bash

# 保护实验进程脚本
# 使用nohup将前台进程转移到后台，防止SSH断开时进程被杀死

set -e

cd /home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code:/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code/experiments/scripts/utils
export POL_DATA_DIR=/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code/data

LOG_DIR="experiments/logs/tuning_2025-11-19"
RESULT_DIR="experiments/results/tuning_2025-11-19"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"

echo "=== 保护实验进程 ==="
echo "当前时间: $(date)"
echo ""

# 检查进程是否还在运行
echo "=== 检查当前进程 ==="
ps aux | grep "run_rq" | grep -v grep | wc -l

echo ""
echo "=== 使用nohup重新启动实验（防止SSH断开时中断） ==="
echo ""

# 如果进程还在运行，先杀死它们
echo "杀死旧进程..."
pkill -f "run_rq1_security.py.*MNIST" || true
pkill -f "run_rq2_ablation.py.*MNIST" || true
sleep 2

# 使用nohup重新启动实验
echo "启动RQ1 MNIST (GPU 0)..."
nohup bash -c 'cd /home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code && \
export CUBLAS_WORKSPACE_CONFIG=:4096:8 && \
export PYTHONPATH=/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code:/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code/experiments/scripts/utils && \
export POL_DATA_DIR=/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code/data && \
CUDA_VISIBLE_DEVICES=0 python experiments/scripts/runners/run_rq1_security.py \
  --dataset MNIST --num_rounds 10 \
  --output_dir experiments/results/tuning_2025-11-19/rq1_mnist_tuning' > "$LOG_DIR/rq1_mnist_nohup.log" 2>&1 &

echo "启动RQ2 MNIST (GPU 1)..."
nohup bash -c 'cd /home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code && \
export CUBLAS_WORKSPACE_CONFIG=:4096:8 && \
export PYTHONPATH=/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code:/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code/experiments/scripts/utils && \
export POL_DATA_DIR=/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code/data && \
CUDA_VISIBLE_DEVICES=1 python experiments/scripts/runners/run_rq2_ablation.py \
  --dataset MNIST --num_rounds 10' > "$LOG_DIR/rq2_mnist_nohup.log" 2>&1 &

sleep 3

echo ""
echo "=== 新进程已启动 ==="
ps aux | grep "run_rq" | grep -v grep | wc -l
echo ""
echo "进程现在在后台运行，SSH断开不会中断"
echo "查看日志: tail -f $LOG_DIR/rq1_mnist_nohup.log"
echo "查看日志: tail -f $LOG_DIR/rq2_mnist_nohup.log"

