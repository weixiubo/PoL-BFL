#!/bin/bash
# RQ3 验证实验：MNIST + CIFAR10 各8轮对照
# 验证性能约束：PoL-FL ≤ 1.5× Vanilla, PoL-FL+ZKP ≤ 2× Vanilla

set -e

PYTHON_BIN="python"
CODE_DIR="PoL-BFL/Code"
LOG_DIR="${CODE_DIR}/experiments/smoke_logs"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "RQ3 验证实验开始"
echo "时间: $(date)"
echo "=========================================="

# MNIST 8轮
echo ""
echo "[1/2] 运行 MNIST 8轮对照实验..."
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
  ${PYTHON_BIN} -u ${CODE_DIR}/experiments/scripts/runners/run_rq3_overhead.py \
  --dataset MNIST --rounds 8 --num_clients 10 --clients_per_round 5 \
  2>&1 | tee ${LOG_DIR}/rq3_overhead_mnist_8r_validation.log

echo ""
echo "[PASS] MNIST 8轮完成"

# CIFAR10 8轮
echo ""
echo "[2/2] 运行 CIFAR10 8轮对照实验..."
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
  ${PYTHON_BIN} -u ${CODE_DIR}/experiments/scripts/runners/run_rq3_overhead.py \
  --dataset CIFAR10 --rounds 8 --num_clients 20 --clients_per_round 10 \
  2>&1 | tee ${LOG_DIR}/rq3_overhead_cifar10_8r_validation.log

echo ""
echo "[PASS] CIFAR10 8轮完成"

echo ""
echo "=========================================="
echo "RQ3 验证实验全部完成。"
echo "时间: $(date)"
echo "=========================================="
echo ""
echo "结果位置:"
echo "  - ${CODE_DIR}/experiments/results/rq3_overhead/"
echo ""
echo "日志位置:"
echo "  - ${LOG_DIR}/rq3_overhead_mnist_8r_validation.log"
echo "  - ${LOG_DIR}/rq3_overhead_cifar10_8r_validation.log"

