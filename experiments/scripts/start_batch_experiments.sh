#!/bin/bash
POLBFL_ROOT="${POLBFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# 并行实验启动脚本 - 充分利用双GPU
# 执行计划：
#   Batch A (estimated 0-2 hours):
#     GPU0: RQ1 CIFAR-10 (15轮)
#     GPU1: RQ2 CIFAR-10 (10轮)
#   Batch B (estimated 2-4 hours):
#     GPU0: RQ1 MNIST 补齐 (3种攻击)
#     GPU1: RQ2 MNIST 补齐 (pol_incentive rep2 + pol_zkp_incentive)

set -e

BASE_DIR="$POLBFL_ROOT"
LOG_DIR="$BASE_DIR/experiments/logs/parameter_evaluation"
RESULT_DIR="$BASE_DIR/experiments/results/parameter_evaluation"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$BASE_DIR:$BASE_DIR/experiments/scripts/utils"
export POL_DATA_DIR="$BASE_DIR/data"

echo "=========================================="
echo "PoL-BFL 并行实验启动"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Batch A: CIFAR-10 parallel execution
echo ""
echo "Batch A: starting CIFAR-10 experiments on GPU 0 and GPU 1"
echo "预期时间: 1.5-2 小时"
echo ""

# GPU0: RQ1 CIFAR-10
echo "GPU0: 启动 RQ1 CIFAR-10 (15轮)..."
nohup bash -c "
  cd $BASE_DIR
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONPATH=$BASE_DIR:$BASE_DIR/experiments/scripts/utils
  export POL_DATA_DIR=$BASE_DIR/data
  export CUDA_VISIBLE_DEVICES=0
  python experiments/scripts/runners/run_rq1_security.py \
    --dataset CIFAR10 --num_rounds 15 \
    --output_dir $RESULT_DIR/rq1_cifar10_tuning \
    2>&1 | tee $LOG_DIR/rq1_cifar10_tuning.log
" > /dev/null 2>&1 &
PID_RQ1_CIFAR10=$!
echo "  PID: $PID_RQ1_CIFAR10"

# GPU1: RQ2 CIFAR-10
echo "GPU1: 启动 RQ2 CIFAR-10 (10轮)..."
nohup bash -c "
  cd $BASE_DIR
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONPATH=$BASE_DIR:$BASE_DIR/experiments/scripts/utils
  export POL_DATA_DIR=$BASE_DIR/data
  export CUDA_VISIBLE_DEVICES=1
  python experiments/scripts/runners/run_rq2_ablation.py \
    --dataset CIFAR10 --num_rounds 10 \
    2>&1 | tee $LOG_DIR/rq2_cifar10_tuning.log
" > /dev/null 2>&1 &
PID_RQ2_CIFAR10=$!
echo "  PID: $PID_RQ2_CIFAR10"

# Wait for Batch A.
echo ""
echo "Waiting for Batch A to complete..."
wait $PID_RQ1_CIFAR10 $PID_RQ2_CIFAR10
echo "Batch A completed."

# Batch B: MNIST coverage in parallel
echo ""
echo "Batch B: starting MNIST coverage experiments on GPU 0 and GPU 1"
echo "预期时间: 2-3 小时"
echo ""

# GPU0: RQ1 MNIST 补齐 (3种攻击)
echo "GPU0: 启动 RQ1 MNIST 补齐..."
for attack in "byzantine_random_noise" "label_flipping" "no_training"; do
  echo "  - $attack..."
  nohup bash -c "
    cd $BASE_DIR
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export PYTHONPATH=$BASE_DIR:$BASE_DIR/experiments/scripts/utils
    export POL_DATA_DIR=$BASE_DIR/data
    export CUDA_VISIBLE_DEVICES=0
    python experiments/scripts/runners/run_rq1_security.py \
      --dataset MNIST --num_rounds 10 \
      --output_dir $RESULT_DIR/rq1_mnist_tuning \
      --attacks $attack \
      --baselines PoL_FL \
      2>&1 | tee $LOG_DIR/rq1_mnist_${attack}_tuning.log
  " > /dev/null 2>&1 &
  wait
done

# GPU1: RQ2 MNIST 补齐
echo "GPU1: 启动 RQ2 MNIST 补齐..."
echo "  - pol_incentive rep2..."
nohup bash -c "
  cd $BASE_DIR
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONPATH=$BASE_DIR:$BASE_DIR/experiments/scripts/utils
  export POL_DATA_DIR=$BASE_DIR/data
  export CUDA_VISIBLE_DEVICES=1
  python experiments/scripts/runners/run_rq2_ablation.py \
    --dataset MNIST --num_rounds 10 --num_repetitions 1 \
    --variants pol_incentive \
    2>&1 | tee $LOG_DIR/rq2_mnist_pol_incentive_rep2.log
" > /dev/null 2>&1 &
wait

echo "  - pol_zkp_incentive (3次重复)..."
nohup bash -c "
  cd $BASE_DIR
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONPATH=$BASE_DIR:$BASE_DIR/experiments/scripts/utils
  export POL_DATA_DIR=$BASE_DIR/data
  export CUDA_VISIBLE_DEVICES=1
  python experiments/scripts/runners/run_rq2_ablation.py \
    --dataset MNIST --num_rounds 10 --num_repetitions 3 \
    --variants pol_zkp_incentive \
    2>&1 | tee $LOG_DIR/rq2_mnist_pol_zkp_incentive.log
" > /dev/null 2>&1 &
wait

echo ""
echo "=========================================="
echo "所有实验完成。"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "结果位置："
echo "  - RQ1 CIFAR-10: $RESULT_DIR/rq1_cifar10_tuning/"
echo "  - RQ2 CIFAR-10: $RESULT_DIR/rq2_ablation/"
echo "  - RQ1 MNIST 补齐: $RESULT_DIR/rq1_mnist_tuning/"
echo "  - RQ2 MNIST 补齐: $RESULT_DIR/rq2_ablation/"
echo ""
echo "日志位置："
echo "  - $LOG_DIR/"
