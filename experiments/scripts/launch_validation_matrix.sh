#!/bin/bash
POLBFL_ROOT="${POLBFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# MNIST validation matrix launcher
# 21个MNIST实验，并行使用GPU0和GPU1

set -e

# 基础配置
export NUM_WORKERS_OVERRIDE=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export POL_VERIFICATION_RATE=0.3
export POL_ALWAYS_VERIFY_LAST_K=2
export POL_RANDOM_Q=3
export POL_MIN_PAIR_SUCCESS_RATE=0.99
export POL_FINAL_DELTA_OVERRIDE=50

# 日志目录
LOG_DIR="/tmp/validation_queue"
mkdir -p "$LOG_DIR"

# Python解释器
PYTHON="python"

# 工作目录
cd $POLBFL_ROOT

echo "=========================================="
echo "MNIST validation matrix launch"
echo "时间: $(date)"
echo "=========================================="

# RQ1: 9个攻击场景
ATTACKS=(
    "no_attack"
    "byzantine_random_noise"
    "byzantine_label_flipping"
    "byzantine_model_replacement"
    "byzantine_gradient_inversion"
    "free_riding_no_training"
    "free_riding_lazy_training"
    "free_riding_minimal_update"
    "sybil_attack"
)

# GPU分配函数
get_gpu() {
    local idx=$1
    echo $((idx % 2))  # 0或1
}

# 启动RQ1实验
echo ""
echo "启动RQ1实验（9个攻击场景）..."
for i in "${!ATTACKS[@]}"; do
    attack="${ATTACKS[$i]}"
    gpu=$(get_gpu $i)
    log_file="$LOG_DIR/rq1_mnist_${attack}.log"

    echo "[$((i+1))/9] 启动: $attack (GPU $gpu)"

    CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON \
        experiments/scripts/runners/run_rq1_security.py \
        --dataset MNIST \
        --model SimpleCNN \
        --num_rounds 20 \
        --attacks "$attack" \
        --baselines PoL_FL \
        > "$log_file" 2>&1 &

    PID=$!
    echo "    PID: $PID, Log: $log_file"

    # 错开启动时间，避免同时初始化
    sleep 5
done

echo ""
echo "=========================================="
echo "MNIST validation matrix jobs started"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""
echo "监控命令:"
echo "  watch -n 10 'nvidia-smi'"
echo "  tail -f $LOG_DIR/rq1_mnist_*.log"
echo ""
echo "检查进度:"
echo "  grep -h 'Round' $LOG_DIR/*.log | tail -20"
echo ""
