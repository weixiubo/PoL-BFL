#!/bin/bash

# Phase 1 队列执行脚本
# GPU0和GPU1各自串行执行队列，互不干扰

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
LOG_DIR="/tmp/phase1_clearance"
mkdir -p "$LOG_DIR"

# Python解释器
PYTHON="/home/wxb/miniconda3/envs/wxb__veryfl_pol/bin/python"

# 工作目录
cd /home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code

# 主日志
MAIN_LOG="$LOG_DIR/phase1_runner.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log "=========================================="
log "Phase 1 队列执行脚本启动"
log "=========================================="

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

# 将攻击分配到两个GPU队列
GPU0_QUEUE=()
GPU1_QUEUE=()

for i in "${!ATTACKS[@]}"; do
    if [ $((i % 2)) -eq 0 ]; then
        GPU0_QUEUE+=("${ATTACKS[$i]}")
    else
        GPU1_QUEUE+=("${ATTACKS[$i]}")
    fi
done

log "GPU0 队列 (${#GPU0_QUEUE[@]}个): ${GPU0_QUEUE[*]}"
log "GPU1 队列 (${#GPU1_QUEUE[@]}个): ${GPU1_QUEUE[*]}"
log ""

# GPU0队列执行函数
run_gpu0_queue() {
    local gpu=0
    log "[GPU$gpu] 开始执行队列"
    
    for attack in "${GPU0_QUEUE[@]}"; do
        local exp_log="$LOG_DIR/rq1_mnist_${attack}.log"
        log "[GPU$gpu] 开始: $attack"
        
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON \
            experiments/scripts/runners/run_rq1_security.py \
            --dataset MNIST \
            --model SimpleCNN \
            --num_rounds 20 \
            --attacks "$attack" \
            --baselines PoL_FL \
            > "$exp_log" 2>&1
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log "[GPU$gpu] ✅ 完成: $attack"
        else
            log "[GPU$gpu] ❌ 失败: $attack (exit code: $exit_code)"
        fi
    done
    
    log "[GPU$gpu] 队列执行完毕"
}

# GPU1队列执行函数
run_gpu1_queue() {
    local gpu=1
    log "[GPU$gpu] 开始执行队列"
    
    for attack in "${GPU1_QUEUE[@]}"; do
        local exp_log="$LOG_DIR/rq1_mnist_${attack}.log"
        log "[GPU$gpu] 开始: $attack"
        
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON \
            experiments/scripts/runners/run_rq1_security.py \
            --dataset MNIST \
            --model SimpleCNN \
            --num_rounds 20 \
            --attacks "$attack" \
            --baselines PoL_FL \
            > "$exp_log" 2>&1
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log "[GPU$gpu] ✅ 完成: $attack"
        else
            log "[GPU$gpu] ❌ 失败: $attack (exit code: $exit_code)"
        fi
    done
    
    log "[GPU$gpu] 队列执行完毕"
}

# 并行执行两个GPU队列
log "启动GPU0和GPU1并行队列..."
run_gpu0_queue &
PID_GPU0=$!
run_gpu1_queue &
PID_GPU1=$!

log "GPU0 队列 PID: $PID_GPU0"
log "GPU1 队列 PID: $PID_GPU1"
log ""
log "监控命令:"
log "  tail -f $MAIN_LOG"
log "  watch -n 10 'nvidia-smi'"
log ""

# 等待两个队列都完成
wait $PID_GPU0
EXIT0=$?
wait $PID_GPU1
EXIT1=$?

log ""
log "=========================================="
log "Phase 1 队列执行完成"
log "GPU0 退出码: $EXIT0"
log "GPU1 退出码: $EXIT1"
log "=========================================="

# 生成总结报告
log ""
log "实验结果总结:"
for attack in "${ATTACKS[@]}"; do
    exp_log="$LOG_DIR/rq1_mnist_${attack}.log"
    if grep -q "Experiment completed successfully" "$exp_log" 2>/dev/null; then
        log "  ✅ $attack"
    elif [ -f "$exp_log" ]; then
        log "  ❌ $attack (检查日志: $exp_log)"
    else
        log "  ⚠️  $attack (日志不存在)"
    fi
done

log ""
log "详细日志目录: $LOG_DIR"

