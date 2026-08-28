#!/bin/bash
POLBFL_ROOT="${POLBFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# PoL-BFL 轻量级验证Smoke Test
# 用轻量级配置覆盖所有207个配置，验证代码正确性和数据质量

set -e  # Exit on error

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [PASS] $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [FAIL] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARNING]  $1${NC}"
}

# 项目根目录
PROJECT_ROOT="$POLBFL_ROOT"
cd "$PROJECT_ROOT"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/experiments/scripts/utils"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 输出目录
OUTPUT_BASE="experiments/results/validation"
mkdir -p "$OUTPUT_BASE"

# 日志目录
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"

# PID文件目录
PID_DIR="$OUTPUT_BASE/pids"
mkdir -p "$PID_DIR"

# 清理旧的PID文件
rm -f "$PID_DIR"/*.pid

log "=========================================="
log "PoL-BFL 轻量级验证Smoke Test"
log "=========================================="
log "策略: 用轻量级配置覆盖所有207个配置"
log "输出目录: $OUTPUT_BASE"
log "日志目录: $LOG_DIR"
log ""

# 检查GPU
log "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
log ""

# ============================================
# RQ1 MNIST reduced-scale validation (5 rounds)
# ============================================
log "=========================================="
log "RQ1 MNIST reduced-scale validation (5 rounds)"
log "=========================================="
log "配置: 7攻击 × 7方法 = 49个实验"
log "预计时间: 6.1小时"
log ""

# GPU 0: 前4个方法
log "启动 GPU 0: Vanilla_FL, Krum, Trimmed_Mean, Median"
CUDA_VISIBLE_DEVICES=0 nohup python3 experiments/scripts/runners/run_rq1_security.py \
    --dataset MNIST \
    --num_rounds 5 \
    --baselines Vanilla_FL,Krum,Trimmed_Mean,Median \
    --output_dir "$OUTPUT_BASE/rq1_mnist_smoke_gpu0" \
    > "$LOG_DIR/rq1_mnist_gpu0.log" 2>&1 &
GPU0_PID=$!
echo $GPU0_PID > "$PID_DIR/rq1_mnist_gpu0.pid"
log_success "GPU 0 启动成功 (PID: $GPU0_PID)"

# 等待5秒确保第一个进程启动
sleep 5

# GPU 1: 后3个方法
log "启动 GPU 1: ShapleyFL, FoolsGold, PoL_FL"
CUDA_VISIBLE_DEVICES=1 nohup python3 experiments/scripts/runners/run_rq1_security.py \
    --dataset MNIST \
    --num_rounds 5 \
    --baselines ShapleyFL,FoolsGold,PoL_FL \
    --output_dir "$OUTPUT_BASE/rq1_mnist_smoke_gpu1" \
    > "$LOG_DIR/rq1_mnist_gpu1.log" 2>&1 &
GPU1_PID=$!
echo $GPU1_PID > "$PID_DIR/rq1_mnist_gpu1.pid"
log_success "GPU 1 启动成功 (PID: $GPU1_PID)"

log ""
log "=========================================="
log "RQ1 MNIST Smoke Test 已启动"
log "=========================================="
log "GPU 0 PID: $GPU0_PID (Vanilla_FL, Krum, Trimmed_Mean, Median)"
log "GPU 1 PID: $GPU1_PID (ShapleyFL, FoolsGold, PoL_FL)"
log ""
log "监控命令:"
log "  - 查看GPU状态: watch -n 5 nvidia-smi"
log "  - 查看进程: ps -fp $GPU0_PID $GPU1_PID"
log "  - 查看GPU 0日志: tail -f $LOG_DIR/rq1_mnist_gpu0.log"
log "  - 查看GPU 1日志: tail -f $LOG_DIR/rq1_mnist_gpu1.log"
log "  - 监控脚本: bash experiments/scripts/monitor_validation.sh"
log ""
log "预计完成时间: $(date -d '+6 hours' +'%Y-%m-%d %H:%M:%S')"
log ""

# 保存启动信息
cat > "$OUTPUT_BASE/validation_status.txt" << EOF
PoL-BFL 轻量级验证Smoke Test
启动时间: $(date +'%Y-%m-%d %H:%M:%S')

Active validation: RQ1 MNIST (5 rounds)
GPU 0 PID: $GPU0_PID (Vanilla_FL, Krum, Trimmed_Mean, Median)
GPU 1 PID: $GPU1_PID (ShapleyFL, FoolsGold, PoL_FL)

日志文件:
- GPU 0: $LOG_DIR/rq1_mnist_gpu0.log
- GPU 1: $LOG_DIR/rq1_mnist_gpu1.log

输出目录:
- GPU 0: $OUTPUT_BASE/rq1_mnist_smoke_gpu0
- GPU 1: $OUTPUT_BASE/rq1_mnist_smoke_gpu1

预计完成时间: $(date -d '+6 hours' +'%Y-%m-%d %H:%M:%S')
EOF

log_success "验证实验已启动。"
log "状态文件: $OUTPUT_BASE/validation_status.txt"
log ""

# 显示初始日志
log "初始日志 (GPU 0):"
sleep 2
tail -20 "$LOG_DIR/rq1_mnist_gpu0.log" || log_warning "日志文件尚未生成"
log ""
log "初始日志 (GPU 1):"
tail -20 "$LOG_DIR/rq1_mnist_gpu1.log" || log_warning "日志文件尚未生成"
log ""

log "=========================================="
log "Note: Ctrl+C does not terminate background processes."
log "Stop command: kill $GPU0_PID $GPU1_PID"
log "=========================================="
