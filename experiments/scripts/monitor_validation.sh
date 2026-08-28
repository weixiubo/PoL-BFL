#!/bin/bash
POLBFL_ROOT="${POLBFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 监控验证实验进度

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PROJECT_ROOT="$POLBFL_ROOT"
OUTPUT_BASE="$PROJECT_ROOT/experiments/results/validation"
PID_DIR="$OUTPUT_BASE/pids"
LOG_DIR="$OUTPUT_BASE/logs"

clear
echo -e "${BLUE}=========================================="
echo "PoL-BFL 验证实验监控"
echo -e "==========================================${NC}"
echo ""

# 检查状态文件
if [ -f "$OUTPUT_BASE/validation_status.txt" ]; then
    echo -e "${CYAN}【启动信息】${NC}"
    cat "$OUTPUT_BASE/validation_status.txt"
    echo ""
fi

# 检查进程状态
echo -e "${CYAN}【进程状态】${NC}"
if [ -d "$PID_DIR" ]; then
    for pidfile in "$PID_DIR"/*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            name=$(basename "$pidfile" .pid)
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${GREEN}[PASS] $name (PID: $pid) - 运行中${NC}"
            else
                echo -e "${RED}[FAIL] $name (PID: $pid) - 已停止${NC}"
            fi
        fi
    done
else
    echo -e "${YELLOW}[WARNING]  未找到PID目录${NC}"
fi
echo ""

# GPU状态
echo -e "${CYAN}【GPU状态】${NC}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{printf "GPU %s: %s | 使用率: %s | 内存: %s / %s\n", $1, $2, $3, $4, $5}'
echo ""

# 检查结果文件
echo -e "${CYAN}【实验进度】${NC}"

check_results() {
    local dir=$1
    local name=$2

    if [ -d "$dir" ]; then
        local csv_count=$(find "$dir" -name "*.csv" -type f 2>/dev/null | wc -l)
        local json_count=$(find "$dir" -name "*.json" -type f 2>/dev/null | wc -l)
        local latest_csv=$(find "$dir" -name "*.csv" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

        if [ $csv_count -gt 0 ]; then
            echo -e "${GREEN}[PASS] $name: $csv_count CSV文件, $json_count JSON文件${NC}"
            if [ -n "$latest_csv" ]; then
                local last_modified=$(stat -c %y "$latest_csv" | cut -d'.' -f1)
                echo -e "   最新文件: $(basename "$latest_csv")"
                echo -e "   更新时间: $last_modified"

                # 显示最后几行
                if [ -f "$latest_csv" ]; then
                    local lines=$(wc -l < "$latest_csv")
                    if [ $lines -gt 1 ]; then
                        echo -e "   进度: $lines 行数据"
                        echo -e "   最新数据:"
                        tail -3 "$latest_csv" | head -2 | while read line; do
                            echo -e "     $line"
                        done
                    fi
                fi
            fi
        else
            echo -e "${YELLOW}[WAITING] $name: 等待结果...${NC}"
        fi
    else
        echo -e "${YELLOW}[WAITING] $name: 目录不存在${NC}"
    fi
    echo ""
}

# RQ1 MNIST
check_results "$OUTPUT_BASE/rq1_mnist_smoke_gpu0" "RQ1 MNIST GPU0 (Vanilla_FL, Krum, Trimmed_Mean, Median)"
check_results "$OUTPUT_BASE/rq1_mnist_smoke_gpu1" "RQ1 MNIST GPU1 (ShapleyFL, FoolsGold, PoL_FL)"

# 日志尾部
echo -e "${CYAN}【最新日志】${NC}"
if [ -f "$LOG_DIR/rq1_mnist_gpu0.log" ]; then
    echo -e "${YELLOW}GPU 0 最新日志:${NC}"
    tail -5 "$LOG_DIR/rq1_mnist_gpu0.log" | sed 's/^/  /'
    echo ""
fi

if [ -f "$LOG_DIR/rq1_mnist_gpu1.log" ]; then
    echo -e "${YELLOW}GPU 1 最新日志:${NC}"
    tail -5 "$LOG_DIR/rq1_mnist_gpu1.log" | sed 's/^/  /'
    echo ""
fi

# 错误检查
echo -e "${CYAN}【错误检查】${NC}"
error_count=0
if [ -f "$LOG_DIR/rq1_mnist_gpu0.log" ]; then
    errors=$(grep -i "error\|exception\|traceback" "$LOG_DIR/rq1_mnist_gpu0.log" | tail -3)
    if [ -n "$errors" ]; then
        echo -e "${RED}GPU 0 发现错误:${NC}"
        echo "$errors" | sed 's/^/  /'
        error_count=$((error_count + 1))
    fi
fi

if [ -f "$LOG_DIR/rq1_mnist_gpu1.log" ]; then
    errors=$(grep -i "error\|exception\|traceback" "$LOG_DIR/rq1_mnist_gpu1.log" | tail -3)
    if [ -n "$errors" ]; then
        echo -e "${RED}GPU 1 发现错误:${NC}"
        echo "$errors" | sed 's/^/  /'
        error_count=$((error_count + 1))
    fi
fi

if [ $error_count -eq 0 ]; then
    echo -e "${GREEN}[PASS] 未发现错误${NC}"
fi
echo ""

echo -e "${BLUE}=========================================="
echo "刷新时间: $(date +'%Y-%m-%d %H:%M:%S')"
echo -e "==========================================${NC}"
echo ""
echo "Refresh command: watch -n 60 bash experiments/scripts/monitor_validation.sh"
