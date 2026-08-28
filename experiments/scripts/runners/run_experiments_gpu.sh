#!/bin/bash

# PoL-BFL GPU并行实验运行脚本
# 将4个实验分配到2个GPU上，防止SSH断开中断

set -e

echo "======================================================================="
echo "PoL-BFL GPU并行实验启动脚本"
echo "======================================================================="
echo ""

# 检查conda环境
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "polbfl" ]]; then
    echo "Error: activate the polbfl environment or set PYTHON_BIN."
    echo "运行: conda activate polbfl"
    exit 1
fi

echo "[PASS] Conda环境: ${CONDA_DEFAULT_ENV}"
echo ""

# 检查GPU
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

# 创建日志目录
LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}
echo "[PASS] 日志目录: ${LOG_DIR}"
echo ""

# 创建结果目录
mkdir -p results/rq1_security
mkdir -p results/rq2_overhead
mkdir -p results/rq3_scalability
mkdir -p results/rq4_incentive
echo "[PASS] 结果目录已创建"
echo ""

# GPU任务分配策略：
# GPU 0: RQ1 (最耗时) -> RQ3 (中等耗时)
# GPU 1: RQ2 (最快) -> RQ4 (中等耗时)
# 这样可以让两个GPU尽可能同时完成

echo "======================================================================="
echo "任务分配方案:"
echo "======================================================================="
echo "GPU 0: RQ1 (安全性评估, ~30-60分钟) -> RQ3 (可扩展性, ~15-30分钟)"
echo "GPU 1: RQ2 (系统开销, ~5-10分钟) -> RQ4 (经济激励, ~10-20分钟)"
echo ""
echo "预计总时间: ~45-90分钟 (两个GPU并行)"
echo "======================================================================="
echo ""

# 函数: 在指定GPU上运行实验
run_on_gpu() {
    local gpu_id=$1
    local exp_name=$2
    local exp_script=$3
    local log_file="${LOG_DIR}/gpu${gpu_id}_${exp_name}.log"

    echo "[GPU ${gpu_id}] 启动 ${exp_name}..."
    echo "[GPU ${gpu_id}] 日志文件: ${log_file}"

    # 使用nohup在后台运行，指定GPU
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python ${exp_script} \
        > ${log_file} 2>&1 &

    local pid=$!
    echo "[GPU ${gpu_id}] ${exp_name} 已启动 (PID: ${pid})"
    echo ${pid} > ${LOG_DIR}/gpu${gpu_id}_${exp_name}.pid

    return ${pid}
}

# 函数: 等待进程完成
wait_for_process() {
    local pid=$1
    local name=$2
    local gpu_id=$3

    echo "[GPU ${gpu_id}] 等待 ${name} 完成 (PID: ${pid})..."

    while kill -0 ${pid} 2>/dev/null; do
        sleep 5
    done

    wait ${pid}
    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        echo "[GPU ${gpu_id}] [PASS] ${name} 完成"
    else
        echo "[GPU ${gpu_id}] [FAIL] ${name} 失败 (退出码: ${exit_code})"
    fi

    return ${exit_code}
}

# 启动时间戳
START_TIME=$(date +%s)
echo "开始时间: $(date)"
echo ""

# ========== GPU 0 任务队列 ==========
echo "======================================================================="
echo "启动 GPU 0 任务队列"
echo "======================================================================="

# GPU 0: RQ1
run_on_gpu 0 "RQ1" "run_rq1_security.py"
RQ1_PID=$!
sleep 2

# ========== GPU 1 任务队列 ==========
echo ""
echo "======================================================================="
echo "启动 GPU 1 任务队列"
echo "======================================================================="

# GPU 1: RQ2
run_on_gpu 1 "RQ2" "run_rq2_overhead.py"
RQ2_PID=$!
sleep 2

echo ""
echo "======================================================================="
echo "所有实验已启动，正在后台运行"
echo "======================================================================="
echo ""
echo "进程信息:"
echo "  GPU 0 - RQ1 (PID: ${RQ1_PID})"
echo "  GPU 1 - RQ2 (PID: ${RQ2_PID})"
echo ""
echo "日志目录: ${LOG_DIR}"
echo ""
echo "监控命令:"
echo "  查看所有日志: tail -f ${LOG_DIR}/*.log"
echo "  查看GPU 0: tail -f ${LOG_DIR}/gpu0_*.log"
echo "  查看GPU 1: tail -f ${LOG_DIR}/gpu1_*.log"
echo "  查看进程: ps aux | grep python | grep run_rq"
echo "  查看GPU状态: watch -n 1 nvidia-smi"
echo ""
echo "======================================================================="
echo ""

# 保存启动信息
cat > ${LOG_DIR}/experiment_info.txt <<EOF
实验启动信息
============

启动时间: $(date)
Conda环境: ${CONDA_DEFAULT_ENV}

GPU分配:
  GPU 0: RQ1 (PID: ${RQ1_PID}) -> RQ3
  GPU 1: RQ2 (PID: ${RQ2_PID}) -> RQ4

日志目录: ${LOG_DIR}

监控命令:
  tail -f ${LOG_DIR}/*.log
  ps aux | grep python | grep run_rq
  watch -n 1 nvidia-smi
EOF

echo "实验信息已保存到: ${LOG_DIR}/experiment_info.txt"
echo ""

# 等待RQ2完成（最快的）
echo "等待 RQ2 完成..."
wait_for_process ${RQ2_PID} "RQ2" 1
RQ2_EXIT=$?

# RQ2完成后，在GPU 1上启动RQ4
if [ ${RQ2_EXIT} -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "RQ2 完成，在 GPU 1 上启动 RQ4"
    echo "======================================================================="
    run_on_gpu 1 "RQ4" "run_rq4_incentive.py"
    RQ4_PID=$!
    echo "  GPU 1 - RQ4 (PID: ${RQ4_PID})"
    echo ""
fi

# 等待RQ1完成
echo "等待 RQ1 完成..."
wait_for_process ${RQ1_PID} "RQ1" 0
RQ1_EXIT=$?

# RQ1完成后，在GPU 0上启动RQ3
if [ ${RQ1_EXIT} -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "RQ1 完成，在 GPU 0 上启动 RQ3"
    echo "======================================================================="
    run_on_gpu 0 "RQ3" "run_rq3_scalability.py"
    RQ3_PID=$!
    echo "  GPU 0 - RQ3 (PID: ${RQ3_PID})"
    echo ""
fi

# 等待所有实验完成
echo ""
echo "======================================================================="
echo "等待所有实验完成..."
echo "======================================================================="

if [ ! -z "${RQ3_PID}" ]; then
    wait_for_process ${RQ3_PID} "RQ3" 0
    RQ3_EXIT=$?
fi

if [ ! -z "${RQ4_PID}" ]; then
    wait_for_process ${RQ4_PID} "RQ4" 1
    RQ4_EXIT=$?
fi

# 计算总时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

# 生成总结报告
echo ""
echo "======================================================================="
echo "所有实验完成。"
echo "======================================================================="
echo ""
echo "完成时间: $(date)"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""
echo "实验结果:"
echo "  RQ1: $([ ${RQ1_EXIT:-1} -eq 0 ] && echo '[PASS] 成功' || echo '[FAIL] 失败')"
echo "  RQ2: $([ ${RQ2_EXIT:-1} -eq 0 ] && echo '[PASS] 成功' || echo '[FAIL] 失败')"
echo "  RQ3: $([ ${RQ3_EXIT:-1} -eq 0 ] && echo '[PASS] 成功' || echo '[FAIL] 失败')"
echo "  RQ4: $([ ${RQ4_EXIT:-1} -eq 0 ] && echo '[PASS] 成功' || echo '[FAIL] 失败')"
echo ""
echo "结果文件:"
echo "  - results/rq1_security/rq1_results.json"
echo "  - results/rq2_overhead/rq2_results.json"
echo "  - results/rq3_scalability/rq3_results.json"
echo "  - results/rq4_incentive/rq4_results.json"
echo ""
echo "日志文件: ${LOG_DIR}/"
echo "======================================================================="
echo ""

# 保存总结报告
cat > ${LOG_DIR}/summary.txt <<EOF
实验总结报告
============

完成时间: $(date)
总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒

实验结果:
  RQ1 (安全性评估): $([ ${RQ1_EXIT:-1} -eq 0 ] && echo '成功' || echo '失败')
  RQ2 (系统开销): $([ ${RQ2_EXIT:-1} -eq 0 ] && echo '成功' || echo '失败')
  RQ3 (可扩展性): $([ ${RQ3_EXIT:-1} -eq 0 ] && echo '成功' || echo '失败')
  RQ4 (经济激励): $([ ${RQ4_EXIT:-1} -eq 0 ] && echo '成功' || echo '失败')

结果文件:
  - results/rq1_security/rq1_results.json
  - results/rq2_overhead/rq2_results.json
  - results/rq3_scalability/rq3_results.json
  - results/rq4_incentive/rq4_results.json

日志文件: ${LOG_DIR}/
EOF

echo "总结报告已保存到: ${LOG_DIR}/summary.txt"
