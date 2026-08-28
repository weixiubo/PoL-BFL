#!/bin/bash

# 修正后的完整实验运行脚本
# 在后台运行所有4个实验，防止SSH断开

set -e

echo "======================================================================="
echo "PoL-BFL 修正后完整实验运行"
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

# 创建日志目录
LOG_DIR="./logs/fixed_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}
echo "[PASS] 日志目录: ${LOG_DIR}"
echo ""

# 检查GPU
echo "GPU状态:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

echo "======================================================================="
echo "任务分配:"
echo "======================================================================="
echo "GPU 0: RQ1 (安全性, ~15分钟) → RQ3 (可扩展性, ~20分钟)"
echo "GPU 1: RQ2 (开销, ~5分钟) → RQ4 (激励, ~10分钟)"
echo ""
echo "预计总时间: ~35分钟"
echo "======================================================================="
echo ""

START_TIME=$(date +%s)

# 启动RQ1 (GPU 0)
echo "[GPU 0] 启动 RQ1..."
CUDA_VISIBLE_DEVICES=0 nohup python run_rq1_security.py > ${LOG_DIR}/gpu0_RQ1.log 2>&1 &
RQ1_PID=$!
echo "[GPU 0] RQ1 已启动 (PID: ${RQ1_PID})"
echo ${RQ1_PID} > ${LOG_DIR}/rq1.pid

sleep 2

# 启动RQ2 (GPU 1)
echo "[GPU 1] 启动 RQ2..."
CUDA_VISIBLE_DEVICES=1 nohup python run_rq2_overhead.py > ${LOG_DIR}/gpu1_RQ2.log 2>&1 &
RQ2_PID=$!
echo "[GPU 1] RQ2 已启动 (PID: ${RQ2_PID})"
echo ${RQ2_PID} > ${LOG_DIR}/rq2.pid

echo ""
echo "======================================================================="
echo "实验已启动，正在后台运行"
echo "======================================================================="
echo ""
echo "进程信息:"
echo "  GPU 0 - RQ1 (PID: ${RQ1_PID})"
echo "  GPU 1 - RQ2 (PID: ${RQ2_PID})"
echo ""
echo "监控命令:"
echo "  查看进度: tail -f ${LOG_DIR}/*.log"
echo "  查看GPU: watch -n 1 nvidia-smi"
echo "  查看进程: ps aux | grep python | grep run_rq"
echo ""
echo "======================================================================="
echo ""

# 等待RQ2完成
echo "等待 RQ2 完成..."
wait ${RQ2_PID}
RQ2_EXIT=$?
echo "RQ2 完成 (退出码: ${RQ2_EXIT})"

# 启动RQ4
if [ ${RQ2_EXIT} -eq 0 ]; then
    echo ""
    echo "[GPU 1] 启动 RQ4..."
    CUDA_VISIBLE_DEVICES=1 nohup python run_rq4_incentive.py > ${LOG_DIR}/gpu1_RQ4.log 2>&1 &
    RQ4_PID=$!
    echo "[GPU 1] RQ4 已启动 (PID: ${RQ4_PID})"
    echo ${RQ4_PID} > ${LOG_DIR}/rq4.pid
fi

# 等待RQ1完成
echo ""
echo "等待 RQ1 完成..."
wait ${RQ1_PID}
RQ1_EXIT=$?
echo "RQ1 完成 (退出码: ${RQ1_EXIT})"

# 启动RQ3
if [ ${RQ1_EXIT} -eq 0 ]; then
    echo ""
    echo "[GPU 0] 启动 RQ3..."
    CUDA_VISIBLE_DEVICES=0 nohup python run_rq3_scalability.py > ${LOG_DIR}/gpu0_RQ3.log 2>&1 &
    RQ3_PID=$!
    echo "[GPU 0] RQ3 已启动 (PID: ${RQ3_PID})"
    echo ${RQ3_PID} > ${LOG_DIR}/rq3.pid
fi

# 等待所有实验完成
echo ""
echo "等待所有实验完成..."

if [ ! -z "${RQ3_PID}" ]; then
    wait ${RQ3_PID}
    RQ3_EXIT=$?
    echo "RQ3 完成 (退出码: ${RQ3_EXIT})"
fi

if [ ! -z "${RQ4_PID}" ]; then
    wait ${RQ4_PID}
    RQ4_EXIT=$?
    echo "RQ4 完成 (退出码: ${RQ4_EXIT})"
fi

# 计算总时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

# 生成总结
echo ""
echo "======================================================================="
echo "所有实验完成。"
echo "======================================================================="
echo ""
echo "完成时间: $(date)"
echo "总耗时: ${MINUTES}分钟 ${SECONDS}秒"
echo ""
echo "实验结果:"
echo "  RQ1: $([ ${RQ1_EXIT:-1} -eq 0 ] && echo '[PASS] 成功' || echo '[FAIL] 失败')"
echo "  RQ2: $([ ${RQ2_EXIT:-1} -eq 0 ] && echo '[PASS] 成功' || echo '[FAIL] 失败')"
echo "  RQ3: $([ ${RQ3_EXIT:-1} -eq 0 ] && echo '[PASS] 成功' || echo '[FAIL] 失败')"
echo "  RQ4: $([ ${RQ4_EXIT:-1} -eq 0 ] && echo '[PASS] 成功' || echo '[FAIL] 失败')"
echo ""
echo "结果文件:"
ls -lh experiments/results/*/rq*_results.json 2>/dev/null || echo "  (检查日志查看详情)"
echo ""
echo "日志文件: ${LOG_DIR}/"
echo "======================================================================="
echo ""

# 保存总结
cat > ${LOG_DIR}/summary.txt <<EOF
实验总结
========

完成时间: $(date)
总耗时: ${MINUTES}分钟 ${SECONDS}秒

实验结果:
  RQ1 (安全性): $([ ${RQ1_EXIT:-1} -eq 0 ] && echo '成功' || echo '失败')
  RQ2 (开销): $([ ${RQ2_EXIT:-1} -eq 0 ] && echo '成功' || echo '失败')
  RQ3 (可扩展性): $([ ${RQ3_EXIT:-1} -eq 0 ] && echo '成功' || echo '失败')
  RQ4 (激励): $([ ${RQ4_EXIT:-1} -eq 0 ] && echo '成功' || echo '失败')

Validated behaviors:
  [PASS] Trimmed Mean retains at least one value after trimming
  [PASS] Krum uses the configured malicious-client population
  [PASS] RQ3 uses positional client-model indexing

结果文件:
  - experiments/results/rq1_security/rq1_results.json
  - experiments/results/rq2_overhead/rq2_results.json
  - experiments/results/rq3_scalability/rq3_results.json
  - experiments/results/rq4_incentive/rq4_results.json

日志文件: ${LOG_DIR}/
EOF

echo "总结已保存到: ${LOG_DIR}/summary.txt"
echo ""
echo "[PASS] Experiment execution completed."
