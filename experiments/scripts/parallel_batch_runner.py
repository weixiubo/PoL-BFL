#!/usr/bin/env python3
"""
并行实验执行器 - 充分利用双GPU
执行计划：
  Batch A (estimated 0-2 hours):
    GPU0: RQ1 CIFAR-10 (15轮)
    GPU1: RQ2 CIFAR-10 (10轮)

  Batch B (estimated 2-4 hours):
    GPU0: RQ1 MNIST 补齐 (3种攻击)
    GPU1: RQ2 MNIST 补齐 (pol_incentive rep2 + pol_zkp_incentive)
"""

import subprocess
import time
import os
from pathlib import Path
from datetime import datetime

# 基础配置
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "experiments" / "scripts" / "runners"
LOG_DIR = BASE_DIR / "experiments" / "logs" / "parameter_evaluation"
RESULT_DIR = BASE_DIR / "experiments" / "results" / "parameter_evaluation"

# 确保目录存在
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def run_command(cmd, gpu_id, task_name, log_file):
    """运行命令并记录日志"""
    print(f"\n{'='*80}")
    print(f"[GPU{gpu_id}] 启动任务: {task_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志: {log_file}")
    print(f"{'='*80}\n")

    env = os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['PYTHONPATH'] = f"{BASE_DIR}:{BASE_DIR}/experiments/scripts/utils"
    env['POL_DATA_DIR'] = f"{BASE_DIR}/data"
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with open(log_file, 'w') as f:
        result = subprocess.run(
            cmd,
            shell=True,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR)
        )

    status = "[PASS] 完成" if result.returncode == 0 else "[FAIL] 失败"
    print(f"[GPU{gpu_id}] {task_name} {status}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return result.returncode == 0

def main():
    print("\n" + "="*80)
    print("PoL-BFL 并行实验执行器")
    print("="*80)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU数量: 2")
    print(f"预期完成时间: 6-8 小时后")
    print("="*80 + "\n")

    # Batch A: CIFAR-10
    print("\nBatch A: CIFAR-10 (estimated 0-2 hours)")
    print("-" * 80)

    # GPU0: RQ1 CIFAR-10
    cmd_rq1_cifar10 = (
        f"python {SCRIPTS_DIR}/run_rq1_security.py "
        f"--dataset CIFAR10 --num_rounds 15 "
        f"--output_dir {RESULT_DIR}/rq1_cifar10_tuning"
    )
    log_rq1_cifar10 = LOG_DIR / "rq1_cifar10_tuning.log"

    # GPU1: RQ2 CIFAR-10
    cmd_rq2_cifar10 = (
        f"python {SCRIPTS_DIR}/run_rq2_ablation.py "
        f"--dataset CIFAR10 --num_rounds 10"
    )
    log_rq2_cifar10 = LOG_DIR / "rq2_cifar10_tuning.log"

    # 并行启动
    import threading

    t0 = threading.Thread(
        target=run_command,
        args=(cmd_rq1_cifar10, 0, "RQ1 CIFAR-10 (15轮)", log_rq1_cifar10)
    )
    t1 = threading.Thread(
        target=run_command,
        args=(cmd_rq2_cifar10, 1, "RQ2 CIFAR-10 (10轮)", log_rq2_cifar10)
    )

    t0.start()
    t1.start()
    t0.join()
    t1.join()

    print("\nBatch A completed")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Batch B: MNIST coverage
    print("\nBatch B: MNIST coverage (estimated 2-4 hours)")
    print("-" * 80)

    # GPU0: RQ1 MNIST 补齐 (3种攻击)
    attacks = ["byzantine_random_noise", "label_flipping", "no_training"]
    for attack in attacks:
        cmd = (
            f"python {SCRIPTS_DIR}/run_rq1_security.py "
            f"--dataset MNIST --num_rounds 10 "
            f"--output_dir {RESULT_DIR}/rq1_mnist_tuning "
            f"--attacks {attack} "
            f"--baselines PoL_FL"
        )
        log_file = LOG_DIR / f"rq1_mnist_{attack}_tuning.log"
        run_command(cmd, 0, f"RQ1 MNIST - {attack} (PoL_FL)", log_file)

    # GPU1: RQ2 MNIST 补齐
    # pol_incentive rep2
    cmd_pol_incentive_rep2 = (
        f"python {SCRIPTS_DIR}/run_rq2_ablation.py "
        f"--dataset MNIST --num_rounds 10 --num_repetitions 1 "
        f"--variants pol_incentive"
    )
    log_pol_incentive_rep2 = LOG_DIR / "rq2_mnist_pol_incentive_rep2.log"
    run_command(cmd_pol_incentive_rep2, 1, "RQ2 MNIST - pol_incentive rep2", log_pol_incentive_rep2)

    # pol_zkp_incentive 完整
    cmd_pol_zkp_incentive = (
        f"python {SCRIPTS_DIR}/run_rq2_ablation.py "
        f"--dataset MNIST --num_rounds 10 --num_repetitions 3 "
        f"--variants pol_zkp_incentive"
    )
    log_pol_zkp_incentive = LOG_DIR / "rq2_mnist_pol_zkp_incentive.log"
    run_command(cmd_pol_zkp_incentive, 1, "RQ2 MNIST - pol_zkp_incentive (3次重复)", log_pol_zkp_incentive)

    print("\n【所有实验完成】")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n明天起来可以查看结果：")
    print(f"  - RQ1 CIFAR-10: {RESULT_DIR}/rq1_cifar10_tuning/")
    print(f"  - RQ2 CIFAR-10: {RESULT_DIR}/rq2_ablation/")
    print(f"  - RQ1 MNIST 补齐: {RESULT_DIR}/rq1_mnist_tuning/")
    print(f"  - RQ2 MNIST 补齐: {RESULT_DIR}/rq2_ablation/")

if __name__ == "__main__":
    main()
