#!/usr/bin/env python3
"""
实时监控RQ1批次2实验进度
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

def check_experiment_progress(base_dir, attack_name):
    """检查单个攻击实验的进度"""
    csv_file = base_dir / f"rq1_rounds_MNIST_{attack_name}_PoL_FL.csv"

    if not csv_file.exists():
        return None

    # 读取CSV文件的行数（减去header）
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if len(lines) <= 1:
            return 0

        # 解析最后一行获取详细信息
        last_line = lines[-1].strip()
        if last_line:
            parts = last_line.split(',')
            round_num = int(parts[0])
            accuracy = float(parts[1])
            tpr = float(parts[3]) if len(parts) > 3 else 0.0
            return {
                'round': round_num,
                'accuracy': accuracy,
                'tpr': tpr,
                'total_rounds': 20
            }
    return 0

def format_progress_bar(current, total, width=30):
    """生成进度条"""
    if current is None:
        return f"[{'?' * width}] 未启动"

    if isinstance(current, dict):
        current_val = current['round']
    else:
        current_val = current

    filled = int(width * current_val / total)
    bar = '█' * filled + '░' * (width - filled)
    percentage = (current_val / total) * 100

    if isinstance(current, dict):
        return f"[{bar}] {current_val}/{total} ({percentage:.0f}%) - Acc:{current['accuracy']:.4f} TPR:{current['tpr']:.2f}"
    else:
        return f"[{bar}] {current_val}/{total} ({percentage:.0f}%)"

def main():
    gpu0_dir = Path("experiments/results/rq1_mnist_polfl_20r_gpu0_batch2")
    gpu1_dir = Path("experiments/results/rq1_mnist_polfl_20r_gpu1_batch2")

    gpu0_attacks = [
        "byzantine_model_replacement",
        "byzantine_gradient_inversion",
        "byzantine_label_flipping",
        "byzantine_alie"
    ]

    gpu1_attacks = [
        "byzantine_minmax",
        "free_riding_no_training",
        "free_riding_lazy_training",
        "free_riding_minimal_update"
    ]

    print("\n" + "=" * 100)
    print("RQ1 批次2 实验进度监控")
    print("=" * 100)
    print(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # GPU 0进度
    print("[GPU 0] GPU 0 进度:")
    print("-" * 100)
    gpu0_completed = 0
    gpu0_total_rounds = 0
    for attack in gpu0_attacks:
        progress = check_experiment_progress(gpu0_dir, attack)
        if progress and isinstance(progress, dict):
            gpu0_total_rounds += progress['round']
            if progress['round'] >= 20:
                gpu0_completed += 1

        bar = format_progress_bar(progress, 20)
        status = "[PASS]" if (progress and isinstance(progress, dict) and progress['round'] >= 20) else "[RUNNING]"
        print(f"  {status} {attack:35s} {bar}")

    print()

    # GPU 1进度
    print("[GPU 1] GPU 1 进度:")
    print("-" * 100)
    gpu1_completed = 0
    gpu1_total_rounds = 0
    for attack in gpu1_attacks:
        progress = check_experiment_progress(gpu1_dir, attack)
        if progress and isinstance(progress, dict):
            gpu1_total_rounds += progress['round']
            if progress['round'] >= 20:
                gpu1_completed += 1

        bar = format_progress_bar(progress, 20)
        status = "[PASS]" if (progress and isinstance(progress, dict) and progress['round'] >= 20) else "[RUNNING]"
        print(f"  {status} {attack:35s} {bar}")

    print()
    print("=" * 100)

    # 总体统计
    total_attacks = len(gpu0_attacks) + len(gpu1_attacks)
    total_completed = gpu0_completed + gpu1_completed
    total_rounds_done = gpu0_total_rounds + gpu1_total_rounds
    total_rounds_needed = total_attacks * 20

    print(f"[RESULT] 总体进度:")
    print(f"  - 已完成攻击: {total_completed}/{total_attacks} ({total_completed/total_attacks*100:.1f}%)")
    print(f"  - 已完成轮数: {total_rounds_done}/{total_rounds_needed} ({total_rounds_done/total_rounds_needed*100:.1f}%)")
    print(f"  - GPU 0: {gpu0_completed}/{len(gpu0_attacks)} 攻击完成")
    print(f"  - GPU 1: {gpu1_completed}/{len(gpu1_attacks)} 攻击完成")

    # 检查进程状态
    print()
    print("[CHECK] 进程状态:")
    gpu0_log = Path("experiments/results/rq1_mnist_polfl_20r_gpu0_batch2.log")
    gpu1_log = Path("experiments/results/rq1_mnist_polfl_20r_gpu1_batch2.log")

    if gpu0_log.exists():
        mtime = datetime.fromtimestamp(gpu0_log.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds()
        status = "[ACTIVE] 活跃" if age < 300 else "[STALLED] 可能停滞"
        print(f"  - GPU 0 日志: {status} (最后更新: {age:.0f}秒前)")
    else:
        print(f"  - GPU 0 日志: [FAIL] 不存在")

    if gpu1_log.exists():
        mtime = datetime.fromtimestamp(gpu1_log.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds()
        status = "[ACTIVE] 活跃" if age < 300 else "[STALLED] 可能停滞"
        print(f"  - GPU 1 日志: {status} (最后更新: {age:.0f}秒前)")
    else:
        print(f"  - GPU 1 日志: [FAIL] 不存在")

    print()
    print("=" * 100)

    # 如果全部完成，显示成功消息
    if total_completed == total_attacks:
        print()
        print("[PASS] " * 20)
        print("所有实验已完成。")
        print("[PASS] " * 20)
        return True

    return False

if __name__ == "__main__":
    # 切换到正确的目录
    script_dir = Path(__file__).parent.parent.parent
    os.chdir(script_dir)

    completed = main()

    if not completed:
        print()
        print("[NOTE] Monitoring notes:")
        print("  - 使用 'watch -n 60 python3 experiments/scripts/monitor_rq1_batch2.py' 自动刷新")
        print("  - 查看详细日志: tail -f experiments/results/rq1_mnist_polfl_20r_gpu0_batch2.log")
