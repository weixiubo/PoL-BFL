#!/usr/bin/env python3
"""
实验监控脚本
定期检查RQ1实验的进度和状态
"""

import json
import time
from pathlib import Path
from datetime import datetime

def check_experiment_progress(output_dir):
    """检查实验进度"""
    output_path = Path(output_dir)

    if not output_path.exists():
        return {
            'status': 'not_started',
            'message': f'输出目录不存在: {output_dir}'
        }

    # 检查配置文件
    config_file = output_path / 'config.json'
    if not config_file.exists():
        return {
            'status': 'initializing',
            'message': '配置文件尚未生成'
        }

    with open(config_file) as f:
        config = json.load(f)

    # 检查结果文件
    results_file = output_path / 'rq1_results.json'
    if not results_file.exists():
        return {
            'status': 'running',
            'message': '实验运行中，结果文件尚未生成',
            'config': config
        }

    with open(results_file) as f:
        results = json.load(f)

    # 分析进度
    total_experiments = len(results)
    attacks = set(exp['attack_type'] for exp in results)

    # 检查每个实验的轮数
    rounds_info = {}
    for exp in results:
        attack = exp['attack_type']
        rounds = exp.get('rounds', [])
        rounds_info[attack] = len(rounds)

    expected_rounds = config.get('num_rounds', 20)
    completed_attacks = sum(1 for r in rounds_info.values() if r >= expected_rounds)

    return {
        'status': 'running' if completed_attacks < len(attacks) else 'completed',
        'total_experiments': total_experiments,
        'attacks': list(attacks),
        'rounds_info': rounds_info,
        'expected_rounds': expected_rounds,
        'completed_attacks': completed_attacks,
        'total_attacks': len(attacks),
        'progress': f'{completed_attacks}/{len(attacks)} 攻击完成'
    }

def format_status_report(gpu_id, output_dir):
    """格式化状态报告"""
    status = check_experiment_progress(output_dir)

    report = f"\n{'='*80}\n"
    report += f"GPU {gpu_id} 实验状态 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"{'='*80}\n"
    report += f"输出目录: {output_dir}\n"
    report += f"状态: {status['status']}\n"

    if status['status'] == 'running' or status['status'] == 'completed':
        report += f"\n进度: {status.get('progress', 'N/A')}\n"
        report += f"预期轮数: {status.get('expected_rounds', 'N/A')}\n"
        report += f"\n各攻击完成轮数:\n"
        for attack, rounds in status.get('rounds_info', {}).items():
            expected = status.get('expected_rounds', 20)
            progress_bar = '█' * (rounds * 20 // expected) + '░' * (20 - rounds * 20 // expected)
            report += f"  {attack:40s}: [{progress_bar}] {rounds}/{expected}\n"
    else:
        report += f"消息: {status.get('message', 'N/A')}\n"

    report += f"{'='*80}\n"

    return report

def main():
    """主函数"""
    gpu0_dir = 'experiments/results/rq1_mnist_polfl_20r_gpu0'
    gpu1_dir = 'experiments/results/rq1_mnist_polfl_20r_gpu1'

    print("\n" + "="*80)
    print("RQ1 实验监控")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("按 Ctrl+C 停止监控\n")

    try:
        while True:
            # 清屏（可选）
            # print("\033[2J\033[H")

            # 检查GPU 0
            print(format_status_report(0, gpu0_dir))

            # 检查GPU 1
            print(format_status_report(1, gpu1_dir))

            # 检查是否都完成
            status0 = check_experiment_progress(gpu0_dir)
            status1 = check_experiment_progress(gpu1_dir)

            if status0['status'] == 'completed' and status1['status'] == 'completed':
                print("\n[PASS] 所有实验已完成。\n")
                break

            # 等待30秒
            print(f"下次检查: {(datetime.now().timestamp() + 30):.0f} (30秒后)")
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n监控已停止。\n")

if __name__ == '__main__':
    main()

