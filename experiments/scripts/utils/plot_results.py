#!/usr/bin/env python3
"""
生成论文图表
基于实验结果JSON文件生成高质量的PDF图表
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 设置matplotlib参数以生成高质量图表
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 12

# 颜色方案
COLORS = {
    'Vanilla_FL': '#E74C3C',      # 红色
    'Krum': '#3498DB',             # 蓝色
    'Trimmed_Mean': '#2ECC71',     # 绿色
    'PoL_FL': '#9B59B6',           # 紫色
}

def load_results():
    """加载所有实验结果"""
    from experiment_config import OUTPUT_CONFIG
    results_dir = Path(OUTPUT_CONFIG['results_dir'])

    results = {}

    # RQ1: Security
    rq1_file = results_dir / 'rq1_security' / 'rq1_results.json'
    if rq1_file.exists():
        with open(rq1_file, 'r') as f:
            results['rq1'] = json.load(f)

    # RQ2: Overhead
    rq2_file = results_dir / 'rq2_overhead' / 'rq2_results.json'
    if rq2_file.exists():
        with open(rq2_file, 'r') as f:
            results['rq2'] = json.load(f)

    # RQ3: Scalability
    rq3_file = results_dir / 'rq3_scalability' / 'rq3_results.json'
    if rq3_file.exists():
        with open(rq3_file, 'r') as f:
            results['rq3'] = json.load(f)

    # RQ4: Incentive
    rq4_file = results_dir / 'rq4_incentive' / 'rq4_results.json'
    if rq4_file.exists():
        with open(rq4_file, 'r') as f:
            results['rq4'] = json.load(f)

    return results

def plot_rq1_convergence(results, output_dir):
    """RQ1: 收敛曲线图"""

    if 'rq1' not in results:
        print("RQ1 results not found, skipping...")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Byzantine attack
    byzantine_data = [r for r in results['rq1'] if 'byzantine' in r['attack_type']]
    for method_data in byzantine_data:
        method = method_data['baseline_method']
        accuracies = method_data['test_accuracies']
        rounds = list(range(1, len(accuracies) + 1))
        ax1.plot(rounds, accuracies, marker='o', label=method,
                color=COLORS.get(method, 'gray'), linewidth=2, markersize=5)

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('(a) Byzantine Attack (20% malicious)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Free-riding attack
    freeriding_data = [r for r in results['rq1'] if 'free_riding' in r['attack_type']]
    for method_data in freeriding_data:
        method = method_data['baseline_method']
        accuracies = method_data['test_accuracies']
        rounds = list(range(1, len(accuracies) + 1))
        ax2.plot(rounds, accuracies, marker='s', label=method,
                color=COLORS.get(method, 'gray'), linewidth=2, markersize=5)

    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('(b) Free-Riding Attack (20% lazy clients)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    output_file = output_dir / 'rq1_convergence.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()

def plot_rq2_overhead(results, output_dir):
    """RQ2: 系统开销对比图"""

    if 'rq2' not in results:
        print("RQ2 results not found, skipping...")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    methods = []
    training_times = []
    comm_overheads = []
    storage_mbs = []

    for method_data in results['rq2']:
        method = method_data['method']
        methods.append(method.replace('_', ' '))
        training_times.append(method_data['total_training_time'])
        comm_overheads.append(method_data['total_communication_mb'])

        if 'total_storage_mb' in method_data:
            storage_mbs.append(method_data['total_storage_mb'])
        else:
            storage_mbs.append(0)

    # Training time
    x = np.arange(len(methods))
    width = 0.6

    bars1 = axes[0].bar(x, training_times, width, color=['#E74C3C', '#9B59B6'])
    axes[0].set_ylabel('Training Time (s)')
    axes[0].set_title('(a) Training Time')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s',
                    ha='center', va='bottom', fontsize=9)

    # Communication overhead
    bars2 = axes[1].bar(x, comm_overheads, width, color=['#E74C3C', '#9B59B6'])
    axes[1].set_ylabel('Communication (MB)')
    axes[1].set_title('(b) Communication Overhead')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}MB',
                    ha='center', va='bottom', fontsize=9)

    # Storage overhead
    bars3 = axes[2].bar(x, [s/1024 for s in storage_mbs], width, color=['#E74C3C', '#9B59B6'])
    axes[2].set_ylabel('Storage (GB)')
    axes[2].set_title('(c) Storage Overhead')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')

    for bar, storage in zip(bars3, storage_mbs):
        height = bar.get_height()
        if storage > 0:
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{storage/1024:.2f}GB',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / 'rq2_overhead.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()

def plot_rq4_incentive(results, output_dir):
    """RQ4: 激励机制效果图"""

    if 'rq4' not in results:
        print("RQ4 results not found, skipping...")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    scenarios = []
    honest_utils = []
    rational_utils = []
    malicious_utils = []

    for scenario_data in results['rq4']:
        scenario = scenario_data.get('scenario', 'Unknown')
        scenarios.append(scenario.replace('_', ' ').title())

        utilities = scenario_data.get('utilities', {})
        honest_utils.append(utilities.get('honest', 0))
        rational_utils.append(utilities.get('rational', 0))
        malicious_utils.append(utilities.get('malicious', 0))

    x = np.arange(len(scenarios))
    width = 0.25

    ax.bar(x - width, honest_utils, width, label='Honest', color='#2ECC71')
    ax.bar(x, rational_utils, width, label='Rational', color='#3498DB')
    ax.bar(x + width, malicious_utils, width, label='Malicious', color='#E74C3C')

    ax.set_ylabel('Expected Utility')
    ax.set_title('Incentive Mechanism Effectiveness')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / 'rq4_incentive.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()

def main():
    """主函数"""

    print("="*70)
    print("Generating Paper Figures")
    print("="*70)

    # 创建输出目录
    from experiment_config import OUTPUT_CONFIG
    output_dir = Path(OUTPUT_CONFIG['plots_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载结果
    print("\nLoading results...")
    results = load_results()

    print(f"Found results for: {list(results.keys())}")
    print()

    # 生成图表
    print("Generating figures...")

    if 'rq1' in results:
        plot_rq1_convergence(results, output_dir)

    if 'rq2' in results:
        plot_rq2_overhead(results, output_dir)

    if 'rq4' in results:
        plot_rq4_incentive(results, output_dir)

    print()
    print("="*70)
    print("Done. Figures saved to experiments/plots/")
    print("="*70)
    print("\nGenerated files:")
    for pdf_file in sorted(output_dir.glob('*.pdf')):
        print(f"  - {pdf_file.name}")

if __name__ == '__main__':
    main()

