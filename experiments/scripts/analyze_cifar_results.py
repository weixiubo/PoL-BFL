#!/usr/bin/env python3
"""
分析CIFAR-10/100实验结果的质量
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_cifar_results(results_dir):
    """分析CIFAR实验结果"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"[FAIL] 目录不存在: {results_dir}")
        return None

    print(f"\n{'='*80}")
    print(f"分析目录: {results_dir}")
    print(f"{'='*80}\n")

    # 查找所有JSON文件
    json_files = list(results_path.rglob('*.json'))
    print(f"找到 {len(json_files)} 个JSON文件\n")

    # 分析每个JSON文件
    results_summary = {
        'total_files': len(json_files),
        'experiments': [],
        'attacks': set(),
        'baselines': set(),
        'accuracies': [],
        'rounds': []
    }

    for json_file in json_files:
        if json_file.name in ['config.json', 'metadata.json']:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            # 提取关键信息
            if isinstance(data, list):
                # 可能是结果列表
                for exp in data:
                    attack = exp.get('attack_type', 'unknown')
                    baseline = exp.get('baseline_method', 'unknown')
                    final_acc = exp.get('final_accuracy', 0)
                    rounds = exp.get('rounds', [])

                    results_summary['attacks'].add(attack)
                    results_summary['baselines'].add(baseline)
                    results_summary['accuracies'].append(final_acc)
                    results_summary['rounds'].append(len(rounds))

                    results_summary['experiments'].append({
                        'attack': attack,
                        'baseline': baseline,
                        'final_accuracy': final_acc,
                        'num_rounds': len(rounds),
                        'file': str(json_file.relative_to(results_path))
                    })
            elif isinstance(data, dict):
                # 可能是单个实验
                attack = data.get('attack_type', 'unknown')
                baseline = data.get('baseline_method', 'unknown')
                final_acc = data.get('final_accuracy', 0)
                rounds = data.get('rounds', [])

                if attack != 'unknown':
                    results_summary['attacks'].add(attack)
                    results_summary['baselines'].add(baseline)
                    results_summary['accuracies'].append(final_acc)
                    results_summary['rounds'].append(len(rounds))

                    results_summary['experiments'].append({
                        'attack': attack,
                        'baseline': baseline,
                        'final_accuracy': final_acc,
                        'num_rounds': len(rounds),
                        'file': str(json_file.relative_to(results_path))
                    })
        except Exception as e:
            print(f"[WARNING] 无法解析 {json_file.name}: {e}")

    # 生成报告
    print(f"### 实验概况")
    print(f"-" * 80)
    print(f"总实验数: {len(results_summary['experiments'])}")
    print(f"攻击类型: {len(results_summary['attacks'])} 种")
    print(f"Baseline方法: {len(results_summary['baselines'])} 种")

    if results_summary['rounds']:
        avg_rounds = sum(results_summary['rounds']) / len(results_summary['rounds'])
        min_rounds = min(results_summary['rounds'])
        max_rounds = max(results_summary['rounds'])
        print(f"轮数: 平均={avg_rounds:.1f}, 最小={min_rounds}, 最大={max_rounds}")

    if results_summary['accuracies']:
        avg_acc = sum(results_summary['accuracies']) / len(results_summary['accuracies'])
        min_acc = min(results_summary['accuracies'])
        max_acc = max(results_summary['accuracies'])
        print(f"准确率: 平均={avg_acc:.4f}, 最小={min_acc:.4f}, 最大={max_acc:.4f}")

    print(f"\n### 攻击类型")
    print(f"-" * 80)
    for attack in sorted(results_summary['attacks']):
        count = sum(1 for exp in results_summary['experiments'] if exp['attack'] == attack)
        print(f"  {attack:40s}: {count} 个实验")

    print(f"\n### Baseline方法")
    print(f"-" * 80)
    for baseline in sorted(results_summary['baselines']):
        count = sum(1 for exp in results_summary['experiments'] if exp['baseline'] == baseline)
        print(f"  {baseline:40s}: {count} 个实验")

    # 检查数据质量
    print(f"\n### 数据质量评估")
    print(f"-" * 80)

    issues = []

    # 检查轮数
    if results_summary['rounds']:
        if min(results_summary['rounds']) < 10:
            issues.append(f"[WARNING] 部分实验轮数过少（最小={min(results_summary['rounds'])}）")
        if max(results_summary['rounds']) < 20:
            issues.append(f"[WARNING] 所有实验轮数不足20轮（最大={max(results_summary['rounds'])}）")

    # 检查准确率
    if results_summary['accuracies']:
        low_acc_count = sum(1 for acc in results_summary['accuracies'] if acc < 0.5)
        if low_acc_count > 0:
            issues.append(f"[WARNING] {low_acc_count} 个实验准确率异常低（<0.5）")

    # 检查完整性
    expected_attacks = 10  # 假设应该有10种攻击
    if len(results_summary['attacks']) < expected_attacks:
        issues.append(f"[WARNING] 攻击类型不完整（{len(results_summary['attacks'])}/{expected_attacks}）")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("[PASS] 数据质量良好")

    # 建议
    print(f"\n### 建议")
    print(f"-" * 80)

    if not results_summary['experiments']:
        print("[FAIL] 无有效实验数据，建议重新运行")
    elif issues:
        print("[WARNING] 数据存在问题，建议：")
        if any('轮数' in issue for issue in issues):
            print("  - 重新运行实验，增加轮数到20轮")
        if any('准确率' in issue for issue in issues):
            print("  - 检查实验配置和攻击参数")
        if any('不完整' in issue for issue in issues):
            print("  - 补充缺失的攻击类型")
    else:
        print("[PASS] 数据可以使用")

    print(f"\n{'='*80}\n")

    return results_summary

def main():
    """主函数"""
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # 默认分析CIFAR-10和CIFAR-100
        print("分析CIFAR实验结果\n")

        cifar10_dir = 'experiments/results/rq1_L3_cifar10_polfl'
        cifar100_dir = 'experiments/results/rq1_L4_cifar100_polfl'

        print("### CIFAR-10")
        analyze_cifar_results(cifar10_dir)

        print("\n### CIFAR-100")
        analyze_cifar_results(cifar100_dir)

if __name__ == '__main__':
    main()

