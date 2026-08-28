#!/usr/bin/env python3
"""RQ1 MNIST验证实验最终数据质量分析"""

import pandas as pd
import numpy as np
from pathlib import Path

# 已知的方法列表
KNOWN_METHODS = ['Vanilla_FL', 'Krum', 'Trimmed_Mean', 'Median', 'ShapleyFL', 'FoolsGold', 'PoL_FL']

def parse_filename(filename):
    """解析文件名: rq1_rounds_MNIST_<attack>_<method>.csv"""
    stem = filename.replace('.csv', '')

    # 尝试匹配已知方法
    for method in KNOWN_METHODS:
        if stem.endswith('_' + method):
            # 提取攻击类型
            attack = stem[len('rq1_rounds_MNIST_'):-len('_' + method)]
            return attack, method

    return None, None

def main():
    print("=" * 120)
    print("RQ1 MNIST 验证实验最终数据质量分析")
    print("=" * 120)
    print()

    # 分析所有CSV文件
    gpu0_dir = Path('experiments/results/validation/rq1_mnist_smoke_gpu0')
    gpu1_dir = Path('experiments/results/validation/rq1_mnist_smoke_gpu1')

    all_results = []

    for gpu_dir, gpu_name in [(gpu0_dir, 'GPU0'), (gpu1_dir, 'GPU1')]:
        if not gpu_dir.exists():
            continue

        for csv_file in sorted(gpu_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_file)

                attack, method = parse_filename(csv_file.name)
                if attack is None or method is None:
                    print(f"[WARNING]  无法解析文件名: {csv_file.name}")
                    continue

                result = {
                    'gpu': gpu_name,
                    'attack': attack,
                    'method': method,
                    'file': csv_file.name,
                    'rounds': len(df),
                    'final_acc': df['test_accuracy'].iloc[-1],
                    'avg_acc': df['test_accuracy'].mean(),
                }

                # PoL特有指标
                if 'detection_tpr_e2e' in df.columns:
                    result['final_tpr'] = df['detection_tpr_e2e'].iloc[-1]
                    result['avg_tpr'] = df['detection_tpr_e2e'].mean()
                    result['final_fpr'] = df['detection_fpr'].iloc[-1]
                    result['avg_fpr'] = df['detection_fpr'].mean()
                else:
                    result['final_tpr'] = None
                    result['avg_tpr'] = None
                    result['final_fpr'] = None
                    result['avg_fpr'] = None

                all_results.append(result)

            except Exception as e:
                print(f"[FAIL] 错误读取 {csv_file.name}: {e}")

    df_results = pd.DataFrame(all_results)

    print(f"总文件数: {len(df_results)}")
    print(f"GPU 0: {len(df_results[df_results['gpu']=='GPU0'])} 个")
    print(f"GPU 1: {len(df_results[df_results['gpu']=='GPU1'])} 个")
    print()

    # 按方法统计
    print("【按方法统计】")
    print("-" * 120)
    print(f"{'方法':<20} {'实验数':<10} {'平均准确率':<15} {'准确率范围':<30}")
    print("-" * 120)

    for method in KNOWN_METHODS:
        method_df = df_results[df_results['method'] == method]
        if len(method_df) > 0:
            avg_acc = method_df['final_acc'].mean()
            min_acc = method_df['final_acc'].min()
            max_acc = method_df['final_acc'].max()

            print(f"{method:<20} {len(method_df):<10} {avg_acc:<15.4f} [{min_acc:.4f}, {max_acc:.4f}]")

    print()

    # PoL_FL详细分析
    print("【PoL_FL 检测性能】")
    print("-" * 120)

    pol_df = df_results[df_results['method'] == 'PoL_FL'].copy()
    if len(pol_df) > 0:
        print(f"{'攻击类型':<35} {'最终准确率':<15} {'最终TPR':<15} {'最终FPR':<15}")
        print("-" * 120)

        for _, row in pol_df.iterrows():
            tpr = row['final_tpr']
            fpr = row['final_fpr']
            tpr_str = f"{tpr:.4f}" if pd.notna(tpr) else "N/A"
            fpr_str = f"{fpr:.4f}" if pd.notna(fpr) else "N/A"
            print(f"{row['attack']:<35} {row['final_acc']:<15.4f} {tpr_str:<15} {fpr_str:<15}")

        # PoL统计
        pol_with_tpr = pol_df[pol_df['final_tpr'].notna()]
        if len(pol_with_tpr) > 0:
            print()
            print(f"PoL_FL统计 ({len(pol_with_tpr)} 个有检测数据的实验):")
            print(f"  - 平均TPR: {pol_with_tpr['final_tpr'].mean():.4f}")
            print(f"  - 平均FPR: {pol_with_tpr['final_fpr'].mean():.4f}")
            print(f"  - 平均准确率: {pol_with_tpr['final_acc'].mean():.4f}")

    print()

    # 数据质量检查
    print("【数据质量检查】")
    print("-" * 120)

    # 检查NaN
    nan_acc = df_results[df_results['final_acc'].isna()]
    if len(nan_acc) > 0:
        print(f"[FAIL] {len(nan_acc)} 个实验准确率为NaN")
        for _, row in nan_acc.iterrows():
            print(f"  - {row['file']}")

    # 检查轮数
    wrong_rounds = df_results[df_results['rounds'] != 5]
    if len(wrong_rounds) > 0:
        print(f"[FAIL] {len(wrong_rounds)} 个实验轮数不是5")
        for _, row in wrong_rounds.iterrows():
            print(f"  - {row['file']}: {row['rounds']} 轮")

    # 检查准确率异常（排除已知会导致低准确率的情况）
    low_acc = df_results[df_results['final_acc'] < 0.5]
    if len(low_acc) > 0:
        print(f"\n[WARNING]  {len(low_acc)} 个实验最终准确率 < 0.5:")
        print("(这可能是正常的，因为某些攻击会严重降低准确率)")
        for _, row in low_acc.head(10).iterrows():
            print(f"  - {row['attack']:<35} × {row['method']:<15}: {row['final_acc']:.4f}")
        if len(low_acc) > 10:
            print(f"  ... 还有 {len(low_acc) - 10} 个")

    print()

    if len(nan_acc) == 0 and len(wrong_rounds) == 0:
        print("[PASS] 核心数据质量检查通过。")
        print()
        print("关键发现:")
        print(f"  - 所有实验都完成了5轮训练")
        print(f"  - 总实验数: {len(df_results)}")
        print(f"  - 准确率范围: [{df_results['final_acc'].min():.4f}, {df_results['final_acc'].max():.4f}]")
        print(f"  - 平均准确率: {df_results['final_acc'].mean():.4f}")
    else:
        print("[FAIL] 发现数据质量问题，需要进一步检查")

    print()
    print("=" * 120)

    # 方法对比（按攻击类型）
    print()
    print("【方法性能对比 - 按攻击类型】")
    print("-" * 120)

    attack_categories = {
        'Byzantine攻击': ['byzantine_random_noise', 'byzantine_model_replacement', 'byzantine_gradient_inversion',
                        'byzantine_label_flipping', 'byzantine_alie', 'byzantine_ipm', 'byzantine_minmax'],
        'Free-riding攻击': ['free_riding_no_training', 'free_riding_lazy_training', 'free_riding_minimal_update'],
        '无攻击': ['no_attack']
    }

    for category, attacks in attack_categories.items():
        print(f"\n{category}:")
        print(f"{'方法':<20} {'实验数':<10} {'平均准确率':<15}")
        print("-" * 60)

        for method in KNOWN_METHODS:
            method_attack_results = df_results[
                (df_results['method'] == method) &
                (df_results['attack'].isin(attacks))
            ]

            if len(method_attack_results) > 0:
                avg_acc = method_attack_results['final_acc'].mean()
                print(f"{method:<20} {len(method_attack_results):<10} {avg_acc:.4f}")

    print()
    print("=" * 120)

if __name__ == '__main__':
    main()

