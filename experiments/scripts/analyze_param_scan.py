#!/usr/bin/env python3
"""
参数扫描结果分析脚本
分析阶段1粗调结果，推荐最优参数组合
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_results(input_dir):
    """加载所有实验结果"""
    results = []
    input_path = Path(input_dir)
    
    for result_file in input_path.rglob("rq1_results.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                if data:
                    # 提取参数信息（从目录名）
                    dir_name = result_file.parent.name
                    parts = dir_name.split('_')
                    
                    # 解析参数
                    delta = None
                    vr = None
                    attack = None
                    
                    for i, part in enumerate(parts):
                        if part.startswith('delta'):
                            delta = float(part.replace('delta', ''))
                        elif part.startswith('vr'):
                            vr = float(part.replace('vr', ''))
                        elif i >= 2:  # 攻击类型在后面
                            if attack is None:
                                attack = part
                            else:
                                attack += '_' + part
                    
                    # 提取指标
                    metrics = data[0]['detection_metrics']
                    
                    results.append({
                        'delta': delta,
                        'verification_rate': vr,
                        'attack_type': attack,
                        'TPR': metrics.get('TPR', 0),
                        'FPR': metrics.get('FPR', 0),
                        'TPR_conditional': metrics.get('TPR_conditional', 0),
                        'precision': metrics.get('Precision', 0),
                        'recall': metrics.get('Recall', 0),
                        'f1': metrics.get('F1', 0),
                        'final_accuracy': data[0].get('final_accuracy', 0),
                        'participation_rate': metrics.get('participation_rate', 0),
                        'verification_pass_rate': metrics.get('verification_pass_rate', 0),
                    })
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return results


def analyze_results(results):
    """分析结果并推荐参数"""
    if not results:
        print("No results found!")
        return None
    
    df = pd.DataFrame(results)
    
    # 按参数组合聚合
    grouped = df.groupby(['delta', 'verification_rate']).agg({
        'TPR': 'mean',
        'FPR': 'mean',
        'TPR_conditional': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'final_accuracy': 'mean',
        'participation_rate': 'mean',
        'verification_pass_rate': 'mean',
    }).reset_index()
    
    # 计算综合得分
    # 权重: TPR(40%) + (1-FPR)(30%) + Accuracy(20%) + F1(10%)
    grouped['score'] = (
        0.40 * grouped['TPR'] +
        0.30 * (1 - grouped['FPR']) +
        0.20 * grouped['final_accuracy'] +
        0.10 * grouped['f1']
    )
    
    # 排序
    grouped = grouped.sort_values('score', ascending=False)
    
    # 筛选条件
    # 1. TPR >= 0.90
    # 2. FPR <= 0.10
    # 3. Accuracy >= 0.80
    qualified = grouped[
        (grouped['TPR'] >= 0.90) &
        (grouped['FPR'] <= 0.10) &
        (grouped['final_accuracy'] >= 0.80)
    ]
    
    analysis = {
        'total_combinations': len(grouped),
        'qualified_combinations': len(qualified),
        'all_results': grouped.to_dict('records'),
        'qualified_results': qualified.to_dict('records'),
        'top_3': grouped.head(3).to_dict('records'),
        'recommendations': []
    }
    
    # 生成推荐
    if len(qualified) > 0:
        top = qualified.iloc[0]
        analysis['recommendations'].append({
            'rank': 1,
            'delta': top['delta'],
            'verification_rate': top['verification_rate'],
            'TPR': top['TPR'],
            'FPR': top['FPR'],
            'accuracy': top['final_accuracy'],
            'score': top['score'],
            'reason': 'Best overall score among qualified combinations'
        })
        
        # 推荐高TPR配置
        high_tpr = qualified.nlargest(1, 'TPR').iloc[0]
        if high_tpr['delta'] != top['delta'] or high_tpr['verification_rate'] != top['verification_rate']:
            analysis['recommendations'].append({
                'rank': 2,
                'delta': high_tpr['delta'],
                'verification_rate': high_tpr['verification_rate'],
                'TPR': high_tpr['TPR'],
                'FPR': high_tpr['FPR'],
                'accuracy': high_tpr['final_accuracy'],
                'score': high_tpr['score'],
                'reason': 'Highest TPR among qualified combinations'
            })
        
        # 推荐低FPR配置
        low_fpr = qualified.nsmallest(1, 'FPR').iloc[0]
        if (low_fpr['delta'] != top['delta'] or low_fpr['verification_rate'] != top['verification_rate']) and \
           (low_fpr['delta'] != high_tpr['delta'] or low_fpr['verification_rate'] != high_tpr['verification_rate']):
            analysis['recommendations'].append({
                'rank': 3,
                'delta': low_fpr['delta'],
                'verification_rate': low_fpr['verification_rate'],
                'TPR': low_fpr['TPR'],
                'FPR': low_fpr['FPR'],
                'accuracy': low_fpr['final_accuracy'],
                'score': low_fpr['score'],
                'reason': 'Lowest FPR among qualified combinations'
            })
    else:
        print("Warning: No qualified combinations found!")
        print("Relaxing criteria...")
        # 放宽条件
        relaxed = grouped[
            (grouped['TPR'] >= 0.80) &
            (grouped['FPR'] <= 0.15)
        ]
        if len(relaxed) > 0:
            top = relaxed.iloc[0]
            analysis['recommendations'].append({
                'rank': 1,
                'delta': top['delta'],
                'verification_rate': top['verification_rate'],
                'TPR': top['TPR'],
                'FPR': top['FPR'],
                'accuracy': top['final_accuracy'],
                'score': top['score'],
                'reason': 'Best under relaxed criteria (TPR>=0.80, FPR<=0.15)'
            })
    
    return analysis


def print_summary(analysis):
    """打印分析摘要"""
    print("\n" + "="*60)
    print("参数扫描分析摘要")
    print("="*60)
    
    print(f"\n总参数组合数: {analysis['total_combinations']}")
    print(f"合格组合数: {analysis['qualified_combinations']}")
    
    print("\n" + "-"*60)
    print("Top 3 参数组合（按综合得分）:")
    print("-"*60)
    
    for i, result in enumerate(analysis['top_3'], 1):
        print(f"\n#{i}")
        print(f"  Delta: {result['delta']}")
        print(f"  Verification Rate: {result['verification_rate']}")
        print(f"  TPR: {result['TPR']:.4f}")
        print(f"  FPR: {result['FPR']:.4f}")
        print(f"  Accuracy: {result['final_accuracy']:.4f}")
        print(f"  Score: {result['score']:.4f}")
    
    if analysis['recommendations']:
        print("\n" + "-"*60)
        print("推荐配置:")
        print("-"*60)
        
        for rec in analysis['recommendations']:
            print(f"\n推荐 #{rec['rank']}: {rec['reason']}")
            print(f"  Delta: {rec['delta']}")
            print(f"  Verification Rate: {rec['verification_rate']}")
            print(f"  TPR: {rec['TPR']:.4f}")
            print(f"  FPR: {rec['FPR']:.4f}")
            print(f"  Accuracy: {rec['accuracy']:.4f}")
            print(f"  Score: {rec['score']:.4f}")
    else:
        print("\n⚠️  警告: 没有找到合格的参数组合！")
        print("建议:")
        print("  1. 扩大参数搜索范围")
        print("  2. 检查代码实现")
        print("  3. 调整评估标准")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='分析参数扫描结果')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录（包含实验结果）')
    parser.add_argument('--output', type=str, required=True,
                        help='输出文件（JSON格式）')
    
    args = parser.parse_args()
    
    print(f"加载结果从: {args.input_dir}")
    results = load_results(args.input_dir)
    print(f"找到 {len(results)} 个实验结果")
    
    if results:
        analysis = analyze_results(results)
        
        # 保存结果
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n分析结果已保存到: {args.output}")
        
        # 打印摘要
        print_summary(analysis)
    else:
        print("没有找到有效的实验结果！")


if __name__ == '__main__':
    main()

