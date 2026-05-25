#!/usr/bin/env python3
"""分析RQ1 MNIST清障实验结果的数据质量"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_csv_file(csv_path):
    """分析单个CSV文件"""
    try:
        df = pd.read_csv(csv_path)
        
        # 基本检查
        if len(df) == 0:
            return {'error': 'Empty file'}
        
        # 提取关键指标
        result = {
            'file': csv_path.name,
            'rounds': len(df),
            'final_acc': df['Main_Accuracy'].iloc[-1] if 'Main_Accuracy' in df.columns else None,
            'avg_acc': df['Main_Accuracy'].mean() if 'Main_Accuracy' in df.columns else None,
            'has_detection': 'TPR' in df.columns or 'TPR_e2e' in df.columns,
            'final_tpr': None,
            'final_fpr': None,
            'avg_tpr': None,
            'avg_fpr': None,
        }
        
        # 检测率指标
        if 'TPR_e2e' in df.columns:
            result['final_tpr'] = df['TPR_e2e'].iloc[-1]
            result['avg_tpr'] = df['TPR_e2e'].mean()
        elif 'TPR' in df.columns:
            result['final_tpr'] = df['TPR'].iloc[-1]
            result['avg_tpr'] = df['TPR'].mean()
        
        if 'FPR' in df.columns:
            result['final_fpr'] = df['FPR'].iloc[-1]
            result['avg_fpr'] = df['FPR'].mean()
        
        # 数据质量检查
        if result['final_acc'] is not None:
            if result['final_acc'] < 0.5:
                result['warning'] = 'Low accuracy (<0.5)'
            elif np.isnan(result['final_acc']):
                result['error'] = 'NaN accuracy'
        
        return result
        
    except Exception as e:
        return {'file': csv_path.name, 'error': str(e)}

def main():
    print("=" * 120)
    print(f"RQ1 MNIST 清障实验数据质量分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)
    print()
    
    # 分析GPU 0
    print("【GPU 0】 Vanilla_FL, Krum, Trimmed_Mean, Median")
    print("-" * 120)
    
    gpu0_dir = Path('experiments/results/clearance/rq1_mnist_smoke_gpu0')
    gpu0_results = []
    
    if gpu0_dir.exists():
        for csv_file in sorted(gpu0_dir.glob('*.csv')):
            result = analyze_csv_file(csv_file)
            gpu0_results.append(result)
    
    # 按方法分组统计
    methods = ['Vanilla_FL', 'Krum', 'Trimmed_Mean', 'Median']
    for method in methods:
        method_results = [r for r in gpu0_results if method in r['file']]
        if method_results:
            valid_results = [r for r in method_results if 'error' not in r]
            if valid_results:
                avg_final_acc = np.mean([r['final_acc'] for r in valid_results if r['final_acc'] is not None])
                print(f"\n  {method}:")
                print(f"    实验数: {len(method_results)}")
                print(f"    平均最终准确率: {avg_final_acc:.4f}")
                
                # 检查异常
                errors = [r for r in method_results if 'error' in r]
                warnings = [r for r in method_results if 'warning' in r]
                if errors:
                    print(f"    ❌ 错误: {len(errors)} 个")
                    for e in errors[:3]:
                        print(f"       - {e['file']}: {e['error']}")
                if warnings:
                    print(f"    ⚠️  警告: {len(warnings)} 个")
                    for w in warnings[:3]:
                        print(f"       - {w['file']}: {w['warning']}")
    
    print()
    print("=" * 120)
    
    # 分析GPU 1
    print("【GPU 1】 ShapleyFL, FoolsGold, PoL_FL")
    print("-" * 120)
    
    gpu1_dir = Path('experiments/results/clearance/rq1_mnist_smoke_gpu1')
    gpu1_results = []
    
    if gpu1_dir.exists():
        for csv_file in sorted(gpu1_dir.glob('*.csv')):
            result = analyze_csv_file(csv_file)
            gpu1_results.append(result)
    
    # 按方法分组统计
    methods = ['ShapleyFL', 'FoolsGold', 'PoL_FL']
    for method in methods:
        method_results = [r for r in gpu1_results if method in r['file']]
        if method_results:
            valid_results = [r for r in method_results if 'error' not in r]
            if valid_results:
                avg_final_acc = np.mean([r['final_acc'] for r in valid_results if r['final_acc'] is not None])
                
                # PoL_FL有检测率
                has_detection = any(r['has_detection'] for r in valid_results)
                
                print(f"\n  {method}:")
                print(f"    实验数: {len(method_results)}")
                print(f"    平均最终准确率: {avg_final_acc:.4f}")
                
                if has_detection and method == 'PoL_FL':
                    tpr_results = [r for r in valid_results if r['final_tpr'] is not None]
                    if tpr_results:
                        avg_tpr = np.mean([r['final_tpr'] for r in tpr_results])
                        avg_fpr = np.mean([r['final_fpr'] for r in tpr_results if r['final_fpr'] is not None])
                        print(f"    平均TPR: {avg_tpr:.4f}")
                        print(f"    平均FPR: {avg_fpr:.4f}")
                
                # 检查异常
                errors = [r for r in method_results if 'error' in r]
                warnings = [r for r in method_results if 'warning' in r]
                if errors:
                    print(f"    ❌ 错误: {len(errors)} 个")
                    for e in errors[:3]:
                        print(f"       - {e['file']}: {e['error']}")
                if warnings:
                    print(f"    ⚠️  警告: {len(warnings)} 个")
                    for w in warnings[:3]:
                        print(f"       - {w['file']}: {w['warning']}")
    
    print()
    print("=" * 120)
    
    # 总体统计
    all_results = gpu0_results + gpu1_results
    total_files = len(all_results)
    total_errors = len([r for r in all_results if 'error' in r])
    total_warnings = len([r for r in all_results if 'warning' in r])
    
    print("【总体统计】")
    print(f"  总文件数: {total_files}")
    print(f"  错误数: {total_errors}")
    print(f"  警告数: {total_warnings}")
    
    if total_errors == 0 and total_warnings == 0:
        print()
        print("  ✅ 所有数据质量检查通过！")
    else:
        print()
        print("  ⚠️  发现异常，需要进一步检查")
    
    print()
    print("=" * 120)
    
    # 详细的方法对比分析
    print()
    print("【方法性能对比】")
    print("-" * 120)
    
    # 按攻击类型分组
    attack_types = {
        'byzantine': ['byzantine_random_noise', 'byzantine_model_replacement', 'byzantine_gradient_inversion',
                     'byzantine_label_flipping', 'byzantine_alie', 'byzantine_ipm', 'byzantine_minmax'],
        'free_riding': ['free_riding_no_training', 'free_riding_lazy_training', 'free_riding_minimal_update'],
        'no_attack': ['no_attack']
    }
    
    all_methods = ['Vanilla_FL', 'Krum', 'Trimmed_Mean', 'Median', 'ShapleyFL', 'FoolsGold', 'PoL_FL']
    
    for attack_category, attacks in attack_types.items():
        print(f"\n{attack_category.upper()}:")
        print(f"{'方法':<15} {'实验数':<10} {'平均准确率':<15} {'最终准确率范围':<25}")
        print("-" * 80)
        
        for method in all_methods:
            method_attack_results = [
                r for r in all_results 
                if method in r['file'] and any(attack in r['file'] for attack in attacks)
                and 'error' not in r and r['final_acc'] is not None
            ]
            
            if method_attack_results:
                avg_acc = np.mean([r['final_acc'] for r in method_attack_results])
                min_acc = np.min([r['final_acc'] for r in method_attack_results])
                max_acc = np.max([r['final_acc'] for r in method_attack_results])
                
                print(f"{method:<15} {len(method_attack_results):<10} {avg_acc:<15.4f} [{min_acc:.4f}, {max_acc:.4f}]")
    
    print()
    print("=" * 120)

if __name__ == '__main__':
    main()

