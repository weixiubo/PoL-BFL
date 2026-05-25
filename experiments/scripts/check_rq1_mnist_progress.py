#!/usr/bin/env python3
"""检查RQ1 MNIST清障实验进度"""

import os
from pathlib import Path
from datetime import datetime

# 预期的配置
attacks = [
    'byzantine_random_noise',
    'byzantine_model_replacement', 
    'byzantine_gradient_inversion',
    'byzantine_label_flipping',
    'byzantine_alie',
    'byzantine_ipm',
    'byzantine_minmax',
    'free_riding_no_training',
    'free_riding_lazy_training',
    'free_riding_minimal_update',
    'no_attack'
]

methods_gpu0 = ['Vanilla_FL', 'Krum', 'Trimmed_Mean', 'Median']
methods_gpu1 = ['ShapleyFL', 'FoolsGold', 'PoL_FL']

def check_gpu_progress(gpu_id, methods, result_dir):
    """检查单个GPU的进度"""
    actual_files = set()
    if result_dir.exists():
        for f in result_dir.glob('*.csv'):
            actual_files.add(f.name)
    
    total = len(attacks) * len(methods)
    completed = []
    missing = []
    
    for attack in attacks:
        for method in methods:
            expected_file = f"rq1_rounds_MNIST_{attack}_{method}.csv"
            if expected_file in actual_files:
                completed.append(expected_file)
            else:
                missing.append((attack, method))
    
    return {
        'total': total,
        'completed': len(completed),
        'missing': missing,
        'completed_files': completed
    }

def main():
    print("=" * 100)
    print(f"RQ1 MNIST 清障实验进度检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print()
    
    # GPU 0
    gpu0_dir = Path('experiments/results/clearance/rq1_mnist_smoke_gpu0')
    gpu0_progress = check_gpu_progress(0, methods_gpu0, gpu0_dir)
    
    print(f"【GPU 0】 Vanilla_FL, Krum, Trimmed_Mean, Median")
    print(f"  进度: {gpu0_progress['completed']}/{gpu0_progress['total']} ({gpu0_progress['completed']/gpu0_progress['total']*100:.1f}%)")
    if gpu0_progress['missing']:
        print(f"  ❌ 缺失 {len(gpu0_progress['missing'])} 个配置:")
        for attack, method in gpu0_progress['missing'][:5]:
            print(f"     - {attack} × {method}")
        if len(gpu0_progress['missing']) > 5:
            print(f"     ... 还有 {len(gpu0_progress['missing']) - 5} 个")
    else:
        print(f"  ✅ 所有配置已完成!")
    print()
    
    # GPU 1
    gpu1_dir = Path('experiments/results/clearance/rq1_mnist_smoke_gpu1')
    gpu1_progress = check_gpu_progress(1, methods_gpu1, gpu1_dir)
    
    print(f"【GPU 1】 ShapleyFL, FoolsGold, PoL_FL")
    print(f"  进度: {gpu1_progress['completed']}/{gpu1_progress['total']} ({gpu1_progress['completed']/gpu1_progress['total']*100:.1f}%)")
    if gpu1_progress['missing']:
        print(f"  ⏳ 待完成 {len(gpu1_progress['missing'])} 个配置:")
        for attack, method in gpu1_progress['missing']:
            print(f"     - {attack} × {method}")
    else:
        print(f"  ✅ 所有配置已完成!")
    print()
    
    # 总计
    total_configs = gpu0_progress['total'] + gpu1_progress['total']
    total_completed = gpu0_progress['completed'] + gpu1_progress['completed']
    
    print("=" * 100)
    print(f"【总进度】 {total_completed}/{total_configs} ({total_completed/total_configs*100:.1f}%)")
    print("=" * 100)
    
    if total_completed == total_configs:
        print()
        print("🎉🎉🎉 RQ1 MNIST 清障实验全部完成！🎉🎉🎉")
        print()
    else:
        remaining = total_configs - total_completed
        print(f"还剩 {remaining} 个配置待完成")
        print()

if __name__ == '__main__':
    main()

