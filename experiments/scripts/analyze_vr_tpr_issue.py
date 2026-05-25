#!/usr/bin/env python3
"""
深入分析为什么 vr≥0.3 的 TPR=0

分析内容：
1. Merkle 失败是否对应攻击者检测
2. vr=0.1 和 vr=0.3 验证的客户端是否包含攻击者
3. 验证逻辑是否有与 vr 相关的问题
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

def parse_experiment_log(log_path):
    """解析实验日志，提取关键信息"""
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 提取 verification_rate
    vr_match = re.search(r'Verification rate:\s+([\d.]+)', content)
    vr = float(vr_match.group(1)) if vr_match else None
    
    # 按实验分割日志
    experiments = re.split(r'Running:\s+(\w+)\s+vs\s+(\w+)', content)
    
    results = []
    
    for i in range(1, len(experiments), 3):
        defense = experiments[i]
        attack = experiments[i+1]
        exp_content = experiments[i+2]
        
        # 只分析 PoL_FL 的实验
        if defense != 'PoL_FL':
            continue
        
        # 提取每轮的信息
        rounds = re.split(r'Round\s+(\d+)/10', exp_content)
        
        round_data = []
        for j in range(1, len(rounds), 2):
            round_num = int(rounds[j])
            round_content = rounds[j+1]
            
            # 提取恶意客户端
            malicious_clients = re.findall(r'\[Malicious\]\s+(client_\d+)', round_content)
            
            # 提取验证的客户端索引
            verified_match = re.search(r'Selected indices:\s+\[([^\]]+)\]', round_content)
            verified_indices = []
            if verified_match:
                verified_indices = [int(x.strip()) for x in verified_match.group(1).split(',')]
            
            # 提取验证结果
            merkle_failures = []
            for match in re.finditer(r'(client_\d+):.*?Merkle.*?verification failed', round_content):
                merkle_failures.append(match.group(1))
            
            step_failures = []
            for match in re.finditer(r'(client_\d+):\s+total_steps.*?Marking verification failed', round_content):
                step_failures.append(match.group(1))
            
            round_data.append({
                'round': round_num,
                'malicious_clients': malicious_clients,
                'verified_indices': verified_indices,
                'merkle_failures': merkle_failures,
                'step_failures': step_failures,
            })
        
        # 提取最终指标
        tpr_match = re.search(r'TPR.*?:\s+([\d.]+)', exp_content)
        fpr_match = re.search(r'FPR.*?:\s+([\d.]+)', exp_content)
        
        results.append({
            'defense': defense,
            'attack': attack,
            'vr': vr,
            'rounds': round_data,
            'tpr': float(tpr_match.group(1)) if tpr_match else None,
            'fpr': float(fpr_match.group(1)) if fpr_match else None,
        })
    
    return results

def analyze_verification_coverage(results):
    """分析验证覆盖率：验证的客户端是否包含攻击者"""
    
    print("=" * 80)
    print("分析 1: 验证覆盖率 - 验证的客户端是否包含攻击者")
    print("=" * 80)
    
    for exp in results:
        attack = exp['attack']
        vr = exp['vr']
        
        # 跳过无攻击场景
        if attack == 'no_attack':
            continue
        
        total_rounds = len(exp['rounds'])
        rounds_with_malicious = 0
        rounds_verified_malicious = 0
        
        for round_data in exp['rounds']:
            malicious = set(round_data['malicious_clients'])
            
            if not malicious:
                continue
            
            rounds_with_malicious += 1
            
            # 检查验证的客户端是否包含攻击者
            # 注意：verified_indices 是相对索引（0-9），需要映射到 client_id
            # 但我们没有这个映射信息，所以暂时跳过这个分析
            
        print(f"\n{attack} (vr={vr}):")
        print(f"  总轮次: {total_rounds}")
        print(f"  有攻击者的轮次: {rounds_with_malicious}")

def analyze_merkle_failures(results):
    """分析 Merkle 失败与攻击者的关系"""
    
    print("\n" + "=" * 80)
    print("分析 2: Merkle 失败统计")
    print("=" * 80)
    
    for exp in results:
        attack = exp['attack']
        vr = exp['vr']
        
        total_merkle_failures = 0
        total_step_failures = 0
        
        for round_data in exp['rounds']:
            total_merkle_failures += len(round_data['merkle_failures'])
            total_step_failures += len(round_data['step_failures'])
        
        print(f"\n{attack} (vr={vr}):")
        print(f"  Merkle 失败: {total_merkle_failures}")
        print(f"  训练步数不足: {total_step_failures}")
        print(f"  TPR: {exp['tpr']}")
        print(f"  FPR: {exp['fpr']}")

def compare_vr_experiments(log_vr01, log_vr03):
    """对比 vr=0.1 和 vr=0.3 的实验"""
    
    print("\n" + "=" * 80)
    print("分析 3: vr=0.1 vs vr=0.3 对比")
    print("=" * 80)
    
    results_01 = parse_experiment_log(log_vr01)
    results_03 = parse_experiment_log(log_vr03)
    
    # 按攻击类型对比
    attacks = set([r['attack'] for r in results_01])
    
    print(f"\n{'攻击类型':<30} {'vr=0.1 Merkle失败':<20} {'vr=0.3 Merkle失败':<20} {'vr=0.1 TPR':<15} {'vr=0.3 TPR':<15}")
    print("-" * 100)
    
    for attack in sorted(attacks):
        r01 = next((r for r in results_01 if r['attack'] == attack), None)
        r03 = next((r for r in results_03 if r['attack'] == attack), None)
        
        if not r01 or not r03:
            continue
        
        merkle_01 = sum(len(rd['merkle_failures']) for rd in r01['rounds'])
        merkle_03 = sum(len(rd['merkle_failures']) for rd in r03['rounds'])
        
        tpr_01 = r01['tpr'] if r01['tpr'] is not None else 0.0
        tpr_03 = r03['tpr'] if r03['tpr'] is not None else 0.0
        
        print(f"{attack:<30} {merkle_01:<20} {merkle_03:<20} {tpr_01:<15.4f} {tpr_03:<15.4f}")

def main():
    log_dir = Path('experiments/logs')
    
    log_vr01 = log_dir / 'rq1_param_scan_20251115_003516_vr01_scan.log'
    log_vr03 = log_dir / 'rq1_param_scan_20251115_050931_vr03_scan.log'
    
    if not log_vr01.exists():
        print(f"错误: 找不到日志文件 {log_vr01}")
        return 1
    
    if not log_vr03.exists():
        print(f"错误: 找不到日志文件 {log_vr03}")
        return 1
    
    print("开始分析 vr≥0.3 的 TPR=0 问题...")
    print(f"日志文件:")
    print(f"  vr=0.1: {log_vr01}")
    print(f"  vr=0.3: {log_vr03}")
    print()
    
    # 解析日志
    print("正在解析日志...")
    results_01 = parse_experiment_log(log_vr01)
    results_03 = parse_experiment_log(log_vr03)
    
    print(f"vr=0.1: 解析了 {len(results_01)} 个实验")
    print(f"vr=0.3: 解析了 {len(results_03)} 个实验")
    
    # 分析
    analyze_merkle_failures(results_01)
    analyze_merkle_failures(results_03)
    
    compare_vr_experiments(log_vr01, log_vr03)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

