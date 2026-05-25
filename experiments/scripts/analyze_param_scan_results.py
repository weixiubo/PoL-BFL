#!/usr/bin/env python3
"""
分析参数扫描结果并生成对比报告

用法:
    python analyze_param_scan_results.py
"""

import re
import os
from pathlib import Path
from collections import defaultdict
import json

def parse_log_file(log_path):
    """解析单个日志文件，提取关键指标"""
    
    results = {
        'config': {},
        'experiments': []
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 提取配置参数
    vr_match = re.search(r'PARAM_NAME=verification_rate PARAM_VALUE=([\d.]+)', content)
    if vr_match:
        results['config']['verification_rate'] = float(vr_match.group(1))
    else:
        # 尝试从 pol_config 中提取
        vr_match = re.search(r'"verification_rate":\s*([\d.]+)', content)
        if vr_match:
            results['config']['verification_rate'] = float(vr_match.group(1))
    
    delta_match = re.search(r'PARAM_NAME=delta PARAM_VALUE=([\d.]+)', content)
    if delta_match:
        results['config']['delta'] = float(delta_match.group(1))
    else:
        delta_match = re.search(r'"delta":\s*([\d.]+)', content)
        if delta_match:
            results['config']['delta'] = float(delta_match.group(1))
    
    # 提取实验结果
    # 查找所有 "Running: X vs Y" 的组合
    running_pattern = r'Running:\s+(\w+)\s+vs\s+(\w+)'
    running_matches = re.finditer(running_pattern, content)
    
    # 查找所有结果摘要
    summary_start = content.find('RQ1: Security Evaluation Summary')
    if summary_start == -1:
        return results
    
    summary_content = content[summary_start:]
    
    # 解析每个实验的结果
    exp_pattern = r'(\w+)\s+vs\s+(\w+):\s+.*?Final Accuracy:\s+([\d.]+).*?(?:TPR.*?:\s+([\d.]+).*?FPR.*?:\s+([\d.]+).*?Precision:\s+([\d.]+).*?Recall:\s+([\d.]+).*?F1 Score:\s+([\d.]+))?'
    
    for match in re.finditer(exp_pattern, summary_content, re.DOTALL):
        aggregator = match.group(1)
        attack = match.group(2)
        accuracy = float(match.group(3))
        
        exp_result = {
            'aggregator': aggregator,
            'attack': attack,
            'accuracy': accuracy
        }
        
        # 如果是 PoL_FL，提取检测指标
        if aggregator == 'PoL_FL' and match.group(4):
            exp_result['tpr'] = float(match.group(4))
            exp_result['fpr'] = float(match.group(5))
            exp_result['precision'] = float(match.group(6))
            exp_result['recall'] = float(match.group(7))
            exp_result['f1'] = float(match.group(8))
        
        results['experiments'].append(exp_result)
    
    return results

def analyze_verification_rate_impact(all_results):
    """分析 verification_rate 对性能的影响"""
    
    print("\n" + "="*80)
    print("Verification Rate 影响分析")
    print("="*80)
    
    # 按 verification_rate 分组
    vr_groups = defaultdict(list)
    for log_file, results in all_results.items():
        vr = results['config'].get('verification_rate')
        if vr is not None:
            vr_groups[vr].append(results)
    
    # 分析无攻击场景下的 FPR
    print("\n1. 无攻击场景下的 False Positive Rate (FPR)")
    print("-" * 80)
    print(f"{'VR':<8} {'FPR':<10} {'PoL Accuracy':<15} {'评价':<20}")
    print("-" * 80)
    
    for vr in sorted(vr_groups.keys()):
        results_list = vr_groups[vr]
        if not results_list:
            continue
        
        results = results_list[0]  # 取第一个（应该只有一个）
        
        # 查找 PoL_FL vs no_attack
        pol_no_attack = None
        for exp in results['experiments']:
            if exp['aggregator'] == 'PoL_FL' and exp['attack'] == 'no_attack':
                pol_no_attack = exp
                break
        
        if pol_no_attack:
            fpr = pol_no_attack.get('fpr', 0.0)
            accuracy = pol_no_attack.get('accuracy', 0.0)
            
            if fpr == 0.0:
                rating = "✅ 优秀"
            elif fpr < 0.1:
                rating = "⚠️ 可接受"
            else:
                rating = "❌ 过高"
            
            print(f"{vr:<8.1f} {fpr:<10.4f} {accuracy:<15.4f} {rating:<20}")
    
    # 分析不同攻击下的 TPR
    print("\n2. 不同攻击类型的 True Positive Rate (TPR)")
    print("-" * 80)
    
    # 收集所有攻击类型
    all_attacks = set()
    for results_list in vr_groups.values():
        for results in results_list:
            for exp in results['experiments']:
                if exp['aggregator'] == 'PoL_FL' and exp['attack'] != 'no_attack':
                    all_attacks.add(exp['attack'])
    
    # 为每个攻击类型生成表格
    for attack in sorted(all_attacks):
        print(f"\n攻击类型: {attack}")
        print(f"{'VR':<8} {'TPR':<10} {'FPR':<10} {'Precision':<12} {'F1':<10}")
        print("-" * 60)
        
        for vr in sorted(vr_groups.keys()):
            results_list = vr_groups[vr]
            if not results_list:
                continue
            
            results = results_list[0]
            
            # 查找对应的实验
            exp = None
            for e in results['experiments']:
                if e['aggregator'] == 'PoL_FL' and e['attack'] == attack:
                    exp = e
                    break
            
            if exp and 'tpr' in exp:
                tpr = exp.get('tpr', 0.0)
                fpr = exp.get('fpr', 0.0)
                precision = exp.get('precision', 0.0)
                f1 = exp.get('f1', 0.0)
                
                print(f"{vr:<8.1f} {tpr:<10.4f} {fpr:<10.4f} {precision:<12.4f} {f1:<10.4f}")
    
    # 分析准确率
    print("\n3. PoL_FL 在不同场景下的准确率")
    print("-" * 80)
    print(f"{'VR':<8} {'无攻击':<12} {'Random Noise':<15} {'Label Flip':<15} {'Grad Inv':<15}")
    print("-" * 80)
    
    for vr in sorted(vr_groups.keys()):
        results_list = vr_groups[vr]
        if not results_list:
            continue
        
        results = results_list[0]
        
        accuracies = {}
        for exp in results['experiments']:
            if exp['aggregator'] == 'PoL_FL':
                attack = exp['attack']
                if attack == 'no_attack':
                    accuracies['no_attack'] = exp['accuracy']
                elif 'random_noise' in attack:
                    accuracies['random_noise'] = exp['accuracy']
                elif 'label_flipping' in attack:
                    accuracies['label_flip'] = exp['accuracy']
                elif 'gradient_inversion' in attack:
                    accuracies['grad_inv'] = exp['accuracy']
        
        print(f"{vr:<8.1f} "
              f"{accuracies.get('no_attack', 0.0):<12.4f} "
              f"{accuracies.get('random_noise', 0.0):<15.4f} "
              f"{accuracies.get('label_flip', 0.0):<15.4f} "
              f"{accuracies.get('grad_inv', 0.0):<15.4f}")

def generate_summary_report(all_results):
    """生成总结报告"""
    
    print("\n" + "="*80)
    print("参数扫描总结报告")
    print("="*80)
    
    print(f"\n已分析的日志文件数量: {len(all_results)}")
    
    # 统计配置
    vr_values = set()
    delta_values = set()
    
    for results in all_results.values():
        vr = results['config'].get('verification_rate')
        delta = results['config'].get('delta')
        if vr is not None:
            vr_values.add(vr)
        if delta is not None:
            delta_values.add(delta)
    
    print(f"\nVerification Rate 配置: {sorted(vr_values)}")
    print(f"Delta 配置: {sorted(delta_values)}")
    
    # 推荐配置
    print("\n" + "="*80)
    print("推荐配置")
    print("="*80)
    
    # 基于 FPR=0 的配置
    best_vr = None
    best_vr_accuracy = 0.0
    
    for log_file, results in all_results.items():
        vr = results['config'].get('verification_rate')
        if vr is None:
            continue
        
        # 查找 PoL_FL vs no_attack
        for exp in results['experiments']:
            if exp['aggregator'] == 'PoL_FL' and exp['attack'] == 'no_attack':
                fpr = exp.get('fpr', 1.0)
                accuracy = exp.get('accuracy', 0.0)
                
                if fpr == 0.0 and accuracy > best_vr_accuracy:
                    best_vr = vr
                    best_vr_accuracy = accuracy
    
    if best_vr is not None:
        print(f"\n✅ 推荐 verification_rate: {best_vr}")
        print(f"   理由: FPR=0，准确率最高 ({best_vr_accuracy:.4f})")
    else:
        print("\n⚠️ 未找到 FPR=0 的配置")

def main():
    """主函数"""
    
    # 查找所有参数扫描日志
    log_dir = Path('experiments/logs')
    log_files = []
    
    # RQ1 默认配置
    default_log = log_dir / 'rq1_20251114_212528_quick_clearance_v2.log'
    if default_log.exists():
        log_files.append(default_log)
    
    # 参数扫描日志
    for log_file in log_dir.glob('rq1_param_scan_*.log'):
        log_files.append(log_file)
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    # 解析所有日志
    all_results = {}
    for log_file in log_files:
        print(f"解析: {log_file.name}")
        try:
            results = parse_log_file(log_file)
            if results['experiments']:
                all_results[log_file.name] = results
                print(f"  ✅ 成功解析 {len(results['experiments'])} 个实验")
            else:
                print(f"  ⚠️ 未找到实验结果（可能还在运行）")
        except Exception as e:
            print(f"  ❌ 解析失败: {e}")
    
    if not all_results:
        print("\n❌ 没有可分析的数据")
        return
    
    # 生成分析报告
    analyze_verification_rate_impact(all_results)
    generate_summary_report(all_results)
    
    # 保存 JSON 结果
    output_file = 'experiments/results/param_scan_analysis.json'
    os.makedirs('experiments/results', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ 分析结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

