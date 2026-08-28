#!/usr/bin/env python3
"""
分析RQ1实验结果
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_rq1_results(gpu0_dir, gpu1_dir):
    """分析RQ1实验结果"""

    print("=" * 80)
    print("RQ1 实验结果分析")
    print("=" * 80)
    print()

    # 合并两个GPU的结果
    all_results = []

    for gpu_id, result_dir in enumerate([gpu0_dir, gpu1_dir]):
        result_path = Path(result_dir)
        results_file = result_path / 'rq1_results.json'

        if not results_file.exists():
            print(f"[WARNING] GPU {gpu_id} 结果文件不存在: {results_file}")
            continue

        with open(results_file) as f:
            data = json.load(f)

        print(f"GPU {gpu_id}: 加载了 {len(data)} 个实验结果")
        all_results.extend(data)

    if not all_results:
        print("[FAIL] 没有找到任何实验结果")
        return

    print(f"\n总计: {len(all_results)} 个实验结果\n")

    # 按攻击类型分组
    by_attack = defaultdict(list)
    for exp in all_results:
        attack = exp['attack_type']
        by_attack[attack].append(exp)

    # 分析每个攻击
    print("### 各攻击类型结果")
    print("-" * 80)
    print(f"{'攻击类型':<40} {'最终准确率':<15} {'轮数':<10} {'状态'}")
    print("-" * 80)

    issues = []

    for attack in sorted(by_attack.keys()):
        exps = by_attack[attack]

        # 取第一个实验的数据（应该只有一个）
        exp = exps[0]
        final_acc = exp.get('final_accuracy', 0)
        rounds = exp.get('rounds', [])
        num_rounds = len(rounds)

        # 判断状态
        if num_rounds < 20:
            status = f"[WARNING] 轮数不足 ({num_rounds}/20)"
            issues.append(f"{attack}: 轮数不足")
        elif final_acc < 0.90:
            status = f"[WARNING] 准确率低 ({final_acc:.4f})"
            issues.append(f"{attack}: 准确率低")
        else:
            status = "[PASS] 正常"

        print(f"{attack:<40} {final_acc:>14.4f} {num_rounds:>9} {status}")

    # 检查检测率（如果有）
    print("\n### 检测率分析")
    print("-" * 80)

    detection_file_0 = Path(gpu0_dir) / 'rq1_with_detection.json'
    detection_file_1 = Path(gpu1_dir) / 'rq1_with_detection.json'

    detection_data = []
    for det_file in [detection_file_0, detection_file_1]:
        if det_file.exists():
            with open(det_file) as f:
                detection_data.extend(json.load(f))

    if detection_data:
        print(f"{'攻击类型':<40} {'TPR':<10} {'FPR':<10} {'状态'}")
        print("-" * 80)

        for det in detection_data:
            attack = det['attack_type']
            metrics = det.get('detection_metrics', {})
            tpr = metrics.get('TPR', 0)
            fpr = metrics.get('FPR', 0)

            # 判断状态
            if attack == 'no_attack':
                # 无攻击场景，TPR应该为0
                if tpr == 0 and fpr == 0:
                    status = "[PASS] 正常"
                else:
                    status = f"[WARNING] 异常"
                    issues.append(f"{attack}: 检测率异常")
            elif attack == 'byzantine_label_flipping':
                # Label flipping是PoL的设计局限
                if tpr == 0:
                    status = "[PASS] 预期（PoL局限）"
                else:
                    status = "[WARNING] 意外检测到"
            else:
                # 其他攻击应该被检测到
                if tpr >= 0.95 and fpr <= 0.1:
                    status = "[PASS] Meets target"
                elif tpr >= 0.8:
                    status = "[WARNING] 可接受"
                else:
                    status = f"[FAIL] TPR过低"
                    issues.append(f"{attack}: TPR={tpr:.2f} 过低")

            print(f"{attack:<40} {tpr:>9.4f} {fpr:>9.4f} {status}")
    else:
        print("[WARNING] 未找到检测率数据")

    # 总结
    print("\n### 总结")
    print("-" * 80)

    if issues:
        print("发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n建议: 检查实验配置和参数")
    else:
        print("[PASS] 所有实验结果正常")
        print("[PASS] 可以继续进行后续实验")

    print("\n" + "=" * 80)

def main():
    """主函数"""
    gpu0_dir = 'experiments/results/rq1_mnist_polfl_20r_gpu0'
    gpu1_dir = 'experiments/results/rq1_mnist_polfl_20r_gpu1'

    analyze_rq1_results(gpu0_dir, gpu1_dir)

if __name__ == '__main__':
    main()
