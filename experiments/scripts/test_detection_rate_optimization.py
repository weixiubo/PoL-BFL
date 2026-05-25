"""
Detection Rate Optimization Test

测试不同 verification_rate 对 PoL-BFL 检测率的影响

测试配置:
- Dataset: MNIST
- Rounds: 20
- Clients: 10 (5 per round)
- Malicious ratio: 20% (1 malicious client per round)
- Attack: byzantine_alie (Blades attack)
- Verification rates: 0.3, 0.5, 1.0

目标:
- 找到最优的 verification_rate
- 确保检测率 > 90%
- 确保准确率损失 < 2%
"""

import os
import sys
import subprocess
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_test(verification_rate, num_rounds=20):
    """运行单个测试"""
    logger.info("=" * 80)
    logger.info(f"测试 verification_rate = {verification_rate}")
    logger.info("=" * 80)
    
    # 设置环境变量
    env = os.environ.copy()
    env['POL_VERIFICATION_RATE'] = str(verification_rate)
    
    # 构建命令
    cmd = [
        'python', 'experiments/scripts/runners/run_rq1_security.py',
        '--dataset', 'MNIST',
        '--num_rounds', str(num_rounds),
        '--num_clients', '10',
        '--clients_per_round', '5',
        '--attacks', 'byzantine_alie',
        '--baselines', 'PoL_FL',  # 只测试 PoL_FL
    ]
    
    # 运行测试
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 分钟超时
        )
        
        # 解析结果
        output = result.stdout
        
        # 提取关键指标
        metrics = extract_metrics(output)
        metrics['verification_rate'] = verification_rate
        
        return metrics
        
    except subprocess.TimeoutExpired:
        logger.error(f"测试超时 (verification_rate={verification_rate})")
        return None
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return None


def extract_metrics(output):
    """从输出中提取关键指标"""
    metrics = {
        'final_accuracy': None,
        'tpr': None,
        'fpr': None,
        'precision': None,
        'recall': None,
        'f1_score': None,
    }
    
    # 解析输出
    lines = output.split('\n')
    
    for line in lines:
        if 'Final Accuracy:' in line:
            try:
                metrics['final_accuracy'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'TPR (Detection Rate):' in line:
            try:
                metrics['tpr'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'FPR (False Positive Rate):' in line:
            try:
                metrics['fpr'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'Precision:' in line:
            try:
                metrics['precision'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'Recall:' in line:
            try:
                metrics['recall'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'F1 Score:' in line:
            try:
                metrics['f1_score'] = float(line.split(':')[1].strip())
            except:
                pass
    
    return metrics


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("Detection Rate Optimization Test")
    logger.info("=" * 80)

    # 测试不同的 verification_rate
    # 先用 5 轮快速测试，确认无误后再用 20 轮
    num_rounds = int(os.getenv('TEST_ROUNDS', '5'))
    logger.info(f"测试轮数: {num_rounds}")

    verification_rates = [0.3, 0.5, 1.0]
    results = []

    for vr in verification_rates:
        logger.info(f"\n开始测试 verification_rate = {vr}...")
        metrics = run_test(vr, num_rounds=num_rounds)
        
        if metrics:
            results.append(metrics)
            logger.info(f"\n结果:")
            logger.info(f"  Final Accuracy: {metrics.get('final_accuracy', 'N/A')}")
            logger.info(f"  TPR (Detection Rate): {metrics.get('tpr', 'N/A')}")
            logger.info(f"  FPR: {metrics.get('fpr', 'N/A')}")
            logger.info(f"  F1 Score: {metrics.get('f1_score', 'N/A')}")
        else:
            logger.error(f"测试失败 (verification_rate={vr})")
    
    # 保存结果
    output_dir = Path(__file__).parent.parent.parent / 'experiments' / 'results' / 'detection_optimization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n结果已保存到: {output_file}")
    
    # 分析结果
    logger.info("\n" + "=" * 80)
    logger.info("结果分析")
    logger.info("=" * 80)
    
    if results:
        # 找到最优配置
        best_config = max(results, key=lambda x: x.get('tpr', 0) or 0)
        
        logger.info(f"\n最优配置:")
        logger.info(f"  Verification Rate: {best_config['verification_rate']}")
        logger.info(f"  TPR (Detection Rate): {best_config.get('tpr', 'N/A')}")
        logger.info(f"  Final Accuracy: {best_config.get('final_accuracy', 'N/A')}")
        logger.info(f"  F1 Score: {best_config.get('f1_score', 'N/A')}")
        
        # 检查是否满足要求
        tpr = best_config.get('tpr', 0) or 0
        accuracy = best_config.get('final_accuracy', 0) or 0
        
        logger.info("\n" + "=" * 80)
        logger.info("清障检查")
        logger.info("=" * 80)
        
        checks = [
            ('检测率 > 90%', tpr > 0.9, f'TPR = {tpr:.2%}'),
            ('准确率 > 95%', accuracy > 0.95, f'Accuracy = {accuracy:.2%}'),
            ('假阳性率 < 5%', (best_config.get('fpr', 0) or 0) < 0.05, f'FPR = {best_config.get("fpr", 0):.2%}'),
        ]
        
        all_passed = True
        for check_name, passed, detail in checks:
            status = "✅" if passed else "❌"
            logger.info(f"{status} {check_name}: {detail}")
            if not passed:
                all_passed = False
        
        if all_passed:
            logger.info("\n🎉 所有检查通过！数据对 paper 有利！")
            logger.info(f"\n建议使用 verification_rate = {best_config['verification_rate']}")
        else:
            logger.info("\n⚠️ 部分检查未通过，需要进一步优化")
            logger.info("\n建议:")
            logger.info("  1. 调整 delta 参数")
            logger.info("  2. 增加训练轮数")
            logger.info("  3. 优化验证策略")
    else:
        logger.error("\n❌ 所有测试都失败了")
    
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()

