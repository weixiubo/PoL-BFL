#!/usr/bin/env python3
"""
快速测试checkpoint验证修复

测试配置:
- Dataset: CIFAR-10
- Model: ResNet18
- Rounds: 3 (足够验证跨轮次的checkpoint验证)
- Repetitions: 1
- Variant: pol_only (最简单的PoL变体)

预期结果:
- Round 1-3的checkpoint验证成功率应该都在90%+
- TPR应该提升到90-100%
- FPR应该降低到5-10%
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from pathlib import Path
from datetime import datetime

from run_rq2_ablation import AblationStudyExperiment
from experiment_config import (
    FL_CONFIG, POL_CONFIG, DATASET_CONFIG, MODEL_CONFIG
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Test checkpoint verification fix')
    parser.add_argument('--num_rounds', type=int, default=3,
                       help='Number of FL rounds (default: 3)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Checkpoint Verification Fix Test")
    logger.info("=" * 70)
    logger.info(f"Rounds: {args.num_rounds}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Delta: {POL_CONFIG['delta']}")
    logger.info("=" * 70)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / 'results' / 'checkpoint_fix_test' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建测试实例
    study = AblationStudyExperiment(
        output_dir=output_dir,
        device=args.device
    )

    # 只测试pol_only变体
    logger.info("\n" + "=" * 70)
    logger.info("Testing variant: pol_only")
    logger.info("=" * 70)

    try:
        results = study.run_variant(
            variant_name='pol_only',
            num_rounds=args.num_rounds,
            num_repetitions=1,
            enable_zkp=False,
            enable_incentive=False
        )

        # 打印结果
        logger.info("\n" + "=" * 70)
        logger.info("Test Results")
        logger.info("=" * 70)
        logger.info(f"Final Accuracy: {results['final_accuracy']:.4f}")
        logger.info(f"Detection TPR: {results['tpr']:.4f} (Target: >0.90)")
        logger.info(f"Detection FPR: {results['fpr']:.4f} (Target: <0.10)")
        logger.info(f"Participation Rate: {results['participation_rate']:.4f}")
        logger.info("=" * 70)

        # 评估修复效果
        logger.info("\n" + "=" * 70)
        logger.info("Fix Evaluation")
        logger.info("=" * 70)

        if results['tpr'] >= 0.90:
            logger.info("✅ TPR >= 90%: PASS")
        else:
            logger.warning(f"⚠️ TPR = {results['tpr']:.1%} < 90%: NEEDS IMPROVEMENT")

        if results['fpr'] <= 0.10:
            logger.info("✅ FPR <= 10%: PASS")
        else:
            logger.warning(f"⚠️ FPR = {results['fpr']:.1%} > 10%: NEEDS IMPROVEMENT")

        logger.info("=" * 70)

        # 保存结果
        import json
        result_file = output_dir / 'test_results.json'
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {result_file}")

        # 检查日志中的checkpoint验证信息
        logger.info("\n" + "=" * 70)
        logger.info("Checkpoint Verification Analysis")
        logger.info("=" * 70)
        logger.info("Please check the log for:")
        logger.info("1. 'Using step-based challenge' messages (should appear)")
        logger.info("2. 'Verification result: X/Y passed' (success rate should be >90%)")
        logger.info("3. 'Checkpoint at step X not found' warnings (should be minimal)")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

