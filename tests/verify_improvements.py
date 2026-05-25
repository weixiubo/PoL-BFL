#!/usr/bin/env python3
"""
验证三个改进是否正确应用
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_improvements():
    """验证所有改进"""
    
    logger.info("=" * 60)
    logger.info("验证三个改进")
    logger.info("=" * 60)
    
    # 验证改进1: Top-Q验证
    logger.info("\n[改进1] 验证Top-Q验证是否启用...")
    try:
        from config.pol_config import POL_CONFIG
        use_top_q = POL_CONFIG.get('use_top_q')
        if use_top_q:
            logger.info("✅ Top-Q验证已启用: use_top_q = True")
        else:
            logger.error("❌ Top-Q验证未启用: use_top_q = False")
            return False
    except Exception as e:
        logger.error(f"❌ 验证Top-Q失败: {e}")
        return False
    
    # 验证改进2: Delta阈值
    logger.info("\n[改进2] 验证Delta阈值是否调整...")
    try:
        delta = POL_CONFIG.get('delta')
        if delta == 200.0:
            logger.info(f"✅ Delta阈值已调整: delta = {delta}")
        else:
            logger.error(f"❌ Delta阈值未正确调整: delta = {delta} (期望: 200.0)")
            return False
    except Exception as e:
        logger.error(f"❌ 验证Delta失败: {e}")
        return False
    
    # 验证改进3: Cosine距离
    logger.info("\n[改进3] 验证Cosine距离是否启用...")
    try:
        distance_metric = POL_CONFIG.get('distance_metric')
        if distance_metric == 'cosine':
            logger.info(f"✅ Cosine距离已启用: distance_metric = {distance_metric}")
        else:
            logger.error(f"❌ Cosine距离未启用: distance_metric = {distance_metric} (期望: cosine)")
            return False
    except Exception as e:
        logger.error(f"❌ 验证Cosine距离失败: {e}")
        return False
    
    # 验证PoLVerifier中的Cosine实现
    logger.info("\n[改进3补充] 验证PoLVerifier中的Cosine实现...")
    try:
        import torch
        from server.pol.PoLVerifier import PoLVerifier

        # 创建一个简单的PoLVerifier实例
        args = {
            'delta': 200.0,
            'distance_metric': 'cosine',
            'device': 'cpu',
            'top_q': 5
        }
        verifier = PoLVerifier(args)

        # 测试Cosine距离计算
        from collections import OrderedDict
        state1 = OrderedDict([('layer1.weight', torch.randn(10, 5))])
        state2 = OrderedDict([('layer1.weight', torch.randn(10, 5))])

        distance = verifier._compute_parameter_distance(state1, state2, metric='cosine')

        if 0 <= distance <= 2.0:
            logger.info(f"✅ Cosine距离计算正确: distance = {distance:.6f}")
        else:
            logger.error(f"❌ Cosine距离计算异常: distance = {distance} (应在[0, 2]范围内)")
            return False
    except Exception as e:
        logger.error(f"❌ 验证PoLVerifier Cosine实现失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ 所有改进验证通过！")
    logger.info("=" * 60)
    
    logger.info("\n📊 改进总结:")
    logger.info(f"  1. Top-Q验证: ✅ 已启用")
    logger.info(f"  2. Delta阈值: ✅ 已调整为 200.0")
    logger.info(f"  3. Cosine距离: ✅ 已启用")
    
    logger.info("\n🎯 预期效果:")
    logger.info(f"  FPR: 26.9% → ≤8%")
    logger.info(f"  TPR: 42.5% → >75%")
    
    return True

if __name__ == '__main__':
    success = verify_improvements()
    sys.exit(0 if success else 1)

