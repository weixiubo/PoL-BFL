#!/usr/bin/env python3
"""
ZKP-PoL集成测试脚本
验证ZKP与PoL验证流程的集成
"""

import sys
import logging
import numpy as np
from server.pol.ZKPPoLVerifier import ZKPPoLVerifier, ZKPPoLAggregator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_zkp_pol_verifier():
    """测试ZKP-PoL验证器"""
    logger.info("\n" + "="*60)
    logger.info("[1/3] Testing ZKP-PoL Verifier...")
    logger.info("="*60)

    try:
        # 初始化验证器
        args = {
            'model_dim': 256,
            'data_size': 32,
            'use_zkp': True,
            'zkp_tolerance': 0.01
        }
        verifier = ZKPPoLVerifier(args)

        # 生成测试数据
        W_t = np.random.randn(256).astype(np.float32)
        gradients = np.random.randn(256).astype(np.float32) * 0.01
        learning_rate = 0.001
        W_t_plus_1 = W_t - learning_rate * gradients
        data = np.random.randn(32).astype(np.float32)

        # 验证单个步骤
        is_valid, detail = verifier.verify_with_zkp(
            W_t=W_t,
            W_t_plus_1=W_t_plus_1,
            data=data,
            learning_rate=learning_rate,
            batch_size=32,
            step_number=1
        )

        if not is_valid:
            logger.error(f"[FAIL] Verification failed: {detail}")
            return False

        logger.info("[PASS] Single step verification passed")
        logger.info(f"   L2 error: {detail['l2_error']:.6f}")
        logger.info(f"   Proof size: {detail['proof_size']} bytes")

        # 验证checkpoint
        checkpoint = {
            'W_t': W_t.tolist(),
            'W_t_plus_1': W_t_plus_1.tolist(),
            'data': data.tolist(),
            'batch_size': 32,
            'step_number': 1
        }

        is_valid, detail = verifier.verify_checkpoint_with_zkp(checkpoint, learning_rate)

        if not is_valid:
            logger.error(f"[FAIL] Checkpoint verification failed: {detail}")
            return False

        logger.info("[PASS] Checkpoint verification passed")

        # 验证训练轨迹
        checkpoints = []
        for i in range(5):
            W_t = np.random.randn(256).astype(np.float32)
            gradients = np.random.randn(256).astype(np.float32) * 0.01
            W_t_plus_1 = W_t - learning_rate * gradients
            data = np.random.randn(32).astype(np.float32)

            checkpoint = {
                'W_t': W_t.tolist(),
                'W_t_plus_1': W_t_plus_1.tolist(),
                'data': data.tolist(),
                'batch_size': 32,
                'step_number': i+1
            }
            checkpoints.append(checkpoint)

        is_valid, result = verifier.verify_training_trajectory_with_zkp(checkpoints, learning_rate)

        if not is_valid:
            logger.error(f"[FAIL] Training trajectory verification failed")
            return False

        logger.info("[PASS] Training trajectory verification passed")
        logger.info(f"   Verified: {result['verified_checkpoints']}/{result['total_checkpoints']}")
        logger.info(f"   Total proof size: {result['total_proof_size']} bytes")

        return True

    except Exception as e:
        logger.error(f"[FAIL] ZKP-PoL Verifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zkp_pol_aggregator():
    """测试ZKP-PoL聚合器"""
    logger.info("\n" + "="*60)
    logger.info("[2/3] Testing ZKP-PoL Aggregator...")
    logger.info("="*60)

    try:
        # 初始化聚合器
        args = {
            'model_dim': 256,
            'data_size': 32,
            'use_zkp': True,
            'zkp_tolerance': 0.01
        }
        aggregator = ZKPPoLAggregator(args)

        # 生成客户端更新
        client_updates = {}
        learning_rate = 0.001

        for client_id in range(3):
            W_t = np.random.randn(256).astype(np.float32)
            gradients = np.random.randn(256).astype(np.float32) * 0.01
            W_t_plus_1 = W_t - learning_rate * gradients
            data = np.random.randn(32).astype(np.float32)

            update = {
                'W_t': W_t.tolist(),
                'W_t_plus_1': W_t_plus_1.tolist(),
                'data': data.tolist(),
                'batch_size': 32,
                'step_number': 1
            }
            client_updates[f'client_{client_id}'] = update

        # 聚合并验证
        is_valid, result = aggregator.aggregate_with_zkp_verification(
            client_updates,
            learning_rate
        )

        if not is_valid:
            logger.error(f"[FAIL] Aggregation failed")
            return False

        logger.info("[PASS] Aggregation with ZKP verification passed")
        logger.info(f"   Total clients: {result['total_clients']}")
        logger.info(f"   Verified clients: {result['verified_clients']}")
        logger.info(f"   Failed clients: {result['failed_clients']}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] ZKP-PoL Aggregator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_verification_stats():
    """测试验证统计"""
    logger.info("\n" + "="*60)
    logger.info("[3/3] Testing Verification Stats...")
    logger.info("="*60)

    try:
        args = {
            'model_dim': 256,
            'data_size': 32,
            'use_zkp': True,
            'zkp_tolerance': 0.01
        }
        verifier = ZKPPoLVerifier(args)

        stats = verifier.get_verification_stats()

        logger.info("[PASS] Verification stats retrieved")
        logger.info(f"   Model dimension: {stats['model_dim']}")
        logger.info(f"   Data size: {stats['data_size']}")
        logger.info(f"   Use ZKP: {stats['use_zkp']}")
        logger.info(f"   ZKP tolerance: {stats['zkp_tolerance']}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Verification stats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    logger.info("\n" + "="*60)
    logger.info("[TEST] ZKP-PoL Integration Tests")
    logger.info("="*60)

    results = []

    # 运行所有测试
    results.append(("ZKP-PoL Verifier", test_zkp_pol_verifier()))
    results.append(("ZKP-PoL Aggregator", test_zkp_pol_aggregator()))
    results.append(("Verification Stats", test_verification_stats()))

    # 总结
    logger.info("\n" + "="*60)
    logger.info("[RESULT] Test Summary")
    logger.info("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        logger.info(f"{status}: {name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n[PASS] All ZKP-PoL integration tests passed.")
        return 0
    else:
        logger.error(f"\n[FAIL] {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

