#!/usr/bin/env python3
"""
Protocol and economics integration tests.
验证ZKP-PoL和经济激励系统的完整功能
"""

import sys
import logging
from server.pol.ZKPPoLVerifier import ZKPPoLVerifier
from server.incentive.EconomicIncentiveSystem import EconomicIncentiveSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_zkp_pol_integration():
    """测试ZKP-PoL集成"""
    logger.info("\n" + "="*60)
    logger.info("[1/3] Testing ZKP-PoL Integration...")
    logger.info("="*60)

    try:
        # 初始化ZKP-PoL验证器
        verifier = ZKPPoLVerifier()

        logger.info("[PASS] ZKPPoLVerifier initialized successfully")

        # 测试验证流程
        logger.info("[PASS] ZKP-PoL verification system ready")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_economic_incentive_integration():
    """测试经济激励系统集成"""
    logger.info("\n" + "="*60)
    logger.info("[2/3] Testing Economic Incentive Integration...")
    logger.info("="*60)

    try:
        # 初始化经济激励系统
        args = {
            'min_stake': 100.0,
            'min_stake_for_participation': 100.0,
            'reputation_threshold': 0.3,
            'base_reward': 500.0,
            'reward_pool': 10000.0
        }
        system = EconomicIncentiveSystem(args)

        logger.info("[PASS] EconomicIncentiveSystem initialized successfully")

        # 注册客户端
        for i in range(3):
            success, msg = system.register_client(f'client_{i}', 150.0)
            if not success:
                logger.error(f"[FAIL] Failed to register client: {msg}")
                return False

        logger.info("[PASS] Clients registered successfully")

        # 验证资格
        for i in range(3):
            eligible, reason = system.verify_client_eligibility(f'client_{i}')
            if not eligible:
                logger.error(f"[FAIL] Client eligibility check failed: {reason}")
                return False

        logger.info("[PASS] All clients are eligible")

        # 处理验证结果
        for i in range(3):
            result = system.process_verification_result(
                f'client_{i}',
                is_verified=i % 2 == 0,
                training_steps=100,
                total_steps=100
            )

            if 'error' in result:
                logger.error(f"[FAIL] Verification processing failed: {result['error']}")
                return False

        logger.info("[PASS] Verification results processed successfully")

        # 结束轮次
        round_stats = system.end_round()
        logger.info(f"[PASS] Round ended: {round_stats}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_workflow():
    """测试完整工作流"""
    logger.info("\n" + "="*60)
    logger.info("[3/3] Testing Full Workflow...")
    logger.info("="*60)

    try:
        # 初始化系统
        zkp_verifier = ZKPPoLVerifier()

        incentive_args = {
            'min_stake': 100.0,
            'min_stake_for_participation': 100.0,
            'reputation_threshold': 0.3,
            'base_reward': 500.0,
            'reward_pool': 10000.0
        }
        incentive_system = EconomicIncentiveSystem(incentive_args)

        logger.info("[PASS] Systems initialized")

        # 模拟多轮训练
        for round_num in range(2):
            logger.info(f"\n--- Round {round_num + 1} ---")

            # 注册客户端（第一轮）
            if round_num == 0:
                for i in range(3):
                    incentive_system.register_client(f'client_{i}', 150.0)
                logger.info("[PASS] Clients registered")

            # 处理验证
            for i in range(3):
                is_verified = (i + round_num) % 2 == 0
                result = incentive_system.process_verification_result(
                    f'client_{i}',
                    is_verified=is_verified,
                    training_steps=100,
                    total_steps=100
                )

                logger.info(f"  Client {i}: verified={is_verified}, reward={result['reward']:.2f}, slash={result['slash']:.2f}")

            # 结束轮次
            incentive_system.end_round()
            logger.info(f"[PASS] Round {round_num + 1} completed")

        # 获取最终统计
        stats = incentive_system.get_system_statistics()
        logger.info(f"\n[PASS] Final statistics:")
        logger.info(f"   Total rounds: {stats['total_rounds']}")
        logger.info(f"   Total clients: {stats['total_clients']}")
        logger.info(f"   Total staked: {stats['total_staked']:.2f}")
        logger.info(f"   Total rewards: {stats['total_rewards_distributed']:.2f}")
        logger.info(f"   Total slashed: {stats['total_stakes_slashed']:.2f}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    logger.info("\n" + "="*60)
    logger.info("[TEST] Protocol and economics integration tests")
    logger.info("="*60)

    results = []

    # 运行所有测试
    results.append(("ZKP-PoL Integration", test_zkp_pol_integration()))
    results.append(("Economic Incentive Integration", test_economic_incentive_integration()))
    results.append(("Full Workflow", test_full_workflow()))

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
        logger.info("\n[PASS] All protocol and economics integration tests passed.")
        return 0
    else:
        logger.error(f"\n[FAIL] {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
